"""Persistence operations for workflows, tasks and agent events."""
from __future__ import annotations

from collections import Counter

from sqlalchemy import delete, select

from app.core.security import redact
from app.core.state import utcnow
from app.database.models import EventRow, SessionLocal, TaskRow, WorkflowRow, init_db

init_db()
TASK_FIELDS = ["title", "description", "feature", "category", "category_confidence", "priority", "priority_confidence",
               "complexity", "status", "assigned_agent", "attempts", "dependency_ids", "artifacts", "qa_report", "error",
               "risk_probability", "risk_category"]


def create_workflow(workflow_id: str, requirement: str) -> dict:
    now = utcnow()
    with SessionLocal.begin() as s:
        s.add(WorkflowRow(id=workflow_id, requirement=requirement, status="created", created_at=now, updated_at=now, state={}))
    return get_workflow(workflow_id)


def save_state(state: dict) -> None:
    wid = state["workflow_id"]
    slim = {k: v for k, v in state.items() if k not in ("tasks", "messages")}
    with SessionLocal.begin() as s:
        wf = s.get(WorkflowRow, wid)
        if wf is None:
            return
        wf.status = state.get("status", wf.status)
        wf.updated_at = utcnow()
        wf.state = slim
        wf.summary = state.get("summary")
        # upsert by (workflow_id, task_id) so task primary keys stay stable for the enhancement tables' foreign keys
        existing = {r.task_id: r for r in s.scalars(select(TaskRow).where(TaskRow.workflow_id == wid))}
        seen = set()
        for t in state.get("tasks", []):
            seen.add(t["task_id"])
            row = existing.get(t["task_id"])
            if row is None:
                s.add(TaskRow(workflow_id=wid, task_id=t["task_id"], **{f: t.get(f) for f in TASK_FIELDS}))
            else:
                for f in TASK_FIELDS:
                    setattr(row, f, t.get(f))
        stale = [tid for tid in existing if tid not in seen]
        if stale:
            s.execute(delete(TaskRow).where(TaskRow.workflow_id == wid, TaskRow.task_id.in_(stale)))


def add_events(workflow_id: str, messages: list[dict]) -> None:
    if not messages:
        return
    with SessionLocal.begin() as s:
        for m in messages:
            s.add(EventRow(workflow_id=workflow_id, timestamp=m.get("timestamp", utcnow()), sender=m.get("sender"),
                           recipient=m.get("recipient"), kind=m.get("kind"),
                           content={k: redact(v) if isinstance(v, str) else v for k, v in m.get("content", {}).items()}))


def set_status(workflow_id: str, status: str, approval: dict | None = None) -> None:
    with SessionLocal.begin() as s:
        wf = s.get(WorkflowRow, workflow_id)
        if wf:
            wf.status, wf.updated_at = status, utcnow()
            if approval is not None:
                wf.approval = approval


def _task_dict(r: TaskRow) -> dict:
    return {"workflow_id": r.workflow_id, "task_id": r.task_id, **{f: getattr(r, f) for f in TASK_FIELDS}}


def get_workflow(workflow_id: str, with_details: bool = True) -> dict | None:
    with SessionLocal() as s:
        wf = s.get(WorkflowRow, workflow_id)
        if wf is None:
            return None
        out = {"workflow_id": wf.id, "requirement": wf.requirement, "status": wf.status, "created_at": wf.created_at,
               "updated_at": wf.updated_at, "summary": wf.summary, "approval": wf.approval}
        if with_details:
            out["state"] = wf.state
            out["tasks"] = [_task_dict(t) for t in s.scalars(select(TaskRow).where(TaskRow.workflow_id == workflow_id).order_by(TaskRow.task_id))]
            out["events"] = [{"timestamp": e.timestamp, "sender": e.sender, "recipient": e.recipient, "kind": e.kind, "content": e.content}
                             for e in s.scalars(select(EventRow).where(EventRow.workflow_id == workflow_id).order_by(EventRow.id))]
        return out


def list_workflows(limit: int = 50) -> list[dict]:
    with SessionLocal() as s:
        rows = s.scalars(select(WorkflowRow).order_by(WorkflowRow.created_at.desc()).limit(limit)).all()
        return [{"workflow_id": w.id, "requirement": w.requirement, "status": w.status, "created_at": w.created_at,
                 "updated_at": w.updated_at, "summary": w.summary} for w in rows]


def list_tasks(workflow_id: str | None = None, status: str | None = None, limit: int = 500) -> list[dict]:
    with SessionLocal() as s:
        q = select(TaskRow)
        if workflow_id:
            q = q.where(TaskRow.workflow_id == workflow_id)
        if status:
            q = q.where(TaskRow.status == status)
        return [_task_dict(t) for t in s.scalars(q.order_by(TaskRow.pk.desc()).limit(limit))]


def task_pk(workflow_id: str | None, task_id: str | None) -> int | None:
    if not workflow_id or not task_id:
        return None
    with SessionLocal() as s:
        return s.scalars(select(TaskRow.pk).where(TaskRow.workflow_id == workflow_id, TaskRow.task_id == task_id)).first()


def get_task(task_id: str, workflow_id: str | None = None) -> dict | None:
    with SessionLocal() as s:
        q = select(TaskRow).where(TaskRow.task_id == task_id)
        if workflow_id:
            q = q.where(TaskRow.workflow_id == workflow_id)
        r = s.scalars(q.order_by(TaskRow.pk.desc())).first()
        return _task_dict(r) if r else None


def metrics() -> dict:
    with SessionLocal() as s:
        tasks = s.scalars(select(TaskRow)).all()
        wfs = s.scalars(select(WorkflowRow)).all()
    st = Counter(t.status for t in tasks)
    total = len(tasks)
    per_agent = Counter(t.assigned_agent for t in tasks if t.assigned_agent)
    active = {t.assigned_agent for t in tasks if t.status in ("assigned", "in_progress") and t.assigned_agent}
    return {
        "total_workflows": len(wfs), "workflow_status": dict(Counter(w.status for w in wfs)),
        "total_tasks": total, "completed_tasks": st.get("completed", 0), "failed_tasks": st.get("failed", 0),
        "blocked_tasks": st.get("blocked", 0),
        "pending_tasks": total - st.get("completed", 0) - st.get("failed", 0) - st.get("blocked", 0),
        "completion_pct": round(100 * st.get("completed", 0) / total, 1) if total else 0.0,
        "active_agents": len(active), "tasks_per_agent": dict(per_agent),
        "by_category": dict(Counter(t.category for t in tasks)), "by_priority": dict(Counter(t.priority for t in tasks)),
        "mean_attempts": round(sum(t.attempts or 0 for t in tasks) / total, 3) if total else 0.0,
    }
