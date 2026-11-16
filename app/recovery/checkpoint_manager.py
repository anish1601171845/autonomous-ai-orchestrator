"""Durable workflow checkpoints stored in the existing SQL database (SQLite/PostgreSQL).

Why not LangGraph's checkpointer? The project compiles its graph with LangGraph's InMemorySaver, which
does not survive a process crash, and the durable SQLite/Postgres savers are separate packages that
would duplicate the database this project already uses. A full WorkflowState snapshot is therefore
written to `workflow_checkpoints` after every node (same transaction boundary as the task upsert),
and `restore_checkpoint` rebuilds a resumable state from it.
"""
from __future__ import annotations

import copy

from sqlalchemy import delete, func, select

from app.database.models import SessionLocal, WorkflowCheckpointRow

KEEP_LAST = 60


def save_checkpoint(state: dict, node: str) -> int:
    wid = state["workflow_id"]
    snapshot = {k: v for k, v in state.items() if k != "messages"}  # messages already persisted as events
    with SessionLocal.begin() as s:
        seq = (s.scalar(select(func.max(WorkflowCheckpointRow.seq)).where(WorkflowCheckpointRow.workflow_id == wid)) or 0) + 1
        s.add(WorkflowCheckpointRow(workflow_id=wid, seq=seq, node=node, state=copy.deepcopy(snapshot)))
        cutoff = seq - KEEP_LAST
        if cutoff > 0:
            s.execute(delete(WorkflowCheckpointRow).where(WorkflowCheckpointRow.workflow_id == wid, WorkflowCheckpointRow.seq <= cutoff))
    return seq


def latest_checkpoint(workflow_id: str) -> dict | None:
    with SessionLocal() as s:
        row = s.scalars(select(WorkflowCheckpointRow).where(WorkflowCheckpointRow.workflow_id == workflow_id)
                        .order_by(WorkflowCheckpointRow.seq.desc())).first()
        return {"seq": row.seq, "node": row.node, "state": row.state} if row else None


def restore_checkpoint(workflow_id: str, seq: int | None = None) -> dict:
    """Return a resumable state. A task that was mid-execution when the crash happened is
    *partially completed*: it is put back at the front of the queue and re-executed. Re-execution is
    safe because workspace writes are idempotent (files are overwritten, never appended) and external
    writes are guarded by the idempotency ledger."""
    with SessionLocal() as s:
        q = select(WorkflowCheckpointRow).where(WorkflowCheckpointRow.workflow_id == workflow_id)
        q = q.where(WorkflowCheckpointRow.seq == seq) if seq else q.order_by(WorkflowCheckpointRow.seq.desc())
        row = s.scalars(q).first()
    if row is None:
        raise LookupError(f"no checkpoint for workflow {workflow_id}")
    state = copy.deepcopy(row.state)
    state["messages"] = []
    cur = state.get("current_task_id")
    queue = list(state.get("queue", []))
    for t in state.get("tasks", []):
        if t.get("status") in ("in_progress", "qa_failed") or t.get("task_id") == cur:
            t["status"] = "pending"
            t["interrupted"] = True
            if t["task_id"] not in queue:
                queue.insert(0, t["task_id"])
    state["queue"], state["current_task_id"] = queue, None
    state["restored_from"] = {"seq": row.seq, "node": row.node}
    return state


def list_checkpoints(workflow_id: str) -> list[dict]:
    with SessionLocal() as s:
        rows = s.scalars(select(WorkflowCheckpointRow).where(WorkflowCheckpointRow.workflow_id == workflow_id)
                         .order_by(WorkflowCheckpointRow.seq)).all()
        return [{"seq": r.seq, "node": r.node, "created_at": r.created_at.isoformat() if r.created_at else None,
                 "queue": len(r.state.get("queue", [])), "status": r.state.get("status")} for r in rows]
