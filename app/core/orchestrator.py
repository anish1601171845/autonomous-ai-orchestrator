"""LangGraph orchestration of the AegisFlow multi-agent workflow.

scope -> classify -> plan -> prioritize -> allocate(+risk) -> select_task -> execute(+memory, recovery) -> debate -> qa
                                                                   ^                                              |
                                                                   +---------------(fail, bounded retry)----------+
select_task --(queue empty)--> finalize -> END   (status: awaiting_approval)

Every node persists the task table, the agent message log and a durable checkpoint. A worker crash is
recovered by restoring the latest checkpoint and resuming at `select_task` (RECOVERY_MODE checkpoint/full).
"""
from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents import (
    classification_agent,
    coding_agent,
    github_agent,
    planning_agent,
    priority_agent,
    qa_agent,
    scoping_agent,
    workload_agent,
)
from app.core import enhancements
from app.core.config import settings
from app.core.security import get_logger, redact
from app.core.state import AgentMessage, TaskStatus, WorkflowState, WorkflowStatus
from app.database import repository as repo
from app.recovery import checkpoint_manager
from app.recovery.fault_injection import INJECTOR, SimulatedFault
from app.recovery.recovery_controller import RecoveryController, RecoveryExhausted, detect_failure, record_recovery_event

log = get_logger(__name__)
TERMINAL = {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.BLOCKED.value}


def _msg(sender: str, recipient: str, kind: str, **content) -> dict:
    return AgentMessage(sender=sender, recipient=recipient, kind=kind, content=content).model_dump()


def _task(state: WorkflowState, task_id: str) -> dict:
    return next(t for t in state["tasks"] if t["task_id"] == task_id)


# ---------------------------------------------------------------- nodes
def scope_node(state: WorkflowState) -> dict:
    if state.get("ticket"):
        from app.agents import intake_agent

        reqs, tasks = intake_agent.scope_ticket(state["ticket"], state["ticket"].get("triage"))
    else:
        reqs, tasks = scoping_agent.run(state["requirement"])
    return {"requirements": reqs, "tasks": [t.model_dump(mode="json") for t in tasks], "status": WorkflowStatus.RUNNING.value,
            "messages": [_msg(scoping_agent.NAME, classification_agent.NAME, "tasks_scoped", count=len(tasks), backend=reqs["backend"],
                              functional=reqs.get("functional"), non_functional=reqs.get("non_functional"))]}


def classify_node(state: WorkflowState) -> dict:
    tasks = classification_agent.run(state["tasks"])
    return {"tasks": tasks, "messages": [_msg(classification_agent.NAME, planning_agent.NAME, "tasks_classified",
                                              categories={t["task_id"]: t["category"] for t in tasks})]}


def plan_node(state: WorkflowState) -> dict:
    tasks = state["tasks"]
    plan = planning_agent.run(tasks)
    return {"tasks": tasks, "edges": plan["edges"],
            "summary": {**state.get("summary", {}), "dependency_method": plan["method"], "encoder_backend": plan["encoder_backend"],
                        "rejected_edges": len(plan["rejected_edges"]), "semantic_candidates": plan["semantic_candidates"][:10]},
            "messages": [_msg(planning_agent.NAME, priority_agent.NAME, "dependencies_mapped", edges=len(plan["edges"]),
                              method=plan["method"], is_dag=plan["is_dag"], rejected=len(plan["rejected_edges"]))]}


def prioritize_node(state: WorkflowState) -> dict:
    tasks = priority_agent.run(state["tasks"])
    return {"tasks": tasks, "messages": [_msg(priority_agent.NAME, workload_agent.NAME, "priorities_predicted",
                                              priorities={t["task_id"]: t["priority"] for t in tasks})]}


def allocate_node(state: WorkflowState) -> dict:
    tasks = state["tasks"]
    alloc = workload_agent.run(tasks)
    order = planning_agent.execution_order(tasks)
    msgs = [_msg(workload_agent.NAME, "executors", "tasks_assigned", scheduler=alloc["scheduler"], assignments=alloc["assignments"])]
    try:
        risk = enhancements.assess_risk(state["workflow_id"], tasks, order, {**state, "tasks": tasks})
    except Exception as e:  # enhancement must never break the core workflow
        log.warning("risk assessment failed: %s", e)
        risk = {"enabled": True, "available": False, "reason": f"{type(e).__name__}: {e}"}
    if risk.get("available"):
        msgs.append(_msg("risk_predictor", "orchestrator", "risk_assessed", high_risk=[t["task_id"] for t in tasks if t.get("risk_category") == "High"],
                         assessed=risk["assessed"]))
    return {"tasks": tasks, "assignments": alloc["assignments"], "scheduler_used": alloc["scheduler"], "execution_order": order,
            "queue": list(order), "summary": {**state.get("summary", {}), "scheduler": alloc["scheduler"], "risk": risk,
                                              "simulated_plan_metrics": alloc["simulated_plan_metrics"]},
            "messages": msgs}


def select_task_node(state: WorkflowState) -> dict:
    tasks, queue, msgs = state["tasks"], list(state.get("queue", [])), []
    by_id = {t["task_id"]: t for t in tasks}
    while queue:
        tid = queue.pop(0)
        t = by_id[tid]
        bad = [d for d in t.get("dependency_ids", []) if by_id.get(d, {}).get("status") in (TaskStatus.FAILED.value, TaskStatus.BLOCKED.value)]
        if bad:
            t["status"], t["error"] = TaskStatus.BLOCKED.value, f"blocked by failed prerequisites: {bad}"
            msgs.append(_msg("orchestrator", t.get("assigned_agent") or "-", "task_blocked", task_id=tid, by=bad))
            continue
        t["status"] = TaskStatus.IN_PROGRESS.value
        t["_started_at"] = t.get("_started_at") or time.time()
        msgs.append(_msg("orchestrator", t.get("assigned_agent") or "-", "task_started", task_id=tid, category=t.get("category")))
        return {"tasks": tasks, "queue": queue, "current_task_id": tid, "messages": msgs}
    return {"tasks": tasks, "queue": [], "current_task_id": None, "messages": msgs}


def _produce(state: WorkflowState, t: dict, memory_context: str | None) -> dict:
    """Run the agent that owns this task category. Fault-injection hook first (no-op unless a test/demo enabled it)."""
    INJECTOR.check("execute", agent=t.get("assigned_agent"), task_id=t["task_id"])
    tasks, wid, cat = state["tasks"], state["workflow_id"], t.get("category")
    feedback = (t.get("qa_report") or {}).get("feedback")
    if cat == "development":
        out = coding_agent.generate_code(wid, t, feedback if t["attempts"] > 1 else None, memory_context=memory_context)
        t["explanation"] = out["explanation"]
        t["test_targets"] = None
        t["memory_context_supplied"] = out.get("memory_context_supplied", False)
    elif cat == "testing":
        by_id = {x["task_id"]: x for x in tasks}
        dep_tests = [Path(a).name for d in t.get("dependency_ids", []) for a in by_id[d].get("artifacts", []) if Path(a).name.startswith("test_")]
        if t.get("feature") == "project" or not dep_tests:
            out = coding_agent.generate_integration_test(wid)
            ws = coding_agent.workspace(wid)
            out["review_files"] = [str(p.relative_to(settings.workspace_dir)) for p in sorted(ws.glob("*.py"))]
            t["test_targets"] = None
        else:
            out = {"files": [], "backend": "qa:execute_dependency_tests",
                   "review_files": [f"{wid}/{n}" for n in dep_tests] + [f"{wid}/{n.removeprefix('test_')}" for n in dep_tests]}
            t["test_targets"] = dep_tests
    elif cat == "documentation":
        out = coding_agent.generate_docs(wid, t, state.get("requirements", {}), tasks)
    elif cat == "deployment":
        out = coding_agent.generate_deployment(wid, t)
    else:  # analysis or unknown
        out = coding_agent.generate_analysis(wid, t, state.get("requirements", {}), tasks)
    return out


def execute_node(state: WorkflowState) -> dict:
    tasks, wid = state["tasks"], state["workflow_id"]
    t = _task(state, state["current_task_id"])
    t["attempts"] = t.get("attempts", 0) + 1
    msgs = []
    memory_context = enhancements.retrieve_memory(wid, t) if t.get("category") == "development" else None
    if t.get("memory_hits"):
        msgs.append(_msg("rag_memory", t.get("assigned_agent") or coding_agent.NAME, "memory_retrieved", task_id=t["task_id"], hits=t["memory_hits"]))
    try:
        out = RecoveryController().retry_task(lambda: _produce(state, t, memory_context), workflow_id=wid, task=t,
                                              operation=f"execute:{t.get('category')}")
        t["artifacts"] = sorted(set(t.get("artifacts", [])) | set(out["files"]))
        t["review_files"] = out.get("review_files", out["files"])
        t["generation_backend"] = out["backend"]
        t["error"] = None
        msg = _msg(t.get("assigned_agent") or coding_agent.NAME, qa_agent.NAME, "artifacts_ready", task_id=t["task_id"],
                   files=out["files"], backend=out["backend"], attempt=t["attempts"])
    except Exception as e:  # agent failure after recovery attempts -> recorded, workflow continues
        cause = e.__cause__ if isinstance(e, RecoveryExhausted) and e.__cause__ else e
        t["error"] = redact(f"{type(cause).__name__}: {cause}")
        t["review_files"] = []
        log.error("execution failed for %s: %s", t["task_id"], traceback.format_exc(limit=3))
        msg = _msg(coding_agent.NAME, "orchestrator", "agent_error", task_id=t["task_id"], error=t["error"], escalated=t.get("escalated", False))
    return {"tasks": tasks, "messages": [*msgs, msg]}


def debate_node(state: WorkflowState) -> dict:
    tasks, wid = state["tasks"], state["workflow_id"]
    t = _task(state, state["current_task_id"])
    res = enhancements.run_debate(wid, t)
    if not res:
        return {"tasks": tasks, "messages": []}
    return {"tasks": tasks, "messages": [_msg("judge_agent", qa_agent.NAME, "debate_finished", task_id=t["task_id"], **res)]}


def qa_node(state: WorkflowState) -> dict:
    tasks = state["tasks"]
    t = _task(state, state["current_task_id"])
    if t.get("error"):
        report = {"passed": False, "kind": "error", "feedback": [t["error"]]}
    elif t.get("category") in ("development", "testing"):
        report = qa_agent.review(t.get("review_files", []), t.get("test_targets"))
    else:
        missing = [f for f in t.get("review_files", []) if not (settings.workspace_dir / f).is_file() or (settings.workspace_dir / f).stat().st_size == 0]
        report = {"passed": not missing and bool(t.get("review_files")), "kind": "artifact_check",
                  "feedback": [f"missing or empty: {m}" for m in missing]}
    if report.get("passed") and INJECTOR.force_qa_rejection(t["task_id"]):
        report = {**report, "passed": False, "simulated": True, "feedback": ["simulated QA rejection (fault injection)"]}
    t["qa_report"] = report
    wid, msgs = state["workflow_id"], []
    budget = settings.max_qa_retries + t.get("extra_attempts", 0)
    retry = (not report["passed"] and t.get("category") == "development" and not t.get("error") and t["attempts"] <= budget)
    if not report["passed"]:
        f = detect_failure(qa_report=report)
        record_recovery_event(workflow_id=wid, task_id=t["task_id"], failure_type=f.type, action="detected", status="failed",
                              attempt=t["attempts"], agent_id=t.get("assigned_agent"), detail={"feedback": report.get("feedback", [])[:3]},
                              simulated=f.simulated)
        # safe reassignment after the QA retry budget is spent (once per task, RECOVERY_MODE=full)
        if (not retry and settings.recovery_mode == "full" and t.get("category") == "development" and not t.get("error")
                and not t.get("qa_reassigned")):
            ctl = RecoveryController()
            prev = t.get("assigned_agent")
            new = ctl.reassign_task(t, {prev})
            record_recovery_event(workflow_id=wid, task_id=t["task_id"], failure_type=f.type, action="reassign",
                                  status="success" if new else "failed", attempt=t["attempts"], agent_id=new,
                                  detail={"from": prev, "to": new, "reason": "QA retry budget exhausted"}, simulated=f.simulated)
            if new:
                t["qa_reassigned"], t["extra_attempts"], retry = True, 1, True
                msgs.append(_msg("recovery_controller", new, "task_reassigned", task_id=t["task_id"], previous=prev))
    if report["passed"]:
        t["status"] = TaskStatus.COMPLETED.value
        if t.get("attempts", 1) > 1 or t.get("qa_reassigned"):
            cause = "worker_crash" if t.get("interrupted") and not t.get("qa_reassigned") else "qa_rejection"
            record_recovery_event(workflow_id=wid, task_id=t["task_id"], failure_type=cause, action="recovered", status="success",
                                  attempt=t["attempts"], agent_id=t.get("assigned_agent"),
                                  detail={"reexecuted_after_checkpoint_restore": bool(t.get("interrupted"))})
        enhancements.save_memory(wid, t)
    elif retry:
        t["status"] = TaskStatus.QA_FAILED.value
    else:
        t["status"] = TaskStatus.FAILED.value
        t["error"] = t.get("error") or "; ".join(report.get("feedback", [])[:3]) or "QA failed"
        if settings.recovery_mode in ("checkpoint", "full") and not t.get("escalated"):
            RecoveryController().escalate_failure(wid, t, detect_failure(qa_report=report) or detect_failure(Exception(t["error"])),
                                                  "task failed after all retries; human review required")
    enhancements.record_debate_qa(t)
    if t["status"] in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
        enhancements.record_execution(wid, t)
    kind = "qa_passed" if report["passed"] else ("qa_feedback" if retry else "qa_rejected")
    return {"tasks": tasks, "messages": [*msgs, _msg(qa_agent.NAME, coding_agent.NAME if retry else "orchestrator", kind,
                                                     task_id=t["task_id"], attempt=t["attempts"], feedback=report.get("feedback", [])[:5])]}


def finalize_node(state: WorkflowState) -> dict:
    tasks = state["tasks"]
    wid = state["workflow_id"]
    ws = coding_agent.workspace(wid)
    suite = qa_agent.run_suite(ws) if any(ws.glob("test_*.py")) else None
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    artifacts = sorted({a for t in tasks for a in t.get("artifacts", [])})
    completed = counts.get("completed", 0)
    status = WorkflowStatus.AWAITING_APPROVAL.value if completed else WorkflowStatus.FAILED.value
    summary = {**state.get("summary", {}), "task_counts": counts, "total_tasks": len(tasks),
               "completion_pct": round(100 * completed / len(tasks), 1) if tasks else 0.0, "artifacts": artifacts,
               "final_test_suite": (suite or {}).get("tests"), "final_suite_passed": (suite or {}).get("passed"),
               "workspace": str(ws), "llm_backend": "llm" if any(t.get("generation_backend") == "llm" for t in tasks) else "template",
               "enhancements": {
                   "memory_hits": sum(len(t.get("memory_hits", [])) for t in tasks),
                   "memories_saved": sum(bool((t.get("memory_saved") or {}).get("saved")) for t in tasks),
                   "debates": sum(bool(t.get("debate_id")) for t in tasks),
                   "high_risk_tasks": sum(t.get("risk_category") == "High" for t in tasks),
                   "escalated_tasks": [t["task_id"] for t in tasks if t.get("escalated")],
                   "reassigned_tasks": [t["task_id"] for t in tasks if t.get("reassignments") or t.get("qa_reassigned")],
                   "restored_from_checkpoint": state.get("restored_from"),
                   "recovery_mode": settings.recovery_mode}}
    (ws / "workflow_summary.md").write_text(_summary_md(state, summary), encoding="utf-8")
    return {"tasks": tasks, "status": status, "summary": summary, "current_task_id": None,
            "messages": [_msg("orchestrator", "human", "approval_requested" if completed else "workflow_failed", **{k: summary[k] for k in ("task_counts", "completion_pct")})]}


def _summary_md(state: WorkflowState, s: dict) -> str:
    lines = [f"# Workflow {state['workflow_id']}", "", f"**Requirement:** {state['requirement']}", "",
             f"- Scheduler: {s.get('scheduler')}", f"- Dependency method: {s.get('dependency_method')} (encoder: {s.get('encoder_backend')})",
             f"- Completion: {s['completion_pct']}% {s['task_counts']}", f"- Final test suite: {s.get('final_test_suite')}", "",
             "| Task | Category | Priority | Agent | Status | Attempts |", "|---|---|---|---|---|---|"]
    for t in state["tasks"]:
        lines.append(f"| {t['task_id']} {t['title']} | {t.get('category')} | {t.get('priority')} | {t.get('assigned_agent')} | {t['status']} | {t.get('attempts', 0)} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- routing
def route_after_select(state: WorkflowState) -> str:
    return "execute" if state.get("current_task_id") else "finalize"


def route_after_qa(state: WorkflowState) -> str:
    t = _task(state, state["current_task_id"])
    return "execute" if t["status"] == TaskStatus.QA_FAILED.value else "select_task"


# ---------------------------------------------------------------- graph
def _persisting(name: str, fn: Callable[[WorkflowState], dict]) -> Callable[[WorkflowState], dict]:
    def wrapper(state: WorkflowState) -> dict:
        # worker-crash fault injection happens OUTSIDE the stage error handler: it aborts the graph run
        INJECTOR.check("node", node=name)
        try:
            update = fn(state)
        except Exception as e:  # unrecoverable stage failure (e.g. models missing)
            err = redact(f"{name}: {type(e).__name__}: {e}")
            log.error(err)
            update = {"status": WorkflowStatus.FAILED.value, "errors": [*state.get("errors", []), err], "current_task_id": None, "queue": [],
                      "messages": [_msg("orchestrator", "human", "stage_failed", stage=name, error=err)]}
        new_msgs = update.get("messages", [])
        update["messages"] = [*state.get("messages", []), *new_msgs]
        merged = {**state, **update}
        _durable_save(name, merged, new_msgs)
        return update
    return wrapper


def _durable_save(node: str, merged: dict, new_msgs: list[dict]) -> None:
    """Persist tasks, events and a durable checkpoint, retrying transient DB outages (bounded)."""
    def write():
        INJECTOR.check("persist")
        repo.save_state(merged)
        repo.add_events(merged["workflow_id"], new_msgs)
        if settings.recovery_mode in ("checkpoint", "full"):
            checkpoint_manager.save_checkpoint(merged, node)
    mode = settings.recovery_mode if settings.recovery_mode != "none" else "none"
    try:
        RecoveryController(mode).retry_task(write, workflow_id=merged["workflow_id"], operation=f"persist:{node}")
    except RecoveryExhausted as e:  # keep running in memory; the next node's save will carry the full state
        log.error("persistence failed after node %s: %s", node, e)


def _guard(next_node: str):
    return lambda s: END if s.get("status") == WorkflowStatus.FAILED.value else next_node


def build_graph(checkpointer=None, entry: str = "scope"):
    g = StateGraph(WorkflowState)
    for name, fn in [("scope", scope_node), ("classify", classify_node), ("plan", plan_node), ("prioritize", prioritize_node),
                     ("allocate", allocate_node), ("select_task", select_task_node), ("execute", execute_node), ("debate", debate_node),
                     ("qa", qa_node), ("finalize", finalize_node)]:
        g.add_node(name, _persisting(name, fn))
    g.add_edge(START, entry)
    for a, b in [("scope", "classify"), ("classify", "plan"), ("plan", "prioritize"), ("prioritize", "allocate"), ("allocate", "select_task")]:
        g.add_conditional_edges(a, _guard(b), [b, END])
    g.add_conditional_edges("select_task", route_after_select, ["execute", "finalize"])
    g.add_edge("execute", "debate")
    g.add_edge("debate", "qa")
    g.add_conditional_edges("qa", route_after_qa, ["execute", "select_task"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


_GRAPHS: dict[str, object] = {}


def graph(entry: str = "scope"):
    if entry not in _GRAPHS:
        _GRAPHS[entry] = build_graph(entry=entry)
    return _GRAPHS[entry]


def new_workflow_id() -> str:
    return "wf_" + uuid.uuid4().hex[:12]


def run_workflow(requirement: str, workflow_id: str | None = None, ticket: dict | None = None) -> dict:
    wid = workflow_id or new_workflow_id()
    if repo.get_workflow(wid, with_details=False) is None:
        repo.create_workflow(wid, requirement)
    init: WorkflowState = {"workflow_id": wid, "requirement": requirement, "status": WorkflowStatus.RUNNING.value,
                           "tasks": [], "messages": [], "errors": [], "summary": {}}
    if ticket:
        init["ticket"] = ticket
    return _run_with_recovery(wid, init)


def _run_with_recovery(wid: str, init: dict) -> dict:
    """Invoke the graph; on a worker crash restore the latest durable checkpoint and resume (bounded)."""
    state, entry, resumes = init, "scope", 0
    while True:
        try:
            return graph(entry).invoke(state, config={"configurable": {"thread_id": f"{wid}:{resumes}"}, "recursion_limit": 500})
        except Exception as e:
            f = detect_failure(e)
            crash_like = isinstance(e, SimulatedFault) or f.retryable
            record_recovery_event(workflow_id=wid, task_id=None, failure_type="worker_crash" if crash_like else f.type, action="detected",
                                  status="failed", attempt=resumes + 1, detail={"error": f.message}, simulated=f.simulated)
            if not crash_like or settings.recovery_mode not in ("checkpoint", "full") or resumes >= settings.max_workflow_resumes:
                repo.set_status(wid, WorkflowStatus.FAILED.value)
                record_recovery_event(workflow_id=wid, task_id=None, failure_type="worker_crash", action="gave_up", status="failed",
                                      detail={"error": f.message, "resumes": resumes}, simulated=f.simulated)
                raise
            resumes += 1
            t0 = time.perf_counter()
            try:
                state, entry = checkpoint_manager.restore_checkpoint(wid), "select_task"
                # checkpoints taken before allocation cannot resume mid-way: restart from the beginning
                if not state.get("execution_order"):
                    state, entry = {**init, "messages": []}, "scope"
            except LookupError:
                state, entry = {**init, "messages": []}, "scope"
            state["resumes"] = resumes
            record_recovery_event(workflow_id=wid, task_id=None, failure_type="worker_crash", action="checkpoint_restore", status="success",
                                  attempt=resumes, duration_ms=round((time.perf_counter() - t0) * 1000, 1), simulated=f.simulated,
                                  detail={"restored_from": state.get("restored_from"), "resume_entry": entry,
                                          "requeued": [t["task_id"] for t in state.get("tasks", []) if t.get("interrupted")]})


def resume_workflow(workflow_id: str) -> dict:
    """Manually resume a crashed/failed workflow from its latest durable checkpoint."""
    state = checkpoint_manager.restore_checkpoint(workflow_id)
    repo.set_status(workflow_id, WorkflowStatus.RUNNING.value)
    record_recovery_event(workflow_id=workflow_id, task_id=None, failure_type="manual", action="checkpoint_restore", status="success",
                          detail={"restored_from": state.get("restored_from")})
    entry = "select_task" if state.get("execution_order") else "scope"
    return graph(entry).invoke(state, config={"configurable": {"thread_id": f"{workflow_id}:manual{uuid.uuid4().hex[:6]}"},
                                              "recursion_limit": 500})


def approve(workflow_id: str, approved: bool, publish_to_github: bool = False, reviewer: str = "user") -> dict:
    wf = repo.get_workflow(workflow_id)
    if wf is None:
        raise KeyError(workflow_id)
    if wf["status"] != WorkflowStatus.AWAITING_APPROVAL.value:
        raise ValueError(f"workflow is '{wf['status']}', not awaiting approval")
    result: dict = {"approved": approved, "reviewer": reviewer, "github": None}
    if approved and publish_to_github:
        files = (wf.get("summary") or {}).get("artifacts", [])
        # idempotent external write: never re-run a PR creation that succeeded or whose outcome is unknown
        result["github"] = RecoveryController().run_external_once(
            f"{workflow_id}:github_publish", "github_publish",
            lambda: github_agent.publish(workflow_id, files, f"Automated artifacts for: {wf['requirement']}", approved=True),
            workflow_id=workflow_id)
    status = WorkflowStatus.COMPLETED.value if approved else "rejected"
    repo.set_status(workflow_id, status, approval=result)
    repo.add_events(workflow_id, [_msg("human", "orchestrator", "approval_decision", approved=approved, github=result["github"])])
    return result
