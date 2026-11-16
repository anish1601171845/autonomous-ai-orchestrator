"""AegisFlow enhancement hooks called by the LangGraph orchestrator.

Each hook is independently switchable (MEMORY_ENABLED, RISK_ENABLED, DEBATE_ENABLED, RECOVERY_MODE)
and is fail-safe: an error inside an enhancement is logged and never breaks the core workflow.
"""
from __future__ import annotations

import time

from app.core.config import settings
from app.core.security import get_logger
from app.database import repository as repo

log = get_logger(__name__)


# ------------------------------------------------------------------ predictive failure detection
def assess_risk(workflow_id: str, tasks: list[dict], order: list[str], state_for_save: dict) -> dict:
    """Pre-execution risk for every task. Uses only features known before execution."""
    if not settings.risk_enabled:
        return {"enabled": False}
    from app.ml import failure_predictor as fp

    if fp.load_bundle() is None:
        return {"enabled": True, "available": False, "reason": "failure model not trained"}
    from app.database.models import SessionLocal, TaskRiskPredictionRow

    by_id = {t["task_id"]: t for t in tasks}
    load: dict[str, int] = {}
    records = []
    for tid in order:
        t = by_id[tid]
        agent = t.get("assigned_agent") or "unknown"
        rec = {"task_category": t.get("category"), "task_priority": t.get("priority"), "agent_id": agent,
               "task_complexity": t.get("complexity", 3), "dependency_count": len(t.get("dependency_ids", [])),
               "estimated_duration": t.get("estimated_duration_h", 7.5), "current_workload": load.get(agent, 0),
               "deadline_slack": 1.5}
        load[agent] = load.get(agent, 0) + 1
        t["_risk_features"] = rec
        records.append((t, rec))
    p_fail = fp.predict_failure_probability([r for _, r in records])
    p_late = fp.predict_deadline_risk([r for _, r in records])
    repo.save_state(state_for_save)  # make sure task rows exist so predictions can reference task_pk
    high = 0
    with SessionLocal.begin() as s:
        for (t, rec), pf, pl in zip(records, p_fail, p_late):
            target = "task_failed" if pf >= pl else "deadline_missed"
            exp = fp.explain_risk_prediction(rec, target)
            t["risk_probability"], t["deadline_risk"] = pf, pl
            t["risk_category"] = fp.risk_category(max(pf, pl))
            t["risk_explanation"] = exp["explanation"]
            high += t["risk_category"] == "High"
            s.add(TaskRiskPredictionRow(workflow_id=workflow_id, task_pk=repo.task_pk(workflow_id, t["task_id"]), task_id=t["task_id"],
                                        agent_id=rec["agent_id"], failure_probability=pf, deadline_miss_probability=pl,
                                        risk_category=t["risk_category"], top_factors=exp["factors"], explanation=exp["explanation"],
                                        model_version=fp.load_bundle()["version"], features=rec))
    return {"enabled": True, "available": True, "high_risk_tasks": high, "assessed": len(records)}


def record_execution(workflow_id: str, t: dict) -> None:
    """Append LIVE execution history (pre-execution features + observed outcome) for future retraining."""
    try:
        from app.database.models import ExecutionMetricRow, SessionLocal

        f = t.get("_risk_features") or {}
        started = t.get("_started_at") or time.time()
        actual = time.time() - started
        deadline = (t.get("complexity") or 3) * settings.task_deadline_s_per_complexity
        with SessionLocal.begin() as s:
            s.add(ExecutionMetricRow(workflow_id=workflow_id, task_pk=repo.task_pk(workflow_id, t["task_id"]), task_id=t["task_id"],
                                     agent_id=t.get("assigned_agent"), task_category=t.get("category"), task_priority=t.get("priority"),
                                     task_complexity=t.get("complexity"), dependency_count=f.get("dependency_count", len(t.get("dependency_ids", []))),
                                     estimated_duration=t.get("estimated_duration_h"), current_workload=f.get("current_workload"),
                                     deadline_s=deadline, actual_duration_s=round(actual, 3), attempts=t.get("attempts"),
                                     task_failed=t.get("status") != "completed", deadline_missed=actual > deadline, source="live"))
    except Exception as e:
        log.warning("could not record execution metric for %s: %s", t.get("task_id"), e)


# ------------------------------------------------------------------ RAG memory
def retrieve_memory(workflow_id: str, t: dict) -> str | None:
    if not settings.memory_enabled:
        return None
    try:
        from app.memory import rag_memory

        hits = rag_memory.retrieve_similar_solutions(t["description"], task_category=t.get("category"), agent_id=t.get("assigned_agent"),
                                                     workflow_id=workflow_id, task_id=t["task_id"])
        t["memory_hits"] = [{"memory_id": h["memory_id"], "score": h["score"]} for h in hits]
        return rag_memory.build_memory_context(hits) or None
    except Exception as e:
        log.warning("memory retrieval failed for %s: %s", t.get("task_id"), e)
        return None


def save_memory(workflow_id: str, t: dict) -> None:
    if not settings.memory_enabled or t.get("category") not in ("development", "testing"):
        return
    try:
        from app.memory import rag_memory

        api = ", ".join(f"{e['name']} ({e['kind']})" for e in t.get("explanation", [])[:8]) or "see artifacts"
        tests = (t.get("qa_report") or {}).get("tests") or {}
        summary = (f"{t.get('generation_backend', 'unknown')} solution. Public API: {api}. "
                   f"Sandboxed tests: {tests.get('passed', 0)} passed, {tests.get('failed', 0)} failed.")
        res = rag_memory.save_verified_solution(t["description"], summary, "qa_passed", task_id=t["task_id"], workflow_id=workflow_id,
                                                agent_id=t.get("assigned_agent"), task_category=t.get("category"),
                                                source_reference=", ".join(t.get("artifacts", [])[:6]),
                                                task_pk=repo.task_pk(workflow_id, t["task_id"]))
        t["memory_saved"] = res
    except Exception as e:
        log.warning("memory save failed for %s: %s", t.get("task_id"), e)


# ------------------------------------------------------------------ debate
def run_debate(workflow_id: str, t: dict) -> dict | None:
    if not settings.debate_enabled or t.get("category") != "development" or t.get("error"):
        return None
    try:
        from app.agents.debate_engine import DebateEngine

        t["_task_pk"] = repo.task_pk(workflow_id, t["task_id"])
        res = DebateEngine(workflow_id, t).run()
        t["debate_id"] = res.debate_id
        return {"debate_id": res.debate_id, "verdict": res.judge_verdict, "rounds": res.rounds, "revisions": res.revisions,
                "stop_reason": res.stop_reason, "tokens": res.tokens_used}
    except Exception as e:
        log.warning("debate failed for %s: %s", t.get("task_id"), e)
        return {"error": f"{type(e).__name__}: {e}"}


def record_debate_qa(t: dict) -> None:
    if t.get("debate_id"):
        try:
            from app.agents.debate_engine import record_qa_result

            tests = ((t.get("qa_report") or {}).get("tests") or {})
            record_qa_result(t["debate_id"], bool((t.get("qa_report") or {}).get("passed")),
                             f"{tests.get('passed', 0)} passed, {tests.get('failed', 0)} failed")
        except Exception as e:
            log.warning("could not record debate QA result: %s", e)
