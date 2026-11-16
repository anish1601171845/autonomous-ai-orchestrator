"""REST endpoints for the AegisFlow enhancements: memory, risk, debate, self-healing recovery."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.config import CATEGORIES, PRIORITIES, settings
from app.database.models import SessionLocal, TaskRiskPredictionRow

router = APIRouter(prefix="/api", tags=["AegisFlow enhancements"])


class MemorySearch(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    task_category: str | None = None
    agent_id: str | None = None
    top_k: int = Field(5, ge=1, le=20)
    min_score: float = Field(0.0, ge=0, le=1)


class RiskRequest(BaseModel):
    task_category: str = Field("development", description=f"one of {CATEGORIES}")
    task_priority: str = Field("medium", description=f"one of {PRIORITIES}")
    agent_id: str = "coder-1"
    task_complexity: int = Field(3, ge=1, le=5)
    dependency_count: int = Field(0, ge=0)
    estimated_duration: float = Field(7.5, gt=0)
    current_workload: int = Field(0, ge=0)
    deadline_slack: float = Field(1.5, gt=0, description="deadline / estimated duration")


class FaultDemo(BaseModel):
    scenario: str = Field(..., description="api_timeout | rate_limit | network_error | agent_crash | db_outage | qa_rejection | repeated_failure | worker_crash")
    requirement: str = Field("Build a notes app with login, note search and automated testing.", min_length=10)
    times: int | None = Field(None, ge=1, le=20)


# ------------------------------------------------------------------ memory
@router.get("/memory")
def list_memory(limit: int = 200):
    from app.memory import rag_memory

    svc = rag_memory.service()
    return {"backend": svc.store.backend_name, "memories": svc.list_memories(limit)}


@router.post("/memory/search")
def search_memory(body: MemorySearch):
    from app.memory import rag_memory

    return rag_memory.retrieve_similar_solutions(body.query, body.top_k, task_category=body.task_category, agent_id=body.agent_id,
                                                 min_score=body.min_score, record=False)


@router.get("/memory/retrievals")
def memory_retrievals(limit: int = 200):
    from app.memory import rag_memory

    return rag_memory.service().retrieval_history(limit)


@router.delete("/memory/{memory_id}")
def remove_memory(memory_id: str):
    from app.memory import rag_memory

    if not rag_memory.delete_memory(memory_id):
        raise HTTPException(404, f"memory {memory_id} not found")
    return {"deleted": memory_id}


@router.post("/memory/rebuild")
def rebuild_memory_index():
    from app.memory import rag_memory

    return rag_memory.rebuild_index()


# ------------------------------------------------------------------ risk
@router.get("/risk/predictions")
def risk_predictions(workflow_id: str | None = None, limit: int = 500):
    with SessionLocal() as s:
        q = select(TaskRiskPredictionRow).order_by(TaskRiskPredictionRow.id.desc()).limit(limit)
        if workflow_id:
            q = q.where(TaskRiskPredictionRow.workflow_id == workflow_id)
        return [{"workflow_id": r.workflow_id, "task_id": r.task_id, "agent_id": r.agent_id, "failure_probability": r.failure_probability,
                 "deadline_miss_probability": r.deadline_miss_probability, "risk_category": r.risk_category, "top_factors": r.top_factors,
                 "explanation": r.explanation, "priority": (r.features or {}).get("task_priority"),
                 "category": (r.features or {}).get("task_category"), "model_version": r.model_version,
                 "created_at": r.created_at.isoformat() if r.created_at else None} for r in s.scalars(q)]


@router.get("/risk/metrics")
def risk_metrics():
    p = settings.reports_dir / "failure_metrics.json"
    if not p.exists():
        raise HTTPException(404, "failure model not trained; run `python scripts/evaluate_enhancements.py --only risk`")
    return json.loads(p.read_text(encoding="utf-8"))


@router.post("/risk/predict")
def risk_predict(body: RiskRequest):
    from app.ml import failure_predictor as fp

    rec = body.model_dump()
    try:
        return {"task_failed": fp.explain_risk_prediction(rec, "task_failed"), "deadline_missed": fp.explain_risk_prediction(rec, "deadline_missed"),
                "thresholds": {"medium": settings.risk_medium_threshold, "high": settings.risk_high_threshold}}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from None


# ------------------------------------------------------------------ debate
@router.get("/debates")
def debates(workflow_id: str | None = None, limit: int = 100):
    from app.agents import debate_engine

    return debate_engine.list_debates(limit, workflow_id)


@router.get("/debates/{debate_id}")
def debate_detail(debate_id: str):
    from app.agents import debate_engine

    d = debate_engine.get_debate_history(debate_id)
    if d is None:
        raise HTTPException(404, f"debate {debate_id} not found")
    return d


# ------------------------------------------------------------------ recovery
@router.get("/recovery/events")
def recovery_events(workflow_id: str | None = None, limit: int = 500):
    from app.recovery.recovery_controller import list_recovery_events

    return list_recovery_events(limit, workflow_id)


@router.get("/recovery/checkpoints/{workflow_id}")
def checkpoints(workflow_id: str):
    from app.recovery.checkpoint_manager import list_checkpoints

    return list_checkpoints(workflow_id)


@router.post("/recovery/demo")
def fault_demo(body: FaultDemo):
    """Run a workflow with a SIMULATED fault injected (clearly labelled in recovery_events)."""
    from app.core import orchestrator
    from app.recovery import fault_injection as fi

    if body.scenario not in fi.SCENARIOS:
        raise HTTPException(422, f"unknown scenario; choose from {sorted(fi.SCENARIOS)}")
    kw = {"times": body.times}
    if body.scenario == "agent_crash":
        kw["agent"] = "coder-1"
    if body.scenario == "repeated_failure":
        kw["task_id"] = "T04"
    if body.scenario == "worker_crash":
        kw["node"] = "qa"
    from app.database import repository as repo

    wid = orchestrator.new_workflow_id()
    repo.create_workflow(wid, body.requirement)
    # label the run without touching the requirement text (the scoping agent parses it)
    repo.add_events(wid, [{"sender": "fault_injector", "recipient": "orchestrator", "kind": "fault_demo_started",
                           "content": {"scenario": body.scenario, "simulated": True, **{k: v for k, v in kw.items() if v is not None}}}])
    with fi.inject(body.scenario, **kw) as spec:
        try:
            state = orchestrator.run_workflow(body.requirement, wid)
            outcome = state.get("status")
        except Exception as e:
            outcome = f"failed: {type(e).__name__}"
    from app.recovery.recovery_controller import list_recovery_events

    return {"workflow_id": wid, "scenario": body.scenario, "simulated": True, "faults_fired": spec.fired, "workflow_status": outcome,
            "recovery_events": list_recovery_events(200, wid)}


@router.post("/workflows/{workflow_id}/resume")
def resume(workflow_id: str):
    from app.core import orchestrator

    try:
        state = orchestrator.resume_workflow(workflow_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from None
    return {"workflow_id": workflow_id, "status": state.get("status"), "task_counts": (state.get("summary") or {}).get("task_counts")}


@router.get("/enhancements/evaluation")
def enhancement_evaluation():
    p = settings.reports_dir / "enhancement_evaluation.json"
    if not p.exists():
        raise HTTPException(404, "run `python scripts/evaluate_enhancements.py` first")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/enhancements/status")
def enhancement_status():
    from app.ml import failure_predictor as fp

    return {"memory": {"enabled": settings.memory_enabled, "backend_setting": settings.memory_backend},
            "risk": {"enabled": settings.risk_enabled, "model_trained": fp.MODEL_PATH.exists(),
                     "thresholds": {"medium": settings.risk_medium_threshold, "high": settings.risk_high_threshold}},
            "debate": {"enabled": settings.debate_enabled, "max_rounds": settings.debate_max_rounds,
                       "time_budget_s": settings.debate_time_budget_s, "token_budget": settings.debate_token_budget},
            "recovery": {"mode": settings.recovery_mode, "max_attempts": settings.retry_max_attempts,
                         "max_workflow_resumes": settings.max_workflow_resumes}}
