from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from app.agents import github_agent, planning_agent, workload_agent
from app.api.schemas import CategoryRequest, DependencyRequest, DependencyResponse, Prediction, PriorityRequest
from app.core.config import settings
from app.database import repository as repo
from app.ml import classification, gnn_model, priority_prediction
from app.rl import scheduler

router = APIRouter(prefix="/api", tags=["models & system"])


def _models_ready() -> dict:
    return {"task_classifier": classification.MODEL_PATH.exists(), "priority_model": priority_prediction.MODEL_PATH.exists(),
            "gnn_dependency": gnn_model.CKPT.exists(), "rl_scheduler": scheduler.PPO_PATH.exists(),
            "requirements_classifier (real)": (settings.models_dir / "requirements_classifier.joblib").exists(),
            "ticket_triage (real)": (settings.models_dir / "ticket_triage.joblib").exists()}


@router.get("/health")
def health():
    models = _models_ready()
    return {"status": "ok" if models["task_classifier"] and models["priority_model"] else "degraded", "models": models,
            "llm_configured": bool(settings.llm_api_key), "github": github_agent.status(), "scheduler_auto_choice": scheduler.best_policy()}


@router.post("/predict/category", response_model=list[Prediction])
def predict_category(body: CategoryRequest):
    try:
        return [Prediction(label=lbl, confidence=c) for lbl, c in classification.predict(body.descriptions)]
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from None


@router.post("/predict/priority", response_model=list[Prediction])
def predict_priority(body: PriorityRequest):
    try:
        recs = [t.model_dump() for t in body.tasks]
        missing = [i for i, r in enumerate(recs) if not r["category"]]
        if missing:
            for i, (lbl, _) in zip(missing, classification.predict([recs[i]["description"] for i in missing])):
                recs[i]["category"] = lbl
        return [Prediction(label=lbl, confidence=c) for lbl, c in priority_prediction.predict(recs)]
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from None


@router.post("/dependencies/predict", response_model=DependencyResponse)
def predict_dependencies(body: DependencyRequest):
    tasks = [t.model_dump() for t in body.tasks]
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise HTTPException(422, "task_id values must be unique")
    missing = [t for t in tasks if not t["category"]]
    if missing:
        for t, (lbl, _) in zip(missing, classification.predict([t["description"] for t in missing])):
            t["category"] = lbl
    plan = planning_agent.run(tasks)
    return DependencyResponse(method=plan["method"], edges=plan["edges"], rejected_edges=plan["rejected_edges"],
                              execution_order=planning_agent.execution_order(tasks), is_dag=plan["is_dag"])


@router.get("/agents")
def agents():
    tasks = repo.list_tasks(limit=5000)
    out = []
    for a in workload_agent.agents():
        mine = [t for t in tasks if t["assigned_agent"] == a["name"]]
        active = [t for t in mine if t["status"] in ("assigned", "in_progress", "qa_failed")]
        out.append({**a, "status": "busy" if active else "idle", "current_workload": len(active), "total_assigned": len(mine),
                    "completed": sum(t["status"] == "completed" for t in mine), "failed": sum(t["status"] == "failed" for t in mine),
                    "recent_tasks": [{"workflow_id": t["workflow_id"], "task_id": t["task_id"], "title": t["title"], "status": t["status"]} for t in mine[:10]]})
    return out


@router.get("/metrics")
def metrics():
    reports = {}
    for name in ("classification_metrics", "priority_metrics", "gnn_metrics", "rl_metrics", "semantic_metrics", "dataset_report",
                 "requirements_metrics", "ticket_metrics"):
        p = settings.reports_dir / f"{name}.json"
        if p.exists():
            reports[name] = json.loads(p.read_text(encoding="utf-8"))
    return {"runtime": repo.metrics(), "models": reports}


@router.get("/github")
def github():
    return github_agent.overview()
