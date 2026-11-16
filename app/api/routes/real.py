"""Endpoints backed by models trained on the REAL datasets (PROMISE requirements, multilingual tickets)."""
from __future__ import annotations

import json

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.agents import intake_agent
from app.agents.scoping_agent import split_requirements
from app.api.schemas import RequirementsClassifyRequest, TicketIn
from app.core import orchestrator
from app.core.config import settings
from app.core.security import get_logger
from app.database import repository as repo
from app.ml import real_data, requirements_classifier, ticket_models

router = APIRouter(prefix="/api", tags=["real datasets"])
log = get_logger(__name__)


@router.get("/datasets")
def datasets():
    out = {"available": real_data.available(),
           "models": {"requirements_classifier": requirements_classifier.MODEL_PATH.exists(),
                      "ticket_triage": ticket_models.MODEL_PATH.exists()}}
    for name in ("requirements_metrics", "ticket_metrics"):
        p = settings.reports_dir / f"{name}.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            out[name] = m
    return out


@router.post("/requirements/classify")
def classify_requirements(body: RequirementsClassifyRequest):
    statements = body.statements or (split_requirements(body.text) if body.text else [])
    if not statements:
        raise HTTPException(422, "provide `text` or `statements`")
    try:
        preds = requirements_classifier.predict(statements)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from None
    counts: dict[str, int] = {}
    for p in preds:
        counts[p["label_name"]] = counts.get(p["label_name"], 0) + 1
    return {"count": len(preds), "by_type": counts, "statements": preds}


def _run_ticket(requirement: str, wid: str, ticket: dict) -> None:
    try:
        orchestrator.run_workflow(requirement, wid, ticket=ticket)
    except Exception as e:
        log.error("ticket workflow %s crashed: %s", wid, e)
        repo.set_status(wid, "failed")


@router.post("/tickets/triage")
def triage_ticket(body: TicketIn, background: BackgroundTasks):
    ticket = body.model_dump(exclude={"create_workflow"})
    try:
        result = intake_agent.triage(ticket)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from None
    response = {"triage": result, "workflow_id": None}
    if body.create_workflow:
        wid = orchestrator.new_workflow_id()
        requirement = f"[ticket {result['type']}/{result['priority']} -> {result['queue']}] {body.subject or body.body[:120]}"
        repo.create_workflow(wid, requirement)
        repo.set_status(wid, "running")
        background.add_task(_run_ticket, requirement, wid, {**ticket, "triage": result})
        response["workflow_id"] = wid
    return response
