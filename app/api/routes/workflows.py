from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.api.schemas import ApprovalRequest, TaskOut, WorkflowCreate, WorkflowCreated, WorkflowDetail, WorkflowSummary
from app.core import orchestrator
from app.core.security import get_logger
from app.database import repository as repo

router = APIRouter(prefix="/api", tags=["workflows"])
log = get_logger(__name__)


def _run(requirement: str, wid: str) -> None:
    try:
        orchestrator.run_workflow(requirement, wid)
    except Exception as e:  # never leave a workflow stuck in 'running'
        log.error("workflow %s crashed: %s", wid, e)
        repo.set_status(wid, "failed")


@router.post("/workflows", response_model=WorkflowCreated, status_code=202)
def create_workflow(body: WorkflowCreate, background: BackgroundTasks):
    wid = orchestrator.new_workflow_id()
    repo.create_workflow(wid, body.requirement)
    repo.set_status(wid, "running")
    if body.run_async:
        background.add_task(_run, body.requirement, wid)
        return WorkflowCreated(workflow_id=wid, status="running")
    _run(body.requirement, wid)
    return WorkflowCreated(workflow_id=wid, status=repo.get_workflow(wid, with_details=False)["status"])


@router.get("/workflows", response_model=list[WorkflowSummary])
def list_workflows(limit: int = Query(50, ge=1, le=500)):
    return repo.list_workflows(limit)


@router.get("/workflows/{workflow_id}", response_model=WorkflowDetail)
def get_workflow(workflow_id: str):
    wf = repo.get_workflow(workflow_id)
    if wf is None:
        raise HTTPException(404, f"workflow {workflow_id} not found")
    return wf


@router.post("/workflows/{workflow_id}/approve")
def approve(workflow_id: str, body: ApprovalRequest):
    try:
        return orchestrator.approve(workflow_id, body.approved, body.publish_to_github, body.reviewer)
    except KeyError:
        raise HTTPException(404, f"workflow {workflow_id} not found") from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from None


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks(workflow_id: str | None = None, status: str | None = None, limit: int = Query(500, ge=1, le=5000)):
    return repo.list_tasks(workflow_id, status, limit)


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: str, workflow_id: str | None = None):
    t = repo.get_task(task_id, workflow_id)
    if t is None:
        raise HTTPException(404, f"task {task_id} not found")
    return t
