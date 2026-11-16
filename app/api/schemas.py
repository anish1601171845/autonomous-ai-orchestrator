"""Pydantic request/response schemas for the REST API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.config import CATEGORIES, PRIORITIES


class WorkflowCreate(BaseModel):
    requirement: str = Field(..., min_length=10, max_length=5000,
                             examples=["Build a student management system with registration, login, database integration, "
                                       "student dashboard, and automated testing."])
    run_async: bool = Field(True, description="Run in the background and return immediately")


class WorkflowCreated(BaseModel):
    workflow_id: str
    status: str


class WorkflowSummary(BaseModel):
    workflow_id: str
    requirement: str
    status: str
    created_at: str
    updated_at: str
    summary: dict[str, Any] | None = None


class TaskOut(BaseModel):
    workflow_id: str
    task_id: str
    title: str | None = None
    description: str | None = None
    feature: str | None = None
    category: str | None = None
    category_confidence: float | None = None
    priority: str | None = None
    priority_confidence: float | None = None
    complexity: int | None = None
    status: str | None = None
    assigned_agent: str | None = None
    attempts: int | None = 0
    dependency_ids: list[str] | None = None
    artifacts: list[str] | None = None
    qa_report: dict[str, Any] | None = None
    error: str | None = None
    risk_probability: float | None = None
    risk_category: str | None = None


class WorkflowDetail(WorkflowSummary):
    approval: dict[str, Any] | None = None
    state: dict[str, Any] = {}
    tasks: list[TaskOut] = []
    events: list[dict[str, Any]] = []


class CategoryRequest(BaseModel):
    descriptions: list[str] = Field(..., min_length=1, max_length=200)


class Prediction(BaseModel):
    label: str
    confidence: float | None


class PriorityItem(BaseModel):
    description: str = Field(..., min_length=3)
    category: str | None = Field(None, description=f"one of {CATEGORIES}; predicted if omitted")
    complexity: int = Field(3, ge=1, le=5)
    urgency: int = Field(3, ge=1, le=5)
    business_impact: int = Field(3, ge=1, le=5)
    num_dependents: int = Field(0, ge=0)


class PriorityRequest(BaseModel):
    tasks: list[PriorityItem] = Field(..., min_length=1, max_length=200)


class DependencyTask(BaseModel):
    task_id: str
    description: str
    feature: str = "general"
    category: str | None = None
    complexity: int = Field(3, ge=1, le=5)


class DependencyRequest(BaseModel):
    tasks: list[DependencyTask] = Field(..., min_length=2, max_length=100)


class DependencyResponse(BaseModel):
    method: str
    edges: list[dict[str, Any]]
    rejected_edges: list[dict[str, Any]]
    execution_order: list[str]
    is_dag: bool


class ApprovalRequest(BaseModel):
    approved: bool
    publish_to_github: bool = False
    reviewer: str = Field("user", max_length=80)


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None


PRIORITY_LABELS = PRIORITIES


class RequirementsClassifyRequest(BaseModel):
    text: str | None = Field(None, description="A requirements document; one requirement per line or running prose")
    statements: list[str] | None = Field(None, max_length=500, description="Alternatively, individual requirement statements")


class TicketIn(BaseModel):
    subject: str | None = Field(None, max_length=500)
    body: str = Field(..., min_length=10, max_length=20000)
    language: str | None = Field(None, description="en, de, es, fr or pt (optional)")
    business_type: str | None = Field(None, examples=["IT Services", "Tech Online Store"])
    create_workflow: bool = Field(False, description="Also turn the ticket into an engineering workflow")
