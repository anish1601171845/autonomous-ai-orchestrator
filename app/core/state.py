"""Typed shared state and structured messages passed between agents."""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    QA_FAILED = "qa_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class WorkflowStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    task_id: str
    title: str
    description: str
    feature: str = "general"
    category: str | None = None
    category_confidence: float | None = None
    priority: str | None = None
    priority_confidence: float | None = None
    complexity: int = Field(3, ge=1, le=5)
    urgency: int = Field(3, ge=1, le=5)
    business_impact: int = Field(3, ge=1, le=5)
    estimated_duration_h: float = 4.0
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)
    assigned_agent: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    artifacts: list[str] = Field(default_factory=list)
    qa_report: dict[str, Any] | None = None
    error: str | None = None


class AgentMessage(BaseModel):
    sender: str
    recipient: str
    kind: str
    content: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utcnow)


class DependencyEdge(BaseModel):
    source: str  # prerequisite task
    target: str  # dependent task
    score: float
    method: str


class WorkflowState(TypedDict, total=False):
    workflow_id: str
    requirement: str
    ticket: dict[str, Any] | None
    requirements: dict[str, Any]
    tasks: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    execution_order: list[str]
    assignments: dict[str, str]
    scheduler_used: str
    current_task_id: str | None
    queue: list[str]
    messages: list[dict[str, Any]]
    errors: list[str]
    status: str
    summary: dict[str, Any]
    restored_from: dict[str, Any] | None
    resumes: int
