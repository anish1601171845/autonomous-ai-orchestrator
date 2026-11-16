"""SQLAlchemy ORM models (SQLite by default; set DATABASE_URL=postgresql+psycopg://... for PostgreSQL).

Core tables: workflows, tasks, events.
AegisFlow enhancement tables (added by migration 1):
agent_memories, memory_retrievals, task_risk_predictions, execution_metrics, agent_debates,
debate_messages, workflow_checkpoints, recovery_events, external_operations.
New timestamps are timezone-aware (UTC).
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

Base = declarative_base()


def now_utc() -> datetime:
    return datetime.now(UTC)


class WorkflowRow(Base):
    __tablename__ = "workflows"
    id = Column(String(40), primary_key=True)
    requirement = Column(Text, nullable=False)
    status = Column(String(30), nullable=False, index=True)
    created_at = Column(String(40), nullable=False)
    updated_at = Column(String(40), nullable=False)
    state = Column(JSON, nullable=False, default=dict)
    summary = Column(JSON, nullable=True)
    approval = Column(JSON, nullable=True)


class TaskRow(Base):
    __tablename__ = "tasks"
    pk = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(40), ForeignKey("workflows.id", ondelete="CASCADE"), index=True, nullable=False)
    task_id = Column(String(20), nullable=False)
    title = Column(String(200))
    description = Column(Text)
    feature = Column(String(120))
    category = Column(String(30), index=True)
    category_confidence = Column(Float)
    priority = Column(String(20), index=True)
    priority_confidence = Column(Float)
    complexity = Column(Integer)
    status = Column(String(20), index=True)
    assigned_agent = Column(String(40), index=True)
    attempts = Column(Integer, default=0)
    dependency_ids = Column(JSON, default=list)
    artifacts = Column(JSON, default=list)
    qa_report = Column(JSON)
    error = Column(Text)
    # added by migration 2
    risk_probability = Column(Float)
    risk_category = Column(String(10))


class EventRow(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(40), index=True, nullable=False)
    timestamp = Column(String(40), nullable=False)
    sender = Column(String(40))
    recipient = Column(String(40))
    kind = Column(String(40))
    content = Column(JSON)


# ------------------------------------------------------------------ enhancement tables
class AgentMemoryRow(Base):
    __tablename__ = "agent_memories"
    memory_id = Column(String(40), primary_key=True)
    workflow_id = Column(String(40), ForeignKey("workflows.id", ondelete="SET NULL"), index=True, nullable=True)
    task_pk = Column(Integer, ForeignKey("tasks.pk", ondelete="SET NULL"), nullable=True)
    task_id = Column(String(20))
    agent_id = Column(String(40), index=True)
    project_id = Column(String(80), index=True, default="default")
    task_description = Column(Text, nullable=False)
    solution_summary = Column(Text, nullable=False)
    verified_outcome = Column(String(30), nullable=False)
    task_category = Column(String(30), index=True)
    source_reference = Column(Text)
    allowed_roles = Column(JSON, default=list)  # empty = visible to every role
    content_hash = Column(String(64), unique=True, nullable=False)
    embedding_model_version = Column(String(120))
    retrieval_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)


class MemoryRetrievalRow(Base):
    __tablename__ = "memory_retrievals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(40), ForeignKey("workflows.id", ondelete="CASCADE"), index=True, nullable=True)
    task_id = Column(String(20))
    agent_id = Column(String(40))
    memory_id = Column(String(40), ForeignKey("agent_memories.memory_id", ondelete="CASCADE"), index=True)
    query = Column(Text)
    score = Column(Float)
    rank = Column(Integer)
    backend = Column(String(80))
    latency_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)


class TaskRiskPredictionRow(Base):
    __tablename__ = "task_risk_predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(40), ForeignKey("workflows.id", ondelete="CASCADE"), index=True, nullable=False)
    task_pk = Column(Integer, ForeignKey("tasks.pk", ondelete="SET NULL"), nullable=True)
    task_id = Column(String(20), nullable=False)
    agent_id = Column(String(40))
    failure_probability = Column(Float)
    deadline_miss_probability = Column(Float)
    risk_category = Column(String(10), index=True)
    top_factors = Column(JSON)
    explanation = Column(Text)
    model_version = Column(String(80))
    features = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)


class ExecutionMetricRow(Base):
    """Execution history: pre-execution features + observed outcomes (training data for the failure model)."""
    __tablename__ = "execution_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(40), ForeignKey("workflows.id", ondelete="CASCADE"), index=True, nullable=False)
    task_pk = Column(Integer, ForeignKey("tasks.pk", ondelete="SET NULL"), nullable=True)
    task_id = Column(String(20), nullable=False)
    agent_id = Column(String(40))
    task_category = Column(String(30))
    task_priority = Column(String(20))
    task_complexity = Column(Integer)
    dependency_count = Column(Integer)
    estimated_duration = Column(Float)
    current_workload = Column(Integer)
    deadline_s = Column(Float)
    actual_duration_s = Column(Float)
    attempts = Column(Integer)
    task_failed = Column(Boolean)
    deadline_missed = Column(Boolean)
    source = Column(String(20), default="live")  # live | synthetic
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)


class AgentDebateRow(Base):
    __tablename__ = "agent_debates"
    debate_id = Column(String(40), primary_key=True)
    workflow_id = Column(String(40), ForeignKey("workflows.id", ondelete="CASCADE"), index=True, nullable=True)
    task_pk = Column(Integer, ForeignKey("tasks.pk", ondelete="SET NULL"), nullable=True)
    task_id = Column(String(20))
    status = Column(String(20))
    rounds = Column(Integer, default=0)
    revisions = Column(Integer, default=0)
    stop_reason = Column(String(60))
    judge_verdict = Column(String(20))
    qa_passed = Column(Boolean, nullable=True)
    tokens_used = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)
    latency_ms = Column(Float)
    backend = Column(String(30))
    task_context = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)


class DebateMessageRow(Base):
    __tablename__ = "debate_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    debate_id = Column(String(40), ForeignKey("agent_debates.debate_id", ondelete="CASCADE"), index=True, nullable=False)
    round = Column(Integer)
    role = Column(String(20))
    kind = Column(String(30))
    content = Column(Text)
    payload = Column(JSON)
    tokens = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)


class WorkflowCheckpointRow(Base):
    __tablename__ = "workflow_checkpoints"
    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(40), ForeignKey("workflows.id", ondelete="CASCADE"), index=True, nullable=False)
    seq = Column(Integer, nullable=False)
    node = Column(String(40))
    state = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    __table_args__ = (UniqueConstraint("workflow_id", "seq", name="uq_checkpoint_seq"),)


class RecoveryEventRow(Base):
    __tablename__ = "recovery_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(40), ForeignKey("workflows.id", ondelete="CASCADE"), index=True, nullable=True)
    task_pk = Column(Integer, ForeignKey("tasks.pk", ondelete="SET NULL"), nullable=True)
    task_id = Column(String(20))
    agent_id = Column(String(40))
    failure_type = Column(String(40), index=True)
    action = Column(String(40))  # detected | retry | reassign | checkpoint_restore | escalate | recovered | gave_up
    attempt = Column(Integer)
    status = Column(String(20))
    detail = Column(JSON)
    duration_ms = Column(Float)
    simulated = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)


class ExternalOperationRow(Base):
    """Idempotency ledger for irreversible external writes (GitHub branch/PR, deployments)."""
    __tablename__ = "external_operations"
    idempotency_key = Column(String(120), primary_key=True)
    workflow_id = Column(String(40), ForeignKey("workflows.id", ondelete="CASCADE"), index=True, nullable=True)
    operation = Column(String(60))
    status = Column(String(20))  # pending | succeeded | failed | unknown
    result = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)


_kw = {"connect_args": {"check_same_thread": False}} if settings.database_url.startswith("sqlite") else {"pool_pre_ping": True}
engine = create_engine(settings.database_url, **_kw)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    from app.database.migrations import migrate

    migrate(engine)
