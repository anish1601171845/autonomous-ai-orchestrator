"""Self-healing recovery controller.

Strategies (RECOVERY_MODE):
  none        - no recovery: the first failure fails the task / workflow
  fixed_retry - bounded retries with a fixed delay, same agent, no checkpoint resume
  checkpoint  - bounded retries with exponential backoff + full jitter, and resume from durable checkpoints
  full        - checkpoint + safe reassignment to another capable agent + human escalation

Safety
* Retries are bounded (RETRY_MAX_ATTEMPTS); escalation follows exhaustion.
* Only retryable failure classes are retried (timeouts, rate limits, network, crashes, DB outages).
* External writes (GitHub branch/PR, deployments) go through `run_external_once` with an idempotency
  key: a succeeded operation is never repeated, and an operation whose outcome is unknown is NEVER
  retried automatically - it is escalated for reconciliation.
* Every detection / retry / reassignment / restore / escalation is written to recovery_events.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sqlalchemy import select

from app.core.config import settings
from app.core.security import get_logger, redact
from app.database.models import ExternalOperationRow, RecoveryEventRow, SessionLocal, now_utc

log = get_logger(__name__)


class FailureType(StrEnum):
    LLM_TIMEOUT = "llm_timeout"
    RATE_LIMIT = "rate_limit"
    AGENT_TIMEOUT = "agent_timeout"
    WORKER_CRASH = "worker_crash"
    AGENT_CRASH = "agent_unavailable"
    NETWORK = "network_error"
    DB_UNAVAILABLE = "db_unavailable"
    QA_REJECTION = "qa_rejection"
    NON_RETRYABLE = "non_retryable"


RETRYABLE = {FailureType.LLM_TIMEOUT, FailureType.RATE_LIMIT, FailureType.AGENT_TIMEOUT, FailureType.NETWORK,
             FailureType.DB_UNAVAILABLE, FailureType.AGENT_CRASH, FailureType.WORKER_CRASH}


@dataclass
class Failure:
    type: FailureType
    retryable: bool
    message: str
    simulated: bool = False
    retry_after: float | None = None


class RecoveryExhausted(RuntimeError):
    def __init__(self, failure: Failure, attempts: int):
        super().__init__(f"{failure.type}: gave up after {attempts} attempt(s): {failure.message}")
        self.failure, self.attempts = failure, attempts


class NeedsReconciliation(RuntimeError):
    pass


# ------------------------------------------------------------------ detection / classification
def classify_failure(exc: BaseException) -> FailureType:
    from app.recovery import fault_injection as fi

    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if isinstance(exc, fi.SimulatedRateLimit) or "429" in msg or "rate limit" in msg or "ratelimit" in name:
        return FailureType.RATE_LIMIT
    if isinstance(exc, fi.SimulatedAgentCrash) or "agent unavailable" in msg:
        return FailureType.AGENT_CRASH
    if isinstance(exc, fi.SimulatedWorkerCrash):
        return FailureType.WORKER_CRASH
    if isinstance(exc, fi.SimulatedDBOutage) or "operationalerror" in name or "database is locked" in msg:
        return FailureType.DB_UNAVAILABLE
    if isinstance(exc, fi.SimulatedLLMTimeout) or "llmunavailable" in name or ("timeout" in name and "llm" in msg):
        return FailureType.LLM_TIMEOUT
    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in msg:
        return FailureType.AGENT_TIMEOUT
    if isinstance(exc, ConnectionError) or "connect" in name or "network" in msg:
        return FailureType.NETWORK
    return FailureType.NON_RETRYABLE


def detect_failure(exc: BaseException | None = None, qa_report: dict | None = None) -> Failure | None:
    if exc is not None:
        ft = classify_failure(exc)
        return Failure(ft, ft in RETRYABLE, redact(f"{type(exc).__name__}: {exc}")[:500],
                       simulated=getattr(exc, "simulated", False), retry_after=getattr(exc, "retry_after", None))
    if qa_report is not None and not qa_report.get("passed", True):
        return Failure(FailureType.QA_REJECTION, True, "; ".join(qa_report.get("feedback", [])[:3])[:500],
                       simulated=bool(qa_report.get("simulated")))
    return None


# ------------------------------------------------------------------ events
def record_recovery_event(*, workflow_id: str | None, task_id: str | None, failure_type: str, action: str, status: str,
                          agent_id: str | None = None, attempt: int | None = None, detail: dict | None = None,
                          duration_ms: float | None = None, simulated: bool = False) -> None:
    try:
        from app.database import repository as repo

        with SessionLocal.begin() as s:
            s.add(RecoveryEventRow(workflow_id=workflow_id, task_pk=repo.task_pk(workflow_id, task_id), task_id=task_id,
                                   agent_id=agent_id, failure_type=str(failure_type), action=action, attempt=attempt, status=status,
                                   detail=detail or {}, duration_ms=duration_ms, simulated=simulated))
    except Exception as e:  # never let audit logging crash recovery (e.g. during a DB outage)
        log.error("could not persist recovery event (%s): %s %s %s", type(e).__name__, workflow_id, task_id, action)


def list_recovery_events(limit: int = 500, workflow_id: str | None = None) -> list[dict]:
    with SessionLocal() as s:
        q = select(RecoveryEventRow).order_by(RecoveryEventRow.id.desc()).limit(limit)
        if workflow_id:
            q = q.where(RecoveryEventRow.workflow_id == workflow_id)
        return [{"id": r.id, "workflow_id": r.workflow_id, "task_id": r.task_id, "agent_id": r.agent_id, "failure_type": r.failure_type,
                 "action": r.action, "attempt": r.attempt, "status": r.status, "detail": r.detail, "duration_ms": r.duration_ms,
                 "simulated": r.simulated, "created_at": r.created_at.isoformat() if r.created_at else None} for r in s.scalars(q)]


# ------------------------------------------------------------------ controller
class RecoveryController:
    def __init__(self, mode: str | None = None, *, max_attempts: int | None = None, base_delay: float | None = None,
                 max_delay: float | None = None, sleep: Callable[[float], None] = time.sleep, seed: int | None = None):
        self.mode = mode or settings.recovery_mode
        if self.mode not in ("none", "fixed_retry", "checkpoint", "full"):
            raise ValueError(f"unknown recovery mode {self.mode}")
        self.max_attempts = max_attempts or settings.retry_max_attempts
        self.base_delay = settings.retry_base_delay_s if base_delay is None else base_delay
        self.max_delay = settings.retry_max_delay_s if max_delay is None else max_delay
        self._sleep = sleep
        self._rng = random.Random(seed)
        self.stats = {"retries": 0, "reassignments": 0, "escalations": 0, "recovered": 0, "gave_up": 0, "duplicate_writes_prevented": 0}

    def backoff_delay(self, attempt: int, failure: Failure) -> float:
        if self.mode == "fixed_retry":
            return self.base_delay
        if failure.retry_after:
            return min(self.max_delay, failure.retry_after)
        return self._rng.uniform(0, min(self.max_delay, self.base_delay * (2 ** (attempt - 1))))  # full jitter

    def reassign_task(self, task: dict, failed_agents: set[str]) -> str | None:
        cat = task.get("category") or "development"
        elig = [a for a in settings.agent_pool if cat in a["capabilities"] and a["name"] not in failed_agents]
        if not elig:
            return None
        new = sorted(elig, key=lambda a: (a["capabilities"][0] != cat, -a["skill"]))[0]["name"]
        task["assigned_agent"] = new
        task.setdefault("reassignments", []).append(new)
        self.stats["reassignments"] += 1
        return new

    def escalate_failure(self, workflow_id: str | None, task: dict | None, failure: Failure, reason: str) -> None:
        self.stats["escalations"] += 1
        if task is not None:
            task["escalated"] = True
            task["escalation_reason"] = reason
        record_recovery_event(workflow_id=workflow_id, task_id=(task or {}).get("task_id"), failure_type=failure.type, action="escalate",
                              status="needs_human", agent_id=(task or {}).get("assigned_agent"),
                              detail={"reason": reason, "error": failure.message}, simulated=failure.simulated)

    def retry_task(self, fn: Callable[[], Any], *, workflow_id: str | None, task: dict | None = None, operation: str = "execute") -> Any:
        """Run fn with bounded, classified retries. Reassigns on agent crash (mode=full). Raises RecoveryExhausted."""
        attempt, failed_agents, t_fail, last = 0, set(), None, None
        while True:
            attempt += 1
            try:
                result = fn()
                if attempt > 1 and last is not None:
                    self.stats["recovered"] += 1
                    record_recovery_event(workflow_id=workflow_id, task_id=(task or {}).get("task_id"), failure_type=last.type,
                                          action="recovered", status="success", attempt=attempt, agent_id=(task or {}).get("assigned_agent"),
                                          duration_ms=round((time.perf_counter() - t_fail) * 1000, 1), simulated=last.simulated,
                                          detail={"operation": operation})
                return result
            except Exception as exc:
                last = detect_failure(exc)
                t_fail = t_fail or time.perf_counter()
                agent = (task or {}).get("assigned_agent")
                record_recovery_event(workflow_id=workflow_id, task_id=(task or {}).get("task_id"), failure_type=last.type,
                                      action="detected", status="failed", attempt=attempt, agent_id=agent,
                                      detail={"operation": operation, "error": last.message}, simulated=last.simulated)
                if self.mode == "none" or not last.retryable or attempt >= self.max_attempts:
                    self.stats["gave_up"] += 1
                    if self.mode in ("checkpoint", "full"):
                        self.escalate_failure(workflow_id, task, last, f"{operation} failed after {attempt} attempt(s)")
                    raise RecoveryExhausted(last, attempt) from exc
                if last.type == FailureType.AGENT_CRASH and task is not None:
                    if self.mode != "full":
                        self.stats["gave_up"] += 1
                        raise RecoveryExhausted(last, attempt) from exc  # same agent will keep failing
                    failed_agents.add(agent)
                    new = self.reassign_task(task, failed_agents)
                    record_recovery_event(workflow_id=workflow_id, task_id=task.get("task_id"), failure_type=last.type, action="reassign",
                                          status="success" if new else "failed", attempt=attempt, agent_id=new,
                                          detail={"from": agent, "to": new}, simulated=last.simulated)
                    if new is None:
                        self.escalate_failure(workflow_id, task, last, "no other capable agent available")
                        raise RecoveryExhausted(last, attempt) from exc
                delay = self.backoff_delay(attempt, last)
                self.stats["retries"] += 1
                record_recovery_event(workflow_id=workflow_id, task_id=(task or {}).get("task_id"), failure_type=last.type, action="retry",
                                      status="scheduled", attempt=attempt + 1, agent_id=(task or {}).get("assigned_agent"),
                                      detail={"delay_s": round(delay, 3), "operation": operation}, simulated=last.simulated)
                self._sleep(delay)

    # -------------------------------------------------------------- idempotent external writes
    def run_external_once(self, idempotency_key: str, operation: str, fn: Callable[[], dict], *, workflow_id: str | None = None) -> dict:
        with SessionLocal.begin() as s:
            row = s.get(ExternalOperationRow, idempotency_key)
            if row is not None:
                if row.status == "succeeded":
                    self.stats["duplicate_writes_prevented"] += 1
                    return {"duplicate": True, **(row.result or {})}
                if row.status in ("pending", "unknown"):
                    raise NeedsReconciliation(f"{operation} {idempotency_key} has status '{row.status}': reconcile before retrying")
                row.status, row.updated_at = "pending", now_utc()  # previous attempt definitively failed -> allowed
            else:
                s.add(ExternalOperationRow(idempotency_key=idempotency_key, workflow_id=workflow_id, operation=operation, status="pending"))
        try:
            result = fn()
        except Exception as e:
            definite = getattr(e, "definitely_not_applied", False)
            with SessionLocal.begin() as s:
                row = s.get(ExternalOperationRow, idempotency_key)
                row.status, row.result, row.updated_at = ("failed" if definite else "unknown"), {"error": redact(str(e))[:300]}, now_utc()
            record_recovery_event(workflow_id=workflow_id, task_id=None, failure_type=classify_failure(e), action="escalate",
                                  status="needs_reconciliation" if not definite else "failed",
                                  detail={"operation": operation, "idempotency_key": idempotency_key})
            raise
        with SessionLocal.begin() as s:
            row = s.get(ExternalOperationRow, idempotency_key)
            row.status, row.result, row.updated_at = "succeeded", result, now_utc()
        return {"duplicate": False, **result}


def controller(**kw) -> RecoveryController:
    return RecoveryController(**kw)
