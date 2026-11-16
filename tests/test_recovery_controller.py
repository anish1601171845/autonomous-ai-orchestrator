import pytest

from app.database import repository
from app.recovery import checkpoint_manager as cm
from app.recovery import fault_injection as fi
from app.recovery.recovery_controller import (
    FailureType,
    NeedsReconciliation,
    RecoveryController,
    RecoveryExhausted,
    classify_failure,
    list_recovery_events,
)


@pytest.fixture
def wid():
    w = "wf_rec_" + __import__("uuid").uuid4().hex[:6]
    repository.create_workflow(w, "recovery test")
    return w


def ctl(mode="full"):
    return RecoveryController(mode, sleep=lambda s: None, seed=0)


def _task():
    return {"task_id": "T1", "category": "development", "assigned_agent": "coder-1"}


def _op(task):
    def fn():
        fi.INJECTOR.check("execute", agent=task["assigned_agent"], task_id=task["task_id"])
        return "done"
    return fn


def test_classification():
    assert classify_failure(fi.SimulatedRateLimit()) == FailureType.RATE_LIMIT
    assert classify_failure(fi.SimulatedLLMTimeout("x")) == FailureType.LLM_TIMEOUT
    assert classify_failure(ConnectionError("reset")) == FailureType.NETWORK
    assert classify_failure(ValueError("bad input")) == FailureType.NON_RETRYABLE


def test_retry_success(wid):
    t = _task()
    with fi.inject("api_timeout", 2):
        assert ctl().retry_task(_op(t), workflow_id=wid, task=t) == "done"
    actions = [e["action"] for e in list_recovery_events(workflow_id=wid)]
    assert actions.count("retry") == 2 and "recovered" in actions


def test_retry_exhaustion_and_escalation(wid):
    t = _task()
    c = ctl()
    with fi.inject("repeated_failure"), pytest.raises(RecoveryExhausted) as e:
        c.retry_task(_op(t), workflow_id=wid, task=t)
    assert e.value.attempts == c.max_attempts and t["escalated"]
    assert any(ev["action"] == "escalate" and ev["status"] == "needs_human" for ev in list_recovery_events(workflow_id=wid))


def test_non_retryable_is_not_retried(wid):
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("deterministic bug")

    with pytest.raises(RecoveryExhausted):
        ctl().retry_task(bad, workflow_id=wid, task=_task())
    assert calls["n"] == 1


def test_no_recovery_mode_fails_fast(wid):
    t = _task()
    with fi.inject("api_timeout", 1), pytest.raises(RecoveryExhausted) as e:
        ctl("none").retry_task(_op(t), workflow_id=wid, task=t)
    assert e.value.attempts == 1


def test_task_reassignment_on_agent_crash(wid):
    t = _task()
    with fi.inject("agent_crash", agent="coder-1"):
        assert ctl().retry_task(_op(t), workflow_id=wid, task=t) == "done"
    from app.core.config import settings

    new = next(a for a in settings.agent_pool if a["name"] == t["assigned_agent"])
    assert t["assigned_agent"] != "coder-1" and "development" in new["capabilities"]
    assert any(e["action"] == "reassign" for e in list_recovery_events(workflow_id=wid))


def test_checkpoint_restoration_requeues_partial_task(wid):
    state = {"workflow_id": wid, "requirement": "x", "status": "running", "queue": ["T3"], "current_task_id": "T2",
             "execution_order": ["T1", "T2", "T3"],
             "tasks": [{"task_id": "T1", "status": "completed"}, {"task_id": "T2", "status": "in_progress"}, {"task_id": "T3", "status": "assigned"}]}
    cm.save_checkpoint(state, "execute")
    restored = cm.restore_checkpoint(wid)
    assert restored["queue"] == ["T2", "T3"] and restored["current_task_id"] is None
    t2 = next(t for t in restored["tasks"] if t["task_id"] == "T2")
    assert t2["status"] == "pending" and t2["interrupted"]
    assert next(t for t in restored["tasks"] if t["task_id"] == "T1")["status"] == "completed"


def test_duplicate_external_action_prevented(wid):
    calls = {"n": 0}

    def create_pr():
        calls["n"] += 1
        return {"pr": 1}

    c = ctl()
    key = f"{wid}:github_publish"
    assert c.run_external_once(key, "github_publish", create_pr, workflow_id=wid)["duplicate"] is False
    assert c.run_external_once(key, "github_publish", create_pr, workflow_id=wid)["duplicate"] is True
    assert calls["n"] == 1


def test_unknown_outcome_blocks_automatic_retry(wid):
    def flaky_push():
        raise ConnectionError("connection reset after request was sent")

    c = ctl()
    key = f"{wid}:push"
    with pytest.raises(ConnectionError):
        c.run_external_once(key, "git_push", flaky_push, workflow_id=wid)
    with pytest.raises(NeedsReconciliation):
        c.run_external_once(key, "git_push", flaky_push, workflow_id=wid)


def test_backoff_has_jitter_and_respects_cap():
    c = RecoveryController("full", base_delay=1.0, max_delay=4.0, sleep=lambda s: None, seed=3)
    from app.recovery.recovery_controller import Failure

    f = Failure(FailureType.NETWORK, True, "x")
    delays = [c.backoff_delay(a, f) for a in range(1, 8)]
    assert all(0 <= d <= 4.0 for d in delays) and len(set(delays)) > 1
    assert c.backoff_delay(1, Failure(FailureType.RATE_LIMIT, True, "x", retry_after=2.0)) == 2.0
    assert RecoveryController("fixed_retry", base_delay=0.3, sleep=lambda s: None).backoff_delay(5, f) == 0.3
