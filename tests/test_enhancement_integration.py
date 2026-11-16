"""End-to-end checks that the four enhancements are wired into the real LangGraph workflow."""
from collections import Counter

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core import orchestrator
from app.memory import rag_memory
from app.ml import failure_predictor as fp
from app.recovery import fault_injection as fi
from app.recovery.recovery_controller import list_recovery_events
from tests.conftest import EXAMPLE, needs_models

client = TestClient(app)
REQ = "Build a notes app with login, note search and automated testing."
risk_ready = pytest.mark.skipif(not fp.MODEL_PATH.exists(), reason="run scripts/evaluate_enhancements.py --only risk")


@pytest.fixture(autouse=True)
def fresh_memory():
    rag_memory.reset_service()
    yield
    rag_memory.reset_service()


@needs_models
def test_existing_workflow_still_functions_with_all_enhancements():
    s = orchestrator.run_workflow(EXAMPLE)
    assert s["status"] == "awaiting_approval" and all(t["status"] == "completed" for t in s["tasks"])
    assert s["summary"]["final_suite_passed"] is True
    enh = s["summary"]["enhancements"]
    assert enh["debates"] >= 1 and enh["recovery_mode"] == "full"


@needs_models
def test_memory_saved_after_verification_and_fed_to_coding_agent():
    orchestrator.run_workflow(REQ)
    s2 = orchestrator.run_workflow(REQ)
    dev = [t for t in s2["tasks"] if t["category"] == "development"]
    assert any(t.get("memory_hits") for t in dev)
    assert any(t.get("memory_context_supplied") for t in dev)
    kinds = [e["kind"] for e in client.get(f"/api/workflows/{s2['workflow_id']}").json()["events"]]
    assert "memory_retrieved" in kinds
    assert all(m["verified_outcome"] in rag_memory.VERIFIED_OUTCOMES for m in rag_memory.service().list_memories())


@needs_models
@risk_ready
def test_risk_prediction_is_recorded():
    s = orchestrator.run_workflow(REQ)
    assert all(t.get("risk_category") in ("Low", "Medium", "High") for t in s["tasks"])
    preds = client.get("/api/risk/predictions", params={"workflow_id": s["workflow_id"]}).json()
    assert len(preds) == len(s["tasks"]) and preds[0]["explanation"]
    rows = client.get("/api/tasks", params={"workflow_id": s["workflow_id"]}).json()
    assert all(r["risk_category"] for r in rows)


@needs_models
def test_debate_produces_review_history_and_qa_result():
    s = orchestrator.run_workflow(REQ)
    debates = client.get("/api/debates", params={"workflow_id": s["workflow_id"]}).json()
    assert debates and all(d["qa_passed"] is True for d in debates)
    d = client.get(f"/api/debates/{debates[0]['debate_id']}").json()
    assert {"developer", "critic", "judge", "qa"} <= {m["role"] for m in d["messages"]}


@needs_models
def test_recovery_updates_workflow_status_after_worker_crash():
    with fi.inject("worker_crash", node="qa") as spec:
        s = orchestrator.run_workflow(REQ)
    assert spec.fired == 1 and s["status"] == "awaiting_approval"
    assert s["summary"]["enhancements"]["restored_from_checkpoint"]
    ev = Counter(e["action"] for e in list_recovery_events(workflow_id=s["workflow_id"]))
    assert ev["checkpoint_restore"] == 1 and all(t["status"] == "completed" for t in s["tasks"])


@needs_models
def test_repeated_failure_escalates_and_blocks_dependants():
    with fi.inject("repeated_failure", task_id="T04"):
        s = orchestrator.run_workflow(REQ)
    t4 = next(t for t in s["tasks"] if t["task_id"] == "T04")
    assert t4["status"] == "failed" and t4["escalated"]
    assert any(t["status"] == "blocked" for t in s["tasks"])
    assert s["summary"]["enhancements"]["escalated_tasks"] == ["T04"]


@needs_models
def test_dashboard_endpoints_read_persisted_data():
    r = client.post("/api/recovery/demo", json={"scenario": "api_timeout"})
    assert r.status_code == 200 and r.json()["simulated"] and r.json()["faults_fired"] == 2
    assert any(e["action"] == "recovered" for e in r.json()["recovery_events"])
    assert client.get("/api/memory").status_code == 200
    assert client.get("/api/recovery/events").json()
    assert client.get("/api/enhancements/status").json()["recovery"]["mode"] == "full"
    assert client.post("/api/recovery/demo", json={"scenario": "nope"}).status_code == 422
