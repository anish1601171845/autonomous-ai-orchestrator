import pytest
from fastapi.testclient import TestClient

from app.agents import coding_agent
from app.api.main import app
from app.core import orchestrator
from app.database import repository as repo
from tests.conftest import EXAMPLE, needs_models

client = TestClient(app)


@needs_models
def test_end_to_end_workflow_persists_and_awaits_approval():
    s = orchestrator.run_workflow(EXAMPLE)
    assert s["status"] == "awaiting_approval"
    assert all(t["status"] == "completed" for t in s["tasks"])
    assert s["summary"]["final_suite_passed"] is True
    wf = repo.get_workflow(s["workflow_id"])
    assert wf["status"] == "awaiting_approval" and len(wf["tasks"]) == len(s["tasks"])
    kinds = [e["kind"] for e in wf["events"]]
    assert kinds[0] == "tasks_scoped" and "qa_passed" in kinds and kinds[-1] == "approval_requested"
    res = orchestrator.approve(s["workflow_id"], approved=True)
    assert res["approved"] and repo.get_workflow(s["workflow_id"], False)["status"] == "completed"
    with pytest.raises(ValueError):  # approval only allowed once, in the right state
        orchestrator.approve(s["workflow_id"], approved=True)


@needs_models
def test_agent_failure_recovery_and_bounded_retries(monkeypatch):
    real = coding_agent.generate_code
    calls = {}

    def flaky(wid, task, feedback=None, memory_context=None):
        calls[task["task_id"]] = calls.get(task["task_id"], 0) + 1
        if "login" in task["description"].lower():
            out = real(wid, task, feedback, memory_context)  # then break the module so QA fails every time
            (coding_agent.settings.workspace_dir / out["files"][0]).write_text("def broken(:\n")
            return out
        if "dashboard" in task["description"].lower():
            raise TimeoutError("simulated LLM timeout")
        return real(wid, task, feedback, memory_context)

    monkeypatch.setattr(coding_agent, "generate_code", flaky)
    s = orchestrator.run_workflow(EXAMPLE)
    by_title = {t["title"]: t for t in s["tasks"]}
    login = by_title["Implement login"]
    # QA retry budget + one extra attempt after safe reassignment to another coding agent (RECOVERY_MODE=full)
    assert login["status"] == "failed" and login["attempts"] == orchestrator.settings.max_qa_retries + 2
    assert login["qa_reassigned"] and login["escalated"]
    assert by_title["Implement student dashboard"]["status"] == "failed"
    assert "TimeoutError" in by_title["Implement student dashboard"]["error"]
    assert by_title["Test login"]["status"] == "blocked"
    assert by_title["Implement registration"]["status"] == "completed"  # unaffected branch continues
    assert s["status"] == "awaiting_approval"


def test_api_health_and_validation():
    r = client.get("/api/health")
    assert r.status_code == 200 and "models" in r.json()
    r = client.post("/api/workflows", json={"requirement": "short"})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert client.get("/api/workflows/nope").status_code == 404
    assert client.post("/api/predict/priority", json={"tasks": [{"description": "x", "urgency": 9}]}).status_code == 422


@needs_models
def test_api_prediction_endpoints():
    r = client.post("/api/predict/category", json={"descriptions": ["Write unit tests for billing"]})
    assert r.status_code == 200 and r.json()[0]["label"] == "testing"
    r = client.post("/api/predict/priority", json={"tasks": [{"description": "urgent: Fix login outage", "urgency": 5, "business_impact": 5}]})
    assert r.status_code == 200 and r.json()[0]["label"] in ("high", "critical")
    body = {"tasks": [{"task_id": "A", "description": "Implement search engine backend service", "feature": "search"},
                      {"task_id": "B", "description": "Write unit tests for search engine", "feature": "search"}]}
    r = client.post("/api/dependencies/predict", json=body)
    assert r.status_code == 200 and r.json()["is_dag"] and r.json()["execution_order"] == ["A", "B"]
    dup = {"tasks": [body["tasks"][0], body["tasks"][0]]}
    assert client.post("/api/dependencies/predict", json=dup).status_code == 422


@needs_models
def test_api_workflow_lifecycle():
    r = client.post("/api/workflows", json={"requirement": "Create a leave management tool with leave request and approval dashboard", "run_async": False})
    assert r.status_code == 202
    wid = r.json()["workflow_id"]
    wf = client.get(f"/api/workflows/{wid}").json()
    assert wf["status"] == "awaiting_approval" and wf["tasks"]
    assert any(w["workflow_id"] == wid for w in client.get("/api/workflows").json())
    tid = wf["tasks"][0]["task_id"]
    assert client.get(f"/api/tasks/{tid}", params={"workflow_id": wid}).json()["workflow_id"] == wid
    assert client.get("/api/tasks", params={"workflow_id": wid}).status_code == 200
    agents = client.get("/api/agents").json()
    assert {a["name"] for a in agents} >= {"coder-1", "qa-1"}
    m = client.get("/api/metrics").json()
    assert m["runtime"]["total_tasks"] >= len(wf["tasks"])
    r = client.post(f"/api/workflows/{wid}/approve", json={"approved": False})
    assert r.status_code == 200 and client.get(f"/api/workflows/{wid}").json()["status"] == "rejected"
    assert client.post(f"/api/workflows/{wid}/approve", json={"approved": True}).status_code == 409
