"""Tests for the models and agents built on the REAL datasets (PROMISE NFR + multilingual tickets)."""
import pytest
from fastapi.testclient import TestClient

from app.agents import intake_agent, scoping_agent
from app.api.main import app
from app.core import orchestrator
from app.ml import real_data, requirements_classifier, ticket_models

client = TestClient(app)
req_ready = pytest.mark.skipif(not requirements_classifier.MODEL_PATH.exists(), reason="run scripts/train_real.py")
tix_ready = pytest.mark.skipif(not ticket_models.MODEL_PATH.exists(), reason="run scripts/train_real.py")

DOC = """F: Teachers shall be able to create a new homework assignment for a class.
F: Students shall be able to upload their homework submission before the deadline.
F: Teachers shall be able to grade a submission and add written feedback.
SE: Only authenticated users with the teacher role shall be able to change grades.
SE: All passwords shall be stored encrypted and user sessions shall expire after 15 minutes of inactivity.
PE: The system shall return search results within 2 seconds for 95% of queries.
L: The system must comply with the data protection regulations for student records."""


def test_promise_loader_cleans_and_labels():
    tr = real_data.load_promise(real_data.PROMISE_TRAIN)
    assert len(tr) > 500 and set(tr["label"]) <= set(real_data.REQUIREMENT_TYPES)
    assert not tr["text"].str.contains("�").any()
    assert tr["is_functional"].sum() == (tr["label"] == "F").sum()


def test_ticket_loader():
    df = real_data.load_tickets()
    assert len(df) > 3900 and set(df["language"]) == {"en", "de", "es", "fr", "pt"}
    assert not df["text"].str.contains("<name>").any()


def test_document_detection_and_split():
    stmts = scoping_agent.split_requirements(DOC)
    assert len(stmts) == 7 and not stmts[0].startswith("F:")
    assert scoping_agent.is_requirements_document(DOC)
    assert not scoping_agent.is_requirements_document("Build a todo app with login and reminders.")


@req_ready
def test_requirements_classifier_predictions():
    preds = requirements_classifier.predict(scoping_agent.split_requirements(DOC))
    labels = [p["label"] for p in preds]
    assert labels[:3].count("F") >= 2 and labels[3:5] == ["SE", "SE"] and labels[5] == "PE" and labels[6] == "L"
    assert all(0 < p["confidence"] <= 1 and p["is_functional"] == (p["label"] == "F") for p in preds)


@req_ready
def test_document_mode_scoping_creates_features_and_nfr_tasks():
    reqs, tasks = scoping_agent.run(DOC)
    assert reqs["flags"]["document_mode"] and reqs["statements"] == 7
    titles = [t.title for t in tasks]
    assert any(t.startswith("Implement ") for t in titles) and any(t.startswith("Test ") for t in titles)
    assert "Security controls" in titles
    sec = next(t for t in tasks if t.title == "Security controls")
    assert any("teacher role" in c or "encrypted" in c for c in sec.acceptance_criteria)  # NFR sentences become acceptance criteria
    assert len({t.feature for t in tasks if t.feature != "project"}) == len([t for t in titles if t.startswith("Implement ")]) + (1 if "Security controls" in titles else 0)


@tix_ready
def test_ticket_triage_multilingual():
    out = ticket_models.predict([
        {"subject": "Server down", "body": "Our production server crashed, all customers affected, fix immediately.", "language": "en"},
        {"subject": "Rechnung falsch", "body": "Meine letzte Rechnung enthält eine doppelte Abbuchung. Bitte erstatten.", "language": "de"},
    ])
    assert {"type", "priority", "queue"} <= out[0].keys()
    assert out[0]["type"] in {"Incident", "Problem", "Request", "Change"}
    assert out[0]["priority"] in {"high", "medium", "low"}


@tix_ready
def test_ticket_workflow_end_to_end():
    ticket = {"subject": "Checkout payment failure", "body": "Every checkout fails at the payment step since this morning. Urgent.",
              "language": "en", "business_type": "Tech Online Store"}
    tri = intake_agent.triage(ticket)
    s = orchestrator.run_workflow("[ticket] Checkout payment failure", ticket={**ticket, "triage": tri})
    assert s["requirements"]["flags"]["ticket_mode"] and s["requirements"]["triage"]["queue"] == tri["queue"]
    assert s["status"] == "awaiting_approval"
    assert all(t["status"] == "completed" for t in s["tasks"])


@req_ready
@tix_ready
def test_real_data_api():
    r = client.post("/api/requirements/classify", json={"text": DOC})
    assert r.status_code == 200 and r.json()["count"] == 7
    assert client.post("/api/requirements/classify", json={}).status_code == 422
    r = client.post("/api/tickets/triage", json={"subject": "Login broken", "body": "I cannot log in to my account since the update."})
    assert r.status_code == 200 and r.json()["workflow_id"] is None and r.json()["triage"]["queue"]
    assert client.post("/api/tickets/triage", json={"body": "short"}).status_code == 422
    d = client.get("/api/datasets").json()
    assert d["available"] == {"promise": True, "tickets": True} and "ticket_metrics" in d
