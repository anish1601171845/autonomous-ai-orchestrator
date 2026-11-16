"""Agent 9 — Ticket Intake & Triage.

Classifies an incoming enterprise support ticket (any of en/de/es/fr/pt) by type, priority and queue
with models trained on the real multilingual ticket dataset, then optionally turns it into an
engineering workflow whose tasks depend on the ticket type.
"""
from __future__ import annotations

import re

from app.core.state import Task
from app.ml import ticket_models

NAME = "intake_agent"
URGENCY = {"high": 5, "medium": 3, "low": 1}
IMPACT = {"Incident": 4, "Problem": 4, "Change": 3, "Request": 3}
# queue -> which internal agent role is the natural owner (used for display / routing hints)
QUEUE_OWNER = {
    "Technical Support": "coding", "IT Support": "devops", "Product Support": "coding",
    "Service Outages and Maintenance": "devops", "Billing and Payments": "coding", "Customer Service": "documentation",
    "Returns and Exchanges": "documentation", "Sales and Pre-Sales": "documentation", "General Inquiry": "documentation",
    "Human Resources": "documentation",
}
_STOP = {"the", "and", "for", "with", "issue", "issues", "problem", "request", "urgent", "help", "need", "regarding",
         "about", "assistance", "support", "please", "inquiry", "our", "your", "from", "has", "are", "not"}


def triage(ticket: dict) -> dict:
    pred = ticket_models.predict([ticket])[0]
    pred["owner_role"] = QUEUE_OWNER.get(pred["queue"], "coding")
    return pred


def _feature(ticket: dict, triage_result: dict) -> str:
    words = [w for w in re.findall(r"[a-z]{3,}", (ticket.get("subject") or "").lower()) if w not in _STOP]
    return " ".join(words[:3]) or triage_result["queue"].lower()


def scope_ticket(ticket: dict, triage_result: dict | None = None) -> tuple[dict, list[Task]]:
    tri = triage_result or triage(ticket)
    kind, prio = tri["type"], tri["priority"]
    feature = _feature(ticket, tri)
    subject = (ticket.get("subject") or feature).strip()[:100]
    urgency, impact = URGENCY.get(prio, 3), IMPACT.get(kind, 3) + (1 if prio == "high" else 0)
    tasks: list[Task] = []

    def add(title: str, desc: str, feat: str, cx: int, criteria: list[str]) -> None:
        tasks.append(Task(task_id=f"T{len(tasks) + 1:02d}", title=title[:120], description=desc, feature=feat, complexity=cx,
                          urgency=urgency, business_impact=min(5, impact), estimated_duration_h=round(cx * 2.5, 1),
                          acceptance_criteria=criteria))

    first = {"Incident": f"Investigate current behaviour and analyse root cause of {feature}",
             "Problem": f"Investigate recurring failures and analyse root cause of {feature}",
             "Change": f"Assess impact and define acceptance criteria for the {feature} change",
             "Request": f"Gather and analyse requirements for {feature}"}[kind]
    add("Triage analysis", first, feature, 2, [f"Ticket: {subject}", f"Predicted {kind} / {prio} / {tri['queue']}"])
    add(f"Implement {feature}", f"Implement {feature} backend service with input validation and error handling", feature, 3,
        ["Fix or feature implemented as an importable module", "Inputs validated with clear errors"])
    add(f"Test {feature}", f"Write unit tests for {feature}", feature, 2, ["Regression tests reproduce the reported scenario"])
    if kind in ("Incident", "Change") or prio == "high":
        add("Release", f"Deploy the {feature} fix to production with a rollback plan", "project", 3,
            ["CI pipeline and container definition exist", "Rollback plan documented"])
    add("Resolution notes", f"Document the {feature} resolution and update the support runbook", feature, 1,
        ["Customer-facing resolution summary written", "Runbook updated"])
    requirements = {"subject": subject, "functional": [feature], "non_functional": [], "backend": "ticket_triage",
                    "flags": {"ticket_mode": True}, "ticket": {k: ticket.get(k) for k in ("subject", "language", "business_type")},
                    "triage": tri}
    return requirements, tasks
