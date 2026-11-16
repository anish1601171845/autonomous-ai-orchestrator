"""Agent 4 — Priority Prediction (urgency, complexity, business impact, dependency fan-out)."""
from __future__ import annotations

from app.ml import priority_prediction

NAME = "priority_agent"


def run(tasks: list[dict]) -> list[dict]:
    dependents = {t["task_id"]: 0 for t in tasks}
    for t in tasks:
        for d in t.get("dependency_ids", []):
            if d in dependents:
                dependents[d] += 1
    records = [{"description": t["description"], "category": t.get("category"), "complexity": t["complexity"],
                "urgency": t["urgency"], "business_impact": t["business_impact"], "num_dependents": dependents[t["task_id"]]}
               for t in tasks]
    for t, (label, conf) in zip(tasks, priority_prediction.predict(records)):
        t["priority"], t["priority_confidence"] = label, conf
    return tasks
