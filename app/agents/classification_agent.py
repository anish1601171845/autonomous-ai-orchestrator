"""Agent 3 — Task Classification (trained ML model)."""
from __future__ import annotations

from app.ml import classification

NAME = "classification_agent"


def run(tasks: list[dict]) -> list[dict]:
    preds = classification.predict([t["description"] for t in tasks])
    for t, (label, conf) in zip(tasks, preds):
        t["category"], t["category_confidence"] = label, conf
    return tasks
