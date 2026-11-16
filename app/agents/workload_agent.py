"""Agent 7 — Workload Allocation: capability-aware assignment via RL (PPO) or rule-based schedulers."""
from __future__ import annotations

from app.core.config import settings
from app.rl import scheduler

NAME = "workload_agent"


def agents() -> list[dict]:
    return settings.agent_pool


def run(tasks: list[dict], policy: str | None = None) -> dict:
    assignments, used, sim_metrics = scheduler.schedule(tasks, policy)
    for t in tasks:
        t["assigned_agent"] = assignments.get(t["task_id"])
        t["status"] = "assigned"
    return {"assignments": assignments, "scheduler": used, "simulated_plan_metrics": sim_metrics}
