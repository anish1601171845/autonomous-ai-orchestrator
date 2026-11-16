"""Gymnasium environment for workload-balanced task scheduling (list scheduling formulation).

Each step the environment presents the highest-priority *ready* task (all prerequisites already
scheduled); the action chooses which agent gets it.

State  : per-agent [relative free time, capability match, skill, load share] + task complexity,
         priority one-hot, category one-hot, fraction of tasks remaining.
Action : index of the agent to receive the task.
Reward : +1 success / -0.5 failure, minus priority-weighted waiting+processing time, minus load
         imbalance; -1 for assigning to an agent lacking the capability (task is not assigned).
Workloads are SIMULATED; results measure the simulator, not production throughput.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.core.config import CATEGORIES, PRIORITIES, settings

PRIO_W = {"low": 0.5, "medium": 1.0, "high": 1.6, "critical": 2.5}
STAGE = {"analysis": 0, "development": 1, "testing": 2, "documentation": 3, "deployment": 3}


@dataclass
class SimTask:
    task_id: str
    category: str
    complexity: int
    priority: str
    deps: list[str] = field(default_factory=list)


def random_tasks(rng: np.random.Generator, n: int | None = None) -> list[SimTask]:
    n = n or int(rng.integers(8, 21))
    tasks: list[SimTask] = []
    for i in range(n):
        cat = str(rng.choice(CATEGORIES, p=[0.38, 0.27, 0.12, 0.10, 0.13]))
        prev = [t for t in tasks if STAGE[t.category] < STAGE[cat]]
        k = min(len(prev), int(rng.integers(0, 3)))
        deps = [t.task_id for t in rng.choice(prev, size=k, replace=False)] if k else []
        tasks.append(SimTask(f"S{i}", cat, int(rng.integers(1, 6)), str(rng.choice(PRIORITIES, p=[0.3, 0.35, 0.25, 0.1])), deps))
    return tasks


class SchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, agents: list[dict] | None = None, tasks: list[SimTask] | None = None, seed: int | None = None,
                 max_invalid: int = 3, stochastic: bool = True):
        super().__init__()
        self.agents = agents or settings.agent_pool
        self.n = len(self.agents)
        self.fixed_tasks = tasks
        self.max_invalid = max_invalid
        self.stochastic = stochastic
        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.Box(-1.0, 5.0, shape=(self.n * 4 + 1 + len(PRIORITIES) + len(CATEGORIES) + 1,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    # -------------------------------------------------------------- helpers
    def eligible(self, task: SimTask) -> list[int]:
        return [i for i, a in enumerate(self.agents) if task.category in a["capabilities"]]

    def duration(self, task: SimTask, agent_idx: int) -> float:
        a = self.agents[agent_idx]
        specialist = a["capabilities"][0] == task.category
        return task.complexity * 2.0 / a["skill"] * (0.85 if specialist else 1.0)

    def fail_prob(self, task: SimTask, agent_idx: int) -> float:
        a = self.agents[agent_idx]
        specialist = a["capabilities"][0] == task.category
        return float(np.clip(0.02 + 0.035 * (task.complexity - 1) / a["skill"] - (0.04 if specialist else 0), 0, 0.5))

    def _next_task(self) -> SimTask | None:
        ready = [t for t in self.pending if all(d in self.finish for d in t.deps)]
        if not ready:
            return None
        order = {t.task_id: k for k, t in enumerate(self.pending)}
        return max(ready, key=lambda t: (PRIO_W[t.priority], -order[t.task_id]))

    def _ready_time(self, t: SimTask) -> float:
        return max([self.finish[d] for d in t.deps], default=0.0)

    def _obs(self) -> np.ndarray:
        t = self.current
        busy = np.array(self.busy)
        if t is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        rt = self._ready_time(t)
        rel = np.clip((np.array(self.free) - rt) / 20.0, -1, 5)
        cap = np.array([1.0 if t.category in a["capabilities"] else 0.0 for a in self.agents])
        skill = np.array([a["skill"] for a in self.agents])
        share = busy / busy.sum() if busy.sum() > 0 else np.zeros(self.n)
        pr = np.eye(len(PRIORITIES))[PRIORITIES.index(t.priority)]
        ct = np.eye(len(CATEGORIES))[CATEGORIES.index(t.category)]
        per_agent = np.stack([rel, cap, skill, share], 1).ravel()
        return np.concatenate([per_agent, [t.complexity / 5], pr, ct, [len(self.pending) / max(self.total, 1)]]).astype(np.float32)

    # -------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        tasks = (options or {}).get("tasks") or self.fixed_tasks or random_tasks(self._rng)
        self.pending = list(tasks)
        self.total = len(tasks)
        self.free = [0.0] * self.n
        self.busy = [0.0] * self.n
        self.finish: dict[str, float] = {}
        self.assignments: dict[str, str] = {}
        self.successes = self.failures = self.invalid = 0
        self.weighted_completion = 0.0
        self._invalid_streak = 0
        self.current = self._next_task()
        return self._obs(), {}

    def _assign(self, t: SimTask, i: int) -> float:
        start = max(self.free[i], self._ready_time(t))
        dur = self.duration(t, i)
        fail = self.stochastic and self._rng.random() < self.fail_prob(t, i)
        if fail:
            dur *= 1.6  # rework
        end = start + dur
        self.free[i] = end
        self.busy[i] += dur
        self.finish[t.task_id] = end
        self.assignments[t.task_id] = self.agents[i]["name"]
        self.pending.remove(t)
        self.weighted_completion += PRIO_W[t.priority] * end
        if fail:
            self.failures += 1
        else:
            self.successes += 1
        busy = np.array(self.busy)
        imbalance = busy.std() / (busy.mean() + 1e-9)
        return (1.0 if not fail else -0.5) - 0.02 * PRIO_W[t.priority] * (end - self._ready_time(t)) - 0.1 * imbalance

    def step(self, action: int):
        t = self.current
        i = int(action)
        if t.category not in self.agents[i]["capabilities"]:
            self.invalid += 1
            self._invalid_streak += 1
            reward = -1.0
            if self._invalid_streak < self.max_invalid:
                return self._obs(), reward, False, False, {"invalid": True}
            i = min(self.eligible(t), key=lambda k: self.free[k])  # forced fallback
            reward += self._assign(t, i) - 0.5
        else:
            reward = self._assign(t, i)
        self._invalid_streak = 0
        self.current = self._next_task()
        done = self.current is None
        if done:
            reward -= 0.02 * self.makespan
        return self._obs(), float(reward), done, False, {}

    @property
    def makespan(self) -> float:
        return max(self.finish.values(), default=0.0)

    def episode_metrics(self) -> dict:
        busy = np.array(self.busy)
        return {"makespan": self.makespan, "load_imbalance_cv": float(busy.std() / (busy.mean() + 1e-9)),
                "success_rate": self.successes / max(self.total, 1), "invalid_actions": self.invalid,
                "priority_weighted_completion": self.weighted_completion / max(self.total, 1)}
