"""Scheduling policies (round-robin, least-loaded, trained PPO) and evaluation harness."""
from __future__ import annotations

import json
from functools import lru_cache
from itertools import cycle

import numpy as np

from app.core.config import settings
from app.rl.environment import SchedulingEnv, SimTask, random_tasks

PPO_PATH = settings.models_dir / "rl_scheduler_ppo.zip"
METRICS_PATH = settings.reports_dir / "rl_metrics.json"


class RoundRobinPolicy:
    name = "round_robin"

    def __init__(self, env: SchedulingEnv):
        self.env, self._it = env, cycle(range(env.n))

    def act(self, obs) -> int:
        elig = set(self.env.eligible(self.env.current))
        for _ in range(self.env.n):
            i = next(self._it)
            if i in elig:
                return i
        return next(iter(elig))


class LeastLoadedPolicy:
    name = "least_loaded"

    def __init__(self, env: SchedulingEnv):
        self.env = env

    def act(self, obs) -> int:
        t = self.env.current
        return min(self.env.eligible(t), key=lambda i: max(self.env.free[i], self.env._ready_time(t)) + self.env.duration(t, i))


class PPOPolicy:
    name = "rl_ppo"

    def __init__(self, env: SchedulingEnv):
        self.env, self.model = env, load_ppo()
        if self.model is None:
            raise FileNotFoundError("PPO scheduler not trained; run `python -m app.rl.train`")

    def act(self, obs) -> int:
        a, _ = self.model.predict(obs, deterministic=True)
        return int(a)


POLICIES = {"round_robin": RoundRobinPolicy, "least_loaded": LeastLoadedPolicy, "rl_ppo": PPOPolicy}


@lru_cache(maxsize=1)
def load_ppo():
    if not PPO_PATH.exists():
        return None
    from stable_baselines3 import PPO

    return PPO.load(PPO_PATH, device="cpu")


def run_episode(policy_name: str, tasks: list[SimTask] | None = None, seed: int = 0, stochastic: bool = True) -> tuple[dict, dict]:
    env = SchedulingEnv(seed=seed, stochastic=stochastic)
    obs, _ = env.reset(seed=seed, options={"tasks": tasks} if tasks else None)
    policy = POLICIES[policy_name](env)
    done, steps = env.current is None, 0
    while not done and steps < 10 * env.total + 10:
        obs, _, done, _, _ = env.step(policy.act(obs))
        steps += 1
    return env.assignments, env.episode_metrics()


def evaluate(policies: list[str], episodes: int = 200, seed: int = 10_000) -> dict:
    out = {}
    for p in policies:
        rows = []
        for k in range(episodes):
            tasks = random_tasks(np.random.default_rng(seed + k))
            rows.append(run_episode(p, tasks, seed=seed + k)[1])
        out[p] = {m: round(float(np.mean([r[m] for r in rows])), 4) for m in rows[0]}
        out[p]["makespan_std"] = round(float(np.std([r["makespan"] for r in rows])), 4)
    return out


def best_policy() -> str:
    """'auto' => use PPO only if the saved evaluation shows it beats least-loaded on makespan."""
    if settings.scheduler != "auto":
        return settings.scheduler
    if PPO_PATH.exists() and METRICS_PATH.exists():
        ev = json.loads(METRICS_PATH.read_text(encoding="utf-8")).get("evaluation", {})
        if "rl_ppo" in ev and ev["rl_ppo"]["makespan"] < ev["least_loaded"]["makespan"] and ev["rl_ppo"]["invalid_actions"] < 0.5:
            return "rl_ppo"
    return "least_loaded"


def schedule(tasks: list[dict], policy: str | None = None) -> tuple[dict[str, str], str, dict]:
    """Assign real workflow tasks to agents. tasks: dicts with task_id, category, complexity, priority, dependency_ids."""
    policy = policy or best_policy()
    sim = [SimTask(t["task_id"], t.get("category") or "development", int(t.get("complexity", 3)), t.get("priority") or "medium",
                   list(t.get("dependency_ids", []))) for t in tasks]
    if not sim:
        return {}, policy, {}
    assignments, metrics = run_episode(policy, sim, seed=0, stochastic=False)
    return assignments, policy, metrics
