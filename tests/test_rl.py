import numpy as np
from gymnasium.utils.env_checker import check_env

from app.core.config import settings
from app.rl import scheduler
from app.rl.environment import SchedulingEnv, SimTask, random_tasks


def test_env_is_gymnasium_compliant():
    check_env(SchedulingEnv(seed=0), skip_render_check=True)


def test_invalid_action_penalised_and_not_assigned():
    env = SchedulingEnv(seed=0, tasks=[SimTask("x", "documentation", 2, "low")])
    env.reset(seed=0)
    bad = next(i for i, a in enumerate(env.agents) if "documentation" not in a["capabilities"])
    _, r, done, _, info = env.step(bad)
    assert r < 0 and info.get("invalid") and not done and "x" not in env.assignments


def test_dependencies_respected():
    tasks = [SimTask("a", "development", 3, "high"), SimTask("b", "testing", 2, "critical", ["a"])]
    assignments, _ = scheduler.run_episode("least_loaded", tasks, stochastic=False)
    env = SchedulingEnv(tasks=tasks, stochastic=False)
    env.reset(seed=0)
    for _ in range(2):
        env.step(scheduler.LeastLoadedPolicy(env).act(None))
    assert env.finish["b"] > env.finish["a"]
    assert set(assignments) == {"a", "b"}


def test_baselines_assign_only_eligible_agents():
    caps = {a["name"]: a["capabilities"] for a in settings.agent_pool}
    tasks = random_tasks(np.random.default_rng(3), 15)
    cat = {t.task_id: t.category for t in tasks}
    for p in ("round_robin", "least_loaded"):
        assignments, m = scheduler.run_episode(p, tasks, seed=3)
        assert len(assignments) == 15 and m["invalid_actions"] == 0
        assert all(cat[t] in caps[a] for t, a in assignments.items())


def test_schedule_real_tasks():
    tasks = [{"task_id": "T1", "category": "development", "complexity": 3, "priority": "high", "dependency_ids": []},
             {"task_id": "T2", "category": "documentation", "complexity": 2, "priority": "low", "dependency_ids": ["T1"]}]
    assignments, used, metrics = scheduler.schedule(tasks)
    assert used in ("least_loaded", "rl_ppo") and set(assignments) == {"T1", "T2"}
    assert metrics["success_rate"] == 1.0


def test_trained_policy_loads_if_present():
    if not scheduler.PPO_PATH.exists():
        return
    _, m = scheduler.run_episode("rl_ppo", random_tasks(np.random.default_rng(5)), seed=5)
    assert m["makespan"] > 0
