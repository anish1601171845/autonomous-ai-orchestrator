"""Train the PPO scheduler on simulated workloads and compare against rule-based baselines.

Usage: python -m app.rl.train [--timesteps 80000]
"""
from __future__ import annotations

import argparse
import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from app.rl.environment import SchedulingEnv
from app.rl.scheduler import METRICS_PATH, PPO_PATH, evaluate, load_ppo


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.curve: list[dict] = []
        self._ep: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep.append(info["episode"]["r"])
        if self._ep and self.num_timesteps % 2048 < self.training_env.num_envs:
            self.curve.append({"timesteps": int(self.num_timesteps), "mean_episode_reward": round(sum(self._ep[-50:]) / len(self._ep[-50:]), 3)})
        return True


def train(timesteps: int = 80_000, seed: int = 42, eval_episodes: int = 200) -> dict:
    env = make_vec_env(lambda: SchedulingEnv(), n_envs=4, seed=seed)
    model = PPO("MlpPolicy", env, seed=seed, n_steps=512, batch_size=256, learning_rate=3e-4, gamma=0.99,
                ent_coef=0.01, verbose=0, device="cpu", policy_kwargs={"net_arch": [128, 128]})
    cb = RewardLogger()
    model.learn(total_timesteps=timesteps, callback=cb)
    model.save(PPO_PATH)
    load_ppo.cache_clear()
    ev = evaluate(["round_robin", "least_loaded", "rl_ppo"], episodes=eval_episodes)
    rl, ll = ev["rl_ppo"], ev["least_loaded"]
    verdict = ("PPO achieved lower mean makespan than least-loaded on the simulated benchmark."
               if rl["makespan"] < ll["makespan"] else
               "PPO did NOT beat the least-loaded heuristic on makespan; the orchestrator keeps least-loaded in 'auto' mode.")
    m = {"data": "SIMULATED workloads", "algorithm": "PPO (stable-baselines3)", "timesteps": timesteps,
         "eval_episodes": eval_episodes, "evaluation": ev, "verdict": verdict, "training_curve": cb.curve}
    METRICS_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=80_000)
    args = ap.parse_args()
    res = train(args.timesteps)
    print(json.dumps(res["evaluation"], indent=2))
    print(res["verdict"])
