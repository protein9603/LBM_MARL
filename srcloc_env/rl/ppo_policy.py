"""Evaluation adapter: a trained PPO checkpoint as a baselines.policies.Policy (D10-4).

The evaluation runner (eval/run_eval.py) gives every method the same environment, belief, episodes and metrics; this class lets a
checkpoint of rl/train.py take part under an alias name.  By default the action is SAMPLED from the learned policy (the stochastic
policy PPO optimised; config.PPO_EVAL_DETERMINISTIC); deterministic=True takes the arg-max action (reference rows only).  The
sampling generator is seeded from (source, frame, scale) of the episode, so repeated evaluation reproduces the episode.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from srcloc_env import config
from srcloc_env.baselines.policies import Policy, _masks
from srcloc_env.rl.ppo import ActorCritic, net_from_state


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class PPOPolicy(Policy):
    def __init__(self, checkpoint: str | Path, alias: str = "ppo", deterministic: bool = config.PPO_EVAL_DETERMINISTIC) -> None:
        torch.set_num_threads(1)
        self.checkpoint = str(checkpoint)
        self.name = alias
        self.deterministic = bool(deterministic)
        ck = load_checkpoint(checkpoint)
        self.net: ActorCritic = net_from_state(ck["learner"])
        self.net.eval()
        self.meta = {k: ck.get(k) for k in ("iteration", "env_steps", "n_drones", "run_name", "truth_mode", "max_steps", "obs_version")}
        self.meta["wind_level"] = str(ck.get("wind_level", config.WIND_LEVEL_DEFAULT))          # checkpoints before D13 are W2
        self.meta["obs_wind"] = str(ck.get("obs_wind", "model"))                                  # checkpoints before D15 read the model wind
        ekw = dict(ck.get("env_kw") or {})
        self.meta["wind_u"] = float(ekw.get("wind_u", config.WIND_MEAN_U)) if self.meta["wind_level"] != "W2" else None
        self.gen = torch.Generator().manual_seed(0)

    def reset(self, env, info: dict) -> None:   # noqa: ANN001
        n = int(getattr(env, "n_drones", 1))
        d = int(getattr(env, "agent_obs_dim", env.obs_dim))
        if d != self.net.obs_dim:
            raise ValueError(f"checkpoint {self.checkpoint} expects {self.net.obs_dim} observation inputs, the {n}-drone environment gives {d}")
        for key, val in (("truth_mode", getattr(env, "truth_mode", None)), ("max_steps", getattr(env, "max_steps", None)), ("obs_version", getattr(env, "obs_version", None)),
                         ("wind_level", getattr(getattr(env, "scene", None), "wind_level", None)),
                         ("obs_wind", getattr(getattr(env, "scene", None), "obs_wind", None)),
                         ("wind_u", (round(float(getattr(env.scene, "wind_u")), 3) if getattr(env, "scene", None) is not None and getattr(env.scene, "wind_level", "W2") != "W2" else None))):
            if self.meta.get(key) is not None and val is not None and self.meta[key] != val:
                warnings.warn(f"checkpoint {self.checkpoint} was trained with {key} = {self.meta[key]} but is evaluated with {key} = {val}", stacklevel=2)
        seed = np.random.SeedSequence([int(info["source"]), int(info["frame"]), int(round(float(info["scale"]) * 1e6)), 13])
        self.gen = torch.Generator().manual_seed(int(seed.generate_state(1)[0]))

    def act(self, env, obs: np.ndarray, info: dict) -> np.ndarray:   # noqa: ANN001
        n = int(getattr(env, "n_drones", 1))
        o = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(n, -1))
        m = torch.from_numpy(np.asarray(_masks(env), dtype=bool).reshape(n, -1))
        a, _, _ = self.net.act(o, m, self.gen, deterministic=self.deterministic)
        return a.numpy().astype(np.int64)
