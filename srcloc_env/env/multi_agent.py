"""Parameter-shared multi-drone environment on top of SourceLocEnv (plan 4.5 다중 드론, 4.6 파라미터 공유 PPO; D9-1).

``MultiDroneEnv(scene, n_drones=2, **SourceLocEnv kwargs)`` runs ``n_drones`` drones in ONE scene with ONE shared
RB-PF belief (plan 4.3: the drones' measurements are fused sequentially into the central PF) and returns per-agent
observations for a parameter-shared policy:

    obs[i] = SourceLocEnv observation of drone i (56)  +  for every teammate j != i (ascending j):
             ((x_j - x_i) / 1000, (y_j - y_i) / 1000, teammate's most recent normalised log count)      -> 56 + 3 (n - 1)
    (v1: 2 drones 59, 3 drones 62; v2: 67 / 70; config.agent_obs_dim; config.ENV_TEAMMATE_DIM = 3)

Step: actions (n,) -> each drone in index order: mask (DroneKinematics; a masked action is a stay), move, one count
measurement at its own position, RBPF.update; then ONE entropy evaluation, ONE GMM refresh and ONE team reward
(plan 4.5 팀 공통 보상: time cost once per step + normalised entropy reduction of the shared belief + terminal term),
success / truncation exactly as SourceLocEnv.  Every drone receives the same scalar reward.
Reset: drone 0 is drawn exactly as SourceLocEnv (same rng stream -> identical start for the 1- and n-drone
environments under the same seed), drones 1..n-1 by the same rule with a minimum separation
config.ENV_START_MIN_SEPARATION_M; options["start_xy"] may be (2,) for drone 0 only or (n, 2).
n_drones = 1 reproduces SourceLocEnv bit for bit (observations (1, 56), same rewards; regression test).
Spaces: observation Box (n, 56 + 3 (n - 1)), action MultiDiscrete([9] * n); ``action_masks()`` -> (n, 9) bool.
This is not a gymnasium single-agent env (the env_checker does not apply); scripts/validate_env.py --n-drones runs the
random-policy safety check (T2-1) and the timing (T2-3) for it.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from gymnasium import spaces

from srcloc_env import config
from srcloc_env.env.drone import ACTION_STAY
from srcloc_env.env.source_env import Scene, SourceLocEnv


class MultiDroneEnv(SourceLocEnv):
    """n drones, one shared belief, per-agent observations and a team reward (module docstring)."""

    def __init__(self, scene: Scene, n_drones: int = 2, min_separation: float = config.ENV_START_MIN_SEPARATION_M,
                 **kwargs: Any) -> None:
        super().__init__(scene, **kwargs)
        if int(n_drones) < 1:
            raise ValueError("n_drones must be >= 1")
        self.n_drones = int(n_drones)
        self.min_separation = float(min_separation)
        self.teammate_dim = int(config.ENV_TEAMMATE_DIM)
        self.agent_obs_dim = self.obs_dim + self.teammate_dim * (self.n_drones - 1)
        self.observation_space = spaces.Box(-config.ENV_OBS_BOUND, config.ENV_OBS_BOUND,
                                            shape=(self.n_drones, self.agent_obs_dim), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([config.DRONE_N_ACTIONS] * self.n_drones)
        self.xys = np.zeros((self.n_drones, 2))
        self._recents: list[list[tuple[float, float, float]]] = [[] for _ in range(self.n_drones)]
        self._since_det = np.zeros(self.n_drones, dtype=np.int64)
        self._seen = np.zeros(self.n_drones, dtype=bool)
        self._last_count_norm = np.zeros(self.n_drones)

    # ------------------------------------------------------------------ per-drone state swapping
    def _swap_in(self, i: int) -> None:
        self.xy = self.xys[i].copy()
        self._recent = self._recents[i]
        self.steps_since_detection = int(self._since_det[i])
        self._seen_det = bool(self._seen[i])

    def _swap_out(self, i: int) -> None:
        self.xys[i] = self.xy
        self._recents[i] = self._recent
        self._since_det[i] = self.steps_since_detection
        self._seen[i] = self._seen_det

    def privileged(self) -> np.ndarray:
        """(n, config.ENV_PRIV_DIM) training-only critic features of every drone (SourceLocEnv.privileged)."""
        return np.vstack([self._privileged_of(self.xys[i]) for i in range(self.n_drones)])

    def action_masks(self) -> np.ndarray:
        """(n, 9) bool: allowed actions of every drone at its current position."""
        return np.vstack([self.kin.action_mask(self.xys[i]) for i in range(self.n_drones)])

    def action_mask(self) -> np.ndarray:
        """(9,) mask of the drone whose state is currently swapped in (SourceLocEnv API; use action_masks())."""
        return self.kin.action_mask(self.xy)

    # ------------------------------------------------------------------ observations / info
    def _observations(self) -> np.ndarray:
        obs = np.zeros((self.n_drones, self.agent_obs_dim), dtype=np.float32)
        for i in range(self.n_drones):
            self._swap_in(i)
            obs[i, :self.obs_dim] = super()._observation()
            k = self.obs_dim
            for j in range(self.n_drones):
                if j == i:
                    continue
                rel = (self.xys[j] - self.xys[i]) / config.ENV_REL_NORM_M
                obs[i, k:k + 3] = (rel[0], rel[1], self._last_count_norm[j])
                k += 3
        return np.clip(obs, -config.ENV_OBS_BOUND, config.ENV_OBS_BOUND).astype(np.float32)

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = {"source": self.source, "truth_xy": self.truth_xy.copy(), "frame": self.frame_meas,
                "scale": self.scale, "reflected": self.reflected, "t": self.t, "drone_xy": self.xys.copy(), "start_type": self.start_type,
                "action_mask": self.action_masks(), "map_error_m": self._map_error(), "entropy": self.h_prev,
                "top_sigma_m": self.gmm.top_sigma(), "n_drones": self.n_drones}
        info.update(extra)
        return info

    # ------------------------------------------------------------------ gym-like API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        opt = dict(options or {})
        starts = None
        if "start_xy" in opt:
            arr = np.asarray(opt["start_xy"], dtype=np.float64)
            if arr.ndim == 2:
                if arr.shape != (self.n_drones, 2):
                    raise ValueError(f"start_xy must be (2,) or ({self.n_drones}, 2)")
                starts = arr.copy()
                opt["start_xy"] = arr[0]
        super().reset(seed=seed, options=opt)             # drone 0, PF, GMM: identical to SourceLocEnv
        xys = [self.xy.copy()]
        start_type0 = self.start_type                     # team start type = the type of drone 0 (the draw shared with the 1-drone episode)
        for i in range(1, self.n_drones):
            if starts is not None:
                p = starts[i]
                if not self.scene.obstacles.is_free(p, self.z):
                    raise ValueError(f"start {p} of drone {i} is not a free position")
            else:
                p = None
                for _ in range(config.ENV_START_MAX_BATCHES):
                    cand = self._sample_start(self.np_random)
                    if all(np.hypot(*(cand - q)) >= self.min_separation for q in xys):
                        p = cand
                        break
                if p is None:
                    raise RuntimeError(f"no start >= {self.min_separation} m from the other drones found")
            xys.append(np.asarray(p, dtype=np.float64))
        self.xys = np.vstack(xys)
        self.start_type = start_type0
        self._recents = [[] for _ in range(self.n_drones)]
        self._since_det = np.zeros(self.n_drones, dtype=np.int64)
        self._seen = np.zeros(self.n_drones, dtype=bool)
        self._last_count_norm = np.zeros(self.n_drones)
        self._swap_in(0)
        return self._observations(), self._info(success=False, y=[None] * self.n_drones)

    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict]:   # noqa: ANN001 - int or (n,) sequence
        if self.pf is None:
            raise RuntimeError("call reset() first")
        t_all = time.perf_counter()
        acts = np.atleast_1d(np.asarray(actions, dtype=np.int64)).reshape(-1)
        if acts.shape != (self.n_drones,):
            raise ValueError(f"actions must have shape ({self.n_drones},), got {acts.shape}")
        if np.any((acts < 0) | (acts >= config.DRONE_N_ACTIONS)):
            raise ValueError(f"actions must be in 0..{config.DRONE_N_ACTIONS - 1}")
        applied, ys, dens = [], [], []
        t_pf = 0.0
        for i in range(self.n_drones):
            self._swap_in(i)
            a = int(acts[i])
            ok = bool(self.kin.action_mask(self.xy)[a])
            if not ok:
                a = ACTION_STAY
            self.xy, _ = self.kin.step(self.xy, a)
            t0 = time.perf_counter()
            y, d = self._measure()
            self.pf.update(y, np.array([self.xy[0], self.xy[1], self.z]))
            t_pf += time.perf_counter() - t0
            yn = float(self.det.normalise(y))
            self._recent.append((yn, float(self.xy[0]), float(self.xy[1])))
            if len(self._recent) > self.n_recent:
                del self._recent[:-self.n_recent]
            hit = bool(self.det.is_detection(y))
            self.steps_since_detection = 0 if hit else self.steps_since_detection + 1
            self._seen_det = self._seen_det or hit
            self._last_count_norm[i] = yn
            self._swap_out(i)
            applied.append(ok); ys.append(y); dens.append(d)
        exited = not bool(np.all(self.scene.obstacles.in_domain(self.xys)))
        self.t += 1
        h = self.pf.entropy_xy()
        gain = (self.h_prev - h) / self.h0
        self.h_prev = h
        t0 = time.perf_counter()
        gmm_refreshed = (self.t % self.gmm_every == 0)
        if gmm_refreshed:
            self._refresh_gmm()
        t_gmm = time.perf_counter() - t0
        err = self._map_error()
        sigma = self.gmm.top_sigma()
        success = bool(gmm_refreshed and sigma < self.success_sigma and err < self.success_error)
        strict = bool(gmm_refreshed and sigma < config.SUCCESS_SIGMA_M and err < config.SUCCESS_ERROR_M)
        terminated = (success and self.terminate_on_success) or exited
        truncated = (not terminated) and self.t >= self.max_steps
        reward, shaped = self._reward(gain, exited, success, truncated or terminated, err)
        t0 = time.perf_counter()
        obs = self._observations()
        t_obs = time.perf_counter() - t0
        self._swap_in(0)
        self.last_timing = {"pf_s": t_pf, "gmm_s": t_gmm, "obs_s": t_obs, "total_s": time.perf_counter() - t_all}
        info = self._info(success=success, success_strict=strict, y=ys, density=dens, applied=applied, exited=exited, info_gain=gain,
                          gmm_refreshed=gmm_refreshed, shaping_reward=shaped)
        return obs, float(reward), terminated, truncated, info
