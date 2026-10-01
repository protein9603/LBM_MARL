"""Baseline policies on the shared environment (plan 4.7 베이스라인; D9-2).

All policies act on a ``SourceLocEnv`` or ``MultiDroneEnv`` through the same interface and therefore share the RB-PF,
sensor, reward and success test of the environment (plan 4.7 "모두 같은 RBPF·센서·보상·성공 판정 코드를 공유"):

    policy.reset(env, info)                      # after env.reset
    actions = policy.act(env, obs, info)         # (n_drones,) int64, every entry an allowed action

random      uniform over the allowed actions of each drone (R: random-walk baseline).
lawnmower   each drone follows a precomputed sawtooth lawnmower (scripts/validate_pf_adjoint.lawnmower_path: sweep width
            config.PF_ADJ_SWEEP_WIDTH_M, advancing upwind = -x, obstacles skipped) from its own start; the allowed action
            whose landing point is nearest to the current waypoint is taken (waypoint index = step).  Several drones sweep
            their own bands around their own starts (영역 분할), alternating phase.
greedy_map  move towards the mean of the top GMM component of the shared belief (allowed action minimising the distance of
            its landing point to the target); stay when already within half a step.
gmm_infotaxis   one-step expected entropy reduction (Infotaxis, R6 / GMM-Infotaxis R7; D9-3: vectorised, common random
            numbers across the candidate actions so that actions differ only through their predicted rates): for every allowed action the
            drone's landing point p' is scored by the expected posterior entropy of the belief after one measurement
            there, estimated on a weighted subsample of config.INFOTAXIS_N_SUB particles with config.INFOTAXIS_N_SAMPLES
            predictive count samples (y ~ likelihood(lambda), lambda = (kappa g(theta, p') + b) T with (theta, kappa)
            drawn from the subsample and its kappa grid; posterior weights by the PF's own kappa-marginalised likelihood;
            entropy = PF_ENTROPY_CELL_M histogram entropy, the same H as the reward).  Drones are chosen sequentially
            (greedy): after drone i's action the subsample is virtually updated with the expected count at p'_i so that
            drone i + 1 scores the already-informed belief (plan 4.7 "순차 탐욕").
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson as _poisson

from srcloc_env import config
from srcloc_env.env.drone import ACTION_STAY
from srcloc_env.scripts.validate_pf_adjoint import lawnmower_path


def _n_drones(env) -> int:
    return int(getattr(env, "n_drones", 1))


def _positions(env) -> np.ndarray:
    return env.xys.copy() if hasattr(env, "xys") else env.xy[None, :].copy()


def _masks(env) -> np.ndarray:
    return env.action_masks() if hasattr(env, "action_masks") else env.action_mask()[None, :]


def _landing(env, xy: np.ndarray) -> np.ndarray:
    """(9, 2) landing points of every action from xy (masked actions land on xy itself, like the kinematics)."""
    land = env.kin.landing_points(xy)[0]
    mask = env.kin.action_mask(xy)
    return np.where(mask[:, None], land, xy[None, :])


class Policy:
    name = "base"

    def reset(self, env, info: dict) -> None:   # noqa: D401
        pass

    def act(self, env, obs: np.ndarray, info: dict) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, env, info: dict) -> None:
        self.rng = np.random.default_rng([int(info.get("t", 0)), int(info["source"]), int(info["frame"]), 7])

    def act(self, env, obs, info) -> np.ndarray:
        masks = _masks(env)
        return np.array([int(self.rng.choice(np.flatnonzero(m))) for m in masks], dtype=np.int64)


class LawnmowerPolicy(Policy):
    name = "lawnmower"

    def __init__(self, sweep_width: float = config.PF_ADJ_SWEEP_WIDTH_M, direction: float = -1.0) -> None:
        self.sweep_width = float(sweep_width)
        self.direction = float(direction)
        self.paths: list[np.ndarray] = []

    def reset(self, env, info: dict) -> None:
        self.paths = []
        for i, xy in enumerate(_positions(env)):
            path, _ = lawnmower_path(env.scene.obstacles, float(xy[0]), float(xy[1]), env.max_steps + 1,
                                     step_m=env.kin.step_m, sweep_width=self.sweep_width, direction=self.direction,
                                     z=env.z, prior_x=env.prior_x, prior_y=env.prior_y, phase_up=(i % 2 == 0))
            self.paths.append(path)

    def act(self, env, obs, info) -> np.ndarray:
        t = int(info["t"])
        acts = []
        for i, xy in enumerate(_positions(env)):
            target = self.paths[i][min(t + 1, len(self.paths[i]) - 1)]
            land = _landing(env, xy)
            mask = env.kin.action_mask(xy)
            d = np.hypot(*(land - target).T)
            d[~mask] = np.inf
            acts.append(int(np.argmin(d)))
        return np.array(acts, dtype=np.int64)


class GreedyMapPolicy(Policy):
    name = "greedy_map"

    def act(self, env, obs, info) -> np.ndarray:
        target = env.gmm.means[0]
        acts = []
        for xy in _positions(env):
            if np.hypot(*(xy - target)) < 0.5 * env.kin.step_m:
                acts.append(ACTION_STAY)
                continue
            land = _landing(env, xy)
            mask = env.kin.action_mask(xy)
            d = np.hypot(*(land - target).T)
            d[~mask] = np.inf
            acts.append(int(np.argmin(d)))
        return np.array(acts, dtype=np.int64)


def _log_pmf(y: float, lam: np.ndarray, likelihood: str, r: float) -> np.ndarray:
    """log p(y | lam) of the PF's count likelihood (Poisson or negative-binomial with dispersion r), elementwise."""
    if likelihood == "poisson":
        return y * np.log(lam) - lam - gammaln(y + 1.0)
    t = np.log(lam + r)
    return y * (np.log(lam) - t) + r * (np.log(r) - t) + gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)


def _hist_entropy(xy: np.ndarray, w: np.ndarray, cell: float, prior_x, prior_y) -> float:
    nx = max(1, int(np.ceil((prior_x[1] - prior_x[0]) / cell)))
    ny = max(1, int(np.ceil((prior_y[1] - prior_y[0]) / cell)))
    ix = np.clip(((xy[:, 0] - prior_x[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - prior_y[0]) / cell).astype(int), 0, ny - 1)
    h = np.bincount(ix * ny + iy, weights=w, minlength=nx * ny)
    p = h[h > 0.0]
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


class GmmInfotaxisPolicy(Policy):
    """One-step expected-entropy-reduction policy on the shared PF belief (module docstring)."""
    name = "gmm_infotaxis"

    def __init__(self, n_sub: int = config.INFOTAXIS_N_SUB, n_samples: int = config.INFOTAXIS_N_SAMPLES,
                 cell: float = config.PF_ENTROPY_CELL_M, seed: int = 0) -> None:
        self.n_sub, self.n_samples, self.cell = int(n_sub), int(n_samples), float(cell)
        self.rng = np.random.default_rng(seed)
        self.last_scores: list[np.ndarray] = []

    def reset(self, env, info: dict) -> None:
        self.rng = np.random.default_rng([int(info["source"]), int(info["frame"]), 11])

    def _subsample(self, env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pf = env.pf
        w = pf.weights()
        n = min(self.n_sub, pf.N)
        idx = self.rng.choice(pf.N, size=n, replace=True, p=w)            # weighted bootstrap -> uniform weights
        logv = pf.logv[idx].astype(np.float64)
        logv = logv - logsumexp(logv, axis=1, keepdims=True)               # normalised kappa grid per particle
        return pf.xy[idx].copy(), np.full(n, 1.0 / n), logv

    def _draw_common(self, env, xy_sub, w_sub, logv_sub) -> dict:
        """Common random numbers of one decision (shared by every candidate action): particle index, kappa index,
        Gamma mixing variable (negbin = Poisson(lambda G), G ~ Gamma(r, 1/r); G = 1 for Poisson) and a uniform."""
        pf = env.pf
        m = self.rng.choice(xy_sub.shape[0], size=self.n_samples, p=w_sub)
        v = np.exp(logv_sub[m])
        v /= v.sum(axis=1, keepdims=True)
        cum = np.cumsum(v, axis=1)
        gidx = np.minimum((cum < self.rng.random(self.n_samples)[:, None]).sum(axis=1), pf.G - 1)
        if pf.likelihood == "poisson":
            gam = np.ones(self.n_samples)
        else:
            r = float(pf.nb_r)
            gam = self.rng.gamma(r, 1.0 / r, size=self.n_samples)
        u = self.rng.random(self.n_samples)
        return {"m": m, "g": gidx, "gam": gam, "u": u}

    def _expected_entropy(self, env, xy_sub, w_sub, logv_sub, p_xy: np.ndarray, crn: dict) -> tuple[float, float]:
        """(expected posterior entropy after one measurement at p_xy, expected count) on the subsample with the
        common random numbers ``crn`` (vectorised over the predictive samples)."""
        pf = env.pf
        g = env.scene.model.unit_response(xy_sub, np.array([p_xy[0], p_xy[1], env.z]))[:, 0]        # (n,)
        lam = (pf.kgrid[None, :] * g[:, None] + pf.background) * pf.T                                  # (n, G)
        lik = "poisson" if pf.likelihood == "poisson" else "negbin"
        r = float(pf.nb_r)
        lam_s = lam[crn["m"], crn["g"]]                                                                # (S,)
        ys = _poisson.ppf(crn["u"], lam_s * crn["gam"]).astype(np.float64)                             # CRN counts
        log_lam = np.log(lam)
        if lik == "poisson":
            base = logv_sub[None, :, :] + ys[:, None, None] * log_lam[None, :, :] - lam[None, :, :]   # (S, n, G) up to const(y)
        else:
            t = np.log(lam + r)
            base = (logv_sub[None, :, :] + ys[:, None, None] * (log_lam - t)[None, :, :] + (r * (np.log(r) - t))[None, :, :])
        mx = base.max(axis=2, keepdims=True)
        ll = np.log(np.exp(base - mx).sum(axis=2)) + mx[:, :, 0]                                        # (S, n) marginal over kappa (const(y) cancels)
        lw = np.log(w_sub)[None, :] + ll
        lw -= lw.max(axis=1, keepdims=True)
        w_post = np.exp(lw)
        w_post /= w_post.sum(axis=1, keepdims=True)
        h = np.mean([_hist_entropy(xy_sub, w_post[k], self.cell, env.prior_x, env.prior_y) for k in range(w_post.shape[0])])
        return float(h), float(np.mean(lam_s))

    def _virtual_update(self, env, xy_sub, w_sub, logv_sub, p_xy: np.ndarray, y_bar: float):
        pf = env.pf
        g = env.scene.model.unit_response(xy_sub, np.array([p_xy[0], p_xy[1], env.z]))[:, 0]
        lam = (pf.kgrid[None, :] * g[:, None] + pf.background) * pf.T
        lik = "poisson" if pf.likelihood == "poisson" else "negbin"
        lg = _log_pmf(float(round(y_bar)), lam, lik, float(pf.nb_r))
        logv_new = logv_sub + lg
        ll = logsumexp(logv_new, axis=1)
        logv_new -= ll[:, None]
        lw = np.log(w_sub) + ll
        return np.exp(lw - logsumexp(lw)), logv_new

    def act(self, env, obs, info) -> np.ndarray:
        xy_sub, w_sub, logv_sub = self._subsample(env)
        acts, self.last_scores = [], []
        for xy in _positions(env):
            crn = self._draw_common(env, xy_sub, w_sub, logv_sub)          # shared by all actions of this drone
            mask = env.kin.action_mask(xy)
            land = _landing(env, xy)
            scores = np.full(config.DRONE_N_ACTIONS, np.inf)
            ybar = np.zeros(config.DRONE_N_ACTIONS)
            for a in np.flatnonzero(mask):
                scores[a], ybar[a] = self._expected_entropy(env, xy_sub, w_sub, logv_sub, land[a], crn)
            a_best = int(np.argmin(scores))                                 # minimal expected posterior entropy
            acts.append(a_best)
            self.last_scores.append(scores)
            if _n_drones(env) > 1:
                w_sub, logv_sub = self._virtual_update(env, xy_sub, w_sub, logv_sub, land[a_best], ybar[a_best])
        return np.array(acts, dtype=np.int64)


POLICIES = {"random": RandomPolicy, "lawnmower": LawnmowerPolicy, "greedy_map": GreedyMapPolicy, "gmm_infotaxis": GmmInfotaxisPolicy}


def make_policy(name: str, **kwargs: Any) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {list(POLICIES)}")
    return POLICIES[name](**kwargs)
