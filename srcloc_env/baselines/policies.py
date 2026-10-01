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
from srcloc_env.baselines.planning import GeodesicField
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


_GEO_CACHE: dict = {}


def _geodesic(env) -> GeodesicField:
    """One GeodesicField per scene (obstacle raster) and flight height, built lazily and cached per process."""
    key = (id(env.scene.obstacles), float(env.z))
    if key not in _GEO_CACHE:
        _GEO_CACHE[key] = GeodesicField(env.scene.obstacles, env.z)
    return _GEO_CACHE[key]


class _TargetFollower:
    """Shared geodesic target following: the shortest-path field to the target is recomputed when the target has moved by
    more than ``retarget_m``; the action is the allowed one with the smallest field value at its landing point."""

    def __init__(self, retarget_m: float = 20.0) -> None:
        self.retarget_m = float(retarget_m)
        self._target = None
        self._field = None

    def reset_target(self) -> None:
        self._target, self._field = None, None

    def landing_values(self, env, target: np.ndarray, land: np.ndarray) -> np.ndarray:
        """Geodesic values (distance-to-go [m]) of landing points (k, 2) for the current target."""
        geo = _geodesic(env)
        target = np.asarray(target, dtype=float)
        if self._target is None or np.hypot(*(target - self._target)) > self.retarget_m:
            self._target, self._field = target.copy(), geo.field(target)
        return geo.value(self._field, land)

    def step_towards(self, env, xy: np.ndarray, target: np.ndarray) -> int:
        geo = _geodesic(env)
        target = np.asarray(target, dtype=float)
        if self._target is None or np.hypot(*(target - self._target)) > self.retarget_m:
            self._target, self._field = target.copy(), geo.field(target)
        if np.hypot(*(xy - target)) < 0.5 * env.kin.step_m:
            return ACTION_STAY
        land = _landing(env, xy)
        mask = env.kin.action_mask(xy)
        val = geo.value(self._field, land)
        val = np.where(mask, val, np.inf)
        return int(np.argmin(val))


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
    """Every drone follows the shortest free path (around buildings, GeodesicField) to the mean of the top GMM component of
    the shared belief (D9-4: the first version took the nearest landing point in Euclidean distance and was trapped by the
    first building between drone and target)."""
    name = "greedy_map"

    def __init__(self) -> None:
        self._followers: list[_TargetFollower] = []

    def reset(self, env, info: dict) -> None:
        self._followers = [_TargetFollower() for _ in range(_n_drones(env))]

    def act(self, env, obs, info) -> np.ndarray:
        target = env.gmm.means[0]
        if len(self._followers) != _n_drones(env):
            self._followers = [_TargetFollower() for _ in range(_n_drones(env))]
        return np.array([f.step_towards(env, xy, target) for f, xy in zip(self._followers, _positions(env))], dtype=np.int64)


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
    """One-step expected-entropy-reduction policy on the shared PF belief (module docstring).  D9-4: when several allowed
    actions are within ``tie_tol`` nats of the best expected entropy (flat information, e.g. no detection yet) the drone
    follows the shortest free path to the belief's top component (greedy tie-break) instead of an arbitrary pick."""
    name = "gmm_infotaxis"

    def __init__(self, n_sub: int = config.INFOTAXIS_N_SUB, n_samples: int = config.INFOTAXIS_N_SAMPLES,
                 cell: float = config.PF_ENTROPY_CELL_M, seed: int = 0, tie_tol: float = config.INFOTAXIS_TIE_TOL_NATS) -> None:
        self.n_sub, self.n_samples, self.cell = int(n_sub), int(n_samples), float(cell)
        self.tie_tol = float(tie_tol)
        self.rng = np.random.default_rng(seed)
        self.last_scores: list[np.ndarray] = []
        self._followers: list[_TargetFollower] = []

    def reset(self, env, info: dict) -> None:
        self.rng = np.random.default_rng([int(info["source"]), int(info["frame"]), 11])
        self._followers = [_TargetFollower() for _ in range(_n_drones(env))]

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
        if len(self._followers) != _n_drones(env):
            self._followers = [_TargetFollower() for _ in range(_n_drones(env))]
        for i_drone, xy in enumerate(_positions(env)):
            crn = self._draw_common(env, xy_sub, w_sub, logv_sub)          # shared by all actions of this drone
            mask = env.kin.action_mask(xy)
            land = _landing(env, xy)
            scores = np.full(config.DRONE_N_ACTIONS, np.inf)
            ybar = np.zeros(config.DRONE_N_ACTIONS)
            for a in np.flatnonzero(mask):
                scores[a], ybar[a] = self._expected_entropy(env, xy_sub, w_sub, logv_sub, land[a], crn)
            a_best = int(np.argmin(scores))                                 # minimal expected posterior entropy
            near = np.flatnonzero(mask & (scores <= scores.min() + self.tie_tol))
            if near.size > 1:                                               # flat information (no signal): follow the belief (greedy tie-break, D9-4)
                vals = self._followers[i_drone].landing_values(env, env.gmm.means[0], land[near])
                a_best = int(near[int(np.argmin(vals))])
            acts.append(a_best)
            self.last_scores.append(scores)
            if _n_drones(env) > 1:
                w_sub, logv_sub = self._virtual_update(env, xy_sub, w_sub, logv_sub, land[a_best], ybar[a_best])
        return np.array(acts, dtype=np.int64)


class OracleLoiterPolicy(Policy):
    """VERIFICATION ONLY (plan rule 3: other methods only where code verification needs them): every drone follows the
    shortest free path to the PLUME PEAK of the episode's source (privileged information: the smoothed 15 m density maximum
    of the truth slab within 200 m of the source - the plume rises and drifts downwind, so the concentration at the source
    itself is ~0) and loiters there (random allowed action that stays within ``loiter_m`` of the peak).  It bounds what the
    shared PF / sensor / success test achieves when the search problem is removed; it is never reported as a method."""
    name = "oracle_loiter"

    def __init__(self, loiter_m: float = 25.0, seed: int = 0, radius_m: float = 200.0) -> None:
        self.loiter_m, self.radius_m = float(loiter_m), float(radius_m)
        self.rng = np.random.default_rng(seed)
        self.peak = None
        self._followers: list[_TargetFollower] = []

    def reset(self, env, info: dict) -> None:
        self.rng = np.random.default_rng([int(info["source"]), int(info["frame"]), 13])
        self._followers = [_TargetFollower(retarget_m=1e9) for _ in range(_n_drones(env))]
        self.peak = self._plume_peak(env)

    def _plume_peak(self, env) -> np.ndarray:
        truth = np.asarray(env.truth_xy, dtype=float)
        try:
            from scipy.ndimage import gaussian_filter
            sf = env.scene.backend.slab(env.frame0)
            g = sf.grid
            dens = sf.density[list(sf.sources).index(env.source), sf.z_index(env.z)].astype(float)
            if env.scene.reflected:
                return truth                                              # not used in evaluation; keep it simple
            sm = gaussian_filter(dens, 2.0)
            xx, yy = np.meshgrid(g.x_centres, g.y_centres)
            sm[np.hypot(xx - truth[0], yy - truth[1]) > self.radius_m] = 0.0
            if sm.max() <= 0.0:
                return truth
            j = int(np.argmax(sm))
            return np.array([xx.flat[j], yy.flat[j]])
        except AttributeError:
            return truth                                                  # backend without slabs (unit tests)

    def act(self, env, obs, info) -> np.ndarray:
        acts = []
        for f, xy in zip(self._followers, _positions(env)):
            land = _landing(env, xy)
            mask = env.kin.action_mask(xy)
            d = np.hypot(*(land - self.peak).T)
            if np.hypot(*(xy - self.peak)) > self.loiter_m:
                acts.append(f.step_towards(env, xy, self.peak))
            else:
                ok = np.flatnonzero(mask & (d <= self.loiter_m))
                acts.append(int(self.rng.choice(ok)) if ok.size else ACTION_STAY)
        return np.array(acts, dtype=np.int64)


POLICIES = {"random": RandomPolicy, "lawnmower": LawnmowerPolicy, "greedy_map": GreedyMapPolicy, "gmm_infotaxis": GmmInfotaxisPolicy}
VERIFICATION_POLICIES = {"oracle_loiter": OracleLoiterPolicy}         # privileged-information bounds, excluded from the default method lists


def make_policy(name: str, **kwargs: Any) -> Policy:
    table = {**POLICIES, **VERIFICATION_POLICIES}
    if name not in table:
        raise ValueError(f"unknown policy {name!r}; choose from {list(table)}")
    return table[name](**kwargs)
