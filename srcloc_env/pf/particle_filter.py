"""Rao-Blackwellised particle filter over the source position (plan 4.3, `pf/particle_filter.py`).

References (docs/references.md)
    R5  Doucet, de Freitas, Murphy & Russell (2000), Rao-Blackwellised particle filtering: the position
        theta = (x_s, y_s) is carried by N particles while the nuisance scale kappa = k * q (sensitivity times
        emission rate) is marginalised analytically per particle, so the PF state stays 2-D.
    R3  Hutchinson et al. (2018) / Ristic et al. (2016): Poisson count likelihood y ~ Poisson((kappa g + b) T)
        with g the unit response of the forward model (plan 4.2) and b the background rate.
    R6  Gelman et al. (2013) BDA3 Ch. 2: Gamma-Poisson conjugacy -> negative-binomial marginal likelihood,
        used on the b = 0 fast path (unit test T1-1) to validate the log-grid path.
    R22 Yee & Chan (1997): instantaneous plume concentrations fluctuate around their mean with a gamma-type
        distribution, so one frozen LDM snapshot seen through a mean-field forward model is over-dispersed.
    R23 Hilbe (2011) Ch. 1-2: Gamma-Poisson mixture = negative-binomial count model, Var = mu + mu^2 / r.

Two kappa-marginalisation paths (plan 4.3)
    mode = "grid" (default, any background b >= 0): every particle i keeps log-weights logv[i, g] over the
        shared log grid kgrid = kappa_ref * 10**linspace(-grid_decades, +grid_decades, n_grid) with a
        log-uniform prior (plan 4.3: kappa_ref * 10^[-2.5, +2.5], G = 26, 0.2 decade spacing).  Per update
            logv[i, g] += y log(lam_ig) - lam_ig - gammaln(y + 1),   lam_ig = (kgrid[g] g_i + b) T
            logL_i     = logsumexp_g logv[i, g]        (marginal likelihood over kappa; the kappa posterior of
            logv[i, :] -= logL_i                        particle i stays normalised)
        Memory N x G = 2000 x 26 = 52,000 floats.  ``kappa_prior="gamma"`` puts the Gamma(alpha0, 1/kappa_ref)
        prior mass (times the log-grid Jacobian kappa) on the grid instead, which makes a fine grid reproduce
        the NB path exactly up to quadrature error (unit test T1-1 cross-check).
    mode = "nb" (b = 0 only, unit tests): Gamma(a_i, beta_i) conjugate prior per particle, a_0 = alpha0 (1),
        beta_0 = 1 / kappa_ref (plan 4.3); with f = g_i T and p = beta / (beta + f)
            logL_i = gammaln(a + y) - gammaln(a) - gammaln(y + 1) + a log p + y log(1 - p);  a += y; beta += f
        which is the closed-form Gamma-Poisson marginal (R6).  It is only valid when b = 0 (the Poisson mean
        must be proportional to kappa), so the constructor raises for background > 0 and mode = "nb".
    Both paths need lam > 0, i.e. g > 0 whenever b = 0 (GaussianPlume floors g at config.FWD_G_FLOOR); a stub
    forward model returning g = 0 with b = 0 and y = 0 would produce 0 * log 0 = NaN weights.

Count likelihood (D7-1; plan 4.3 강건화, S1 T1-2 / T1-4; ``likelihood`` = config.PF_LIKELIHOOD)
    "poisson" (R3, default): the grid block above, unchanged (bit-identical to D3).
    "negbin"  (grid mode only; R22, R23): y | lam ~ NB(r, lam) with mean lam = (kgrid g + b) T and
        Var = lam + lam^2 / r, i.e. the Gamma-Poisson marginal of a count rate fluctuating as lam Gamma(r, 1/r):
            log p(y | lam) = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r log(r / (r + lam)) + y log(lam / (r + lam))
        evaluated in place in ``dtype`` as ll = y (log lam - t) + r (log r - t) + const(y) with t = log(lam + r),
        so the (N, G) block costs two logs instead of one.  r = ``nb_r`` (config.PF_NB_DISPERSION_R = 0.3: T1-4
        diagnostic of 2026-09-30, dense-cell log-residual std ~1.2 -> CV ~1.8 -> r = 1 / CV^2); r -> inf recovers
        the Poisson likelihood.  The heavy NB tail stops the overconfident collapse of the Poisson likelihood on
        the spatially correlated clumps of the frozen LDM snapshot (validation_log 'G1 FAIL 원인과 치료').  The
        "nb" mode (Gamma-kappa conjugacy) is only defined for the Poisson likelihood and raises otherwise.

Robustification (plan 4.3 "강건화"): the analytic plume misses trapped sources by up to 8x (report 2.6), so
    logL_i <- logaddexp(log(1 - eps) + logL_i, log(eps) + logmeanexp_j logL_j)
mixes every particle likelihood with the plain particle-average likelihood (eps = config.PF_EPS_MIX = 0.05;
0.1 if trapped sources fail, risk R1).  eps = 0 disables it (exact path for T1-1).

Weights and resampling: logw is kept normalised (logsumexp = 0); after every update, if
N_eff = 1 / sum w^2 < resample_frac * N (plan 4.3: N/2) the particles are systematically resampled, the parent's
kappa statistics (logv or a, beta) are copied, and the positions are jittered with N(0, jitter_m^2) (3 m).
A jittered position that leaves the prior box or lands on a blocked cell of the obstacle map (no-fly at
altitude z, env/drone.py) is re-drawn from its parent up to config.PF_JITTER_MAX_TRIES times, then the parent
position is kept.  Measurements of several drones are fused by calling ``update`` sequentially on the one
central filter (plan 4.3).

Return value of ``update``: the predictive log-likelihood of the measurement under the belief before the
update, log sum_i w_i L_i (after robustification), i.e. the marginal log-likelihood used by likelihood-based
diagnostics; for N = 1 and eps = 0 it is exactly the particle's marginal (T1-1).

Summaries for the reward and the GMM stage (plan 4.4 / 4.5): ``mean``, ``covariance``, ``map_estimate``
(weighted mode: the densest config.PF_ENTROPY_CELL_M cell of the weighted 2-D histogram, refined by the
weighted mean of the particles in its 3 x 3 cell neighbourhood; ``method="top"`` gives the weighted mean of
the top config.PF_MAP_TOP_FRACTION particles by weight instead), ``entropy_xy`` (entropy of the weighted 2-D
histogram on config.PF_ENTROPY_CELL_M = 20 m cells over the prior box, in nats; plan 4.5 reward),
``posterior_kappa`` (per-particle or weighted-mixture kappa posterior on the grid), ``posterior_kappa_quantiles``
(cell-edge CDF in log10 kappa: the median of the flat prior is kappa_ref),
``kappa_edge_mass`` (weight on the two extreme grid cells; diagnostic of risk R14) and
``positions_and_weights`` for the GMM summary.

Everything is numpy-vectorised over (N, G).  The grid update is dominated by one log and one exp over the
N x G block (52,000 floats); on the D3 machine that is ~2 ms in float64 and ~0.5 ms with ``dtype=np.float32``
(scripts/validate_kappa_grid.py measures both).  float32 log-weights are renormalised every update, so their
~1e-3 accumulated error over an episode is negligible for the position posterior; the default stays float64 so
that the T1-1 grid-vs-NB check holds to 1e-3.  All defaults come from srcloc_env.config.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.special import gammaln

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.sensor.detector import expected_counts_from_kappa   # reference formula (unit-tested against update)


class ForwardModel(Protocol):
    """Hypothesis -> unit response interface of pf/forward_model.GaussianPlume (plan 4.2)."""

    def unit_response(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> np.ndarray:
        """(N, M) response of N source hypotheses at M drone positions."""
        ...


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Numerically safe log(sum(exp(a))) in plain numpy (scipy.special.logsumexp costs ~2.6 ms for (2000, 26)
    rows on the D3 machine versus 0.4 ms here; the PF update budget is 1 ms, config.PF_UPDATE_TIME_TARGET_S)."""
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return float(out.item()) if axis is None else np.squeeze(out, axis=axis)


def logmeanexp(x: np.ndarray) -> float:
    """log of the plain mean of exp(x) over a 1-D array (used by the robust mixture, plan 4.3)."""
    x = np.asarray(x, dtype=np.float64)
    return float(logsumexp(x) - np.log(x.size))


class RBPF:
    """Rao-Blackwellised PF over the source position with kappa marginalised per particle (plan 4.3; R5, R3, R6).

    Parameters (defaults from config, provenance there)
        forward       object with unit_response(source_xy (N,2), drone_xyz (M,3)) -> (N, M) (GaussianPlume)
        n_particles   N (config.PF_N_PARTICLES = 2000)
        kappa_ref     centre of the kappa log grid / Gamma prior scale (config.KAPPA_REF; T1-3 sets it)
        grid_decades  half-width of the log grid in decades (config.KAPPA_GRID_DECADES = 2.5)
        n_grid        G grid points (config.KAPPA_G = 26)
        background    b [cps] (config.SENSOR_BACKGROUND_CPS); T integration time [s] (config.SENSOR_T)
        eps_mix       robust mixture weight (config.PF_EPS_MIX); jitter_m resampling jitter std [m]
        prior_x/y     uniform prior box (config.PF_PRIOR_X / PF_PRIOR_Y = LDM region, plan 4.3)
        obstacles     ObstacleMap or None: positions must be free at altitude z (config.DRONE_Z)
        mode          "grid" (default) or "nb" (b = 0 only); kappa_prior "loguniform" (default) or "gamma"
        alpha0        Gamma prior shape of the nb path / gamma grid prior (config.PF_NB_ALPHA0 = 1)
        resample_frac resample when N_eff < resample_frac * N (config.PF_RESAMPLE_NEFF_FRACTION = 0.5)
        rng           numpy Generator (default: default_rng())
        dtype         float dtype of the (N, G) kappa-grid block (np.float64 default; np.float32 is ~4x faster)
        likelihood    "poisson" (R3) or "negbin" (grid mode only; R22 / R23, D7-1); None = config.PF_LIKELIHOOD
                      (negbin since D7-4) except in mode "nb", which is Poisson by construction
        nb_r          negative-binomial dispersion r > 0 of the "negbin" likelihood (config.PF_NB_DISPERSION_R)
    ``reset`` must be called (the constructor calls it) before ``update``.
    """

    def __init__(self, forward: ForwardModel, n_particles: int = config.PF_N_PARTICLES,
                 kappa_ref: float = config.KAPPA_REF, grid_decades: float = config.KAPPA_GRID_DECADES,
                 n_grid: int = config.KAPPA_G, background: float = config.SENSOR_BACKGROUND_CPS,
                 T: float = config.SENSOR_T, eps_mix: float = config.PF_EPS_MIX,
                 jitter_m: float = config.PF_JITTER_M,
                 prior_x: tuple[float, float] = config.PF_PRIOR_X,
                 prior_y: tuple[float, float] = config.PF_PRIOR_Y,
                 obstacles: ObstacleMap | None = None, mode: str = "grid",
                 rng: np.random.Generator | None = None, z: float = config.DRONE_Z,
                 kappa_prior: str = "loguniform", alpha0: float = config.PF_NB_ALPHA0,
                 resample_frac: float = config.PF_RESAMPLE_NEFF_FRACTION,
                 max_jitter_tries: int = config.PF_JITTER_MAX_TRIES, dtype: type = np.float64,
                 likelihood: str | None = None, nb_r: float = config.PF_NB_DISPERSION_R) -> None:
        if mode not in ("grid", "nb"):
            raise ValueError(f"mode must be 'grid' or 'nb', got {mode!r}")
        if kappa_prior not in ("loguniform", "gamma"):
            raise ValueError(f"kappa_prior must be 'loguniform' or 'gamma', got {kappa_prior!r}")
        if likelihood is None:                       # config default; the conjugate 'nb' mode is Poisson by definition
            likelihood = "poisson" if mode == "nb" else config.PF_LIKELIHOOD
        if likelihood not in config.PF_LIKELIHOODS:
            raise ValueError(f"likelihood must be one of {config.PF_LIKELIHOODS}, got {likelihood!r}")
        if mode == "nb" and likelihood != "poisson":
            raise ValueError("mode='nb' (Gamma-kappa conjugacy) supports only likelihood='poisson' (plan D7-1)")
        if not np.isfinite(nb_r) or nb_r <= 0.0:
            raise ValueError("need a finite nb_r > 0")
        if mode == "nb" and background > 0.0:
            raise ValueError("mode='nb' (Gamma-Poisson conjugacy) is only valid for background == 0 (plan 4.3)")
        if n_particles < 1 or n_grid < 2 or kappa_ref <= 0.0 or grid_decades <= 0.0:
            raise ValueError("need n_particles >= 1, n_grid >= 2, kappa_ref > 0, grid_decades > 0")
        if background < 0.0 or T <= 0.0 or not 0.0 <= eps_mix < 1.0 or jitter_m < 0.0 or alpha0 <= 0.0:
            raise ValueError("need background >= 0, T > 0, 0 <= eps_mix < 1, jitter_m >= 0, alpha0 > 0")
        if prior_x[1] <= prior_x[0] or prior_y[1] <= prior_y[0]:
            raise ValueError("prior box must have positive extent")
        self.forward = forward
        self.N = int(n_particles)
        self.kappa_ref = float(kappa_ref)
        self.grid_decades = float(grid_decades)
        self.G = int(n_grid)
        self.background = float(background)
        self.T = float(T)
        self.eps_mix = float(eps_mix)
        self.jitter_m = float(jitter_m)
        self.prior_x = (float(prior_x[0]), float(prior_x[1]))
        self.prior_y = (float(prior_y[0]), float(prior_y[1]))
        self.obstacles = obstacles
        self.mode = mode
        self.z = float(z)
        self.kappa_prior = kappa_prior
        self.alpha0 = float(alpha0)
        self.beta0 = 1.0 / self.kappa_ref
        self.resample_frac = float(resample_frac)
        self.max_jitter_tries = int(max_jitter_tries)
        self.kgrid: np.ndarray = self.kappa_ref * 10.0 ** np.linspace(-self.grid_decades, self.grid_decades, self.G)
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise ValueError("dtype must be a float type")
        self._kgrid_T = (self.kgrid * self.T).astype(self.dtype)          # lam = g (kgrid T) + b T, see update
        self._bT = self.dtype.type(self.background * self.T)
        self.likelihood = likelihood
        self.nb_r = float(nb_r)
        self._r = self.dtype.type(self.nb_r)                                   # NB dispersion in the block dtype
        self._nb_const_r = float(self.nb_r * np.log(self.nb_r) - gammaln(self.nb_r))   # r log r - gammaln(r)
        self.rng = rng if rng is not None else np.random.default_rng()
        # state (filled by reset)
        self.xy: np.ndarray = np.empty((self.N, 2))
        self.logw: np.ndarray = np.empty(self.N)
        self.logv: np.ndarray = np.empty((self.N, self.G))
        self.a: np.ndarray = np.empty(self.N)
        self.beta: np.ndarray = np.empty(self.N)
        self.last_loglik: np.ndarray = np.empty(self.N)
        self.n_updates = 0
        self.n_resamples = 0
        self.reset(self.rng)

    # ------------------------------------------------------------------ prior / reset
    def _log_prior_grid(self) -> np.ndarray:
        """Normalised log prior mass on the kappa grid: log-uniform, or Gamma(alpha0, beta0) mass * Jacobian."""
        if self.kappa_prior == "loguniform":
            return np.full(self.G, -np.log(self.G))
        lp = self.alpha0 * np.log(self.kgrid) - self.beta0 * self.kgrid      # kappa^(alpha0-1) e^{-beta0 kappa} * kappa
        return lp - logsumexp(lp)

    def _valid(self, xy: np.ndarray) -> np.ndarray:
        """bool (n,): inside the (closed) prior box and free at altitude z when an obstacle map is given."""
        ok = ((xy[:, 0] >= self.prior_x[0]) & (xy[:, 0] <= self.prior_x[1])
              & (xy[:, 1] >= self.prior_y[0]) & (xy[:, 1] <= self.prior_y[1]))
        if self.obstacles is not None:
            ok &= self.obstacles.is_free(xy, self.z)
        return ok

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Uniform prior over the prior box (positions blocked at altitude z are rejected), flat kappa prior."""
        if rng is not None:
            self.rng = rng
        xy = np.empty((self.N, 2))
        todo = np.ones(self.N, dtype=bool)
        for _ in range(config.PF_RESET_MAX_ROUNDS):
            n = int(todo.sum())
            if n == 0:
                break
            cand = np.column_stack([self.rng.uniform(*self.prior_x, n), self.rng.uniform(*self.prior_y, n)])
            good = self._valid(cand)
            idx = np.flatnonzero(todo)[good]
            xy[idx] = cand[good]
            todo[idx] = False
        if todo.any():
            raise RuntimeError("reset: could not draw enough free particles (prior box mostly blocked?)")
        self.xy = xy
        self.logw = np.full(self.N, -np.log(self.N))
        self.logv = np.tile(self._log_prior_grid(), (self.N, 1)).astype(self.dtype)
        self.a = np.full(self.N, self.alpha0)
        self.beta = np.full(self.N, self.beta0)
        self.last_loglik = np.zeros(self.N)
        self.n_updates = 0
        self.n_resamples = 0

    # ------------------------------------------------------------------ measurement update
    def update(self, y: int, drone_xyz: np.ndarray) -> float:
        """Sequential Bayes update with one count y at drone_xyz (3,); returns the predictive log-likelihood.

        grid mode: logv += y log(lam) - lam - gammaln(y+1), lam = (kgrid g + b) T (the same formula as
        sensor/detector.expected_counts_from_kappa, evaluated here in-place in ``dtype``); logL = logsumexp_g logv;
        likelihood='negbin': logv += y (log lam - t) + r (log r - t) + gammaln(y + r) - gammaln(r) - gammaln(y + 1)
        with t = log(lam + r) (module docstring, R22 / R23);
        nb mode (b = 0): negative-binomial marginal with the Gamma(a, beta) prior, then a += y, beta += g T.
        Then the robust mixture (eps_mix), logw += logL, normalisation and resampling when N_eff < frac * N.
        """
        y = int(y)
        if y < 0:
            raise ValueError("counts must be non-negative")
        drone = np.asarray(drone_xyz, dtype=np.float64).reshape(-1)
        if drone.size != 3:
            raise ValueError(f"drone_xyz must have 3 components, got {drone.size}")
        g = np.asarray(self.forward.unit_response(self.xy, drone[None, :]), dtype=np.float64)[:, 0]   # (N,)
        if self.mode == "grid":
            # in-place (N, G) arithmetic in self.dtype: one log (two for negbin), one exp and a few cheap passes
            lam = np.outer(g.astype(self.dtype, copy=False), self._kgrid_T)          # (N, G) = g kgrid T
            lam += self._bT                                                          # + b T  (detector formula)
            if self.likelihood == "poisson":
                tmp = np.log(lam)
                tmp *= self.dtype.type(y)
                tmp -= lam
                self.logv += tmp
                self.logv -= self.dtype.type(gammaln(y + 1.0))
            else:
                # negative binomial (R22 / R23, D7-1): ll = y (log lam - t) + r (log r - t) + const(y), t = log(lam + r)
                tmp = lam + self._r
                np.log(tmp, out=tmp)                                                 # t = log(lam + r)
                np.log(lam, out=lam)                                                 # log lam
                lam -= tmp
                lam *= self.dtype.type(y)                                            # y (log lam - t)
                tmp *= self._r                                                       # r t
                lam -= tmp
                self.logv += lam
                self.logv += self.dtype.type(gammaln(y + self.nb_r) - gammaln(y + 1.0) + self._nb_const_r)
            m = self.logv.max(axis=1)
            np.subtract(self.logv, m[:, None], out=tmp)
            np.exp(tmp, out=tmp)
            logL = (m + np.log(tmp.sum(axis=1))).astype(np.float64)                  # logsumexp over kappa
            self.logv -= logL[:, None].astype(self.dtype)
        else:
            f = g * self.T
            p = self.beta / (self.beta + f)
            logL = (gammaln(self.a + y) - gammaln(self.a) - gammaln(y + 1.0)
                    + self.a * np.log(p) + y * np.log1p(-p))
            self.a += y
            self.beta += f
        self.last_loglik = logL.copy()
        if self.eps_mix > 0.0:
            logL = np.logaddexp(np.log1p(-self.eps_mix) + logL, np.log(self.eps_mix) + logmeanexp(logL))
        marginal = float(logsumexp(self.logw + logL))
        self.logw = self.logw + logL - marginal
        self.n_updates += 1
        if self.neff() < self.resample_frac * self.N:
            self.resample_and_jitter()
        return marginal

    # ------------------------------------------------------------------ resampling
    def resample_and_jitter(self) -> None:
        """Systematic resampling, parent kappa statistics copied, N(0, jitter_m^2) jitter with validity re-draws."""
        w = np.exp(self.logw)
        cum = np.cumsum(w)
        cum[-1] = 1.0
        u = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.minimum(np.searchsorted(cum, u, side="right"), self.N - 1)
        parent = self.xy[idx]
        new = parent.copy()
        bad = np.ones(self.N, dtype=bool)
        if self.jitter_m > 0.0:
            for _ in range(self.max_jitter_tries):
                where = np.flatnonzero(bad)
                if where.size == 0:
                    break
                cand = parent[where] + self.rng.normal(0.0, self.jitter_m, (where.size, 2))
                good = self._valid(cand)
                new[where[good]] = cand[good]
                bad[where[good]] = False
        # particles still bad keep the parent position (new == parent there already)
        self.xy = new
        if self.mode == "grid":
            self.logv = self.logv[idx].copy()
        else:
            self.a = self.a[idx].copy()
            self.beta = self.beta[idx].copy()
        self.last_loglik = self.last_loglik[idx].copy()
        self.logw = np.full(self.N, -np.log(self.N))
        self.n_resamples += 1

    # ------------------------------------------------------------------ summaries
    def weights(self) -> np.ndarray:
        """(N,) normalised weights."""
        return np.exp(self.logw)

    def neff(self) -> float:
        """Effective sample size 1 / sum w_i^2."""
        return float(1.0 / np.sum(np.exp(2.0 * self.logw)))

    def mean(self) -> np.ndarray:
        """Weighted posterior mean position (2,)."""
        return self.weights() @ self.xy

    def covariance(self) -> np.ndarray:
        """Weighted 2 x 2 posterior covariance of the position."""
        w = self.weights()
        d = self.xy - (w @ self.xy)
        return (d * w[:, None]).T @ d

    def positions_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """(xy (N,2) copy, weights (N,)) for the GMM summary stage (plan 4.4)."""
        return self.xy.copy(), self.weights()

    def _hist_xy(self, cell: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Weighted 2-D histogram (nx, ny) on `cell`-sized bins over the prior box, plus per-particle bin ids."""
        nx = int(np.ceil((self.prior_x[1] - self.prior_x[0]) / cell))
        ny = int(np.ceil((self.prior_y[1] - self.prior_y[0]) / cell))
        ix = np.clip(np.floor((self.xy[:, 0] - self.prior_x[0]) / cell).astype(np.int64), 0, nx - 1)
        iy = np.clip(np.floor((self.xy[:, 1] - self.prior_y[0]) / cell).astype(np.int64), 0, ny - 1)
        h = np.bincount(ix * ny + iy, weights=self.weights(), minlength=nx * ny).reshape(nx, ny)
        return h, ix, iy, nx, ny

    def entropy_xy(self, cell: float = config.PF_ENTROPY_CELL_M) -> float:
        """Shannon entropy [nats] of the weighted 2-D position histogram on `cell` m cells (plan 4.5 reward)."""
        h = self._hist_xy(float(cell))[0].ravel()
        p = h[h > 0.0]
        p = p / p.sum()
        return float(-np.sum(p * np.log(p)))

    def map_estimate(self, method: str = "mode", cell: float = config.PF_ENTROPY_CELL_M,
                     top_fraction: float = config.PF_MAP_TOP_FRACTION) -> np.ndarray:
        """Point estimate (2,) of the source position.

        method = "mode" (default): the weighted mode = densest `cell` m histogram cell, refined by the weighted
        mean of the particles inside that cell and its 8 neighbours (well defined also right after resampling,
        when all weights are equal and the highest-weight particle would be arbitrary).
        method = "top": weighted mean of the top `top_fraction` particles by weight (plan 4.3 alternative).
        """
        if method == "top":
            k = max(1, int(np.ceil(top_fraction * self.N)))
            w = self.weights()
            idx = np.argpartition(-w, k - 1)[:k]
            return (w[idx] @ self.xy[idx]) / w[idx].sum()
        if method != "mode":
            raise ValueError("method must be 'mode' or 'top'")
        h, ix, iy, nx, ny = self._hist_xy(float(cell))
        bx, by = np.unravel_index(int(np.argmax(h)), h.shape)
        sel = (np.abs(ix - bx) <= 1) & (np.abs(iy - by) <= 1)
        w = self.weights()[sel]
        return (w @ self.xy[sel]) / w.sum()

    def posterior_kappa(self, particle: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(kgrid (G,), weights (G,)) kappa posterior of one particle, or the weighted mixture over particles.

        nb mode: the Gamma(a, beta) posterior is evaluated as log-cell masses on the same grid
        (kappa^a e^{-beta kappa}, normalised), so both modes share this representation.
        """
        if self.mode == "grid":
            lv = self.logv
        else:
            lv = self.a[:, None] * np.log(self.kgrid)[None, :] - self.beta[:, None] * self.kgrid[None, :]
            lv = lv - logsumexp(lv, axis=1)[:, None]
        if particle is not None:
            return self.kgrid, np.exp(lv[int(particle)])
        mix = logsumexp(lv + self.logw[:, None], axis=0)
        return self.kgrid, np.exp(mix - logsumexp(mix))

    def posterior_kappa_quantiles(self, q: tuple[float, ...] = (0.05, 0.5, 0.95)) -> np.ndarray:
        """Quantiles of the mixture kappa posterior (kappa units).

        Each grid cell's mass is spread uniformly in log10(kappa) over the cell [edge_g, edge_{g+1}] whose edges are
        the mid-points between neighbouring nodes (the outer cells are extended symmetrically), so the CDF is
        piecewise-linear between the cell edges and the median of the log-uniform prior is exactly kappa_ref.
        (A step CDF evaluated at the nodes would bias every quantile low by half a cell, 0.1 decade.)
        """
        kg, w = self.posterior_kappa()
        lk = np.log10(kg)
        half = 0.5 * np.diff(lk)
        edges = np.concatenate(([lk[0] - half[0]], lk[:-1] + half, [lk[-1] + half[-1]]))     # (G + 1,)
        cdf = np.concatenate(([0.0], np.cumsum(w)))
        cdf = cdf / cdf[-1]
        return 10.0 ** np.interp(np.asarray(q, dtype=np.float64), cdf, edges)

    def kappa_edge_mass(self) -> tuple[float, float]:
        """Mixture posterior mass on the lowest and highest grid cell (risk R14 diagnostic: grid too narrow)."""
        _, w = self.posterior_kappa()
        return float(w[0]), float(w[-1])