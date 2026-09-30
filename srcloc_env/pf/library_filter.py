"""Discrete Bayesian candidate filter over the K real sources with the library forward model (plan 4.2
"라이브러리 모델", S1 "라이브러리 필터 상한" / T1-4 library upper bound; `pf/library_filter.py`, D5-4).

Model (plan 4.2 / 4.3, references R3, R5, R6)
    The unknown source is one of K candidates (config.ALL_SOURCES, the 13 LDM release positions of report 2.6)
    and the scale kappa is a nuisance.  Each measurement is a Poisson count y ~ Poisson((kappa g_j(p) + b) T)
    (R3, sensor/detector.expected_counts_from_kappa) where g_j(p) is candidate j's response at the drone
    position p, delivered by any callable ``unit_response_fn(drone_xyz) -> (K,)``:
      (a) ``library_unit_response_fn``: the cached LDM slab of source j (pf/forward_model.LibraryModel), i.e. the
          data itself as the model - the library filter is the UPPER BOUND on what any forward model can do,
          since the truth field and the model field coincide (plan T1-4; validate_library_filter).  CAVEAT
          (plan 4.2): the slab is the density for the ACTUAL release of the simulation (config.
          PARTICLES_PER_INDEX_STEP_PER_SOURCE per index step), not a per-unit-q response, so kappa absorbs only
          k * scale (= config.SENSOR_K0 x scale = 1000 for scale 1), not k * scale * q.
      (b) ``adjoint_unit_response_fn``: the calibrated adjoint unit-source fields of the 13 sources
          (scripts/calibrate_adjoint.py -> config.LIB_ADJOINT_FIELDS_NPZ, plan 4.2b / T1-3b) sampled
          bilinearly at the drone (the same cell-centre convention as field/concentration_field.py); kappa then
          absorbs k * scale * q as in the RB-PF.  Returns None when the file does not exist.
    kappa is marginalised exactly as in the RB-PF grid mode (pf/particle_filter.RBPF, plan 4.3, R5): each
    candidate j keeps log kappa-posterior masses logv[j, g] on the shared log grid
    kgrid = kappa_ref 10^linspace(-grid_decades, grid_decades, n_grid) with a log-uniform prior, and per update
        logv[j, g] += y log lam_jg - lam_jg - gammaln(y + 1),   lam_jg = (kgrid[g] g_j + b) T
        logL_j      = logsumexp_g logv[j, g];  logv[j, :] -= logL_j
    (the marginal likelihood over kappa and the renormalised kappa posterior of candidate j).  g_j is floored
    at g_floor (config.FWD_G_FLOOR) so that lam > 0 also with b = 0 (same convention as GaussianPlume).
    The robust mixture of plan 4.3 is applied to the candidate likelihoods in the same form as in the RB-PF,
        logL_j <- logaddexp(log(1 - eps) + logL_j, log(eps) + logmeanexp_j logL_j),
    which keeps every candidate posterior strictly positive (eps > 0): one wildly inconsistent count cannot
    delete a candidate for good.  Then log p_j += logL_j and the K posteriors are renormalised.

    The filter is the exact Bayesian solution of the K-hypothesis problem (no particles, no resampling), so it is
    deterministic given the measurements; K x G = 13 x 31 floats of state.  ``update`` returns the predictive
    log-likelihood log sum_j p_j L_j (after the mixture) like RBPF.update.

Summaries: ``posterior`` (K,), ``map_candidate`` (candidate id) / ``map_index``, ``entropy`` [nats] (log K at
reset), ``posterior_kappa`` (kgrid, weights) of one candidate or the posterior-weighted mixture and
``posterior_kappa_quantiles`` (same cell-edge CDF convention as RBPF).  All defaults come from srcloc_env.config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.special import gammaln

from srcloc_env import config
from srcloc_env.field.concentration_field import FieldBackend
from srcloc_env.pf.forward_model import LibraryModel
from srcloc_env.pf.particle_filter import logmeanexp, logsumexp

UnitResponseFn = Callable[[np.ndarray], np.ndarray]


class CandidateFilter:
    """Exact discrete Bayes filter over K candidate sources with the kappa log-grid marginalisation of the
    RB-PF (plan 4.2 library model / 4.3 kappa marginalisation; R3, R5).

    Parameters (defaults from config, provenance there)
        unit_response_fn  callable drone_xyz (3,) -> (K,) responses of the candidates (see the module docstring)
        candidates        candidate ids in the order of the response vector (config.ALL_SOURCES)
        kappa_ref, grid_decades, n_grid   the shared kappa log grid (config.KAPPA_REF, KAPPA_GRID_DECADES, KAPPA_G)
        background b [cps], T [s]         config.SENSOR_BACKGROUND_CPS, SENSOR_T
        eps_mix           robust mixture weight (config.PF_EPS_MIX); 0 disables it
        g_floor           lower clip of the responses (config.FWD_G_FLOOR)
        log_prior         optional (K,) log prior over the candidates (default uniform)
    """

    def __init__(self, unit_response_fn: UnitResponseFn, candidates: Sequence[int] = config.ALL_SOURCES,
                 kappa_ref: float = config.KAPPA_REF, grid_decades: float = config.KAPPA_GRID_DECADES,
                 n_grid: int = config.KAPPA_G, background: float = config.SENSOR_BACKGROUND_CPS,
                 T: float = config.SENSOR_T, eps_mix: float = config.PF_EPS_MIX,
                 g_floor: float = config.FWD_G_FLOOR, log_prior: np.ndarray | None = None) -> None:
        self.candidates = tuple(int(c) for c in candidates)
        self.K = len(self.candidates)
        if self.K < 1 or len(set(self.candidates)) != self.K:
            raise ValueError("candidates must be a non-empty sequence of distinct ids")
        if n_grid < 2 or kappa_ref <= 0.0 or grid_decades <= 0.0:
            raise ValueError("need n_grid >= 2, kappa_ref > 0, grid_decades > 0")
        if background < 0.0 or T <= 0.0 or not 0.0 <= eps_mix < 1.0 or g_floor <= 0.0:
            raise ValueError("need background >= 0, T > 0, 0 <= eps_mix < 1, g_floor > 0")
        if not callable(unit_response_fn):
            raise TypeError("unit_response_fn must be callable: drone_xyz (3,) -> (K,)")
        self.unit_response_fn = unit_response_fn
        self.kappa_ref = float(kappa_ref)
        self.grid_decades = float(grid_decades)
        self.G = int(n_grid)
        self.background = float(background)
        self.T = float(T)
        self.eps_mix = float(eps_mix)
        self.g_floor = float(g_floor)
        self.kgrid: np.ndarray = self.kappa_ref * 10.0 ** np.linspace(-self.grid_decades, self.grid_decades, self.G)
        self._kgrid_T = self.kgrid * self.T
        self._bT = self.background * self.T
        if log_prior is None:
            self._log_prior = np.full(self.K, -np.log(self.K))
        else:
            lp = np.asarray(log_prior, dtype=np.float64).reshape(-1)
            if lp.shape != (self.K,):
                raise ValueError(f"log_prior must have shape ({self.K},), got {lp.shape}")
            self._log_prior = lp - logsumexp(lp)
        self._index = {c: i for i, c in enumerate(self.candidates)}
        self.logp: np.ndarray = np.empty(self.K)
        self.logv: np.ndarray = np.empty((self.K, self.G))
        self.last_loglik: np.ndarray = np.empty(self.K)
        self.last_response: np.ndarray = np.empty(self.K)
        self.n_updates = 0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Prior over the candidates (uniform unless log_prior was given), log-uniform kappa prior per candidate."""
        self.logp = self._log_prior.copy()
        self.logv = np.full((self.K, self.G), -np.log(self.G))
        self.last_loglik = np.zeros(self.K)
        self.last_response = np.zeros(self.K)
        self.n_updates = 0

    def index_of(self, candidate: int) -> int:
        """Position of a candidate id in the posterior vector."""
        try:
            return self._index[int(candidate)]
        except KeyError:
            raise KeyError(f"unknown candidate {candidate}; candidates are {self.candidates}") from None

    # ------------------------------------------------------------------ measurement update
    def update(self, y: int, drone_xyz: np.ndarray) -> float:
        """Bayes update with one count y at drone_xyz (3,); returns the predictive log-likelihood log sum_j p_j L_j.

        g = max(unit_response_fn(drone_xyz), g_floor) (K,); lam = (kgrid g + b) T (K, G) - the formula of
        sensor/detector.expected_counts_from_kappa; logv += y log lam - lam - gammaln(y + 1); logL_j =
        logsumexp_g; logv renormalised; robust mixture (eps_mix); logp += logL; renormalised (plan 4.3).
        """
        y = int(y)
        if y < 0:
            raise ValueError("counts must be non-negative")
        drone = np.asarray(drone_xyz, dtype=np.float64).reshape(-1)
        if drone.size != 3:
            raise ValueError(f"drone_xyz must have 3 components, got {drone.size}")
        g = np.asarray(self.unit_response_fn(drone), dtype=np.float64).reshape(-1)
        if g.shape != (self.K,):
            raise ValueError(f"unit_response_fn must return ({self.K},), got {g.shape}")
        g = np.maximum(g, self.g_floor)
        self.last_response = g
        lam = g[:, None] * self._kgrid_T[None, :] + self._bT                    # (K, G)
        self.logv += y * np.log(lam) - lam - gammaln(y + 1.0)
        logL = logsumexp(self.logv, axis=1)                                     # (K,) marginal over kappa
        self.logv -= logL[:, None]
        self.last_loglik = np.asarray(logL, dtype=np.float64).copy()
        if self.eps_mix > 0.0:
            logL = np.logaddexp(np.log1p(-self.eps_mix) + logL, np.log(self.eps_mix) + logmeanexp(logL))
        marginal = float(logsumexp(self.logp + logL))
        self.logp = self.logp + logL - marginal
        self.n_updates += 1
        return marginal

    # ------------------------------------------------------------------ summaries
    def posterior(self) -> np.ndarray:
        """(K,) posterior probabilities of the candidates (sums to 1)."""
        return np.exp(self.logp)

    def map_index(self) -> int:
        """Index of the most probable candidate."""
        return int(np.argmax(self.logp))

    def map_candidate(self) -> int:
        """Id of the most probable candidate."""
        return self.candidates[self.map_index()]

    def posterior_of(self, candidate: int) -> float:
        """Posterior probability of one candidate id."""
        return float(np.exp(self.logp[self.index_of(candidate)]))

    def entropy(self) -> float:
        """Shannon entropy [nats] of the candidate posterior (log K at reset with the uniform prior)."""
        p = self.posterior()
        p = p[p > 0.0]
        return float(-np.sum(p * np.log(p)))

    def posterior_kappa(self, candidate: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(kgrid (G,), weights (G,)): kappa posterior of one candidate id, or the posterior-weighted mixture."""
        if candidate is not None:
            return self.kgrid, np.exp(self.logv[self.index_of(candidate)])
        mix = logsumexp(self.logv + self.logp[:, None], axis=0)
        return self.kgrid, np.exp(mix - logsumexp(mix))

    def posterior_kappa_quantiles(self, q: tuple[float, ...] = (0.05, 0.5, 0.95),
                                  candidate: int | None = None) -> np.ndarray:
        """Quantiles [kappa units] of the kappa posterior with the cell-edge CDF of RBPF.posterior_kappa_quantiles
        (each grid cell's mass spread uniformly in log10 kappa between the mid-points to its neighbours)."""
        kg, w = self.posterior_kappa(candidate)
        lk = np.log10(kg)
        half = 0.5 * np.diff(lk)
        edges = np.concatenate(([lk[0] - half[0]], lk[:-1] + half, [lk[-1] + half[-1]]))
        cdf = np.concatenate(([0.0], np.cumsum(w)))
        cdf = cdf / cdf[-1]
        return 10.0 ** np.interp(np.asarray(q, dtype=np.float64), cdf, edges)


# ---------------------------------------------------------------------------------------- response builders
def library_unit_response_fn(backend: FieldBackend, frame_index: int = config.LIB_FRAME_INDEX,
                             z: float = config.DRONE_Z, scale: float = 1.0,
                             candidates: Sequence[int] = config.ALL_SOURCES,
                             flip_y: bool = False) -> tuple[UnitResponseFn, LibraryModel]:
    """(fn, model): fn(drone_xyz (3,)) -> (K,) slab densities of the candidates from LibraryModel (plan 4.2 (a)).

    The responses are the cached slab densities for the ACTUAL release rate (kappa absorbs k * scale only,
    plan 4.2 caveat).  ``flip_y`` is forwarded for the y-reflection augmentation."""
    model = LibraryModel(backend, frame_index, z=z, scale=scale, candidates=tuple(int(c) for c in candidates))

    def fn(drone_xyz: np.ndarray) -> np.ndarray:
        return model.unit_response(np.asarray(drone_xyz, dtype=np.float64).reshape(1, 3), flip_y)[:, 0]

    return fn, model


def bilinear_fields_at(fields: np.ndarray, x0: float, y0: float, res: float, xy: np.ndarray) -> np.ndarray:
    """(K, n) bilinear samples of (K, ny, nx) cell-centred fields at xy (n, 2); cell (iy, ix) is centred at
    x0 + res (ix + 0.5), y0 + res (iy + 0.5), the edge value is held in the outer half cell and points outside
    the grid give 0 (the convention of field/concentration_field.LdmSlabBackend.density)."""
    f = np.asarray(fields, dtype=np.float64)
    if f.ndim != 2 and f.ndim != 3:
        raise ValueError(f"fields must be (K, ny, nx) or (ny, nx), got {f.shape}")
    if f.ndim == 2:
        f = f[None]
    ny, nx = f.shape[1:]
    p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    fx = (p[:, 0] - x0) / res - 0.5
    fy = (p[:, 1] - y0) / res - 0.5
    inside = (fx >= -0.5) & (fx <= nx - 0.5) & (fy >= -0.5) & (fy <= ny - 0.5)
    fx = np.clip(fx, 0.0, nx - 1.0)
    fy = np.clip(fy, 0.0, ny - 1.0)
    ix0 = np.minimum(np.floor(fx).astype(np.intp), nx - 2)
    iy0 = np.minimum(np.floor(fy).astype(np.intp), ny - 2)
    tx, ty = fx - ix0, fy - iy0
    out = ((1.0 - ty) * ((1.0 - tx) * f[:, iy0, ix0] + tx * f[:, iy0, ix0 + 1])
           + ty * ((1.0 - tx) * f[:, iy0 + 1, ix0] + tx * f[:, iy0 + 1, ix0 + 1]))
    out[:, ~inside] = 0.0
    return out


def adjoint_unit_response_fn(path: Path = config.LIB_ADJOINT_FIELDS_NPZ,
                             candidates: Sequence[int] = config.ALL_SOURCES) -> tuple[UnitResponseFn, dict] | None:
    """(fn, meta) from the calibrated adjoint unit-source fields (scripts/calibrate_adjoint.py, plan 4.2b /
    T1-3b), or None when ``path`` does not exist (plan 4.2 (b), optional).

    The npz holds fields (13, ny, nx) [particles/m^3 per particle/s] in the slab grid (grid_x0, grid_y0,
    grid_res, grid_nx, grid_ny) with ``sources``; the candidate rows are picked by id.  fn(drone_xyz) samples
    them bilinearly (``bilinear_fields_at``).  meta records K, lam, wind_layer, z, h_layer, index."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as npz:
        fields = np.asarray(npz["fields"], dtype=np.float64)
        sources = [int(s) for s in np.asarray(npz["sources"]).ravel()]
        x0, y0, res = float(npz["grid_x0"]), float(npz["grid_y0"]), float(npz["grid_res"])
        meta = {k: (npz[k].item() if npz[k].ndim == 0 else npz[k].tolist())
                for k in ("K", "lam", "wind_layer", "z", "h_layer", "index", "grid_nx", "grid_ny") if k in npz.files}
    if fields.ndim != 3 or fields.shape[0] != len(sources):
        raise ValueError(f"{path}: fields shape {fields.shape} inconsistent with sources {len(sources)}")
    rows = [sources.index(int(c)) for c in candidates]
    sel = np.ascontiguousarray(fields[rows])
    meta.update({"path": str(path), "candidates": [int(c) for c in candidates], "grid_x0": x0, "grid_y0": y0,
                 "grid_res": res})

    def fn(drone_xyz: np.ndarray) -> np.ndarray:
        d = np.asarray(drone_xyz, dtype=np.float64).reshape(-1)
        return bilinear_fields_at(sel, x0, y0, res, d[:2])[:, 0]

    return fn, meta