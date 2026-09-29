"""Unit tests for pf/particle_filter.py (plan 4.3, T1-1 and the synthetic convergence check; no raw data).

T1-1: the NB (Gamma-Poisson, R6) marginal log-likelihood of 5 counts for ONE particle equals the numerical
integral over kappa of prod Poisson(y_i; kappa g_i T) Gamma(kappa; alpha0 = 1, rate 1/kappa_ref), trapezoid in
ln(kappa) on a 2,000-point log-spaced grid (|dlogL| < 1e-6 is reached: ~1e-13); the grid path with b = 0 and a
fine grid (n_grid = 2000, grid_decades = 6, Gamma prior mass on the grid) matches the NB value within 1e-3.

Convergence: truth from GaussianPlume, source (700, 0), 120 Poisson counts (b = 20) from a 2-drone lawnmower at
z = 15 m, RBPF grid mode N = 2000 -> MAP error < 30 m, kappa posterior median within a factor 3 of the truth,
entropy below the prior.  kappa_true = 3e4 (inside the T1-2 sampling range kappa_ref * 10^[-1, 1]); the plan's
example value 3e3 gives a peak signal of only 1.4 cps over the 20 cps background at 15 m altitude with the
default plume (sigma_z = 0.6 sigma_y), which no filter can localise in 120 measurements.
"""
import time

import numpy as np
import pytest
from scipy.special import gammaln, logsumexp

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.pf.forward_model import ForwardParams, GaussianPlume
from srcloc_env.pf.particle_filter import RBPF, logmeanexp
from srcloc_env.sensor.detector import expected_counts_from_kappa

# ---- T1-1 fixtures (test-local numbers, not physics constants) --------------------------------
G_SEQ = np.array([2e-4, 5e-4, 1e-4, 3e-4, 8e-4])     # unit responses of the 5 measurements
Y_SEQ = np.array([1, 3, 0, 2, 4])                    # counts (kappa_ref * g * T ~ 0.7 .. 5.6 expected)
INTEGRAL_POINTS = 2000
INTEGRAL_DECADES = (-6.0, 3.0)                       # kappa range relative to kappa_ref (Gamma tail e^{-kappa/kappa_ref})
TOL_NB_VS_INTEGRAL = 1e-6
TOL_GRID_VS_NB = 1e-3
# ---- convergence scenario ---------------------------------------------------------------------
SRC_TRUE = np.array([700.0, 0.0])
KAPPA_TRUE = 3e4
ROWS_A, ROWS_B, Y_HALF, N_PER_ROW = (680.0, 720.0, 760.0), (800.0, 850.0, 900.0), 50.0, 20
MAP_TOL_M = 30.0
MAP_TOP_TOL_M = 100.0                                # loose bound for map_estimate(method='top')
KAPPA_FACTOR = 3.0
RUNTIME_S = 20.0


class DroneXAsG:
    """Stub forward model: g = drone x coordinate for every particle (exact control of g in T1-1)."""

    def unit_response(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(source_xy)
        d = np.atleast_2d(drone_xyz)
        return np.broadcast_to(d[None, :, 0], (s.shape[0], d.shape[0])).copy()


def _nb_sequence(pf: RBPF) -> float:
    return float(sum(pf.update(int(y), np.array([g, 0.0, config.DRONE_Z])) for y, g in zip(Y_SEQ, G_SEQ)))


def _exact_log_marginal(kappa_ref: float, n_points: int = INTEGRAL_POINTS) -> float:
    """Trapezoid in ln(kappa) of prod Poisson(y_i; kappa g_i T) * Gamma(kappa; alpha0, beta0) on a log grid."""
    a0, b0, T = config.PF_NB_ALPHA0, 1.0 / kappa_ref, config.SENSOR_T
    lk = np.linspace(np.log10(kappa_ref) + INTEGRAL_DECADES[0], np.log10(kappa_ref) + INTEGRAL_DECADES[1], n_points)
    k = 10.0 ** lk
    lam = k[None, :] * G_SEQ[:, None] * T
    li = np.sum(Y_SEQ[:, None] * np.log(lam) - lam - gammaln(Y_SEQ + 1)[:, None], axis=0)
    li += a0 * np.log(b0) - gammaln(a0) + (a0 - 1.0) * np.log(k) - b0 * k     # Gamma pdf
    li += np.log(k)                                                             # d kappa = kappa d ln(kappa)
    dl = np.log(10.0) * (lk[1] - lk[0])
    wt = np.full(n_points, dl)
    wt[0] = wt[-1] = dl / 2.0
    return float(logsumexp(li + np.log(wt)))


def test_t1_1_nb_marginal_matches_numerical_integral():
    pf = RBPF(DroneXAsG(), n_particles=1, background=0.0, mode="nb", eps_mix=0.0, rng=np.random.default_rng(0))
    assert pf.a[0] == config.PF_NB_ALPHA0 and np.isclose(pf.beta[0], 1.0 / config.KAPPA_REF)
    nb = _nb_sequence(pf)
    assert abs(nb - _exact_log_marginal(config.KAPPA_REF)) < TOL_NB_VS_INTEGRAL
    assert pf.a[0] == config.PF_NB_ALPHA0 + Y_SEQ.sum() and np.isclose(pf.beta[0], 1.0 / config.KAPPA_REF + G_SEQ.sum() * config.SENSOR_T)


def test_t1_1_fine_grid_matches_nb():
    nb = _nb_sequence(RBPF(DroneXAsG(), n_particles=1, background=0.0, mode="nb", eps_mix=0.0,
                           rng=np.random.default_rng(0)))
    grid = RBPF(DroneXAsG(), n_particles=1, background=0.0, mode="grid", n_grid=2000, grid_decades=6.0,
                kappa_prior="gamma", eps_mix=0.0, rng=np.random.default_rng(0))
    assert abs(_nb_sequence(grid) - nb) < TOL_GRID_VS_NB
    # the per-particle kappa posterior on the grid stays normalised
    assert np.isclose(logsumexp(grid.logv, axis=1), 0.0, atol=1e-9).all()


def test_grid_and_prior_layout():
    pf = RBPF(GaussianPlume(ForwardParams()), n_particles=50, rng=np.random.default_rng(1))
    assert pf.kgrid.shape == (config.KAPPA_G,)
    assert np.isclose(pf.kgrid[0], config.KAPPA_REF * 10.0 ** -config.KAPPA_GRID_DECADES)
    assert np.isclose(pf.kgrid[-1], config.KAPPA_REF * 10.0 ** config.KAPPA_GRID_DECADES)
    assert np.allclose(np.diff(np.log10(pf.kgrid)), 2.0 * config.KAPPA_GRID_DECADES / (config.KAPPA_G - 1))
    assert np.allclose(pf.logv, -np.log(config.KAPPA_G))                       # log-uniform prior
    assert np.isclose(logsumexp(pf.logw), 0.0)
    kg, w = pf.posterior_kappa()
    assert np.allclose(w, 1.0 / config.KAPPA_G) and np.isclose(w.sum(), 1.0)
    x, y = pf.xy[:, 0], pf.xy[:, 1]
    assert (x >= config.PF_PRIOR_X[0]).all() and (x <= config.PF_PRIOR_X[1]).all()
    assert (y >= config.PF_PRIOR_Y[0]).all() and (y <= config.PF_PRIOR_Y[1]).all()
    # one update reproduces the detector formula lam = (kgrid g + b) T in the grid log-likelihood
    g0 = 3e-4
    pf1 = RBPF(DroneXAsG(), n_particles=1, rng=np.random.default_rng(0))
    ll = pf1.update(25, np.array([g0, 0.0, config.DRONE_Z]))
    lam = expected_counts_from_kappa(pf1.kgrid, g0)
    ref = 25 * np.log(lam) - lam - gammaln(26.0) - np.log(config.KAPPA_G)
    assert np.isclose(ll, logsumexp(ref)) and np.allclose(pf1.logv[0], ref - logsumexp(ref))
    # float32 grid block gives the same weights to ~1e-4
    pf32 = RBPF(GaussianPlume(ForwardParams()), n_particles=50, rng=np.random.default_rng(1), dtype=np.float32)
    assert pf32.logv.dtype == np.float32
    pf64 = RBPF(GaussianPlume(ForwardParams()), n_particles=50, rng=np.random.default_rng(1))
    d = np.array([800.0, 10.0, config.DRONE_Z])
    for y in (30, 18, 45):
        l32, l64 = pf32.update(y, d), pf64.update(y, d)
        assert abs(l32 - l64) < 1e-3
    assert np.allclose(pf32.logw, pf64.logw, atol=1e-3) and pf32.logw.dtype == np.float64
    with pytest.raises(ValueError):
        RBPF(DroneXAsG(), n_particles=5, mode="nb")                            # background > 0 with nb
    with pytest.raises(ValueError):
        RBPF(DroneXAsG(), n_particles=5, mode="other")


def test_weights_normalised_after_updates_and_resampling():
    plume = GaussianPlume(ForwardParams())
    rng = np.random.default_rng(2)
    pf = RBPF(plume, n_particles=300, rng=rng)
    drones = np.column_stack([rng.uniform(700, 900, 8), rng.uniform(-50, 50, 8), np.full(8, config.DRONE_Z)])
    for d in drones:
        ll = pf.update(int(rng.poisson(40.0)), d)
        assert np.isfinite(ll)
        assert np.isclose(logsumexp(pf.logw), 0.0, atol=1e-9)
        assert np.isclose(logsumexp(pf.logv, axis=1), 0.0, atol=1e-9).all()
        assert pf.xy.shape == (300, 2) and pf.logv.shape == (300, config.KAPPA_G)
    pf.resample_and_jitter()
    assert pf.xy.shape == (300, 2) and np.allclose(pf.logw, -np.log(300))
    assert np.isclose(pf.neff(), 300.0)
    _, w = pf.posterior_kappa()
    assert np.isclose(w.sum(), 1.0)


def test_jitter_keeps_particles_in_box_and_off_buildings():
    # synthetic 2 m raster: a tall block in the middle of a small prior box; prior box = raster extent
    n, res = 40, 2.0
    occ = np.zeros((n, n), dtype=bool)
    hmap = np.zeros((n, n), dtype=np.float32)
    occ[15:25, 15:25] = True
    hmap[15:25, 15:25] = 40.0
    box = (0.0, res * (n - 1))
    om = ObstacleMap.from_arrays(0.0, 0.0, res, occ, hmap, domain_x=box, domain_y=box)
    pf = RBPF(DroneXAsG(), n_particles=400, prior_x=box, prior_y=box, obstacles=om, jitter_m=6.0,
              rng=np.random.default_rng(3))
    assert om.is_free(pf.xy, config.DRONE_Z).all()
    for _ in range(5):
        pf.logw = np.log(np.random.default_rng(4).dirichlet(np.ones(400) * 0.05))   # peaked weights -> resample
        pf.resample_and_jitter()
        assert pf.xy.shape == (400, 2)
        assert (pf.xy >= box[0]).all() and (pf.xy <= box[1]).all()
        assert om.is_free(pf.xy, config.DRONE_Z).all()


def test_robust_mixture_and_logmeanexp():
    x = np.array([-1.0, -2.0, -3.0])
    assert np.isclose(logmeanexp(x), np.log(np.mean(np.exp(x))))
    pf = RBPF(DroneXAsG(), n_particles=3, eps_mix=0.5, rng=np.random.default_rng(0))
    pf.xy[:] = 0.0
    pf.update(30, np.array([5e-4, 0.0, config.DRONE_Z]))
    assert np.allclose(pf.logw, -np.log(3))          # identical particles -> identical weights, whatever eps


def _lawnmower(rows: tuple[float, ...], y_half: float, n_per_row: int) -> np.ndarray:
    pts = []
    for i, x in enumerate(rows):
        ys = np.linspace(-y_half, y_half, n_per_row)
        pts += [(x, y) for y in (ys if i % 2 == 0 else ys[::-1])]
    return np.array(pts)


def test_synthetic_convergence_two_drone_lawnmower():
    plume = GaussianPlume(ForwardParams())
    pa, pb = _lawnmower(ROWS_A, Y_HALF, N_PER_ROW), _lawnmower(ROWS_B, Y_HALF, N_PER_ROW)
    path = np.empty((2 * N_PER_ROW * len(ROWS_A), 3))
    path[0::2, :2], path[1::2, :2], path[:, 2] = pa, pb, config.DRONE_Z
    g_true = plume.unit_response(SRC_TRUE, path)[0]
    rng = np.random.default_rng(0)
    counts = rng.poisson((KAPPA_TRUE * g_true + config.SENSOR_BACKGROUND_CPS) * config.SENSOR_T)
    pf = RBPF(plume, n_particles=config.PF_N_PARTICLES, rng=np.random.default_rng(100))
    h_prior = pf.entropy_xy()
    t0 = time.perf_counter()
    for y, p in zip(counts, path):
        pf.update(int(y), p)
    elapsed = time.perf_counter() - t0
    assert elapsed < RUNTIME_S
    err = float(np.hypot(*(pf.map_estimate() - SRC_TRUE)))
    assert err < MAP_TOL_M, err
    # "top" (top 5 % by weight) is the documented weaker alternative: right after a resampling step the weights
    # are equal and the selection is arbitrary, so only a loose bound is checked
    assert float(np.hypot(*(pf.map_estimate(method="top") - SRC_TRUE))) < MAP_TOP_TOL_M
    k_med = pf.posterior_kappa_quantiles((0.5,))[0]
    assert KAPPA_TRUE / KAPPA_FACTOR < k_med < KAPPA_TRUE * KAPPA_FACTOR, k_med
    assert pf.entropy_xy() < h_prior
    lo, hi = pf.kappa_edge_mass()
    assert lo < 0.01 and hi < 0.01
    cov = pf.covariance()
    assert cov.shape == (2, 2) and cov[0, 0] > 0 and cov[1, 1] > 0
    xy, w = pf.positions_and_weights()
    assert xy.shape == (config.PF_N_PARTICLES, 2) and np.isclose(w.sum(), 1.0)


def test_systematic_resampling_counts_parent_copy_and_kappa_quantiles():
    """R16 systematic resampling (review D3): every particle i gets floor(N w_i) or ceil(N w_i) children, the
    children inherit the parent's position (jitter_m = 0), kappa statistics (logv or a/beta) and last_loglik;
    posterior_kappa_quantiles: the median of the log-uniform prior is kappa_ref and a posterior concentrated on
    one grid cell has all its quantiles inside that cell (half-cell edges in log10 kappa)."""
    n = 200
    for mode in ("grid", "nb"):
        pf = RBPF(DroneXAsG(), n_particles=n, jitter_m=0.0, background=0.0, mode=mode, rng=np.random.default_rng(5))
        xy0 = pf.xy.copy()
        tag = np.arange(n, dtype=np.float64)
        if mode == "grid":
            pf.logv[:, 0] = tag
        else:
            pf.a[:] = tag
            pf.beta[:] = 2.0 * tag
        pf.last_loglik = tag.copy()
        w = np.random.default_rng(6).dirichlet(np.full(n, 0.2))
        pf.logw = np.log(w)
        pf.resample_and_jitter()
        parents = np.array([int(np.flatnonzero((xy0 == p).all(axis=1))[0]) for p in pf.xy])
        cnt = np.bincount(parents, minlength=n)
        assert cnt.sum() == n
        assert (cnt >= np.floor(n * w)).all() and (cnt <= np.ceil(n * w)).all()
        assert np.array_equal(pf.xy, xy0[parents])
        if mode == "grid":
            assert np.array_equal(pf.logv[:, 0], tag[parents])
        else:
            assert np.array_equal(pf.a, tag[parents]) and np.array_equal(pf.beta, 2.0 * tag[parents])
        assert np.array_equal(pf.last_loglik, tag[parents])
        assert np.allclose(pf.logw, -np.log(n)) and pf.n_resamples == 1
    # kappa quantiles
    pf = RBPF(DroneXAsG(), n_particles=5, rng=np.random.default_rng(0))
    assert np.isclose(pf.posterior_kappa_quantiles((0.5,))[0], config.KAPPA_REF, rtol=1e-9)
    j = 7
    step = np.log10(pf.kgrid[1] / pf.kgrid[0])
    pf.logv[:] = -1e3                                   # finite: cell masses e^-1000 ~ 0 without -inf columns
    pf.logv[:, j] = 0.0
    qs = pf.posterior_kappa_quantiles((0.05, 0.5, 0.95))
    assert (qs > pf.kgrid[j] * 10.0 ** (-0.5 * step)).all() and (qs < pf.kgrid[j] * 10.0 ** (0.5 * step)).all()
    assert np.isclose(qs[1], pf.kgrid[j], rtol=1e-9)
