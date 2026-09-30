"""Unit tests for pf/library_filter.CandidateFilter (plan 4.2 library model, S1 library upper bound / T1-4; D5-4)
on a synthetic 3-candidate problem only (no raw data): distinct response vectors along a short path, counts
from Detector with one candidate as truth.  Checks: the posterior concentrates on the true candidate; the
kappa marginalisation makes the selection independent of the truth scale (config.LIB_TEST_KAPPA_DECADES) and
the kappa posterior tracks kappa_true; an exact scale identity (responses x 10^step, kappa_ref / 10^step
-> identical posteriors); the eps mixture keeps all posteriors > 0; the update formula equals the
RBPF grid update for one particle per candidate with and without the robust mixture (same likelihood code
path, R3 / R5); the bilinear helper and the missing-file path of adjoint_unit_response_fn."""
from pathlib import Path

import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.pf.library_filter import (CandidateFilter, adjoint_unit_response_fn, bilinear_fields_at)
from srcloc_env.pf.particle_filter import RBPF
from srcloc_env.sensor.detector import Detector

# test-local sizes (not physics constants)
N_STEPS_TEST = 40
COUNTS_MAX_TEST = 200.0          # kappa_true scaled so the largest expected signal count is ~200 (background 20)
CANDIDATES = (1, 2, 3)


def make_problem() -> tuple[np.ndarray, np.ndarray]:
    """(path (n, 3), G (3, n)): three Gaussian bumps at different positions along a straight path."""
    x = np.linspace(0.0, 200.0, N_STEPS_TEST)
    path = np.column_stack([x, np.zeros_like(x), np.full_like(x, config.DRONE_Z)])
    centres = (40.0, 100.0, 160.0)
    widths = (25.0, 30.0, 20.0)
    G = np.stack([np.exp(-0.5 * ((x - c) / w) ** 2) for c, w in zip(centres, widths)])   # (3, n) unit responses
    return path, G


class _TableResponse:
    """unit_response_fn: looks the path point up in the response table (K, n)."""

    def __init__(self, path: np.ndarray, G: np.ndarray, factor: float = 1.0) -> None:
        self.path, self.G, self.factor = path, G, factor

    def __call__(self, drone_xyz: np.ndarray) -> np.ndarray:
        i = int(np.argmin(np.abs(self.path[:, 0] - float(drone_xyz[0]))))
        return self.factor * self.G[:, i]


def run(cf: CandidateFilter, path: np.ndarray, counts: np.ndarray) -> np.ndarray:
    for k in range(path.shape[0]):
        cf.update(int(counts[k]), path[k])
    return cf.posterior()


def counts_for(G: np.ndarray, true_idx: int, kappa_true: float, seed: int, background: float = config.SENSOR_BACKGROUND_CPS) -> np.ndarray:
    det = Detector(background=background)
    rng = np.random.default_rng(seed)
    return det.measure(G[true_idx] * kappa_true / det.k0, 1.0, rng)


def test_posterior_concentrates_on_true_candidate():
    path, G = make_problem()
    kappa_true = COUNTS_MAX_TEST / config.SENSOR_T       # expected signal counts = kappa g T, g <= 1
    for true_idx in range(3):
        cf = CandidateFilter(_TableResponse(path, G), candidates=CANDIDATES, kappa_ref=kappa_true)
        assert cf.entropy() == pytest.approx(np.log(3))
        post = run(cf, path, counts_for(G, true_idx, kappa_true, seed=true_idx))
        assert post.sum() == pytest.approx(1.0)
        assert cf.map_candidate() == CANDIDATES[true_idx]
        assert post[true_idx] > 0.9
        assert cf.entropy() < 0.5
        assert cf.n_updates == N_STEPS_TEST
    cf.reset()
    assert np.allclose(cf.posterior(), 1.0 / 3.0) and cf.n_updates == 0


def test_selection_invariant_to_truth_scale_and_kappa_tracks():
    """kappa marginalisation (plan 4.3, R5): the true candidate wins for kappa_true over 3 decades around
    kappa_ref and the posterior kappa median lands within 0.3 decade of kappa_true."""
    path, G = make_problem()
    kappa_ref = COUNTS_MAX_TEST / config.SENSOR_T
    for d in config.LIB_TEST_KAPPA_DECADES:
        kappa_true = kappa_ref * 10.0 ** d
        cf = CandidateFilter(_TableResponse(path, G), candidates=CANDIDATES, kappa_ref=kappa_ref)
        post = run(cf, path, counts_for(G, 1, kappa_true, seed=11))
        assert cf.map_candidate() == 2, d
        assert post[1] > 0.9, d
        q = cf.posterior_kappa_quantiles((0.05, 0.5, 0.95), candidate=2)
        assert abs(np.log10(q[1] / kappa_true)) < 0.3, (d, q)
        assert q[2] / q[0] < 10.0 ** 0.6        # the 90 % interval is at most three grid cells wide (0.2 decade each)


def test_exact_grid_shift_identity():
    """b = 0, eps = 0: scaling every response by one grid step s = 10^(2 decades / (G - 1)) and kappa_ref by
    1/s leaves every product kgrid[g] * response unchanged node for node (kappa_ref/s * 10^l * s * g_j), so
    the two filters must give the same posterior - the kappa grid is a pure scale and the filter must not
    depend on the absolute level of the responses in any other way.  The only deviation is the g_floor clip
    (config.FWD_G_FLOOR), which acts on the unscaled responses of candidates far from the truth (< 1e-9 at
    the path ends); their posterior mass is ~0 in both filters, hence the atol."""
    path, G = make_problem()
    kappa_ref = COUNTS_MAX_TEST / config.SENSOR_T
    step = 10.0 ** (2.0 * config.KAPPA_GRID_DECADES / (config.KAPPA_G - 1))
    y = counts_for(G, 0, kappa_ref, seed=5, background=0.0)
    a = CandidateFilter(_TableResponse(path, G), candidates=CANDIDATES, kappa_ref=kappa_ref, background=0.0, eps_mix=0.0)
    b = CandidateFilter(_TableResponse(path, G, factor=step), candidates=CANDIDATES, kappa_ref=kappa_ref / step,
                        background=0.0, eps_mix=0.0)
    assert np.allclose(a.kgrid, b.kgrid * step, rtol=1e-12)          # the lam grids coincide exactly
    pa, pb = run(a, path, y), run(b, path, y)
    assert np.allclose(pa, pb, rtol=1e-6, atol=1e-9)
    assert pa[0] > 0.99
    # the kappa posterior of the winner is the same posterior shifted by exactly one node
    _, wa = a.posterior_kappa(CANDIDATES[0])
    _, wb = b.posterior_kappa(CANDIDATES[0])
    assert np.allclose(wa, wb, rtol=1e-6, atol=1e-9)


def test_eps_mixture_keeps_all_posteriors_positive():
    path, G = make_problem()
    kappa_true = COUNTS_MAX_TEST / config.SENSOR_T
    y = counts_for(G, 2, kappa_true, seed=3)
    exact = CandidateFilter(_TableResponse(path, G), candidates=CANDIDATES, kappa_ref=kappa_true, eps_mix=0.0)
    robust = CandidateFilter(_TableResponse(path, G), candidates=CANDIDATES, kappa_ref=kappa_true, eps_mix=config.PF_EPS_MIX)
    pe, pr = run(exact, path, y), run(robust, path, y)
    assert np.all(pr > 0.0) and np.all(np.isfinite(robust.logp))
    assert pr.min() > pe.min()                      # the mixture lifts the losers
    assert robust.map_candidate() == 3 and pr[2] > 0.9
    # with eps > 0 every candidate keeps at least eps/K of ONE update's average likelihood ratio: never exactly 0
    assert np.all(robust.posterior() > 1e-300)


@pytest.mark.parametrize("eps_mix", [0.0, config.PF_EPS_MIX])
def test_update_matches_rbpf_grid_path(eps_mix: float):
    """One RBPF particle placed on each candidate with a table forward model must give the same per-candidate
    marginal likelihoods (last_loglik), predictive log-likelihood, weights and kappa posteriors as
    CandidateFilter (shared formula), with and without the robust mixture (plan 4.3)."""
    path, G = make_problem()
    kappa_ref = COUNTS_MAX_TEST / config.SENSOR_T
    y = counts_for(G, 1, kappa_ref, seed=9)

    class _Fwd:
        def __init__(self, table: _TableResponse) -> None:
            self.table = table

        def unit_response(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> np.ndarray:
            g = self.table(np.asarray(drone_xyz).reshape(-1))
            return np.maximum(g, config.FWD_G_FLOOR)[:, None]

    table = _TableResponse(path, G)
    cf = CandidateFilter(table, candidates=CANDIDATES, kappa_ref=kappa_ref, eps_mix=eps_mix)
    pf = RBPF(_Fwd(table), n_particles=3, kappa_ref=kappa_ref, eps_mix=eps_mix, jitter_m=0.0, resample_frac=0.0, likelihood="poisson",
              rng=np.random.default_rng(0))
    for k in range(path.shape[0]):
        m1 = cf.update(int(y[k]), path[k])
        m2 = pf.update(int(y[k]), path[k])
        assert cf.last_loglik == pytest.approx(pf.last_loglik, rel=1e-9, abs=1e-9)
        assert m1 == pytest.approx(m2, rel=1e-9, abs=1e-9)
    assert pf.n_resamples == 0
    assert np.allclose(cf.posterior(), pf.weights(), atol=1e-9)
    for j in range(3):
        assert np.allclose(cf.posterior_kappa(CANDIDATES[j])[1], pf.posterior_kappa(j)[1], atol=1e-9)
    assert np.allclose(cf.posterior_kappa()[1], pf.posterior_kappa()[1], atol=1e-9)
    assert np.allclose(cf.posterior_kappa_quantiles(), pf.posterior_kappa_quantiles(), rtol=1e-9)


def test_input_validation():
    path, G = make_problem()
    fn = _TableResponse(path, G)
    with pytest.raises(ValueError):
        CandidateFilter(fn, candidates=(1, 1, 2))
    with pytest.raises(ValueError):
        CandidateFilter(fn, candidates=CANDIDATES, n_grid=1)
    with pytest.raises(ValueError):
        CandidateFilter(fn, candidates=CANDIDATES, eps_mix=1.0)
    with pytest.raises(TypeError):
        CandidateFilter(3.0, candidates=CANDIDATES)
    cf = CandidateFilter(fn, candidates=CANDIDATES)
    with pytest.raises(ValueError):
        cf.update(-1, path[0])
    with pytest.raises(ValueError):
        cf.update(1, np.zeros(2))
    with pytest.raises(ValueError):
        CandidateFilter(lambda p: np.ones(2), candidates=CANDIDATES).update(1, path[0])
    with pytest.raises(KeyError):
        cf.index_of(99)
    with pytest.raises(ValueError):
        CandidateFilter(fn, candidates=CANDIDATES, log_prior=np.zeros(2))


def test_bilinear_fields_at_and_missing_npz(tmp_path: Path):
    x0, y0, res = 10.0, -5.0, 2.0
    ny, nx = 4, 5
    xx, yy = np.meshgrid(x0 + res * (np.arange(nx) + 0.5), y0 + res * (np.arange(ny) + 0.5))
    fields = np.stack([xx + 2.0 * yy, 3.0 * np.ones_like(xx)])              # linear field -> bilinear is exact
    pts = np.array([[x0 + 3.3, y0 + 2.7], [x0 + 1.0, y0 + 1.0], [x0 + 100.0, y0]])   # interior, centre, outside
    out = bilinear_fields_at(fields, x0, y0, res, pts)
    assert out.shape == (2, 3)
    assert out[0, 0] == pytest.approx(pts[0, 0] + 2.0 * pts[0, 1])
    assert out[1, 0] == pytest.approx(3.0) and out[1, 1] == pytest.approx(3.0)
    assert out[0, 2] == 0.0 and out[1, 2] == 0.0
    assert adjoint_unit_response_fn(tmp_path / "missing.npz") is None
    np.savez(tmp_path / "f.npz", fields=fields.astype(np.float32), sources=np.array([7, 8]), grid_x0=x0,
             grid_y0=y0, grid_res=res, grid_nx=nx, grid_ny=ny, K=4.0, lam=0.02, wind_layer="single_15m", z=15.0,
             h_layer=10.0, index=599)
    fn, meta = adjoint_unit_response_fn(tmp_path / "f.npz", candidates=(8, 7))
    g = fn(np.array([pts[0, 0], pts[0, 1], config.DRONE_Z]))
    assert g.shape == (2,) and g[0] == pytest.approx(3.0) and g[1] == pytest.approx(pts[0, 0] + 2.0 * pts[0, 1], rel=1e-6)
    assert meta["K"] == 4.0 and meta["candidates"] == [8, 7]