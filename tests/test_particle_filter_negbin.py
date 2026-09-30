"""Unit tests for the negative-binomial (Gamma-Poisson) count likelihood of pf/particle_filter.RBPF (D7-1; plan 4.3
강건화, S1 T1-2 / T1-4; R22 Yee & Chan 1997, R23 Hilbe 2011).  Synthetic stubs only, no raw data.

(1) r = 1e8 reproduces the Poisson log-likelihood increments to |dlogL| < 1e-4 for several (y, lam);
(2) the NB pmf sums to 1 over y for r in {0.3, 1, 3} and matches scipy.stats.nbinom(n = r, p = r / (r + lam));
(3) counts sampled from the Gamma-Poisson mixture lam Gamma(r, 1/r) -> Poisson have Var ~ lam + lam^2 / r (10 %);
(4) an RBPF update with 'negbin' keeps the weights and the per-particle kappa posteriors normalised (float64 and
    float32) and gives a flatter likelihood than Poisson for an outlier count: the log-likelihood ratio between two
    hypotheses shrinks;
(5) ValueError for an unknown likelihood, for mode='nb' with 'negbin' and for nb_r <= 0; the explicit 'poisson'
    choice is bit-identical to the default, and validate_kappa_bias.make_filter passes the choice to all four filters.
"""
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import nbinom, poisson

from srcloc_env import config
from srcloc_env.pf.particle_filter import RBPF
from srcloc_env.scripts.validate_kappa_bias import FILTER_NAMES, make_filter

# ---- test-local numbers (not physics constants) ----------------------------------------------------------------
BACKGROUND = 2.0                       # cps: small background so that lam = kappa g T + b T can be set to 3..40
T = 1.0
G0 = 1.0                               # unit response of the constant stub; kappa_ref = (lam - b T) / (G0 T) sets the mean
LAMS = (3.0, 12.0, 40.0)
R_VALUES = (0.3, 1.0, 3.0)
R_POISSON_LIMIT = 1e8
TOL_POISSON_LIMIT = 1e-4
TOL_NB_EXACT = 1e-7                    # vs the two-node grid mixture of scipy nbinom log-pmfs (same lam nodes)
TOL_NB_CENTRE = 1e-3                   # vs nbinom / poisson at the centre lam (the two nodes differ by 4.6e-6 in kappa)
Y_MAX = 4000                           # pmf summed over 0..Y_MAX (r = 0.3, lam = 40: tail mass ~1e-13 beyond)
TOL_PMF_SUM = 1e-6
N_SAMPLES = 200_000
LAM_SAMPLE = 30.0
TOL_VAR = 0.10
TOL_MEAN = 0.05
KAPPA_TWO = 100.0                      # two-hypothesis filter: g = 1e-3 x -> lam = 100 g + 20 = 60 / 150 counts
X_HYP = (400.0, 1300.0)
OUTLIER_Y = 600
LLR_SHRINK = 0.1                       # NB log-likelihood ratio must be below this fraction of the Poisson one
TOL_NORM = 1e-9
TOL_NORM_32 = 1e-4
N_GRID_PARTICLES = 200
N_GRID_UPDATES = 10
LAM_GRID_COUNTS = 60.0


class ConstG:
    """Stub forward model: the same unit response G0 for every hypothesis and drone."""

    def unit_response(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> np.ndarray:
        return np.full((np.atleast_2d(source_xy).shape[0], np.atleast_2d(drone_xyz).shape[0]), G0)


class SourceXAsG:
    """Stub forward model: g = 1e-3 x_s, so hypotheses differ in their expected count."""

    def unit_response(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(source_xy)
        d = np.atleast_2d(drone_xyz)
        return np.broadcast_to(1e-3 * s[:, 0][:, None], (s.shape[0], d.shape[0])).copy()


DRONE = np.array([500.0, 0.0, config.DRONE_Z])


def _fixed_lam_filter(lam: float, likelihood: str, nb_r: float = config.PF_NB_DISPERSION_R) -> RBPF:
    """One particle, kappa fixed (two-node grid of half-width config.T1_2_FIXED_KAPPA_DECADES) so that the count
    mean is lam = kappa_ref G0 T + BACKGROUND T; eps_mix = 0 -> update() returns the log-likelihood of y."""
    return RBPF(ConstG(), n_particles=1, kappa_ref=(lam - BACKGROUND * T) / (G0 * T),
                grid_decades=config.T1_2_FIXED_KAPPA_DECADES, n_grid=2, background=BACKGROUND, T=T, eps_mix=0.0,
                jitter_m=0.0, likelihood=likelihood, nb_r=nb_r, rng=np.random.default_rng(0))


def _node_lams(pf: RBPF) -> np.ndarray:
    return pf.kgrid * G0 * T + BACKGROUND * T


def _logpmf(pf: RBPF, y: int) -> float:
    pf.reset(np.random.default_rng(0))
    return pf.update(y, DRONE)


def _nb_ref(y: int, lams: np.ndarray, r: float) -> float:
    """log of the equal-prior two-node mixture of NB pmfs at the node means (what the degenerate grid computes)."""
    return float(logsumexp(nbinom.logpmf(y, r, r / (r + lams))) - np.log(lams.size))


# ---- (1) Poisson limit --------------------------------------------------------------------------------------------
def test_large_r_reproduces_poisson():
    for lam in LAMS:
        pf_nb = _fixed_lam_filter(lam, "negbin", R_POISSON_LIMIT)
        pf_p = _fixed_lam_filter(lam, "poisson")
        for y in (0, 1, int(lam), int(3 * lam), 100):
            l_nb = _logpmf(pf_nb, y)
            l_p = _logpmf(pf_p, y)
            assert abs(l_nb - l_p) < TOL_POISSON_LIMIT, (lam, y, l_nb, l_p)
            assert abs(l_p - poisson.logpmf(y, lam)) < TOL_NB_CENTRE


# ---- (2) pmf normalisation and scipy agreement ---------------------------------------------------------------------
@pytest.mark.parametrize("r", R_VALUES)
def test_nb_pmf_normalised_and_matches_scipy(r):
    ys = np.arange(Y_MAX + 1)
    for lam in LAMS:
        pf = _fixed_lam_filter(lam, "negbin", r)
        lams = _node_lams(pf)
        assert lams == pytest.approx(lam, rel=1e-5)
        lp = np.array([_logpmf(pf, int(y)) for y in ys])
        assert abs(np.exp(lp).sum() - 1.0) < TOL_PMF_SUM, (r, lam)
        ref = np.array([_nb_ref(int(y), lams, r) for y in ys])
        assert np.max(np.abs(lp - ref)) < TOL_NB_EXACT, (r, lam)
        centre = nbinom.logpmf(ys, r, r / (r + lam))
        assert np.max(np.abs(lp - centre)[: int(10 * lam)]) < TOL_NB_CENTRE, (r, lam)


# ---- (3) Gamma-Poisson variance ------------------------------------------------------------------------------------
@pytest.mark.parametrize("r", R_VALUES)
def test_gamma_poisson_samples_have_nb_variance(r):
    rng = np.random.default_rng(7)
    rates = LAM_SAMPLE * rng.gamma(r, 1.0 / r, N_SAMPLES)       # E = lam, Var(rate) = lam^2 / r
    y = rng.poisson(rates)
    expected_var = LAM_SAMPLE + LAM_SAMPLE ** 2 / r
    assert expected_var == pytest.approx(nbinom.var(r, r / (r + LAM_SAMPLE)))
    assert abs(y.mean() - LAM_SAMPLE) < TOL_MEAN * LAM_SAMPLE
    assert abs(y.var() - expected_var) < TOL_VAR * expected_var


# ---- (4) RBPF update: normalisation and flatter likelihood ---------------------------------------------------------
def _two_hypothesis_llr(likelihood: str) -> float:
    pf = RBPF(SourceXAsG(), n_particles=2, kappa_ref=KAPPA_TWO, grid_decades=config.T1_2_FIXED_KAPPA_DECADES,
              n_grid=2, eps_mix=0.0, jitter_m=0.0, resample_frac=0.0, likelihood=likelihood,
              rng=np.random.default_rng(0))
    pf.xy = np.array([[X_HYP[0], 0.0], [X_HYP[1], 0.0]])
    pf.update(OUTLIER_Y, DRONE)
    assert abs(logsumexp(pf.logw)) < TOL_NORM
    assert np.allclose(logsumexp(pf.logv, axis=1), 0.0, atol=TOL_NORM)
    return float(pf.last_loglik[1] - pf.last_loglik[0])


def test_negbin_update_normalised_and_flatter_than_poisson():
    llr_p = _two_hypothesis_llr("poisson")
    llr_nb = _two_hypothesis_llr("negbin")
    assert llr_p > 0.0 and llr_nb > 0.0                  # both prefer the hypothesis nearer the outlier
    assert llr_nb < LLR_SHRINK * llr_p, (llr_nb, llr_p)
    # full config grid, many particles: weights and per-particle kappa posteriors stay normalised (float64 / float32)
    for dtype, tol in ((np.float64, TOL_NORM), (np.float32, TOL_NORM_32)):
        pf = RBPF(SourceXAsG(), n_particles=N_GRID_PARTICLES, likelihood="negbin", dtype=dtype,
                  rng=np.random.default_rng(3))
        rng = np.random.default_rng(4)
        for _ in range(N_GRID_UPDATES):
            pf.update(int(rng.poisson(LAM_GRID_COUNTS)), DRONE)
            assert abs(logsumexp(pf.logw)) < tol
            assert np.allclose(logsumexp(pf.logv.astype(np.float64), axis=1), 0.0, atol=tol)
            assert np.all(np.isfinite(pf.logv)) and np.all(np.isfinite(pf.logw))
        assert pf.n_updates == N_GRID_UPDATES and pf.logv.dtype == dtype
        kg, w = pf.posterior_kappa()
        assert w.sum() == pytest.approx(1.0, abs=1e-6) and kg.shape == (config.KAPPA_G,)


# ---- (5) argument validation, Poisson path unchanged, script plumbing ------------------------------------------------
def test_invalid_likelihood_and_nb_mode():
    with pytest.raises(ValueError):
        RBPF(ConstG(), n_particles=2, likelihood="gamma")
    with pytest.raises(ValueError):
        RBPF(ConstG(), n_particles=2, mode="nb", background=0.0, likelihood="negbin")
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            RBPF(ConstG(), n_particles=2, likelihood="negbin", nb_r=bad)
    assert RBPF(ConstG(), n_particles=2, mode="nb", background=0.0, likelihood="poisson").likelihood == "poisson"
    assert config.PF_LIKELIHOOD in config.PF_LIKELIHOODS and config.PF_NB_DISPERSION_R > 0.0


def test_explicit_config_likelihood_is_bit_identical_to_default():
    a = RBPF(SourceXAsG(), n_particles=100, rng=np.random.default_rng(11))
    b = RBPF(SourceXAsG(), n_particles=100, rng=np.random.default_rng(11), likelihood=config.PF_LIKELIHOOD)
    rng = np.random.default_rng(12)
    for _ in range(8):
        y = int(rng.poisson(50.0))
        assert a.update(y, DRONE) == b.update(y, DRONE)
    assert np.array_equal(a.logv, b.logv) and np.array_equal(a.logw, b.logw) and np.array_equal(a.xy, b.xy)


def test_make_filter_passes_likelihood_to_every_filter():
    for name in FILTER_NAMES:
        pf = make_filter(name, SourceXAsG(), 1e3, np.random.default_rng(0), n_particles=5, obstacles=None,
                         likelihood="negbin", nb_r=0.5)
        assert pf.likelihood == "negbin" and pf.nb_r == 0.5
        assert make_filter(name, SourceXAsG(), 1e3, np.random.default_rng(0), n_particles=5, obstacles=None).likelihood == config.PF_LIKELIHOOD