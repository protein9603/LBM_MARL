"""Unit tests for sensor/detector.py (plan 4.1; R3 Poisson sensor, R4 Currie threshold). Synthetic inputs only."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.sensor.detector import Detector, expected_counts_from_kappa

B, T = 20.0, 1.0


@pytest.fixture
def det():
    return Detector(k0=config.SENSOR_K0, background=B, T=T)


@pytest.mark.parametrize("lam", [20.0, 100.0, 1000.0])
def test_poisson_sample_mean_within_1pct(lam):
    rng = np.random.default_rng(1234)
    det = Detector(k0=1.0, background=0.0, T=1.0)
    y = det.measure(np.full(config.VALIDATE_DETECTOR_N_POISSON, lam), 1.0, rng)
    assert y.dtype == np.int64
    assert abs(y.mean() - lam) / lam < config.VALIDATE_DETECTOR_MEAN_TOL
    assert abs(y.var() - lam) / lam < 0.05                       # Poisson: variance == mean


def test_zero_density_gives_background_only(det):
    rng = np.random.default_rng(0)
    assert det.expected_rate(0.0) == B
    assert det.expected_counts(np.zeros(4), 2.5).tolist() == [B * T] * 4
    y = det.measure(np.zeros(50_000), 3.0, rng)
    assert abs(y.mean() - B * T) / (B * T) < 0.02


def test_expected_rate_formula_and_scale(det):
    n = np.array([0.0, 1.0, 38.4])
    np.testing.assert_allclose(det.expected_rate(n, 2.0), config.SENSOR_K0 * 2.0 * n + B)
    np.testing.assert_allclose(det.expected_counts(n, 0.5), (config.SENSOR_K0 * 0.5 * n + B) * T)
    # y_max reproduces the plan 4.1 number ~3,050 counts at scale 3, 38.4 particles/m^3
    expected_ymax = (max(config.SENSOR_SCALE_RANGE) * config.SENSOR_K0 * config.SENSOR_REF_DENSITY + config.SENSOR_BACKGROUND_CPS) * config.SENSOR_T
    assert abs(config.SENSOR_Y_MAX - expected_ymax) < 1e-6
    assert abs(det.expected_counts(config.SENSOR_REF_DENSITY, max(config.SENSOR_SCALE_RANGE)) - config.SENSOR_Y_MAX) < 1e-9


def test_normalise_in_unit_interval_and_monotone(det):
    y = np.array([0, 1, 5, 33, 100, 1000, config.SENSOR_Y_MAX / 10, config.SENSOR_Y_MAX, 10 * config.SENSOR_Y_MAX])
    z = det.normalise(y)
    assert z.shape == y.shape
    assert np.all(z >= 0.0) and np.all(z <= 1.0)
    assert z[0] == 0.0
    assert np.all(np.diff(z) >= 0.0)
    assert z[-1] == 1.0 and z[-2] == 1.0                          # clipped above y_max
    assert abs(det.normalise(config.SENSOR_Y_MAX) - 1.0) < 1e-12
    assert det.normalise(-3.0) == 0.0                              # negative input clipped


def test_currie_threshold_value_and_detection(det):
    thr = det.detection_threshold_cps()
    assert abs(thr - (B + 3.0 * np.sqrt(B))) < 1e-12
    assert round(thr, 2) == 33.42
    flags = det.is_detection(np.array([0, 33, 34, 100]))
    assert flags.tolist() == [False, False, True, True]
    # T enters as b + k*sqrt(b*T)/T
    d4 = Detector(k0=1.0, background=B, T=4.0)
    assert abs(d4.detection_threshold_cps() - (B + 3.0 * np.sqrt(B * 4.0) / 4.0)) < 1e-12


def test_counts_vs_cps_units_for_T_not_1():
    """Rate is cps, counts are rate*T, detection compares counts/T (cps) with the cps threshold (plan 4.1).

    All other tests use T = 1, where a cps/counts mix-up is invisible; this one pins the units at T = 4 s.
    """
    T4 = 4.0
    d4 = Detector(k0=config.SENSOR_K0, background=B, T=T4)
    n = np.array([0.0, 0.5, 2.0])
    rate = d4.expected_rate(n, 1.0)
    np.testing.assert_allclose(rate, config.SENSOR_K0 * n + B)                  # cps: independent of T
    np.testing.assert_allclose(d4.expected_counts(n, 1.0), rate * T4)           # counts: rate * T
    thr = d4.detection_threshold_cps()                                          # 20 + 3*sqrt(80)/4 = 26.71 cps
    counts = np.array([thr * T4 - 1.0, thr * T4 + 1.0, thr * T4 / 2.0])         # 105.8, 107.8, 53.4 counts
    assert d4.is_detection(counts).tolist() == [False, True, False]
    # 53 counts over 4 s (13.4 cps) is below threshold although 53 > 33.4 (the T = 1 threshold in cps)
    assert not d4.is_detection(53.0)
    # measure_per_source expected vector also carries the factor T
    rng = np.random.default_rng(5)
    counts4, truth4 = d4.measure_per_source(np.ones((3, 13)), 2.0, rng)
    np.testing.assert_allclose(truth4, config.SENSOR_K0 * 2.0 * 1.0 * T4)
    # Poisson mean scales with T: 20k draws at density 0 -> mean ~ B*T4 = 80 (rel std 0.1 %)
    y = d4.measure(np.zeros(20_000), 1.0, rng)
    assert abs(y.mean() - B * T4) / (B * T4) < 0.01
    # kappa helper agrees with the T = 4 detector
    np.testing.assert_allclose(expected_counts_from_kappa(config.SENSOR_K0, n, b=B, T=T4), d4.expected_counts(n, 1.0))


def test_measure_per_source_sums_and_shapes(det):
    rng = np.random.default_rng(7)
    n, n_src = 6, 13
    d = rng.uniform(0.0, 2.0, size=(n, n_src))
    scale = 1.7
    counts, truth = det.measure_per_source(d, scale, rng)
    assert counts.shape == (n,) and counts.dtype == np.int64
    assert truth.shape == (n, n_src)
    np.testing.assert_allclose(truth, config.SENSOR_K0 * scale * d * T)
    # per-source expected vector sums to the total expected minus background
    np.testing.assert_allclose(truth.sum(axis=1) + B * T, det.expected_counts(d.sum(axis=1), scale))
    # (n_src,) single position -> (1,) counts, (1, n_src) truth
    c1, t1 = det.measure_per_source(d[0], scale, rng)
    assert c1.shape == (1,) and t1.shape == (1, n_src)
    # per-row scale (n,)
    c2, t2 = det.measure_per_source(d, np.linspace(0.3, 3.0, n), rng)
    assert c2.shape == (n,) and t2.shape == (n, n_src)
    np.testing.assert_allclose(t2[:, 0], config.SENSOR_K0 * np.linspace(0.3, 3.0, n) * d[:, 0] * T)
    with pytest.raises(ValueError):
        det.measure_per_source(d, np.ones(n + 1), rng)


def test_measure_per_source_single_draw_statistics(det):
    """Total counts follow Poisson(sum + b*T): mean and variance of the draw match (not per-source sums)."""
    rng = np.random.default_rng(11)
    d = np.tile(np.linspace(0.0, 1.0, 13), (config.VALIDATE_DETECTOR_N_POISSON, 1))
    counts, truth = det.measure_per_source(d, 1.0, rng)
    lam = truth[0].sum() + B * T
    assert abs(counts.mean() - lam) / lam < config.VALIDATE_DETECTOR_MEAN_TOL
    assert abs(counts.var() - lam) / lam < 0.05


def test_shapes_scalar_vector_matrix(det):
    rng = np.random.default_rng(3)
    assert np.ndim(det.expected_rate(1.0)) == 0
    assert det.measure(1.0, 1.0, rng).shape == ()
    assert det.measure(np.ones(5), 1.0, rng).shape == (5,)
    assert det.measure(np.ones((5, 13)), 1.0, rng).shape == (5, 13)
    assert det.measure(np.ones(5), np.linspace(0.3, 3.0, 5), rng).shape == (5,)
    assert det.normalise(np.ones((5, 13))).shape == (5, 13)
    assert det.is_detection(np.ones((2, 3))).shape == (2, 3)
    with pytest.raises(TypeError):
        det.measure(1.0, 1.0, np.random.RandomState(0))


def test_sample_scale_range_and_log_uniform():
    rng = np.random.default_rng(2024)
    lo, hi = config.SENSOR_SCALE_RANGE
    s = np.array([Detector.sample_scale(rng) for _ in range(2000)])
    assert np.all(s >= lo) and np.all(s <= hi)
    s = Detector.sample_scale(rng, size=20_000)
    assert s.shape == (20_000,)
    assert np.all(s >= lo) and np.all(s <= hi)
    med = np.median(s)
    assert abs(med - np.sqrt(lo * hi)) / np.sqrt(lo * hi) < 0.10
    # log-uniform: log(s) is uniform -> the quartiles of log s sit at 1/4 and 3/4 of the log range
    q1, q3 = np.percentile(np.log(s), [25, 75])
    lr = np.log(hi) - np.log(lo)
    assert abs(q1 - (np.log(lo) + 0.25 * lr)) < 0.05 * lr
    assert abs(q3 - (np.log(lo) + 0.75 * lr)) < 0.05 * lr
    with pytest.raises(ValueError):
        Detector.sample_scale(rng, lo=0.0, hi=1.0)


def test_expected_counts_from_kappa_broadcast():
    kgrid = config.KAPPA_REF * 10.0 ** np.linspace(-config.KAPPA_GRID_DECADES, config.KAPPA_GRID_DECADES, config.KAPPA_G)
    g = np.array([1e-4, 2e-3, 0.0])                              # (N,)
    lam = expected_counts_from_kappa(kgrid[None, :], g[:, None])
    assert lam.shape == (3, config.KAPPA_G)
    np.testing.assert_allclose(lam, (kgrid[None, :] * g[:, None] + config.SENSOR_BACKGROUND_CPS) * config.SENSOR_T)
    np.testing.assert_allclose(lam[2], config.SENSOR_BACKGROUND_CPS * config.SENSOR_T)  # zero response -> background
    assert expected_counts_from_kappa(2.0, 3.0, b=1.0, T=2.0) == 14.0
    # consistent with Detector: kappa = k0*scale, g = density
    det = Detector()
    np.testing.assert_allclose(expected_counts_from_kappa(config.SENSOR_K0 * 1.5, g), det.expected_counts(g, 1.5))


def test_detector_argument_validation():
    with pytest.raises(ValueError):
        Detector(T=0.0)
    with pytest.raises(ValueError):
        Detector(background=-1.0)