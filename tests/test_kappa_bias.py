"""Unit tests for scripts/validate_kappa_bias.py (plan S1 T1-2, D5-3; R3, R5): the four filter configurations
run end-to-end on a tiny synthetic GaussianPlume set-up (no data; N = config.T1_2_TEST_N_PARTICLES,
2 x T1_2_TEST_N_STEPS = 40 measurements, T1_2_TEST_N_REPEATS repeats) and the metric / summary / pass structures
have the right shapes and finite values; the fixed-kappa filters sit at the intended kappa."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.pf.forward_model import GaussianPlume
from srcloc_env.scripts.validate_kappa_bias import (FILTER_LABELS, FILTER_NAMES, evaluate_pass, make_filter, run_case,
                                                    sample_kappa_true, simulate_counts, summarise)
from srcloc_env.scripts.validate_pf_adjoint import two_drone_paths
from srcloc_env.sensor.detector import Detector

PRIOR_X, PRIOR_Y = (0.0, 500.0), (0.0, 300.0)      # test-local prior box (no obstacles)
TRUE_XY = (150.0, 150.0)


def test_sample_kappa_true_range_and_seed():
    rng = np.random.default_rng(3)
    k = sample_kappa_true(rng, 500)
    lo, hi = config.KAPPA_REF * 10.0 ** (-config.T1_2_KAPPA_TRUE_DECADES), config.KAPPA_REF * 10.0 ** config.T1_2_KAPPA_TRUE_DECADES
    assert k.shape == (500,) and np.all(k >= lo) and np.all(k <= hi)
    assert 0.3 < np.mean(np.log10(k / config.KAPPA_REF) > 0.0) < 0.7            # log-uniform: half above the centre
    assert np.array_equal(sample_kappa_true(np.random.default_rng(3), 500), k)


def test_make_filter_kappa_grids():
    """(i) at kappa_true, (iii) / (iv) at the fixed factors (two nodes within 1e-5 relative), (ii) the config grid."""
    model = GaussianPlume()
    kt = 3.3e4
    for name, fac in (("kappa_known", 1.0), ("kappa_fixed_high", config.T1_2_KAPPA_FIXED_FACTORS[0]),
                      ("kappa_fixed_low", config.T1_2_KAPPA_FIXED_FACTORS[1])):
        pf = make_filter(name, model, kt, np.random.default_rng(0), n_particles=20, prior_x=PRIOR_X, prior_y=PRIOR_Y)
        assert pf.G == 2 and np.allclose(pf.kgrid, fac * kt, rtol=1e-5)
        assert pf.N == 20 and pf.mode == "grid"
    pf = make_filter("rbpf_grid", model, kt, np.random.default_rng(0), n_particles=20, prior_x=PRIOR_X, prior_y=PRIOR_Y)
    assert pf.G == config.KAPPA_G and pf.kappa_ref == config.KAPPA_REF and pf.grid_decades == config.KAPPA_GRID_DECADES
    with pytest.raises(ValueError):
        make_filter("nope", model, kt, np.random.default_rng(0))


def test_simulate_counts_matches_detector_semantics():
    model = GaussianPlume()
    det = Detector()
    drn = np.array([[250.0, 150.0, config.DRONE_Z], [50.0, 150.0, config.DRONE_Z]])   # downwind / upwind of TRUE_XY
    kt = 5.0e4
    counts, expected = simulate_counts(model, np.asarray(TRUE_XY), drn, kt, det, np.random.default_rng(1))
    g = model.unit_response(np.asarray(TRUE_XY).reshape(1, 2), drn)[0]
    assert counts.shape == (2,) and counts.dtype == np.int64 and expected.shape == (2,)
    assert np.allclose(expected, (kt * g + det.background) * det.T)
    assert expected[1] == pytest.approx(det.background * det.T, rel=1e-4)       # upwind: background + kappa g_floor T (5e-5)
    assert expected[0] > 2.0 * det.background * det.T                            # downwind: the plume is seen


def test_four_filters_end_to_end_small():
    """Tiny synthetic run of run_case: shapes, finite metrics, summary / pass structure, oracle sanity."""
    model = GaussianPlume()
    det = Detector()
    paths, _ = two_drone_paths(None, TRUE_XY, config.T1_2_TEST_N_STEPS, prior_x=PRIOR_X, prior_y=PRIOR_Y)
    n_meas = config.T1_2_TEST_N_STEPS * config.PF_ADJ_N_DRONES
    report_at = (n_meas // 2, n_meas)
    res = run_case(model, np.asarray(TRUE_XY), paths, seed=0, n_repeats=config.T1_2_TEST_N_REPEATS, det=det,
                   obstacles=None, n_particles=config.T1_2_TEST_N_PARTICLES, prior_x=PRIOR_X, prior_y=PRIOR_Y,
                   report_at=report_at)
    assert res["n_measurements"] == n_meas == 40
    assert len(res["repeats"]) == config.T1_2_TEST_N_REPEATS and len(res["kappa_true"]) == config.T1_2_TEST_N_REPEATS
    for r in res["repeats"]:
        assert set(r["filters"]) == set(FILTER_NAMES)
        assert r["counts_max"] >= 0 and np.isfinite(r["expected_counts_max"])
        assert np.isfinite(r["map_distance_ii_i_m"]) and np.isfinite(r["map_error_diff_ii_minus_i_m"])
        for name in FILTER_NAMES:
            f = r["filters"][name]
            assert len(f["map_xy"]) == 2 and np.all(np.isfinite(f["map_xy"]))
            for key in ("map_error_m", "bias_x_m", "bias_y_m", "mean_error_m", "posterior_spread_m",
                        "kappa_median_over_true", "neff_final"):
                assert np.isfinite(f[key]), (name, key)
            assert f["map_error_m"] == pytest.approx(np.hypot(f["bias_x_m"], f["bias_y_m"]))
            assert f["n_updates"] == n_meas and f["n_resamples"] >= 0
            assert set(f["map_error_at"]) == {str(k) for k in report_at}
            assert f["map_error_at"][str(n_meas)] == pytest.approx(f["map_error_m"])
            assert len(f["kappa_q05_q50_q95"]) == 3 and np.all(np.isfinite(f["kappa_q05_q50_q95"]))
        assert r["filters"]["kappa_known"]["kappa_median_over_true"] == pytest.approx(1.0, rel=1e-4)
        assert r["filters"]["kappa_fixed_high"]["kappa_median_over_true"] == pytest.approx(config.T1_2_KAPPA_FIXED_FACTORS[0], rel=1e-4)
        assert r["filters"]["kappa_fixed_low"]["kappa_median_over_true"] == pytest.approx(config.T1_2_KAPPA_FIXED_FACTORS[1], rel=1e-4)
        # the MAP estimate stays inside the prior box for every filter
        for name in FILTER_NAMES:
            mx, my = r["filters"][name]["map_xy"]
            assert PRIOR_X[0] <= mx <= PRIOR_X[1] and PRIOR_Y[0] <= my <= PRIOR_Y[1]
    s = res["summary"]
    assert s["n_repeats"] == config.T1_2_TEST_N_REPEATS and set(s["filters"]) == set(FILTER_NAMES)
    for name in FILTER_NAMES:
        st = s["filters"][name]
        assert st["label"] == FILTER_LABELS[name]
        for key in ("map_error_m", "bias_x_m", "abs_bias_x_m", "kappa_median_over_true", "posterior_spread_m"):
            d = st[key]
            assert d["n"] == config.T1_2_TEST_N_REPEATS and d["q25"] <= d["median"] <= d["q75"] and np.isfinite(d["iqr"])
        assert 0.0 <= st["fraction_bias_negative"] <= 1.0
    assert s["map_distance_ii_i_m"]["n"] == config.T1_2_TEST_N_REPEATS
    p = res["pass"]
    assert p == evaluate_pass(s)
    assert set(p) >= {"pass_map_diff", "pass_bias", "pass_bias_high", "pass_bias_low", "abs_bias_x_median_m", "bias_x_median_sign"}
    assert p["pass_bias"] == (p["pass_bias_high"] and p["pass_bias_low"])
    assert p["pass_map_diff"] == (p["map_distance_ii_i_median_m"] < config.T1_2_MAP_DIFF_PASS_M)


def test_summarise_subset_of_filters():
    """summarise / run_case work for a subset of filters (no (ii)-vs-(i) block, no pass block)."""
    model = GaussianPlume()
    paths, _ = two_drone_paths(None, TRUE_XY, 5, prior_x=PRIOR_X, prior_y=PRIOR_Y)
    res = run_case(model, np.asarray(TRUE_XY), paths, seed=1, n_repeats=1, det=Detector(), obstacles=None,
                   n_particles=50, prior_x=PRIOR_X, prior_y=PRIOR_Y, report_at=(10,), filters=("kappa_known",))
    assert "pass" not in res and "map_distance_ii_i_m" not in res["summary"]
    assert list(res["summary"]["filters"]) == ["kappa_known"]
    assert summarise(res["repeats"], ("kappa_known",))["filters"]["kappa_known"]["map_error_m"]["n"] == 1