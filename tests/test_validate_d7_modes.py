"""Unit tests for scripts/validate_d7_modes.py and the Mode T measurement support of scripts/validate_t1_4.py
(plan D7-2; plan 1 시간 모드: frame f(t) = min(N_FILES - 1, round(FRAME_START_MODE_T[0] + FILES_PER_RL_STEP t))):
the frame schedule, mode_t_densities / generate_measurements on a stub field backend, the likelihood settings and
the recommendation rule on synthetic tables (no data)."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.scripts.validate_d7_modes import (config_key, final_error_table, likelihood_settings, per_source_table,
                                                  recommend, table_markdown)
from srcloc_env.scripts.validate_t1_4 import (aggregate, frame_index_mode_t, frame_schedule_mode_t,
                                               generate_measurements, mode_t_densities, truth_densities)
from srcloc_env.sensor.detector import Detector

OPEN = (101, 108, 109, 111, 113)
REG = (102, 104, 106)


# ---------------------------------------------------------------------------------------- frame schedule
def test_frame_index_mode_t_schedule():
    start = config.FRAME_START_MODE_T[0]
    last = config.N_FILES - 1
    assert config.T1_4_MODE_T_START_INDEX == start
    assert frame_index_mode_t(0) == start
    assert frame_index_mode_t(1) == round(start + config.FILES_PER_RL_STEP) == 102
    f = [frame_index_mode_t(t) for t in range(700)]
    assert all(b >= a for a, b in zip(f, f[1:]))                       # monotone non-decreasing
    assert all(0 <= v <= last for v in f) and f[-1] == last             # capped at N_FILES - 1
    t_cap = int(np.ceil((last - start) / config.FILES_PER_RL_STEP))     # first step at the cap
    assert f[t_cap] == last and f[t_cap - 1] < last
    sched = frame_schedule_mode_t(700)
    assert sched.dtype == np.int64 and sched.shape == (700,) and sched.tolist() == f
    assert frame_schedule_mode_t(config.D7_2_N_STEPS)[-1] == 338       # 100 + 1.6 x 149 = 338.4 -> 338 < 599
    assert frame_schedule_mode_t(config.D7_2_N_STEPS)[0] == start
    assert frame_index_mode_t(5, start=350) == 358 and frame_index_mode_t(1000, start=350) == last
    assert frame_schedule_mode_t(3, start=10, files_per_step=2.0, n_files=14).tolist() == [10, 12, 13]
    with pytest.raises(ValueError):
        frame_index_mode_t(-1)


# ---------------------------------------------------------------------------------------- stub backend
class _StubBackend:
    """density = 1e-3 frame + 1e-4 x + 1e-5 y + 1e-2 src (deterministic, frame-dependent); records every call."""

    def __init__(self):
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def density(self, src_ids, xy, frame_index, z=config.DRONE_Z, scale=1.0):
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        self.calls.append((tuple(int(s) for s in src_ids), int(frame_index)))
        return float(scale) * (1e-3 * frame_index + 1e-4 * pts[:, 0] + 1e-5 * pts[:, 1] + 1e-2 * sum(src_ids))


def _paths(n_steps: int, n_drones: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform([400.0, -300.0], [1000.0, 300.0], size=(n_steps, n_drones, 2))


def test_mode_t_densities_stub_backend():
    be = _StubBackend()
    p1, p2 = _paths(20, seed=1), _paths(12, seed=2)                    # different episode lengths
    d = mode_t_densities(be, {101: p1, 102: p2}, start=100)
    assert set(d) == {101, 102} and d[101].shape == (20, 2) and d[102].shape == (12, 2)
    for s, p in ((101, p1), (102, p2)):
        for t in range(p.shape[0]):
            ref = _StubBackend().density([s], p[t], frame_index_mode_t(t, start=100))
            assert np.allclose(d[s][t], ref)
    frames = [fr for _, fr in be.calls]
    assert frames == sorted(frames)                                     # ascending frame pass
    assert len(set(be.calls)) == len(be.calls)                          # each (source, frame) queried once
    assert set(frames) == set(frame_schedule_mode_t(20, start=100).tolist())
    assert mode_t_densities(be, {}) == {}
    # truth_densities dispatch
    assert np.allclose(truth_densities(be, 101, p1, "T"), d[101])
    ref_f = be.density([101], p1.reshape(-1, 2), 599).reshape(20, 2)
    assert np.array_equal(truth_densities(be, 101, p1, "F", frame_index=599), ref_f)
    with pytest.raises(ValueError):
        truth_densities(be, 101, p1, "X")


def test_generate_measurements_modes_and_cache():
    be, det = _StubBackend(), Detector()
    paths = _paths(15, seed=3)
    counts_f, exp_f = generate_measurements(be, det, 109, paths, 4, mode="F")
    # bit-identical to the D6 draw: flat densities of the fixed frame, rng [seed, source, 1]
    flat = _StubBackend().density([109], paths.reshape(-1, 2), config.T1_4_FRAME_INDEX)
    ref = det.measure(flat, config.T1_4_SENSOR_SCALE, np.random.default_rng([4, 109, 1])).reshape(15, 2)
    assert np.array_equal(counts_f, ref) and np.allclose(exp_f, det.expected_counts(flat, config.T1_4_SENSOR_SCALE).reshape(15, 2))
    # the densities cache reproduces the field query exactly
    dens_f = truth_densities(be, 109, paths, "F")
    counts_c, exp_c = generate_measurements(be, det, 109, paths, 4, densities=dens_f)
    assert np.array_equal(counts_c, counts_f) and np.array_equal(exp_c, exp_f)
    # mode T: same shapes and rng stream, different truth
    counts_t, exp_t = generate_measurements(be, det, 109, paths, 4, mode="T")
    dens_t = mode_t_densities(_StubBackend(), {109: paths})[109]
    assert counts_t.shape == (15, 2) and np.allclose(exp_t, det.expected_counts(dens_t, config.T1_4_SENSOR_SCALE))
    assert not np.allclose(exp_t, exp_f)
    counts_t2, _ = generate_measurements(be, det, 109, paths, 4, densities=dens_t)
    assert np.array_equal(counts_t2, counts_t)


# ---------------------------------------------------------------------------------------- settings / keys
def test_likelihood_settings_and_config_key():
    st = likelihood_settings()
    assert [s["label"] for s in st] == ["poisson", "negbin_r0.3", "negbin_r1", "negbin_r3"]
    assert st[0]["likelihood"] == "poisson" and st[0]["nb_r"] is None
    assert [s["nb_r"] for s in st[1:]] == [0.3, 1.0, 3.0] and all(s["likelihood"] == "negbin" for s in st[1:])
    assert config_key("poisson", "F") == "poisson_F" and config_key(st[1]["label"], "T") == "negbin_r0.3_T"
    with pytest.raises(ValueError):
        config_key("poisson", "X")
    with pytest.raises(ValueError):
        likelihood_settings((0.3, -1.0))


# ---------------------------------------------------------------------------------------- recommendation rule
def _row(open_vals, reg_vals) -> dict:
    return {**dict(zip(OPEN, open_vals)), **dict(zip(REG, reg_vals))}


def synthetic_table() -> dict:
    return {
        "poisson_F": _row([140, 160, 110, 340, 60], [19, 12, 16]),        # open med 140, reg med 16 -> limit 24
        "negbin_r0.3_F": _row([44, 12, 92, 43, 30], [20, 14, 18]),        # 43 / 18 admissible
        "negbin_r1_F": _row([49, 12, 100, 44, 25], [30, 25, 28]),         # 44 / 28 > 24 inadmissible
        "negbin_r3_F": _row([80, 70, 100, 90, 60], [15, 15, 15]),         # 80 / 15
        "poisson_T": _row([90, 95, 110, 85, 70], [20, 20, 20]),           # 90 / 20
        "negbin_r0.3_T": _row([20, 15, 25, 18, 22], [40, 40, 40]),        # 20 (best) / 40 inadmissible
        "negbin_r1_T": _row([50, 45, 55, 40, 60], [22, 22, 22]),          # 50 / 22
        "negbin_r3_T": _row([70, 65, 75, 60, 80], [23, 23, 23]),          # 70 / 23
    }


def test_recommend_rule_all_and_within_mode():
    table = synthetic_table()
    r = recommend(table, OPEN, REG, baseline_key="poisson_F", tolerance=1.5, n_top=3)
    assert r["baseline_regression_median_m"] == 16.0 and r["regression_limit_m"] == pytest.approx(24.0)
    assert r["top"] == ["negbin_r0.3_F", "negbin_r1_T", "negbin_r3_T"] and r["recommended"] == "negbin_r0.3_F"
    assert r["ranking"] == ["negbin_r0.3_F", "negbin_r1_T", "negbin_r3_T", "negbin_r3_F", "poisson_T", "poisson_F"]
    assert r["n_admissible"] == 6
    pc = r["per_config"]
    assert pc["negbin_r0.3_T"]["admissible"] is False and pc["negbin_r0.3_T"]["rank"] is None
    assert pc["negbin_r0.3_T"]["open_median_m"] == 20.0                # best statistic but violates the constraint
    assert pc["negbin_r1_F"]["admissible"] is False and pc["negbin_r1_F"]["regression_ratio_to_baseline"] == pytest.approx(28 / 16)
    assert pc["negbin_r0.3_F"]["rank"] == 1 and pc["poisson_F"]["rank"] == 6 and pc["poisson_F"]["admissible"] is True
    assert "1.5" in r["rule"] and "poisson_F" in r["rule"]
    # within mode T (candidates restricted, Poisson-F baseline kept)
    rt = recommend(table, OPEN, REG, candidates=["poisson_T", "negbin_r0.3_T", "negbin_r1_T", "negbin_r3_T"])
    assert rt["ranking"] == ["negbin_r1_T", "negbin_r3_T", "poisson_T"] and rt["recommended"] == "negbin_r1_T"
    assert rt["regression_limit_m"] == pytest.approx(24.0) and set(rt["per_config"]) == set(rt["candidates"])
    rf = recommend(table, OPEN, REG, candidates=["poisson_F", "negbin_r0.3_F", "negbin_r1_F", "negbin_r3_F"])
    assert rf["ranking"] == ["negbin_r0.3_F", "negbin_r3_F", "poisson_F"]
    # vacuous constraint (no regression sources): every config admissible, pure open-median ranking
    r0 = recommend(table, OPEN, (), n_top=2)
    assert r0["regression_limit_m"] is None and r0["n_admissible"] == 8 and r0["top"] == ["negbin_r0.3_T", "negbin_r0.3_F"]
    # a tolerance so tight that nothing but the baseline passes
    r1 = recommend(table, OPEN, REG, tolerance=1.0)
    assert r1["ranking"] == ["negbin_r3_F", "poisson_F"] and r1["recommended"] == "negbin_r3_F"
    with pytest.raises(KeyError):
        recommend(table, OPEN, REG, baseline_key="missing")
    # candidates absent from the table are ignored
    assert recommend(table, OPEN, REG, candidates=["poisson_F", "nope"])["candidates"] == ["poisson_F"]


def _rec(final: float, success: bool, first, n_steps: int = 150) -> dict:
    traj = np.linspace(300.0, final, n_steps)
    return {"map_error_at": {str(c): float(traj[c - 1]) for c in (50, 100, 150)}, "map_error_trajectory_m": traj.tolist(),
            "final_map_error_m": float(final), "final_mean_error_m": float(final) + 1.0,
            "final_top_sigma_m": 8.0 if success else 45.0, "success": success, "first_success_step_median": None,
            "first_success_step": first, "entropy_at": {"0": 6.0, "50": 3.0, "150": 1.0}, "kappa_median": 1e3,
            "n_resamples": 10, "wall_seconds": 1.0}


def test_tables_from_aggregates():
    runs = {"poisson_F": {101: [_rec(140.0, False, None), _rec(100.0, False, None), _rec(160.0, False, None)],
                          102: [_rec(19.0, True, 40), _rec(10.0, True, 30), _rec(25.0, False, None)]},
            "negbin_r0.3_F": {101: [_rec(44.0, True, 90), _rec(30.0, True, 80), _rec(50.0, False, None)],
                              102: [_rec(20.0, True, 50)] * 3}}
    agg = aggregate(runs, (50, 100, 150))
    table = final_error_table(agg)
    assert table == {"poisson_F": {101: 140.0, 102: 19.0}, "negbin_r0.3_F": {101: 44.0, 102: 20.0}}
    per = per_source_table(agg, (50, 100, 150))
    p = per["negbin_r0.3_F"]["101"]
    assert p["final_error_median_m"] == 44.0 and p["success_rate"] == pytest.approx(2 / 3) and p["n_seeds"] == 3
    assert set(p["map_error_median_at"]) == {"50", "100", "150"} and p["map_error_median_at"]["150"] == 44.0
    assert p["top_sigma_median_m"] == 8.0 and p["first_success_step_median"] == 85.0
    assert per["poisson_F"]["101"]["success_rate"] == 0.0 and per["poisson_F"]["101"]["top_sigma_median_m"] == 45.0
    md = table_markdown(table, (101, 102), ["poisson_F", "negbin_r0.3_F"], per)
    lines = md.splitlines()
    assert lines[0] == "| source | type | poisson_F | negbin_r0.3_F |" and lines[1] == "|---|---|---|---|"
    assert lines[2] == "| 101 | open | 140 (0%) | 44 (67%) |" and lines[3].startswith("| 102 | trapped | 19 (67%) | 20 (100%) |")
    assert table_markdown(table, (101,), ["poisson_F", "missing"]).splitlines()[2] == "| 101 | open | 140 | n/a |"
    r = recommend(table, (101,), (102,))
    assert r["recommended"] == "negbin_r0.3_F" and r["per_config"]["negbin_r0.3_F"]["regression_ratio_to_baseline"] == pytest.approx(20 / 19)