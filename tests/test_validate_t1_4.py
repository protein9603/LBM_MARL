"""Unit tests for the T1-4 aggregation helpers of scripts/validate_t1_4.py (plan S1 T1-4 / G1) on synthetic per-run
records only (no data): medians / p90 over seeds, success rate, first-success step, the G1 verdicts and the Table 1
markdown string; plus run_filter on a stub forward model (record keys, checkpoints, snapshot shapes)."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.pf.particle_filter import RBPF
from srcloc_env.scripts.validate_t1_4 import (aggregate, final_verdicts, load_baseline, load_offset_report, offset_annotation,
                                               pooled_error_curve, run_filter, source_type, table1_final_markdown,
                                               table1_markdown, verdicts)

CHECKS = (25, 50, 100, 150)


def make_record(final: float, success: bool, first: int | None, at: dict | None = None, n_steps: int = 150) -> dict:
    traj = np.linspace(300.0, final, n_steps)
    at = at if at is not None else {str(c): float(traj[c - 1]) for c in CHECKS}
    return {"map_error_at": at, "map_error_trajectory_m": traj.tolist(), "final_map_error_m": float(final),
            "final_mean_error_m": float(final) + 1.0, "final_top_sigma_m": 10.0 if success else 40.0,
            "success": success, "first_success_step": first, "entropy_at": {"0": 6.0, "50": 3.0, "150": 1.0},
            "kappa_median": 1000.0, "n_resamples": 20, "wall_seconds": 1.0}


def synthetic_runs() -> dict:
    a = {109: [make_record(10.0, True, 40), make_record(20.0, True, 60), make_record(50.0, False, None),
               make_record(5.0, True, 20), make_record(100.0, False, None)],
         110: [make_record(200.0, False, None)] * 3,
         102: [make_record(60.0, False, None), make_record(15.0, True, 100)]}
    b = {109: [make_record(8.0, True, 30)] * 5, 110: [make_record(250.0, False, None)] * 3,
         102: [make_record(12.0, True, 50), make_record(18.0, True, 70)]}
    return {"A": a, "B": b, "A2": {102: [make_record(30.0, False, None)]}, "B2": {102: [make_record(9.0, True, 45)]}}


def test_aggregate_medians_success_rate_first_step():
    agg = aggregate(synthetic_runs(), CHECKS)
    a109 = agg["A"]["109"]
    assert a109["n_seeds"] == 5
    assert a109["final_error_median"] == pytest.approx(20.0)
    assert a109["final_error_p90"] == pytest.approx(np.percentile([10, 20, 50, 5, 100], 90))
    assert a109["map_error_median"]["150"] == pytest.approx(20.0)
    assert a109["map_error_median"]["25"] == pytest.approx(np.median([np.linspace(300, f, 150)[24] for f in (10, 20, 50, 5, 100)]))
    assert a109["success_rate"] == pytest.approx(0.6) and a109["n_success"] == 3
    assert a109["first_success_step_median"] == pytest.approx(40.0)     # median over the successful seeds only
    assert agg["A"]["110"]["success_rate"] == 0.0 and agg["A"]["110"]["first_success_step_median"] is None
    assert agg["B"]["109"]["final_error_p90"] == pytest.approx(8.0)
    assert agg["B"]["102"]["first_success_step_median"] == pytest.approx(60.0)
    assert set(agg) == {"A", "B", "A2", "B2"} and set(agg["A2"]) == {"102"}
    curve = pooled_error_curve(synthetic_runs()["A"], (109, 110))
    assert curve.shape == (150,) and curve[-1] == pytest.approx(np.median([10, 20, 50, 5, 100, 200, 200, 200]))
    assert pooled_error_curve(synthetic_runs()["A"], (999,)) is None


def test_verdicts_and_table():
    runs = synthetic_runs()
    agg = aggregate(runs, CHECKS)
    max_counts = {109: 546.0, 110: 25.0, 102: 543.0}
    v = verdicts(agg, max_counts, (109, 110, 102), open_sources=(109,), pass_m=30.0, eps_sources=(102,),
                 unobservable_source=110, background_counts=20.0, unobservable_factor=1.5)
    assert v["i_A_open"]["per_source"]["109"]["pass"] and v["i_A_open"]["overall_pass"] and v["i_A_open"]["n_pass"] == 1
    assert v["ii_B_open"]["overall_pass"]
    assert v["iii_B_better_than_A"]["count"] == 2 and v["iii_B_better_than_A"]["n_sources"] == 3   # 109, 102 better; 110 worse
    assert v["iii_B_better_than_A"]["per_source"]["110"]["B_better"] is False
    assert set(v["iv_eps_trapped"]["102"]) == {"A", "B", "A2", "B2"}
    assert v["iv_eps_trapped"]["102"]["B2"]["final_error_median_m"] == pytest.approx(9.0)
    assert v["v_unobservable"]["unobservable_at_15m"] is True and v["v_unobservable"]["threshold_counts"] == 30.0
    assert v["v_unobservable"]["label"] == "unobservable at 15 m"
    v2 = verdicts(agg, {110: 60.0}, (110,), open_sources=(109,), unobservable_source=110)
    assert v2["v_unobservable"]["unobservable_at_15m"] is False and v2["v_unobservable"]["label"] == "observable"
    assert v2["v_unobservable"]["rule"] == "factor x background" and v2["v_unobservable"]["below_currie"] is None
    # Currie rule (D6-1 review): 32 counts is above 1.5 x 20 = 30 but below the 33.4-count decision threshold
    v3 = verdicts(agg, {110: 32.0}, (110,), open_sources=(109,), unobservable_source=110, background_counts=20.0,
                  unobservable_factor=1.5, currie_counts=33.4)
    assert v3["v_unobservable"]["unobservable_at_15m"] is True and v3["v_unobservable"]["threshold_counts"] == pytest.approx(33.4)
    assert v3["v_unobservable"]["below_factor_rule"] is False and v3["v_unobservable"]["below_currie"] is True
    assert v3["v_unobservable"]["threshold_factor_counts"] == pytest.approx(30.0) and v3["v_unobservable"]["label"] == "unobservable at 15 m"
    v4 = verdicts(agg, {110: 32.0}, (110,), open_sources=(109,), unobservable_source=110, background_counts=20.0,
                  unobservable_factor=1.5, currie_counts=25.0)                     # the larger of the two thresholds wins
    assert v4["v_unobservable"]["threshold_counts"] == pytest.approx(30.0) and v4["v_unobservable"]["unobservable_at_15m"] is False
    # an open source failing the criterion flips the overall verdict
    agg_fail = aggregate({"A": {109: [make_record(45.0, False, None)] * 3}, "B": {109: [make_record(5.0, True, 10)]}}, CHECKS)
    vf = verdicts(agg_fail, {}, (109,), open_sources=(109,), pass_m=30.0, eps_sources=(), unobservable_source=110)
    assert vf["i_A_open"]["overall_pass"] is False and vf["ii_B_open"]["overall_pass"] is True
    assert vf["v_unobservable"]["unobservable_at_15m"] is None and vf["v_unobservable"]["label"] == "not run"
    table = table1_markdown(agg, (109, 110, 102), {109: 1.50, 110: 2.84}, {109: 1.41, 110: 1.11},
                            {109: {"first_step_median": 13.0, "selection_rate": 1.0}, 110: {"first_step_median": None, "selection_rate": 0.8}},
                            max_counts, unobservable=(110,))
    lines = table.splitlines()
    assert len(lines) == 2 + 3 and lines[0].startswith("| source | type |") and lines[1].startswith("|---|")
    assert lines[2].startswith("| 109 | open(holdout) | 1.50 | 1.41 | 20 / ") and "| 60% | 100% | 13 (100%) | 546 |" in lines[2]
    assert "unobservable at 15 m" in lines[3] and "| 200 / 200 | 250 / 250 | 0% | 0% | n/a (80%) | 25 |" in lines[3]
    assert lines[4].startswith("| 102 | trapped | n/a | n/a |")
    assert source_type(109) == "open(holdout)" and source_type(110) == "trapped" and source_type(103) == "holdout"
    assert source_type(101) == "open" and source_type(104) == "train"


def test_final_table_and_final_verdicts(tmp_path):
    """D7-3: FINAL Table 1 (Poisson baseline vs NB re-run side by side, 110 tagged, 109 annotated) and the per-source
    G1 re-judgement (open sources < pass_m for A(NB) / B(NB), regression sources not regressed vs Poisson)."""
    import json
    agg_p = aggregate(synthetic_runs(), CHECKS)                       # baseline: A 109 median 20, B 109 8, B 102 15
    runs_n = {"A": {109: [make_record(8.0, True, 30)] * 3, 110: [make_record(180.0, False, None)] * 2,
                    102: [make_record(50.0, False, None)]},
              "B": {109: [make_record(25.0, True, 40), make_record(29.0, True, 50), make_record(26.0, False, None)],
                    110: [make_record(260.0, False, None)] * 2, 102: [make_record(40.0, False, None)]}}
    agg_n = aggregate(runs_n, CHECKS)
    # offset report round trip through a temporary calibrate_timeavg-like JSON
    rep = {"mean_peak_distance_m": 30.37, "mean_peak_direction_deg": 10.05, "instantaneous_peak_distance_m": 80.8,
           "instantaneous_peak_direction_deg": 22.0, "wind_dir_deg": 12.6}
    f_off = tmp_path / "timeavg.json"
    f_off.write_text(json.dumps({"offset_report": {"109": rep}}), encoding="utf-8")
    assert load_offset_report(f_off, 109) == rep and load_offset_report(f_off, 101) is None
    assert load_offset_report(tmp_path / "missing.json", 109) is None
    note = offset_annotation(rep, frame_index=599)
    assert note == "[systematic offset: 15 m slab peak 30 m (time mean) / 81 m (frame 599) from the source, 10-22 deg; wind 13 deg]"
    assert "missing" in offset_annotation(None)
    # baseline loader
    f_base = tmp_path / "base.json"
    f_base.write_text(json.dumps({"created": "t", "mode": "F", "pf": {"likelihood": "poisson", "nb_r": 0.3}, "n_seeds": 5,
                                  "aggregates": agg_p}), encoding="utf-8")
    base = load_baseline(f_base)
    assert base["likelihood"] == "poisson" and base["mode"] == "F" and base["aggregates"]["B"]["109"]["final_error_median"] == 8.0
    assert load_baseline(tmp_path / "none.json") is None
    table = table1_final_markdown(agg_n, base["aggregates"], (109, 110, 102), {109: 1.50, 110: 2.84}, {109: 1.41, 110: 1.11},
                                  {109: {"first_step_median": 13.0, "selection_rate": 1.0}}, {109: 546.0, 110: 32.0, 102: 543.0},
                                  unobservable=(110,), offset_source=109, offset_note=note, nb_label="negbin r=1")
    lines = table.splitlines()
    assert len(lines) == 5 and lines[0].count("|") == 12 and lines[1] == "|" + "---|" * 11
    assert "A(negbin r=1) final err" in lines[0] and "B(negbin r=1) success" in lines[0]
    assert lines[2].startswith("| 109 | open(holdout) [systematic offset: 15 m slab peak 30 m")
    assert "| 1.50 | 1.41 | 20 / 80 | 8 / 8 | 8 / 8 | 26 / 28 | 67% | 13 (100%) | 546 |" in lines[2]
    assert "| 110 | trapped [unobservable at 15 m] |" in lines[3] and "| 200 / 200 | 180 / 180 | 250 / 250 | 260 / 260 | 0% | n/a (n/a%) | 32 |" in lines[3]
    assert lines[4].startswith("| 102 | trapped | n/a | n/a | 38 / 56 | 50 / 50 | 15 / 17 | 40 / 40 | 0% |")
    # verdicts: open 109 passes for A (8 < 30) and B (26 < 30); 102 regressed (40 > max(1.5 x 15, 30) = 30)
    v = final_verdicts(agg_n, base["aggregates"], open_sources=(109,), regression_sources=(102, 110), pass_m=30.0, tolerance=1.5)
    assert v["open"]["A"]["per_source"]["109"] == {"nb_median_m": 8.0, "poisson_median_m": 20.0, "pass": True, "improved": True}
    assert v["open"]["B"]["per_source"]["109"]["pass"] is True and v["open"]["B"]["per_source"]["109"]["improved"] is False
    assert v["open"]["A"]["overall_pass"] and v["open"]["B"]["overall_pass"] and v["open"]["B"]["n_pass"] == 1
    r102 = v["regression"]["per_source"]["102"]
    assert r102["limit_m"] == 30.0 and r102["not_regressed"] is False and r102["ratio"] == pytest.approx(40.0 / 15.0)
    assert v["regression"]["per_source"]["110"]["not_regressed"] is True           # 260 <= max(1.5 x 250, 30) = 375
    assert v["regression"]["overall_pass"] is False and v["overall_pass"] is False
    v2 = final_verdicts(agg_n, base["aggregates"], open_sources=(109,), regression_sources=(110,), pass_m=30.0, tolerance=1.5)
    assert v2["regression"]["per_source"]["110"]["not_regressed"] is True and v2["regression"]["pooled"]["pass"] is True
    assert v2["regression"]["overall_pass"] and v2["overall_pass"]
    v3 = final_verdicts(agg_n, base["aggregates"], open_sources=(109, 102), regression_sources=(), pass_m=30.0)
    assert v3["open"]["A"]["overall_pass"] is False and v3["regression"]["pooled"] is None and v3["overall_pass"] is False


class _StubForward:
    """Unit response = Gaussian bump around the drone (no data, deterministic)."""

    def __init__(self, width: float = 40.0):
        self.width = width

    def unit_response(self, source_xy, drone_xyz):
        s = np.asarray(source_xy, dtype=float).reshape(-1, 2)
        d = np.asarray(drone_xyz, dtype=float).reshape(-1, 3)[:, :2]
        r2 = ((s[:, None, :] - d[None, :, :]) ** 2).sum(-1)
        return np.maximum(1e-3 * np.exp(-0.5 * r2 / self.width ** 2), config.FWD_G_FLOOR)


def test_run_filter_record_and_snapshot_shapes():
    """run_filter on a stub model: record keys / checkpoint semantics, success detection, snapshot array shapes."""
    true_xy = (600.0, 0.0)
    n_steps, n_drones, n_particles = 20, 2, 300
    fwd = _StubForward()
    rng = np.random.default_rng(3)
    # two drones circling the source: counts from the stub response with kappa 1e5
    ang = np.linspace(0.0, 2.0 * np.pi, n_steps, endpoint=False)
    paths = np.stack([np.column_stack([600.0 + 30.0 * np.cos(ang), 30.0 * np.sin(ang)]),
                      np.column_stack([600.0 + 60.0 * np.cos(ang + 1.0), 60.0 * np.sin(ang + 1.0)])], axis=1)
    g = fwd.unit_response(np.asarray(true_xy), np.column_stack([paths.reshape(-1, 2), np.full(n_steps * n_drones, config.DRONE_Z)]))[0]
    counts = rng.poisson((1e5 * g + config.SENSOR_BACKGROUND_CPS) * config.SENSOR_T).reshape(n_steps, n_drones)
    pf = RBPF(fwd, n_particles=n_particles, mode="grid", obstacles=None, kappa_ref=1e5, prior_x=(500.0, 700.0),
              prior_y=(-100.0, 100.0), rng=np.random.default_rng(1))          # small prior box: 300 particles converge in 20 steps
    rec, snap = run_filter(pf, counts, paths, true_xy, checkpoints=(5, 10, 20, 50), entropy_steps=(0, 10, 20),
                           gmm_every=5, snapshot=True)
    assert set(rec["map_error_at"]) == {"5", "10", "20"}                       # 50 > n_steps is dropped
    assert rec["map_error_at"]["20"] == pytest.approx(rec["final_map_error_m"])
    assert len(rec["map_error_trajectory_m"]) == n_steps and rec["min_map_error_m"] <= rec["final_map_error_m"]
    assert set(rec["entropy_at"]) == {"0", "10", "20"} and rec["entropy_at"]["0"] > rec["entropy_at"]["20"]
    assert rec["n_checks"] == 4 and 0 <= rec["n_success_checks"] <= 4
    assert rec["success"] == (rec["first_success_step"] is not None)
    if rec["success"]:
        assert rec["first_success_step"] % 5 == 0
    assert rec["final_map_error_m"] < 30.0 and np.isfinite(rec["final_top_sigma_m"])
    assert rec["kappa_median"] > 0 and rec["n_resamples"] >= 0 and rec["wall_seconds"] > 0
    assert snap["steps"].tolist() == [0, 5, 10, 15, 20]
    assert snap["xy"].shape == (5, n_particles, 2) and snap["xy"].dtype == np.float32
    assert snap["logw"].shape == (5, n_particles) and snap["logw"].dtype == np.float32
    assert snap["drone_xy"].shape == (5, n_drones, 2) and snap["map_xy"].shape == (5, 2)
    assert np.allclose(snap["true_xy"], true_xy) and np.allclose(snap["drone_xy"][1], paths[4])
    rec2, snap2 = run_filter(RBPF(fwd, n_particles=50, obstacles=None, prior_x=(500.0, 700.0), prior_y=(-100.0, 100.0),
                                  rng=np.random.default_rng(1)), counts, paths, true_xy,
                             checkpoints=(20,), gmm_every=7, snapshot=False)
    assert snap2 is None and rec2["n_checks"] == 3                              # steps 7, 14 and the final step 20