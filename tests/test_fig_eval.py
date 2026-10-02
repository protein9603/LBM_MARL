"""Tests of scripts/fig_eval.py on small synthetic evaluation directories (records CSV, step logs, summary.json; no real data)."""
import csv
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.eval import metrics
from srcloc_env.scripts import fig_eval

SOURCES = tuple(config.ALL_SOURCES)            # 13 sources including the unobservable 110
HORIZON = 30                                   # synthetic horizon (summary.json max_steps)
FIELDS = ["method", "n_drones", "episode_id", "seed", "source", "frame", "scale", "mode", "start_type", "success", "steps", "success_strict",
          "steps_strict", "min_error_m", "final_error_m", "first_detection_step", "declared_step", "declared_error_m", "path_length_m", "n_masked",
          "entropy_final", "top_sigma_final_m", "wall_s", "step_ms_median", "tie_tol", "tie_frac", "all_tied_frac", "calib_2sigma_frac", "step_log"]
# (method, n_drones, success probability): oracle best, random worst
CONFIGS = (("gmm_infotaxis", 1, 0.4), ("gmm_infotaxis", 2, 0.55), ("ppo_x", 1, 0.5), ("ppo_x", 2, 0.7), ("oracle_loiter", 1, 0.9))


class FakeSlab:
    """Minimal SlabFrame look-alike: a Gaussian blob per source on a 40 x 40 grid of 10 m cells."""

    class Grid:
        x0, y0, nx, ny, res = 0.0, -100.0, 40, 40, 10.0

    grid = Grid()
    sources = tuple(config.ALL_SOURCES)
    z_levels = (config.DRONE_Z,)

    def __init__(self, frame: int) -> None:
        yy, xx = np.mgrid[0:40, 0:40]
        blob = np.exp(-(((xx - 20) ** 2 + (yy - 20) ** 2) / 60.0))
        self.density = np.stack([blob * (1.0 + 0.01 * frame) for _ in self.sources])[:, None]

    def z_index(self, z: float) -> int:
        return 0


class FakeBackend:
    def __init__(self) -> None:
        self.frames: list[int] = []

    def slab(self, frame: int) -> FakeSlab:
        self.frames.append(int(frame))
        return FakeSlab(frame)


class FakeObstacles:
    x = np.arange(0.0, 400.0, 10.0)
    y = np.arange(-100.0, 300.0, 10.0)
    occ = np.zeros((40, 40), dtype=bool)
    occ[5:9, 25:30] = True


def synth_episode(cfg_idx: int, ep: int, source: int, p_success: float, horizon: int) -> dict:
    """Deterministic synthetic episode: error 250 m / sigma 100 m until the success step s, afterwards 15 m / 10 m (strict) or 40 m / 25 m (primary only)."""
    rng = np.random.default_rng(1000 * cfg_idx + ep)
    success = bool(rng.random() < p_success) and source != 110
    return {"success": success, "s": int(rng.integers(5, horizon - 4)), "strict": bool(rng.random() < 0.5),
            "start_type": "plume" if ep % 2 == 0 else "random", "scale": float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))}


def step_arrays(spec: dict, source: int, n_drones: int, horizon: int, new_log: bool) -> tuple[dict, dict]:
    truth = np.array(config.SOURCES_XY[source], dtype=float) * 0.25 + np.array([50.0, 50.0])      # inside the fake 400 x 400 window
    T, s = horizon, spec["s"]
    err = np.full(T, 250.0)
    sig = np.full(T, 100.0)
    ent = np.full(T, 7.0)
    if spec["success"]:
        err[s - 1:] = 15.0 if spec["strict"] else 40.0
        sig[s - 1:] = 10.0 if spec["strict"] else 25.0
        ent[s - 1:] = 6.0
    start = truth + np.array([[120.0, 30.0 * (d + 1)] for d in range(n_drones)])
    tt = np.linspace(0.0, 1.0, T)[:, None, None]
    drone_xy = start[None] * (1 - tt) + (truth[None, None] + 20.0) * tt                          # (T, D, 2)
    mu = np.zeros((T, 3, 2))
    mu[:, 0] = truth + np.stack([err, np.zeros(T)], axis=1)
    mu[:, 1:] = truth + 80.0
    cov = np.zeros((T, 3, 2, 2))
    for k in range(3):
        cov[:, k] = np.eye(2) * (sig[:, None, None] ** 2)
    arrs = {"drone_xy": drone_xy, "y": np.random.default_rng(1).integers(15, 60, size=(T, n_drones)), "action": np.zeros((T, n_drones), int),
            "applied": np.ones((T, n_drones), bool), "gmm_w": np.tile([0.7, 0.2, 0.1], (T, 1)), "gmm_mu": mu, "gmm_cov": cov,
            "gmm_mask": np.ones((T, 3), bool), "map_xy": mu[:, 0].copy(), "entropy": ent, "top_sigma": sig, "map_error": err, "truth_xy": truth}
    meta = {"source": source, "scale": spec["scale"], "seed": 7, "success": spec["success"], "n_drones": n_drones, "start_type": spec["start_type"]}
    if new_log:
        arrs.update({"start_xy": start, "reward": np.zeros(T), "frame": np.full(T, 450), "calib_inside": err <= 2 * sig})
        meta.update({"mode": "F", "max_steps": horizon})
    return arrs, meta


def build_eval_dir(root: Path, name: str, mode: str = "F", horizon: int = HORIZON, records: bool = True, steps: bool = True, summary: bool = True,
                   configs=CONFIGS, per_source: int = 1, new_log: bool = False) -> tuple[Path, list[dict]]:
    """Write episodes.csv, records_*.csv, steps/*.npz and summary.json of a synthetic evaluation; returns (dir, ground-truth rows)."""
    d = Path(root) / name
    (d / "steps").mkdir(parents=True)
    eps = [(i * per_source + k, src) for i, src in enumerate(SOURCES) for k in range(per_source)]
    with (d / "episodes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode_id", "seed", "source", "frame", "scale", "mode", "reflect"])
        for ep, src in eps:
            w.writerow([ep, 20261001 + 1000 * ep, src, 430 + ep % 20, 1.0, mode, False])
    truth_rows = []
    for ci, (method, n, p) in enumerate(configs):
        rows = []
        for ep, src in eps:
            spec = synth_episode(ci, ep, src, p, horizon)
            arrs, meta = step_arrays(spec, src, n, horizon, new_log)
            meta.update({"method": method, "frame": 430 + ep % 20})
            path = d / "steps" / f"{method}_{n}drones_ep{ep:04d}.npz"
            if steps:
                np.savez_compressed(path, **arrs, meta=json.dumps(meta))
            ok, s = spec["success"], (spec["s"] if spec["success"] else horizon)
            err, sig = arrs["map_error"], arrs["top_sigma"]
            row = {"method": method, "n_drones": n, "episode_id": ep, "seed": 20261001 + 1000 * ep, "source": src, "frame": 430 + ep % 20,
                   "scale": spec["scale"], "mode": mode, "start_type": spec["start_type"], "success": ok, "steps": s,
                   "success_strict": bool(ok and spec["strict"]), "steps_strict": s if (ok and spec["strict"]) else horizon,
                   "min_error_m": float(err.min()), "final_error_m": float(err[-1]), "first_detection_step": 3, "declared_step": s if ok else "",
                   "declared_error_m": float(err[s - 1]) if ok else "", "path_length_m": 100.0, "n_masked": 0, "entropy_final": float(arrs["entropy"][-1]),
                   "top_sigma_final_m": float(sig[-1]), "wall_s": 1.0, "step_ms_median": 10.0, "tie_tol": "", "tie_frac": "", "all_tied_frac": "",
                   "calib_2sigma_frac": "", "step_log": str(path)}
            rows.append(row)
            truth_rows.append({**row, "spec": spec})
        if records:
            with (d / f"records_{method}_{n}drones.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
    if summary:
        (d / "summary.json").write_text(json.dumps({"tag": name, "mode": mode, "max_steps": horizon}), encoding="utf-8")
    return d, truth_rows


@pytest.fixture(autouse=True)
def fast_figures(monkeypatch):
    monkeypatch.setattr(fig_eval, "RECORDS_LOADER", fig_eval._load_records_light)   # the real run_eval.load_records imports the whole environment stack
    monkeypatch.setattr(fig_eval, "FIG_DPI", 40)
    monkeypatch.setattr(fig_eval, "get_scene_backends", lambda: (FakeBackend(), FakeObstacles()))


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One complete make_all on a synthetic Mode F directory with old-format step logs (no frame / calib_inside / start_xy)."""
    root = tmp_path_factory.mktemp("fig_eval")
    d, truth = build_eval_dir(root, "synthA")
    backend = FakeBackend()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fig_eval, "RECORDS_LOADER", fig_eval._load_records_light)
        mp.setattr(fig_eval, "FIG_DPI", 40)
        manifest = fig_eval.make_all(d, root / "out", "syn", backend=backend, obstacles=FakeObstacles(), n_boot=100)
    return {"manifest": manifest, "out": root / "out", "dir": d, "truth": truth, "root": root, "backend": backend}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def truth_of(full_run, method, n):
    return [r for r in full_run["truth"] if r["method"] == method and r["n_drones"] == n]


def obs(rows):
    return [r for r in rows if r["source"] != 110]


OK_OUTPUTS = ("table2_md", "table2_csv", "table2_png", "success_bars", "success_cdf", "map_error_vs_step", "source_heatmap", "paired_diff", "trajectories",
              "entropy_curves", "calibration", "threshold_curve", "start_type_bars", "start_type_table", "sensitivity_scale", "error_at_steps", "efficiency",
              "compute_cost", "figure_sources")


def test_manifest_and_files(full_run):
    man = full_run["manifest"]
    for k in OK_OUTPUTS:
        assert man[k]["status"] == "ok", (k, man[k])
        p = Path(man[k]["path"])
        assert p.is_file() and p.stat().st_size > 0, k
    assert man["sensitivity"]["status"] == "skipped" and "several" in man["sensitivity"]["reason"]
    saved = json.loads((full_run["out"] / "syn_manifest.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(man))
    rows = read_csv(man["figure_sources"]["path"])
    assert {r["output"] for r in rows} == set(OK_OUTPUTS) - {"figure_sources"}
    assert all(r["data_files"] and r["columns_used"] and r["what_it_shows"] for r in rows)
    png = Path(man["table2_png"]["path"]).read_bytes()[:8]
    assert png == b"\x89PNG\r\n\x1a\n"


def test_table2_matches_metrics(full_run):
    rows = read_csv(full_run["out"] / "syn_table2.csv")
    assert {(r["method"], int(r["n_drones"])) for r in rows} == {(m, n) for m, n, _ in CONFIGS}
    got = next(r for r in rows if r["method"] == "ppo_x" and r["n_drones"] == "2" and r["group"] == "all_observable")
    t = obs(truth_of(full_run, "ppo_x", 2))
    k, n = sum(r["success"] for r in t), len(t)
    assert int(got["n"]) == n == 12 and int(got["n_success"]) == k
    rate, lo, hi = metrics.wilson_ci(k, n)
    assert float(got["success_rate"]) == pytest.approx(rate) and float(got["success_ci_lo"]) == pytest.approx(lo) and float(got["success_ci_hi"]) == pytest.approx(hi)
    ks = sum(r["success_strict"] for r in t)
    assert int(got["n_success_strict"]) == ks
    cens = np.median([r["steps"] if r["success"] else HORIZON for r in t])
    assert float(got["censored_step_median"]) == pytest.approx(cens)
    unob = next(r for r in rows if r["method"] == "gmm_infotaxis" and r["n_drones"] == "1" and r["group"] == "unobservable")
    assert int(unob["n"]) == 1 and int(unob["n_success"]) == 0
    md = Path(full_run["manifest"]["table2_md"]["path"]).read_text(encoding="utf-8")
    assert "| Method (drones) | Group |" in md and "ppo_x (2) | all_observable | 12 |" in md and f"horizon {HORIZON}" in md
    assert "oracle_loiter is a verification baseline" in md


def test_threshold_curve_values(full_run):
    rows = [r for r in read_csv(full_run["manifest"]["threshold_curve"]["data_csv"]) if r["method"] == "ppo_x" and r["n_drones"] == "2"]
    t = obs(truth_of(full_run, "ppo_x", 2))
    by_thr = {float(r["error_threshold_m"]): float(r["success_rate"]) for r in rows}
    assert sorted(by_thr) == [float(x) for x in fig_eval.THRESHOLD_GRID_M]
    n_strict = sum(r["success"] and r["spec"]["strict"] for r in t)                # error 15 m (sigma 10 m): below every threshold from 20 m on
    n_any = sum(r["success"] for r in t)                                           # error 40 m (sigma 25 m): below 45 m, not below 40 m (strict inequality)
    assert by_thr[20.0] == pytest.approx(n_strict / len(t))
    assert by_thr[40.0] == pytest.approx(n_strict / len(t))
    assert by_thr[45.0] == pytest.approx(n_any / len(t))
    assert by_thr[100.0] == pytest.approx(n_any / len(t))


def test_map_error_and_entropy_curves(full_run):
    t = obs(truth_of(full_run, "gmm_infotaxis", 1))
    rows = [r for r in read_csv(full_run["manifest"]["map_error_vs_step"]["data_csv"]) if r["method"] == "gmm_infotaxis" and r["n_drones"] == "1"]
    last = next(r for r in rows if int(r["step"]) == HORIZON)
    errs = [r["final_error_m"] for r in t]
    assert float(last["center"]) == pytest.approx(np.median(errs)) and float(last["q25"]) == pytest.approx(np.quantile(errs, 0.25))
    assert float(rows[0]["center"]) == pytest.approx(250.0) and int(rows[0]["n_episodes"]) == len(t)
    ent = [r for r in read_csv(full_run["manifest"]["entropy_curves"]["data_csv"]) if r["method"] == "ppo_x" and r["n_drones"] == "1"]
    t2 = obs(truth_of(full_run, "ppo_x", 1))
    assert float(ent[-1]["center"]) == pytest.approx(np.median([r["entropy_final"] for r in t2]) / 7.0)
    assert full_run["manifest"]["entropy_curves"]["h0_nats"] == pytest.approx(7.0)


def test_calibration_recomputed_from_gmm(full_run):
    assert fig_eval.CALIB_NOMINAL == pytest.approx(0.8647, abs=1e-4)
    t = obs(truth_of(full_run, "ppo_x", 1))
    cal = [r for r in read_csv(full_run["manifest"]["calibration"]["data_csv"]) if r["method"] == "ppo_x" and r["n_drones"] == "1"]
    expect = np.mean([r["success"] for r in t])                    # truth inside the 2-sigma ellipse exactly for the successful synthetic episodes (15 <= 20, 40 <= 50, 250 > 200)
    assert float(cal[-1]["center"]) == pytest.approx(expect)
    assert float(cal[0]["center"]) == pytest.approx(0.0)


def test_heatmap_marks_unobservable_source(full_run):
    rows = read_csv(full_run["manifest"]["source_heatmap"]["data_csv"])
    assert {int(r["source"]) for r in rows} == set(SOURCES)
    assert all((r["observable"] == "True") == (int(r["source"]) != 110) for r in rows)
    row = next(r for r in rows if r["method"] == "oracle_loiter" and r["source"] == "101")
    t = [r for r in truth_of(full_run, "oracle_loiter", 1) if r["source"] == 101]
    assert int(row["n"]) == len(t) and float(row["success_rate"]) == pytest.approx(np.mean([r["success"] for r in t]))


def test_paired_differences_against_default_reference(full_run):
    man = full_run["manifest"]["paired_diff"]
    assert man["reference"] == "gmm_infotaxis"
    rows = read_csv(man["data_csv"])
    assert "oracle_loiter" not in {r["method"] for r in rows}               # verification method is never compared as a competitor
    row = next(r for r in rows if r["method"] == "ppo_x" and r["n_drones"] == "2" and r["scope"] == "all observable" and r["comparison"] == "method_vs_reference")
    a, b = obs(truth_of(full_run, "ppo_x", 2)), obs(truth_of(full_run, "gmm_infotaxis", 2))
    assert int(row["n"]) == len(a)
    assert float(row["success_diff"]) == pytest.approx(np.mean([x["success"] for x in a]) - np.mean([x["success"] for x in b]))
    steps = [(x["steps"] if x["success"] else HORIZON) - (y["steps"] if y["success"] else HORIZON) for x, y in zip(a, b)]
    assert float(row["censored_step_diff_median"]) == pytest.approx(np.median(steps))
    assert float(row["success_diff_lo"]) <= float(row["success_diff"]) <= float(row["success_diff_hi"])
    one_source = next(r for r in rows if r["method"] == "ppo_x" and r["n_drones"] == "1" and r["scope"] == "101" and r["comparison"] == "method_vs_reference")
    assert int(one_source["n"]) == 1
    assert not any(r["scope"] == "110" for r in rows)


def test_paired_reference_missing_is_skipped(tmp_path):
    d, _ = build_eval_dir(tmp_path, "synthR")
    ev = fig_eval.load_eval_set(d)
    out = fig_eval.Outputs(tmp_path / "o", "p", ev.tag, "")
    with pytest.raises(fig_eval.Skip, match="not in this evaluation directory"):
        fig_eval.build_paired(ev, out, "no_such_method", None, 50)


def test_trajectory_selection_is_representative(full_run):
    ev = fig_eval.load_eval_set(full_run["dir"])
    sel = {s["key"]: s for s in fig_eval.select_trajectory_episodes(ev)}
    assert list(sel) == ["open_success", "trapped_success", "failure", "multi_drone"]
    a, b, c, e = (sel[k]["record"] for k in sel)
    assert a["method"] == "ppo_x" and a["n_drones"] == 1 and a["success"] and a["source"] in fig_eval.OPEN_SOURCES
    assert b["success"] and b["source"] in fig_eval.TRAPPED_SOURCES and not sel["trapped_success"]["note"]
    assert not c["success"] and c["source"] != 110
    assert e["n_drones"] == 2 and e["method"] == "ppo_x" and e["success"]
    pool = [r["steps"] for r in ev.recs("ppo_x", 1) if r["success"] and r["source"] in fig_eval.OPEN_SOURCES]
    assert abs(a["steps"] - np.median(pool)) == min(abs(np.array(pool) - np.median(pool)))      # the success closest to the median step
    forced = fig_eval.select_trajectory_episodes(ev, preferred="gmm_infotaxis")
    assert forced[0]["record"]["method"] == "gmm_infotaxis"


def test_trajectories_figure_panels(full_run):
    man = full_run["manifest"]["trajectories"]
    panels = read_csv(man["data_csv"])
    assert [p["panel"] for p in panels] == ["a", "b", "c", "d"]
    assert {p["key"] for p in panels} == {"open_success", "trapped_success", "failure", "multi_drone"}
    assert all(Path(p["step_log"]).is_file() for p in panels)
    assert len(full_run["backend"].frames) == 4 and set(full_run["backend"].frames) <= {430 + k for k in range(20)}    # Mode F: the frozen frame of each episode
    assert panels[3]["n_drones"] == "2" and panels[3]["episode_id"] != panels[0]["episode_id"]                         # largest drone count, another scene than panel a


def test_trajectory_without_trapped_success_notes_fallback(tmp_path):
    cfgs = (("ppo_x", 1, 0.5),)
    d, truth = build_eval_dir(tmp_path, "synthT", configs=cfgs)
    ev = fig_eval.load_eval_set(d)
    for r in ev.records:                                                       # no success on the trapped sources, no multi-drone configuration
        if r["source"] in fig_eval.TRAPPED_SOURCES:
            r["success"] = False
    sel = {s["key"]: s for s in fig_eval.select_trajectory_episodes(ev)}
    assert sel["trapped_success"]["record"]["source"] not in fig_eval.TRAPPED_SOURCES and "no success on a trapped source" in sel["trapped_success"]["note"]
    assert sel["multi_drone"]["record"] is None and "no multi-drone" in sel["multi_drone"]["note"]
    out = fig_eval.Outputs(tmp_path / "o", "p", ev.tag, "")
    fig_eval.build_trajectories(ev, out, None, None, backend=FakeBackend(), obstacles=FakeObstacles())
    assert out.manifest["trajectories"]["status"] == "ok" and "multi_drone: no multi-drone" in out.manifest["trajectories"]["reason"]


def test_episode_density_frame_modes():
    arrs = {"drone_xy": np.zeros((150, 1, 2))}
    assert fig_eval.episode_density_frame(arrs, {"frame": 433, "mode": "F"}, "F") == (433, 150)
    assert fig_eval.episode_density_frame(arrs, {"frame": 410}, "T2") == (410 + 75, 76)                      # old log: start frame + middle step index
    assert fig_eval.episode_density_frame({"drone_xy": np.zeros((400, 1, 2))}, {"frame": 450}, "T2") == (599, 201)   # clipped at the last cached frame
    arrs["frame"] = np.arange(412, 412 + 150)
    assert fig_eval.episode_density_frame(arrs, {"frame": 412, "mode": "T2"}, "F") == (412 + 75, 76)           # logged frame array and meta mode win


def test_calibration_inside_matches_direct_mahalanobis():
    rng = np.random.default_rng(3)
    T = 40
    truth = np.array([300.0, -50.0])
    mu = truth + rng.normal(scale=40.0, size=(T, 3, 2))
    a = rng.normal(size=(T, 3, 2, 2))
    cov = a @ a.transpose(0, 1, 3, 2) * 300.0 + np.eye(2) * 50.0
    w = rng.dirichlet(np.ones(3), size=T)
    mask = np.ones((T, 3), bool)
    mask[::5, 0] = False                                                        # the top component is the largest VALID weight
    arrs = {"gmm_mu": mu, "gmm_cov": cov, "gmm_w": w, "gmm_mask": mask, "truth_xy": truth}
    got = fig_eval.calibration_inside(arrs)
    top = np.argmax(np.where(mask, w, -np.inf), axis=1)
    ref = np.array([(truth - mu[t, k]) @ np.linalg.solve(cov[t, k], truth - mu[t, k]) <= 4.0 for t, k in enumerate(top)])
    assert got.dtype == bool and np.array_equal(got, ref) and 0 < ref.sum() < T
    logged = np.zeros(T, bool)
    assert not fig_eval.calibration_inside({**arrs, "calib_inside": logged}).any()                  # the logged flag takes precedence
    assert fig_eval.calibration_inside({"truth_xy": truth}) is None


SMALL = (("random", 1, 0.15), ("ppo_x", 2, 0.7))


def test_rebuild_records_from_step_logs_when_records_are_missing(tmp_path):
    d, truth = build_eval_dir(tmp_path, "synthB", mode="T2", records=False, summary=False, configs=SMALL[1:])
    ev = fig_eval.load_eval_set(d)
    assert ev.mode == "T2" and ev.horizon == config.T2_MAX_STEPS == 150                       # no summary.json: episodes.csv mode, default horizon of Mode T2
    assert len(ev.records) == len(truth) and len(ev.rebuilt) == 1 and any("rebuilt" in n for n in ev.notes)
    by = {(r["method"], r["n_drones"], r["episode_id"]): r for r in ev.records}
    for t in truth:
        r = by[(t["method"], t["n_drones"], t["episode_id"])]
        assert r["success"] == t["success"] and r["steps"] == t["steps"] and r["success_strict"] == t["success_strict"]
        assert r["source"] == t["source"] and r["start_type"] == t["start_type"] and r["mode"] == "T2" and r["frame"] == t["frame"]
        assert r["final_error_m"] == pytest.approx(t["final_error_m"]) and r["declared_step"] == (t["steps"] if t["success"] else None)
        assert r["step_log"].endswith(f"{t['method']}_{t['n_drones']}drones_ep{t['episode_id']:04d}.npz")
        ys = np.load(r["step_log"])["y"]                                                     # first step with a count above the Currie threshold (33.4 cps for b = 20, T = 1 s)
        thr = (config.SENSOR_BACKGROUND_CPS + config.SENSOR_CURRIE_K * np.sqrt(config.SENSOR_BACKGROUND_CPS * config.SENSOR_T) / config.SENSOR_T) * config.SENSOR_T
        hits = np.flatnonzero((ys > thr).any(axis=1))
        assert r["first_detection_step"] == (int(hits[0]) + 1 if hits.size else None)
    dc, truth_c = build_eval_dir(tmp_path, "synthC", records=False, summary=False, new_log=True, configs=SMALL[:1])
    new = fig_eval.load_eval_set(dc)
    assert new.mode == "F" and new.horizon == HORIZON                                             # new logs carry max_steps in their meta
    for t in truth_c:                                                                             # logged calib_inside: inside from the success step on (15 <= 20, 40 <= 50), never before
        r = next(x for x in new.records if x["episode_id"] == t["episode_id"])
        expect = (HORIZON - t["spec"]["s"] + 1) / HORIZON if t["success"] else 0.0
        assert r["calib_2sigma_frac"] == pytest.approx(expect)


def test_builders_skip_without_step_logs_and_on_empty_directory(tmp_path):
    d, _ = build_eval_dir(tmp_path, "synthN", steps=False, configs=SMALL)
    ev = fig_eval.load_eval_set(d)
    out = fig_eval.Outputs(tmp_path / "o1", "p", ev.tag, "")
    for build in (lambda: fig_eval.build_map_error(ev, out, None, None), lambda: fig_eval.build_entropy(ev, out, None, None),
                  lambda: fig_eval.build_calibration(ev, out, None, None), lambda: fig_eval.build_threshold_curve(ev, out, None, None),
                  lambda: fig_eval.build_error_at_steps(ev, out, None, None)):
        with pytest.raises(fig_eval.Skip, match="no step logs found"):
            build()
    with pytest.raises(fig_eval.Skip, match="step log not found"):
        fig_eval.build_trajectories(ev, out, None, None, backend=FakeBackend(), obstacles=FakeObstacles())
    empty = tmp_path / "empty"
    empty.mkdir()
    man = fig_eval.make_all(empty, tmp_path / "o2", "q")
    assert all(v["status"] == "skipped" for k, v in man.items() if k != "figure_sources") and man["table2_png"]["reason"] == "no evaluation records"
    assert (tmp_path / "o2" / "q_manifest.json").is_file() and man["figure_sources"]["status"] == "ok"
    assert read_csv(man["figure_sources"]["path"]) == []


def test_bad_inputs_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        fig_eval.make_all("no_such_tag_for_fig_eval_test", tmp_path, "p")
    with pytest.raises(ValueError):
        fig_eval.make_all([], tmp_path, "p")


def test_several_tags_make_separate_sets_and_a_sensitivity_figure(tmp_path, monkeypatch):
    a, _ = build_eval_dir(tmp_path, "tagF", mode="F", horizon=HORIZON, steps=False, configs=SMALL)
    b, _ = build_eval_dir(tmp_path, "tagT2", mode="T2", horizon=15, steps=False, configs=SMALL)
    calls = []

    def stub(ev, out_dir, prefix, suffix="", *args):
        calls.append((ev.tag, ev.mode, ev.horizon, prefix, suffix))
        return fig_eval.Outputs(out_dir, prefix, ev.tag, suffix)

    monkeypatch.setattr(fig_eval, "make_for_set", stub)                          # per-tag figures are tested elsewhere; here only the wiring and the sensitivity figure
    man = fig_eval.make_all([a, b], tmp_path / "o", "cmp", backend=FakeBackend(), obstacles=FakeObstacles())
    assert calls == [("tagF", "F", HORIZON, "cmp_tagF", "@tagF"), ("tagT2", "T2", 15, "cmp_tagT2", "@tagT2")]
    assert man["sensitivity"]["status"] == "ok" and "tagF, tagT2" in man["sensitivity"]["reason"] and Path(man["sensitivity"]["path"]).is_file()
    rows = read_csv(man["sensitivity"]["data_csv"])
    assert {(r["tag"], r["mode"], r["horizon"]) for r in rows} == {("tagF", "F", "30"), ("tagT2", "T2", "15")} and {r["method"] for r in rows} == {"random", "ppo_x"}
    only_a = fig_eval.make_all([a, b], tmp_path / "o2", "cmp2", compare_tags=[str(a)])
    assert only_a["sensitivity"]["status"] == "skipped" and "several" in only_a["sensitivity"]["reason"]


def test_series_cap_and_method_order(tmp_path):
    d, _ = build_eval_dir(tmp_path, "synthS", configs=SMALL[1:], per_source=2)
    ev = fig_eval.load_eval_set(d)
    s = ev.series("ppo_x", 2, per_source_cap=1)
    assert s.err.shape == (len(SOURCES), HORIZON) and s.n_missing == 0 and set(s.source) == set(SOURCES)
    assert ev.series("ppo_x", 2).err.shape[0] == 2 * len(SOURCES)
    assert fig_eval.order_methods(["oracle_loiter", "ppo_m2", "random", "gmm_infotaxis", "lawnmower_band", "ppo_m1", "lawnmower", "greedy_map"]) == [
        "random", "lawnmower", "greedy_map", "gmm_infotaxis", "lawnmower_band", "ppo_m1", "ppo_m2", "oracle_loiter"]
    ms = ["random", "ppo_m1", "ppo_m2"]
    assert fig_eval.method_colour("ppo_m1", ms) != fig_eval.method_colour("ppo_m2", ms) and fig_eval.method_colour("random", ms) == fig_eval.FIXED_COLOURS["random"]


def test_cli_arguments_and_manifest_printout(tmp_path, monkeypatch, capsys):
    seen = {}

    def stub(eval_dirs, out_dir, prefix, reference_method=None, compare_tags=None, **kw):
        seen.update(eval_dirs=eval_dirs, out_dir=out_dir, prefix=prefix, reference=reference_method, compare=compare_tags, kw=kw)
        return {"table2_md": {"path": "a.md", "status": "ok", "reason": ""}, "sensitivity": {"path": "", "status": "skipped", "reason": "needs several tags"}}

    monkeypatch.setattr(fig_eval, "make_all", stub)
    man = fig_eval.main(["--tag", "t_one", "--tag2", "t_two", "t_three", "--out-dir", str(tmp_path), "--prefix", "cli", "--reference-method", "gmm_infotaxis",
                         "--compare-tags", "t_one", "t_two", "--methods", "ppo_x", "random", "--episodes-per-source", "1", "--traj-method", "ppo_x", "--dpi", "300"])
    out = capsys.readouterr().out
    assert seen["eval_dirs"] == ["t_one", "t_two", "t_three"] and seen["prefix"] == "cli" and seen["reference"] == "gmm_infotaxis" and seen["compare"] == ["t_one", "t_two"]
    assert seen["kw"]["methods"] == ["ppo_x", "random"] and seen["kw"]["log_eps_per_source"] == 1 and seen["kw"]["traj_method"] == "ppo_x" and seen["kw"]["dpi"] == 300
    assert "table2_md" in out and "skipped" in out and "needs several tags" in out and man["table2_md"]["status"] == "ok"
    with pytest.raises(SystemExit):
        fig_eval.main(["--out-dir", str(tmp_path)])                                # no tag given


def test_zoom_limits_and_square_window():
    steps = np.arange(1, 11)
    base = ("gmm_infotaxis", 1, steps, np.linspace(1.0, 0.98, 10), None, None)
    oracle = ("oracle_loiter", 1, steps, np.linspace(1.0, 0.2, 10), None, None)
    lim = fig_eval._zoom_limits([base, oracle], 1.0)
    assert lim is not None and lim[0] < 0.98 < 1.0 < lim[1]
    assert fig_eval._zoom_limits([base], 1.0) is None and fig_eval._zoom_limits([base, oracle], None) is None
    far = ("gmm_infotaxis", 1, steps, np.linspace(1.0, 0.5, 10), None, None)
    assert fig_eval._zoom_limits([far, oracle], 1.0) is None                      # the baselines already use more than a quarter of the range
    x0, y0, x1, y1 = fig_eval._square_window(np.array([[100.0, 0.0], [400.0, 50.0]]), 90.0, 300.0)
    assert x1 - x0 == pytest.approx(480.0) and y1 - y0 == pytest.approx(480.0) and (x0 + x1) / 2 == pytest.approx(250.0) and (y0 + y1) / 2 == pytest.approx(25.0)
    x0, y0, x1, y1 = fig_eval._square_window(np.array([[0.0, 0.0], [10.0, 10.0]]), 10.0, 300.0)
    assert x1 - x0 == pytest.approx(300.0) and y1 - y0 == pytest.approx(300.0)


# ------------------------------------------------------------------------------------------------ review fixes and spec 9 / 10 gaps
@pytest.fixture
def spy_figures(monkeypatch):
    """Record layout facts of every saved figure (before it is closed): title box inside the figure, x tick labels, overlaps, texts."""
    seen: list[dict] = []
    orig = fig_eval._save

    def spy(fig, path):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        sup = getattr(fig, "_suptitle", None)
        axes = []
        for ax in fig.axes:
            if not ax.axison:
                continue
            labs = [t for t in ax.get_xticklabels() if t.get_text() and t.get_visible()]
            boxes = [t.get_window_extent(r) for t in labs]
            axes.append({"xlabels": [t.get_text() for t in labs], "labelbottom": ax.xaxis.get_tick_params(which="major").get("labelbottom"),
                         "overlap": any(boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))),
                         "texts": [t.get_text() for t in ax.texts]})
        sb = sup.get_window_extent(r) if sup is not None else None
        seen.append({"name": Path(path).name, "suptitle": sup.get_text() if sup is not None else "", "axes": axes,
                     "title_inside": None if sb is None else bool(sb.x0 >= -1 and sb.x1 <= fig.bbox.x1 + 1 and sb.y1 <= fig.bbox.y1 + 1)})
        return orig(fig, path)

    monkeypatch.setattr(fig_eval, "_save", spy)
    return seen


def test_table2_has_declared_fields(full_run):
    rows = read_csv(full_run["out"] / "syn_table2.csv")
    for col in ("declared_n", "declared_success_rate", "declared_conditional_rate", "declared_overconfident_rate", "declaration_delay_median"):
        assert col in rows[0], col
    got = next(r for r in rows if r["method"] == "ppo_x" and r["n_drones"] == "2" and r["group"] == "all_observable")
    t = obs(truth_of(full_run, "ppo_x", 2))
    n_succ = sum(r["success"] for r in t)
    assert n_succ > 0 and int(got["declared_n"]) == n_succ
    assert float(got["declared_success_rate"]) == pytest.approx(n_succ / len(t))                 # all episodes in the denominator (spec 9.7)
    assert float(got["declared_conditional_rate"]) == pytest.approx(1.0)                         # declared errors are 15 / 40 m < 50 m
    n_over = sum(r["success"] and not r["spec"]["strict"] for r in t)                            # 40 m >= 20 m
    assert float(got["declared_overconfident_rate"]) == pytest.approx(n_over / n_succ)
    assert float(got["declaration_delay_median"]) == pytest.approx(0.0)                          # declared at the success step
    md = Path(full_run["manifest"]["table2_md"]["path"]).read_text(encoding="utf-8").splitlines()
    head = next(ln for ln in md if ln.startswith("| Method (drones)"))
    for col in ("Declared success rate (all episodes)", "Declared correct (share of declaring episodes)", "Overconfident declarations", "Declaration delay median [steps]"):
        assert col in head, col
    table = [ln for ln in md if ln.startswith("|")]
    assert len({ln.count("|") for ln in table}) == 1                                              # every row has the header's column count
    row = next(ln for ln in table if ln.startswith("| ppo_x (2) | all_observable |"))
    assert f"| 100% of {n_succ} |" in row and f"| {100 * n_succ / len(t):.0f}% |" in row
    assert Path(full_run["manifest"]["table2_png"]["path"]).stat().st_size > 0


def test_table2_markdown_rows_without_declarations():
    a = {"n": 5, "success_rate": 0.0, "success_ci": [0.0, 0.43], "success_strict_rate": 0.0, "success_step_median": float("nan"), "success_step_ci": [float("nan")] * 2,
         "censored_step_median": 30.0, "final_error_median_m": 200.0, "final_error_p90_m": 300.0, "declared_n": 0, "declared_success_rate": 0.0,
         "declared_conditional_rate": None, "declared_overconfident_rate": None, "declaration_delay_median": None}
    lines = fig_eval.table2_markdown_rows({"random (1)": {"all_observable": a}})
    assert lines[2].endswith("| 0% | - | - | - |") and lines[2].count("|") == lines[0].count("|")


def test_step_log_titles_state_the_episodes_used(tmp_path, monkeypatch):
    d, _ = build_eval_dir(tmp_path, "synthE", configs=SMALL[1:], per_source=3, horizon=15)       # 39 episodes per configuration, 36 on observable sources
    ev = fig_eval.load_eval_set(d)
    titles = []
    monkeypatch.setattr(fig_eval, "_suptitle", lambda fig, text, fontsize=11.0: titles.append(text))
    out = fig_eval.Outputs(tmp_path / "o", "p", ev.tag, "")
    builds = {"map_error_vs_step": fig_eval.build_map_error, "entropy_curves": fig_eval.build_entropy, "calibration": fig_eval.build_calibration,
              "threshold_curve": fig_eval.build_threshold_curve, "error_at_steps": fig_eval.build_error_at_steps}
    for key, build in builds.items():
        build(ev, out, None, 1)
        assert "12 of 39 episodes per configuration (first 1 per source)" in titles[-1], (key, titles[-1])
        assert out.manifest[key]["episodes_used"] == {"ppo_x (2)": {"used": 12, "total": 39}} and "12 of 39" in out.manifest[key]["episodes_title"]
    fig_eval.build_map_error(ev, out, None, None)
    assert "36 episodes per configuration" in titles[-1] and " of " not in titles[-1].splitlines()[1]
    victim = next(r for r in ev.recs("ppo_x", 2) if r["source"] == 101)
    Path(victim["step_log"]).unlink()                                                             # one unreadable log is reported, not hidden
    ev2 = fig_eval.load_eval_set(d)
    fig_eval.build_map_error(ev2, out, None, None)
    assert "35 of 39 episodes per configuration (1 without readable step log)" in titles[-1] and out.manifest["map_error_vs_step"]["n_missing_logs"] == 1
    assert fig_eval._title(ev, "T", "7 of 9 episodes per configuration").endswith("7 of 9 episodes per configuration")


def test_paired_drone_count_pairs_groups_and_ratios(full_run):
    man = full_run["manifest"]["paired_diff"]
    rows = read_csv(man["data_csv"])
    dc = [r for r in rows if r["comparison"] == "drone_count"]
    assert {(r["method"], r["n_drones"], r["n_drones_reference"]) for r in dc} == {("gmm_infotaxis", "2", "1"), ("ppo_x", "2", "1")}   # oracle: one drone count only
    assert "ppo_x: 2 vs 1" in man["drone_count_pairs"] and all(r["reference"] == r["method"] for r in dc)
    groups = {"train_open", "train_other", "holdout", "train", "all observable"}
    assert groups <= {r["scope"] for r in dc} and groups <= {r["scope"] for r in rows if r["comparison"] == "method_vs_reference"}
    assert {r["scope_kind"] for r in rows if r["scope"] in ("holdout", "train")} == {"group"} and next(r for r in rows if r["scope"] == "101")["scope_kind"] == "source"
    hold = set(metrics.GROUPS["holdout"])
    g = next(r for r in dc if r["method"] == "ppo_x" and r["scope"] == "holdout")
    a, b = [r for r in truth_of(full_run, "ppo_x", 2) if r["source"] in hold], [r for r in truth_of(full_run, "ppo_x", 1) if r["source"] in hold]
    assert int(g["n"]) == len(hold) and float(g["success_diff"]) == pytest.approx(np.mean([r["success"] for r in a]) - np.mean([r["success"] for r in b]))
    allr = next(r for r in dc if r["method"] == "ppo_x" and r["scope"] == "all observable")
    a, b = obs(truth_of(full_run, "ppo_x", 2)), obs(truth_of(full_run, "ppo_x", 1))
    sa, sb = [x["steps"] for x in a if x["success"]], [x["steps"] for x in b if x["success"]]
    both = [x["steps"] / y["steps"] for x, y in zip(a, b) if x["success"] and y["success"]]
    assert both and int(allr["n_both_success"]) == len(both) and int(allr["n_common"]) == len(a)
    assert float(allr["ideal_step_ratio"]) == 0.5
    assert float(allr["step_ratio_of_medians"]) == pytest.approx(np.median(sa) / np.median(sb))
    assert float(allr["step_ratio_paired_median"]) == pytest.approx(np.median(both))
    assert float(allr["step_ratio_paired_lo"]) <= float(allr["step_ratio_paired_median"]) <= float(allr["step_ratio_paired_hi"])
    assert float(allr["flight_time_median_hi"]) == pytest.approx(2 * np.median(sa)) and float(allr["flight_time_median_lo"]) == pytest.approx(np.median(sb))
    assert float(allr["flight_time_ratio_of_medians"]) == pytest.approx(2 * np.median(sa) / np.median(sb))
    assert float(allr["flight_time_paired_ratio_median"]) == pytest.approx(2 * np.median(both))
    assert next(r for r in rows if r["comparison"] == "method_vs_reference")["step_ratio_paired_median"] == ""          # plain pairs carry no ratio
    assert (int(allr["n_drones"]), int(allr["n_drones_reference"])) == (2, 1)


def _rec(i, ok, steps):
    return {"episode_id": i, "success": ok, "steps": steps}


def test_ratio_stats_hand_made_records():
    hi = [_rec(0, True, 10), _rec(1, True, 20), _rec(2, True, 30), _rec(3, False, 100)]
    lo = [_rec(0, True, 20), _rec(1, True, 40), _rec(2, True, 60), _rec(3, True, 90), _rec(4, True, 5)]                  # episode 4 is not common
    s = fig_eval.ratio_stats(hi, lo, 2, 1, 100, 200)
    assert s["n_common"] == 4 and s["n_success_hi"] == 3 and s["n_success_lo"] == 4 and s["n_both_success"] == 3
    assert (s["step_median_hi"], s["step_median_lo"]) == (20.0, 50.0) and s["step_ratio_of_medians"] == pytest.approx(0.4)
    assert (s["censored_median_hi"], s["censored_median_lo"], s["censored_ratio"]) == (25.0, 50.0, pytest.approx(0.5))
    assert s["step_ratio_paired_median"] == pytest.approx(0.5) and s["step_ratio_paired_lo"] == pytest.approx(0.5) and s["step_ratio_paired_hi"] == pytest.approx(0.5)
    assert s["ideal_step_ratio"] == 0.5 and (s["flight_time_median_hi"], s["flight_time_median_lo"]) == (40.0, 50.0)
    assert s["flight_time_ratio_of_medians"] == pytest.approx(0.8) and s["flight_time_paired_ratio_median"] == pytest.approx(1.0)
    assert s["censored_flight_time_ratio"] == pytest.approx(1.0)
    empty = fig_eval.ratio_stats([], lo, 2, 1, 100, 50)
    assert empty["n_common"] == 0 and np.isnan(empty["step_ratio_of_medians"])


def test_multi_drone_panel_uses_largest_count_and_another_episode(tmp_path):
    d, _ = build_eval_dir(tmp_path, "synthM", configs=(("ppo_x", 1, 0.5), ("ppo_x", 2, 0.6), ("ppo_x", 3, 0.7)), per_source=2)
    sel = {s["key"]: s for s in fig_eval.select_trajectory_episodes(fig_eval.load_eval_set(d))}
    e = sel["multi_drone"]["record"]
    assert e["n_drones"] == 3 and e["success"] and sel["multi_drone"]["title"] == "3-drone episode" and not sel["multi_drone"]["note"]
    assert e["episode_id"] not in {sel[k]["record"]["episode_id"] for k in ("open_success", "trapped_success")}
    d2, _ = build_eval_dir(tmp_path, "synthM2", configs=(("ppo_x", 2, 0.7), ("gmm_infotaxis", 2, 0.5)), per_source=2)        # every configuration flies two drones
    ev2 = fig_eval.load_eval_set(d2)
    sel2 = {s["key"]: s for s in fig_eval.select_trajectory_episodes(ev2)}
    a, e2 = sel2["open_success"]["record"], sel2["multi_drone"]["record"]
    assert a["n_drones"] == e2["n_drones"] == 2 and e2["success"] and e2["episode_id"] != a["episode_id"]
    for r in ev2.records:                                                                       # a single successful episode: its scene is reused and flagged
        r["success"] = r["success"] and r["method"] == "ppo_x" and r["episode_id"] == a["episode_id"]
    sel3 = {s["key"]: s for s in fig_eval.select_trajectory_episodes(ev2)}
    assert sel3["multi_drone"]["record"]["episode_id"] == a["episode_id"] and "same scene as panel a" in sel3["multi_drone"]["note"]


def test_detection_mask_and_rings_on_trajectory_panel():
    thr = fig_eval.currie_threshold_cps() * config.SENSOR_T
    assert thr == pytest.approx(33.4, abs=0.1)
    assert fig_eval.detection_mask(np.array([[10, 40], [34, 33]])).tolist() == [[False, True], [True, False]]
    spec = synth_episode(0, 4, 101, 1.0, HORIZON)
    arrs, meta = step_arrays(spec, 101, 2, HORIZON, True)
    meta["frame"] = 440
    fig, ax = plt.subplots()
    sc = fig_eval.draw_episode_panel(ax, arrs, meta, "F", FakeBackend(), FakeObstacles(), "t")
    colour = matplotlib.colors.to_rgba(fig_eval.R_DETECTION_COLOUR)
    rings = [c for c in ax.collections if len(c.get_edgecolor()) and np.allclose(c.get_edgecolor()[0], colour) and len(c.get_offsets())]
    expected = int(fig_eval.detection_mask(arrs["y"]).sum())
    assert 0 < expected < arrs["y"].size and sum(len(c.get_offsets()) for c in rings) == expected
    assert len(sc.get_array()) == HORIZON and sc.get_array()[0] == 1 and sc.get_array()[-1] == HORIZON          # the path keeps its step colouring (colour bar)
    plt.close(fig)


def test_threshold_curve_reference_lines_are_labelled_by_error(tmp_path, spy_figures):
    d, _ = build_eval_dir(tmp_path, "synthTh", configs=SMALL[1:])
    ev = fig_eval.load_eval_set(d)
    fig_eval.build_threshold_curve(ev, fig_eval.Outputs(tmp_path / "o", "p", ev.tag, ""), None, None)
    texts = [t for a in spy_figures[-1]["axes"] for t in a["texts"]]
    assert "20 m error" in texts and "50 m error (primary)" in texts and not any("strict" in t for t in texts)


def test_trajectory_frame_fallback_uses_files_per_step_with_round(monkeypatch):
    old = {"drone_xy": np.zeros((150, 1, 2))}
    assert fig_eval.episode_density_frame(old, {"frame": 410}, "T2", files_per_step=1.6) == (410 + 120, 76)
    assert fig_eval.episode_density_frame(old, {"frame": 410}, "T2", files_per_step=0.5) == (448, 76)      # round(447.5) = 448; int() would give 447
    monkeypatch.setattr(config, "T1_4_MODE_T2_FILES_PER_STEP", 1.6)                                          # read from the config at call time
    assert fig_eval.episode_density_frame(old, {"frame": 410}, "T2") == (530, 76)
    assert fig_eval.episode_density_frame({"drone_xy": np.zeros((400, 1, 2))}, {"frame": 450}, "T2") == (599, 201)   # still clipped at the last cached frame


def test_layout_inner_ticks_overlapping_labels_and_title_clipping(tmp_path, spy_figures):
    methods = ("random", "lawnmower", "greedy_map", "gmm_infotaxis", "lawnmower_alongwind", "ppo_x")
    d, _ = build_eval_dir(tmp_path, "synthL", configs=tuple((m, 1, 0.3) for m in methods), horizon=15)
    ev = fig_eval.load_eval_set(d)
    out = fig_eval.Outputs(tmp_path / "o", "p", ev.tag, "")
    fig_eval.build_map_error(ev, out, None, None)
    me = spy_figures[-1]
    assert len(me["axes"]) == 6 and all(a["labelbottom"] for a in me["axes"]) and all(a["xlabels"] for a in me["axes"])      # 2 x 4 grid: the upper row keeps its x tick numbers
    fig_eval.build_start_type(ev, out, None)
    st = spy_figures[-1]
    assert st["name"].endswith("start_type.png") and not any(a["overlap"] for a in st["axes"]) and len(st["axes"][0]["xlabels"]) == len(methods)
    assert "\n" in st["axes"][0]["xlabels"][-2]                                                              # method names are wrapped
    fig_eval.build_calibration(ev, out, None, None)
    assert spy_figures[-1]["title_inside"] is True                                                           # single-panel title is wrapped, not clipped
    assert fig_eval.wrap_method("lawnmower_alongwind") == "lawnmower\nalongwind" and fig_eval.wrap_method("oracle_loiter").endswith("(verif.)")


def test_show_xticks_restores_the_inner_row_labels():
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    assert not axes[0][0].xaxis.get_tick_params(which="major")["labelbottom"]                     # shared grids hide the upper row
    fig_eval._show_xticks(axes.ravel())
    assert all(a.xaxis.get_tick_params(which="major")["labelbottom"] for a in axes.ravel())
    plt.close(fig)


def test_sensitivity_scale_bins(full_run):
    assert [fig_eval.scale_bin(s) for s in (0.3, 0.6, 0.6001, 1.0, 1.69, 1.7, 3.0)] == [0, 0, 1, 1, 1, 2, 2]
    assert fig_eval.scale_bin_labels() == ["<= 0.6", "0.6-1.7", ">= 1.7"] and fig_eval.R_SCALE_BIN_EDGES == (0.6, 1.7)
    man = full_run["manifest"]["sensitivity_scale"]
    rows = read_csv(man["data_csv"])
    t = obs(truth_of(full_run, "ppo_x", 2))
    for lab, inside in (("<= 0.6", lambda s: s <= 0.6), ("0.6-1.7", lambda s: 0.6 < s < 1.7), (">= 1.7", lambda s: s >= 1.7)):
        got = next((r for r in rows if r["method"] == "ppo_x" and r["n_drones"] == "2" and r["scale_bin"] == lab), None)
        sub = [r for r in t if inside(r["spec"]["scale"])]
        if sub:
            k = sum(r["success"] for r in sub)
            assert int(got["n"]) == len(sub) and int(got["n_success"]) == k
            assert float(got["success_rate"]) == pytest.approx(metrics.wilson_ci(k, len(sub))[0]) and float(got["ci_hi"]) == pytest.approx(metrics.wilson_ci(k, len(sub))[2])
        else:
            assert got is None
    assert man["scale_bins"] == ["<= 0.6", "0.6-1.7", ">= 1.7"]


def test_sensitivity_scale_skipped_when_scale_is_constant(tmp_path):
    d, _ = build_eval_dir(tmp_path, "synthK", configs=SMALL[1:])
    ev = fig_eval.load_eval_set(d)
    for r in ev.records:
        r["scale"] = 1.0
    out = fig_eval.Outputs(tmp_path / "o", "p", ev.tag, "")
    with pytest.raises(fig_eval.Skip, match="does not vary"):
        fig_eval.build_sensitivity_scale(ev, out)


def test_effective_steps_and_error_at_steps_values(tmp_path):
    assert fig_eval._effective_steps(150) == [(50, [50]), (100, [100]), (150, [150, 300])]
    assert fig_eval._effective_steps(300) == [(50, [50]), (100, [100]), (150, [150]), (300, [300])]
    assert fig_eval.R_ERROR_AT_STEPS == (50, 100, 150, 300)
    horizon = 60
    d, truth = build_eval_dir(tmp_path, "synthH", horizon=horizon, configs=(("ppo_x", 2, 0.7), ("gmm_infotaxis", 1, 0.4)), per_source=2)
    ev = fig_eval.load_eval_set(d)
    out = fig_eval.Outputs(tmp_path / "o", "p", ev.tag, "")
    fig_eval.build_error_at_steps(ev, out, None, None)
    man = out.manifest["error_at_steps"]
    assert man["steps_effective"] == [50, 60] and Path(man["path"]).is_file()
    rows = [r for r in read_csv(man["data_csv"]) if r["method"] == "ppo_x" and r["group"] == "all_observable"]
    assert [(int(r["requested_step"]), int(r["step"]), r["clipped"]) for r in rows] == [(50, 50, "False"), (100, 60, "True"), (150, 60, "True"), (300, 60, "True")]
    t = [r for r in truth if r["method"] == "ppo_x" and r["source"] != 110]

    def err_at(r, k):                                                                          # synthetic error: 250 m until the success step, then 15 / 40 m
        spec = r["spec"]
        return (15.0 if spec["strict"] else 40.0) if spec["success"] and spec["s"] <= k else 250.0
    for row in rows:
        k = int(row["step"])
        vals = [err_at(r, k) for r in t]
        assert int(row["n_episodes"]) == len(t) and float(row["median_error_m"]) == pytest.approx(np.median(vals))
        assert float(row["q25_error_m"]) == pytest.approx(np.quantile(vals, 0.25)) and float(row["p90_error_m"]) == pytest.approx(np.quantile(vals, 0.9))
    assert float(next(r for r in rows if r["step"] == "60")["median_error_m"]) == pytest.approx(np.median([r["final_error_m"] for r in t]))
    holdout = [r for r in read_csv(man["data_csv"]) if r["method"] == "ppo_x" and r["group"] == "holdout" and r["requested_step"] == "50"]
    assert int(holdout[0]["n_episodes"]) == 2 * len(metrics.GROUPS["holdout"])


def test_efficiency_distributions_and_drone_count_rows(tmp_path):
    d, _ = build_eval_dir(tmp_path, "synthF", configs=(("ppo_x", 1, 0.6), ("ppo_x", 2, 0.7), ("random", 1, 0.2)), per_source=2, steps=False)
    ev = fig_eval.load_eval_set(d)
    for r in ev.records:                                                                       # make the per-episode columns vary
        r["path_length_m"] = 10.0 * r["episode_id"] + 5.0 * r["n_drones"]
        r["n_masked"] = r["episode_id"] % 3
        r["first_detection_step"] = None if r["episode_id"] % 4 == 0 else 2 + r["episode_id"]
    out = fig_eval.Outputs(tmp_path / "o", "p", ev.tag, "")
    fig_eval.build_efficiency(ev, out, None, 60)
    man = out.manifest["efficiency"]
    assert Path(man["path"]).is_file() and Path(man["data_csv"]).is_file() and Path(man["data_csv_drone_count"]).is_file()
    rows = read_csv(man["data_csv"])
    recs = [r for r in ev.recs("ppo_x", 2) if r["source"] != 110]

    def get(metric):
        return next(r for r in rows if r["metric"] == metric and r["method"] == "ppo_x" and r["n_drones"] == "2")
    pl = [r["path_length_m"] for r in recs]
    assert int(get("path_length_m")["n_episodes"]) == len(recs) == 24 and float(get("path_length_m")["median"]) == pytest.approx(np.median(pl))
    assert float(get("path_length_m")["q75"]) == pytest.approx(np.quantile(pl, 0.75)) and float(get("path_length_m")["mean"]) == pytest.approx(np.mean(pl))
    assert float(get("masked_actions")["median"]) == pytest.approx(np.median([r["n_masked"] for r in recs]))
    fd = [r["first_detection_step"] for r in recs if r["first_detection_step"] is not None]
    assert int(get("first_detection_step")["n_valid"]) == len(fd) < len(recs) and float(get("first_detection_step")["median"]) == pytest.approx(np.median(fd))
    steps = [r["steps"] for r in recs if r["success"]]
    assert int(get("success_step")["n_valid"]) == len(steps) and float(get("success_step")["median"]) == pytest.approx(np.median(steps))
    assert float(get("flight_time_drone_steps")["median"]) == pytest.approx(2 * np.median(steps))
    assert float(get("success_step")["median_ci_lo"]) <= float(get("success_step")["median"]) <= float(get("success_step")["median_ci_hi"])
    dc = read_csv(man["data_csv_drone_count"])
    assert {(r["method"], r["n_drones"], r["n_drones_reference"]) for r in dc} == {("ppo_x", "2", "1")}                  # random has a single drone count
    assert {"train", "holdout", "all observable"} <= {r["scope"] for r in dc} and {float(r["ideal_step_ratio"]) for r in dc} == {0.5}
    assert man["drone_count_pairs"] == ["ppo_x: 2 vs 1"]
    d1, _ = build_eval_dir(tmp_path, "synthF1", configs=(("random", 1, 0.2),), steps=False)    # no multi-drone method: the figure still exists
    ev1 = fig_eval.load_eval_set(d1)
    out1 = fig_eval.Outputs(tmp_path / "o1", "p", ev1.tag, "")
    fig_eval.build_efficiency(ev1, out1, None, 20)
    assert Path(out1.manifest["efficiency"]["path"]).is_file() and out1.manifest["efficiency"]["drone_count_pairs"] == []


def test_compute_cost_from_records_and_skip_without_timing(tmp_path):
    d, _ = build_eval_dir(tmp_path, "synthC2", configs=SMALL, steps=False)
    ev = fig_eval.load_eval_set(d)
    for r in ev.records:
        r["step_ms_median"] = 5.0 + r["episode_id"] if r["method"] == "random" else 40.0
        r["wall_s"] = 0.01 * ev.horizon * (1 + r["episode_id"] % 2)
    out = fig_eval.Outputs(tmp_path / "o", "p", ev.tag, "")
    fig_eval.build_compute_cost(ev, out)
    rows = read_csv(out.manifest["compute_cost"]["data_csv"])
    a = next(r for r in rows if r["metric"] == "step_ms_median" and r["method"] == "random")
    ids = [r["episode_id"] for r in ev.recs("random", 1)]
    assert float(a["median_ms"]) == pytest.approx(np.median([5.0 + i for i in ids])) and int(a["n_episodes"]) == len(ids)
    w = next(r for r in rows if r["metric"] == "wall_ms_per_step" and r["method"] == "ppo_x")
    assert float(w["median_ms"]) == pytest.approx(np.median([10.0 * (1 + i % 2) for i in [r["episode_id"] for r in ev.recs("ppo_x", 2)]]))
    assert float(w["q25_ms"]) <= float(w["median_ms"]) <= float(w["q75_ms"])
    dn, _ = build_eval_dir(tmp_path, "synthC3", records=False, configs=SMALL, horizon=15)      # records rebuilt from step logs carry no timing
    evn = fig_eval.load_eval_set(dn)
    with pytest.raises(fig_eval.Skip, match="no timing"):
        fig_eval.build_compute_cost(evn, fig_eval.Outputs(tmp_path / "o2", "p", evn.tag, ""))


def test_figure_sources_slide_column_and_output_csv(full_run):
    rows = {r["output"]: r for r in read_csv(full_run["manifest"]["figure_sources"]["path"])}
    assert {k for k in rows} == set(OK_OUTPUTS) - {"figure_sources"}
    for k in ("table2_md", "table2_csv", "table2_png", "success_bars"):
        assert rows[k]["slide"] == "11", k
    assert rows["trajectories"]["slide"] == "12"
    assert rows["sensitivity_scale"]["slide"] == "13"
    for k in ("success_cdf", "map_error_vs_step", "paired_diff", "threshold_curve", "calibration", "start_type_bars", "error_at_steps", "efficiency", "compute_cost"):
        assert rows[k]["slide"] == "unassigned", k
    assert all(r["slide"] in ("11", "12", "13", "unassigned") and r["slide_basis"] for r in rows.values())
    assert rows["efficiency"]["output_csv_columns"].startswith("tag, metric, method, n_drones") and rows["table2_csv"]["output_csv_columns"].startswith("tag, mode, horizon")
    assert rows["table2_png"]["output_csv"] == full_run["manifest"]["table2_csv"]["path"]
    assert fig_eval.R_SLIDE_MAP["sensitivity"][0] == "13"


def test_cross_tag_sensitivity_is_assigned_to_its_slide(tmp_path, monkeypatch):
    a, _ = build_eval_dir(tmp_path, "tagF", mode="F", horizon=HORIZON, steps=False, configs=SMALL)
    b, _ = build_eval_dir(tmp_path, "tagT2", mode="T2", horizon=15, steps=False, configs=SMALL)
    monkeypatch.setattr(fig_eval, "make_for_set", lambda ev, out_dir, prefix, suffix="", *args: fig_eval.Outputs(out_dir, prefix, ev.tag, suffix))
    man = fig_eval.make_all([a, b], tmp_path / "o", "cmp", backend=FakeBackend(), obstacles=FakeObstacles())
    rows = {r["output"]: r for r in read_csv(man["figure_sources"]["path"])}
    assert rows["sensitivity"]["slide"] == "13" and rows["sensitivity"]["output_csv"].endswith("cmp_sensitivity.csv")


def test_slide_map_matches_the_plan_slide_table():
    plan = Path(config.CACHE_DIR).parent / "ICRS15_연구수행계획.md"
    if not plan.is_file():
        pytest.skip("plan file not available")
    titles = {}
    for ln in plan.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|", ln)
        if m and 1 <= int(m.group(1)) <= 15:
            titles.setdefault(int(m.group(1)), m.group(2))
    assert "비교표" in titles[11] and "궤적" in titles[12] and "민감도" in titles[13]                   # slide 11 = Table 2, 12 = Figure 7, 13 = Table 3
    assert set(fig_eval.R_PLAN_SLIDES) == {11, 12, 13} and {v[0] for v in fig_eval.R_SLIDE_MAP.values()} == {"11", "12", "13"}


def test_dpi_override_and_make_all_dpi(tmp_path, monkeypatch):
    d, _ = build_eval_dir(tmp_path, "synthD", configs=SMALL[1:])
    ev = fig_eval.load_eval_set(d)
    out = fig_eval.Outputs(tmp_path / "o", "p", ev.tag, "")
    shapes = []
    for dpi in (40, 80):
        with fig_eval._dpi_override(dpi):
            fig_eval.build_success_bars(ev, out, None)
        shapes.append(plt.imread(out.manifest["success_bars"]["path"]).shape)
    assert abs(shapes[1][0] - 2 * shapes[0][0]) <= 2 and abs(shapes[1][1] - 2 * shapes[0][1]) <= 2
    assert fig_eval.FIG_DPI == 40                                                                # restored (the fixture's value)
    seen = []
    monkeypatch.setattr(fig_eval, "_make_all", lambda *a, **k: seen.append(fig_eval.FIG_DPI) or {})
    fig_eval.make_all(d, tmp_path / "o3", "q", dpi=300)
    fig_eval.make_all(d, tmp_path / "o3", "q")
    assert seen == [300, 40] and fig_eval.FIG_DPI == 40 and fig_eval.R_DEFAULT_DPI == 150
