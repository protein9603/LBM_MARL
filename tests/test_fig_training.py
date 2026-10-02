"""Unit tests of scripts/fig_training.py on small synthetic run directories (no raw data, Agg backend; spec 7 items 1-8)."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.scripts import fig_training as ft

LOG_COLS = ["iteration", "env_steps", "success_ma", "strict_ma", "return_ma", "length_ma", "policy_loss", "value_loss", "entropy", "approx_kl",
            "clip_frac", "explained_var", "masked_share", "steps_per_s"]
EP_COLS = ["env_steps", "proc", "episode_idx", "source", "reflected", "start_type", "success", "success_strict", "truncated", "length", "ret",
           "final_error_m", "top_sigma_m", "entropy_drop"]
DECOMP = ["ret_info", "ret_time", "ret_terminal"]
SOURCES = [101, 102, 103, 108, 110, 112]            # 110 (unobservable) is in the records and must stay out of the heat map


@pytest.fixture(autouse=True)
def _fast_dpi(monkeypatch):
    monkeypatch.setattr(ft, "FIG_DPI", 50)


def _write(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _step_log(path, n_drones, steps, seed, new_keys, mode="F", map_xy=(465.0, 15.0)):
    rng = np.random.default_rng(seed)
    T, D = steps, n_drones
    xy = np.cumsum(rng.normal(0, 6, (T, D, 2)), axis=0) + np.array([420.0, 0.0])
    cov = np.tile(np.diag([400.0, 900.0]), (T, 3, 1, 1))
    arrs = {"drone_xy": xy, "y": rng.poisson(30, (T, D)), "action": rng.integers(0, 9, (T, D)), "applied": np.ones((T, D), bool),
            "gmm_w": np.tile([0.6, 0.3, 0.1], (T, 1)), "gmm_mu": np.tile([[470.0, 20.0], [440.0, -30.0], [0.0, 0.0]], (T, 1, 1)), "gmm_cov": cov,
            "gmm_mask": np.tile([1, 1, 0], (T, 1)).astype(bool), "map_xy": np.tile(map_xy, (T, 1)), "entropy": np.linspace(7, 5, T),
            "top_sigma": np.linspace(60, 20, T), "map_error": np.linspace(80, 30, T), "truth_xy": np.array([480.0, 10.0])}
    meta = {"source": 101, "frame": 450, "scale": 1.2, "seed": seed, "success": bool(seed % 2), "method": "ppo", "n_drones": D, "start_type": "plume"}
    if new_keys:
        arrs["frame"] = (450 + np.arange(T) if mode == "T2" else np.full(T, 450)).astype(int)
        arrs["reward"] = rng.normal(0, 1, T)
        arrs["calib_inside"] = np.ones(T, bool)
        arrs["start_xy"] = xy[0] - 5.0
        meta.update({"mode": mode, "max_steps": T})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrs, meta=json.dumps(meta))


def make_run(root, name, n_drones=1, mode="F", decomp=True, steps_logs=True, new_keys=True, iters=6, scale=1.0, n_ckpt=3, map_xy=(465.0, 15.0)):
    d = root / name
    d.mkdir(parents=True)
    horizon = 150 if mode == "T2" else 300
    (d / "config.json").write_text(json.dumps({"n_drones": n_drones, "truth_mode": mode, "max_steps": horizon, "args": {"ma_window": 5}}), encoding="utf-8")
    rows = [{"iteration": i + 1, "env_steps": (i + 1) * 300, "success_ma": 0.02 * i * scale, "strict_ma": 0.01 * i * scale, "return_ma": -8 + 0.3 * i * scale,
             "length_ma": horizon - 5 * i, "policy_loss": -0.01, "value_loss": 0.1 / (i + 1), "entropy": 2.1 - 0.02 * i, "approx_kl": 0.003, "clip_frac": 0.05,
             "explained_var": 0.1 * i, "masked_share": 0.05, "steps_per_s": 100} for i in range(iters)]
    _write(d / "train_log.csv", LOG_COLS, rows)
    rng = np.random.default_rng(3)
    eps = []
    for i in range(iters * 4):
        ln = int(rng.integers(40, horizon + 1))
        e = {"env_steps": (i // 4 + 1) * 300, "proc": i % 3, "episode_idx": i, "source": 101, "reflected": False, "start_type": "plume",
             "success": ln < horizon, "success_strict": False, "truncated": ln == horizon, "length": ln, "ret": -0.02 * ln + 1.0 + (10.0 if ln < horizon else -2.0),
             "final_error_m": 50.0, "top_sigma_m": 30.0, "entropy_drop": 0.3}
        e.update({"ret_info": 1.0, "ret_time": -0.02 * ln, "ret_terminal": e["ret"] - 1.0 + 0.02 * ln})
        eps.append(e)
    _write(d / "episodes.csv", EP_COLS + (DECOMP if decomp else []), eps)
    stems = [(f"ckpt_{(j + 1) * 600:08d}", (j + 1) * 600) for j in range(n_ckpt - 1)] + [("final", iters * 300)]      # n_ckpt = 3: ckpt_600, ckpt_1200, final
    for k, (stem, steps) in enumerate(stems):
        recs = [{"source": s, "success": str(bool((k + s) % 2)), "success_strict": "False", "final_error_m": 40.0 + 5 * k} for s in SOURCES for _ in range(2)]
        (d / "ckpt_eval").mkdir(exist_ok=True)
        _write(d / "ckpt_eval" / f"{stem}.csv", ["source", "success", "success_strict", "final_error_m"], recs)
        if steps_logs:
            for eid in (0, 1):
                _step_log(d / "ckpt_eval" / f"{stem}_steps" / f"ep{eid:04d}.npz", n_drones, 20, seed=10 * k + eid, new_keys=new_keys, mode=mode, map_xy=map_xy)
    return d


class FakeOm:
    """ObstacleMap-like: x0, y0, res, nx, ny, x, y and occ[ix, iy]."""

    def __init__(self):
        self.x0, self.y0, self.res, self.nx, self.ny = 300.0, -100.0, 2.0, 150, 100
        self.x = self.x0 + self.res * np.arange(self.nx)
        self.y = self.y0 + self.res * np.arange(self.ny)
        self.occ = np.zeros((self.nx, self.ny), bool)
        self.occ[20:40, 10:30] = True
        self.occ[90:110, 50:80] = True


class FakeBackend:
    """Slab backend stand-in that records the requested frames; the density grows with the frame index."""

    def __init__(self):
        self.frames = []
        self.grid = SimpleNamespace(x0=300.0, y0=-100.0, nx=60, ny=40, res=5.0)

    def slab(self, frame):
        self.frames.append(int(frame))
        yy, xx = np.meshgrid(self.grid.y0 + self.grid.res * (np.arange(40) + 0.5), self.grid.x0 + self.grid.res * (np.arange(60) + 0.5), indexing="ij")
        dens = np.exp(-((xx - 480.0) ** 2 + (yy - 10.0) ** 2) / (2 * 60.0 ** 2)) * (1.0 + frame / 600.0)
        return SimpleNamespace(density=np.stack([dens, 0.5 * dens])[:, None], sources=(101, 102), grid=self.grid, z_index=lambda z: 0)


def make_tag(root, name, mode, horizon, n_drones=1):
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"mode": mode, "max_steps": horizon}), encoding="utf-8")
    for m, rate in (("random", 0.0), ("lawnmower", 0.25), ("greedy_map", 0.5)):
        rows = []
        for s in (101, 102, 103, 108):                                   # 103 is a held-out source: not part of the training-source baselines
            for k in range(4):
                ok = k < 4 * rate
                rows.append({"method": m, "n_drones": n_drones, "episode_id": k, "source": s, "mode": mode, "success": str(ok), "steps": 40 + 20 * k if ok else horizon})
        _write(d / f"records_{m}_{n_drones}drones.csv", ["method", "n_drones", "episode_id", "source", "mode", "success", "steps"], rows)
    return d


def png_ok(path, min_w=300, min_h=150):
    img = mpimg.imread(path)
    assert img.shape[1] >= min_w and img.shape[0] >= min_h, (path, img.shape)
    assert float(np.std(img[..., :3])) > 0.02, f"{path} looks blank"


@pytest.fixture
def scene(monkeypatch):
    be = FakeBackend()
    monkeypatch.setattr(ft, "load_scene_helpers", lambda: (FakeOm(), be))
    return be


# ---------------------------------------------------------------------------------------------------------------- numeric helpers
def test_trailing_helpers_and_per_iteration():
    x = np.array([1.0, 3.0, np.nan, 5.0, 7.0])
    assert np.allclose(ft.trailing_mean(x, 2), [1.0, 2.0, 3.0, 5.0, 6.0])         # partial window at the start, NaN ignored
    length = np.array([100.0, 300.0, 50.0, 300.0, 80.0])
    succ = np.array([1, 0, 1, 0, 1])
    med = ft.trailing_success_median(length, succ, 3)
    assert np.allclose(med, [100.0, 100.0, 75.0, 50.0, 65.0])                        # median over the successes among the last 3 episodes
    assert np.isnan(ft.trailing_success_median(np.array([300.0]), np.array([0]), 3)[0])
    s = ft._per_iteration({"env_steps": np.array([10, 10, 20, 20, 20, 30]), "v": np.arange(6)})
    assert list(s["env_steps"]) == [10, 20, 30] and list(s["v"]) == [1, 4, 5]        # last episode of every iteration


def test_episode_frames_logged_and_derived():
    meta = {"frame": 410}
    assert list(ft.episode_frames({"frame": np.array([7, 8, 9])}, meta, 3)) == [7, 8, 9]            # newer logs: the logged frames
    assert list(ft.episode_frames({}, meta, 4, "F")) == [410] * 4                                      # Mode F: constant
    t2 = ft.episode_frames({}, meta, 3, "T2")
    assert list(t2) == [410, 411, 412]                                                                 # Mode T2: start frame + t, 1 frame per step
    assert list(ft.episode_frames({}, {"frame": 598, "mode": "T2"}, 4, "F")) == [598, 599, 599, 599]   # meta mode wins; clipped to the cached frames


def test_read_episodes_and_columns(tmp_path):
    d = make_run(tmp_path, "r", decomp=False)
    ep = ft.read_episodes(d)
    assert ep["success"].dtype == float and set(np.unique(ep["success"])) <= {0.0, 1.0}
    assert ft._num(ep, "ret_info") is None and ft._num(ep, "ret") is not None
    assert ft.run_mode(d) == ("F", 300) and ft.run_n_drones(d) == 1 and ft.ma_window(d) == 5


# ---------------------------------------------------------------------------------------------------------------- baselines
def test_baseline_info_mode_check_and_horizon(tmp_path):
    t_f, t_t2 = make_tag(tmp_path, "tagF", "F", 300), make_tag(tmp_path, "tagT2", "T2", 150)
    b = ft.baseline_info(t_f, 1, "F", 300)
    assert b["horizon"] == 300 and b["tag_mode"] == "F" and set(b["rates"]) == {"random", "lawnmower", "greedy_map"}
    assert b["rates"]["random"] == 0.0 and b["rates"]["lawnmower"] == 0.25 and b["rates"]["greedy_map"] == 0.5
    assert all(n == 12 for n in b["n"].values())                                   # 3 training sources x 4 episodes (held-out 103 excluded)
    mis = ft.baseline_info(t_f, 1, "T2", 150)                                       # Mode F tag for a Mode T2 run: no lines, with the reason
    assert mis["rates"] == {} and "Mode F" in mis["note"] and "Mode T2" in mis["note"]
    own = ft.baseline_info(t_t2, 1, "T2", 150)                                      # the records' own horizon (150), not 300
    assert own["horizon"] == 150 and own["rates"]["greedy_map"] == 0.5
    cut = ft.baseline_info(t_f, 1, "F", 50)                                         # shorter run horizon: only successes within 50 steps (k = 0 -> step 40)
    assert cut["horizon"] == 50 and cut["rates"]["greedy_map"] == pytest.approx(0.25)
    assert ft.baseline_rates(str(t_f), 1) == b["rates"]                             # legacy helper
    assert "not found" in ft.baseline_info(tmp_path / "nope", 1, "F", 300)["note"]


# ---------------------------------------------------------------------------------------------------------------- the one-call pipeline
def test_make_all_every_output_and_manifest(tmp_path, scene, monkeypatch):
    monkeypatch.setattr(ft, "TRAJ_FRACTIONS", (0.5, 1.0))                         # fewer snapshots: this test is about the outputs, not the panels
    runs = [make_run(tmp_path, "m1_s1"), make_run(tmp_path, "m2_s1", n_drones=2, scale=1.4)]
    tag = make_tag(tmp_path, "tagF", "F", 300)
    out = tmp_path / "figs"
    man = ft.make_all([str(r) for r in runs], "t", out, baseline_tag=str(tag), compare=[str(runs[0]), f"{runs[1]}"])      # n_drones: from the runs
    for name, e in man.items():
        assert set(e) == {"path", "status", "reason"} and e["status"] == "ok", (name, e)
        assert e["path"] and Path(e["path"]).exists() and Path(e["path"]).stat().st_size > 0, (name, e)
    expected = {"curves", "episode_stats", "diagnostics", "source_heatmap", "ckpt_eval", "reward_decomposition", "compare_drones", "manifest", "baselines_csv",
                "trajectories_m1_s1", "trajectories_m2_s1", "curves_csv", "episode_stats_csv", "source_heatmap_csv",
                "ckpt_eval_csv", "reward_decomposition_csv", "compare_drones_csv"}
    assert expected <= set(man)
    for name in ("curves", "episode_stats", "diagnostics", "source_heatmap", "ckpt_eval", "reward_decomposition", "compare_drones",
                 "trajectories_m1_s1", "trajectories_m2_s1"):
        png_ok(man[name]["path"])
    assert "baselines: tag tagF, Mode F" in man["curves"]["reason"] and "300-step" in man["curves"]["reason"]
    assert json.loads((out / "t_manifest.json").read_text(encoding="utf-8")) == man          # the manifest file is written next to the figures
    # the heat map table: 110 absent, 3 checkpoints x 5 observable sources, groups per metrics.GROUPS
    with (out / "t_source_heatmap.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {int(r["source"]) for r in rows} == {101, 102, 103, 108, 112} and 110 not in {int(r["source"]) for r in rows}
    assert {r["group"] for r in rows if r["source"] == "103"} == {"holdout"} and {r["group"] for r in rows if r["source"] == "101"} == {"train_open"}
    assert all(int(r["n"]) == 2 for r in rows)
    with (out / "t_compare.csv").open(encoding="utf-8") as fh:
        comp = list(csv.DictReader(fh))
    assert {int(r["n_drones"]) for r in comp} == {1, 2}


def test_old_run_skips_with_reasons_and_never_raises(tmp_path, scene):
    """A run that predates the reward decomposition and the sample-trajectory logs: those outputs are skipped with the reason, the rest is drawn."""
    old = make_run(tmp_path, "old", decomp=False, steps_logs=False)
    out = tmp_path / "figs"
    man = ft.make_all([str(old)], "o", out, baseline_tag=None, only=["curves", "source_heatmap", "reward_decomposition", "trajectories", "compare_drones"])
    assert man["reward_decomposition"]["status"] == "skipped" and man["reward_decomposition"]["path"] is None
    assert "run predates reward decomposition" in man["reward_decomposition"]["reason"]
    assert man["trajectories_old"]["status"] == "skipped" and "step logs" in man["trajectories_old"]["reason"]
    assert man["compare_drones"]["status"] == "skipped"                              # one run: no groups to compare
    assert "no baseline tag" in man["curves"]["reason"]
    for name in ("curves", "source_heatmap"):
        assert man[name]["status"] == "ok", (name, man[name])
        png_ok(man[name]["path"])
    assert not scene.frames                                                           # no step logs: the slab data was never touched
    assert (out / "o_manifest.json").exists()


def test_run_without_any_result_files_skips_everything(tmp_path, scene):
    empty = tmp_path / "empty_run"
    empty.mkdir()
    man = ft.make_all([str(empty)], "e", tmp_path / "figs")
    assert all(e["status"] == "skipped" for n, e in man.items() if n != "manifest"), man
    assert man["curves"]["reason"] and man["source_heatmap"]["reason"]


def test_make_all_baseline_mode_mismatch_is_noted_not_drawn(tmp_path, scene):
    run = make_run(tmp_path, "t2run", mode="T2")
    tag_f = make_tag(tmp_path, "tagF", "F", 300)
    man = ft.make_all([str(run)], "x", tmp_path / "figs", baseline_tag=str(tag_f), only=["curves"])
    assert man["curves"]["status"] == "ok" and "Mode F" in man["curves"]["reason"] and "Mode T2" in man["curves"]["reason"]
    assert "baseline lines skipped" in man["curves"]["reason"]
    man2 = ft.make_all([str(run)], "y", tmp_path / "figs", baseline_tag=str(make_tag(tmp_path, "tagT2", "T2", 150)), only=["curves"])
    assert set(man) == {"curves", "curves_csv", "manifest"}                              # only= limits the call
    assert "baselines: tag tagT2, Mode T2" in man2["curves"]["reason"] and "150-step" in man2["curves"]["reason"]


# ---------------------------------------------------------------------------------------------------------------- sample trajectories
def test_trajectories_mode_t2_old_logs_use_time_varying_frames(tmp_path):
    run = make_run(tmp_path, "t2old", mode="T2", new_keys=False)             # old logs: no frame / start_xy / mode keys
    be = FakeBackend()
    res = ft.fig_trajectories(run, "q", tmp_path, FakeOm(), be, fractions=(0.1, 1.0))
    png_ok(res["png"])
    # snapshots at 10 and 100 % of 20 steps -> steps 2 and 20: frame = start (450) + step index (0-based): 451 and 469
    assert sorted(set(be.frames)) == [451, 469]
    assert res["checkpoints"] == ["ckpt_00000600", "ckpt_00001200", "final"] and res["episode_id"] in (0, 1)


def test_trajectories_mode_f_one_frame_and_selection(tmp_path):
    run = make_run(tmp_path, "f2", n_drones=2)
    be = FakeBackend()
    res = ft.fig_trajectories(run, "q", tmp_path, FakeOm(), be, episode_id=1)
    assert res["steps"] == [[2, 6, 12, 20]] * 3                                  # default: 4 snapshots (10, 30, 60, 100 %) for each of 3 checkpoints
    assert set(be.frames) == {450} and res["episode_id"] == 1                  # frozen frame; the requested episode
    assert res["source"] == 101
    with pytest.raises(ft.Skip, match="not among"):
        ft.fig_trajectories(run, "q", tmp_path, FakeOm(), FakeBackend(), source=108)
    # only one checkpoint with logs: skipped, not an error
    for stem in ("ckpt_00001200_steps", "final_steps"):
        for f in (run / "ckpt_eval" / stem).glob("*.npz"):
            f.unlink()
        (run / "ckpt_eval" / stem).rmdir()
    with pytest.raises(ft.Skip, match="at least 2 checkpoints"):
        ft.fig_trajectories(run, "q", tmp_path, FakeOm(), FakeBackend())


def test_trajectory_log_missing_required_key_is_skipped(tmp_path):
    run = make_run(tmp_path, "bad")
    f = run / "ckpt_eval" / "final_steps" / "ep0000.npz"
    z = np.load(f)
    np.savez(f, **{k: z[k] for k in z.files if k != "gmm_cov"})
    with pytest.raises(ft.Skip, match="lacks keys"):
        ft.fig_trajectories(run, "q", tmp_path, FakeOm(), FakeBackend(), episode_id=0)


# ---------------------------------------------------------------------------------------------------------------- comparison, errors, CLI
def test_compare_auto_groups_by_drone_count_and_needs_two_groups(tmp_path, scene):
    r1, r2 = make_run(tmp_path, "a1"), make_run(tmp_path, "a2", n_drones=2)
    man = ft.make_all([str(r1), str(r2)], "c", tmp_path / "figs", only=["compare_drones"])
    assert man["compare_drones"]["status"] == "ok" and "drone counts" in man["compare_drones"]["reason"]
    with pytest.raises(ft.Skip, match="at least two"):
        ft.fig_compare([[r1]], "c", tmp_path)


def test_missing_run_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ft.make_all([str(tmp_path / "no_such_run")], "z", tmp_path / "figs")


def test_cli_prints_manifest_and_writes_json(tmp_path, scene, capsys):
    run = make_run(tmp_path, "cli_run")
    tag = make_tag(tmp_path, "tagF", "F", 300)
    res = ft.main(["--runs", str(run), "--prefix", "cli", "--baseline-tag", str(tag), "--out-dir", str(tmp_path / "o"),
                   "--only", "curves", "reward_decomposition"])
    txt = capsys.readouterr().out
    assert "[fig_training] curves" in txt and "reward_decomposition" in txt and "cli_manifest.json" in txt
    assert set(res["manifest"]) == {"curves", "curves_csv", "baselines_csv", "reward_decomposition", "reward_decomposition_csv", "manifest"}
    assert (tmp_path / "o" / "cli_manifest.json").exists() and res["manifest"]["curves"]["status"] == "ok"
    assert res["final_success_ma"] == [pytest.approx(0.1)] and len(res["ckpt_eval"]) == 3          # legacy return values kept
    assert config.T2_MAX_STEPS == 150                                                                  # the Mode T2 horizon assumed by the tests


# ================================================================================================================ review follow-up (D11)
@pytest.fixture
def figs(monkeypatch):
    """The figures the script draws, kept open until the end of the test: {png file name: Figure}."""
    got = {}

    def save(fig, path, dpi):
        got[Path(path).name] = fig
        fig.savefig(path, dpi=ft._dpi(dpi))

    monkeypatch.setattr(ft, "_save", save)
    yield got
    for fg in got.values():
        plt.close(fg)


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _legend_texts(fig):
    return [t.get_text() for lg in fig.legends for t in lg.get_texts()] + [t.get_text() for ax in fig.axes for t in (ax.get_legend().get_texts() if ax.get_legend() else [])]


# ---------------------------------------------------------------------------------------------------------------- baselines: Wilson CI
def test_baseline_info_carries_k_and_wilson_interval(tmp_path):
    tag = make_tag(tmp_path, "tagF", "F", 300)
    b = ft.baseline_info(tag, 1, "F", 300)
    assert b["k"] == {"random": 0, "lawnmower": 3, "greedy_map": 6} and b["n_drones"] == 1 and b["tag"] == "tagF"
    for m, (lo, hi) in b["ci"].items():
        assert lo <= b["rates"][m] <= hi and 0.0 <= lo and hi <= 1.0
    assert b["ci"]["random"][0] == 0.0 and b["ci"]["random"][1] == pytest.approx(0.2425, abs=1e-3)       # Wilson 0/12
    assert b["ci"]["greedy_map"] == pytest.approx((0.2539, 0.7461), abs=1e-3)                              # Wilson 6/12


def test_baseline_lines_show_ci_in_legend_and_baselines_csv(tmp_path, figs):
    run = make_run(tmp_path, "m1_s1")
    tag = make_tag(tmp_path, "tagF", "F", 300)
    out = tmp_path / "o"
    man = ft.make_all([str(run)], "q", out, baseline_tag=str(tag), only=["curves"])
    assert man["baselines_csv"]["status"] == "ok" and Path(man["baselines_csv"]["path"]).name == "q_baselines.csv"
    rows = _read_csv(man["baselines_csv"]["path"])
    assert [r["method"] for r in rows] == ["random", "lawnmower", "greedy_map"]
    assert {r["n_drones"] for r in rows} == {"1"} and {r["n"] for r in rows} == {"12"} and {r["truth_mode"] for r in rows} == {"F"}
    gm = rows[2]
    assert (gm["k"], float(gm["rate"])) == ("6", 0.5) and float(gm["ci95_lo"]) == pytest.approx(0.2539, abs=1e-3) and float(gm["ci95_hi"]) == pytest.approx(0.7461, abs=1e-3)
    legend = " | ".join(_legend_texts(figs["q_curves.png"]))
    assert "greedy_map: 50.0 %, 95 % CI 25.4-74.6, n=12" in legend and "Wilson 95 % CI" in legend
    # no baselines (no tag / other truth mode): the key is absent, not a skipped entry
    man2 = ft.make_all([str(run)], "r", out, baseline_tag=None, only=["curves"])
    assert "baselines_csv" not in man2 and not (out / "r_baselines.csv").exists()


# ---------------------------------------------------------------------------------------------------------------- --n-drones default
def test_n_drones_default_comes_from_the_runs_and_a_mismatch_warns(tmp_path, recwarn):
    run2 = make_run(tmp_path, "m2_s1", n_drones=2)
    tag = make_tag(tmp_path, "tag2", "F", 300, n_drones=2)
    out = tmp_path / "o"
    man = ft.make_all([str(run2)], "d", out, baseline_tag=str(tag), only=["curves"])            # n_drones omitted: 2-drone records
    assert "2-drone records" in man["curves"]["reason"] and "disagrees" not in man["curves"]["reason"]
    assert {r["n_drones"] for r in _read_csv(man["baselines_csv"]["path"])} == {"2"}
    assert not [w for w in recwarn if "disagrees" in str(w.message)]
    with pytest.warns(UserWarning, match="--n-drones 1 disagrees"):
        man2 = ft.make_all([str(run2)], "e", out, baseline_tag=str(tag), n_drones=1, only=["curves"])      # explicit value that disagrees
    assert "disagrees with the drone count of the runs (2 drones: m2_s1)" in man2["curves"]["reason"]
    assert "baselines_csv" not in man2                                                              # the tag has no 1-drone records
    with pytest.warns(UserWarning, match="disagrees"):                                              # also through the CLI
        ft.main(["--runs", str(run2), "--prefix", "f", "--baseline-tag", str(tag), "--n-drones", "1", "--out-dir", str(out), "--only", "curves"])
    ft.main(["--runs", str(run2), "--prefix", "g", "--baseline-tag", str(tag), "--out-dir", str(out), "--only", "curves"])      # default: no warning
    assert (out / "g_baselines.csv").exists()


# ---------------------------------------------------------------------------------------------------------------- runs of different drone counts are not pooled
def test_different_drone_counts_are_separate_groups_in_items_1_2_6(tmp_path, figs):
    runs = [make_run(tmp_path, "a1", n_drones=1, scale=1.0), make_run(tmp_path, "a2", n_drones=1, scale=1.2),
            make_run(tmp_path, "b1", n_drones=2, scale=2.0), make_run(tmp_path, "b2", n_drones=2, scale=2.4)]
    out = tmp_path / "o"
    man = ft.make_all([str(r) for r in runs], "g", out, only=["curves", "episode_stats", "reward_decomposition"])
    for k in ("curves", "episode_stats", "reward_decomposition"):
        assert "different drone counts are drawn as separate groups, not pooled (1 drone: a1, a2; 2 drones: b1, b2)" in man[k]["reason"]
    cur = _read_csv(man["curves_csv"]["path"])
    assert {"success_ma_mean_1d", "success_ma_sd_1d", "success_ma_mean_2d", "success_ma_sd_2d", "success_ma_seed_a1", "success_ma_seed_b2"} <= set(cur[0])
    assert "success_ma_mean" not in cur[0]                                                          # no pooled column
    row = cur[-1]                                                                                   # end of the common grid: every seed defined
    seeds_1d = [float(row[f"success_ma_seed_{n}"]) for n in ("a1", "a2")]
    seeds_2d = [float(row[f"success_ma_seed_{n}"]) for n in ("b1", "b2")]
    assert float(row["success_ma_mean_1d"]) == pytest.approx(np.mean(seeds_1d)) and float(row["success_ma_mean_2d"]) == pytest.approx(np.mean(seeds_2d))
    assert float(row["success_ma_mean_1d"]) != pytest.approx(float(row["success_ma_mean_2d"]))      # they differ: pooling would have blurred them
    assert float(row["success_ma_sd_2d"]) == pytest.approx(np.std(seeds_2d, ddof=1))
    assert any("2 drones: mean of 2 seeds" in t for t in _legend_texts(figs["g_curves.png"]))
    for key in ("episode_stats_csv", "reward_decomposition_csv"):
        head = set(_read_csv(man[key]["path"])[0])
        assert any(h.endswith("_mean_1d") for h in head) and any(h.endswith("_mean_2d") for h in head), (key, head)
    # the same runs of ONE drone count keep the plain column names
    man1 = ft.make_all([str(runs[0]), str(runs[1])], "h", out, only=["curves"])
    assert "success_ma_mean" in _read_csv(man1["curves_csv"]["path"])[0] and "different drone counts" not in man1["curves"]["reason"]


# ---------------------------------------------------------------------------------------------------------------- labels use the run ma_window
def test_moving_average_labels_follow_ma_window(tmp_path, figs):
    run = make_run(tmp_path, "w5")                                                   # config.json: args.ma_window = 5
    r2 = make_run(tmp_path, "w5b", n_drones=2)
    cfg = json.loads((r2 / "config.json").read_text(encoding="utf-8"))
    cfg["args"]["ma_window"] = 20
    (r2 / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    ft.make_all([str(run)], "m", tmp_path / "o", only=["curves", "episode_stats", "reward_decomposition"])
    titles = [ax.get_title() for ax in figs["m_curves.png"].axes] + [ax.get_ylabel() for ax in figs["m_curves.png"].axes]
    assert any("5-episode mean" in t for t in titles) and not any("100-episode" in t for t in titles)
    assert "trailing windows of 5 episodes" in figs["m_returns.png"]._suptitle.get_text()
    assert "moving averages of 5 episodes" in figs["m_reward_decomposition.png"]._suptitle.get_text()
    ft.make_all([str(run), str(r2)], "n", tmp_path / "o", only=["curves", "compare_drones"])                # different windows: both values are shown
    assert any("5/20-episode mean" in ax.get_title() for ax in figs["n_curves.png"].axes)
    assert any("5/20-episode mean" in ax.get_ylabel() for ax in figs["n_compare.png"].axes)
    assert "100-episode" not in " ".join(ax.get_ylabel() + ax.get_title() for ax in figs["n_compare.png"].axes)


# ---------------------------------------------------------------------------------------------------------------- --only is validated
def test_only_with_a_misspelled_name_raises_a_clear_error(tmp_path):
    run = make_run(tmp_path, "r1")
    out = tmp_path / "o"
    with pytest.raises(ValueError) as ei:
        ft.make_all([str(run)], "x", out, only=["curvs", "reward_decomposition"])
    msg = str(ei.value)
    assert "'curvs'" in msg and "did you mean 'curves'" in msg and "valid names:" in msg and "compare_drones" in msg and "trajectories_r1" in msg
    assert "reward_decomposition" not in msg.split("valid names")[0]                          # only the bad name is blamed
    assert not out.exists()                                                                      # nothing was drawn or created
    with pytest.raises(ValueError, match="'trajectories_nope'"):
        ft.make_all([str(run)], "x", out, only=["trajectories_nope"])
    assert ft.validate_only("curves") == ["curves"] and ft.validate_only(["trajectories_r1", "compare_drones"], ["r1"]) == ["trajectories_r1", "compare_drones"]
    man = ft.make_all([str(run)], "x", out, only="diagnostics")                                  # a single string is one name, not characters
    assert set(man) == {"diagnostics", "manifest"}
    with pytest.raises(SystemExit) as se:                                                       # CLI: a parser error, exit status 2
        ft.main(["--runs", str(run), "--out-dir", str(tmp_path / "o2"), "--only", "curvs"])
    assert se.value.code == 2 and not (tmp_path / "o2").exists()


# ---------------------------------------------------------------------------------------------------------------- logs that are still being written
def _truncate_last_row(path, keep_chars):
    """Cut the file inside its last row (as a writer that is in the middle of the row): the final line break and the end of the row are gone."""
    text = path.read_text(encoding="utf-8")
    body = text.rstrip("\r\n")
    last_start = body.rfind("\n") + 1
    path.write_text(text[:last_start + keep_chars], encoding="utf-8")


def test_read_csv_rows_drops_an_incomplete_last_row(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b,c\n1,2,3\n4,5,6\n7,8", encoding="utf-8")                               # last row: no line break, too few fields
    header, rows = ft.read_csv_rows(f)
    assert header == ["a", "b", "c"] and [r["a"] for r in rows] == ["1", "4"]
    f.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9.", encoding="utf-8")                           # all fields present but the number is cut (no line break): dropped too
    assert [r["a"] for r in ft.read_csv_rows(f)[1]] == ["1", "4"]
    f.write_text("a,b,c\n1,2,3\n4,,6\n", encoding="utf-8")                                    # complete file, empty field in the middle: kept
    assert [r["b"] for r in ft.read_csv_rows(f)[1]] == ["2", ""]
    f.write_text("a,b,c\n", encoding="utf-8")
    assert ft.read_csv_rows(f) == (["a", "b", "c"], [])
    f.write_text("", encoding="utf-8")
    assert ft.read_csv_rows(f) == ([], [])


def test_truncated_numbers_in_live_logs_do_not_raise(tmp_path, scene, monkeypatch):
    monkeypatch.setattr(ft, "TRAJ_FRACTIONS", (0.5, 1.0))                                  # fewer snapshots: the subject here is the CSV reading
    run = make_run(tmp_path, "live", iters=6)
    n_log = len(ft.read_log(run)["env_steps"])
    _truncate_last_row(run / "train_log.csv", 9)                                              # the last row ends inside a number
    log = ft.read_log(run)
    assert len(log["env_steps"]) == n_log - 1 and ft.has_log_rows(run)                           # the incomplete row is dropped
    n_ep = len(ft.read_episodes(run)["env_steps"])
    _truncate_last_row(run / "episodes.csv", 25)
    ep = ft.read_episodes(run)
    assert len(ep["env_steps"]) == n_ep - 1 and ep["ret"].dtype == float
    # a garbled number in the middle of a numeric column: that field is NaN, the column stays numeric
    text = (run / "episodes.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    cols = text[0].strip().split(",")
    fields = text[2].rstrip("\r\n").split(",")
    fields[cols.index("ret")] = "-3.5e"
    text[2] = ",".join(fields) + "\r\n"
    (run / "episodes.csv").write_text("".join(text), encoding="utf-8")
    ep = ft.read_episodes(run)
    assert ep["ret"].dtype == float and np.isnan(ep["ret"][1]) and np.isfinite(ep["ret"][0]) and ep["start_type"].dtype == object
    # the quick-evaluation CSV and the baseline records are tolerated as well
    _truncate_last_row(run / "ckpt_eval" / "final.csv", 3)
    assert all(r["n"] >= 11 for r in ft.read_ckpt_eval(run))
    tag = make_tag(tmp_path, "tagF", "F", 300)
    _truncate_last_row(tag / "records_random_1drones.csv", 14)
    b = ft.baseline_info(tag, 1, "F", 300)
    assert b["n"]["random"] == 11 and b["n"]["greedy_map"] == 12
    man = ft.make_all([str(run)], "l", tmp_path / "o", baseline_tag=str(tag))                      # the whole pipeline on the half-written run
    assert all(e["status"] == "ok" for k, e in man.items() if k != "compare_drones"), man                 # (one run: nothing to compare)
    f = tag / "records_greedy_map_1drones.csv"                                                    # a garbled number inside a baseline record: that row is skipped
    txt = f.read_text(encoding="utf-8")
    assert ",40\n" in txt
    f.write_text(txt.replace(",40\n", ",4x\n", 1), encoding="utf-8")
    assert ft.baseline_info(tag, 1, "F", 300)["n"]["greedy_map"] == 11


# ---------------------------------------------------------------------------------------------------------------- heat-map width
def test_source_heatmap_is_capped_in_width_and_wraps_panels(tmp_path):
    ncols, nrows, (w, h) = ft.heatmap_layout(3, 11, 12)
    assert w <= ft.HEATMAP_MAX_WIDTH_IN and ncols >= 1 and nrows == -(-3 // ncols) and nrows >= 2           # 3 seeds x 11 checkpoints wrap into rows
    assert ft.heatmap_layout(1, 3, 12)[0] == 1 and ft.heatmap_layout(1, 3, 12)[2][0] >= ft.HEATMAP_MIN_WIDTH_IN
    assert ft.heatmap_layout(3, 4, 12)[0] == 3                                                         # a few checkpoints: side by side as before
    assert ft.heatmap_layout(1, 60, 12)[2][0] <= ft.HEATMAP_MAX_WIDTH_IN                              # extreme checkpoint counts do not widen the figure either
    runs = [make_run(tmp_path, f"s{i}", iters=25, n_ckpt=11, steps_logs=False) for i in range(3)]
    res = ft.fig_source_heatmap(runs, "hm", tmp_path)
    img = mpimg.imread(res["png"])
    assert img.shape[1] <= ft.HEATMAP_MAX_WIDTH_IN * 50 + 2 and img.shape[0] > img.shape[1] * 0.5       # dpi 50 (fixture): at most 14 in wide
    rows = _read_csv(res["csv"])
    assert {r["run"] for r in rows} == {"s0", "s1", "s2"} and len({r["ckpt"] for r in rows}) == 11


# ---------------------------------------------------------------------------------------------------------------- --dpi
def test_dpi_option_scales_the_figures(tmp_path):
    run = make_run(tmp_path, "d1")
    out = tmp_path / "o"
    ft.make_all([str(run)], "lo", out, only=["diagnostics"])                                     # default: FIG_DPI (50 under the test fixture)
    ft.make_all([str(run)], "hi", out, only=["diagnostics"], dpi=100)
    lo, hi = mpimg.imread(out / "lo_diagnostics.png"), mpimg.imread(out / "hi_diagnostics.png")
    assert hi.shape[1] == pytest.approx(2 * lo.shape[1], abs=2) and hi.shape[0] == pytest.approx(2 * lo.shape[0], abs=2)
    ft.main(["--runs", str(run), "--prefix", "cli", "--out-dir", str(out), "--only", "diagnostics", "--dpi", "75"])
    assert mpimg.imread(out / "cli_diagnostics.png").shape[1] == pytest.approx(1.5 * lo.shape[1], abs=2)
    with pytest.raises(ValueError, match="dpi"):
        ft.make_all([str(run)], "bad", out, only=["diagnostics"], dpi=0)
    with pytest.raises(SystemExit):
        ft.main(["--runs", str(run), "--out-dir", str(out), "--only", "diagnostics", "--dpi", "-5"])
    assert ft.FIG_DPI == 50                                                                       # monkeypatched by the autouse fixture; the module default is 150


# ---------------------------------------------------------------------------------------------------------------- sample trajectories: legend, paths, window, rules
FEW = (0.5, 1.0)                                       # two snapshots per checkpoint keep these tests fast


def test_trajectory_figure_has_one_legend_entry_per_drone_and_paths_under_the_dots(tmp_path, figs):
    run = make_run(tmp_path, "tr", n_drones=2)
    res = ft.fig_trajectories(run, "q", tmp_path, FakeOm(), FakeBackend(), episode_id=1, fractions=FEW)
    fig = figs["q_tr_trajectories.png"]
    texts = _legend_texts(fig)
    for k in (1, 2):
        assert f"drone {k}: path, position" in texts and f"drone {k}: start" in texts
    assert "drone 3: path, position" not in texts
    ax = fig.axes[0]
    paths = [ln for ln in ax.lines if ln.get_linewidth() == ft.TRAJ_PATH_LW]
    assert {ln.get_color() for ln in paths} == {"tab:blue", "tab:green"}                           # every drone has its own path colour
    dots = [c for c in ax.collections if type(c).__name__ == "PathCollection"]
    assert dots and all(c.get_zorder() > ln.get_zorder() for c in dots for ln in paths)               # the paths lie under the dots
    sup = " ".join(fig._suptitle.get_text().split())                                                 # the selection rules are in the title (wrapped to the width)
    assert "initial = first periodic checkpoint" in sup and "not the untrained policy" in sup and "late = last checkpoint" in sup
    assert "episode rule:" in sup and "ckpt_00000600" in sup and "final" in sup
    assert any("initial policy: ckpt_00000600" in a.get_ylabel() for a in fig.axes) and res["checkpoints"][0] == "ckpt_00000600"
    one = make_run(tmp_path, "tr1", n_drones=1)
    ft.fig_trajectories(one, "q", tmp_path, FakeOm(), FakeBackend(), episode_id=0, fractions=FEW)
    assert "drone 2: path, position" not in _legend_texts(figs["q_tr1_trajectories.png"])
    assert ft._select_checkpoints(ft.ckpt_step_logs(run))[0][0] == "initial"


def test_trajectory_window_includes_a_nearby_map_and_arrows_to_a_far_one(tmp_path, figs):
    base = ft.fig_trajectories(make_run(tmp_path, "m0"), "q", tmp_path, FakeOm(), FakeBackend(), episode_id=0, fractions=FEW)["window"]
    near = ft.fig_trajectories(make_run(tmp_path, "m1", map_xy=(700.0, 10.0)), "q", tmp_path, FakeOm(), FakeBackend(), episode_id=0, fractions=FEW)["window"]
    far = ft.fig_trajectories(make_run(tmp_path, "m2", map_xy=(2500.0, 400.0)), "q", tmp_path, FakeOm(), FakeBackend(), episode_id=0, fractions=FEW)["window"]
    assert near[2] >= 700.0 + ft.TRAJ_MARGIN_M - 1e-6 and near[2] > base[2] + 100.0                  # a MAP within reach widens the window to contain it
    assert far[2] < 1000.0 and far == pytest.approx(base)                                              # a MAP 2 km away does not squash the paths
    arrows = [t.get_text() for ax in figs["q_m2_trajectories.png"].axes for t in ax.texts if "outside" in t.get_text()]
    assert arrows and all(a.startswith("MAP ") and a.endswith(" m outside") for a in arrows)             # an arrow with the distance instead
    assert not [t for ax in figs["q_m1_trajectories.png"].axes for t in ax.texts if "outside" in t.get_text()]
    assert ft._map_in_reach(np.array([[0.0, 0.0], [10.0, 10.0]]), np.array([[100.0, 5.0], [900.0, 0.0]])).tolist() == [[100.0, 5.0]]


def test_unreadable_step_log_is_avoided_or_skipped(tmp_path):
    run = make_run(tmp_path, "u")
    (run / "ckpt_eval" / "final_steps" / "ep0000.npz").write_bytes(b"PK-not-an-archive")           # being written / truncated
    res = ft.fig_trajectories(run, "q", tmp_path, FakeOm(), FakeBackend(), fractions=FEW)
    assert res["episode_id"] == 1                                                                    # the unreadable candidate is dropped
    (run / "ckpt_eval" / "ckpt_00001200_steps" / "ep0001.npz").write_bytes(b"x")
    with pytest.raises(ft.Skip, match="unreadable"):
        ft.fig_trajectories(run, "q", tmp_path, FakeOm(), FakeBackend())


# ---------------------------------------------------------------------------------------------------------------- curve CSVs: sd (ddof = 1) and per-seed columns
def test_nanstd0_is_the_sample_sd_and_nan_for_one_value():
    a = np.array([[1.0, 2.0, np.nan, 4.0], [3.0, np.nan, np.nan, np.nan], [5.0, 6.0, np.nan, 8.0]])
    sd = ft._nanstd0(a)
    assert sd[0] == pytest.approx(2.0) and sd[1] == pytest.approx(np.std([2.0, 6.0], ddof=1)) and np.isnan(sd[2]) and sd[3] == pytest.approx(np.std([4.0, 8.0], ddof=1))
    assert np.isnan(ft._nanstd0(np.array([[1.0, 2.0]]))).all()                                      # a single seed has no spread


def test_curve_csvs_have_mean_sd_and_one_column_per_seed(tmp_path):
    runs = [make_run(tmp_path, f"seed{i}", scale=1.0 + 0.5 * i) for i in range(3)]
    man = ft.make_all([str(r) for r in runs], "c", tmp_path / "o", only=["curves", "episode_stats", "reward_decomposition"])
    for csv_key, keys in (("curves_csv", ("success_ma", "return_ma")), ("episode_stats_csv", ("ret", "length")), ("reward_decomposition_csv", ("ret_info", "ret_terminal", "ret"))):
        rows = _read_csv(man[csv_key]["path"])
        n_checked = 0
        for key in keys:
            assert {f"{key}_mean", f"{key}_sd", f"{key}_seed_seed0", f"{key}_seed_seed1", f"{key}_seed_seed2"} <= set(rows[0]), (csv_key, key)
            for r in rows:
                v = np.array([float(r[f"{key}_seed_seed{i}"]) for i in range(3)])
                if np.isfinite(v).all():
                    n_checked += 1
                    assert float(r[f"{key}_mean"]) == pytest.approx(v.mean()) and float(r[f"{key}_sd"]) == pytest.approx(np.std(v, ddof=1)), (csv_key, key)
        assert n_checked > 20, csv_key


def test_compare_csv_has_sd_seed_count_and_per_run_columns(tmp_path):
    g1 = [make_run(tmp_path, "p1", scale=1.0), make_run(tmp_path, "p2", scale=1.5)]
    g2 = [make_run(tmp_path, "q1", n_drones=2, scale=2.0), make_run(tmp_path, "q2", n_drones=2, scale=3.0)]
    res = ft.fig_compare([g1, g2], "k", tmp_path)
    rows = _read_csv(res["csv"])
    assert {"success_ma_sd", "return_ma_sd", "n_seeds", "success_ma_seed_p1", "return_ma_seed_q2"} <= set(rows[0]) and {r["n_seeds"] for r in rows} == {"2"}
    r = [x for x in rows if x["n_drones"] == "2"][-1]
    seeds = [float(r["success_ma_seed_q1"]), float(r["success_ma_seed_q2"])]
    assert float(r["success_ma_sd"]) == pytest.approx(np.std(seeds, ddof=1)) and r["success_ma_seed_p1"] == ""      # the other group runs stay empty


def test_axis_unit_and_unique_names():
    assert ft._steps_unit(5.0e4) == (1.0e3, "k") and ft._steps_unit(2.0e6) == (1.0e6, "M") and ft._steps_unit(ft.STEPS_UNIT_SWITCH)[1] == "M"
    assert ft._unique_names([Path("a/x"), Path("b/x"), Path("c/y")]) == ["x#1", "x#2", "y"]


def test_axis_label_of_a_short_run_is_in_thousands(tmp_path, figs):
    run = make_run(tmp_path, "short")
    ft.make_all([str(run)], "u", tmp_path / "o", only=["curves"])
    assert {ax.get_xlabel() for ax in figs["u_curves.png"].axes} == {"team steps [k]"}
