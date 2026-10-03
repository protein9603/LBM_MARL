"""Evaluation-result tables and figures of docs/training_evaluation_spec.md sections 9 and 10 (spec 10 items 1-11, spec 9 items 1-12).

Usage: python -m srcloc_env.scripts.fig_eval --tag t2_4_v3 [--tag t2mode_prelim ...] --out-dir <dir> --prefix <name>
           [--compare-tags TAG ...] [--reference-method gmm_infotaxis] [--methods random lawnmower ...] [--traj-method ppo_m2]
           [--episodes-per-source K] [--n-boot N] [--dpi 150|300]
Reads ONE evaluation directory per tag (``<CACHE_DIR>/eval/<tag>/``, or an explicit directory): ``records_<method>_<n>drones.csv``
(run_eval.load_records), ``episodes.csv``, ``summary.json`` (mode / horizon) and the step logs ``steps/*.npz`` (or the paths in the records'
``step_log`` column).  Configurations that have step logs but no records yet (a batch still running) are rebuilt from their logs
(success flags, first success step, errors, declared stop ...), so partial directories can be previewed read-only.  Older step logs lack
``frame`` / ``reward`` / ``calib_inside`` / ``start_xy``: every consumer falls back or skips, nothing crashes.

Outputs (``<out-dir>/<prefix>[_<tag>]_<name>.<ext>``; with several tags every tag gets its own set, never mixing Mode F and Mode T2):
  table2.md / table2.csv / table2.png   Table 2: method x drones x source group (success rate with Wilson 95 % CI, strict success,
                                        success-step median with bootstrap CI, censored median, final error median / p90, and the declared
                                        stop: declared success rate over ALL episodes, correct share and overconfident share among the
                                        declaring episodes, declaration delay median); statistics come from eval/metrics.aggregate
  success_bars.png                      success rate per source group with CI error bars (methods x drone counts)
  success_cdf.png                       CDF of the success step per method (1 drone solid, 2 drones dashed; failures are censored at the horizon)
  map_error_vs_step.png                 MAP error versus step, one panel per method (median and interquartile band per drone count, log y; from step logs)
  source_heatmap.png                    source x method success heatmap (110 grey = unobservable)
  paired_diff.png                       paired differences of every method minus the reference method on the common episode ids (per source and per
                                        group train_open / train_other / holdout / train), the same-method n-versus-1-drone pairs, and the n/1
                                        success-step ratio and total flight time (drones x steps) of those pairs
  trajectories.png                      four representative episodes chosen from the records (open success, trapped success, failure, multi-drone
                                        episode of the largest drone count and not the episode of panel a); paths coloured by step, detections ringed
  entropy_curves.png                    belief entropy H_t / H_0 versus step (median over episodes; second row zooms on the baselines when they stay near H_0)
  calibration.png                       fraction of episodes whose truth lies inside the 2-sigma ellipse of the top GMM component versus step
  threshold_curve.png                   success rate versus MAP-error threshold 20..100 m (top sigma < 30 m; from step logs)
  start_type.md / .csv / .png           success stratified by start type (plume / random)
  error_at_steps.png / .csv             median MAP error at the steps 50 / 100 / 150 / 300 clipped to the horizon (spec 9.8; from step logs)
  efficiency.png / .csv                 path length, masked actions, first detection step, and the drone-count effect: n/1 success-step ratio
                                        and total flight time (spec 9.5, 9.9, 9.11; ``efficiency_drone_count.csv`` holds the drone-count rows)
  compute_cost.png / .csv               compute cost per step by method (records' step_ms_median = environment step; episode wall time / horizon)
  sensitivity_scale.png / .csv          within-tag sensitivity: success rate by sensor-scale bin (<= 0.6, 0.6-1.7, >= 1.7 of the records' scale column)
  sensitivity.png / .csv                success rate per tag (only with several tags: ``compare_tags`` / repeated ``--tag``)
  figure_sources.csv                    numbers-to-source table (spec 10.11): one row per produced output with the slide that uses it (plan chapter 7,
                                        'unassigned' when none), its data files and columns, and the columns of its own CSV
  manifest.json                         {output_name: {path, status ok|skipped, reason}}
Curve data behind each figure is written next to it as ``<name>.csv``.  Missing inputs skip only the affected outputs (reason in the
manifest); programming errors raise.  Figures: Agg backend, dpi ``--dpi`` (default 150; config.FIG_DPI_FINAL = 300 for final figures), English
labels.  The horizon is taken from summary.json (max_steps; default 300 in Mode F and 150 in Mode T2); failures are censored at that horizon.
Step-log figure titles state the episodes actually drawn ('12 of 130 episodes per configuration' when --episodes-per-source subsamples).

Figure-layout constants are defined at the top of this module (config.py is not part of this tool); every evaluation threshold (success sigma /
error, strict thresholds, horizons, source groups, bootstrap count) comes from srcloc_env.config and srcloc_env.eval.metrics.  New constants of
this tool carry the prefix ``R_`` (to be moved into config.py by the maintainer).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import re
import sys
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.colors import LogNorm    # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Ellipse, Patch   # noqa: E402

from srcloc_env import config            # noqa: E402
from srcloc_env.eval import metrics      # noqa: E402

# ------------------------------------------------------------------------------------------------ module constants
FIG_DPI = 150                                         # figure resolution (task convention; config.FIG_DPI_FINAL is the 300 dpi talk resolution)
CALIB_SIGMAS = 2.0                                    # calibration ellipse radius [sigma] (spec 9 item 10)
CALIB_NOMINAL = 1.0 - math.exp(-0.5 * CALIB_SIGMAS ** 2)   # P(chi2_2 <= 4) = 0.865: nominal 2-D coverage of the 2-sigma ellipse
THRESHOLD_GRID_M = tuple(range(20, 101, 5))           # MAP-error thresholds of the success-versus-threshold curve [m] (spec 9 item 1)
TRAPPED_SOURCES = (102, 104, 106)                     # trapped / regression sources of the train_other group (spec section 8)
OPEN_SOURCES = tuple(config.T1_3_OPEN_SOURCES)        # open sources used to pick the 'open success' trajectory (109 included)
BOOTSTRAP_SEED = 0                                    # seed of the bootstrap of the success-difference CI
TRAJ_MARGIN_M = 90.0                                  # margin around paths and truth of a trajectory panel [m]
TRAJ_MIN_SIDE_M = 300.0                               # minimum side of the (square) window of a trajectory panel [m]
HEATMAP_CMAP = "Blues"                                # one-hue sequential map of the source heatmap
PAIRED_MAX_METHODS = 8                                # at most this many methods are compared with the reference in the forest figure
CDF_GROUPS = ("all_observable", "train_open", "train_other", "holdout")
BAR_GROUPS = ("train_open", "train_other", "holdout", "train", "all_observable")
TABLE_GROUPS = ("train_open", "train_other", "holdout", "train", "all_observable")
CURVE_GROUP = "all_observable"                        # source group of the step-log curves (MAP error, entropy, calibration, threshold)
START_TYPES = ("plume", "random")
MIN_CURVE_EPISODES = 2                                # steps with fewer finite values are not drawn

# constants added with the review fixes (prefix R_: to be moved into config.py)
R_SCALE_BIN_EDGES = (0.6, 1.7)                        # sensor-scale bins of the within-tag sensitivity: <= 0.6, 0.6-1.7, >= 1.7 (task decision 2026-10-02)
R_ERROR_AT_STEPS = (50, 100, 150, 300)                # step budgets of the error-versus-time table (spec 9.8), clipped to the horizon
R_ERROR_AT_STEP_GROUPS = ("all_observable", "train", "holdout")   # source groups of the error-at-steps table
R_BOX_WHISKER_PCT = (5, 95)                           # whiskers of the efficiency box plots [percentile]
R_DEFAULT_DPI = FIG_DPI                               # --dpi default
R_DETECTION_COLOUR = "#e8590c"                        # ring colour of the detection steps on the trajectory paths
R_DETECTION_MARKER_AREA = 34.0                        # scatter area [pt^2] of the detection rings
R_PLAN_SLIDES = {11: "Results 1: comparison table", 12: "Results 2: learning curves and trajectories", 13: "(optional) sensitivity / ablation"}
# slide that uses an output (ICRS15_연구수행계획.md chapter 7 slide table, column 'figure / table source'): output -> (slide, basis); anything else is 'unassigned'
R_SLIDE_MAP = {
    "table2_md": ("11", "slide 11 cites Table 2 (S2-S4)"), "table2_csv": ("11", "slide 11 cites Table 2 (S2-S4)"),
    "table2_png": ("11", "slide 11 cites Table 2 (S2-S4)"),
    "success_bars": ("11", "graphical form of Table 2: slide 11 content = 1 vs 2 drones, train / holdout, CI included (spec 10.2)"),
    "trajectories": ("12", "slide 12 cites Figure 7 = spec 10.7 trajectory panel (2-drone trajectory + counts + belief ellipses)"),
    "sensitivity": ("13", "slide 13 cites Table 3 (stretch): Mode T2 vs F, scale"),
    "sensitivity_scale": ("13", "slide 13 cites Table 3 (stretch): sensor-scale sensitivity (scale {0.3, 1, 3} bins)"),
}
R_SLIDE_DEFAULT = ("unassigned", "not cited by the plan's slide table (backup / appendix / paper candidate)")

CANONICAL_METHODS = ("random", "lawnmower", "greedy_map", "gmm_infotaxis")
VARIANT_METHODS = ("lawnmower_band", "lawnmower_alongwind")
VERIFICATION_METHODS = ("oracle_loiter",)             # verification only: drawn dashed / hatched, never a competing method
FIXED_COLOURS = {"random": "#7f7f7f", "lawnmower": "#8c564b", "greedy_map": "#17becf", "gmm_infotaxis": "#e377c2",
                 "lawnmower_band": "#bcbd22", "lawnmower_alongwind": "#c49c94", "oracle_loiter": "#000000"}   # fig_training BASE_COLORS + variants
PPO_COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#393b79")           # PPO aliases, in order of appearance
DRONE_COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}                                    # drone count colours of the forest plot
DRONE_LINESTYLES = {1: "-", 2: "--", 3: ":"}
DRONE_MARKERS = {1: "o", 2: "s", 3: "^"}

STEP_LOG_RE = re.compile(r"^(?P<method>.+)_(?P<n>\d+)drones_ep(?P<id>\d+)\.npz$")
_READ_ERRORS = (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile)

GROUP_ORDER_SOURCES = tuple(s for g in ("train_open", "train_other", "holdout", "unobservable") for s in metrics.GROUPS[g])
GROUP_SHORT = {"train_open": "train\nopen", "train_other": "train\nother", "holdout": "holdout", "train": "train\n(9)",
               "all_observable": "all obs.\n(12)", "unobservable": "unobs.\n(110)"}


def group_label(name: str) -> str:
    srcs = metrics.GROUPS[name]
    return f"{name} ({', '.join(str(s) for s in srcs)})" if len(srcs) <= 6 else f"{name} ({len(srcs)} sources)"


def default_horizon(mode: str) -> int:
    """Episode horizon of a truth mode: config.T2_MAX_STEPS in Mode T2, config.MAX_EPISODE_STEPS in Mode F."""
    return int(config.T2_MAX_STEPS if str(mode) == "T2" else config.MAX_EPISODE_STEPS)


class Skip(Exception):
    """Raised by an output builder whose inputs are missing: the output is recorded as 'skipped' with the message as the reason."""


# ------------------------------------------------------------------------------------------------ method ordering and styles
def order_methods(methods: Iterable[str]) -> list[str]:
    """Canonical baselines first, then lawnmower variants, then everything else (PPO aliases) alphabetically, verification methods last."""
    ms = sorted(set(methods))
    head = [m for m in CANONICAL_METHODS if m in ms]
    var = [m for m in VARIANT_METHODS if m in ms]
    ver = [m for m in VERIFICATION_METHODS if m in ms]
    rest = [m for m in ms if m not in head + var + ver]
    return head + var + rest + ver


def method_colour(method: str, ordered: Sequence[str]) -> str:
    if method in FIXED_COLOURS:
        return FIXED_COLOURS[method]
    others = [m for m in ordered if m not in FIXED_COLOURS]
    return PPO_COLOURS[others.index(method) % len(PPO_COLOURS)] if method in others else "#444444"


def method_label(method: str) -> str:
    return f"{method} (verification only)" if method in VERIFICATION_METHODS else method


def cfg_label(method: str, n: int) -> str:
    return f"{method} ({n})"


def wrap_method(method: str) -> str:
    """Method name broken at its underscores (and 'verification only' shortened) so that tick labels of many methods never overlap."""
    return method.replace("_", "\n") + ("\n(verif.)" if method in VERIFICATION_METHODS else "")


# ------------------------------------------------------------------------------------------------ records
def _load_records_light(out_dir: Path) -> list[dict]:
    """Records CSV reader with the same typing as run_eval.load_records, without importing the evaluation stack (used by the tests)."""
    ints = ("n_drones", "episode_id", "seed", "source", "frame", "steps", "steps_strict", "n_masked")
    floats = ("scale", "final_error_m", "min_error_m", "path_length_m", "entropy_final", "top_sigma_final_m", "wall_s", "step_ms_median",
              "tie_tol", "tie_frac", "all_tied_frac", "calib_2sigma_frac")
    out = []
    for f in sorted(Path(out_dir).glob("records_*.csv")):
        with f.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                r = dict(row)
                for k in ints:
                    r[k] = int(float(row[k]))
                for k in floats:
                    r[k] = float(row[k]) if row.get(k) not in (None, "") else float("nan")
                for k in ("first_detection_step", "declared_step"):
                    r[k] = int(float(row[k])) if row.get(k) not in (None, "", "None") else None
                r["declared_error_m"] = float(row["declared_error_m"]) if row.get("declared_error_m") not in (None, "", "None") else None
                for k in ("success", "success_strict"):
                    r[k] = str(row[k]) == "True"
                out.append(r)
    return out


def _load_records(out_dir: Path) -> list[dict]:
    """run_eval.load_records (imported lazily: it pulls in the whole environment stack, which a figure script otherwise does not need)."""
    from srcloc_env.eval.run_eval import load_records
    return load_records(out_dir)


RECORDS_LOADER: Callable[[Path], list[dict]] = _load_records     # tests replace it by _load_records_light


def currie_threshold_cps() -> float:
    """Detection threshold b + k sqrt(b T) / T [cps] of sensor/detector.Detector.detection_threshold_cps (config constants)."""
    b, T = config.SENSOR_BACKGROUND_CPS, config.SENSOR_T
    return float(b + config.SENSOR_CURRIE_K * math.sqrt(b * T) / T)


def detection_mask(y: np.ndarray) -> np.ndarray:
    """Counts above the Currie threshold (sensor/detector.Detector.is_detection: counts / T > b + k sqrt(b T) / T) of a step log's ``y`` (T, D) -> bool."""
    return np.asarray(y, dtype=float) / config.SENSOR_T > currie_threshold_cps()


def read_step_arrays(path: Path, keys: Iterable[str] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """(arrays, meta) of a step-log NPZ; only the requested keys that exist are read (older logs lack frame / reward / calib_inside / start_xy)."""
    with np.load(path, allow_pickle=False) as z:
        want = list(z.files) if keys is None else [k for k in keys if k in z.files]
        arrs = {k: z[k] for k in want if k != "meta"}
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return arrs, meta


def calibration_inside(arrs: dict[str, np.ndarray]) -> np.ndarray | None:
    """(T,) bool: is the truth inside the 2-sigma ellipse of the top GMM component at every step.  Uses the logged ``calib_inside`` when
    present, otherwise recomputes the Mahalanobis distance squared <= 4 from gmm_mu / gmm_cov / truth_xy (top = largest valid weight);
    None when neither is available."""
    if "calib_inside" in arrs:
        return np.asarray(arrs["calib_inside"], dtype=bool)
    if not all(k in arrs for k in ("gmm_mu", "gmm_cov", "truth_xy")):
        return None
    mu, cov = np.asarray(arrs["gmm_mu"], dtype=float), np.asarray(arrs["gmm_cov"], dtype=float)
    T = mu.shape[0]
    if "gmm_w" in arrs:
        w = np.asarray(arrs["gmm_w"], dtype=float)
        if "gmm_mask" in arrs:
            w = np.where(np.asarray(arrs["gmm_mask"], dtype=bool), w, -np.inf)
        k = np.argmax(w, axis=1)
    else:
        k = np.zeros(T, dtype=int)
    idx = np.arange(T)
    d = np.asarray(arrs["truth_xy"], dtype=float).reshape(1, 2) - mu[idx, k]
    c = cov[idx, k]
    a, b, e = c[:, 0, 0], 0.5 * (c[:, 0, 1] + c[:, 1, 0]), c[:, 1, 1]
    det = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        m2 = (e * d[:, 0] ** 2 - 2.0 * b * d[:, 0] * d[:, 1] + a * d[:, 1] ** 2) / det
    return (det > 0) & np.isfinite(m2) & (m2 <= CALIB_SIGMAS ** 2)


def record_from_step_log(path: Path, method: str, n_drones: int, episode_id: int, episode_row: dict | None = None,
                         default_mode: str = "F") -> tuple[dict, bool]:
    """Rebuild one evaluation record (run_eval RECORD_FIELDS) from a step log, for configurations whose records are not written yet.
    Success rules as in the environment (GMM top sigma < ENV_SUCCESS_SIGMA_M and MAP error < ENV_SUCCESS_ERROR_M at ANY step, strict
    SUCCESS_SIGMA_M / SUCCESS_ERROR_M, declared stop = first step with sigma < ENV_SUCCESS_SIGMA_M); steps are 1-based.  Returns
    (record, consistent) where consistent is False when the logged ``meta['success']`` disagrees with the recomputed flag.
    Not recoverable from a log: wall time, per-step time (nan), tie statistics (nan), path length of the first move of old logs without start_xy."""
    arrs, meta = read_step_arrays(path)
    err, sig = np.asarray(arrs["map_error"], dtype=float), np.asarray(arrs["top_sigma"], dtype=float)
    T = int(err.shape[0])
    ok = (sig < config.ENV_SUCCESS_SIGMA_M) & (err < config.ENV_SUCCESS_ERROR_M)
    strict = (sig < config.SUCCESS_SIGMA_M) & (err < config.SUCCESS_ERROR_M)
    decl = np.flatnonzero(sig < config.ENV_SUCCESS_SIGMA_M)
    pos = np.asarray(arrs["drone_xy"], dtype=float)
    if "start_xy" in arrs:
        pos = np.concatenate([np.asarray(arrs["start_xy"], dtype=float).reshape(1, *pos.shape[1:]), pos], axis=0)
    path_len = float(np.sum(np.hypot(*np.diff(pos, axis=0).transpose(2, 0, 1)))) if pos.shape[0] > 1 else 0.0
    ys = np.asarray(arrs["y"], dtype=float).reshape(T, -1)
    det = np.flatnonzero(detection_mask(ys).any(axis=1))
    inside = calibration_inside(arrs)
    row = dict(episode_row or {})
    ent = np.asarray(arrs.get("entropy", np.full(T, np.nan)), dtype=float)
    rec = {"method": method, "n_drones": int(n_drones), "episode_id": int(episode_id),
           "seed": int(meta.get("seed", row.get("seed", -1))), "source": int(meta.get("source", row.get("source", -1))),
           "frame": int(meta.get("frame", row.get("frame", -1))), "scale": float(meta.get("scale", row.get("scale", float("nan")))),
           "mode": str(meta.get("mode", row.get("mode", default_mode))), "start_type": str(meta.get("start_type", "")),
           "wind_level": str(meta.get("wind_level", row.get("wind_level", config.WIND_LEVEL_DEFAULT))), "trained_wind_level": str(row.get("trained_wind_level", "")),
           "wind_u": float(meta.get("wind_u", row.get("wind_u", config.WIND_MEAN_U))), "wind_dir_deg": float(meta.get("wind_dir_deg", row.get("wind_dir_deg", config.WIND_MEAN_DIR_DEG))),
           "success": bool(ok.any()), "steps": int(np.argmax(ok)) + 1 if ok.any() else T,
           "success_strict": bool(strict.any()), "steps_strict": int(np.argmax(strict)) + 1 if strict.any() else T,
           "min_error_m": float(np.min(err)), "final_error_m": float(err[-1]),
           "first_detection_step": int(det[0]) + 1 if det.size else None,
           "declared_step": int(decl[0]) + 1 if decl.size else None,
           "declared_error_m": float(err[decl[0]]) if decl.size else None,
           "path_length_m": path_len, "n_masked": int((~np.asarray(arrs["applied"], dtype=bool)).sum()) if "applied" in arrs else 0,
           "entropy_final": float(ent[-1]), "top_sigma_final_m": float(sig[-1]), "wall_s": float("nan"), "step_ms_median": float("nan"),
           "tie_tol": float("nan"), "tie_frac": float("nan"), "all_tied_frac": float("nan"),
           "calib_2sigma_frac": float(np.mean(inside)) if inside is not None else float("nan"), "step_log": str(path)}
    consistent = ("success" not in meta) or (bool(meta["success"]) == rec["success"])
    return rec, consistent


@dataclass
class StepSeries:
    """Per-step arrays of one (method, n_drones) configuration from its step logs, padded with NaN to the longest log."""
    method: str
    n_drones: int
    episode_id: np.ndarray
    source: np.ndarray
    start_type: np.ndarray
    success: np.ndarray
    err: np.ndarray             # (E, H) MAP error [m]
    sigma: np.ndarray           # (E, H) top GMM sigma [m]
    entropy: np.ndarray         # (E, H) PF entropy [nats]
    inside: np.ndarray          # (E, H) 1.0 / 0.0 truth inside the 2-sigma ellipse, NaN where unknown
    n_missing: int = 0          # records without a readable step log

    def select(self, sources: Iterable[int]) -> np.ndarray:
        return np.isin(self.source, list(sources))


@dataclass
class EvalSet:
    """One evaluation directory: typed records (plus rebuilt ones), step-log index, mode and horizon."""
    tag: str
    path: Path
    mode: str
    horizon: int
    records: list[dict]
    step_index: dict[tuple[str, int, int], Path]
    notes: list[str] = field(default_factory=list)
    rebuilt: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    _agg: dict = field(default_factory=dict, repr=False)
    _series: dict = field(default_factory=dict, repr=False)

    def configs(self, methods: Iterable[str] | None = None) -> list[tuple[str, int]]:
        keep = None if methods is None else set(methods)
        found = {(r["method"], int(r["n_drones"])) for r in self.records if keep is None or r["method"] in keep}
        order = order_methods({m for m, _ in found})
        return sorted(found, key=lambda c: (order.index(c[0]), c[1]))

    def methods(self) -> list[str]:
        return order_methods({m for m, _ in self.configs()})

    def drone_counts(self, methods: Iterable[str] | None = None) -> list[int]:
        return sorted({n for _, n in self.configs(methods)})

    def recs(self, method: str, n: int) -> list[dict]:
        return sorted((r for r in self.records if r["method"] == method and int(r["n_drones"]) == int(n)), key=lambda r: r["episode_id"])

    def aggregate(self, method: str, n: int, start_type: str | None = None) -> dict:
        key = (method, int(n), start_type)
        if key not in self._agg:
            self._agg[key] = metrics.aggregate(self.recs(method, n), max_steps=self.horizon, start_type=start_type)
        return self._agg[key]

    def step_log_path(self, rec: dict) -> Path | None:
        p = rec.get("step_log")
        if p:
            pp = Path(str(p))
            if pp.is_file():
                return pp
        return self.step_index.get((rec["method"], int(rec["n_drones"]), int(rec["episode_id"])))

    def series(self, method: str, n: int, per_source_cap: int | None = None) -> StepSeries | None:
        """StepSeries of a configuration (cached); ``per_source_cap`` keeps at most that many episodes per source (smoke runs)."""
        key = (method, int(n), per_source_cap)
        if key in self._series:
            return self._series[key]
        recs, seen, missing = [], {}, 0
        for r in self.recs(method, n):
            c = seen.get(r["source"], 0)
            if per_source_cap is not None and c >= per_source_cap:
                continue
            seen[r["source"]] = c + 1
            recs.append(r)
        rows = []
        for r in recs:
            p = self.step_log_path(r)
            if p is None:
                missing += 1
                continue
            try:
                arrs, _ = read_step_arrays(p, ("map_error", "top_sigma", "entropy", "calib_inside", "gmm_w", "gmm_mask", "gmm_mu", "gmm_cov", "truth_xy"))
            except _READ_ERRORS:
                missing += 1
                continue
            if "map_error" not in arrs or "top_sigma" not in arrs:
                missing += 1
                continue
            ins = calibration_inside(arrs)
            rows.append((r, np.asarray(arrs["map_error"], float), np.asarray(arrs["top_sigma"], float),
                         np.asarray(arrs.get("entropy", np.full(len(arrs["map_error"]), np.nan)), float),
                         None if ins is None else ins.astype(float)))
        if not rows:
            self._series[key] = None
            return None
        H = max(len(x[1]) for x in rows)

        def pad(a: np.ndarray | None) -> np.ndarray:
            out = np.full(H, np.nan)
            if a is not None:
                out[:len(a)] = a
            return out

        s = StepSeries(method=method, n_drones=int(n), episode_id=np.array([x[0]["episode_id"] for x in rows]),
                       source=np.array([x[0]["source"] for x in rows]), start_type=np.array([x[0].get("start_type", "") for x in rows]),
                       success=np.array([bool(x[0]["success"]) for x in rows]),
                       err=np.stack([pad(x[1]) for x in rows]), sigma=np.stack([pad(x[2]) for x in rows]),
                       entropy=np.stack([pad(x[3]) for x in rows]), inside=np.stack([pad(x[4]) for x in rows]), n_missing=missing)
        self._series[key] = s
        return s


def resolve_eval_dir(spec: str | Path) -> Path:
    """An existing directory, else a tag under <CACHE_DIR>/eval/."""
    p = Path(spec)
    if p.is_dir():
        return p
    q = config.CACHE_DIR / "eval" / str(spec)
    if q.is_dir():
        return q
    raise FileNotFoundError(f"evaluation directory or tag not found: {spec}")


def load_eval_set(spec: str | Path, loader: Callable[[Path], list[dict]] | None = None) -> EvalSet:
    """Load records / summary / step-log index of one evaluation directory; rebuild records of configurations that only have step logs."""
    d = resolve_eval_dir(spec)
    notes: list[str] = []
    summary: dict = {}
    sp = d / "summary.json"
    if sp.is_file():
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            notes.append(f"summary.json unreadable ({e})")
    episodes: dict[int, dict] = {}
    ep = d / "episodes.csv"
    if ep.is_file():
        with ep.open("r", encoding="utf-8", newline="") as fh:
            episodes = {int(r["episode_id"]): r for r in csv.DictReader(fh)}
    records: list[dict] = []
    try:
        records = list((loader or RECORDS_LOADER)(d))
    except _READ_ERRORS as e:
        notes.append(f"records unreadable ({e})")
    step_index: dict[tuple[str, int, int], Path] = {}
    sd = d / "steps"
    if sd.is_dir():
        for p in sorted(sd.glob("*.npz")):
            m = STEP_LOG_RE.match(p.name)
            if m:
                step_index[(m["method"], int(m["n"]), int(m["id"]))] = p
    mode = str(summary.get("mode") or "")
    if not mode and episodes:
        modes = [r.get("mode", "") for r in episodes.values() if r.get("mode")]
        mode = max(set(modes), key=modes.count) if modes else ""
    if not mode and records:
        mode = str(records[0].get("mode", ""))
    mode = mode or "F"
    have = {(r["method"], int(r["n_drones"]), int(r["episode_id"])) for r in records}
    rebuilt_cfgs: dict[tuple[str, int], int] = {}
    inconsistent = unreadable = 0
    for (m, n, i), p in step_index.items():
        if (m, n, i) in have:
            continue
        try:
            rec, ok = record_from_step_log(p, m, n, i, episodes.get(i), default_mode=mode)
        except _READ_ERRORS:
            unreadable += 1                                           # partially written by a running batch
            continue
        records.append(rec)
        rebuilt_cfgs[(m, n)] = rebuilt_cfgs.get((m, n), 0) + 1
        inconsistent += int(not ok)
    rebuilt = [f"{cfg_label(m, n)}: {c} episodes" for (m, n), c in sorted(rebuilt_cfgs.items())]
    if rebuilt:
        notes.append("records rebuilt from step logs (no records_*.csv yet): " + "; ".join(rebuilt))
    if unreadable:
        notes.append(f"{unreadable} step logs unreadable (being written?)")
    if inconsistent:
        notes.append(f"{inconsistent} rebuilt episodes disagree with the logged success flag")
    horizon = summary.get("max_steps")
    if not horizon:
        horizon = 0
        for p in list(step_index.values())[:1]:
            try:
                horizon = int(read_step_arrays(p, ())[1].get("max_steps", 0))
            except _READ_ERRORS:
                horizon = 0
        horizon = horizon or default_horizon(mode)
    return EvalSet(tag=d.name, path=d, mode=mode, horizon=int(horizon), records=records, step_index=step_index, notes=notes,
                   rebuilt=rebuilt, summary=summary)


# ------------------------------------------------------------------------------------------------ manifest
class Outputs:
    """Collector of the manifest and of the numbers-to-source rows (spec 10 item 11) of one evaluation set."""

    def __init__(self, out_dir: Path, prefix: str, tag: str, suffix: str, notes: Sequence[str] = ()) -> None:
        self.out_dir, self.base, self.tag, self.suffix, self.notes = Path(out_dir), prefix, tag, suffix, list(notes)
        self.manifest: dict[str, dict[str, Any]] = {}
        self.rows: list[dict[str, str]] = []
        self._notes_given = False

    def path(self, name: str, ext: str) -> Path:
        return self.out_dir / f"{self.base}_{name}.{ext}"

    def key(self, name: str) -> str:
        return f"{name}{self.suffix}"

    def ok(self, name: str, path: Path, what: str, data_files: Sequence[str], columns: str, section: str, reason: str = "", **extra: Any) -> None:
        notes = self.notes if not self._notes_given else (["data notes: see the first output of this set"] if self.notes else [])
        self._notes_given = True
        r = "; ".join(x for x in [reason, *notes] if x)
        self.manifest[self.key(name)] = {"path": str(path), "status": "ok", "reason": r, **extra}
        slide, basis = R_SLIDE_MAP.get(name, R_SLIDE_DEFAULT)
        csv_path = extra.get("data_csv") or (str(path) if Path(path).suffix.lower() == ".csv" else "")
        self.rows.append({"output": self.key(name), "file": str(path), "spec_section": section, "slide": slide, "slide_basis": basis,
                          "what_it_shows": what, "data_files": "; ".join(data_files), "columns_used": columns, "output_csv": csv_path,
                          "output_csv_columns": _csv_header(csv_path), "notes": r})

    def skip(self, name: str, reason: str) -> None:
        self.manifest[self.key(name)] = {"path": "", "status": "skipped", "reason": reason}


def _csv_header(path: str | Path) -> str:
    """Column names (comma separated) of a CSV file written by this tool; '' when there is none."""
    try:
        if path and Path(path).is_file():
            with Path(path).open("r", encoding="utf-8", newline="") as fh:
                return ", ".join(next(csv.reader(fh), []))
    except (OSError, UnicodeDecodeError):
        pass
    return ""


@contextlib.contextmanager
def _dpi_override(dpi: int | None):
    """Temporarily set the module resolution FIG_DPI (``make_all(dpi=...)``, ``--dpi``); None keeps the current value."""
    global FIG_DPI
    old = FIG_DPI
    if dpi is not None:
        FIG_DPI = int(dpi)
    try:
        yield
    finally:
        FIG_DPI = old


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


def _suptitle(fig: plt.Figure, text: str, fontsize: float = 11.0) -> None:
    """Figure title wrapped to the figure width (single-panel figures are narrower than their long titles and used to be clipped)."""
    width = max(28, int(fig.get_figwidth() * 72.0 / (0.62 * fontsize)))
    lines = []
    for ln in str(text).split("\n"):
        lines += textwrap.wrap(ln, width=width) or [""]
    fig.suptitle("\n".join(lines), fontsize=fontsize)


def _show_xticks(axes: Iterable[plt.Axes]) -> None:
    """Keep the x tick numbers of every panel of a shared-x grid (matplotlib hides them on the inner rows)."""
    for ax in axes:
        if ax.axison:
            ax.tick_params(axis="x", labelbottom=True)


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields) if fields is not None else (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _records_files(ev: EvalSet, cfgs: Iterable[tuple[str, int]]) -> list[str]:
    out = []
    for m, n in cfgs:
        f = ev.path / f"records_{m}_{n}drones.csv"
        out.append(str(f) if f.is_file() else f"{ev.path / 'steps'}/{m}_{n}drones_ep*.npz (rebuilt records)")
    return out


def _steps_files(ev: EvalSet) -> list[str]:
    return [f"{ev.path / 'steps'}/*.npz (or the step_log paths of the records)"]


def _range_text(values: Iterable[int]) -> str:
    v = sorted(set(int(x) for x in values))
    return "0" if not v else (f"{v[0]}" if len(v) == 1 else f"{v[0]}-{v[-1]}")


def _title(ev: EvalSet, text: str, episodes: str | None = None) -> str:
    """Two-line title; ``episodes`` replaces the default '<n> episodes per configuration' (step-log figures state the episodes actually drawn)."""
    if episodes is None:
        episodes = f"{_range_text(len(ev.recs(m, n)) for m, n in ev.configs())} episodes per configuration"
    return f"{text}\n{ev.tag}: Mode {ev.mode}, horizon {ev.horizon} steps, {episodes}"


def episodes_phrase(ev: EvalSet, series: dict[tuple[str, int], "StepSeries"], cap: int | None) -> tuple[str, dict[str, dict[str, int]]]:
    """('12 of 130 episodes per configuration', {config label: {used, total}}) for a step-log figure: ``used`` = observable-source episodes
    drawn (source 110 is never drawn), ``total`` = all records of the configuration.  The 'of' part appears only when episodes were left out
    by the per-source cap or by missing step logs."""
    obs = set(metrics.GROUPS[CURVE_GROUP])
    used, total, avail, info = [], [], [], {}
    for (m, n), s in series.items():
        recs = ev.recs(m, n)
        u, t = int(s.select(obs).sum()), len(recs)
        used.append(u)
        total.append(t)
        avail.append(sum(int(r["source"]) in obs for r in recs))
        info[cfg_label(m, n)] = {"used": u, "total": t}
    missing = sum(s.n_missing for s in series.values())
    if used == avail and not missing:
        return f"{_range_text(used)} episodes per configuration", info
    extra = []
    if cap is not None:
        extra.append(f"first {cap} per source")
    if missing:
        extra.append(f"{missing} without readable step log")
    return f"{_range_text(used)} of {_range_text(total)} episodes per configuration" + (f" ({'; '.join(extra)})" if extra else ""), info


def _legend(fig: plt.Figure, methods: Sequence[str], drones: Sequence[int], extra: Sequence = (), ncol: int | None = None) -> None:
    ordered = list(methods)
    handles = [Line2D([0], [0], color=method_colour(m, ordered), lw=2.2, ls="--" if m in VERIFICATION_METHODS else "-",
                      label=method_label(m)) for m in methods]
    if len(drones) > 1:
        handles += [Line2D([0], [0], color="0.25", lw=1.8, ls=DRONE_LINESTYLES.get(n, "-"), label=f"{n} drone{'s' if n > 1 else ''}") for n in drones]
    handles += list(extra)
    fit = max(2, int(fig.get_figwidth() // 1.75))                     # entries per row that fit the figure width (narrow single-panel figures)
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol or min(len(handles), 6, fit), fontsize=8, frameon=False)


def _drone_panels(drones: Sequence[int], w: float = 6.0, h: float = 4.4, sharey: bool = True, vertical: bool = False) -> tuple[plt.Figure, list[plt.Axes]]:
    """One panel per drone count: side by side (w, h = size of ONE panel) or stacked when ``vertical``."""
    if vertical:
        fig, axes = plt.subplots(len(drones), 1, figsize=(w, h * len(drones)), squeeze=False, sharey=sharey, layout="constrained")
        return fig, [a[0] for a in axes]
    fig, axes = plt.subplots(1, len(drones), figsize=(w * len(drones), h), squeeze=False, sharey=sharey, layout="constrained")
    return fig, list(axes[0])


def _style(method: str, n: int, ordered: Sequence[str]) -> dict:
    return {"color": method_colour(method, ordered), "ls": DRONE_LINESTYLES.get(int(n), "-"), "lw": 1.5 if method not in VERIFICATION_METHODS else 1.2}


# ------------------------------------------------------------------------------------------------ Table 2
TABLE2_HEADER = ("| Method (drones) | Group | n | Success rate [95% CI] | Strict success rate | Success step median [CI] | Censored median | "
                 "Final error median / p90 [m] | Declared success rate (all episodes) | Declared correct (share of declaring episodes) | "
                 "Overconfident declarations (error >= 20 m; share of declaring episodes) | "
                 "Declaration delay median [steps] (first success step - declared step) |")
AGG_COLUMNS = ("n", "n_success", "success_rate", "success_ci", "n_success_strict", "success_strict_rate", "success_strict_ci", "success_step_median",
               "success_step_ci", "censored_step_median", "final_error_median_m", "final_error_p90_m", "first_detection_median", "declared_n",
               "declared_success_rate", "declared_conditional_rate", "declared_overconfident_rate", "declaration_delay_median",
               "path_length_median_m", "masked_actions_median")
R_DECLARED_GROUPS = ("train", "holdout", "all_observable")           # groups of the declared-success columns of the Table 2 picture


def _fmt_pct(x: float | None) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def _fmt_num(x: float | None, digits: int = 0) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _cell_rate(a: dict | None) -> str:
    return "-" if a is None else f"{_fmt_pct(a['success_rate'])} [{_fmt_pct(a['success_ci'][0])}, {_fmt_pct(a['success_ci'][1])}]"


def _cell_steps(a: dict | None) -> str:
    if a is None:
        return "-"
    m = a["success_step_median"]
    s = "-" if not np.isfinite(m) else f"{m:.0f} [{a['success_step_ci'][0]:.0f}, {a['success_step_ci'][1]:.0f}]"
    return f"{s} ({a['censored_step_median']:.0f})"


def _cell_err(a: dict | None) -> str:
    return "-" if a is None else f"{a['final_error_median_m']:.0f} / {a['final_error_p90_m']:.0f}"


def _cell_declared_all(a: dict | None) -> str:
    """Declared success rate over ALL episodes of the group (an episode without a declaration is a failure, spec 9.7)."""
    return "-" if a is None else _fmt_pct(a.get("declared_success_rate"))


def _cell_declared_share(a: dict | None, key: str) -> str:
    """Share of the DECLARING episodes ('37% of 8'): declared_conditional_rate (error < success error) or declared_overconfident_rate (error >= strict error)."""
    if a is None or a.get(key) is None:
        return "-"
    return f"{_fmt_pct(a[key])} of {a.get('declared_n', '?')}"


def _cell_delay(a: dict | None) -> str:
    return "-" if a is None else _fmt_num(a.get("declaration_delay_median"))


def table2_markdown_rows(aggs: dict[str, dict], groups: Sequence[str] = ("train_open", "train_other", "holdout", "train", "all_observable", "unobservable")) -> list[str]:
    """Rows of the English Table 2 (header TABLE2_HEADER, one row per configuration label x group), including the declared-stop columns."""
    lines = [TABLE2_HEADER, "|" + "---|" * (TABLE2_HEADER.count("|") - 1)]
    for label, agg in aggs.items():
        for g in groups:
            a = agg.get(g)
            if a is None:
                continue
            ss = f"{a['success_step_median']:.0f} [{a['success_step_ci'][0]:.0f}, {a['success_step_ci'][1]:.0f}]" if np.isfinite(a["success_step_median"]) else "-"
            lines.append(f"| {label} | {g} | {a['n']} | {_cell_rate(a)} | {_fmt_pct(a['success_strict_rate'])} | {ss} | {a['censored_step_median']:.0f} | "
                         f"{_cell_err(a)} | {_cell_declared_all(a)} | {_cell_declared_share(a, 'declared_conditional_rate')} | "
                         f"{_cell_declared_share(a, 'declared_overconfident_rate')} | {_cell_delay(a)} |")
    return lines


def build_table2(ev: EvalSet, out: Outputs, methods: Sequence[str] | None = None) -> None:
    cfgs = ev.configs(methods)
    if not cfgs:
        raise Skip("no evaluation records")
    aggs = {cfg_label(m, n): ev.aggregate(m, n) for m, n in cfgs}
    lines = table2_markdown_rows(aggs)
    notes = [f"Table 2 - {ev.tag} (Mode {ev.mode}, horizon {ev.horizon} steps). Success = top GMM sigma < {config.ENV_SUCCESS_SIGMA_M:.0f} m and MAP error < "
             f"{config.ENV_SUCCESS_ERROR_M:.0f} m at any step (strict {config.SUCCESS_SIGMA_M:.0f} / {config.SUCCESS_ERROR_M:.0f} m); CI = Wilson 95 % (rates) / "
             f"bootstrap {config.EVAL_N_BOOTSTRAP} resamples (step median, successes only); censored median counts failures as {ev.horizon}.",
             f"Declared stop (deployment view, spec 9.7) = the first step with top GMM sigma < {config.ENV_SUCCESS_SIGMA_M:.0f} m; the episode continues. "
             f"Declared success rate = share of ALL episodes whose MAP error at that step is < {config.ENV_SUCCESS_ERROR_M:.0f} m (an episode that never declares is a failure). "
             f"Declared correct and overconfident are shares of the declaring episodes only ('x% of n'): overconfident = declared although the error is >= {config.SUCCESS_ERROR_M:.0f} m. "
             "Declaration delay = declared step minus the first success step, median over episodes that declared and succeeded "
             "(never positive: the belief declares at or before the first success; negative = declared while the estimate was still wrong).",
             "Groups (mutually exclusive): " + "; ".join(group_label(g) for g in ("train_open", "train_other", "holdout"))
             + f"; summaries {group_label('all_observable')}, train (9). Source 110 is unobservable at the 15 m flight altitude and reported apart."]
    if any(m in VERIFICATION_METHODS for m, _ in cfgs):
        notes.append("oracle_loiter is a verification baseline (uses the true source), not a competing method.")
    md_text = notes[0] + "\n\n" + "\n".join(lines) + "\n\n" + "\n\n".join(notes[1:]) + "\n"
    p_md = out.path("table2", "md")
    p_md.parent.mkdir(parents=True, exist_ok=True)
    p_md.write_text(md_text, encoding="utf-8")
    rows = []
    for m, n in cfgs:
        for g, a in ev.aggregate(m, n).items():
            row = {"tag": ev.tag, "mode": ev.mode, "horizon": ev.horizon, "method": m, "n_drones": n, "group": g}
            for k in AGG_COLUMNS:
                v = a.get(k)
                if isinstance(v, list):
                    row[k + "_lo"], row[k + "_hi"] = v
                else:
                    row[k] = v
            rows.append(row)
    fields = ["tag", "mode", "horizon", "method", "n_drones", "group"]
    for k in AGG_COLUMNS:
        fields += [k + "_lo", k + "_hi"] if k.endswith("_ci") else [k]
    p_csv = _write_csv(out.path("table2", "csv"), rows, fields)
    p_png = _render_table2_png(ev, cfgs, aggs, out.path("table2", "png"))
    data = _records_files(ev, cfgs)
    cols = ("method, n_drones, episode_id, source, start_type, success, steps, success_strict, final_error_m, first_detection_step, declared_step, "
            "declared_error_m, path_length_m, n_masked")
    out.ok("table2_md", p_md, "Table 2 (Markdown): success rate [Wilson CI], strict success, success-step median [bootstrap CI], censored median, final error median/p90, "
           "declared success (all episodes), declared correct / overconfident (of declaring) and declaration delay per method x drones x group", data, cols, "10.1",
           data_csv=str(p_csv))
    out.ok("table2_csv", p_csv, "Table 2 numbers, one row per method x drones x group (incl. declared_success_rate, declared_conditional_rate, "
           "declared_overconfident_rate, declaration_delay_median)", data, cols, "10.1")
    out.ok("table2_png", p_png, "Table 2 rendered as a cross-table (success, step median, final error, declared stop by group)", data, cols, "10.1",
           data_csv=str(p_csv))


def _render_table2_png(ev: EvalSet, cfgs: Sequence[tuple[str, int]], aggs: dict[str, dict], path: Path) -> Path:
    labels = [cfg_label(m, n) + (" *" if m in VERIFICATION_METHODS else "") for m, n in cfgs]
    keys = [cfg_label(m, n) for m, n in cfgs]
    groups = TABLE_GROUPS

    def head(g: str) -> str:
        ns = {aggs[k][g]["n"] for k in keys if g in aggs[k]}
        return f"{GROUP_SHORT[g]}\n(n={ns.pop()})" if len(ns) == 1 else f"{GROUP_SHORT[g]}\n(n varies)"

    obs = "all_observable"
    d_groups = [g for g in R_DECLARED_GROUPS if any(g in aggs[k] for k in keys)]
    blocks = [
        ("Success rate [95% Wilson CI]", [head(g) for g in groups] + ["strict\n(all obs.)", "110\n(unobs.)"],
         [[_cell_rate(aggs[k].get(g)) for g in groups] + [_fmt_pct(aggs[k].get(obs, {}).get("success_strict_rate")),
                                                          _cell_rate(aggs[k].get("unobservable"))] for k in keys], True),
        ("Success step median [bootstrap CI] (censored median, failures = horizon)", [head(g) for g in groups],
         [[_cell_steps(aggs[k].get(g)) for g in groups] for k in keys], False),
        ("Final MAP error median / p90 [m]", [head(g) for g in groups], [[_cell_err(aggs[k].get(g)) for g in groups] for k in keys], False),
        (f"Declared stop (first step with top sigma < {config.ENV_SUCCESS_SIGMA_M:.0f} m): success over ALL episodes, then shares of the declaring episodes (all obs.)",
         [f"declared success\n(all episodes)\n{GROUP_SHORT[g].replace(chr(10), ' ')}" for g in d_groups]
         + ["declared correct\n(error < %.0f m)\nof declaring" % config.ENV_SUCCESS_ERROR_M,
            "overconfident\n(error >= %.0f m)\nof declaring" % config.SUCCESS_ERROR_M, "delay median\n[steps]\n(success - declared)"],
         [[_cell_declared_all(aggs[k].get(g)) for g in d_groups] + [_cell_declared_share(aggs[k].get(obs), "declared_conditional_rate"),
                                                                   _cell_declared_share(aggs[k].get(obs), "declared_overconfident_rate"),
                                                                   _cell_delay(aggs[k].get(obs))] for k in keys], False),
    ]
    row_h = 0.27
    heights = [(len(keys) + 2.2) * row_h + 0.5 for _ in blocks]
    fig, axes = plt.subplots(len(blocks), 1, figsize=(15.5, sum(heights) + 1.2), gridspec_kw={"height_ratios": heights}, layout="constrained")
    for ax, (title, cols, cells, mark_best) in zip(axes, blocks):
        ax.axis("off")
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        tab = ax.table(cellText=cells, rowLabels=labels, colLabels=cols, loc="upper center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        tab.scale(1.0, 1.25)
        for (r, c), cell in tab.get_celld().items():
            cell.set_linewidth(0.3)
            cell.set_edgecolor("0.7")
            if r == 0:
                cell.set_facecolor("#e8eef7")
                cell.set_height(cell.get_height() * 1.7)
            elif cfgs[r - 1][0] in VERIFICATION_METHODS:
                cell.set_facecolor("#efefef")
        if mark_best:
            for c in range(len(groups)):                              # highlight the best competing method per group column
                vals = [(aggs[keys[i]].get(groups[c], {}).get("success_rate", -1.0), i) for i in range(len(keys)) if cfgs[i][0] not in VERIFICATION_METHODS]
                if vals:
                    best = max(v for v, _ in vals)
                    for v, i in vals:
                        if v == best and best > 0:
                            tab[(i + 1, c)].set_facecolor("#cfe2f7")
                            tab[(i + 1, c)].get_text().set_fontweight("bold")
    _suptitle(fig, f"Table 2 - {ev.tag} (Mode {ev.mode}, horizon {ev.horizon} steps)", fontsize=12)
    foot = ("success = top GMM sigma < %.0f m and MAP error < %.0f m at any step; strict %.0f / %.0f m; shaded blue = best competing method per group; "
            "* = verification only (true-source oracle); groups are mutually exclusive, 110 is unobservable at 15 m; "
            "declared success counts non-declaring episodes as failures, 'x%% of n' = share of the n declaring episodes"
            % (config.ENV_SUCCESS_SIGMA_M, config.ENV_SUCCESS_ERROR_M, config.SUCCESS_SIGMA_M, config.SUCCESS_ERROR_M))
    fig.text(0.01, 0.003, "\n".join(textwrap.wrap(foot, 190)), fontsize=7.5, color="0.35", ha="left", va="bottom")
    return _save(fig, path)


# ------------------------------------------------------------------------------------------------ 2 success bars, start type
def _bar_axes(ax: plt.Axes, ev: EvalSet, methods: Sequence[str], n: int, groups: Sequence[str]) -> None:
    ms = [m for m in methods if ev.recs(m, n)]
    w = 0.82 / max(len(ms), 1)
    for j, m in enumerate(ms):
        agg = ev.aggregate(m, n)
        xs = [i for i, g in enumerate(groups) if g in agg]
        vals = [agg[groups[i]] for i in xs]
        if not vals:
            continue
        rate = np.array([a["success_rate"] for a in vals])
        ci = np.array([a["success_ci"] for a in vals])
        pos = np.array(xs) - 0.41 + w * (j + 0.5)
        ver = m in VERIFICATION_METHODS
        ax.bar(pos, rate, w * 0.92, yerr=np.vstack([np.maximum(rate - ci[:, 0], 0.0), np.maximum(ci[:, 1] - rate, 0.0)]), color=method_colour(m, ms), alpha=0.55 if ver else 0.95,
               hatch="//" if ver else None, edgecolor="white" if not ver else "0.2", linewidth=0.5, error_kw={"elinewidth": 1.0, "capsize": 1.5, "ecolor": "0.15"})
        ax.plot(pos, [a["success_strict_rate"] for a in vals], ls="none", marker="D", ms=3.2, mfc="white", mec="0.1", mew=0.8)
    ticks = []
    for g in groups:
        ns = sorted({ev.aggregate(m, n)[g]["n"] for m in ms if g in ev.aggregate(m, n)})
        ticks.append(GROUP_SHORT[g].replace("\n", " ") + "\nn=" + ("/".join(str(x) for x in ns) if ns else "-"))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def build_success_bars(ev: EvalSet, out: Outputs, methods: Sequence[str] | None = None) -> None:
    cfgs = ev.configs(methods)
    if not cfgs:
        raise Skip("no evaluation records")
    ms, drones = order_methods({m for m, _ in cfgs}), sorted({n for _, n in cfgs})
    fig, axes = _drone_panels(drones, w=max(9.0, 0.42 * len(BAR_GROUPS) * (len(ms) + 1)), h=3.9, vertical=True)
    rows = []
    for ax, n in zip(axes, drones):
        _bar_axes(ax, ev, ms, n, BAR_GROUPS)
        ax.set_title(f"{n} drone{'s' if n > 1 else ''}", fontsize=10)
        for m in ms:
            for g, a in ev.aggregate(m, n).items():
                if g in BAR_GROUPS:
                    rows.append({"tag": ev.tag, "method": m, "n_drones": n, "group": g, "n": a["n"], "success_rate": a["success_rate"],
                                 "ci_lo": a["success_ci"][0], "ci_hi": a["success_ci"][1], "strict_rate": a["success_strict_rate"]})
    _legend(fig, ms, [], extra=[Line2D([0], [0], marker="D", ls="none", mfc="white", mec="0.1", ms=5, label="strict success rate")], ncol=8)
    _suptitle(fig, _title(ev, "Success rate by source group (bars = primary criterion, error bars = Wilson 95 % CI)"), fontsize=11)
    p = _save(fig, out.path("success_bars", "png"))
    c = _write_csv(out.path("success_bars", "csv"), rows)
    out.ok("success_bars", p, "Success rate per source group and method x drone count with Wilson 95 % CI error bars (diamond = strict criterion)",
           _records_files(ev, cfgs), "method, n_drones, source, success, success_strict", "10.2", data_csv=str(c))


def build_start_type(ev: EvalSet, out: Outputs, methods: Sequence[str] | None = None) -> None:
    cfgs = ev.configs(methods)
    if not cfgs:
        raise Skip("no evaluation records")
    if not any(r.get("start_type") in START_TYPES for r in ev.records):
        raise Skip("records carry no start_type")
    ms, drones = order_methods({m for m, _ in cfgs}), sorted({n for _, n in cfgs})
    rows, md = [], ["| Method (drones) | Start | Group | n | Success rate [95% CI] | Strict success rate |", "|---|---|---|---|---|---|"]
    for m, n in cfgs:
        for st in START_TYPES:
            for g, a in ev.aggregate(m, n, st).items():
                if g == "unobservable":
                    continue
                rows.append({"tag": ev.tag, "method": m, "n_drones": n, "start_type": st, "group": g, "n": a["n"], "success_rate": a["success_rate"],
                             "ci_lo": a["success_ci"][0], "ci_hi": a["success_ci"][1], "strict_rate": a["success_strict_rate"]})
                if g in ("train", "holdout", "all_observable"):
                    md.append(f"| {cfg_label(m, n)} | {st} | {g} | {a['n']} | {_cell_rate(a)} | {_fmt_pct(a['success_strict_rate'])} |")
    fig, axes = _drone_panels(drones, w=max(8.0, 1.2 * len(ms) + 2.0), h=3.8, vertical=True)
    for ax, n in zip(axes, drones):
        mm = [m for m in ms if ev.recs(m, n)]
        w = 0.38
        for k, st in enumerate(START_TYPES):
            for i, m in enumerate(mm):
                a = ev.aggregate(m, n, st).get(CURVE_GROUP)
                if a is None:
                    continue
                x = i + (k - 0.5) * w * 1.05
                lo, hi = a["success_ci"]
                ax.bar(x, a["success_rate"], w, color=method_colour(m, ms), alpha=0.95 if st == "plume" else 0.45, hatch=None if st == "plume" else "..",
                       edgecolor="0.25", linewidth=0.5, yerr=[[max(a["success_rate"] - lo, 0.0)], [max(hi - a["success_rate"], 0.0)]], error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "0.15"})
        ax.set_xticks(range(len(mm)))
        ns = [[ev.aggregate(m, n, st).get(CURVE_GROUP, {}).get("n", 0) for st in START_TYPES] for m in mm]
        ax.set_xticklabels([f"{wrap_method(m)}\n{a} / {b}" for m, (a, b) in zip(mm, ns)], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("success rate (12 observable sources)")
        ax.set_title(f"{n} drone{'s' if n > 1 else ''}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor="0.4", edgecolor="0.25", label="plume start (a drone starts in the detectable plume)"),
                        Patch(facecolor="0.8", edgecolor="0.25", hatch="..", label="random start"),
                        Patch(facecolor="none", edgecolor="none", label="numbers below the method name: n = plume / random episodes")],
               loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    _suptitle(fig, _title(ev, "Success by start type (solid = plume start, dotted = random start; Wilson 95 % CI)"), fontsize=11)
    p = _save(fig, out.path("start_type", "png"))
    c = _write_csv(out.path("start_type", "csv"), rows)
    pm = out.path("start_type", "md")
    pm.write_text(f"Success rate by start type - {ev.tag} (Mode {ev.mode}, horizon {ev.horizon} steps)\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    data = _records_files(ev, cfgs)
    cols = "method, n_drones, episode_id, source, start_type, success, success_strict"
    out.ok("start_type_bars", p, "Success rate of plume-start versus random-start episodes (12 observable sources) with Wilson 95 % CI", data, cols, "9.1", data_csv=str(c))
    out.ok("start_type_table", pm, "Start-type stratified success table (train, holdout, all observable)", data, cols, "9.1", data_csv=str(c))


# ------------------------------------------------------------------------------------------------ 3 CDF
def build_success_cdf(ev: EvalSet, out: Outputs, methods: Sequence[str] | None = None) -> None:
    cfgs = ev.configs(methods)
    if not cfgs:
        raise Skip("no evaluation records")
    ms, drones = order_methods({m for m, _ in cfgs}), sorted({n for _, n in cfgs})
    groups = [g for g in CDF_GROUPS if any(g in ev.aggregate(m, n) for m, n in cfgs)]
    grid = np.arange(0, ev.horizon + 1)
    rows, top = [], 0.0
    ncol = 2 if len(groups) > 1 else 1
    nrow = math.ceil(len(groups) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.1 * nrow + 0.7), squeeze=False, sharex=True, sharey=True, layout="constrained")
    for k in range(len(groups), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    for k, (ax, g) in enumerate(zip(axes.ravel(), groups)):
        srcs = set(metrics.GROUPS[g])
        for m, n in cfgs:
            rs = [r for r in ev.recs(m, n) if int(r["source"]) in srcs]
            if not rs:
                continue
            st = np.sort([r["steps"] for r in rs if r["success"]])
            cdf = np.searchsorted(st, grid, side="right") / len(rs)
            ax.plot(grid, cdf, drawstyle="steps-post", **_style(m, n, ms))
            top = max(top, float(cdf.max()))
            rows += [{"tag": ev.tag, "group": g, "method": m, "n_drones": n, "n_episodes": len(rs), "step": int(s), "success_cdf": float(c)} for s, c in zip(grid, cdf)]
        ax.axvline(ev.horizon, color="0.5", lw=0.8, ls=":")
        ax.set_title(group_label(g) if g != "all_observable" else "all observable (12 sources)", fontsize=9)
        if k + ncol >= len(groups):
            ax.set_xlabel("step of the first success")
        if k % ncol == 0:
            ax.set_ylabel("fraction of episodes succeeded by step t")
        ax.set_xlim(0, ev.horizon)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylim(0, min(1.0, max(0.1, 1.12 * top)))              # shared y axis, scaled to the best curve (verification curves included)
    _show_xticks(axes.ravel())
    _legend(fig, ms, drones)
    _suptitle(fig, _title(ev, f"Success-step CDF (failures censored at the horizon {ev.horizon}: curves end at the success rate)"), fontsize=11)
    p = _save(fig, out.path("success_cdf", "png"))
    c = _write_csv(out.path("success_cdf", "csv"), rows)
    out.ok("success_cdf", p, "CDF of the first success step per method (1 drone solid, 2 drones dashed); failures censored at the horizon", _records_files(ev, cfgs),
           "method, n_drones, episode_id, source, success, steps", "10.3", data_csv=str(c))


# ------------------------------------------------------------------------------------------------ step-log curves
def _series_for(ev: EvalSet, cfgs: Sequence[tuple[str, int]], cap: int | None) -> dict[tuple[str, int], StepSeries]:
    if not cfgs:
        raise Skip("no evaluation records")
    out = {}
    for m, n in cfgs:
        s = ev.series(m, n, cap)
        if s is not None:
            out[(m, n)] = s
    if not out:
        raise Skip("no step logs found (run_eval --log-steps; neither the records' step_log paths nor <dir>/steps/*.npz exist)")
    return out


def _curve_stats(x: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """(len(quantiles), H) nan-quantiles over episodes; columns with fewer than MIN_CURVE_EPISODES finite values are NaN."""
    cnt = np.isfinite(x).sum(axis=0)
    with np.errstate(all="ignore"):
        q = np.nanquantile(x, quantiles, axis=0)
    q[:, cnt < MIN_CURVE_EPISODES] = np.nan
    return q


def _miss(series: dict) -> dict:
    n_miss = sum(s.n_missing for s in series.values())
    return {"n_missing_logs": int(n_miss)} if n_miss else {}


def _curve_of(x: np.ndarray, reducer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(steps, centre, q25, q75, finite count) of an (episodes, steps) matrix: 'median' with quartiles, or 'mean' (q25 = q75 = mean)."""
    steps = np.arange(1, x.shape[1] + 1)
    cnt = np.isfinite(x).sum(axis=0)
    if reducer == "median":
        q25, med, q75 = _curve_stats(x, (0.25, 0.5, 0.75))
    else:
        with np.errstate(all="ignore"):
            med = np.where(cnt >= MIN_CURVE_EPISODES, np.nanmean(x, axis=0), np.nan)
        q25 = q75 = med
    return steps, med, q25, q75, cnt


def _collect_curves(ev: EvalSet, series: dict[tuple[str, int], StepSeries], value: Callable[[StepSeries], np.ndarray], reducer: str) -> tuple[list, list[dict]]:
    """Curves (method, n_drones, steps, centre, q25, q75) of the CURVE_GROUP episodes of every configuration, plus the rows of the curve data."""
    curves, rows = [], []
    for (m, n), s in series.items():
        x = value(s)[s.select(metrics.GROUPS[CURVE_GROUP])]
        if x.shape[0] == 0:
            continue
        steps, med, q25, q75, cnt = _curve_of(x, reducer)
        curves.append((m, n, steps, med, q25, q75))
        rows += [{"tag": ev.tag, "method": m, "n_drones": n, "step": int(t), "n_episodes": int(k), "center": float(c), "q25": float(a), "q75": float(b)}
                 for t, k, c, a, b in zip(steps, cnt, med, q25, q75)]
    return curves, rows


def _zoom_limits(curves: list, zoom_ref: float | None) -> tuple[float, float] | None:
    """y limits of the zoom row: only when the non-verification curves stay within a quarter of the range spanned by all curves."""
    unver = [c for c in curves if c[0] not in VERIFICATION_METHODS and np.isfinite(c[3]).any()]
    if zoom_ref is None or not unver or len(unver) == len(curves):
        return None
    lo_all = min(float(np.nanmin(c[3])) for c in curves if np.isfinite(c[3]).any())
    lo_un = min(float(np.nanmin(c[3])) for c in unver)
    if zoom_ref - lo_un >= 0.25 * (zoom_ref - lo_all):
        return None
    pad = 0.15 * max(zoom_ref - lo_un, 0.005)
    return lo_un - pad, zoom_ref + pad


def _step_curve_figure(ev: EvalSet, out: Outputs, name: str, series: dict[tuple[str, int], StepSeries], value: Callable[[StepSeries], np.ndarray],
                       ylabel: str, title: str, logy: bool, ylim: tuple[float | None, float | None] | None,
                       refs: Sequence[tuple[float, str, str]] = (), reducer: str = "median", zoom_ref: float | None = None,
                       ep_note: str | None = None) -> tuple[Path, list[dict]]:
    """Per-step curve of one value, one panel per drone count and one line per configuration (colour = method, line style = drone count).
    With ``zoom_ref`` (the value every curve starts from) a second row zooms on the non-verification methods when they stay within a quarter of
    the range spanned by all curves; returns (png path, curve rows)."""
    curves, rows = _collect_curves(ev, series, value, reducer)
    ms = order_methods({c[0] for c in curves})
    drones = sorted({c[1] for c in curves})
    zoom = _zoom_limits(curves, zoom_ref)
    nrows = 2 if zoom else 1
    fig, axes = plt.subplots(nrows, len(drones), figsize=(6.6 * len(drones), 4.5 * nrows), squeeze=False, sharex=True, sharey="row", layout="constrained")
    xmax = max((len(c[2]) for c in curves), default=ev.horizon)
    for r in range(nrows):
        for ax, n in zip(axes[r], drones):
            for m, nn, steps, med, q25, q75 in curves:
                if nn == n:
                    ax.plot(steps, med, **_style(m, n, ms))
            for y, label, ls in refs:
                ax.axhline(y, color="0.35", lw=0.9, ls=ls)
                ax.text(0.01, y, label, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=7, color="0.3", clip_on=True)
            if logy:
                ax.set_yscale("log")
            if r == 1:
                ax.set_ylim(*zoom)
            elif ylim:
                ax.set_ylim(*ylim)
            ax.set_xlim(1, xmax)
            ax.grid(alpha=0.3, which="both" if logy else "major")
            if r == 0:
                ax.set_title(f"{n} drone{'s' if n > 1 else ''}", fontsize=10)
            if r == nrows - 1:
                ax.set_xlabel("step")
        axes[r][0].set_ylabel(ylabel if r == 0 else "zoom on the baselines (verification curve clipped)", fontsize=9)
    _show_xticks(axes.ravel())
    _legend(fig, ms, drones)
    _suptitle(fig, _title(ev, f"{title} ({len(metrics.GROUPS[CURVE_GROUP])} observable sources, source 110 excluded)", ep_note), fontsize=11)
    return _save(fig, out.path(name, "png")), rows


def _step_curve_by_method(ev: EvalSet, out: Outputs, name: str, series: dict[tuple[str, int], StepSeries], value: Callable[[StepSeries], np.ndarray],
                          ylabel: str, title: str, logy: bool, ylim: tuple[float | None, float | None] | None,
                          refs: Sequence[tuple[float, str, str]] = (), ep_note: str | None = None) -> tuple[Path, list[dict]]:
    """Small multiples: one panel per method with the median and the interquartile band of every drone count (colour = drone count), so that the
    bands of different methods never overlap; returns (png path, curve rows)."""
    curves, rows = _collect_curves(ev, series, value, "median")
    ms = order_methods({c[0] for c in curves})
    drones = sorted({c[1] for c in curves})
    ncol = min(4, len(ms))
    nrow = math.ceil(len(ms) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow + 0.4), squeeze=False, sharex=True, sharey=True, layout="constrained")
    xmax = max((len(c[2]) for c in curves), default=ev.horizon)
    if logy and ylim and any(np.isfinite(c[4]).any() for c in curves):          # lower limit: one decade floor below the lowest quartile (but not below the configured bound)
        lowest = min([float(np.nanmin(c[4])) for c in curves if np.isfinite(c[4]).any()] + [float(y) for y, _, _ in refs])      # the reference lines stay visible
        ylim = (max(float(ylim[0] or 1.0), 10.0 ** math.floor(math.log10(0.8 * lowest))), ylim[1])
    for k, ax in enumerate(axes.ravel()):
        if k >= len(ms):
            ax.axis("off")
            continue
        m = ms[k]
        for mm, n, steps, med, q25, q75 in curves:
            if mm == m:
                col = DRONE_COLOURS.get(n, "0.3")
                ax.fill_between(steps, q25, q75, color=col, alpha=0.18, linewidth=0)
                ax.plot(steps, med, color=col, ls=DRONE_LINESTYLES.get(n, "-"), lw=1.6)
        for y, label, ls in refs:
            ax.axhline(y, color="0.35", lw=0.9, ls=ls)
            ax.text(0.01, y, label, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=7, color="0.3", clip_on=True)
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(1, xmax)
        ax.set_title(method_label(m), fontsize=9)
        ax.grid(alpha=0.3, which="both" if logy else "major")
        if k + ncol >= len(ms):
            ax.set_xlabel("step")
        if k % ncol == 0:
            ax.set_ylabel(ylabel, fontsize=8)
    handles = [Line2D([0], [0], color=DRONE_COLOURS.get(n, "0.3"), ls=DRONE_LINESTYLES.get(n, "-"), lw=2, label=f"{n} drone{'s' if n > 1 else ''} (median, band = interquartile range)")
               for n in drones]
    fig.legend(handles=handles, loc="outside lower center", ncol=max(1, min(len(handles), int(fig.get_figwidth() // 4.0))), fontsize=8, frameon=False)
    _show_xticks(axes.ravel())
    _suptitle(fig, _title(ev, f"{title} ({len(metrics.GROUPS[CURVE_GROUP])} observable sources, source 110 excluded)", ep_note), fontsize=11)
    return _save(fig, out.path(name, "png")), rows


def build_map_error(ev: EvalSet, out: Outputs, methods: Sequence[str] | None, cap: int | None) -> None:
    series = _series_for(ev, ev.configs(methods), cap)
    refs = [(config.ENV_SUCCESS_ERROR_M, f"{config.ENV_SUCCESS_ERROR_M:.0f} m success", "--"), (config.SUCCESS_ERROR_M, f"{config.SUCCESS_ERROR_M:.0f} m strict", ":")]
    phrase, used = episodes_phrase(ev, series, cap)
    p, rows = _step_curve_by_method(ev, out, "map_error_vs_step", series, lambda s: np.maximum(s.err, 1.0), "MAP error [m]",
                                    "MAP error versus step", logy=True, ylim=(config.T1_4_FIG_YLIM_M[0], None), refs=refs, ep_note=phrase)
    c = _write_csv(out.path("map_error_vs_step", "csv"), rows)
    out.ok("map_error_vs_step", p, "MAP error versus step: median and interquartile band over episodes (log y)", _steps_files(ev),
           "map_error (T) per episode; episode selection from the records: method, n_drones, episode_id, source", "10.4", data_csv=str(c),
           episodes_title=phrase, episodes_used=used, **_miss(series))


def build_entropy(ev: EvalSet, out: Outputs, methods: Sequence[str] | None, cap: int | None) -> None:
    series = _series_for(ev, ev.configs(methods), cap)
    first = np.concatenate([s.entropy[:, 0][np.isfinite(s.entropy[:, 0])] for s in series.values()])
    if first.size == 0:
        raise Skip("step logs carry no entropy")
    h0 = float(np.median(first))                                       # H_0 is not stored in the logs: pooled median of the first-step entropy
    phrase, used = episodes_phrase(ev, series, cap)
    p, rows = _step_curve_figure(ev, out, "entropy_curves", series, lambda s: s.entropy / h0, "belief entropy  H_t / H_0  (median over episodes)",
                                 "Belief entropy reduction", logy=False, ylim=None, refs=[(1.0, "H_0", ":")], zoom_ref=1.0, ep_note=phrase)
    c = _write_csv(out.path("entropy_curves", "csv"), rows)
    what = f"H_t / H_0 versus step, median over episodes (H_0 = pooled median of the first-step entropy = {h0:.3f} nats; the logs do not store H_0)"
    out.ok("entropy_curves", p, what, _steps_files(ev), "entropy (T) per episode", "10.8", data_csv=str(c), h0_nats=h0,
           episodes_title=phrase, episodes_used=used, **_miss(series))


def build_calibration(ev: EvalSet, out: Outputs, methods: Sequence[str] | None, cap: int | None) -> None:
    series = {k: s for k, s in _series_for(ev, ev.configs(methods), cap).items() if np.isfinite(s.inside).any()}
    if not series:
        raise Skip("step logs carry neither calib_inside nor gmm_mu / gmm_cov / truth_xy")
    refs = [(CALIB_NOMINAL, f"nominal {CALIB_NOMINAL:.3f}", "--")]
    phrase, used = episodes_phrase(ev, series, cap)
    p, rows = _step_curve_figure(ev, out, "calibration", series, lambda s: s.inside, "fraction of episodes with truth inside the 2-sigma ellipse",
                                 "Belief calibration of the top GMM component", logy=False, ylim=(0.0, 1.0), refs=refs, reducer="mean", ep_note=phrase)
    c = _write_csv(out.path("calibration", "csv"), rows)
    what = f"Fraction of episodes whose truth lies inside the 2-sigma ellipse of the top GMM component versus step (nominal {CALIB_NOMINAL:.3f})"
    out.ok("calibration", p, what, _steps_files(ev), "calib_inside (T) or gmm_mu, gmm_cov, gmm_w, gmm_mask, truth_xy", "10.10", data_csv=str(c),
           episodes_title=phrase, episodes_used=used, **_miss(series))


def build_threshold_curve(ev: EvalSet, out: Outputs, methods: Sequence[str] | None, cap: int | None) -> None:
    series = _series_for(ev, ev.configs(methods), cap)
    ms = order_methods({m for m, _ in series})
    drones = sorted({n for _, n in series})
    fig, axes = _drone_panels(drones, w=6.2, h=4.6)
    rows = []
    grid = np.array(THRESHOLD_GRID_M, dtype=float)
    for ax, n in zip(axes, drones):
        for (m, nn), s in series.items():
            if nn != n:
                continue
            sel = s.select(metrics.GROUPS[CURVE_GROUP])
            if not sel.any():
                continue
            with np.errstate(invalid="ignore"):
                ok_sigma = s.sigma[sel] < config.ENV_SUCCESS_SIGMA_M
            best = np.min(np.where(ok_sigma, s.err[sel], np.inf), axis=1)    # best MAP error among the steps with sigma < the success sigma
            rate = np.array([np.mean(best < t) for t in grid])
            ax.plot(grid, rate, marker="o", ms=3, **_style(m, n, ms))
            for t, r in zip(grid, rate):
                _, lo, hi = metrics.wilson_ci(int(np.sum(best < t)), int(best.size))
                rows.append({"tag": ev.tag, "method": m, "n_drones": n, "error_threshold_m": float(t), "n_episodes": int(best.size),
                             "success_rate": float(r), "ci_lo": lo, "ci_hi": hi})
        for x, label in ((config.SUCCESS_ERROR_M, f"{config.SUCCESS_ERROR_M:.0f} m error"), (config.ENV_SUCCESS_ERROR_M, f"{config.ENV_SUCCESS_ERROR_M:.0f} m error (primary)")):
            ax.axvline(x, color="0.4", lw=0.9, ls=":")                  # the curve itself always requires top sigma < ENV_SUCCESS_SIGMA_M: the lines mark error thresholds only
            ax.text(x + 0.8, 0.985, label, fontsize=7, color="0.3", va="top")
        ax.set_xlabel(f"MAP error threshold [m]  (top sigma < {config.ENV_SUCCESS_SIGMA_M:.0f} m at the same step)")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{n} drone{'s' if n > 1 else ''}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate")
    _legend(fig, ms, drones)
    phrase, used = episodes_phrase(ev, series, cap)
    _suptitle(fig, _title(ev, f"Success rate versus MAP-error threshold ({len(metrics.GROUPS[CURVE_GROUP])} observable sources)", phrase), fontsize=11)
    p = _save(fig, out.path("threshold_curve", "png"))
    c = _write_csv(out.path("threshold_curve", "csv"), rows)
    what = "Success rate versus MAP-error threshold 20..100 m with top sigma below the success sigma, from the per-step logs"
    out.ok("threshold_curve", p, what, _steps_files(ev), "map_error (T), top_sigma (T) per episode", "9.1", data_csv=str(c),
           episodes_title=phrase, episodes_used=used, **_miss(series))


# ------------------------------------------------------------------------------------------------ 5 heatmap
def build_heatmap(ev: EvalSet, out: Outputs, methods: Sequence[str] | None = None) -> None:
    cfgs = ev.configs(methods)
    if not cfgs:
        raise Skip("no evaluation records")
    srcs = list(GROUP_ORDER_SOURCES)
    mat = np.full((len(cfgs), len(srcs)), np.nan)
    cnt = np.zeros((len(cfgs), len(srcs)), dtype=int)
    rows = []
    for i, (m, n) in enumerate(cfgs):
        recs = ev.recs(m, n)
        for j, s in enumerate(srcs):
            rs = [r for r in recs if int(r["source"]) == s]
            if rs:
                k = sum(bool(r["success"]) for r in rs)
                mat[i, j], cnt[i, j] = k / len(rs), len(rs)
                rows.append({"tag": ev.tag, "method": m, "n_drones": n, "source": s, "n": len(rs), "n_success": k, "success_rate": k / len(rs),
                             "observable": s not in config.EXCLUDED_SOURCES})
    unobs = [j for j, s in enumerate(srcs) if s in config.EXCLUDED_SOURCES]
    shown = np.ma.masked_invalid(mat.copy())
    for j in unobs:
        shown[:, j] = np.ma.masked
    fig, ax = plt.subplots(figsize=(1.0 * len(srcs) + 3.4, 0.42 * len(cfgs) + 2.8), layout="constrained")
    cmap = plt.get_cmap(HEATMAP_CMAP).with_extremes(bad="#d9d9d9")
    im = ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    for i in range(len(cfgs)):
        for j in range(len(srcs)):
            if cnt[i, j] == 0:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="0.45")
                continue
            v = mat[i, j]
            dark = (j not in unobs) and v > 0.55
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8, color="white" if dark else ("0.35" if j in unobs else "0.1"))
    ax.set_xticks(range(len(srcs)))
    ax.set_xticklabels([str(s) if s not in config.EXCLUDED_SOURCES else f"{s}\nunobs." for s in srcs], fontsize=8)
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([cfg_label(m, n) for m, n in cfgs], fontsize=8)
    start = 0
    for g in ("train_open", "train_other", "holdout", "unobservable"):
        k = len(metrics.GROUPS[g])
        if start:
            ax.axvline(start - 0.5, color="k", lw=1.2)
        ax.text(start + k / 2 - 0.5, 1.012, GROUP_SHORT[g].replace("\n", " "), ha="center", va="bottom", fontsize=8, color="0.2", transform=ax.get_xaxis_transform())
        start += k
    ax.set_xlabel("source")
    ax.set_xticks(np.arange(-0.5, len(srcs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cfgs), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("success rate")
    ax.set_title(_title(ev, "Success rate by source and method (cell = % of the episodes of that source; grey = 110, unobservable at 15 m)"), fontsize=10, pad=30)
    p = _save(fig, out.path("source_heatmap", "png"))
    c = _write_csv(out.path("source_heatmap", "csv"), rows)
    out.ok("source_heatmap", p, "Source x method success rate (13 sources; 110 grey = unobservable at 15 m)", _records_files(ev, cfgs),
           "method, n_drones, episode_id, source, success", "10.5", data_csv=str(c))


# ------------------------------------------------------------------------------------------------ 6 paired differences
def default_reference(ev: EvalSet) -> str | None:
    """gmm_infotaxis (the strongest baseline of the plan: M2 - B3) when present, else the first non-verification method."""
    ms = [m for m in ev.methods() if m not in VERIFICATION_METHODS]
    if "gmm_infotaxis" in ms:
        return "gmm_infotaxis"
    return ms[0] if ms else None


def _bootstrap_mean_ci(x: Sequence[float], n_boot: int, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    x = np.asarray(list(x), dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


R_PAIRED_GROUP_SCOPES = ("train_open", "train_other", "holdout", "train")      # metrics.GROUPS keys drawn as group rows of the forest plot (spec 9.6: train / holdout split)
RATIO_COLUMNS = ("n_common", "n_success_hi", "n_success_lo", "n_both_success", "step_median_hi", "step_median_lo", "step_ratio_of_medians",
                 "censored_median_hi", "censored_median_lo", "censored_ratio", "step_ratio_paired_median", "step_ratio_paired_lo", "step_ratio_paired_hi",
                 "ideal_step_ratio", "flight_time_median_hi", "flight_time_median_lo", "flight_time_ratio_of_medians", "flight_time_paired_ratio_median",
                 "flight_time_paired_ratio_lo", "flight_time_paired_ratio_hi", "censored_flight_time_ratio")
PAIRED_FIELDS = ("tag", "comparison", "method", "reference", "n_drones", "n_drones_reference", "scope", "scope_kind", "n", "success_diff", "success_diff_lo",
                 "success_diff_hi", "n_method_only", "n_reference_only", "censored_step_diff_median", "censored_step_diff_lo", "censored_step_diff_hi") + RATIO_COLUMNS


def paired_scopes() -> dict[str, tuple[str, set[int]]]:
    """{scope name: (kind, source ids)} of the forest plot: every observable source, then the groups train_open / train_other / holdout / train,
    then 'all observable' (source 110 is unobservable and never a scope)."""
    scopes: dict[str, tuple[str, set[int]]] = {str(s): ("source", {s}) for s in GROUP_ORDER_SOURCES if s not in config.EXCLUDED_SOURCES}
    for g in R_PAIRED_GROUP_SCOPES:
        scopes[g] = ("group", set(metrics.GROUPS[g]))
    scopes["all observable"] = ("overall", set(metrics.GROUPS["all_observable"]))
    return scopes


def _nanmed(x: Sequence[float]) -> float:
    x = [float(v) for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def _div(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else float("nan")


def ratio_stats(ra: Iterable[dict], rb: Iterable[dict], n_hi: int, n_lo: int, horizon: int, n_boot: int) -> dict:
    """Drone-count effect of ONE method on the common episode ids: ``ra`` = records flown with ``n_hi`` drones, ``rb`` = with ``n_lo`` drones
    (spec 9.11).  Success-step ratio hi / lo (ideal n_lo / n_hi: 0.5 for 2 versus 1) as the ratio of the success-step medians (successes only),
    as the ratio of the censored medians (failures = horizon) and as the median of the per-episode ratios over the episodes both teams
    solved (bootstrap CI); total flight time = drones x steps, as the median per team size and as the ratio hi / lo (ideal 1 = no extra effort)."""
    out: dict[str, float] = dict.fromkeys(RATIO_COLUMNS, float("nan"))
    a = {int(r["episode_id"]): r for r in ra}
    b = {int(r["episode_id"]): r for r in rb}
    pairs = [(a[i], b[i]) for i in sorted(set(a) & set(b))]
    out["n_common"] = len(pairs)
    out["ideal_step_ratio"] = float(n_lo / n_hi)
    if not pairs:
        return out
    sh = [int(h["steps"]) for h, _ in pairs if h["success"]]
    sl = [int(l["steps"]) for _, l in pairs if l["success"]]
    out["n_success_hi"], out["n_success_lo"] = len(sh), len(sl)
    out["step_median_hi"], out["step_median_lo"] = _nanmed(sh), _nanmed(sl)
    out["step_ratio_of_medians"] = _div(out["step_median_hi"], out["step_median_lo"])
    ch = [int(h["steps"]) if h["success"] else int(horizon) for h, _ in pairs]
    cl = [int(l["steps"]) if l["success"] else int(horizon) for _, l in pairs]
    out["censored_median_hi"], out["censored_median_lo"] = _nanmed(ch), _nanmed(cl)
    out["censored_ratio"] = _div(out["censored_median_hi"], out["censored_median_lo"])
    both = [int(h["steps"]) / int(l["steps"]) for h, l in pairs if h["success"] and l["success"] and int(l["steps"]) > 0]
    out["n_both_success"] = len(both)
    if both:
        med, lo, hi = metrics.bootstrap_median_ci(both, n_boot)
        out["step_ratio_paired_median"], out["step_ratio_paired_lo"], out["step_ratio_paired_hi"] = med, lo, hi
        k = n_hi / n_lo
        out["flight_time_paired_ratio_median"], out["flight_time_paired_ratio_lo"], out["flight_time_paired_ratio_hi"] = k * med, k * lo, k * hi
    out["flight_time_median_hi"], out["flight_time_median_lo"] = n_hi * out["step_median_hi"], n_lo * out["step_median_lo"]
    out["flight_time_ratio_of_medians"] = _div(out["flight_time_median_hi"], out["flight_time_median_lo"])
    out["censored_flight_time_ratio"] = _div(n_hi * out["censored_median_hi"], n_lo * out["censored_median_lo"])
    return out


def paired_table(ev: EvalSet, method: str, reference: str, n: int, scopes: dict[str, Any], n_boot: int, n_ref: int | None = None) -> list[dict]:
    """Rows of (method with ``n`` drones) - (reference with ``n_ref`` drones, default ``n``) on the common episode ids, per scope (source id or
    group; ``scopes`` maps a name to a source-id set or to (kind, set) as returned by paired_scopes): success-rate difference with a bootstrap CI
    over episodes (not provided by metrics.paired_differences) and the censored success-step difference from metrics.paired_differences
    (median with bootstrap CI).  With ``n_ref`` different from ``n`` (same method, ``reference`` = ``method``) the row is a drone-count pair
    (comparison 'drone_count') and also carries the ratio_stats columns; plain pairs leave them empty."""
    n_ref = n if n_ref is None else int(n_ref)
    drone_count = n_ref != n
    a_all, b_all = ev.recs(method, n), ev.recs(reference, n_ref)
    rows = []
    for name, spec in scopes.items():
        kind, srcs = spec if isinstance(spec, tuple) else ("source", spec)
        ra = [r for r in a_all if int(r["source"]) in srcs]
        rb = [r for r in b_all if int(r["source"]) in srcs]
        pd = metrics.paired_differences(ra, rb, max_steps=ev.horizon)
        if not pd.get("n"):
            continue
        da = {int(r["episode_id"]): int(bool(r["success"])) for r in ra}
        db = {int(r["episode_id"]): int(bool(r["success"])) for r in rb}
        diff = [da[i] - db[i] for i in sorted(set(da) & set(db))]
        _, lo, hi = _bootstrap_mean_ci(diff, n_boot)
        row = {"tag": ev.tag, "comparison": "drone_count" if drone_count else "method_vs_reference", "method": method, "reference": reference,
               "n_drones": n, "n_drones_reference": n_ref, "scope": name, "scope_kind": kind, "n": pd["n"], "success_diff": pd["success_rate_diff"],
               "success_diff_lo": lo, "success_diff_hi": hi, "n_method_only": pd["n_a_only"], "n_reference_only": pd["n_b_only"],
               "censored_step_diff_median": pd["censored_step_diff_median"], "censored_step_diff_lo": pd["censored_step_diff_ci"][0],
               "censored_step_diff_hi": pd["censored_step_diff_ci"][1]}
        row.update(ratio_stats(ra, rb, n, n_ref, ev.horizon, n_boot) if drone_count else dict.fromkeys(RATIO_COLUMNS))
        rows.append(row)
    return rows


def _ratio_axes(ax: plt.Axes, pairs: Sequence[tuple[str, int, int]], rows: Sequence[dict], key: str, lo_key: str | None, hi_key: str | None, alt_key: str,
                ylabel: str, ideal: Callable[[tuple[str, int, int]], float | None], ideal_label: str, ylim_min: float = 0.0) -> None:
    """Drone-count ratio of every (method, n_hi, n_lo) pair: median of the per-episode ratios with bootstrap CI (disc) and ratio of the group medians
    (hollow diamond), dashed segment = ideal."""
    tops = []
    for k, (m, nh, nl) in enumerate(pairs):
        row = next((r for r in rows if r["method"] == m and r["n_drones"] == nh and r["n_drones_reference"] == nl and r["scope"] == "all observable"), None)
        if row is None:
            continue
        v, a = row.get(key), row.get(alt_key)
        if v is not None and np.isfinite(v):
            lo, hi = row.get(lo_key), row.get(hi_key)
            ax.errorbar(k, v, yerr=[[max(v - lo, 0.0)], [max(hi - v, 0.0)]] if lo is not None and np.isfinite(lo) else None, fmt="o", ms=6, capsize=3,
                        color=DRONE_COLOURS.get(nh, "0.3"), elinewidth=1.4)
            tops.append(hi if hi is not None and np.isfinite(hi) else v)
            ax.annotate(f"n={int(row['n_both_success'])}", (k, v), textcoords="offset points", xytext=(7, 4), fontsize=7, color="0.3")
        else:
            ax.text(k, 0.02, "n=0", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="0.4")      # no episode solved by both teams
        if a is not None and np.isfinite(a):
            ax.plot([k], [a], marker="D", ms=6, mfc="none", mec="0.15", mew=1.2, ls="none")
            tops.append(a)
        i = ideal((m, nh, nl))
        if i is not None:
            ax.hlines(i, k - 0.38, k + 0.38, colors="0.25", linestyles="--", lw=1.1)
            tops.append(i)
    ax.set_xticks(range(len(pairs)))
    many = len(pairs) >= 4                                                    # rotate the tick labels instead of letting neighbouring method names touch
    ax.set_xticklabels([(f"{m} ({nh} vs {nl})" if many else f"{wrap_method(m)}\n{nh} vs {nl}") for m, nh, nl in pairs], fontsize=7,
                       rotation=40 if many else 0, ha="right" if many else "center")
    ax.set_xlim(-0.6, len(pairs) - 0.4)
    ax.set_ylim(ylim_min, max([1.15 * t for t in tops if np.isfinite(t)] + [0.1]))
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(textwrap.fill(ideal_label + "; n = episodes solved by both teams", 60), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=7)


def _forest_axes(ax: plt.Axes, sel: Sequence[dict], names: Sequence[str], kinds: dict[str, str], key: str, lo_k: str, hi_k: str, xlabel: str,
                 span: tuple[float, float], show_names: bool) -> None:
    """One forest panel: a marker per scope row and drone count (colour = drone count, marker size / line width by scope kind: source < group < overall)."""
    ns = sorted({r["n_drones"] for r in sel})
    offs = {n: (0.0 if len(ns) == 1 else -0.17 + 0.34 * i / (len(ns) - 1)) for i, n in enumerate(ns)}
    for n in ns:
        for k, name in enumerate(names):
            row = next((r for r in sel if r["n_drones"] == n and r["scope"] == name), None)
            if row is None:
                continue
            ms_, lw_, cap_ = {"overall": (6.5, 1.8, 2), "group": (5.0, 1.3, 2), "source": (3.6, 0.8, 0)}[kinds[name]]
            ax.errorbar(row[key], k + offs[n], xerr=[[max(row[key] - row[lo_k], 0.0)], [max(row[hi_k] - row[key], 0.0)]], fmt=DRONE_MARKERS.get(n, "o"),
                        color=DRONE_COLOURS.get(n, "0.3"), ms=ms_, lw=lw_, capsize=cap_, alpha=1.0 if kinds[name] != "source" else 0.8)
    ax.axvline(0.0, color="0.2", lw=0.8)
    for k in range(1, len(names)):
        if kinds[names[k]] != kinds[names[k - 1]]:
            ax.axhline(k - 0.5, color="0.6", lw=0.8)
    ax.set_xlim(*span)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names if show_names else [], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", alpha=0.3)


def build_paired(ev: EvalSet, out: Outputs, reference: str | None, methods: Sequence[str] | None, n_boot: int) -> None:
    """Paired-difference figure (spec 9.6 / 10.6): tier 1 = every method minus the reference per source, per group (train_open / train_other / holdout /
    train) and overall; tier 2 = the same-method team-size pairs (n drones minus the fewest drones) per group and overall; tier 3 = their
    success-step ratio (ideal n_lo / n_hi) and total flight time ratio.  The CSV holds every scope (also the per-source pairs)."""
    ref = reference or default_reference(ev)
    if ref is None:
        raise Skip("no reference method available")
    if ref not in ev.methods():
        raise Skip(f"reference method {ref} not in this evaluation directory (methods: {', '.join(ev.methods())})")
    pool = [m for m in ev.methods() if methods is None or m in methods]
    others = [m for m in pool if m != ref and m not in VERIFICATION_METHODS][:PAIRED_MAX_METHODS]
    pair_methods = [m for m in pool if m not in VERIFICATION_METHODS and len(ev.drone_counts([m])) > 1][:PAIRED_MAX_METHODS]
    if not others and not pair_methods:
        raise Skip(f"no method to compare with the reference {ref}")
    scopes = paired_scopes()
    rows = []
    for m in others:
        for n in ev.drone_counts():
            if ev.recs(m, n) and ev.recs(ref, n):
                rows += paired_table(ev, m, ref, n, scopes, n_boot)
    for m in pair_methods:                                                       # same-method team-size pairs: (n drones) - (fewest drones), spec 9.6 / 9.11
        counts = ev.drone_counts([m])
        for n in counts[1:]:
            rows += paired_table(ev, m, m, n, scopes, n_boot, n_ref=counts[0])
    if not rows:
        raise Skip(f"no common episodes between the methods and the reference {ref} (same drone count needed)")
    used = [m for m in others if any(r["method"] == m and r["comparison"] == "method_vs_reference" for r in rows)]
    dc = [r for r in rows if r["comparison"] == "drone_count"]
    pairs = [(m, n, int(next(r["n_drones_reference"] for r in dc if r["method"] == m and r["n_drones"] == n)))
             for m in pair_methods for n in sorted({r["n_drones"] for r in dc if r["method"] == m})]
    pair_cols = list(dict.fromkeys(p[0] for p in pairs))
    names = list(scopes)
    kinds = {k: v[0] for k, v in scopes.items()}
    names2 = [k for k in names if kinds[k] != "source"]                         # tier 2: groups and overall (per-source pairs are in the CSV)
    metric_specs = (("success_diff", "success_diff_lo", "success_diff_hi", "success rate difference\n(positive = better than the reference)", (-1.05, 1.05)),
                    ("censored_step_diff_median", "censored_step_diff_lo", "censored_step_diff_hi", "censored success-step difference [steps]\n(negative = faster than the reference)",
                     (-1.05 * ev.horizon, 1.05 * ev.horizon)))
    ncols = max(len(used), len(pair_cols), 3)
    h1 = 2 * (0.26 * len(names) + 1.6) if used else 0.0
    h2 = 2 * (0.3 * len(names2) + 1.5) if pair_cols else 0.0
    h3 = 3.4 if pairs else 0.0
    tiers = [(k, h) for k, h in (("ref", h1), ("pair", h2), ("ratio", h3)) if h > 0]
    fig = plt.figure(figsize=(max(3.1 * ncols, 9.0), sum(h for _, h in tiers) + 1.4), layout="constrained")
    outer = fig.add_gridspec(len(tiers), 1, height_ratios=[h for _, h in tiers])
    grid = {k: outer[i] for i, (k, _) in enumerate(tiers)}
    if used:
        g = grid["ref"].subgridspec(2, len(used))
        for c, m in enumerate(used):
            sel = [r for r in rows if r["method"] == m and r["comparison"] == "method_vs_reference"]
            for r_i, (key, lo_k, hi_k, xl, span) in enumerate(metric_specs):
                ax = fig.add_subplot(g[r_i, c])
                _forest_axes(ax, sel, names, kinds, key, lo_k, hi_k, xl, span, show_names=c == 0)
                if r_i == 0:
                    ax.set_title(f"{m} - {ref}", fontsize=9)
    if pair_cols:
        g = grid["pair"].subgridspec(2, len(pair_cols))
        for c, m in enumerate(pair_cols):
            sel = [r for r in dc if r["method"] == m]
            ns = sorted({r["n_drones"] for r in sel})
            lows = sorted({r["n_drones_reference"] for r in sel})
            for r_i, (key, lo_k, hi_k, xl, span) in enumerate(metric_specs):
                ax = fig.add_subplot(g[r_i, c])
                _forest_axes(ax, sel, names2, kinds, key, lo_k, hi_k, xl.replace("the reference", f"{lows[0]} drone{'s' if lows[0] > 1 else ''}"), span, show_names=c == 0)
                if r_i == 0:
                    ax.set_title(f"{m}: {'/'.join(str(n) for n in ns)} - {'/'.join(str(n) for n in lows)} drones", fontsize=9)
    if pairs:
        sub = grid["ratio"].subgridspec(1, 2, wspace=0.12)
        ax1 = fig.add_subplot(sub[0, 0])
        _ratio_axes(ax1, pairs, dc, "step_ratio_paired_median", "step_ratio_paired_lo", "step_ratio_paired_hi", "step_ratio_of_medians",
                    "success-step ratio  hi / lo drones", lambda p: p[2] / p[1], "success-step ratio (dashed = ideal n_lo / n_hi, 0.5 for 2 vs 1)")
        ax2 = fig.add_subplot(sub[0, 1])
        _ratio_axes(ax2, pairs, dc, "flight_time_paired_ratio_median", "flight_time_paired_ratio_lo", "flight_time_paired_ratio_hi", "flight_time_ratio_of_medians",
                    "total flight time ratio  hi / lo", lambda p: 1.0, "total flight time = drones x steps (dashed = equal effort, 1)")
    handles = [Line2D([0], [0], marker=DRONE_MARKERS.get(n, "o"), color=DRONE_COLOURS.get(n, "0.3"), lw=0.8, ms=6, label=f"{n} drone{'s' if n > 1 else ''}")
               for n in sorted({r["n_drones"] for r in rows})]
    if pairs:
        handles += [Line2D([0], [0], marker="o", color="0.3", lw=1.2, ms=6, label="median of per-episode ratios (episodes both solved), 95 % CI"),
                    Line2D([0], [0], marker="D", mfc="none", mec="0.15", ls="none", ms=6, label="ratio of the medians (common episodes)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 3 if fig.get_figwidth() < 13 else 5), fontsize=8, frameon=False)
    _suptitle(fig, _title(ev, f"Paired differences: method - {ref} (top: per source and per group), same-method team-size pairs (middle: group rows; per-source in the CSV) "
                              "and their success-step / flight-time ratios (bottom); common episodes, bootstrap 95 % CI, 110 excluded"), fontsize=9)
    p = _save(fig, out.path("paired_diff", "png"))
    c = _write_csv(out.path("paired_diff", "csv"), rows, PAIRED_FIELDS)
    data = _records_files(ev, [(m, n) for m in used + [ref] + pair_methods for n in ev.drone_counts() if ev.recs(m, n)])
    what = (f"Paired differences method - {ref} and same-method n-versus-fewest-drones pairs (success rate and censored success step) per source, per group "
            "(train_open, train_other, holdout, train) and overall, with bootstrap CI; drone-count pairs add the success-step ratio (ideal n_lo / n_hi) and total flight time")
    out.ok("paired_diff", p, what, data, "method, n_drones, episode_id, source, success, steps", "10.6 / 9.6 / 9.11", data_csv=str(c), reference=ref,
           drone_count_pairs=[f"{m}: {nh} vs {nl}" for m, nh, nl in pairs])


# ------------------------------------------------------------------------------------------------ 7 trajectories
def _preferred_configs(ev: EvalSet, n: int, preferred: str | None = None, methods: Sequence[str] | None = None) -> list[tuple[str, int]]:
    """Configurations with ``n`` drones in the order of preference for the representative episodes: the requested method, then trained policies
    (anything that is not a baseline or a verification method), then baselines; within a class the higher all-observable success rate first."""
    cfgs = [(m, k) for m, k in ev.configs(methods) if k == n and m not in VERIFICATION_METHODS]

    def key(c: tuple[str, int]) -> tuple:
        rate = ev.aggregate(*c).get("all_observable", {}).get("success_rate", -1.0)
        klass = 0 if c[0] == preferred else (1 if c[0] not in CANONICAL_METHODS + VARIANT_METHODS else 2)
        return (klass, -rate, c[0])
    return sorted(cfgs, key=key)


def _pick(cands: list[dict], value: Callable[[dict], float]) -> dict | None:
    """Representative candidate: the one closest to the median of ``value`` (ties: smaller episode id)."""
    if not cands:
        return None
    med = float(np.median([value(r) for r in cands]))
    return sorted(cands, key=lambda r: (abs(value(r) - med), r["episode_id"]))[0]


def select_trajectory_episodes(ev: EvalSet, preferred: str | None = None, methods: Sequence[str] | None = None) -> list[dict]:
    """Four representative episodes from the records (no cherry-picking by figure appearance): success on an open source, success on a trapped
    source, a failure, and a multi-drone episode.  Each item: {key, title, record or None, note}.  The median-step success (or the median-error
    failure) of the best configuration that has such an episode is taken; kinds that do not exist are reported in ``note``."""
    drones = ev.drone_counts(methods)
    n1 = min(drones) if drones else 1
    n2 = max(drones) if drones and max(drones) > 1 else None         # the multi-drone panel flies the LARGEST drone count (works when every configuration has several drones)
    obs = set(metrics.GROUPS["all_observable"])
    steps_of, err_of = (lambda r: float(r["steps"])), (lambda r: float(r["final_error_m"]))

    def find(n: int, cond: Callable[[dict], bool], value: Callable[[dict], float]) -> dict | None:
        for c in _preferred_configs(ev, n, preferred, methods):
            r = _pick([x for x in ev.recs(*c) if cond(x)], value)
            if r is not None:
                return r
        return None

    out = []
    r = find(n1, lambda x: x["success"] and int(x["source"]) in OPEN_SOURCES, steps_of)
    out.append({"key": "open_success", "title": "success, open source", "record": r, "note": "" if r else "no success on an open source"})
    r = find(n1, lambda x: x["success"] and int(x["source"]) in TRAPPED_SOURCES, steps_of)
    note = ""
    if r is None:
        r = find(n1, lambda x: x["success"] and int(x["source"]) in obs and int(x["source"]) not in OPEN_SOURCES, steps_of)
        note = "no success on a trapped source (102, 104, 106): another non-open source is shown" if r else "no success on a trapped or other non-open source"
    out.append({"key": "trapped_success", "title": "success, trapped source", "record": r, "note": note})
    r = find(n1, lambda x: (not x["success"]) and int(x["source"]) in obs, err_of)
    out.append({"key": "failure", "title": "failure", "record": r, "note": "" if r else "no failure"})
    if n2 is None:
        out.append({"key": "multi_drone", "title": "2-drone episode", "record": None, "note": "no multi-drone configuration in this evaluation"})
        return out
    first_id = out[0]["record"]["episode_id"] if out[0]["record"] else None
    shown = {x["record"]["episode_id"] for x in out if x["record"]}          # scenes of the panels a-c
    exclusions = [shown, {first_id}, set()]                                   # prefer a scene not shown yet, then a scene other than panel a, then anything
    r, note = None, ""
    for excl in exclusions:
        for c in _preferred_configs(ev, n2, preferred, methods):
            r = _pick([x for x in ev.recs(*c) if x["success"] and int(x["source"]) in obs and x["episode_id"] not in excl], steps_of)
            if r is not None:
                break
        if r is not None:
            break
    if r is None:
        for excl in exclusions:
            for c in _preferred_configs(ev, n2, preferred, methods):
                r = min((x for x in ev.recs(*c) if int(x["source"]) in obs and x["episode_id"] not in excl), key=err_of, default=None)
                if r is not None:
                    note = "no multi-drone success: the episode with the smallest final error is shown"
                    break
            if r is not None:
                break
    if r is not None and first_id is not None and r["episode_id"] == first_id and not note:
        note = "same scene as panel a: no other multi-drone episode available"
    out.append({"key": "multi_drone", "title": f"{n2}-drone episode", "record": r, "note": note if r else "no multi-drone episode"})
    return out


def _square_window(points: np.ndarray, margin: float, min_side: float) -> tuple[float, float, float, float]:
    """Square window (x0, y0, x1, y1) around the points so that every trajectory panel has the same shape and scale."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    side = max(float((hi - lo).max()) + 2.0 * margin, min_side)
    c = 0.5 * (lo + hi)
    return float(c[0] - side / 2), float(c[1] - side / 2), float(c[0] + side / 2), float(c[1] + side / 2)


def _draw_ellipse(ax: plt.Axes, mu: np.ndarray, cov: np.ndarray, weight: float, colour: str) -> None:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * CALIB_SIGMAS * np.sqrt(vals[1]), 2 * CALIB_SIGMAS * np.sqrt(vals[0]), angle=ang, fill=False, edgecolor=colour,
                         linewidth=1.0 + 2.0 * weight, alpha=0.55 + 0.45 * weight, zorder=5))


def episode_density_frame(arrs: dict[str, np.ndarray], meta: dict, mode: str, n_files: int = config.N_FILES,
                          files_per_step: float | None = None) -> tuple[int, int]:
    """(frame, step) of the truth shown behind the trajectory: Mode F = the episode's frozen frame; Mode T2 = the frame at the middle step
    (logged ``frame`` array, else round(start frame + config.T1_4_MODE_T2_FILES_PER_STEP * t) clipped at the last cached frame; t = 0-based step
    index as in SourceLocEnv.frame_at)."""
    T = int(arrs["drone_xy"].shape[0])
    if str(meta.get("mode", mode)) != "T2":
        return int(meta["frame"]), T
    mid = T // 2
    if "frame" in arrs:
        return int(arrs["frame"][min(mid, len(arrs["frame"]) - 1)]), mid + 1
    fps = config.T1_4_MODE_T2_FILES_PER_STEP if files_per_step is None else files_per_step
    return int(min(n_files - 1, round(int(meta["frame"]) + fps * mid))), mid + 1


def draw_episode_panel(ax: plt.Axes, arrs: dict[str, np.ndarray], meta: dict, mode: str, backend: Any, obstacles: Any, title: str) -> Any:
    """One trajectory panel (spec 10 item 7): buildings, truth-frame density of the source, drone paths coloured by step (colour bar), detection steps
    (count above the Currie threshold) ringed, start markers, GMM ellipses at the end, MAP estimate and true source.  Returns the path scatter
    (for the shared colour bar)."""
    T = int(arrs["drone_xy"].shape[0])
    D = int(arrs["drone_xy"].shape[1])
    truth = np.asarray(arrs["truth_xy"], dtype=float)
    frame, mid_step = episode_density_frame(arrs, meta, mode)
    sf = backend.slab(frame)
    g = sf.grid
    dens = np.asarray(sf.density[list(sf.sources).index(int(meta["source"])), sf.z_index(config.DRONE_Z)], dtype=float)
    vmax = max(float(dens.max()), 1e-12)
    ax.imshow(np.log10(np.maximum(dens, vmax * 10.0 ** -config.FIG_LOG_DECADES)), origin="lower",
              extent=(g.x0, g.x0 + g.nx * g.res, g.y0, g.y0 + g.ny * g.res), cmap="Greys", vmin=np.log10(vmax) - config.FIG_LOG_DECADES, vmax=np.log10(vmax),
              alpha=0.75, interpolation="nearest", zorder=1)
    ox, oy = np.meshgrid(obstacles.x, obstacles.y, indexing="ij")
    ax.contour(ox, oy, np.asarray(obstacles.occ, dtype=float), levels=[0.5], colors="0.3", linewidths=0.7, zorder=2)
    pts = [arrs["drone_xy"].reshape(-1, 2), truth.reshape(1, 2)]
    if "start_xy" in arrs:
        pts.append(np.asarray(arrs["start_xy"], dtype=float).reshape(-1, 2))
    win = _square_window(np.vstack(pts), TRAJ_MARGIN_M, TRAJ_MIN_SIDE_M)
    steps = np.arange(1, T + 1)
    sc = None
    det = detection_mask(arrs["y"]).reshape(T, -1) if "y" in arrs and np.asarray(arrs["y"]).size == T * D else None
    for d in range(D):
        xy = arrs["drone_xy"][:, d]
        ax.plot(xy[:, 0], xy[:, 1], "-", color="0.15", lw=0.5, alpha=0.5, zorder=3)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=steps, s=7, cmap="viridis", vmin=1, vmax=max(T, 2), zorder=4, linewidths=0)
        if det is not None and det[:, d].any():                       # ring the steps whose count exceeded the Currie threshold (the step colour stays visible inside)
            hit = np.flatnonzero(det[:, d])
            ax.scatter(xy[hit, 0], xy[hit, 1], s=R_DETECTION_MARKER_AREA, facecolors="none", edgecolors=R_DETECTION_COLOUR, linewidths=1.1, zorder=6)
        st = np.asarray(arrs["start_xy"], dtype=float).reshape(-1, 2)[d] if "start_xy" in arrs else xy[0]
        ax.plot(*st, marker="s" if d == 0 else "D", ms=7, mfc="white", mec="k", mew=1.4, zorder=7, ls="none")
        ax.plot(*xy[-1], marker="o", ms=5, mfc="none", mec="k", mew=1.2, zorder=7, ls="none")
    if all(k in arrs for k in ("gmm_w", "gmm_mu", "gmm_cov", "gmm_mask")):
        w, mu, cov, mask = arrs["gmm_w"][-1], arrs["gmm_mu"][-1], arrs["gmm_cov"][-1], arrs["gmm_mask"][-1]
        for j in range(len(w)):
            if mask[j]:
                _draw_ellipse(ax, mu[j], cov[j], float(w[j]), "tab:red")
    if "map_xy" in arrs:
        ax.plot(*arrs["map_xy"][-1], marker="x", color="tab:red", ms=9, mew=2.2, zorder=8, ls="none")
    ax.plot(*truth, marker="*", color="gold", mec="k", ms=15, zorder=9, ls="none")
    ax.set_xlim(win[0], win[2])
    ax.set_ylim(win[1], win[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    dens_note = f"truth frame {frame} (step {mid_step} of {T})" if str(meta.get("mode", mode)) == "T2" else f"frozen truth frame {frame}"
    ax.set_title(f"{title}\n{dens_note}", fontsize=8)
    return sc


def get_scene_backends() -> tuple[Any, Any]:
    """(slab backend, ObstacleMap): the stacked 15 m slab (all 600 frames) when its cache exists, else the per-frame slab files."""
    from srcloc_env.env.drone import ObstacleMap
    from srcloc_env.field.concentration_field import LdmSlabBackend
    if Path(config.SLAB_STACK_PATH).is_file() and Path(config.SLAB_STACK_META_PATH).is_file():
        from srcloc_env.field.slab_stack import StackedSlabBackend
        be = StackedSlabBackend()
    else:
        be = LdmSlabBackend()
    return be, ObstacleMap.load()


def build_trajectories(ev: EvalSet, out: Outputs, preferred: str | None, methods: Sequence[str] | None, backend: Any = None, obstacles: Any = None) -> None:
    if not ev.records:
        raise Skip("no evaluation records")
    sel = select_trajectory_episodes(ev, preferred, methods)
    if all(s["record"] is None for s in sel):
        raise Skip("no representative episode found in the records")
    panels = []
    for s in sel:
        r = s["record"]
        if r is None:
            panels.append((s, None, None, s["note"]))
            continue
        p = ev.step_log_path(r)
        if p is None:
            panels.append((s, None, None, "step log not found"))
            continue
        try:
            arrs, meta = read_step_arrays(p)
        except _READ_ERRORS as e:
            panels.append((s, None, None, f"step log unreadable ({e})"))
            continue
        panels.append((s, arrs, meta, s["note"]))
    if all(a is None for _, a, _, _ in panels):
        raise Skip("; ".join(f"{s['key']}: {n}" for s, _, _, n in panels))
    if backend is None or obstacles is None:
        try:
            be2, om2 = get_scene_backends()
        except (OSError, ImportError) as e:
            raise Skip(f"slab / building data not available ({e})") from e
        backend, obstacles = backend or be2, obstacles or om2
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 12.4), layout="constrained")
    sc = None
    used, remarks = [], []
    for k, (ax, (s, arrs, meta, note)) in enumerate(zip(axes.ravel(), panels)):
        if arrs is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"({chr(97 + k)}) {s['title']}\nnot available: {note}", ha="center", va="center", fontsize=10, color="0.35", transform=ax.transAxes)
            remarks.append(f"{s['key']}: {note}")
            continue
        r = s["record"]
        ok = f"success at step {r['steps']}" if r["success"] else "no success"
        n_det = int(detection_mask(arrs["y"]).reshape(arrs["drone_xy"].shape[0], -1).any(axis=1).sum()) if "y" in arrs else None
        det_txt = "" if n_det is None else f"; detection on {n_det} of {arrs['drone_xy'].shape[0]} steps"
        title = (f"({chr(97 + k)}) {s['title']}: {r['method']} ({r['n_drones']}), source {r['source']}, episode {r['episode_id']}, scale {r['scale']:.2f}\n"
                 f"{ok}; final error {r['final_error_m']:.0f} m, final sigma {r['top_sigma_final_m']:.0f} m{det_txt}" + (f"\n[{note}]" if note else ""))
        sc2 = draw_episode_panel(ax, arrs, meta, str(meta.get("mode", ev.mode)), backend, obstacles, title)
        sc = sc2 if sc2 is not None else sc
        if note:
            remarks.append(f"{s['key']}: {note}")
        used.append({"panel": chr(97 + k), "key": s["key"], "method": r["method"], "n_drones": r["n_drones"], "episode_id": r["episode_id"], "source": r["source"],
                     "success": r["success"], "step_log": str(ev.step_log_path(r))})
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.55, pad=0.01, location="right")
        cb.set_label("step (drone path colour; rings = detection steps)")
    handles = [Line2D([0], [0], marker="s", ls="none", mfc="white", mec="k", mew=1.4, label="start (drone 1)"),
               Line2D([0], [0], marker="D", ls="none", mfc="white", mec="k", mew=1.4, label="start (drone 2)"),
               Line2D([0], [0], marker="o", ls="none", mfc="none", mec="k", mew=1.2, label="final position"),
               Line2D([0], [0], marker="o", ls="none", mfc="none", mec=R_DETECTION_COLOUR, mew=1.5, ms=8, label=f"detection (count > {currie_threshold_cps() * config.SENSOR_T:.1f})"),
               Line2D([0], [0], color="tab:red", lw=2, label="2-sigma GMM ellipses (end)"),
               Line2D([0], [0], marker="x", color="tab:red", ls="none", mew=2, label="MAP estimate"),
               Line2D([0], [0], marker="*", color="gold", mec="k", ls="none", ms=11, label="true source")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    _suptitle(fig, _title(ev, "Representative episodes (grey = 15 m slab density of the source in the truth frame, grey outlines = buildings)"), fontsize=11)
    p = _save(fig, out.path("trajectories", "png"))
    c = _write_csv(out.path("trajectories", "csv"), used)
    what = ("Four representative episodes (open success, trapped success, failure, multi-drone): buildings, truth-frame density, paths coloured by step, "
            "start markers, final GMM ellipses, MAP, true source")
    cols = ("step log: drone_xy, gmm_w, gmm_mu, gmm_cov, gmm_mask, map_xy, truth_xy, start_xy, frame; "
            "records: method, n_drones, episode_id, source, success, steps, final_error_m")
    out.ok("trajectories", p, what, [u["step_log"] for u in used] + ["slab_stack_z15.npy or slabs/slab_XXX.npz (15 m density)", "occupancy_2m_flowframe.npz"],
           cols, "10.7", reason="; ".join(remarks), data_csv=str(c))


# ------------------------------------------------------------------------------------------------ 9 sensitivity
def build_sensitivity(sets: Sequence[EvalSet], out: Outputs, methods: Sequence[str] | None) -> None:
    if len(sets) < 2:
        raise Skip("needs several evaluation tags (compare_tags / repeated --tag); only one given")
    all_cfgs = {c for ev in sets for c in ev.configs(methods)}
    if not all_cfgs:
        raise Skip("no records in the compared tags")
    ms = order_methods({m for m, _ in all_cfgs})
    cfgs = sorted(all_cfgs, key=lambda c: (ms.index(c[0]), c[1]))
    drones = sorted({n for _, n in cfgs})
    labels = []
    for ev in sets:
        scales = {round(float(r["scale"]), 6) for r in ev.records if np.isfinite(r["scale"])}
        sc = f"scale {next(iter(scales)):g}" if len(scales) == 1 else "scale log-uniform"
        labels.append(f"{ev.tag}\nMode {ev.mode}, {ev.horizon} steps\n{sc}")
    rows = []
    fig, axes = _drone_panels(drones, w=max(6.0, 2.2 * len(sets) + 2.5), h=5.0)
    for ax, n in zip(axes, drones):
        for m, nn in cfgs:
            if nn != n:
                continue
            xs, ys, lo, hi = [], [], [], []
            for i, ev in enumerate(sets):
                a = ev.aggregate(m, n).get(CURVE_GROUP) if ev.recs(m, n) else None
                if a is None:
                    continue
                xs.append(i)
                ys.append(a["success_rate"])
                lo.append(a["success_ci"][0])
                hi.append(a["success_ci"][1])
                rows.append({"tag": ev.tag, "mode": ev.mode, "horizon": ev.horizon, "method": m, "n_drones": n, "n": a["n"], "success_rate": a["success_rate"],
                             "ci_lo": a["success_ci"][0], "ci_hi": a["success_ci"][1], "strict_rate": a["success_strict_rate"], "label": labels[i].replace("\n", ", ")})
            if xs:
                off = (ms.index(m) - (len(ms) - 1) / 2) * 0.03
                ax.errorbar(np.array(xs) + off, ys, yerr=[np.maximum(np.array(ys) - lo, 0.0), np.maximum(np.array(hi) - np.array(ys), 0.0)], marker="o", ms=4, capsize=2, elinewidth=0.8,
                            **_style(m, n, ms))
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.4, len(sets) - 0.6)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{n} drone{'s' if n > 1 else ''}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate (12 observable sources)")
    _legend(fig, ms, drones)
    horizons, modes = {ev.horizon for ev in sets}, {ev.mode for ev in sets}
    warn = ""
    if len(horizons) > 1 or len(modes) > 1:
        warn = (f"Horizons differ ({', '.join(str(h) for h in sorted(horizons))} steps) and truth modes differ ({', '.join(sorted(modes))}): "
                "compare trends, not absolute levels.")
    _suptitle(fig, "Sensitivity: success rate (Wilson 95 % CI) per evaluation tag\n" + warn, fontsize=10)
    p = _save(fig, out.path("sensitivity", "png"))
    c = _write_csv(out.path("sensitivity", "csv"), rows)
    out.ok("sensitivity", p, "Success rate per evaluation tag (e.g. Mode F versus Mode T2 or scale settings) per method x drone count",
           [str(ev.path / "records_*.csv") for ev in sets], "method, n_drones, episode_id, source, success, scale, mode", "10.9",
           reason="compared tags: " + ", ".join(ev.tag for ev in sets), data_csv=str(c))


# ------------------------------------------------------------------------------------------------ within-tag sensitivity (scale bins)
def scale_bin_labels(edges: Sequence[float] = R_SCALE_BIN_EDGES) -> list[str]:
    lo, hi = edges[0], edges[-1]
    return [f"<= {lo:g}", f"{lo:g}-{hi:g}", f">= {hi:g}"]


def scale_bin(scale: float, edges: Sequence[float] = R_SCALE_BIN_EDGES) -> int:
    """0 for scale <= edges[0], 2 for scale >= edges[1], else 1 (the sensor sensitivity factor of an episode)."""
    return 0 if scale <= edges[0] else (2 if scale >= edges[-1] else 1)


def build_sensitivity_scale(ev: EvalSet, out: Outputs, methods: Sequence[str] | None = None) -> None:
    """Within-tag sensitivity (spec 10.9): success rate of the observable sources by sensor-scale bin, from the records' scale column, so that a
    single log-uniform-scale run has a sensitivity result without a second tag."""
    cfgs = ev.configs(methods)
    if not cfgs:
        raise Skip("no evaluation records")
    labels = scale_bin_labels()
    obs = set(metrics.GROUPS["all_observable"])
    rows = []
    for m, n in cfgs:
        recs = [r for r in ev.recs(m, n) if int(r["source"]) in obs and np.isfinite(float(r.get("scale", float("nan"))))]
        for b, lab in enumerate(labels):
            rs = [r for r in recs if scale_bin(float(r["scale"])) == b]
            if not rs:
                continue
            k, ks = sum(bool(r["success"]) for r in rs), sum(bool(r.get("success_strict", False)) for r in rs)
            rate, lo, hi = metrics.wilson_ci(k, len(rs))
            rows.append({"tag": ev.tag, "mode": ev.mode, "horizon": ev.horizon, "method": m, "n_drones": n, "scale_bin": lab, "bin_index": b,
                         "scale_median": float(np.median([float(r["scale"]) for r in rs])), "n": len(rs), "n_success": k, "success_rate": rate,
                         "ci_lo": lo, "ci_hi": hi, "strict_rate": ks / len(rs)})
    populated = sorted({r["bin_index"] for r in rows})
    if len(populated) < 2:
        raise Skip(f"the sensor scale does not vary within this tag ({len(populated)} of {len(labels)} bins populated): use the cross-tag sensitivity")
    ms, drones = order_methods({m for m, _ in cfgs}), sorted({n for _, n in cfgs})
    fig, axes = _drone_panels(drones, w=max(6.0, 1.6 * len(ms) + 2.5), h=4.6)
    for ax, n in zip(axes, drones):
        for m, nn in cfgs:
            if nn != n:
                continue
            sel = sorted((r for r in rows if r["method"] == m and r["n_drones"] == n), key=lambda r: r["bin_index"])
            if not sel:
                continue
            x = np.array([r["bin_index"] for r in sel], dtype=float) + (ms.index(m) - (len(ms) - 1) / 2) * 0.04
            y = np.array([r["success_rate"] for r in sel])
            ax.errorbar(x, y, yerr=[np.maximum(y - [r["ci_lo"] for r in sel], 0.0), np.maximum([r["ci_hi"] for r in sel] - y, 0.0)], marker="o", ms=4, capsize=2,
                        elinewidth=0.8, **_style(m, n, ms))
        ns = [sorted({r["n"] for r in rows if r["n_drones"] == n and r["bin_index"] == b}) for b in range(len(labels))]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"scale {lab}\nn = {'/'.join(str(v) for v in nb) if nb else '-'} per method" for lab, nb in zip(labels, ns)], fontsize=8)
        ax.set_xlim(-0.4, len(labels) - 0.6)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{n} drone{'s' if n > 1 else ''}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate (12 observable sources)")
    _legend(fig, ms, drones)
    _suptitle(fig, _title(ev, "Within-tag sensitivity: success rate by sensor-scale bin (scale = per-episode sensitivity factor; Wilson 95 % CI)"), fontsize=11)
    p = _save(fig, out.path("sensitivity_scale", "png"))
    c = _write_csv(out.path("sensitivity_scale", "csv"), rows)
    out.ok("sensitivity_scale", p, "Success rate of the 12 observable sources by sensor-scale bin (<= 0.6, 0.6-1.7, >= 1.7) within one evaluation tag, per method x drone count",
           _records_files(ev, cfgs), "method, n_drones, episode_id, source, success, success_strict, scale", "10.9", data_csv=str(c),
           scale_bins=labels)


# ------------------------------------------------------------------------------------------------ 9.8 error at fixed steps
def _ffill_tail(x: np.ndarray) -> np.ndarray:
    """Forward-fill the NaN padding at the end of every row: an episode that stopped early keeps its last logged value."""
    out = np.array(x, dtype=float, copy=True)
    for i in range(out.shape[0]):
        fin = np.flatnonzero(np.isfinite(out[i]))
        if fin.size and fin[-1] < out.shape[1] - 1:
            out[i, fin[-1] + 1:] = out[i, fin[-1]]
    return out


def error_at_steps_rows(ev: EvalSet, series: dict[tuple[str, int], StepSeries], steps: Sequence[int] = R_ERROR_AT_STEPS,
                        groups: Sequence[str] = R_ERROR_AT_STEP_GROUPS) -> list[dict]:
    """Median (with quartiles and p90) MAP error over the episodes of each source group at the requested steps clipped to the horizon (spec 9.8);
    step k = k-th environment step (1-based), episodes that stopped early are carried forward."""
    rows = []
    for (m, n), s in series.items():
        err = _ffill_tail(s.err)
        for g in groups:
            sel = s.select(metrics.GROUPS[g])
            if not sel.any():
                continue
            for req in steps:
                eff = min(int(req), int(ev.horizon))
                x = err[sel, min(eff, err.shape[1]) - 1]
                x = x[np.isfinite(x)]
                if x.size == 0:
                    continue
                rows.append({"tag": ev.tag, "method": m, "n_drones": n, "group": g, "requested_step": int(req), "step": eff, "clipped": eff != int(req),
                             "n_episodes": int(x.size), "median_error_m": float(np.median(x)), "q25_error_m": float(np.quantile(x, 0.25)),
                             "q75_error_m": float(np.quantile(x, 0.75)), "p90_error_m": float(np.quantile(x, 0.9))})
    return rows


def _effective_steps(horizon: int, steps: Sequence[int] = R_ERROR_AT_STEPS) -> list[tuple[int, list[int]]]:
    """[(effective step, requested steps that map to it)] in increasing order: with the Mode T2 horizon 150 the requests 150 and 300 coincide."""
    out: dict[int, list[int]] = {}
    for r in steps:
        out.setdefault(min(int(r), int(horizon)), []).append(int(r))
    return sorted(out.items())


def build_error_at_steps(ev: EvalSet, out: Outputs, methods: Sequence[str] | None, cap: int | None) -> None:
    series = _series_for(ev, ev.configs(methods), cap)
    rows = error_at_steps_rows(ev, series)
    if not rows:
        raise Skip("no finite MAP error in the step logs")
    phrase, used = episodes_phrase(ev, series, cap)
    cfgs = [c for c in ev.configs(methods) if c in series]
    groups = [g for g in R_ERROR_AT_STEP_GROUPS if any(r["group"] == g for r in rows)]
    eff = _effective_steps(ev.horizon)
    ncol = len(groups) * len(eff)
    mat = np.full((len(cfgs), ncol), np.nan)
    for r in rows:
        i = cfgs.index((r["method"], r["n_drones"]))
        j = groups.index(r["group"]) * len(eff) + [e for e, _ in eff].index(r["step"])
        mat[i, j] = r["median_error_m"]
    fin = mat[np.isfinite(mat) & (mat > 0)]
    norm = LogNorm(vmin=max(1.0, float(fin.min()) * 0.9), vmax=max(float(fin.max()), 2.0)) if fin.size else None
    fig, ax = plt.subplots(figsize=(0.95 * ncol + 3.8, 0.42 * len(cfgs) + 3.0), layout="constrained")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=plt.get_cmap("Blues").with_extremes(bad="#d9d9d9"), norm=norm, aspect="auto", interpolation="nearest")
    for i in range(len(cfgs)):
        for j in range(ncol):
            if np.isfinite(mat[i, j]):
                dark = norm is not None and norm(mat[i, j]) > 0.6
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if dark else "0.1")
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([(str(e) if len(q) == 1 else f"{e}\n(= requested {'/'.join(str(v) for v in q)})") for _ in groups for e, q in eff], fontsize=8)
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([cfg_label(m, n) + (" *" if m in VERIFICATION_METHODS else "") for m, n in cfgs], fontsize=8)
    for k, g in enumerate(groups):
        if k:
            ax.axvline(k * len(eff) - 0.5, color="k", lw=1.2)
        n_ep = sorted({r["n_episodes"] for r in rows if r["group"] == g})
        ax.text(k * len(eff) + (len(eff) - 1) / 2, 1.012, f"{GROUP_SHORT[g].replace(chr(10), ' ')} (n = {'/'.join(str(v) for v in n_ep)})", ha="center", va="bottom",
                fontsize=8, color="0.2", transform=ax.get_xaxis_transform())
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cfgs), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("environment step (budget)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("median MAP error [m] (log colour scale; darker = larger error)")
    clipped = [f"{q} -> {min(q, ev.horizon)}" for q in R_ERROR_AT_STEPS if q > ev.horizon]
    note = f"requested steps beyond the horizon are clipped ({', '.join(clipped)})" if clipped else "no step is clipped by the horizon"
    _suptitle(fig, _title(ev, f"Median MAP error at fixed step budgets (spec 9.8; {note}; * = verification only)", phrase), fontsize=10)
    p = _save(fig, out.path("error_at_steps", "png"))
    c = _write_csv(out.path("error_at_steps", "csv"), rows)
    out.ok("error_at_steps", p, f"Median (quartiles, p90) MAP error at the steps {', '.join(str(s) for s in R_ERROR_AT_STEPS)} clipped to the horizon {ev.horizon}, per method x drone count "
           "and source group (all observable, train, holdout)", _steps_files(ev), "map_error (T) per episode; episode selection from the records: method, n_drones, episode_id, source",
           "9.8", data_csv=str(c), reason=note, episodes_title=phrase, episodes_used=used, steps_effective=[e for e, _ in eff], **_miss(series))


# ------------------------------------------------------------------------------------------------ 9.5 / 9.9 / 9.11 efficiency
R_HATCH_BY_DRONES = {1: None, 2: "//", 3: "xx"}


def _percentile_stats(v: Sequence[float]) -> dict[str, float]:
    x = np.asarray([a for a in v if a is not None and np.isfinite(a)], dtype=float)
    nan = float("nan")
    if x.size == 0:
        return {"n_valid": 0, "mean": nan, "median": nan, "q25": nan, "q75": nan, "p05": nan, "p95": nan}
    lo, hi = R_BOX_WHISKER_PCT
    return {"n_valid": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)),
            "p05": float(np.percentile(x, lo)), "p95": float(np.percentile(x, hi))}


def _cfg_ticks(ax: plt.Axes, cfgs: Sequence[tuple[str, int]], fontsize: float = 7.0) -> None:
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([cfg_label(m, n) + (" *" if m in VERIFICATION_METHODS else "") for m, n in cfgs], rotation=55, ha="right", fontsize=fontsize)
    ax.set_xlim(-0.7, len(cfgs) - 0.3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)


def _box_axes(ax: plt.Axes, cfgs: Sequence[tuple[str, int]], data: dict[tuple[str, int], list[float]], ylabel: str, ordered: Sequence[str],
              title: str, frac: dict[tuple[str, int], str] | None = None) -> None:
    """One box per configuration (median, quartiles, 5-95 % whiskers; colour = method, hatch = drone count) over faint episode points."""
    rng = np.random.default_rng(0)
    for i, (m, n) in enumerate(cfgs):
        v = np.asarray([a for a in data.get((m, n), []) if a is not None and np.isfinite(a)], dtype=float)
        if v.size == 0:
            continue
        bp = ax.boxplot([v], positions=[i], widths=0.62, whis=R_BOX_WHISKER_PCT, showfliers=False, patch_artist=True, medianprops={"color": "k", "linewidth": 1.4},
                        whiskerprops={"color": "0.25"}, capprops={"color": "0.25"})
        for b in bp["boxes"]:
            b.set(facecolor=method_colour(m, ordered), alpha=0.7, edgecolor="0.2", hatch=R_HATCH_BY_DRONES.get(n))
        ax.plot(i + rng.uniform(-0.22, 0.22, v.size), v, ".", ms=2.5, color="0.15", alpha=0.35, zorder=4)
    if frac:
        for i, c in enumerate(cfgs):
            if c in frac:
                ax.text(i, 1.005, frac[c], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5, color="0.3")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(textwrap.fill(title, 56), fontsize=9, pad=14 if frac else 6)
    _cfg_ticks(ax, cfgs)


def build_efficiency(ev: EvalSet, out: Outputs, methods: Sequence[str] | None, n_boot: int) -> None:
    """Efficiency of the search (spec 9.5, 9.9, 9.11): path length, masked actions, first-detection step, and the drone-count effect (n/1 success-step
    ratio and total flight time = drones x steps), over the 12 observable sources."""
    cfgs = ev.configs(methods)
    if not cfgs:
        raise Skip("no evaluation records")
    obs = set(metrics.GROUPS[CURVE_GROUP])
    ms = order_methods({m for m, _ in cfgs})
    per: dict[str, dict[tuple[str, int], list[float]]] = {k: {} for k in ("path_length_m", "masked_actions", "first_detection_step", "success_step", "flight_time_drone_steps")}
    n_eps: dict[tuple[str, int], int] = {}
    for m, n in cfgs:
        rs = [r for r in ev.recs(m, n) if int(r["source"]) in obs]
        n_eps[(m, n)] = len(rs)
        per["path_length_m"][(m, n)] = [float(r["path_length_m"]) for r in rs]
        per["masked_actions"][(m, n)] = [float(r["n_masked"]) for r in rs]
        per["first_detection_step"][(m, n)] = [float(r["first_detection_step"]) for r in rs if r.get("first_detection_step") is not None]
        per["success_step"][(m, n)] = [float(r["steps"]) for r in rs if r["success"]]
        per["flight_time_drone_steps"][(m, n)] = [float(n * r["steps"]) for r in rs if r["success"]]
    rows = []
    for metric, data in per.items():
        for (m, n), v in data.items():
            st = _percentile_stats(v)
            ci = metrics.bootstrap_median_ci(v, n_boot)[1:] if metric in ("success_step", "flight_time_drone_steps") and len(v) else (float("nan"), float("nan"))
            rows.append({"tag": ev.tag, "metric": metric, "method": m, "n_drones": n, "group": CURVE_GROUP, "n_episodes": n_eps[(m, n)], **st,
                         "median_ci_lo": ci[0], "median_ci_hi": ci[1]})
    # drone-count effect: every method flown with several team sizes, (n drones) versus (fewest drones), groups and overall
    pair_methods = [m for m in order_methods({m for m, _ in cfgs}) if m not in VERIFICATION_METHODS and len(ev.drone_counts([m])) > 1]
    scopes = {k: v for k, v in paired_scopes().items() if v[0] != "source"}
    dc_rows: list[dict] = []
    for m in pair_methods:
        counts = ev.drone_counts([m])
        for n in counts[1:]:
            dc_rows += paired_table(ev, m, m, n, scopes, n_boot, n_ref=counts[0])
    pairs = [(m, n, int(next(r["n_drones_reference"] for r in dc_rows if r["method"] == m and r["n_drones"] == n)))
             for m in pair_methods for n in sorted({r["n_drones"] for r in dc_rows if r["method"] == m})]
    fig = plt.figure(figsize=(max(14.0, 0.45 * len(cfgs) * 3 + 3.0), 11.0), layout="constrained")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.95])
    _box_axes(fig.add_subplot(gs[0, 0:2]), cfgs, per["path_length_m"], "path length per episode [m] (all drones summed)", ms, "(a) path length")
    _box_axes(fig.add_subplot(gs[0, 2:4]), cfgs, per["masked_actions"], "masked actions per episode", ms, "(b) masked actions (collisions are 0 by construction)")
    frac = {c: f"{len(per['first_detection_step'][c])}/{n_eps[c]}" for c in cfgs}
    _box_axes(fig.add_subplot(gs[0, 4:6]), cfgs, per["first_detection_step"], "first step with a count above the Currie threshold", ms,
              "(c) first detection (episodes with a detection; label = detected / episodes)", frac)
    sub = gs[1, 0:3].subgridspec(1, 2, wspace=0.12)
    if pairs:
        _ratio_axes(fig.add_subplot(sub[0, 0]), pairs, dc_rows, "step_ratio_paired_median", "step_ratio_paired_lo", "step_ratio_paired_hi", "step_ratio_of_medians",
                    "success-step ratio  hi / lo drones", lambda p: p[2] / p[1], "(d) success-step ratio (dashed = ideal n_lo / n_hi)")
        _ratio_axes(fig.add_subplot(sub[0, 1]), pairs, dc_rows, "flight_time_paired_ratio_median", "flight_time_paired_ratio_lo", "flight_time_paired_ratio_hi",
                    "flight_time_ratio_of_medians", "total flight time ratio  hi / lo", lambda p: 1.0, "(e) total flight time ratio (dashed = equal effort)")
    else:
        ax = fig.add_subplot(gs[1, 0:3])
        ax.axis("off")
        ax.text(0.5, 0.5, "(d, e) drone-count effect: needs one method evaluated with several drone counts", ha="center", va="center", fontsize=10, color="0.35", transform=ax.transAxes)
    axf = fig.add_subplot(gs[1, 3:6])
    ft = per["flight_time_drone_steps"]
    for i, (m, n) in enumerate(cfgs):
        v = ft[(m, n)]
        if not v:
            axf.text(i, 0.02, "no\nsuccess", transform=axf.get_xaxis_transform(), ha="center", va="bottom", fontsize=6, color="0.4")
            continue
        med, lo, hi = metrics.bootstrap_median_ci(v, n_boot)
        axf.bar(i, med, 0.7, color=method_colour(m, ms), alpha=0.8, hatch=R_HATCH_BY_DRONES.get(n), edgecolor="0.2", linewidth=0.5,
                yerr=[[max(med - lo, 0.0)], [max(hi - med, 0.0)]], error_kw={"elinewidth": 1.0, "capsize": 2, "ecolor": "0.15"})
        axf.text(i, med, f"{med:.0f}", ha="center", va="bottom", fontsize=6.5, color="0.2")
    axf.set_ylabel("median total flight time to success [drone-steps]", fontsize=8)
    axf.set_title("(f) total flight time = drones x success step (successful episodes; 95 % bootstrap CI)", fontsize=9)
    _cfg_ticks(axf, cfgs)
    handles = [Patch(facecolor=method_colour(m, ms), edgecolor="0.2", alpha=0.8, label=method_label(m)) for m in ms]
    handles += [Patch(facecolor="white", edgecolor="0.2", hatch=R_HATCH_BY_DRONES.get(n), label=f"{n} drone{'s' if n > 1 else ''}") for n in sorted({n for _, n in cfgs}) if n > 1]
    if pairs:
        handles += [Line2D([0], [0], marker="o", color="0.3", lw=1.2, ms=6, label="(d, e) median of per-episode ratios, 95 % CI"),
                    Line2D([0], [0], marker="D", mfc="none", mec="0.15", ls="none", ms=6, label="(d, e) ratio of medians")]
    fig.legend(handles=handles, loc="outside lower center", ncol=max(2, min(len(handles), int(fig.get_figwidth() // 1.9))), fontsize=8, frameon=False)
    _suptitle(fig, _title(ev, f"Search efficiency ({len(obs)} observable sources, source 110 excluded; box = quartiles, whiskers = "
                              f"{R_BOX_WHISKER_PCT[0]}-{R_BOX_WHISKER_PCT[1]} %, * = verification only)"), fontsize=11)
    p = _save(fig, out.path("efficiency", "png"))
    c = _write_csv(out.path("efficiency", "csv"), rows)
    c2 = _write_csv(out.path("efficiency_drone_count", "csv"), dc_rows, PAIRED_FIELDS)
    data = _records_files(ev, cfgs)
    out.ok("efficiency", p, "Distributions of path length, masked actions and first-detection step, and the drone-count effect (n/1 success-step ratio, ideal n_lo/n_hi; "
           "total flight time = drones x steps) per method x drone count over the 12 observable sources", data,
           "method, n_drones, episode_id, source, success, steps, path_length_m, n_masked, first_detection_step", "9.5 / 9.9 / 9.11", data_csv=str(c),
           data_csv_drone_count=str(c2), drone_count_pairs=[f"{m}: {nh} vs {nl}" for m, nh, nl in pairs])


# ------------------------------------------------------------------------------------------------ 9.12 compute cost
def build_compute_cost(ev: EvalSet, out: Outputs, methods: Sequence[str] | None = None) -> None:
    """Compute cost per step by method (spec 9.12): the records' step_ms_median (environment step only, per-episode median) and the episode wall time
    divided by the horizon (environment + policy / planner + logging), medians over the episodes with quartile whiskers."""
    cfgs = ev.configs(methods)
    if not cfgs:
        raise Skip("no evaluation records")
    stats: dict[str, dict[tuple[str, int], np.ndarray]] = {"step_ms_median": {}, "wall_ms_per_step": {}}
    for m, n in cfgs:
        rs = ev.recs(m, n)
        a = np.array([float(r["step_ms_median"]) for r in rs], dtype=float)
        w = np.array([1000.0 * float(r["wall_s"]) / ev.horizon for r in rs], dtype=float)
        stats["step_ms_median"][(m, n)] = a[np.isfinite(a) & (a > 0)]
        stats["wall_ms_per_step"][(m, n)] = w[np.isfinite(w) & (w > 0)]
    if not any(v.size for v in stats["step_ms_median"].values()) and not any(v.size for v in stats["wall_ms_per_step"].values()):
        raise Skip("the records carry no timing (step_ms_median / wall_s are empty, e.g. records rebuilt from step logs)")
    rows = []
    for metric, data in stats.items():
        for (m, n), v in data.items():
            if v.size:
                rows.append({"tag": ev.tag, "metric": metric, "method": m, "n_drones": n, "n_episodes": int(v.size), "median_ms": float(np.median(v)),
                             "q25_ms": float(np.quantile(v, 0.25)), "q75_ms": float(np.quantile(v, 0.75)), "p90_ms": float(np.quantile(v, 0.9)),
                             "min_ms": float(v.min()), "max_ms": float(v.max())})
    ms, drones = order_methods({m for m, _ in cfgs}), sorted({n for _, n in cfgs})
    fig, axes = plt.subplots(1, 2, figsize=(max(11.0, 1.15 * len(ms) * 2 + 2.0), 4.9), layout="constrained")
    log_note = [False]
    titles = {"step_ms_median": "(a) environment step (per-episode median; excludes the policy)",
              "wall_ms_per_step": f"(b) episode wall time / {ev.horizon} steps (environment + policy / planner + logging)"}
    for ax, (metric, data) in zip(axes, stats.items()):
        w = 0.8 / max(len(drones), 1)
        for j, n in enumerate(drones):
            xs, med, lo, hi = [], [], [], []
            for i, m in enumerate(ms):
                v = data.get((m, n))
                if v is None or not v.size:
                    continue
                xs.append(i - 0.4 + w * (j + 0.5))
                med.append(float(np.median(v)))
                lo.append(float(np.quantile(v, 0.25)))
                hi.append(float(np.quantile(v, 0.75)))
            if xs:
                ax.bar(xs, med, w * 0.92, color=DRONE_COLOURS.get(n, "0.3"), alpha=0.9, label=f"{n} drone{'s' if n > 1 else ''}",
                       yerr=[np.maximum(np.array(med) - lo, 0.0), np.maximum(np.array(hi) - med, 0.0)], error_kw={"elinewidth": 1.0, "capsize": 2, "ecolor": "0.15"})
                for x, v, h in zip(xs, med, hi):
                    ax.text(x, h, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color="0.2")
        allv = np.concatenate([v for v in data.values() if v.size]) if any(v.size for v in data.values()) else np.array([1.0])
        if np.median(allv) > 0 and (np.quantile(allv, 0.9) / max(np.quantile(allv, 0.1), 1e-9)) > 30:
            ax.set_yscale("log")                                         # bars from zero are honest only when the methods differ by less than ~30x
            log_note[0] = True
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([wrap_method(m) for m in ms], fontsize=8)
        ax.set_ylabel("ms per step (median over episodes; bars = quartiles)", fontsize=8)
        ax.set_title(titles[metric], fontsize=9)
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.set_axisbelow(True)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8, frameon=False)
    elif axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=8, frameon=False)
    _suptitle(fig, _title(ev, "Compute cost per step by method (absolute values depend on the CPU load of concurrent jobs" + ("; log axis)" if log_note[0] else ")")), fontsize=11)
    p = _save(fig, out.path("compute_cost", "png"))
    c = _write_csv(out.path("compute_cost", "csv"), rows)
    out.ok("compute_cost", p, "Compute cost per step by method x drone count: step_ms_median of the records (environment step) and episode wall time / horizon", _records_files(ev, cfgs),
           "method, n_drones, step_ms_median, wall_s", "9.12", data_csv=str(c))


# ------------------------------------------------------------------------------------------------ driver
def _run(out: Outputs, names: Sequence[str], fn: Callable[[], None]) -> None:
    try:
        fn()
    except Skip as e:
        for n in names:
            out.skip(n, str(e))


def make_for_set(ev: EvalSet, out_dir: Path, prefix: str, suffix: str = "", reference_method: str | None = None, methods: Sequence[str] | None = None,
                 traj_method: str | None = None, backend: Any = None, obstacles: Any = None, log_eps_per_source: int | None = None,
                 n_boot: int = config.EVAL_N_BOOTSTRAP) -> Outputs:
    """All per-directory outputs of one EvalSet (manifest keys get ``suffix``)."""
    out = Outputs(out_dir, prefix, ev.tag, suffix, notes=ev.notes)
    tasks: list[tuple[Sequence[str], Callable[[], None]]] = [
        (("table2_md", "table2_csv", "table2_png"), lambda: build_table2(ev, out, methods)),
        (("success_bars",), lambda: build_success_bars(ev, out, methods)),
        (("success_cdf",), lambda: build_success_cdf(ev, out, methods)),
        (("map_error_vs_step",), lambda: build_map_error(ev, out, methods, log_eps_per_source)),
        (("source_heatmap",), lambda: build_heatmap(ev, out, methods)),
        (("paired_diff",), lambda: build_paired(ev, out, reference_method, methods, n_boot)),
        (("trajectories",), lambda: build_trajectories(ev, out, traj_method, methods, backend, obstacles)),
        (("entropy_curves",), lambda: build_entropy(ev, out, methods, log_eps_per_source)),
        (("calibration",), lambda: build_calibration(ev, out, methods, log_eps_per_source)),
        (("threshold_curve",), lambda: build_threshold_curve(ev, out, methods, log_eps_per_source)),
        (("start_type_bars", "start_type_table"), lambda: build_start_type(ev, out, methods)),
        (("sensitivity_scale",), lambda: build_sensitivity_scale(ev, out, methods)),
        (("error_at_steps",), lambda: build_error_at_steps(ev, out, methods, log_eps_per_source)),
        (("efficiency",), lambda: build_efficiency(ev, out, methods, n_boot)),
        (("compute_cost",), lambda: build_compute_cost(ev, out, methods)),
    ]
    for names, fn in tasks:
        _run(out, names, fn)
    return out


def make_all(eval_dirs: str | Path | Sequence[str | Path], out_dir: str | Path, prefix: str, reference_method: str | None = None,
             compare_tags: Sequence[str | Path] | None = None, *, methods: Sequence[str] | None = None, traj_method: str | None = None,
             backend: Any = None, obstacles: Any = None, log_eps_per_source: int | None = None, n_boot: int = config.EVAL_N_BOOTSTRAP,
             dpi: int | None = None) -> dict[str, dict[str, Any]]:
    """Every evaluation figure and table of spec 10 (and the spec 9 outputs) for one or several evaluation directories (tags); returns the manifest
    {output_name: {"path", "status": "ok" | "skipped", "reason"}} (also written as <prefix>_manifest.json together with <prefix>_figure_sources.csv).

    With several directories each one gets its own set of outputs (``<prefix>_<tag>_<name>``, manifest key ``<name>@<tag>``): Mode F and Mode T2
    numbers are never drawn on one axis, except in the sensitivity figure over ``compare_tags`` (default: all given directories when there are
    several).  ``reference_method`` is the baseline of the paired differences (default gmm_infotaxis); ``methods`` restricts the compared methods;
    ``log_eps_per_source`` keeps only that many episodes per source for the step-log figures (smoke runs); ``backend`` / ``obstacles`` inject the
    slab backend and ObstacleMap of the trajectory figure (tests); ``dpi`` = figure resolution (default FIG_DPI = 150; config.FIG_DPI_FINAL = 300 for
    the final figures)."""
    with _dpi_override(dpi):
        return _make_all(eval_dirs, out_dir, prefix, reference_method, compare_tags, methods, traj_method, backend, obstacles, log_eps_per_source, n_boot)


def _make_all(eval_dirs: str | Path | Sequence[str | Path], out_dir: str | Path, prefix: str, reference_method: str | None, compare_tags: Sequence[str | Path] | None,
              methods: Sequence[str] | None, traj_method: str | None, backend: Any, obstacles: Any, log_eps_per_source: int | None, n_boot: int) -> dict[str, dict[str, Any]]:
    specs = [eval_dirs] if isinstance(eval_dirs, (str, Path)) else list(eval_dirs)
    if not specs:
        raise ValueError("eval_dirs is empty")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = [load_eval_set(s) for s in specs]
    manifest: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, str]] = []
    for ev in sets:
        multi = len(sets) > 1
        o = make_for_set(ev, out_dir, f"{prefix}_{ev.tag}" if multi else prefix, f"@{ev.tag}" if multi else "", reference_method, methods, traj_method,
                         backend, obstacles, log_eps_per_source, n_boot)
        manifest.update(o.manifest)
        rows += o.rows
    cmp_specs = list(compare_tags) if compare_tags else (specs if len(specs) > 1 else [])
    cmp_sets = []
    for s in cmp_specs:
        d = resolve_eval_dir(s)
        cmp_sets.append(next((e for e in sets if e.path == d), None) or load_eval_set(d))
    so = Outputs(out_dir, prefix, "+".join(e.tag for e in cmp_sets), "")
    _run(so, ("sensitivity",), lambda: build_sensitivity(cmp_sets, so, methods))
    manifest.update(so.manifest)
    rows += so.rows
    fs = out_dir / f"{prefix}_figure_sources.csv"
    _write_csv(fs, rows, ["output", "file", "spec_section", "slide", "slide_basis", "what_it_shows", "data_files", "columns_used", "output_csv", "output_csv_columns", "notes"])
    manifest["figure_sources"] = {"path": str(fs), "status": "ok", "reason": f"{len(rows)} produced outputs listed"}
    (out_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", action="append", default=[], metavar="TAG_OR_DIR", help="evaluation tag (under <CACHE_DIR>/eval) or directory; repeat for several (one output set each)")
    ap.add_argument("--tag2", nargs="*", default=[], metavar="TAG_OR_DIR", help="further tags (same as repeating --tag)")
    ap.add_argument("--compare-tags", nargs="*", default=None, metavar="TAG_OR_DIR", help="tags of the sensitivity figure (default: all tags when several are given)")
    ap.add_argument("--out-dir", type=Path, default=None, help="output directory (default <FIG_DIR>/eval)")
    ap.add_argument("--prefix", default=None, help="file name prefix (default fig_eval_<first tag>)")
    ap.add_argument("--reference-method", default=None, help="reference of the paired differences (default gmm_infotaxis)")
    ap.add_argument("--methods", nargs="*", default=None, help="restrict the compared methods")
    ap.add_argument("--traj-method", default=None, help="preferred method of the representative trajectories")
    ap.add_argument("--episodes-per-source", type=int, default=None, help="step-log figures: at most this many episodes per source and configuration (smoke runs)")
    ap.add_argument("--n-boot", type=int, default=config.EVAL_N_BOOTSTRAP, help="bootstrap resamples of the success-difference CI")
    ap.add_argument("--dpi", type=int, default=None, help=f"figure resolution in dpi (default {R_DEFAULT_DPI}; {config.FIG_DPI_FINAL} for the final figures)")
    args = ap.parse_args(argv)
    tags = list(args.tag) + list(args.tag2)
    if not tags:
        ap.error("at least one --tag is required")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    out_dir = args.out_dir if args.out_dir is not None else config.FIG_DIR / "eval"
    prefix = args.prefix or f"fig_eval_{Path(str(tags[0])).name}"
    manifest = make_all(tags, out_dir, prefix, args.reference_method, args.compare_tags, methods=args.methods, traj_method=args.traj_method,
                        log_eps_per_source=args.episodes_per_source, n_boot=args.n_boot, dpi=args.dpi)
    for k, v in manifest.items():
        print(f"{k:28s} {v['status']:8s} {v['path'] or '-'}" + (f"  [{v['reason']}]" if v.get("reason") else ""))
    print(f"[fig_eval] manifest {out_dir / (prefix + '_manifest.json')}")
    return manifest


if __name__ == "__main__":
    main()
