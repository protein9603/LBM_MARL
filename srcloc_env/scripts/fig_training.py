"""Training-result figures from the run directories of rl/train.py (docs/training_evaluation_spec.md section 7, items 1-8; D10, D11).

Usage: python -m srcloc_env.scripts.fig_training --runs m1_s1 m1_s2 m1_s3 --prefix fig6_m1 [--baseline-tag t2_4_v3] [--n-drones N] [--dpi 300]
           [--compare m1_s1,m1_s2 m2_s1,m2_s2] [--traj-source 108] [--out-dir DIR] [--only curves reward_decomposition ...]

ONE call (``make_all``, which the CLI uses) writes every training-result visual of spec section 7 into --out-dir:
  item 1  <prefix>_curves.png / .csv            success, strict success, return, length vs team steps (one thin line per seed, mean and sd
                                                  band over seeds; horizontal lines = baselines on the 9 training sources of an evaluation tag, the
                                                  legend gives their Wilson 95 % CI) + <prefix>_baselines.csv (method, k, n, rate, CI)
  item 2  <prefix>_returns.png / .csv           episode return, episode length and median success step vs team steps (same style, from episodes.csv)
  item 3  <prefix>_diagnostics.png              PPO losses, entropy, KL, clip fraction, explained variance, masked share vs iteration
  item 4  <prefix>_source_heatmap.png / .csv    per-source success rate, source x checkpoint (quick evaluation; source 110 absent; panels wrap so that
                                                  the figure stays at most HEATMAP_MAX_WIDTH_IN wide)
  item 5  <prefix>_ckpt_eval.png / .csv         checkpoint quick evaluation, training versus held-out sources
  item 6  <prefix>_reward_decomposition.png/.csv information-gain, time and terminal reward terms (moving averages from episodes.csv)
  item 7  <prefix>_<run>_trajectories.png       the same quick-evaluation episode at the first periodic ("initial"), a middle and the last ("late")
                                                  checkpoint (4 snapshots each: building outlines, 15 m slab density, drone paths, start, 2-sigma GMM
                                                  ellipses, MAP, true source)
  item 8  <prefix>_compare.png / .csv           success and return of run groups with different drone counts (--compare, or auto-grouped by drone count)
  and <prefix>_manifest.json: {output name: {"path", "status": "ok" | "skipped", "reason"}} for all of them (new key: baselines_csv).
Runs with different drone counts are never pooled into one "mean of seeds" band: items 1, 2 and 6 draw one colour group per drone count (CSV columns
<key>_mean_<n>d, <key>_sd_<n>d); the baseline lines of item 1 use the drone count of the runs (--n-drones overrides it and warns on a mismatch).
Curve CSVs carry <key>_mean, <key>_sd (sample standard deviation over the seeds, ddof = 1: also the width of the band in the figures) and one column
<key>_seed_<run> per seed.  Files that a running training / evaluation is still writing are tolerated: an incomplete last CSV row is dropped and an
unparsable number becomes NaN.
An output whose input is missing (no step logs, no reward-decomposition columns, baseline tag of another truth mode, ...) is skipped with the reason in the
manifest; a real bug still raises (so does an unknown --only name).  Mode T2 runs have 150-step episodes: the baseline lines come from an evaluation tag
of the SAME truth mode (summary.json mode versus the run's config.json truth_mode) and use the records' own horizon.  Figures are drawn with matplotlib
(Agg), dpi 150 by default (--dpi 300 for final figures), English labels.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import math
import re
import sys
import textwrap
import warnings
import zipfile
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap    # noqa: E402
from matplotlib.lines import Line2D    # noqa: E402
from matplotlib.patches import Ellipse   # noqa: E402

from srcloc_env import config      # noqa: E402
from srcloc_env.eval.metrics import GROUPS, wilson_ci   # noqa: E402  (light: config + numpy only)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
GROUP_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]          # one colour per drone-count group when runs of several drone counts are drawn together
GROUP_STYLES = ["-", "--", ":", "-."]
BASE_COLORS = {"random": "0.55", "lawnmower": "#8c564b", "greedy_map": "#17becf", "gmm_infotaxis": "#e377c2"}
COMPONENT_COLORS = {"ret_info": "#1f77b4", "ret_time": "#7f7f7f", "ret_terminal": "#d62728", "ret": "k"}
DRONE_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:cyan"]        # drone paths of the trajectory figure
OUTPUT_NAMES = ("curves", "episode_stats", "diagnostics", "source_heatmap", "ckpt_eval", "reward_decomposition", "trajectories", "compare_drones")
# Layout constants of this script (not physical constants; defined here because the figure code is their only user).
FIG_DPI = 150                              # figure resolution of the pipeline outputs (--dpi 300 for the final figures)
MA_WINDOW_DEFAULT = 100                    # episodes of a moving average when config.json of the run has no args.ma_window (train.py default)
GRID_POINTS = 200                          # points of the common x grid on which the seeds are averaged
STEPS_UNIT_SWITCH = 1.0e5                  # runs shorter than this many team steps get an x axis in thousands of steps instead of millions
TRAJ_FRACTIONS = (0.1, 0.3, 0.6, 1.0)      # snapshot times of a sample trajectory (fractions of the episode), as in fig_episode
TRAJ_MARGIN_M = 80.0                       # window margin around the paths and the source [m]
TRAJ_MIN_ASPECT = 0.6                      # smallest height : width ratio of a trajectory window (thin paths would give flat panels)
TRAJ_MAP_REACH_M = 300.0                   # a MAP estimate at most this far from the paths / source box widens the window; a farther one gets an arrow
TRAJ_PANEL_IN = 3.6                        # width of one trajectory panel [in]
TRAJ_DENSITY_DECADES = 4.0                 # decades of the slab density colour scale
TRAJ_PATH_LW = 3.2                         # line width of a drone path (a ribbon in the drone's own colour under the smaller count dots)
TRAJ_DOT_SIZE = 3.2                        # marker area of a measurement dot [pt^2]
TRAJ_COUNT_TICKS = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)       # candidate tick values [counts per step] of the dot colour bar
HEATMAP_ANNOTATE_MAX_COLS = 14             # write "k/n" into the heat-map cells up to this many checkpoints
HEATMAP_MAX_WIDTH_IN = 14.0                # widest heat-map figure [in]; more panels / checkpoints wrap into further rows
HEATMAP_PANEL_BASE_IN = 1.8                # heat-map panel width = base (labels) + HEATMAP_CELL_IN per checkpoint column [in]
HEATMAP_CELL_IN = 0.38
HEATMAP_PANEL_MIN_IN = 3.4
HEATMAP_MIN_WIDTH_IN = 9.0                 # narrowest heat-map figure [in] (the suptitle lines need about this width)
TRAJ_REQUIRED_KEYS = ("drone_xy", "y", "gmm_w", "gmm_mu", "gmm_cov", "gmm_mask", "map_xy", "map_error", "top_sigma", "truth_xy")
_UNREADABLE = (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile)      # what np.load raises for an NPZ that is still being written


class Skip(Exception):
    """An output cannot be produced because an input it needs is missing; the reason goes into the manifest (never to the caller)."""


def _dpi(dpi: int | None) -> int:
    return FIG_DPI if dpi is None else int(dpi)


def _wrap(text: str, fig_width_in: float, fontsize: float) -> str:
    """Wrap every line of a figure title to the figure width (about 0.6 x font size per character) so that a long title is never cut at the edge."""
    width = max(30, int(0.95 * fig_width_in * 72.0 / (0.6 * fontsize)))
    return "\n".join(textwrap.fill(line, width=width) for line in text.split("\n"))


def _save(fig, path: Path, dpi: int | None) -> None:
    fig.savefig(path, dpi=_dpi(dpi))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------ tolerant CSV reading
def read_csv_rows(path: Path | str) -> tuple[list[str], list[dict]]:
    """(header, rows as {column: str}) of a CSV file that another process may still be writing.

    The last row is dropped when it is incomplete: the file does not end with a line break (the writer is in the middle of the row) or the row has
    fewer fields than the header.  Rows in the middle with missing fields get empty strings.  Raises FileNotFoundError for a missing file."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    body = list(csv.reader(io.StringIO(text)))
    body = [r for r in body if r]
    if not body:
        return [], []
    header, rows = [h.strip() for h in body[0]], body[1:]
    if rows and (not text.endswith("\n") or len(rows[-1]) < len(header)):
        rows = rows[:-1]
    return header, [dict(zip(header, r + [""] * (len(header) - len(r)))) for r in rows]


def _parse_float(s) -> float | None:
    """float of a CSV field (booleans as 0/1, empty / nan / None as NaN); None if the field is not a number (e.g. a truncated token)."""
    if s is None or s in ("", "nan", "NaN", "None"):
        return float("nan")
    if s in ("True", "true"):
        return 1.0
    if s in ("False", "false"):
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_float(s) -> float:
    v = _parse_float(s)
    return float("nan") if v is None else v


def _to_int(s) -> int | None:
    v = _parse_float(s)
    return None if v is None or not math.isfinite(v) else int(v)


def _truthy(s) -> bool:
    return str(s) in ("True", "true", "1", "1.0")


# ------------------------------------------------------------------------------------------------------------ run directories
def default_horizon(mode: str) -> int:
    """Episode horizon of a truth mode when config.json does not say (Mode F 300 steps, Mode T2 config.T2_MAX_STEPS)."""
    return int(config.T2_MAX_STEPS if mode == "T2" else config.MAX_EPISODE_STEPS)


def read_config(run: Path) -> dict:
    f = Path(run) / "config.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def run_mode(run: Path) -> tuple[str, int]:
    """(truth mode, episode horizon) of a training run; older runs have no truth_mode in config.json and were trained in Mode F."""
    c = read_config(run)
    mode = str(c.get("truth_mode") or config.ENV_TRUTH_MODE_DEFAULT)
    return mode, int(c.get("max_steps") or default_horizon(mode))


def run_n_drones(run: Path) -> int:
    """Drone count of a run: config.json n_drones, else derived from obs_dim, else 1."""
    c = read_config(run)
    if c.get("n_drones"):
        return int(c["n_drones"])
    if c.get("obs_dim"):
        return int(round((int(c["obs_dim"]) - config.ENV_OBS_DIM) / config.ENV_TEAMMATE_DIM)) + 1
    return 1


def ma_window(run: Path) -> int:
    return int((read_config(run).get("args") or {}).get("ma_window") or MA_WINDOW_DEFAULT)


def _ma_label(runs: Sequence[Path]) -> str:
    """Moving-average window of the runs for axis labels: "100", or "50/100" if the runs differ."""
    return "/".join(str(w) for w in sorted({ma_window(r) for r in runs}))


def _drones_txt(n: int) -> str:
    return f"{n} drone{'s' if n > 1 else ''}"


def _unique_names(runs: Sequence[Path]) -> list[str]:
    """Run directory names; a name used by several runs gets a #<k> suffix (column names and legends must be unique)."""
    names = [Path(r).name for r in runs]
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if names.count(n) > 1:
            seen[n] = seen.get(n, 0) + 1
            out.append(f"{n}#{seen[n]}")
        else:
            out.append(n)
    return out


def resolve_run(r: str | Path) -> Path:
    """A run name under config.TRAIN_ROOT or a path to a run directory (FileNotFoundError if neither exists: a typo, not a missing input)."""
    p = Path(str(r))
    q = p if p.exists() else config.TRAIN_ROOT / str(r)
    if not q.is_dir():
        raise FileNotFoundError(f"training run directory not found: {r} (looked at {p} and {config.TRAIN_ROOT / str(r)})")
    return q


def read_log(run: Path) -> dict[str, np.ndarray]:
    """train_log.csv of a run as {column: float array}; tolerant of a log that is being written (see read_csv_rows)."""
    fields, rows = read_csv_rows(Path(run) / "train_log.csv")
    return {k: np.array([_safe_float(r.get(k)) for r in rows], dtype=float) for k in fields}


def has_log_rows(run: Path) -> bool:
    return (Path(run) / "train_log.csv").exists() and read_log(run).get("env_steps", np.empty(0)).size > 0


def read_episodes(run: Path) -> dict[str, np.ndarray] | None:
    """episodes.csv of a run as {column: array} (numeric columns as float, booleans as 0/1, others as str); None if absent or empty.

    A column is numeric when most of its fields parse as numbers; a field that does not (truncated by a concurrent write) becomes NaN."""
    f = Path(run) / "episodes.csv"
    if not f.exists():
        return None
    fields, rows = read_csv_rows(f)
    if not rows:
        return None
    out: dict[str, np.ndarray] = {}
    for k in fields:
        vals = [r.get(k) for r in rows]
        parsed = [_parse_float(v) for v in vals]
        n_bad = sum(p is None for p in parsed)
        n_good = len(parsed) - n_bad
        if n_bad == 0 or (n_good > 0 and n_good >= n_bad):
            out[k] = np.array([float("nan") if p is None else p for p in parsed], dtype=float)
        else:
            out[k] = np.array(["" if v is None else v for v in vals], dtype=object)
    return out


def _num(ep: dict | None, key: str) -> np.ndarray | None:
    """A numeric column of read_episodes with at least one finite value, else None."""
    a = None if ep is None else ep.get(key)
    if a is None or a.dtype == object or not np.isfinite(a).any():
        return None
    return a


def trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` valid entries at every position (partial window at the start, as the training log's deque)."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    cn = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    idx = np.arange(1, x.size + 1)
    lo = np.maximum(idx - window, 0)
    n = cn[idx] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[idx] - cs[lo]) / np.maximum(n, 1.0), np.nan)


def trailing_success_median(length: np.ndarray, success: np.ndarray, window: int) -> np.ndarray:
    """Median length (= success step: training stops at the first success) of the successful episodes among the last ``window`` episodes; NaN if none."""
    length, success = np.asarray(length, dtype=float), np.asarray(success, dtype=float) > 0.5
    out = np.full(length.size, np.nan)
    for i in range(length.size):
        lo = max(0, i - window + 1)
        v = length[lo:i + 1][success[lo:i + 1]]
        if v.size:
            out[i] = float(np.median(v))
    return out


# ------------------------------------------------------------------------------------------------------------ baselines
def resolve_eval_dir(tag: str | Path) -> Path:
    """<CACHE_DIR>/eval/<tag> (or the tag itself if it is a directory path)."""
    d = config.CACHE_DIR / "eval" / str(tag)
    if d.is_dir():
        return d
    p = Path(str(tag))
    return p if p.is_dir() else d


def baseline_info(tag: str | Path, n_drones: int, run_mode_name: str | None = None, horizon: int | None = None) -> dict:
    """Baseline success rates on the training sources from <eval dir>/records_<method>_<n>drones.csv.

    Returns {"rates": {method: rate}, "n": {method: episodes}, "k": {method: successes}, "ci": {method: (lo, hi)} (Wilson 95 %), "note": str,
    "tag_mode": str | None, "horizon": int | None, "n_drones": int, "tag": str}.  The rates use the records' own horizon (summary.json max_steps,
    else the largest step count of an unsuccessful record); if ``horizon`` (the run's) is shorter, only successes within it count.  If the tag's truth
    mode (summary.json, else the records' mode column) differs from ``run_mode_name`` nothing is returned and the note says why (Mode F and Mode T2
    baselines are never mixed).  Record files that are still being written are read tolerantly (see read_csv_rows)."""
    d = resolve_eval_dir(tag)
    out: dict = {"rates": {}, "n": {}, "k": {}, "ci": {}, "note": "", "tag_mode": None, "horizon": None, "n_drones": int(n_drones), "tag": d.name}
    if not d.is_dir():
        out["note"] = f"baseline lines skipped: evaluation tag {tag} not found"
        return out
    tag_mode, rec_h = None, None
    sj = d / "summary.json"
    if sj.exists():
        s = json.loads(sj.read_text(encoding="utf-8"))
        tag_mode = s.get("mode")
        rec_h = int(s["max_steps"]) if s.get("max_steps") else None
    recs: dict[str, list[tuple[bool, int]]] = {}
    for m in BASE_COLORS:
        f = d / f"records_{m}_{n_drones}drones.csv"
        if not f.exists():
            continue
        _fields, rows = read_csv_rows(f)
        if tag_mode is None and rows and rows[0].get("mode"):
            tag_mode = rows[0]["mode"]
        sel = []
        for r in rows:
            src, st = _to_int(r.get("source")), _to_int(r.get("steps"))
            if src is not None and st is not None and src in config.TRAIN_SOURCES:
                sel.append((_truthy(r.get("success")), st))
        if sel:
            recs[m] = sel
    out["tag_mode"] = tag_mode
    if run_mode_name and tag_mode and tag_mode != run_mode_name:
        out["note"] = f"baseline lines skipped: tag {tag} was evaluated in Mode {tag_mode} but the run was trained in Mode {run_mode_name}"
        return out
    if not recs:
        out["note"] = f"baseline lines skipped: no records_<method>_{n_drones}drones.csv of the baselines on the training sources in {d.name}"
        return out
    if rec_h is None:
        fails = [st for rs in recs.values() for ok, st in rs if not ok]
        rec_h = max(fails) if fails else default_horizon(tag_mode or run_mode_name or config.ENV_TRUTH_MODE_DEFAULT)
    used = rec_h
    cut = horizon is not None and horizon < rec_h
    if cut:
        used = int(horizon)
    for m, rs in recs.items():
        k = int(sum(ok and (st <= used if cut else True) for ok, st in rs))
        _p, lo, hi = wilson_ci(k, len(rs))
        out["rates"][m] = k / len(rs)
        out["n"][m] = len(rs)
        out["k"][m] = k
        out["ci"][m] = (lo, hi)
    out["horizon"] = used
    out["note"] = f"baselines: tag {d.name}, Mode {tag_mode or '?'}, {n_drones}-drone records, {used}-step horizon, training sources"
    if horizon is not None and rec_h < horizon:
        out["note"] += f" (the records horizon {rec_h} is shorter than the run horizon {horizon})"
    return out


def baseline_rates(tag: str, n_drones: int) -> dict[str, float]:
    """Baseline success rates on the training sources (no truth-mode check; see baseline_info)."""
    return baseline_info(tag, n_drones)["rates"]


def _baseline_stats(b: dict, m: str) -> tuple[int, int, float, float, float]:
    """(k, n, rate, Wilson lo, Wilson hi) of method ``m`` of a baseline_info dict (a hand-made dict without k / ci is completed from rate and n)."""
    n = int(b.get("n", {}).get(m, 0))
    rate = float(b["rates"][m])
    k = int(b["k"][m]) if b.get("k") and m in b["k"] else int(round(rate * n))
    if b.get("ci") and m in b["ci"]:
        lo, hi = b["ci"][m]
    else:
        _p, lo, hi = wilson_ci(k, n)
    return k, n, rate, float(lo), float(hi)


def write_baselines_csv(path: Path, sets: Sequence[dict]) -> Path | None:
    """<prefix>_baselines.csv: one row per (drone count, method) of the baseline lines: k successes of n episodes on the training sources, rate and the
    Wilson 95 % interval.  None (nothing written) if there are no baselines."""
    rows = []
    for b in sets:
        for m in b["rates"]:
            k, n, rate, lo, hi = _baseline_stats(b, m)
            rows.append([b.get("tag", ""), b.get("tag_mode") or "", b.get("n_drones", ""), m, k, n, rate, lo, hi, b.get("horizon") or "",
                         "training sources " + " ".join(str(s) for s in config.TRAIN_SOURCES)])
    if not rows:
        return None
    header = ["tag", "truth_mode", "n_drones", "method", "k", "n", "rate", "ci95_lo", "ci95_hi", "horizon", "sources"]
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return Path(path)


# ------------------------------------------------------------------------------------------------------------ seed lines + mean band
def _steps_unit(top: float) -> tuple[float, str]:
    """(divisor, unit name) of the team-step axis: millions for long runs, thousands below STEPS_UNIT_SWITCH steps (smoke runs)."""
    return (1.0e6, "M") if top >= STEPS_UNIT_SWITCH else (1.0e3, "k")


def _interp_nan(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y(x) on ``grid`` ignoring non-finite pairs; NaN outside the range of x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return np.full(grid.shape, np.nan)
    return np.interp(grid, x[ok], y[ok], left=np.nan, right=np.nan)


def _nanmean0(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, 0)


def _nanstd0(a: np.ndarray) -> np.ndarray:
    """Sample standard deviation (ddof = 1) over axis 0 ignoring NaN; NaN where fewer than two values are defined (one seed has no spread)."""
    a = np.asarray(a, dtype=float)
    ok = np.isfinite(a)
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.where(ok, a, 0.0).sum(0) / np.maximum(n, 1), np.nan)
        var = np.where(ok, (a - mean) ** 2, 0.0).sum(0) / np.maximum(n - 1, 1)
    return np.where(n > 1, np.sqrt(var), np.nan)


def _band(ax, logs: list[dict[str, np.ndarray]], key: str, label: str, grid: np.ndarray, *, color: str | None = None, mean_color: str = "k",
          xkey: str = "env_steps", mean_label: str | None = None, names: Sequence[str] | None = None, xlabel: str | None = None,
          xdiv: float = 1.0e6, top: float | None = None) -> np.ndarray:
    """One thin line per run (``logs[i][xkey]`` [steps], ``logs[i][key]``) plus the mean and sd band (ddof = 1) over runs on ``grid`` (several runs);
    returns the runs interpolated on the grid (runs x grid, NaN where undefined or beyond ``top``)."""
    ys = []
    single = len(logs) == 1
    for i, lg in enumerate(logs):
        lab = names[i] if names is not None else None
        if single and mean_label:
            lab = mean_label
        ax.plot(np.asarray(lg[xkey], dtype=float) / xdiv, lg[key], color=color if color is not None else COLORS[i % len(COLORS)],
                lw=1.6 if single else 0.8, alpha=0.9 if single else 0.55, label=lab)
        ys.append(_interp_nan(grid, lg[xkey], lg[key]))
    ys = np.array(ys)
    if top is not None:
        ys[:, grid > top] = np.nan
    if not single:
        m, s = _nanmean0(ys), _nanstd0(ys)
        mc = color if color is not None else mean_color
        ax.plot(grid / xdiv, m, color=mc, lw=1.8, label=mean_label or f"mean of {len(logs)} seeds")
        ax.fill_between(grid / xdiv, m - s, m + s, color=mc, alpha=0.12)
    ax.set_xlabel(xlabel or f"team steps [{'M' if xdiv >= 1.0e6 else 'k'}]")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    return ys


def _group_by_drones(entries: Sequence[tuple[Path, str, dict]]) -> list:
    """Runs split by drone count, sorted by count: [(n_drones, [(name, series)])] from entries (run path, unique name, series)."""
    by: dict[int, list] = {}
    for run, name, s in entries:
        by.setdefault(run_n_drones(run), []).append((name, s))
    return sorted(by.items())


def _group_note(groups: list) -> str:
    if len(groups) < 2:
        return ""
    return "runs with different drone counts are drawn as separate groups, not pooled (" + "; ".join(
        f"{_drones_txt(n)}: {', '.join(nm for nm, _s in items)}" for n, items in groups) + ")"


def _band_groups(ax, groups: list, key: str, label: str, grid: np.ndarray, *, xkey: str = "env_steps", xdiv: float = 1.0e6,
                 seed_legend: bool = False) -> list[np.ndarray]:
    """_band for every drone-count group on the same axes: one colour per group (thin seed lines, mean line and sd band) if there are several
    groups, else the plain seed lines + black mean.  A group's mean stops where its shortest seed ends.  Returns the interpolated seeds per group."""
    multi = len(groups) > 1
    out = []
    for gi, (n, items) in enumerate(groups):
        lgs = [s for _nm, s in items]
        top = min(float(np.nanmax(np.asarray(s[xkey], dtype=float))) for s in lgs)
        if multi:
            ml = f"{_drones_txt(n)}: mean of {len(lgs)} seed{'s' if len(lgs) > 1 else ''}"
            ys = _band(ax, lgs, key, label, grid, color=GROUP_COLORS[gi % len(GROUP_COLORS)], xkey=xkey, mean_label=ml, xdiv=xdiv, top=top)
        else:
            ys = _band(ax, lgs, key, label, grid, xkey=xkey, names=[nm for nm, _s in items] if seed_legend and len(items) > 1 else None, xdiv=xdiv, top=top)
        out.append(ys)
    return out


def _add_curve_columns(table: dict, key: str, groups: list, ys_list: list[np.ndarray]) -> None:
    """CSV columns of one curve: <key>_mean, <key>_sd (ddof = 1) per group (suffix _<n>d if several groups) and <key>_seed_<run> per seed."""
    multi = len(groups) > 1
    for (n, _items), ys in zip(groups, ys_list):
        suf = f"_{n}d" if multi else ""
        table[f"{key}_mean{suf}"] = _nanmean0(ys)
        table[f"{key}_sd{suf}"] = _nanstd0(ys)
    for (_n, items), ys in zip(groups, ys_list):
        for (name, _s), y in zip(items, ys):
            table[f"{key}_seed_{name}"] = y


def _per_iteration(series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep the last episode of every iteration (all episodes of an iteration carry the same env_steps) so that the lines do not jump vertically."""
    x = np.asarray(series["env_steps"], dtype=float)
    keep = np.append(x[1:] != x[:-1], True)
    return {k: np.asarray(v)[keep] for k, v in series.items()}


def _min_yspan(ax, rel: float) -> None:
    """Widen the y range to at least ``rel`` times the magnitude of its centre (min 0.2) so that a nearly constant curve is not magnified."""
    lo, hi = ax.get_ylim()
    span = max(rel * abs(0.5 * (lo + hi)), 0.2)
    if hi - lo < span:
        mid = 0.5 * (lo + hi)
        ax.set_ylim(mid - 0.5 * span, mid + 0.5 * span)


def _note_empty(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="0.4", fontsize=9)


def _write_csv(path: Path, header: Sequence[str], columns: Sequence[Sequence]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(header))
        w.writerows(zip(*columns))


def fmt_steps(x: float) -> str:
    return f"{x / 1e6:.2f} M" if x >= 1e6 else f"{x / 1e3:.1f} k"


# ------------------------------------------------------------------------------------------------------------ item 1: learning curves
def _baseline_sets(base: dict[str, float] | None, base_n: dict[str, int] | None, baselines: Sequence[dict] | None) -> list[dict]:
    if baselines is not None:
        return [b for b in baselines if b.get("rates")]
    if base:
        return [{"rates": dict(base), "n": dict(base_n or {}), "k": {}, "ci": {}, "n_drones": None, "tag": "", "tag_mode": None, "horizon": None}]
    return []


def fig_curves(runs: list[Path], prefix: str, out_dir: Path, base: dict[str, float] | None = None, horizon: int | None = None, base_note: str = "",
               base_n: dict[str, int] | None = None, *, baselines: Sequence[dict] | None = None, dpi: int | None = None) -> dict:
    """Item 1.  ``baselines`` = baseline_info dicts (one per drone count; horizontal lines with Wilson 95 % CI in the legend, and
    <prefix>_baselines.csv); the legacy ``base`` / ``base_n`` (plain rates) draw the lines without an interval.  ``horizon`` = episode horizon of
    the runs (dotted line on the length panel).  Runs of different drone counts form separate colour groups.  Returns the final moving averages per run
    and ``baselines_csv`` (path or None)."""
    names = _unique_names(runs)
    logs = [read_log(r) for r in runs]
    groups = _group_by_drones([(r, nm, lg) for r, nm, lg in zip(runs, names, logs)])
    top = max(min(float(lg["env_steps"][-1]) for _nm, lg in items) for _n, items in groups)
    grid = np.linspace(0.0, top, GRID_POINTS)
    xdiv, unit = _steps_unit(top)
    sets = _baseline_sets(base, base_n, baselines)
    multi_set = len(sets) > 1
    n_lines = sum(len(b["rates"]) for b in sets)
    w = _ma_label(runs)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    table: dict[str, np.ndarray] = {"env_steps": grid}
    for ax, key, lab, ylab in zip(axes.flat, ("success_ma", "strict_ma", "return_ma", "length_ma"),
                                  (f"success rate (primary, {w}-episode mean)", f"strict success rate (sigma<15 m, error<20 m; {w}-episode mean)",
                                   f"team return per episode ({w}-episode mean)", f"episode length [steps] ({w}-episode mean)"),
                                  ("success rate", "strict success rate", "team return", "episode length [steps]")):
        ys_list = _band_groups(ax, groups, key, ylab, grid, xdiv=xdiv, seed_legend=key == "success_ma")
        _add_curve_columns(table, key, groups, ys_list)
        if key in ("success_ma", "strict_ma"):
            tops = [float(np.nanmax(ys)) for ys in ys_list if np.isfinite(ys).any()] or [0.0]
            if key == "success_ma":
                tops += [v for b in sets for v in b["rates"].values()]
            top_y = max(0.1, 1.15 * max(tops))
            ax.set_ylim(-0.03 * top_y, top_y)
        if key == "success_ma":
            li = 0
            for b in sets:
                nd = f", {_drones_txt(b['n_drones'])}" if multi_set and b.get("n_drones") else ""
                for m in b["rates"]:
                    k, n, v, lo, hi = _baseline_stats(b, m)
                    nn = f", n={n}" if n else ""
                    ci = f", 95 % CI {100 * lo:.1f}-{100 * hi:.1f}" if n else ""
                    ax.axhline(v, color=BASE_COLORS.get(m, "0.4"), ls=(3 * li, (4, 2 * n_lines)), lw=1.2,
                               label=f"{m}{nd}: {100 * v:.1f} %{ci}{nn}")       # dash offsets keep equal rates visible
                    li += 1
            if sets:
                ax.plot([], [], " ", label="baseline lines: training sources, interval = Wilson 95 % CI" + (f"; {base_note}" if base_note else ""))
        if key == "length_ma" and horizon:
            ax.axhline(horizon, color="0.5", ls=":", lw=1.0)
            ax.set_ylim(0, 1.08 * horizon)
        ax.set_title(lab, fontsize=9)
    h, lbl = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, lbl, loc="outside lower center", ncol=min(3, len(h)), fontsize=7, frameon=False)
    modes = sorted({run_mode(r)[0] for r in runs})
    if len(modes) > 1:
        mode_txt = f" (Modes {', '.join(modes)}: different episode lengths)"
    else:
        mode_txt = f" (Mode {modes[0]}, {horizon}-step episodes)" if horizon else ""
    fig.suptitle(_wrap(f"{prefix}: training curves of {len(runs)} run(s): " + ", ".join(names) + mode_txt
                       + f"\nthin lines = seeds, thick line = mean of seeds, band = mean +- sd over seeds (sample sd, ddof=1); x axis in {unit} team steps"
                       + (f"\n{_group_note(groups)}" if len(groups) > 1 else ""), fig.get_figwidth(), 9), fontsize=9)
    _save(fig, out_dir / f"{prefix}_curves.png", dpi)
    _write_csv(out_dir / f"{prefix}_curves.csv", list(table), list(table.values()))
    bcsv = write_baselines_csv(out_dir / f"{prefix}_baselines.csv", sets) if sets else None
    return {"final_success_ma": [float(lg["success_ma"][-1]) for lg in logs], "final_return_ma": [float(lg["return_ma"][-1]) for lg in logs],
            "baselines_csv": bcsv}


# ------------------------------------------------------------------------------------------------------------ item 2: return, length, success step
def fig_episode_stats(runs: list[Path], prefix: str, out_dir: Path, dpi: int | None = None) -> dict:
    """Item 2: episode return, episode length and median success step versus team steps from episodes.csv (trailing windows of ma_window episodes);
    one colour group per drone count."""
    entries, used, skipped = [], [], []
    for run in runs:
        ep = read_episodes(run)
        ret, length, succ = _num(ep, "ret"), _num(ep, "length"), _num(ep, "success")
        if ret is None or length is None or succ is None or _num(ep, "env_steps") is None:
            skipped.append(run.name)
            continue
        w = ma_window(run)
        entries.append((run, {"env_steps": ep["env_steps"], "ret": trailing_mean(ret, w), "length": trailing_mean(length, w),
                              "success_step": trailing_success_median(length, succ, w)}))
        used.append(run)
    if not entries:
        raise Skip(f"no episodes.csv with ret / length / success columns in: {', '.join(r.name for r in runs)}")
    names = _unique_names(used)
    groups = _group_by_drones([(r, nm, _per_iteration(s)) for (r, s), nm in zip(entries, names)])
    all_series = [s for _n, items in groups for _nm, s in items]
    top = max(min(float(s["env_steps"][-1]) for _nm, s in items) for _n, items in groups)
    grid = np.linspace(0.0, top, GRID_POINTS)
    xdiv, unit = _steps_unit(top)
    horizons = sorted({run_mode(r)[1] for r in used})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), constrained_layout=True)
    table: dict[str, np.ndarray] = {"env_steps": grid}
    panels = (("ret", "team return per episode", "mean"), ("length", "episode length [steps]", "mean"),
              ("success_step", "median success step [steps]", "median of successes"))
    for ax, (key, lab, how) in zip(axes, panels):
        ys_list = _band_groups(ax, groups, key, lab, grid, xdiv=xdiv, seed_legend=key == "ret")
        _add_curve_columns(table, key, groups, ys_list)
        if key != "ret":
            for h in horizons:
                ax.axhline(h, color="0.5", ls=":", lw=1.0)
            ax.set_ylim(0, 1.08 * max(horizons))
        if key == "ret" and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        if not any(np.isfinite(s[key]).any() for s in all_series):
            _note_empty(ax, "no successful episode in any window yet")
        ax.set_xlim(0.0, 1.02 * max(float(s["env_steps"][-1]) for s in all_series) / xdiv)      # all seeds in full; the success-step line starts later
        ax.set_title(f"{lab} ({how})", fontsize=9)
    fig.suptitle(f"{prefix}: episode statistics of {len(used)} run(s): " + ", ".join(names)
                 + f"\ntrailing windows of {_ma_label(used)} episodes; dotted line = episode horizon; success step = length of the episodes that ended in success; "
                 f"band = mean +- sd over seeds (ddof=1); x axis in {unit} team steps" + (f"\n{_group_note(groups)}" if len(groups) > 1 else ""), fontsize=9)
    path = out_dir / f"{prefix}_returns.png"
    _save(fig, path, dpi)
    _write_csv(out_dir / f"{prefix}_returns.csv", list(table), list(table.values()))
    reasons = []
    if skipped:
        reasons.append("runs without usable episodes.csv skipped: " + ", ".join(skipped))
    if len(groups) > 1:
        reasons.append(_group_note(groups))
    return {"png": path, "csv": out_dir / f"{prefix}_returns.csv", "reason": "; ".join(reasons)}


# ------------------------------------------------------------------------------------------------------------ item 3: PPO diagnostics
def fig_diagnostics(runs: list[Path], prefix: str, out_dir: Path, dpi: int | None = None) -> None:
    logs = [read_log(r) for r in runs]
    keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var", "masked_share", "steps_per_s")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for ax, key in zip(axes.flat, keys):
        for i, lg in enumerate(logs):
            if key in lg and "iteration" in lg:
                ax.plot(lg["iteration"], lg[key], color=COLORS[i % 5], lw=0.8, label=runs[i].name)
        if not ax.lines:
            _note_empty(ax, f"no {key} column in train_log.csv")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"{prefix}: PPO diagnostics (policy entropy starts near ln 9 = 2.2; a collapse or a KL spike marks instability)", fontsize=10)
    _save(fig, out_dir / f"{prefix}_diagnostics.png", dpi)


# ------------------------------------------------------------------------------------------------------------ items 4 and 5: checkpoint quick evaluation
def _last_env_steps(run: Path) -> float:
    if not has_log_rows(run):
        return float("nan")
    return float(read_log(run)["env_steps"][-1])


def _ckpt_env_steps(stem: str, last: float, rows: list[dict]) -> float | None:
    """Team steps of a checkpoint from its file stem (ckpt_<steps>; final = last row of train_log.csv, else the CSV env_steps column)."""
    if stem == "final":
        if np.isfinite(last):
            return float(last)
        v = rows[0].get("env_steps")
        return float(v) if v not in (None, "", "None") else None
    parts = stem.split("_")
    return float(parts[1]) if len(parts) == 2 and parts[0] == "ckpt" and parts[1].isdigit() else None


def _ckpt_eval_files(run: Path) -> list[tuple[Path, float, list[dict]]]:
    """(csv path, team steps of the checkpoint, records) of every quick-evaluation CSV of a run, sorted by steps; rows without a valid source dropped."""
    d = Path(run) / "ckpt_eval"
    out = []
    if not d.is_dir():
        return out
    last = _last_env_steps(run)
    for f in sorted(d.glob("*.csv")):
        _fields, rows = read_csv_rows(f)
        rows = [r for r in rows if _to_int(r.get("source")) is not None]
        if not rows:
            continue
        steps = _ckpt_env_steps(f.stem, last, rows)
        if steps is not None:
            out.append((f, steps, rows))
    return sorted(out, key=lambda t: t[1])


def read_ckpt_eval(run: Path) -> list[dict]:
    """One row per evaluated checkpoint: team steps, primary / strict success and median final error on training and held-out sources."""
    out = []
    for _f, steps, rows in _ckpt_eval_files(run):
        row = {"run": run.name, "env_steps": steps, "n": len(rows)}
        for g, sel in (("train", [r for r in rows if int(r["source"]) not in config.HOLDOUT_SOURCES]),
                       ("holdout", [r for r in rows if int(r["source"]) in config.HOLDOUT_SOURCES])):
            err = [e for e in (_safe_float(r.get("final_error_m")) for r in sel) if math.isfinite(e)]
            row[f"success_{g}"] = float(np.mean([_truthy(r.get("success")) for r in sel])) if sel else float("nan")
            row[f"strict_{g}"] = float(np.mean([_truthy(r.get("success_strict")) for r in sel])) if sel else float("nan")
            row[f"error_{g}"] = float(np.median(err)) if err else float("nan")
            row[f"n_{g}"] = len(sel)
        out.append(row)
    return sorted(out, key=lambda r: r["env_steps"])


def fig_ckpt_eval(runs: list[Path], prefix: str, out_dir: Path, dpi: int | None = None) -> list[dict]:
    """Item 5; returns the rows behind the figure ([] and nothing written if no run has quick-evaluation records)."""
    allrows = [r for run in runs for r in read_ckpt_eval(run)]
    if not allrows:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    xdiv, unit = _steps_unit(max(r["env_steps"] for r in allrows))
    for i, run in enumerate(runs):
        rows = [r for r in allrows if r["run"] == run.name]
        x = [r["env_steps"] / xdiv for r in rows]
        for ax, k, lab in zip(axes, ("success", "strict", "error"), ("primary success rate", "strict success rate", "median final error [m]")):
            ax.plot(x, [r[f"{k}_train"] for r in rows], "o-", color=COLORS[i % 5], lw=1.2, label=f"{run.name} training sources")
            ax.plot(x, [r[f"{k}_holdout"] for r in rows], "s--", color=COLORS[i % 5], lw=1.0, alpha=0.7, label=f"{run.name} held-out sources")
            ax.set_xlabel(f"team steps [{unit}]")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
    for ax, k in zip(axes[:2], ("success", "strict")):
        vals = [r[f"{k}_{g}"] for r in allrows for g in ("train", "holdout") if np.isfinite(r[f"{k}_{g}"])]
        top_y = max(0.1, 1.15 * max(vals, default=0.0))
        ax.set_ylim(-0.03 * top_y, top_y)
    axes[0].legend(fontsize=7)
    n_eps = sorted({r["n"] for r in allrows})
    fig.suptitle(f"{prefix}: checkpoint quick evaluation ({'/'.join(str(n) for n in n_eps)} episodes per checkpoint on the observable sources, "
                 f"full horizon of the training mode, sampled actions; small samples, curves only)", fontsize=10)
    _save(fig, out_dir / f"{prefix}_ckpt_eval.png", dpi)
    with (out_dir / f"{prefix}_ckpt_eval.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    return allrows


# ------------------------------------------------------------------------------------------------------------ item 4: per-source heat map
def _heatmap_sources(present: set[int]) -> list[tuple[int, str]]:
    """Row order of the heat map: the metrics.GROUPS partition (train_open, train_other, holdout), source 110 excluded; (source, group name)."""
    rows = []
    for g in ("train_open", "train_other", "holdout"):
        rows += [(s, g) for s in GROUPS[g] if s in present and s not in config.EXCLUDED_SOURCES]
    seen = {s for s, _ in rows}
    return rows + [(s, "other") for s in sorted(present) if s not in seen and s not in config.EXCLUDED_SOURCES]


def heatmap_layout(n_panels: int, max_checkpoints: int, n_sources: int) -> tuple[int, int, tuple[float, float]]:
    """(columns, rows, figure size [in]) of the heat-map figure: a panel is as wide as its checkpoint columns need, as many panels as fit into
    HEATMAP_MAX_WIDTH_IN go side by side and the rest wraps into further rows (the figure never exceeds HEATMAP_MAX_WIDTH_IN in width)."""
    avail = HEATMAP_MAX_WIDTH_IN - 1.4                                   # colour bar and margins
    pw = float(np.clip(HEATMAP_PANEL_BASE_IN + HEATMAP_CELL_IN * max_checkpoints, HEATMAP_PANEL_MIN_IN, avail))
    ncols = int(max(1, min(n_panels, avail // pw)))
    nrows = math.ceil(n_panels / ncols)
    return ncols, nrows, (max(pw * ncols + 1.4, HEATMAP_MIN_WIDTH_IN), nrows * max(3.6, 0.3 * n_sources + 1.6) + 1.3)


def fig_source_heatmap(runs: list[Path], prefix: str, out_dir: Path, dpi: int | None = None) -> dict:
    """Item 4: per-source primary success rate, source x checkpoint, from ckpt_eval/*.csv (one panel per run, wrapped into rows; source 110 absent)."""
    per_run = [(r, _ckpt_eval_files(r)) for r in runs]
    per_run = [(r, fs) for r, fs in per_run if fs]
    if not per_run:
        raise Skip("no ckpt_eval/*.csv (quick-evaluation records) in " + ", ".join(r.name for r in runs))
    present = {int(rec["source"]) for _r, fs in per_run for _f, _s, rows in fs for rec in rows}
    srcs = _heatmap_sources(present)
    if not srcs:
        raise Skip("the quick evaluation contains no observable source")
    table = []
    n_pan = len(per_run)
    max_ck = max(len(fs) for _r, fs in per_run)
    ncols_fig, nrows_fig, figsize = heatmap_layout(n_pan, max_ck, len(srcs))
    fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=figsize, constrained_layout=True, squeeze=False)
    cmap = plt.get_cmap("Blues").with_extremes(bad="0.9")
    im = None
    for ax, (run, fs) in zip(axes.flat, per_run):
        mat = np.full((len(srcs), len(fs)), np.nan)
        cnt = np.zeros((len(srcs), len(fs)), dtype=int)
        ksuc = np.zeros_like(cnt)
        for j, (f, steps, rows) in enumerate(fs):
            for i, (s, g) in enumerate(srcs):
                sel = [r for r in rows if int(r["source"]) == s]
                if sel:
                    cnt[i, j], ksuc[i, j] = len(sel), sum(_truthy(r.get("success")) for r in sel)
                    mat[i, j] = ksuc[i, j] / cnt[i, j]
                    table.append({"run": run.name, "ckpt": f.stem, "env_steps": steps, "source": s, "group": g, "n": int(cnt[i, j]),
                                  "n_success": int(ksuc[i, j]), "success_rate": float(mat[i, j]),
                                  "n_success_strict": sum(_truthy(r.get("success_strict")) for r in sel)})
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(srcs)))
        ax.set_yticklabels([f"{s} {'held-out' if g == 'holdout' else ('open' if g == 'train_open' else 'other')}" for s, g in srcs], fontsize=7)
        ax.set_xticks(range(len(fs)))
        many = len(fs) > 6
        ax.set_xticklabels([fmt_steps(st) for _f, st, _r in fs], fontsize=7, rotation=45 if many else 0, ha="right" if many else "center")
        ax.set_xlabel("checkpoint (team steps)", fontsize=8)
        gs = [g for _s, g in srcs]
        for i in range(1, len(gs)):
            if gs[i] != gs[i - 1]:
                ax.axhline(i - 0.5, color="k", lw=1.0)
        if len(fs) <= HEATMAP_ANNOTATE_MAX_COLS:
            for i in range(len(srcs)):
                for j in range(len(fs)):
                    if cnt[i, j]:
                        ax.text(j, i, f"{ksuc[i, j]}/{cnt[i, j]}", ha="center", va="center", fontsize=6.5, color="white" if mat[i, j] > 0.55 else "0.15")
        nd = run_n_drones(run)
        ax.set_title(f"{run.name} (Mode {run_mode(run)[0]}, {_drones_txt(nd)})", fontsize=9)
    for ax in list(axes.flat)[n_pan:]:
        ax.set_visible(False)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
    cb.set_label("primary success rate")
    fig.suptitle(f"{prefix}: per-source success rate by checkpoint (quick evaluation; cells = successes / episodes)\n"
                 f"rows grouped as training open, training other (trapped 102, 104, 106 etc.) and held-out; source 110 is not observable and absent", fontsize=9)
    path = out_dir / f"{prefix}_source_heatmap.png"
    _save(fig, path, dpi)
    cols = ["run", "ckpt", "env_steps", "source", "group", "n", "n_success", "success_rate", "n_success_strict"]
    _write_csv(out_dir / f"{prefix}_source_heatmap.csv", cols, [[t[c] for t in table] for c in cols])
    return {"png": path, "csv": out_dir / f"{prefix}_source_heatmap.csv", "reason": ""}


# ------------------------------------------------------------------------------------------------------------ item 6: reward decomposition
def fig_reward_decomposition(runs: list[Path], prefix: str, out_dir: Path, dpi: int | None = None) -> dict:
    """Item 6: moving averages of ret_info, ret_time and ret_terminal (and the total) of the training episodes versus team steps; one colour group
    (panels 1-3) / line style (panel 4) per drone count."""
    comps = ("ret_info", "ret_time", "ret_terminal")
    entries, used, skipped = [], [], []
    for run in runs:
        ep = read_episodes(run)
        cols = {k: _num(ep, k) for k in (*comps, "ret", "env_steps")}
        if any(cols[k] is None for k in (*comps, "env_steps")):
            skipped.append(run.name)
            continue
        w = ma_window(run)
        s = {"env_steps": cols["env_steps"]}
        for k in comps:
            s[k] = trailing_mean(cols[k], w)
        s["ret"] = trailing_mean(cols["ret"], w) if cols["ret"] is not None else s["ret_info"] + s["ret_time"] + s["ret_terminal"]
        entries.append((run, _per_iteration(s)))
        used.append(run)
    if not entries:
        raise Skip("skipped: run predates reward decomposition (no ret_info / ret_time / ret_terminal columns in episodes.csv of "
                   + ", ".join(r.name for r in runs) + ")")
    names = _unique_names(used)
    groups = _group_by_drones([(r, nm, s) for (r, s), nm in zip(entries, names)])
    multi = len(groups) > 1
    top = max(min(float(s["env_steps"][-1]) for _nm, s in items) for _n, items in groups)
    grid = np.linspace(0.0, top, GRID_POINTS)
    xdiv, unit = _steps_unit(top)
    titles = {"ret_info": "information-gain term: sum of (H[t-1] - H[t]) / H0", "ret_time": "time term: -0.02 per step",
              "ret_terminal": "terminal term: success bonus / failure penalty (incl. exit)"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    table: dict[str, np.ndarray] = {"env_steps": grid}
    for ax, k in zip(axes.flat[:3], comps):
        ys_list = _band_groups(ax, groups, k, "reward per episode", grid, xdiv=xdiv, seed_legend=k == "ret_info")
        _add_curve_columns(table, k, groups, ys_list)
        _min_yspan(ax, 0.1)
        ax.set_title(titles[k], fontsize=9)
        if k == "ret_info" and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    ax = axes.flat[3]
    for k, lab in (("ret_info", "information gain"), ("ret_time", "time"), ("ret_terminal", "terminal"), ("ret", "total return")):
        for gi, (n, items) in enumerate(groups):
            ys = np.array([_interp_nan(grid, s["env_steps"], s[k]) for _nm, s in items])
            ys[:, grid > min(float(s["env_steps"][-1]) for _nm, s in items)] = np.nan
            m = _nanmean0(ys)
            if k == "ret":
                suf = f"_{n}d" if multi else ""
                table["ret_mean" + suf], table["ret_sd" + suf] = m, _nanstd0(ys)
                for (nm, _s), y in zip(items, ys):
                    table[f"ret_seed_{nm}"] = y
            ax.plot(grid / xdiv, m, color=COMPONENT_COLORS[k], ls=GROUP_STYLES[gi % len(GROUP_STYLES)], lw=2.0 if k == "ret" else 1.4,
                    label=lab + (f" ({n} d)" if multi else ""))
    ax.axhline(0.0, color="0.6", lw=0.6)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - (0.32 if multi else 0.22) * (hi - lo), hi)                  # free band at the bottom for the legend
    ax.set_xlabel(f"team steps [{unit}]")
    ax.set_ylabel("reward per episode (mean over seeds)")
    ax.set_title("all terms: total = information gain + time + terminal", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower center", ncol=4)
    fig.suptitle(f"{prefix}: reward decomposition of the training episodes (moving averages of {_ma_label(used)} episodes; band = mean +- sd over "
                 f"seeds, ddof=1) of " + ", ".join(names) + (f"\n{_group_note(groups)}" if multi else ""), fontsize=10)
    path = out_dir / f"{prefix}_reward_decomposition.png"
    _save(fig, path, dpi)
    _write_csv(out_dir / f"{prefix}_reward_decomposition.csv", list(table), list(table.values()))
    reasons = []
    if skipped:
        reasons.append("runs predating the reward decomposition skipped: " + ", ".join(skipped))
    if multi:
        reasons.append(_group_note(groups))
    return {"png": path, "csv": out_dir / f"{prefix}_reward_decomposition.csv", "reason": "; ".join(reasons)}


# ------------------------------------------------------------------------------------------------------------ item 7: sample trajectories
def load_step_log(path: Path) -> tuple[dict, dict]:
    """(arrays, meta) of a step-log NPZ (spec 11.7; same contract as scripts/fig_episode.load_step_log, kept local so that this script stays light)."""
    z = np.load(path, allow_pickle=False)
    arrs = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return arrs, meta


def episode_frames(log: dict, meta: dict, n_steps: int, mode_hint: str = "F") -> np.ndarray:
    """Truth frame of every step of a step log: the logged ``frame`` array of newer logs, else derived (Mode F: the constant meta frame; Mode T2:
    frame0 + files_per_step * t for the 0-based measurement index t, clipped to the cached frames; mode from meta, else ``mode_hint``)."""
    if "frame" in log and len(log["frame"]) == n_steps:
        return np.asarray(log["frame"], dtype=int)
    mode = str(meta.get("mode") or mode_hint or "F")
    f0 = int(meta["frame"])
    if mode == "T2":
        return np.minimum(config.N_FILES - 1, np.rint(f0 + config.T1_4_MODE_T2_FILES_PER_STEP * np.arange(n_steps))).astype(int)
    return np.full(n_steps, f0, dtype=int)


def ckpt_step_logs(run: Path) -> list[dict]:
    """Checkpoints of a run that have sample step logs: [{"stem", "env_steps", "episodes": {episode_id: path}}], sorted by team steps."""
    d = Path(run) / "ckpt_eval"
    out = []
    if not d.is_dir():
        return out
    last = _last_env_steps(run)
    for sd in sorted(p for p in d.iterdir() if p.is_dir() and p.name.endswith("_steps")):
        eps = {int(m.group(1)): f for f in sorted(sd.glob("ep*.npz")) if (m := re.fullmatch(r"ep(\d+)\.npz", f.name))}
        stem = sd.name[:-len("_steps")]
        steps = _ckpt_env_steps(stem, last, [{}]) if eps else None
        if steps is not None:
            out.append({"stem": stem, "env_steps": steps, "episodes": eps})
    return sorted(out, key=lambda e: e["env_steps"])


def _peek(path: Path) -> tuple[dict, float] | None:
    """(meta, final MAP error) of a step log without reading the other arrays; None if the file cannot be read (still being written)."""
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        err = float(z["map_error"][-1]) if "map_error" in z.files and len(z["map_error"]) else float("nan")
    except _UNREADABLE:
        return None
    return meta, err


def _select_checkpoints(entries: list[dict]) -> list[tuple[str, dict]]:
    """Initial (the first periodic checkpoint with step logs; not the untrained policy), middle (closest to the midpoint of the interior checkpoints)
    and late (the last checkpoint, normally final)."""
    if len(entries) < 2:
        raise Skip(f"need step logs of at least 2 checkpoints for initial / late policies, found {len(entries)}")
    sel = [("initial", entries[0])]
    if len(entries) >= 3:
        mid_target = 0.5 * (entries[0]["env_steps"] + entries[-1]["env_steps"])
        sel.append(("middle", min(entries[1:-1], key=lambda e: abs(e["env_steps"] - mid_target))))
    sel.append(("late", entries[-1]))
    return sel


def _pick_episode(sel: list[tuple[str, dict]], source: int | None, episode_id: int | None) -> tuple[int, int]:
    """(episode id, source) of the sample episode: the episodes common to all selected checkpoints, filtered by the requested source / id, preferring
    one the late checkpoint solved (then the smallest final MAP error, then the lowest id)."""
    common = set.intersection(*[set(e["episodes"]) for _l, e in sel])
    if not common:
        raise Skip("the selected checkpoints share no episode id")
    late = sel[-1][1]
    cands = []
    for eid in sorted(common):
        pk = _peek(late["episodes"][eid])
        if pk is None:
            continue                                                    # unreadable: still being written
        meta, err = pk
        cands.append((eid, int(meta.get("source", -1)), bool(meta.get("success", False)), err if np.isfinite(err) else float("inf")))
    if episode_id is not None:
        cands = [c for c in cands if c[0] == episode_id]
    if source is not None:
        cands = [c for c in cands if c[1] == source]
    if not cands:
        raise Skip(f"requested sample episode (source {source}, episode {episode_id}) not among the common step logs {sorted(common)}")
    best = min(cands, key=lambda c: (not c[2], c[3], c[0]))
    return best[0], best[1]


def _window(points: np.ndarray, margin: float = TRAJ_MARGIN_M, min_aspect: float = TRAJ_MIN_ASPECT) -> tuple[float, float, float, float]:
    lo, hi = points.min(axis=0) - margin, points.max(axis=0) + margin
    for a, b in ((1, 0), (0, 1)):
        span, other = hi[a] - lo[a], hi[b] - lo[b]
        if span < min_aspect * other:
            mid = 0.5 * (hi[a] + lo[a])
            lo[a], hi[a] = mid - 0.5 * min_aspect * other, mid + 0.5 * min_aspect * other
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def _map_in_reach(box_pts: np.ndarray, maps: np.ndarray, reach: float = TRAJ_MAP_REACH_M) -> np.ndarray:
    """The MAP positions (n, 2) within ``reach`` metres of the bounding box of ``box_pts`` (paths + source): they widen the window, farther ones do not."""
    lo, hi = box_pts.min(axis=0), box_pts.max(axis=0)
    d = np.maximum(np.maximum(lo - maps, maps - hi), 0.0)
    return maps[np.hypot(d[:, 0], d[:, 1]) <= reach]


def _arrow_to_outside(ax, win: tuple[float, float, float, float], target: Sequence[float]) -> None:
    """Red arrow from inside the window towards a MAP estimate beyond its edge, labelled with the distance from the window."""
    cx, cy = 0.5 * (win[0] + win[2]), 0.5 * (win[1] + win[3])
    hx, hy = 0.5 * (win[2] - win[0]), 0.5 * (win[3] - win[1])
    dx, dy = float(target[0]) - cx, float(target[1]) - cy
    s = min(0.92 * hx / abs(dx) if dx else math.inf, 0.92 * hy / abs(dy) if dy else math.inf)
    tip, tail = (cx + s * dx, cy + s * dy), (cx + 0.72 * s * dx, cy + 0.72 * s * dy)
    ax.annotate("", xy=tip, xytext=tail, arrowprops={"arrowstyle": "-|>", "color": "tab:red", "lw": 2.0}, zorder=6)
    out = math.hypot(max(win[0] - target[0], 0.0, target[0] - win[2]), max(win[1] - target[1], 0.0, target[1] - win[3]))
    horizontal = abs(dx) / hx >= abs(dy) / hy                               # the label sits at the tail, on the side of the window centre, so that it stays inside the axes
    ha = ("right" if dx > 0 else "left") if horizontal else "center"
    va = "bottom" if horizontal else ("top" if dy > 0 else "bottom")
    ax.text(tail[0], tail[1], f"MAP {out:.0f} m outside", color="tab:red", fontsize=7, ha=ha, va=va, zorder=8, clip_on=True,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})


def _ellipse_patch(mu, cov, weight: float, colour: str) -> Ellipse:
    """2-sigma ellipse of a GMM component (same convention as fig_episode)."""
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    vals = np.maximum(vals, 1e-6)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return Ellipse(mu, 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang, fill=False, edgecolor=colour, linewidth=0.8 + 1.8 * weight,
                   alpha=0.55 + 0.45 * weight)


def _draw_buildings(ax, om, win) -> None:
    """Building outlines (occupancy contour) inside the window; ``om`` needs x0, y0, res, nx, ny, x, y and occ[ix, iy] (env.drone.ObstacleMap)."""
    ix0, ix1 = max(int((win[0] - om.x0) // om.res) - 1, 0), min(int((win[2] - om.x0) // om.res) + 2, om.nx)
    iy0, iy1 = max(int((win[1] - om.y0) // om.res) - 1, 0), min(int((win[3] - om.y0) // om.res) + 2, om.ny)
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return
    sub = np.asarray(om.occ)[ix0:ix1, iy0:iy1]
    if sub.any() and not sub.all():
        ox, oy = np.meshgrid(np.asarray(om.x)[ix0:ix1], np.asarray(om.y)[iy0:iy1], indexing="ij")
        ax.contour(ox, oy, sub.astype(float), levels=[0.5], colors="0.25", linewidths=0.6)


def load_scene_helpers():
    """Default (ObstacleMap, slab backend) of the real data; Skip if the cache files are missing."""
    try:
        from srcloc_env.env.drone import ObstacleMap
        om = ObstacleMap.load()
        if config.SLAB_STACK_PATH.exists() and config.SLAB_STACK_META_PATH.exists():
            from srcloc_env.field.slab_stack import StackedSlabBackend
            return om, StackedSlabBackend()
        from srcloc_env.field.concentration_field import LdmSlabBackend
        return om, LdmSlabBackend()
    except FileNotFoundError as exc:
        raise Skip(f"building map or slab data not available: {exc}") from exc


def _count_cmap() -> LinearSegmentedColormap:
    """Warm colour map of the measurement dots without the palest yellows (those vanish on the white / grey background)."""
    return LinearSegmentedColormap.from_list("counts", plt.get_cmap("YlOrRd")(np.linspace(0.3, 1.0, 256)))


def fig_trajectories(run: Path, prefix: str, out_dir: Path, om=None, backend=None, source: int | None = None, episode_id: int | None = None,
                     fractions: Sequence[float] | None = None, dpi: int | None = None) -> dict:
    """Item 7: the same quick-evaluation episode (same source, frame, scale and start) at an initial (the first periodic checkpoint), a middle and a late
    (last) checkpoint, one row of ``len(fractions)`` snapshots each (figure 7 format).  Step logs come from ckpt_eval/<ckpt>_steps/ep<id>.npz.  ``om``
    (ObstacleMap-like) and ``backend`` (slab backend: .slab(frame) -> object with .density, .sources, .grid, .z_index) are loaded on demand.  Every drone
    has its own path colour (the path lies under the dots, which are coloured by the measured count), the window includes the MAP estimates within
    TRAJ_MAP_REACH_M (a farther one is marked by an arrow), the suptitle states the checkpoint and episode selection rules.  The result carries the
    window ("window": x0, y0, x1, y1)."""
    fractions = TRAJ_FRACTIONS if fractions is None else fractions
    entries = ckpt_step_logs(run)
    if not entries:
        raise Skip(f"no ckpt_eval/<ckpt>_steps/ep*.npz step logs in {run.name} (run predates the sample-trajectory logs or --ckpt-eval-log-per-source was 0)")
    sel = _select_checkpoints(entries)
    eid, src = _pick_episode(sel, source, episode_id)
    mode_hint = run_mode(run)[0]
    rows = []
    for role, e in sel:
        fpath = e["episodes"][eid]
        try:
            log, meta = load_step_log(fpath)
        except _UNREADABLE as exc:
            raise Skip(f"step log {fpath.name} of checkpoint {e['stem']} is unreadable (still being written?): {exc}") from exc
        miss = [k for k in TRAJ_REQUIRED_KEYS if k not in log]
        if miss:
            raise Skip(f"step log {fpath.name} of checkpoint {e['stem']} lacks keys {miss}")
        rows.append({"role": role, "stem": e["stem"], "env_steps": e["env_steps"], "log": log, "meta": meta})
    if om is None or backend is None:
        om, backend = load_scene_helpers()
    from srcloc_env.sensor.detector import Detector     # light (numpy + config)
    det = Detector()
    ref = rows[-1]
    truth = np.asarray(ref["log"]["truth_xy"], dtype=float)
    steps_of: list[list[int]] = []
    for r in rows:
        T = int(r["log"]["drone_xy"].shape[0])
        D = int(r["log"]["drone_xy"].shape[1])
        steps_of.append([max(1, min(T, int(round(f * T)))) for f in fractions])
        r["frames"] = episode_frames(r["log"], r["meta"], T, mode_hint)
        r["yn"] = det.normalise(np.asarray(r["log"]["y"], dtype=float).reshape(-1)).reshape(T, D)
        r["start"] = np.asarray(r["log"]["start_xy"], dtype=float) if "start_xy" in r["log"] else np.asarray(r["log"]["drone_xy"][0], dtype=float)
    box_pts = np.concatenate([r["log"]["drone_xy"].reshape(-1, 2) for r in rows] + [truth[None]])
    maps = np.array([np.asarray(r["log"]["map_xy"][t - 1], dtype=float) for r, st in zip(rows, steps_of) for t in st])
    win = _window(np.concatenate([box_pts, _map_in_reach(box_pts, maps)]))
    ncol = len(fractions)
    ph = float(np.clip((TRAJ_PANEL_IN - 0.25) * (win[3] - win[1]) / (win[2] - win[0]), 2.0, 4.8))     # the real panel is a bit narrower than the slot
    fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * TRAJ_PANEL_IN + 1.5, len(rows) * (ph + 0.7) + 1.9), constrained_layout=True,
                             squeeze=False, sharex=True, sharey=True)
    # slab density of the source at every snapshot frame (one colour scale over the whole figure)
    dens: dict[int, tuple[np.ndarray, object]] = {}
    for r, steps in zip(rows, steps_of):
        for t in steps:
            fr = int(r["frames"][t - 1])
            if fr not in dens:
                try:
                    sf = backend.slab(fr)
                except FileNotFoundError as exc:
                    raise Skip(f"slab frame {fr} not available: {exc}") from exc
                dens[fr] = (np.asarray(sf.density[list(sf.sources).index(src), sf.z_index(config.DRONE_Z)], dtype=float), sf.grid)
    vmax = max(max(float(d.max()) for d, _g in dens.values()), 1e-12)
    ymin, ymax = min(float(r["yn"].min()) for r in rows), max(float(r["yn"].max()) for r in rows)
    if ymax - ymin < 0.05:
        ymax = ymin + 0.05                                  # colour scale of the dots spans the measured counts of the whole figure
    cmap = _count_cmap()
    D_max = max(int(r["log"]["drone_xy"].shape[1]) for r in rows)
    sc = None
    for ri, r in enumerate(rows):
        log, T = r["log"], int(r["log"]["drone_xy"].shape[0])
        D = int(log["drone_xy"].shape[1])
        yn, start = r["yn"], r["start"]
        for ci, t in enumerate(steps_of[ri]):
            ax = axes[ri][ci]
            d, g = dens[int(r["frames"][t - 1])]
            ax.imshow(np.log10(np.maximum(d, vmax * 10 ** -TRAJ_DENSITY_DECADES)), origin="lower", extent=(g.x0, g.x0 + g.nx * g.res, g.y0, g.y0 + g.ny * g.res),
                      cmap="Greys", vmin=np.log10(vmax) - TRAJ_DENSITY_DECADES, vmax=np.log10(vmax), alpha=0.7, interpolation="nearest")
            _draw_buildings(ax, om, win)
            for k in range(D):
                xy, col = log["drone_xy"][:t, k], DRONE_COLORS[k % len(DRONE_COLORS)]
                ax.plot(xy[:, 0], xy[:, 1], "-", color=col, lw=TRAJ_PATH_LW, alpha=0.9, solid_capstyle="round", zorder=2)      # path under the dots
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=yn[:t, k], s=TRAJ_DOT_SIZE, cmap=cmap, vmin=ymin, vmax=ymax, edgecolors="0.2", linewidths=0.15, zorder=3)
                ax.plot(*xy[-1], marker="o", markersize=6, markerfacecolor=col, markeredgecolor="k", zorder=5)
                ax.plot(*start[k], marker="s", markersize=6, markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.8, zorder=4)
            w, mu, cov, mask = log["gmm_w"][t - 1], log["gmm_mu"][t - 1], log["gmm_cov"][t - 1], log["gmm_mask"][t - 1]
            for j in range(len(w)):
                if mask[j]:
                    ax.add_patch(_ellipse_patch(mu[j], cov[j], float(w[j]), "tab:red"))
            mp = np.asarray(log["map_xy"][t - 1], dtype=float)
            ax.plot(*mp, marker="x", color="tab:red", markersize=8, markeredgewidth=2, zorder=6)
            ax.plot(*truth, marker="*", color="gold", markeredgecolor="k", markersize=14, zorder=7)
            ax.set_xlim(win[0], win[2])
            ax.set_ylim(win[1], win[3])
            ax.set_aspect("equal")
            if not (win[0] <= mp[0] <= win[2] and win[1] <= mp[1] <= win[3]):
                _arrow_to_outside(ax, win, mp)
            ax.tick_params(labelsize=7)
            fr_txt = f", frame {int(r['frames'][t - 1])}" if str(r["meta"].get("mode") or mode_hint) == "T2" else ""
            ax.set_title(f"step {t}/{T}{fr_txt}: MAP error {float(log['map_error'][t - 1]):.0f} m, sigma {float(log['top_sigma'][t - 1]):.0f} m", fontsize=7.5)
        ok = "success" if r["meta"].get("success") else "no success"
        axes[ri][0].set_ylabel(f"{r['role']} policy: {r['stem']}, {fmt_steps(r['env_steps'])} team steps\n({ok}, final error {float(log['map_error'][-1]):.0f} m)\ny [m]",
                               fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("x [m]", fontsize=8)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, aspect=30, pad=0.01)
        pos = [(float(det.normalise(c)), str(c)) for c in TRAJ_COUNT_TICKS if ymin <= float(det.normalise(c)) <= ymax]
        for v in (ymin, ymax):                              # label the ends of the scale too unless a round tick is close
            if all(abs(v - p) > 0.05 for p, _l in pos):
                pos.append((v, f"{np.expm1(v * np.log1p(det.y_max)):.0f}"))
        pos.sort()
        cb.set_ticks([p for p, _l in pos])
        cb.set_ticklabels([lab for _p, lab in pos])
        cb.set_label("measured count y per step (log colour scale)", fontsize=8)
    handles = [Line2D([], [], marker="*", color="gold", markeredgecolor="k", ls="", markersize=11, label="true source"),
               Line2D([], [], marker="x", color="tab:red", ls="", markeredgewidth=2, label="MAP estimate"),
               Line2D([], [], color="tab:red", lw=1.5, label="2-sigma GMM components"),
               Line2D([], [], color="0.25", lw=0.8, label="building outline"),
               Line2D([], [], marker="o", color=cmap(0.6), markeredgecolor="0.15", markeredgewidth=0.4, markersize=4, ls="", label="measurement (colour = count)")]
    for k in range(D_max):
        col = DRONE_COLORS[k % len(DRONE_COLORS)]
        handles.append(Line2D([], [], color=col, lw=TRAJ_PATH_LW, marker="o", markerfacecolor=col, markeredgecolor="k", label=f"drone {k + 1}: path, position"))
        handles.append(Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.8, ls="", label=f"drone {k + 1}: start"))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 5), fontsize=8, frameon=False)
    meta0 = ref["meta"]
    mode = str(meta0.get("mode") or mode_hint)
    fr0 = int(meta0.get("frame", ref["frames"][0]))
    frame_txt = f"time-varying truth from frame {fr0} (Mode T2)" if mode == "T2" else f"frozen LDM frame {fr0} (Mode F)"
    solved = bool(ref["meta"].get("success", False))
    pick = (f"episode {eid} was requested" if episode_id is not None else
            ("the episode the late checkpoint solved, then the smallest final MAP error, then the lowest id" if solved else
             "no candidate was solved by the late checkpoint: the smallest final MAP error, then the lowest id"))
    title = (f"{run.name}: sample episode {eid} (source {src}, scale {float(meta0.get('scale', float('nan'))):.2f}, {frame_txt}); the same scenario is replayed "
             f"at {len(rows)} checkpoints\n"
             + "; ".join(f"{r['role']} = {r['stem']}" for r in rows)
             + " (initial = first periodic checkpoint with step logs, not the untrained policy; middle = interior checkpoint closest to the midpoint; "
             "late = last checkpoint)\n"
             "episode rule: among the episodes logged for all checkpoints" + (f" of source {source}" if source is not None else "")
             + f", {pick}; grey = 15 m slab density of the source (log, {TRAJ_DENSITY_DECADES:.0f} decades)")
    fig.suptitle(_wrap(title, fig.get_figwidth(), 8), fontsize=8)
    path = out_dir / f"{prefix}_{run.name}_trajectories.png"
    _save(fig, path, dpi)
    return {"png": path, "reason": f"episode {eid} (source {src}) at checkpoints " + ", ".join(f"{r['role']}={r['stem']}" for r in rows),
            "episode_id": eid, "source": src, "checkpoints": [r["stem"] for r in rows], "steps": steps_of, "window": win}


# ------------------------------------------------------------------------------------------------------------ item 8: 1 drone versus 2 drones
def _normalise_compare(compare) -> list[list[Path]]:
    groups = []
    for g in compare:
        names = [s for s in g.split(",") if s] if isinstance(g, str) else list(g)
        if names:
            groups.append([resolve_run(n) for n in names])
    return groups


def _group_label(runs: list[Path]) -> str:
    return f"{_drones_txt(run_n_drones(runs[0]))} ({', '.join(r.name for r in runs)})"


def fig_compare(groups: list[list[Path]], prefix: str, out_dir: Path, dpi: int | None = None) -> dict:
    """Item 8: success_ma and return_ma of run groups with different drone counts on the same x axis; top row = team steps, bottom row = agent
    transitions (team steps x drones, what a parameter-shared learner actually sees).  The CSV (long format, one block per group) has the mean, the sd
    (ddof = 1), the number of seeds and one column per run."""
    if len(groups) < 2:
        raise Skip("need at least two run groups with different drone counts to compare")
    glogs = []
    for g in groups:
        miss = [r.name for r in g if not has_log_rows(r)]
        if miss:
            raise Skip("train_log.csv missing or empty for " + ", ".join(miss))
        logs = []
        for r in g:
            lg = read_log(r)
            lg["x_agent"] = lg["env_steps"] * run_n_drones(r)
            logs.append(lg)
        glogs.append((g, run_n_drones(g[0]), logs))
    all_runs = [r for g, _n, _l in glogs for r in g]
    uniq = _unique_names(all_runs)
    w = _ma_label(all_runs)
    div = {xk: _steps_unit(max(min(float(lg[xk][-1]) for lg in logs) for _g, _n, logs in glogs)) for xk in ("env_steps", "x_agent")}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    fields = ["group", "n_drones", "env_steps", "agent_transitions", "n_seeds", "success_ma_mean", "success_ma_sd", "return_ma_mean", "return_ma_sd"]
    fields += [f"{k}_seed_{nm}" for nm in uniq for k in ("success_ma", "return_ma")]
    table_rows: list[dict] = []
    first = 0
    for gi, (g, n, logs) in enumerate(glogs):
        col = COLORS[gi % len(COLORS)]
        lab = _group_label(g)
        grids = {xk: np.linspace(0.0, min(float(lg[xk][-1]) for lg in logs), GRID_POINTS) for xk in ("env_steps", "x_agent")}
        stats = {}
        for row, xk, xname in ((0, "env_steps", "team steps"), (1, "x_agent", "agent transitions = team steps x drones")):
            xdiv, unit = div[xk]
            for cidx, (key, ylab) in enumerate((("success_ma", f"success rate (primary, {w}-episode mean)"), ("return_ma", "team return per episode"))):
                ys = _band(axes[row][cidx], logs, key, ylab, grids[xk], color=col, xkey=xk, mean_label=lab, xlabel=f"{xname} [{unit}]", xdiv=xdiv)
                if xk == "env_steps":
                    stats[key] = ys
        g_env = grids["env_steps"]
        mean_s, sd_s = _nanmean0(stats["success_ma"]), _nanstd0(stats["success_ma"])
        mean_r, sd_r = _nanmean0(stats["return_ma"]), _nanstd0(stats["return_ma"])
        for i in range(g_env.size):
            rowd = {"group": lab, "n_drones": n, "env_steps": g_env[i], "agent_transitions": g_env[i] * n, "n_seeds": len(logs),
                    "success_ma_mean": mean_s[i], "success_ma_sd": sd_s[i], "return_ma_mean": mean_r[i], "return_ma_sd": sd_r[i]}
            for si in range(len(g)):
                rowd[f"success_ma_seed_{uniq[first + si]}"] = stats["success_ma"][si][i]
                rowd[f"return_ma_seed_{uniq[first + si]}"] = stats["return_ma"][si][i]
            table_rows.append(rowd)
        first += len(g)
    for ax in axes.flat:
        ax.legend(fontsize=7, loc="best")
    for ax, t in zip(axes.flat, ("success rate vs team steps", "team return vs team steps", "success rate vs agent transitions",
                                 "team return vs agent transitions")):
        ax.set_title(t, fontsize=9)
    fig.suptitle(f"{prefix}: runs with different drone counts (thin lines = seeds, thick = mean of seeds, band = mean +- sd, ddof=1); "
                 f"one team step = one transition per drone", fontsize=10)
    path = out_dir / f"{prefix}_compare.png"
    _save(fig, path, dpi)
    with (out_dir / f"{prefix}_compare.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, restval="")
        wr.writeheader()
        wr.writerows(table_rows)
    return {"png": path, "csv": out_dir / f"{prefix}_compare.csv", "reason": ""}


# ------------------------------------------------------------------------------------------------------------ one call for everything
def _ok(path: Path | str, reason: str = "") -> dict:
    return {"path": str(path), "status": "ok", "reason": reason}


def _skipped(reason: str) -> dict:
    return {"path": None, "status": "skipped", "reason": reason}


def validate_only(only: Sequence[str] | str, run_names: Sequence[str] = ()) -> list[str]:
    """The ``only`` list of make_all / --only as a list; ValueError naming every unknown output (with the closest valid name) if one is misspelled.
    Valid: OUTPUT_NAMES, and trajectories_<run name> for one run."""
    names = [only] if isinstance(only, str) else list(only)
    valid = list(OUTPUT_NAMES) + [f"trajectories_{n}" for n in run_names]
    bad = [o for o in names if o not in valid]
    if bad:
        parts = []
        for o in bad:
            close = difflib.get_close_matches(o, valid, n=1, cutoff=0.5)
            parts.append(f"{o!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(f"unknown output name(s) in only / --only: {', '.join(parts)}; valid names: {', '.join(OUTPUT_NAMES)}"
                         + "".join(f", trajectories_{n}" for n in run_names) + " (trajectories = every run)")
    return names


def _drone_groups_text(by_n: dict[int, list[str]]) -> str:
    return "; ".join(f"{_drones_txt(n)}: {', '.join(v)}" for n, v in sorted(by_n.items()))


def make_all(runs: Sequence[str | Path], prefix: str, out_dir: str | Path, baseline_tag: str | Path | None = None, n_drones: int | None = None,
             compare: Sequence | None = None, traj_source: int | None = None, traj_episode: int | None = None,
             only: Sequence[str] | None = None, dpi: int | None = None) -> dict[str, dict]:
    """Every training-result visual of spec section 7 for ``runs`` (names under config.TRAIN_ROOT or paths) into ``out_dir``; returns the manifest
    {output name: {"path": str | None, "status": "ok" | "skipped", "reason": str}} and writes <prefix>_manifest.json.  ``baseline_tag``
    (an evaluation tag or directory) adds the horizontal baseline lines of item 1 (+ the new key baselines_csv with their Wilson 95 % CI); the baselines
    use the drone count of the runs, or ``n_drones`` if given (a UserWarning and a manifest reason when it disagrees with the runs).  Runs with different
    drone counts form separate groups in items 1, 2 and 6 (never one pooled band).  ``compare`` = groups of runs for item 8 (lists of names, or "a,b"
    strings), default: the runs grouped by drone count if there are two or more counts.  ``only`` restricts the call to the named outputs
    (curves, episode_stats, diagnostics, source_heatmap, ckpt_eval, reward_decomposition, trajectories, compare_drones; ValueError on any other name;
    the other outputs are then absent from the manifest).  ``dpi`` = figure resolution (default FIG_DPI 150; 300 for final figures).  A missing input
    skips one output (with the reason); a nonexistent run directory or a bug raises."""
    run_paths = [resolve_run(r) for r in runs]
    if only is not None:
        only = validate_only(only, [r.name for r in run_paths])
    if dpi is not None and int(dpi) <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man: dict[str, dict] = {}

    def attempt(name: str, fn: Callable[[], dict | None]) -> None:
        if only is not None and not any(name == o or name.startswith(o + "_") for o in only):
            return
        try:
            res = fn() or {}
        except Skip as exc:
            man[name] = _skipped(str(exc))
            return
        man[name] = _ok(res["png"], res.get("reason", ""))
        if res.get("csv") is not None:
            man[f"{name}_csv"] = _ok(res["csv"], f"numbers behind {name}")
        for key, (p, why) in (res.get("extra") or {}).items():
            man[key] = _ok(p, why)

    with_log = [r for r in run_paths if has_log_rows(r)]
    skip_note = (" (runs without train_log.csv rows skipped: " + ", ".join(r.name for r in run_paths if r not in with_log) + ")") \
        if len(with_log) < len(run_paths) else ""

    def need_logs() -> list[Path]:
        if not with_log:
            raise Skip("no train_log.csv with rows in " + ", ".join(r.name for r in run_paths))
        return with_log

    def curves() -> dict:
        rs = need_logs()
        modes = {run_mode(r) for r in rs}
        mode_names = sorted({m for m, _h in modes})
        same_mode = len(mode_names) == 1
        horizon = sorted(modes)[0][1] if same_mode else None
        by_n: dict[int, list[str]] = {}
        for r in rs:
            by_n.setdefault(run_n_drones(r), []).append(r.name)
        notes: list[str] = []
        if len(by_n) > 1:
            notes.append(f"runs with different drone counts are drawn as separate groups, not pooled ({_drone_groups_text(by_n)})")
        if not same_mode:
            notes.append(f"runs of different truth modes ({', '.join(mode_names)}) share the axes")
        want = sorted(by_n) if n_drones is None else [int(n_drones)]
        if n_drones is not None and any(n != int(n_drones) for n in by_n):
            msg = (f"--n-drones {n_drones} disagrees with the drone count of the runs ({_drone_groups_text(by_n)}); the baselines of "
                   f"{n_drones} drone(s) are drawn as requested (omit n_drones to take the runs' drone count)")
            warnings.warn(msg, UserWarning, stacklevel=3)
            notes.append(msg)
        sets: list[dict] = []
        if baseline_tag is None:
            notes.append("no baseline tag given: no baseline lines")
        elif not same_mode:
            notes.append("baseline lines skipped: the runs have different truth modes")
        else:
            for nd in want:
                info = baseline_info(baseline_tag, nd, mode_names[0], horizon)
                notes.append(info["note"])
                if info["rates"]:
                    sets.append(info)
        hs = sorted({b["horizon"] for b in sets if b["horizon"]})
        res = fig_curves(rs, prefix, out_dir, horizon=horizon, base_note=("records horizon " + "/".join(str(h) for h in hs) + " steps") if hs else "",
                         baselines=sets, dpi=dpi)
        extra = {}
        if res.get("baselines_csv"):
            extra["baselines_csv"] = (res["baselines_csv"], "baseline success rates on the training sources: k, n, rate, Wilson 95 % CI (the lines of curves)")
        return {"png": out_dir / f"{prefix}_curves.png", "csv": out_dir / f"{prefix}_curves.csv", "reason": ("; ".join(notes) + skip_note).strip(),
                "extra": extra}

    def diagnostics() -> dict:
        fig_diagnostics(need_logs(), prefix, out_dir, dpi=dpi)
        return {"png": out_dir / f"{prefix}_diagnostics.png", "reason": skip_note.strip()}

    def ckpt() -> dict:
        if not fig_ckpt_eval(run_paths, prefix, out_dir, dpi=dpi):
            raise Skip("no ckpt_eval/*.csv (quick-evaluation records) in " + ", ".join(r.name for r in run_paths))
        return {"png": out_dir / f"{prefix}_ckpt_eval.png", "csv": out_dir / f"{prefix}_ckpt_eval.csv"}

    attempt("curves", curves)
    attempt("episode_stats", lambda: fig_episode_stats(run_paths, prefix, out_dir, dpi=dpi))
    attempt("diagnostics", diagnostics)
    attempt("source_heatmap", lambda: fig_source_heatmap(run_paths, prefix, out_dir, dpi=dpi))
    attempt("ckpt_eval", ckpt)
    attempt("reward_decomposition", lambda: fig_reward_decomposition(run_paths, prefix, out_dir, dpi=dpi))
    scene: dict = {}
    for run in run_paths:
        def traj(run=run) -> dict:
            if "om" not in scene and ckpt_step_logs(run):
                scene["om"], scene["backend"] = load_scene_helpers()
            return fig_trajectories(run, prefix, out_dir, scene.get("om"), scene.get("backend"), traj_source, traj_episode, dpi=dpi)
        attempt(f"trajectories_{run.name}", traj)

    def comp() -> dict:
        if compare is not None:
            groups, note = _normalise_compare(compare), ""
        else:
            by_n: dict[int, list[Path]] = {}
            for r in run_paths:
                by_n.setdefault(run_n_drones(r), []).append(r)
            if len(by_n) < 2:
                raise Skip("no comparison groups: pass compare=[[1-drone runs], [2-drone runs]] (--compare a,b c,d) or runs with different drone counts")
            groups, note = [by_n[n] for n in sorted(by_n)], "groups built from the drone counts of the runs"
        res = fig_compare(groups, prefix, out_dir, dpi=dpi)
        res["reason"] = note
        return res

    attempt("compare_drones", comp)
    mpath = out_dir / f"{prefix}_manifest.json"
    man["manifest"] = _ok(mpath, "this file")
    mpath.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    return man


def print_manifest(man: dict[str, dict]) -> None:
    for name, e in man.items():
        tail = e["path"] if e["status"] == "ok" else e["reason"]
        extra = f"  [{e['reason']}]" if e["status"] == "ok" and e["reason"] else ""
        print(f"[fig_training] {name:<28s} {e['status']:<8s} {tail}{extra}")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run names under config.TRAIN_ROOT (or paths)")
    ap.add_argument("--prefix", default="fig6_m1")
    ap.add_argument("--n-drones", type=int, default=None,
                    help="drone count of the baseline records drawn as horizontal lines (default: the drone count of the runs, from their config.json; "
                         "a warning is issued if this value disagrees with the runs)")
    ap.add_argument("--baseline-tag", default="t2_4_v3", help="evaluation tag (or directory) of the baseline lines; must be of the runs' truth mode")
    ap.add_argument("--out-dir", type=Path, default=config.FIG_DIR)
    ap.add_argument("--dpi", type=int, default=FIG_DPI, help=f"figure resolution (default {FIG_DPI}; use 300 for the final figures)")
    ap.add_argument("--compare", nargs="+", default=None, metavar="RUN[,RUN...]",
                    help="run groups of different drone counts for the 1-versus-n comparison, e.g. m1_s1,m1_s2 m2_s1,m2_s2 (default: group --runs by drone count)")
    ap.add_argument("--traj-source", type=int, default=None, help="source of the sample-trajectory episode (default: an episode the late checkpoint solved)")
    ap.add_argument("--traj-episode", type=int, default=None, help="episode id of the sample-trajectory episode")
    ap.add_argument("--only", nargs="+", default=None, help="draw only these outputs: " + " ".join(OUTPUT_NAMES) + " (trajectories_<run> for one run)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    runs = [resolve_run(r) for r in args.runs]
    if args.only is not None:
        try:
            validate_only(args.only, [r.name for r in runs])
        except ValueError as exc:
            ap.error(str(exc))
    if args.dpi <= 0:
        ap.error(f"--dpi must be positive, got {args.dpi}")
    man = make_all(runs, args.prefix, args.out_dir, args.baseline_tag, args.n_drones, args.compare, args.traj_source, args.traj_episode, args.only,
                   dpi=args.dpi)
    print_manifest(man)
    logs = [read_log(r) for r in runs if has_log_rows(r)]
    res: dict = {"manifest": man, "final_success_ma": [float(lg["success_ma"][-1]) for lg in logs],
                 "final_return_ma": [float(lg["return_ma"][-1]) for lg in logs], "ckpt_eval": [row for r in runs for row in read_ckpt_eval(r)]}
    print(f"[fig_training] wrote {args.prefix}_* to {args.out_dir}")
    return res


if __name__ == "__main__":
    main()
