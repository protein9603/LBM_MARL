"""D6-1 / plan S1 T1-4 (gate G1): the decisive real-data test of the RB-PF forward models on the LDM slab truth.

Usage: python -m srcloc_env.scripts.validate_t1_4 [--seed 0] [--n-seeds 5] [--n-steps 150] [--sources 109 102]
                                                  [--filters A B] [--out ...] [--fig ...]
Writes config.CACHE_DIR / validate_t1_4.json, config.CACHE_DIR / t1_4_snapshots_{src}.npz (T1-5 / figure 5 input)
and config.FIG_DIR / fig_t1_4_errors.png (config.FIG_DPI_FINAL) + _preview.png (config.FIG_DPI_PREVIEW).

Truth and measurements (plan T1-4, exactly as validate_library_filter.run_filter): for each of the 13 sources the
truth is the cached LDM slab at frame config.T1_4_FRAME_INDEX (599 = step 30000), z = config.DRONE_Z, sensor scale
config.T1_4_SENSOR_SCALE; two drones fly validate_pf_adjoint.two_drone_paths (config.T1_4_N_STEPS = 150 RL steps,
config.DRONE_STEP_M, adjacent config.PF_ADJ_SWEEP_WIDTH_M bands, start config.PF_ADJ_START_DOWNWIND_M downwind,
drone-free cells only) and the counts y ~ Poisson((k0 scale n_s(p) + b) T) are drawn ONCE per (source, seed) with
Detector.measure and rng [seed, source, 1] (R3, plan 4.1), then the SAME sequence is fed to every filter.

Filters (all RBPF, N = config.PF_N_PARTICLES, grid mode, kappa grid from config, obstacles = ObstacleMap.load(),
PF rng [seed, 2] per repeat; plan 4.3):
    A   GaussianPlume(global, ForwardParams(U = config.T1_4_ANALYTIC_U, sigma_v = config.T1_4_ANALYTIC_SIGMA_V)),
        eps_mix = config.PF_EPS_MIX (the T1-3 chosen combination, calibrate_forward.json)
    A2  the same with eps_mix = config.PF_EPS_MIX_TRAPPED (plan S1 보강)
    B   LbmAdjointModel(AdvectionDiffusionOperator.from_data(AdjointParams(K = config.T1_4_ADJOINT_K,
        lam = config.T1_4_ADJOINT_LAM), WindField.load(), ObstacleMap.load())) factorised once and shared,
        eps_mix = config.PF_EPS_MIX (the T1-3b chosen combination, calibrate_adjoint.json; plan 4.2b)
    B2  the same with eps_mix = config.PF_EPS_MIX_TRAPPED
    C   the library CandidateFilter upper bound is NOT re-run: its per-source first step with P(true) > 0.9 and
        selection rate are read from validate_library_filter.json (plan S1 라이브러리 필터 상한).

Per run: MAP error (RBPF.map_estimate, weighted mode; config.SUCCESS_* semantics) after config.T1_4_CHECKPOINT_STEPS
steps and at every step, final weighted-mean error, final GMM top sigma (gmm_summary.summarise_pf().top_sigma()),
success = (top sigma < config.SUCCESS_SIGMA_M and MAP error < config.SUCCESS_ERROR_M) at ANY step (checked every
config.T1_4_GMM_EVERY steps, plan 4.5) and the first success step, entropy at config.T1_4_ENTROPY_STEPS, kappa
posterior median, number of resamples, max expected count along the path (truth field) and wall time.
Belief snapshots (xy, logw float32 every T1_4_GMM_EVERY steps, seed 0, filter T1_4_SNAPSHOT_FILTER) are saved for
config.T1_4_SNAPSHOT_SOURCES (T1-5 / figure 5).

Aggregates (``aggregate``): per source and filter the median / p90 of the MAP error at each checkpoint over seeds,
the success rate and the median first-success step.  Verdicts (``verdicts``, plan G1): (i) filter A open sources
(config.T1_3_OPEN_SOURCES) median final MAP error < config.T1_4_FINAL_ERROR_PASS_M per source and overall; (ii) the
same for B; (iii) the number of sources (of 13) where B beats A on the median final error; (iv) the trapped /
roof-leak sources config.T1_4_EPS_REPORT_SOURCES with eps 0.05 vs 0.1; (v) source config.T1_4_UNOBSERVABLE_SOURCE
marked 'unobservable at 15 m' iff its max expected count <= max(config.T1_4_UNOBSERVABLE_COUNT_FACTOR x background,
Currie decision threshold Detector.detection_threshold_cps() x T when config.T1_4_UNOBSERVABLE_USE_CURRIE; the
rule of validate_pf_adjoint / validation_log 결정 '고정 고도 15 m의 관측 한계').
``table1_markdown`` builds the Table 1 string (source | type | analytic / adjoint std_dense from the calibration
JSONs | A / B final error med/p90 | A / B success rate | library first step | max count).

Figure fig_t1_4_errors.png: (left) per-source grouped bars of the median final MAP error for A and B with p90
whiskers, 30 m / 20 m reference lines, sources coloured open / trapped / holdout / train, the unobservable source
hatched; (right) MAP error vs RL step (median over seeds and open sources) for the four filters.
References: R3 (Poisson likelihood), R5 (RB-PF), R7 (GMM summary), R13 (Gaussian plume), R17 / R18 (adjoint).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.field.wind import WindField
from srcloc_env.pf.forward_model import ForwardParams, GaussianPlume
from srcloc_env.pf.gmm_summary import summarise_pf
from srcloc_env.pf.lbm_adjoint import AdjointParams, AdvectionDiffusionOperator, LbmAdjointModel
from srcloc_env.pf.particle_filter import RBPF
from srcloc_env.scripts.validate_pf_adjoint import two_drone_paths
from srcloc_env.sensor.detector import Detector

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Patch   # noqa: E402

FILTERS = ("A", "A2", "B", "B2")
FILTER_MODEL = {"A": "analytic", "A2": "analytic", "B": "adjoint", "B2": "adjoint"}
FILTER_LABEL = {"A": "A analytic plume (eps 0.05)", "A2": "A2 analytic plume (eps 0.1)",
                "B": "B LBM adjoint (eps 0.05)", "B2": "B2 LBM adjoint (eps 0.1)"}
# figure colours (dataviz palette slots 1 / 2 / 3 + neutral; identity = source type, A / B = tint + legend + hatch)
TYPE_COLORS = {"open": "#2a78d6", "trapped": "#eb6834", "holdout": "#1baf7a", "train": "#8a8a8a"}
FILTER_COLORS = {"A": "#2a78d6", "A2": "#2a78d6", "B": "#eb6834", "B2": "#eb6834"}
FILTER_STYLES = {"A": "-", "A2": "--", "B": "-", "B2": "--"}
INK_MUTED = "#8a8a8a"
UNOBSERVABLE_HATCH = "///"


# ---------------------------------------------------------------------------------------- source typing
def source_type(s: int) -> str:
    """'trapped' (config.T1_3_TRAPPED_SOURCES), 'open' / 'open(holdout)' (T1_3_OPEN_SOURCES), 'holdout'
    (HOLDOUT_SOURCES) or 'train' (plan 1 / T1-3 source classes)."""
    if s in config.T1_3_TRAPPED_SOURCES:
        return "trapped"
    if s in config.T1_3_OPEN_SOURCES:
        return "open(holdout)" if s in config.HOLDOUT_SOURCES else "open"
    if s in config.HOLDOUT_SOURCES:
        return "holdout"
    return "train"


def colour_group(s: int) -> str:
    """Figure colour class: open sources stay 'open' even when held out (109)."""
    t = source_type(s)
    return "open" if t.startswith("open") else t


# ---------------------------------------------------------------------------------------- measurements
def generate_measurements(backend: LdmSlabBackend, det: Detector, source: int, paths: np.ndarray, seed: int,
                          frame_index: int = config.T1_4_FRAME_INDEX, z: float = config.DRONE_Z,
                          scale: float = config.T1_4_SENSOR_SCALE) -> tuple[np.ndarray, np.ndarray]:
    """(counts, expected) both (n_steps, n_drones): slab-truth Poisson counts along paths (n_steps, n_drones, 2),
    drawn with the rng stream [seed, source, 1] of validate_library_filter.run_filter (identical sequences)."""
    n_steps, n_drones = paths.shape[:2]
    flat = paths.reshape(-1, 2)
    dens = backend.density([source], flat, frame_index, z, 1.0)
    rng = np.random.default_rng([int(seed), int(source), 1])
    counts = det.measure(dens, scale, rng).reshape(n_steps, n_drones)
    expected = det.expected_counts(dens, scale).reshape(n_steps, n_drones)
    return counts, expected


# ---------------------------------------------------------------------------------------- one run
def run_filter(pf: RBPF, counts: np.ndarray, paths: np.ndarray, true_xy: Sequence[float],
               checkpoints: tuple[int, ...] = config.T1_4_CHECKPOINT_STEPS,
               entropy_steps: tuple[int, ...] = config.T1_4_ENTROPY_STEPS, gmm_every: int = config.T1_4_GMM_EVERY,
               snapshot: bool = False, z: float = config.DRONE_Z) -> tuple[dict, dict | None]:
    """Feed counts (n_steps, n_drones) sequentially (plan 4.3 fusion) to ``pf`` and record the T1-4 statistics.

    Returns (record, snapshots): record holds the per-step MAP error trajectory, the checkpoint errors, the final
    weighted-mean error, the final GMM top sigma, success / first success step (checked every gmm_every steps with
    the config.SUCCESS_* criteria), entropy at entropy_steps, the kappa posterior median, n resamples and the wall
    time; snapshots (when requested) hold xy / logw (float32), drone_xy, map_xy at step 0 and every gmm_every steps.
    """
    n_steps, n_drones = counts.shape
    true_xy = np.asarray(true_xy, dtype=np.float64)
    err = np.empty(n_steps)
    ent: dict[str, float] = {}
    if 0 in entropy_steps:
        ent["0"] = pf.entropy_xy()
    gmm_rng = np.random.default_rng(config.T1_4_GMM_SEED)
    first_success: int | None = None
    n_success_checks = 0
    n_checks = 0
    top_sigma = float("nan")
    snap: dict | None = None
    if snapshot:
        snap = {"steps": [0], "xy": [pf.xy.astype(np.float32)], "logw": [pf.logw.astype(np.float32)],
                "drone_xy": [paths[0].astype(np.float64)], "map_xy": [pf.map_estimate()]}
    t0 = time.perf_counter()
    for k in range(n_steps):
        step = k + 1
        for d in range(n_drones):
            pf.update(int(counts[k, d]), np.array([paths[k, d, 0], paths[k, d, 1], z]))
        map_xy = pf.map_estimate()
        err[k] = float(np.hypot(*(map_xy - true_xy)))
        if step in entropy_steps:
            ent[str(step)] = pf.entropy_xy()
        if step % gmm_every == 0 or step == n_steps:
            gmm = summarise_pf(pf, rng=gmm_rng)
            top_sigma = gmm.top_sigma()
            n_checks += 1
            ok = bool(top_sigma < config.SUCCESS_SIGMA_M and err[k] < config.SUCCESS_ERROR_M)
            n_success_checks += int(ok)
            if ok and first_success is None:
                first_success = step
            if snap is not None:
                snap["steps"].append(step)
                snap["xy"].append(pf.xy.astype(np.float32))
                snap["logw"].append(pf.logw.astype(np.float32))
                snap["drone_xy"].append(paths[k].astype(np.float64))
                snap["map_xy"].append(map_xy)
    wall = time.perf_counter() - t0
    record = {
        "n_steps": int(n_steps), "n_drones": int(n_drones), "wall_seconds": float(wall),
        "map_error_at": {str(c): float(err[c - 1]) for c in checkpoints if c <= n_steps},
        "map_error_trajectory_m": err.tolist(),
        "final_map_error_m": float(err[-1]),
        "final_mean_error_m": float(np.hypot(*(pf.mean() - true_xy))),
        "final_top_sigma_m": float(top_sigma),
        "success": bool(first_success is not None), "first_success_step": first_success,
        "n_success_checks": int(n_success_checks), "n_checks": int(n_checks),
        "min_map_error_m": float(err.min()),
        "entropy_at": ent,
        "kappa_median": float(pf.posterior_kappa_quantiles((0.5,))[0]),
        "kappa_q05_q95": pf.posterior_kappa_quantiles((0.05, 0.95)).tolist(),
        "kappa_edge_mass": list(pf.kappa_edge_mass()),
        "n_resamples": int(pf.n_resamples), "neff_final": float(pf.neff()),
        "posterior_spread_m": float(np.sqrt(np.trace(pf.covariance()))),
    }
    if snap is not None:
        snap = {"steps": np.asarray(snap["steps"], dtype=np.int64), "xy": np.stack(snap["xy"]),
                "logw": np.stack(snap["logw"]), "drone_xy": np.stack(snap["drone_xy"]),
                "true_xy": true_xy, "map_xy": np.stack(snap["map_xy"])}
    return record, snap

# ---------------------------------------------------------------------------------------- aggregation
def _median_or_none(vals: list) -> float | None:
    v = [float(x) for x in vals if x is not None]
    return float(np.median(v)) if v else None


def aggregate(runs: dict[str, dict[int, list[dict]]],
              checkpoints: tuple[int, ...] = config.T1_4_CHECKPOINT_STEPS) -> dict[str, dict[str, dict]]:
    """Per filter and source (string keys): median / p90 of the MAP error at each checkpoint over the seeds, the
    final-error median / p90, the success rate, the median first-success step (successful seeds only, None if
    none), the median final top sigma / mean error / kappa median and the per-seed lists (plan T1-4 aggregates)."""
    out: dict[str, dict[str, dict]] = {}
    for f, by_src in runs.items():
        out[f] = {}
        for s, rr in by_src.items():
            if not rr:
                continue
            errs = {str(c): [r["map_error_at"][str(c)] for r in rr if str(c) in r["map_error_at"]] for c in checkpoints}
            fin = [r["final_map_error_m"] for r in rr]
            out[f][str(s)] = {
                "n_seeds": len(rr),
                "map_error_median": {c: float(np.median(v)) for c, v in errs.items() if v},
                "map_error_p90": {c: float(np.percentile(v, 90)) for c, v in errs.items() if v},
                "final_error_median": float(np.median(fin)), "final_error_p90": float(np.percentile(fin, 90)),
                "final_error_values": [float(x) for x in fin],
                "mean_error_median": float(np.median([r["final_mean_error_m"] for r in rr])),
                "top_sigma_median": float(np.median([r["final_top_sigma_m"] for r in rr])),
                "success_rate": float(np.mean([bool(r["success"]) for r in rr])),
                "n_success": int(sum(bool(r["success"]) for r in rr)),
                "first_success_step_median": _median_or_none([r["first_success_step"] for r in rr]),
                "first_success_steps": [r["first_success_step"] for r in rr],
                "kappa_median_median": float(np.median([r["kappa_median"] for r in rr])),
                "n_resamples_median": float(np.median([r["n_resamples"] for r in rr])),
                "wall_seconds_median": float(np.median([r["wall_seconds"] for r in rr])),
                "entropy_median": {k: float(np.median([r["entropy_at"][k] for r in rr if k in r["entropy_at"]]))
                                   for k in rr[0]["entropy_at"]},
            }
    return out


def pooled_error_curve(runs_by_src: dict[int, list[dict]], sources: Sequence[int]) -> np.ndarray | None:
    """(n_steps,) median over all (source in sources, seed) runs of the MAP error trajectory; None if empty."""
    rows = [np.asarray(r["map_error_trajectory_m"]) for s in sources for r in runs_by_src.get(s, [])]
    if not rows:
        return None
    return np.median(np.stack(rows), axis=0)


def verdicts(agg: dict[str, dict[str, dict]], max_counts: dict[int, float], sources: Sequence[int],
             open_sources: Sequence[int] = config.T1_3_OPEN_SOURCES,
             pass_m: float = config.T1_4_FINAL_ERROR_PASS_M,
             eps_sources: Sequence[int] = config.T1_4_EPS_REPORT_SOURCES,
             unobservable_source: int = config.T1_4_UNOBSERVABLE_SOURCE,
             background_counts: float = config.SENSOR_BACKGROUND_CPS * config.SENSOR_T,
             unobservable_factor: float = config.T1_4_UNOBSERVABLE_COUNT_FACTOR,
             currie_counts: float | None = None) -> dict:
    """Plan G1 verdicts (i)-(v) from the aggregates (see the module docstring).

    (v): the observability threshold is unobservable_factor x background_counts, raised to currie_counts (the Currie
    decision threshold in counts, Detector.detection_threshold_cps() x T) when that is given and larger."""
    def open_block(f: str) -> dict:
        per = {}
        for s in open_sources:
            a = agg.get(f, {}).get(str(s))
            if a is None:
                continue
            per[str(s)] = {"final_error_median_m": a["final_error_median"], "pass": bool(a["final_error_median"] < pass_m)}
        return {"pass_m": pass_m, "per_source": per, "n_pass": int(sum(v["pass"] for v in per.values())),
                "n_sources": len(per), "overall_pass": bool(per) and all(v["pass"] for v in per.values())}

    b_better, comp = [], {}
    for s in sources:
        a, b = agg.get("A", {}).get(str(s)), agg.get("B", {}).get(str(s))
        if a is None or b is None:
            continue
        better = bool(b["final_error_median"] < a["final_error_median"])
        comp[str(s)] = {"A_median_m": a["final_error_median"], "B_median_m": b["final_error_median"], "B_better": better}
        b_better.append(better)
    eps_block = {}
    for s in eps_sources:
        eps_block[str(s)] = {f: {"final_error_median_m": agg[f][str(s)]["final_error_median"],
                                 "final_error_p90_m": agg[f][str(s)]["final_error_p90"],
                                 "success_rate": agg[f][str(s)]["success_rate"],
                                 "first_success_step_median": agg[f][str(s)]["first_success_step_median"]}
                            for f in agg if str(s) in agg[f]}
    mc = max_counts.get(unobservable_source)
    thr_factor = unobservable_factor * background_counts
    thr = thr_factor if currie_counts is None else max(thr_factor, float(currie_counts))
    unobs = None if mc is None else bool(mc <= thr)
    v_block = {"source": unobservable_source, "max_expected_count": mc, "background_counts": background_counts,
               "threshold_counts": thr, "threshold_factor_counts": thr_factor, "currie_threshold_counts": currie_counts,
               "rule": "max(factor x background, Currie)" if currie_counts is not None else "factor x background",
               "below_factor_rule": None if mc is None else bool(mc <= thr_factor),
               "below_currie": None if mc is None or currie_counts is None else bool(mc <= float(currie_counts)),
               "unobservable_at_15m": unobs,
               "label": "unobservable at 15 m" if unobs else ("observable" if unobs is not None else "not run"),
               "results": {f: {"final_error_median_m": agg[f][str(unobservable_source)]["final_error_median"],
                               "success_rate": agg[f][str(unobservable_source)]["success_rate"]}
                           for f in agg if str(unobservable_source) in agg[f]}}
    return {"i_A_open": open_block("A"), "ii_B_open": open_block("B"),
            "iii_B_better_than_A": {"count": int(sum(b_better)), "n_sources": len(b_better), "per_source": comp},
            "iv_eps_trapped": eps_block, "v_unobservable": v_block}


def _fmt(x: float | None, nd: int = 0) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def table1_markdown(agg: dict[str, dict[str, dict]], sources: Sequence[int], analytic_std: dict[int, float],
                    adjoint_std: dict[int, float], library: dict[int, dict], max_counts: dict[int, float],
                    unobservable: Sequence[int] = ()) -> str:
    """Markdown Table 1 (plan T1-4 산출물): source | type | analytic std_dense | adjoint std_dense | A final err
    med/p90 | B final err med/p90 | A success rate | B success rate | library first step (P > 0.9) | max count."""
    head = ("| source | type | analytic std_dense | adjoint std_dense | A final err med/p90 [m] | B final err med/p90 [m] "
            "| A success | B success | library first step P>0.9 (sel.) | max count |")
    sep = "|" + "---|" * 10
    rows = [head, sep]
    for s in sources:
        a, b = agg.get("A", {}).get(str(s)), agg.get("B", {}).get(str(s))
        lib = library.get(s, {})
        t = source_type(s) + (" [unobservable at 15 m]" if s in unobservable else "")
        rows.append(
            f"| {s} | {t} | {_fmt(analytic_std.get(s), 2)} | {_fmt(adjoint_std.get(s), 2)} "
            f"| {_fmt(None if a is None else a['final_error_median'])} / {_fmt(None if a is None else a['final_error_p90'])} "
            f"| {_fmt(None if b is None else b['final_error_median'])} / {_fmt(None if b is None else b['final_error_p90'])} "
            f"| {_fmt(None if a is None else 100 * a['success_rate'])}% | {_fmt(None if b is None else 100 * b['success_rate'])}% "
            f"| {_fmt(lib.get('first_step_median'))} ({_fmt(None if lib.get('selection_rate') is None else 100 * lib['selection_rate'])}%) "
            f"| {_fmt(max_counts.get(s))} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------- figure
def make_figure(agg: dict[str, dict[str, dict]], runs: dict[str, dict[int, list[dict]]], sources: Sequence[int],
                unobservable: Sequence[int], path: Path, open_sources: Sequence[int] = config.T1_3_OPEN_SOURCES) -> None:
    """Left: grouped bars (A, B) of the median final MAP error per source with p90 whiskers; right: pooled
    open-source error-vs-step curves of the four filters."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16.0, 6.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1.0]})
    x = np.arange(len(sources))
    w = 0.38
    for f, off, alpha in (("A", -w / 2, 0.45), ("B", w / 2, 1.0)):
        med = np.array([agg[f][str(s)]["final_error_median"] if str(s) in agg.get(f, {}) else np.nan for s in sources])
        p90 = np.array([agg[f][str(s)]["final_error_p90"] if str(s) in agg.get(f, {}) else np.nan for s in sources])
        cols = [TYPE_COLORS[colour_group(s)] for s in sources]
        hatch = [UNOBSERVABLE_HATCH if s in unobservable else None for s in sources]
        for xi, m, p, c, h in zip(x + off, med, p90, cols, hatch):
            if np.isnan(m):
                continue
            ax.bar(xi, m, width=w - 0.04, color=c, alpha=alpha, edgecolor="white", linewidth=0.8, hatch=h)
            ax.errorbar(xi, m, yerr=[[0.0], [max(p - m, 0.0)]], color="#333333", linewidth=1.0, capsize=3)
    for ref, ls in zip(config.T1_4_FIG_REF_LINES_M, ("--", ":")):
        ax.axhline(ref, color=INK_MUTED, linewidth=1.0, linestyle=ls)
        ax.text(x[-1] + 0.6, ref, f" {ref:.0f} m", color=INK_MUTED, fontsize=8, va="bottom", ha="left")
    ax.set_yscale("log")
    ax.set_ylim(*config.T1_4_FIG_YLIM_M)
    ax.set_xticks(x, [str(s) for s in sources], fontsize=8)
    ax.set_xlabel("true source (LDM slab truth, frame 599, z = 15 m)")
    ax.set_ylabel("final MAP error after 150 RL steps [m] (bar: median over seeds, whisker: p90)")
    ax.set_title("A analytic plume (light) vs B LBM adjoint (solid); hatched = unobservable at 15 m", fontsize=9.5)
    handles = [Patch(facecolor=TYPE_COLORS[k], label=f"{k} source") for k in ("open", "trapped", "holdout", "train")]
    handles += [Patch(facecolor="#555555", alpha=0.45, label="A: analytic plume"), Patch(facecolor="#555555", label="B: LBM adjoint"),
                Patch(facecolor="white", edgecolor="#555555", hatch=UNOBSERVABLE_HATCH, label="unobservable at 15 m")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # right: pooled open-source error curves
    n_steps = 0
    for f in FILTERS:
        if f not in runs:
            continue
        curve = pooled_error_curve(runs[f], open_sources)
        if curve is None:
            continue
        n_steps = curve.size
        steps = np.arange(1, n_steps + 1)
        ax2.plot(steps, curve, color=FILTER_COLORS[f], linestyle=FILTER_STYLES[f], linewidth=2.0, label=FILTER_LABEL[f])
        ax2.text(steps[-1] + 1.5, curve[-1], f, color=FILTER_COLORS[f], fontsize=8, va="center")
    for ref, ls in zip(config.T1_4_FIG_REF_LINES_M, ("--", ":")):
        ax2.axhline(ref, color=INK_MUTED, linewidth=1.0, linestyle=ls)
    ax2.set_yscale("log")
    ax2.set_ylim(*config.T1_4_FIG_YLIM_M)
    ax2.set_xlim(0, n_steps + 12)
    ax2.set_xlabel("RL step (2 drones, 1 update each)")
    ax2.set_ylabel("MAP error [m], median over open sources x seeds")
    ax2.set_title(f"open sources {list(open_sources)}: error vs step", fontsize=9.5)
    ax2.legend(loc="upper right", fontsize=8, frameon=False)
    ax2.grid(True, color="#e6e6e6", linewidth=0.6)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    fig.suptitle(f"T1-4: RB-PF on the LDM slab truth, 2-drone lawnmower, N = {config.PF_N_PARTICLES}, "
                 f"{config.T1_4_N_SEEDS} seeds; analytic U {config.T1_4_ANALYTIC_U} / sigma_v {config.T1_4_ANALYTIC_SIGMA_V}, "
                 f"adjoint K {config.T1_4_ADJOINT_K} / lambda {config.T1_4_ADJOINT_LAM}", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIG_DPI_FINAL)
    fig.savefig(path.with_name(path.stem + "_preview" + path.suffix), dpi=config.FIG_DPI_PREVIEW)
    plt.close(fig)

# ---------------------------------------------------------------------------------------- inputs
def load_calibration_std(path: Path, key: str = "per_source_chosen") -> tuple[dict[int, float], dict]:
    """({source: std_dense}, chosen) from calibrate_forward.json / calibrate_adjoint.json (T1-3 / T1-3b)."""
    if not Path(path).exists():
        return {}, {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(s): float(v["std_dense"]) for s, v in d.get(key, {}).items()}, d.get("chosen", {})


def load_library_summary(path: Path) -> dict[int, dict]:
    """{source: {first_step_median, selection_rate, n_reached_0p9, expected_counts_max}} from
    validate_library_filter.json (block 'library'); {} if the file is missing (plan S1 라이브러리 필터 상한)."""
    if not Path(path).exists():
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    per = d.get("blocks", {}).get("library", {}).get("per_source", {})
    return {int(s): {"first_step_median": v.get("first_step_median"), "selection_rate": v.get("selection_rate"),
                     "n_reached_0p9": v.get("n_reached_0p9"), "expected_counts_max": v.get("expected_counts_max"),
                     "first_step_above_0p9": v.get("first_step_above_0p9")} for s, v in per.items()}


# ---------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=config.T1_4_N_SEEDS)
    ap.add_argument("--n-steps", type=int, default=config.T1_4_N_STEPS)
    ap.add_argument("--sources", type=int, nargs="*", default=list(config.ALL_SOURCES))
    ap.add_argument("--filters", nargs="*", default=list(FILTERS), choices=list(FILTERS))
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_t1_4.json")
    ap.add_argument("--fig", type=Path, default=config.FIG_DIR / "fig_t1_4_errors.png")
    ap.add_argument("--snapshot-dir", type=Path, default=config.CACHE_DIR)
    args = ap.parse_args(argv)
    sources = [int(s) for s in args.sources]
    filters = [f for f in FILTERS if f in args.filters]

    t_start = time.perf_counter()
    om = ObstacleMap.load()
    backend = LdmSlabBackend()
    det = Detector()
    analytic_std, analytic_chosen = load_calibration_std(config.CACHE_DIR / "calibrate_forward.json")
    adjoint_std, adjoint_chosen = load_calibration_std(config.CACHE_DIR / "calibrate_adjoint.json")
    library = load_library_summary(config.CACHE_DIR / "validate_library_filter.json")
    cal_match = {"analytic": bool(analytic_chosen.get("U") == config.T1_4_ANALYTIC_U and analytic_chosen.get("sigma_v") == config.T1_4_ANALYTIC_SIGMA_V
                                  and analytic_chosen.get("wind_mode", "global") == "global"),
                 "adjoint": bool(adjoint_chosen.get("K") == config.T1_4_ADJOINT_K and adjoint_chosen.get("lam") == config.T1_4_ADJOINT_LAM
                                 and adjoint_chosen.get("wind_band") is None)}
    models: dict[str, object] = {}
    setup: dict[str, float] = {}
    if any(FILTER_MODEL[f] == "analytic" for f in filters):
        models["analytic"] = GaussianPlume(ForwardParams(U=config.T1_4_ANALYTIC_U, sigma_v=config.T1_4_ANALYTIC_SIGMA_V), wind_mode="global")
    if any(FILTER_MODEL[f] == "adjoint" for f in filters):
        t0 = time.perf_counter()
        params = AdjointParams(K=config.T1_4_ADJOINT_K, lam=config.T1_4_ADJOINT_LAM)
        op = AdvectionDiffusionOperator.from_data(params, WindField.load(), om).factorize()
        models["adjoint"] = LbmAdjointModel(op)
        setup["adjoint_seconds"] = time.perf_counter() - t0
        setup["adjoint_n_free"] = op.n_free
    print(f"[setup] filters {filters}, sources {sources}, seeds {args.n_seeds}, steps {args.n_steps}; "
          f"calibration match {cal_match}; setup {time.perf_counter() - t_start:.1f} s", flush=True)

    runs: dict[str, dict[int, list[dict]]] = {f: {} for f in filters}
    meas: dict[str, dict] = {}
    max_counts: dict[int, float] = {}
    path_info: dict[str, list] = {}
    bg = det.background * det.T
    for s in sources:
        true_xy = config.SOURCES_XY[s]
        paths, info = two_drone_paths(om, true_xy, args.n_steps)
        path_info[str(s)] = info
        t_src = time.perf_counter()
        for f in filters:
            runs[f][s] = []
        for i in range(args.n_seeds):
            seed = args.seed + i
            counts, expected = generate_measurements(backend, det, s, paths, seed)
            max_counts[s] = float(expected.max())
            meas[f"{s}_{seed}"] = {"source": s, "seed": seed, "counts_sum": int(counts.sum()), "counts_max": int(counts.max()),
                                   "n_detections": int(det.is_detection(counts).sum()), "expected_max": float(expected.max())}
            for f in filters:
                pf = RBPF(models[FILTER_MODEL[f]], n_particles=config.PF_N_PARTICLES, mode="grid", obstacles=om,
                          eps_mix=config.T1_4_FILTER_EPS[f], rng=np.random.default_rng([seed, 2]))
                want_snap = bool(f == config.T1_4_SNAPSHOT_FILTER and i == 0 and s in config.T1_4_SNAPSHOT_SOURCES)
                rec, snap = run_filter(pf, counts, paths, true_xy, snapshot=want_snap)
                rec.update({"filter": f, "source": s, "seed": seed, "eps_mix": config.T1_4_FILTER_EPS[f]})
                runs[f][s].append(rec)
                if snap is not None:
                    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(args.snapshot_dir / f"t1_4_snapshots_{s}.npz", **snap)
        line = ", ".join(f"{f} {np.median([r['final_map_error_m'] for r in runs[f][s]]):.0f} m "
                         f"({sum(r['success'] for r in runs[f][s])}/{len(runs[f][s])} ok)" for f in filters)
        print(f"[source {s} {source_type(s)}] final MAP error median: {line}; max expected count {max_counts[s]:.0f} "
              f"(bg {bg:.0f}); {time.perf_counter() - t_src:.0f} s", flush=True)

    agg = aggregate(runs)
    currie_counts = det.detection_threshold_cps() * det.T if config.T1_4_UNOBSERVABLE_USE_CURRIE else None
    verd = verdicts(agg, max_counts, sources, currie_counts=currie_counts)
    unobs = [config.T1_4_UNOBSERVABLE_SOURCE] if verd["v_unobservable"]["unobservable_at_15m"] else []
    table = table1_markdown(agg, sources, analytic_std, adjoint_std, library, max_counts, unobs)
    curves = {}
    for f in filters:
        c = pooled_error_curve(runs[f], config.T1_3_OPEN_SOURCES)
        curves[f] = None if c is None else c.tolist()
    make_figure(agg, runs, sources, unobs, args.fig)
    res = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "frame_index": config.T1_4_FRAME_INDEX,
        "step": config.index_to_step(config.T1_4_FRAME_INDEX), "z": config.DRONE_Z,
        "sensor": {"k0": det.k0, "scale": config.T1_4_SENSOR_SCALE, "background_cps": det.background, "T": det.T,
                   "currie_threshold_cps": det.detection_threshold_cps()},
        "pf": {"n_particles": config.PF_N_PARTICLES, "kappa_ref": config.KAPPA_REF, "grid_decades": config.KAPPA_GRID_DECADES,
               "n_grid": config.KAPPA_G, "map_method": "mode", "jitter_m": config.PF_JITTER_M},
        "filters": {f: {"model": FILTER_MODEL[f], "eps_mix": config.T1_4_FILTER_EPS[f], "label": FILTER_LABEL[f]} for f in filters},
        "analytic_params": {"wind_mode": "global", "U": config.T1_4_ANALYTIC_U, "sigma_v": config.T1_4_ANALYTIC_SIGMA_V,
                            "calibration_chosen": analytic_chosen, "matches_calibration": cal_match["analytic"]},
        "adjoint_params": {"K": config.T1_4_ADJOINT_K, "lam": config.T1_4_ADJOINT_LAM, "wind_band": None,
                           "calibration_chosen": adjoint_chosen, "matches_calibration": cal_match["adjoint"], **setup},
        "n_seeds": args.n_seeds, "n_steps": args.n_steps, "n_drones": config.PF_ADJ_N_DRONES, "sources": sources,
        "source_types": {str(s): source_type(s) for s in sources},
        "checkpoints": list(config.T1_4_CHECKPOINT_STEPS), "entropy_steps": list(config.T1_4_ENTROPY_STEPS),
        "gmm_every": config.T1_4_GMM_EVERY, "success_criteria": {"sigma_m": config.SUCCESS_SIGMA_M, "error_m": config.SUCCESS_ERROR_M},
        "path": {"sweep_width_m": config.PF_ADJ_SWEEP_WIDTH_M, "start_downwind_m": config.PF_ADJ_START_DOWNWIND_M,
                 "step_m": config.DRONE_STEP_M, "info": path_info},
        "measurements": meas, "max_expected_counts": {str(s): v for s, v in max_counts.items()},
        "library_reference": {str(s): v for s, v in library.items()},
        "calibration_std_dense": {"analytic": {str(s): v for s, v in analytic_std.items()},
                                  "adjoint": {str(s): v for s, v in adjoint_std.items()}},
        "aggregates": agg, "verdicts": verd, "pooled_open_error_curves": curves, "table1_markdown": table,
        "snapshots": {str(s): str(args.snapshot_dir / f"t1_4_snapshots_{s}.npz") for s in config.T1_4_SNAPSHOT_SOURCES
                      if s in sources and config.T1_4_SNAPSHOT_FILTER in filters},
        "runs": {f: {str(s): [{k: v for k, v in r.items() if k != "map_error_trajectory_m"} for r in rr]
                     for s, rr in by.items()} for f, by in runs.items()},
        "error_trajectories": {f: {str(s): [r["map_error_trajectory_m"] for r in rr] for s, rr in by.items()} for f, by in runs.items()},
        "figure": str(args.fig), "total_seconds": time.perf_counter() - t_start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(table)
    for key in ("i_A_open", "ii_B_open"):
        v = verd[key]
        print(f"{key}: {v['n_pass']}/{v['n_sources']} open sources < {v['pass_m']:.0f} m ->", "PASS" if v["overall_pass"] else "FAIL")
    print(f"iii B better than A: {verd['iii_B_better_than_A']['count']}/{verd['iii_B_better_than_A']['n_sources']} sources")
    print(f"v source {config.T1_4_UNOBSERVABLE_SOURCE}: {verd['v_unobservable']['label']} (max count "
          f"{verd['v_unobservable']['max_expected_count']:.1f} vs threshold {verd['v_unobservable']['threshold_counts']:.1f} counts, "
          f"rule {verd['v_unobservable']['rule']})")
    print(f"total {res['total_seconds']:.0f} s; JSON {args.out}; figure {args.fig}")
    return res


if __name__ == "__main__":
    main()