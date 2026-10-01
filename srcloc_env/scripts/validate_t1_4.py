"""D6-1 / plan S1 T1-4 (gate G1): the decisive real-data test of the RB-PF forward models on the LDM slab truth.

Usage: python -m srcloc_env.scripts.validate_t1_4 [--seed 0] [--n-seeds 5] [--n-steps 150] [--sources 109 102]
                                                  [--filters A B] [--out ...] [--fig ...]
                                                  [--likelihood poisson|negbin] [--nb-r 1.0] [--mode F|T]
                                                  [--baseline-json ...] [--timeavg-json ...]
                                                  [--adjoint-K 16 --adjoint-lam 0.005]   (D8-1 candidate runs; default config)
                                                  --mode T2 = developed-plume time-varying truth, frames 400 + t (D8-2; config.T1_4_MODE_SCHEDULES)
                                                  [--frame-index 599]   (Mode F snapshot other than config.T1_4_FRAME_INDEX; D8-2 frame-luck check)
Writes config.CACHE_DIR / validate_t1_4.json, config.CACHE_DIR / t1_4_snapshots_{src}.npz (T1-5 / figure 5 input)
and config.FIG_DIR / fig_t1_4_errors.png (config.FIG_DPI_FINAL) + _preview.png (config.FIG_DPI_PREVIEW).

Truth and measurements (plan T1-4, exactly as validate_library_filter.run_filter): for each of the 13 sources the
truth is the cached LDM slab at frame config.T1_4_FRAME_INDEX (599 = step 30000), z = config.DRONE_Z, sensor scale
config.T1_4_SENSOR_SCALE; two drones fly validate_pf_adjoint.two_drone_paths (config.T1_4_N_STEPS = 150 RL steps,
config.DRONE_STEP_M, adjacent config.PF_ADJ_SWEEP_WIDTH_M bands, start config.PF_ADJ_START_DOWNWIND_M downwind,
drone-free cells only) and the counts y ~ Poisson((k0 scale n_s(p) + b) T) are drawn ONCE per (source, seed) with
Detector.measure and rng [seed, source, 1] (R3, plan 4.1), then the SAME sequence is fed to every filter.

Truth mode (D7-2; plan 1 시간 모드, config.T1_4_MODES): --mode F (default) is the fixed snapshot above (D6 behaviour);
--mode T is the time-varying truth: RL step t reads frame index frame_index_mode_t(t) = min(config.N_FILES - 1,
round(config.T1_4_MODE_T_START_INDEX + config.FILES_PER_RL_STEP t)) (1.6 files per step, zero-order hold, both drones
of a step share the frame).  The per-step densities are gathered once per source by mode_t_densities (frames visited
in ascending order, all sources' points of a frame in one vectorised query) and reused by every seed and filter; the
Poisson draw uses the same rng stream.  Mode T changes the truth (the plume grows and moves between frames), so its
errors are not directly comparable with Mode F on one axis.

Filters (all RBPF, N = config.PF_N_PARTICLES, grid mode, kappa grid from config, obstacles = ObstacleMap.load(),
PF rng [seed, 2] per repeat, count likelihood --likelihood 'poisson' (R3) or 'negbin' with dispersion --nb-r
(Gamma-Poisson, R22 / R23; D7-1).  Defaults = the D7-3 configuration config.T1_4_D7_3_LIKELIHOOD ('negbin') and
config.T1_4_D7_3_NB_R_MODE_F / _MODE_T (1.0 / 3.0 by --mode), so a flag-less run reproduces the D7-3 gate input;
the D6 Poisson baseline needs --likelihood poisson (D7-3 review); plan 4.3):
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

Final Table 1 and G1 re-judgement (D7-3; plan S1 T1-4 산출물 / G1, 4.3 강건화): when --baseline-json
(config.T1_4_POISSON_BASELINE_JSON = the D6 Poisson / Mode F run, copied before the re-run) exists, ``table1_final_markdown``
puts the baseline and the re-run (negative-binomial likelihood, R22 / R23; D7-2 recommendation r = config.T1_4_D7_3_NB_R_MODE_F)
final errors side by side: source | type | analytic / adjoint std_dense | A(Poisson) | A(NB) | B(Poisson) | B(NB) final err
med/p90 | B(NB) success | library first step | max count, tags the unobservable source and annotates
config.T1_4_OFFSET_SOURCE (109) with its systematic 15 m offset read from --timeavg-json (config.T1_4_TIMEAVG_JSON,
calibrate_timeavg.py 'offset_report'; T1-3c).  ``final_verdicts`` re-judges G1 per source: open sources
(config.T1_3_OPEN_SOURCES) < config.T1_4_FINAL_ERROR_PASS_M for A(NB) and B(NB), and the regression sources
config.T1_4_REGRESSION_SOURCES (102 / 104 / 106, solved by B(Poisson) in D6) not regressed: B(NB) median final error <=
max(config.D7_2_REGRESSION_TOLERANCE x B(Poisson) median, pass_m) per source, plus the D7-2 pooled rule (median over
the regression sources <= tolerance x its Poisson value).

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
def frame_index_mode_t(t: int, start: int = config.T1_4_MODE_T_START_INDEX,
                       files_per_step: float = config.FILES_PER_RL_STEP, n_files: int = config.N_FILES) -> int:
    """Mode T frame index of RL step t (0-based): min(n_files - 1, round(start + files_per_step t)) (plan 1 시간 모드:
    config.FILES_PER_RL_STEP = 1.6 files per RL step under interpretation A, zero-order hold, capped at the last
    cached frame; D7-2)."""
    if t < 0:
        raise ValueError("t must be >= 0")
    return int(min(n_files - 1, round(start + files_per_step * t)))


def frame_schedule_mode_t(n_steps: int, start: int = config.T1_4_MODE_T_START_INDEX,
                          files_per_step: float = config.FILES_PER_RL_STEP, n_files: int = config.N_FILES) -> np.ndarray:
    """(n_steps,) int64 frame indices frame_index_mode_t(t) for t = 0 .. n_steps - 1 (vectorised; np.round is
    round-half-even like Python's round)."""
    t = np.arange(int(n_steps), dtype=np.float64)
    return np.minimum(n_files - 1, np.round(start + files_per_step * t)).astype(np.int64)


def mode_t_densities(backend: LdmSlabBackend, paths_by_source: dict[int, np.ndarray], z: float = config.DRONE_Z,
                     start: int = config.T1_4_MODE_T_START_INDEX, files_per_step: float = config.FILES_PER_RL_STEP,
                     n_files: int = config.N_FILES) -> dict[int, np.ndarray]:
    """Mode T truth densities {source: (n_steps, n_drones)} [particles/m^3] along paths_by_source[source]
    (n_steps, n_drones, 2): RL step t is read from frame frame_index_mode_t(t) (plan 1 시간 모드; D7-2).

    The frames are visited once in ascending order (LdmSlabBackend loads ~0.25 s per frame; 150 steps -> ~150
    frames) and every source's points of a frame are gathered in one vectorised density query, so one pass serves
    all sources, seeds and filters."""
    out: dict[int, np.ndarray] = {}
    if not paths_by_source:
        return out
    arrs = {int(s): np.asarray(p, dtype=np.float64) for s, p in paths_by_source.items()}
    n_max = max(a.shape[0] for a in arrs.values())
    schedule = frame_schedule_mode_t(n_max, start, files_per_step, n_files)
    for s, a in arrs.items():
        out[s] = np.zeros(a.shape[:2], dtype=np.float64)
    for frame in np.unique(schedule):
        steps = np.flatnonzero(schedule == frame)
        for s, a in arrs.items():
            st = steps[steps < a.shape[0]]
            if st.size == 0:
                continue
            out[s][st] = backend.density([s], a[st].reshape(-1, 2), int(frame), z, 1.0).reshape(st.size, a.shape[1])
    return out


def mode_schedule(mode: str) -> tuple[int, float]:
    """(start index, files per RL step) of a time-varying truth mode (config.T1_4_MODE_SCHEDULES: 'T' young plume,
    'T2' developed plume frames >= 400; D8-2).  Mode 'F' has no schedule."""
    if mode not in config.T1_4_MODE_SCHEDULES:
        raise ValueError(f"mode {mode!r} has no frame schedule (time-varying modes: {list(config.T1_4_MODE_SCHEDULES)})")
    start, fps = config.T1_4_MODE_SCHEDULES[mode]
    return int(start), float(fps)


def truth_densities(backend: LdmSlabBackend, source: int, paths: np.ndarray, mode: str = "F",
                    frame_index: int = config.T1_4_FRAME_INDEX, z: float = config.DRONE_Z) -> np.ndarray:
    """(n_steps, n_drones) truth densities [particles/m^3] of ``source`` along paths (n_steps, n_drones, 2):
    mode 'F' = the fixed slab frame frame_index (D6), mode 'T' = mode_t_densities (config.T1_4_MODES; D7-2)."""
    if mode not in config.T1_4_MODES:
        raise ValueError(f"mode must be one of {config.T1_4_MODES}, got {mode!r}")
    n_steps, n_drones = paths.shape[:2]
    if mode == "F":
        return backend.density([source], paths.reshape(-1, 2), frame_index, z, 1.0).reshape(n_steps, n_drones)
    start, fps = mode_schedule(mode)
    return mode_t_densities(backend, {int(source): paths}, z, start, fps)[int(source)]


def generate_measurements(backend: LdmSlabBackend, det: Detector, source: int, paths: np.ndarray, seed: int,
                          frame_index: int = config.T1_4_FRAME_INDEX, z: float = config.DRONE_Z,
                          scale: float = config.T1_4_SENSOR_SCALE, mode: str = "F",
                          densities: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(counts, expected) both (n_steps, n_drones): slab-truth Poisson counts along paths (n_steps, n_drones, 2),
    drawn with the rng stream [seed, source, 1] of validate_library_filter.run_filter (identical sequences).

    mode 'F' (default) reads the fixed frame frame_index (bit-identical to D6), mode 'T' the time-varying frames of
    truth_densities (D7-2); ``densities`` (n_steps, n_drones), when given, replaces the field query (the per-source
    Mode T cache shared by all seeds and filters).  The Poisson draw is the same rng stream in both modes."""
    n_steps, n_drones = paths.shape[:2]
    if densities is None:
        densities = truth_densities(backend, source, paths, mode, frame_index, z)
    dens = np.asarray(densities, dtype=np.float64).reshape(-1)
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


# ---------------------------------------------------------------------------------------- final table (D7-3)
def load_baseline(path: Path) -> dict | None:
    """The Poisson baseline T1-4 JSON (config.T1_4_POISSON_BASELINE_JSON) reduced to {path, created, mode, likelihood,
    nb_r, n_seeds, aggregates}; None if the file is missing (plan D7-3)."""
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    pf = d.get("pf", {})
    return {"path": str(p), "created": d.get("created"), "mode": d.get("mode", "F"),
            "likelihood": pf.get("likelihood", "poisson"), "nb_r": pf.get("nb_r"), "n_seeds": d.get("n_seeds"),
            "aggregates": d.get("aggregates", {})}


def load_offset_report(path: Path, source: int) -> dict | None:
    """The calibrate_timeavg.json 'offset_report' entry of ``source`` (T1-3c: distance / direction of the 15 m slab
    maximum from the source in the time-averaged and the instantaneous field, the adjoint model and the local wind);
    None if the file or the source is missing (plan D7-3)."""
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    rep = d.get("offset_report", {}).get(str(int(source)))
    return None if rep is None else dict(rep)


def offset_annotation(rep: dict | None, frame_index: int = config.T1_4_FRAME_INDEX) -> str:
    """Table 1 note of a systematic offset from an offset_report entry, e.g. '[systematic offset: 15 m slab peak 30 m
    (time mean) / 81 m (frame 599) from the source, 10-22 deg; wind 13 deg]' (validation_log 'G1 FAIL 원인과 치료':
    109's 15 m maximum lies downwind of the release); a placeholder when T1-3c was not run."""
    if not rep:
        return "[systematic offset (T1-3c offset report missing)]"
    parts = []
    if rep.get("mean_peak_distance_m") is not None:
        parts.append(f"{float(rep['mean_peak_distance_m']):.0f} m (time mean)")
    if rep.get("instantaneous_peak_distance_m") is not None:
        parts.append(f"{float(rep['instantaneous_peak_distance_m']):.0f} m (frame {frame_index})")
    dirs = [float(rep[k]) for k in ("mean_peak_direction_deg", "instantaneous_peak_direction_deg") if rep.get(k) is not None]
    d = f", {min(dirs):.0f}-{max(dirs):.0f} deg" if dirs else ""
    w = f"; wind {float(rep['wind_dir_deg']):.0f} deg" if rep.get("wind_dir_deg") is not None else ""
    return "[systematic offset: 15 m slab peak " + " / ".join(parts) + f" from the source{d}{w}]"


def _err_cell(a: dict | None) -> str:
    return "n/a" if a is None else f"{_fmt(a['final_error_median'])} / {_fmt(a['final_error_p90'])}"


def table1_final_markdown(agg_nb: dict[str, dict[str, dict]], agg_poisson: dict[str, dict[str, dict]],
                          sources: Sequence[int], analytic_std: dict[int, float], adjoint_std: dict[int, float],
                          library: dict[int, dict], max_counts: dict[int, float], unobservable: Sequence[int] = (),
                          offset_source: int = config.T1_4_OFFSET_SOURCE, offset_note: str = "",
                          nb_label: str = "NB") -> str:
    """FINAL Table 1 (plan T1-4 산출물, D7-3): source | type | analytic std_dense | adjoint std_dense | A(Poisson) final
    err med/p90 | A(NB) | B(Poisson) | B(NB) | B(NB) success rate | library first step (P > 0.9) | max count.
    ``agg_poisson`` are the baseline aggregates (load_baseline), ``agg_nb`` the re-run's; the unobservable sources
    are tagged '[unobservable at 15 m]' and ``offset_source`` carries ``offset_note`` (offset_annotation)."""
    head = ("| source | type | analytic std_dense | adjoint std_dense | A(Poisson) final err med/p90 [m] "
            f"| A({nb_label}) final err med/p90 [m] | B(Poisson) final err med/p90 [m] | B({nb_label}) final err med/p90 [m] "
            f"| B({nb_label}) success | library first step P>0.9 (sel.) | max count |")
    sep = "|" + "---|" * 11
    rows = [head, sep]
    for s in sources:
        lib = library.get(s, {})
        t = source_type(s)
        if s in unobservable:
            t += " [unobservable at 15 m]"
        if s == offset_source and offset_note:
            t += " " + offset_note
        ap, an = agg_poisson.get("A", {}).get(str(s)), agg_nb.get("A", {}).get(str(s))
        bp, bn = agg_poisson.get("B", {}).get(str(s)), agg_nb.get("B", {}).get(str(s))
        rows.append(
            f"| {s} | {t} | {_fmt(analytic_std.get(s), 2)} | {_fmt(adjoint_std.get(s), 2)} "
            f"| {_err_cell(ap)} | {_err_cell(an)} | {_err_cell(bp)} | {_err_cell(bn)} "
            f"| {_fmt(None if bn is None else 100 * bn['success_rate'])}% "
            f"| {_fmt(lib.get('first_step_median'))} ({_fmt(None if lib.get('selection_rate') is None else 100 * lib['selection_rate'])}%) "
            f"| {_fmt(max_counts.get(s))} |")
    return "\n".join(rows)


def final_verdicts(agg_nb: dict[str, dict[str, dict]], agg_poisson: dict[str, dict[str, dict]],
                   open_sources: Sequence[int] = config.T1_3_OPEN_SOURCES,
                   regression_sources: Sequence[int] = config.T1_4_REGRESSION_SOURCES,
                   pass_m: float = config.T1_4_FINAL_ERROR_PASS_M,
                   tolerance: float = config.D7_2_REGRESSION_TOLERANCE, filters: Sequence[str] = ("A", "B")) -> dict:
    """D7-3 per-source re-judgement of G1 (plan S1 T1-4 / G1) from the re-run aggregates ``agg_nb`` and the Poisson
    baseline ``agg_poisson``:
      'open'       per filter in ``filters``: every open source's median final MAP error < pass_m (and whether it
                   improved on the baseline);
      'regression' filter B on ``regression_sources``: not regressed iff the NB median <= max(tolerance x Poisson
                   median, pass_m) per source, plus the D7-2 pooled rule median_NB <= tolerance x median_Poisson;
      'overall_pass' = all open blocks and the regression block pass."""
    out_open: dict[str, dict] = {}
    for f in filters:
        per = {}
        for s in open_sources:
            n, p = agg_nb.get(f, {}).get(str(s)), agg_poisson.get(f, {}).get(str(s))
            if n is None:
                continue
            per[str(s)] = {"nb_median_m": n["final_error_median"],
                           "poisson_median_m": None if p is None else p["final_error_median"],
                           "pass": bool(n["final_error_median"] < pass_m),
                           "improved": None if p is None else bool(n["final_error_median"] < p["final_error_median"])}
        out_open[f] = {"pass_m": pass_m, "per_source": per, "n_pass": int(sum(v["pass"] for v in per.values())),
                       "n_sources": len(per), "overall_pass": bool(per) and all(v["pass"] for v in per.values())}
    per_reg = {}
    for s in regression_sources:
        n, p = agg_nb.get("B", {}).get(str(s)), agg_poisson.get("B", {}).get(str(s))
        if n is None or p is None:
            continue
        pm, nm = float(p["final_error_median"]), float(n["final_error_median"])
        limit = max(tolerance * pm, pass_m)
        per_reg[str(s)] = {"poisson_median_m": pm, "nb_median_m": nm, "ratio": (nm / pm) if pm > 0 else None,
                           "limit_m": limit, "not_regressed": bool(nm <= limit),
                           "poisson_success_rate": p["success_rate"], "nb_success_rate": n["success_rate"]}
    pooled = None
    if per_reg:
        pm = float(np.median([v["poisson_median_m"] for v in per_reg.values()]))
        nm = float(np.median([v["nb_median_m"] for v in per_reg.values()]))
        pooled = {"poisson_median_m": pm, "nb_median_m": nm, "limit_m": tolerance * pm, "pass": bool(nm <= tolerance * pm)}
    reg = {"tolerance": tolerance, "pass_m": pass_m, "per_source": per_reg, "pooled": pooled,
           "overall_pass": bool(per_reg) and all(v["not_regressed"] for v in per_reg.values()) and bool(pooled and pooled["pass"])}
    return {"open": out_open, "regression": reg,
            "overall_pass": bool(out_open) and all(v["overall_pass"] for v in out_open.values()) and reg["overall_pass"]}


# ---------------------------------------------------------------------------------------- figure
def make_figure(agg: dict[str, dict[str, dict]], runs: dict[str, dict[int, list[dict]]], sources: Sequence[int],
                unobservable: Sequence[int], path: Path, open_sources: Sequence[int] = config.T1_3_OPEN_SOURCES,
                truth_label: str = f"frame {config.T1_4_FRAME_INDEX}", likelihood_label: str = "",
                adjoint_label: str = f"K {config.T1_4_ADJOINT_K} / lambda {config.T1_4_ADJOINT_LAM}") -> None:
    """Left: grouped bars (A, B) of the median final MAP error per source with p90 whiskers; right: pooled
    open-source error-vs-step curves of the four filters.  truth_label names the truth in the x label (Mode F frame
    or the Mode T frame range; D7-2); likelihood_label (e.g. 'negbin r=1') is appended to the title (D7-3)."""
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
    ax.set_xlabel(f"true source (LDM slab truth, {truth_label}, z = {config.DRONE_Z:g} m)")
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
                 f"adjoint {adjoint_label}"
                 + (f"; likelihood {likelihood_label}" if likelihood_label else ""), fontsize=11)
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
    ap.add_argument("--likelihood", choices=list(config.PF_LIKELIHOODS), default=config.T1_4_D7_3_LIKELIHOOD,
                    help="count likelihood of every filter: 'poisson' (R3) or 'negbin' (Gamma-Poisson, R22 / R23; D7-1); "
                         "default = the D7-3 configuration config.T1_4_D7_3_LIKELIHOOD")
    ap.add_argument("--nb-r", type=float, default=None,
                    help="negative-binomial dispersion r (Var = lam + lam^2 / r) of --likelihood negbin; default = "
                         "config.T1_4_D7_3_NB_R_MODE_F (mode F) / config.T1_4_D7_3_NB_R_MODE_T (mode T)")
    ap.add_argument("--mode", choices=list(config.T1_4_MODES), default="F",
                    help="truth mode: 'F' fixed slab frame config.T1_4_FRAME_INDEX (D6), 'T' time-varying frames "
                         "frame_index_mode_t(t) from config.T1_4_MODE_T_START_INDEX (plan 1 Mode T; D7-2)")
    ap.add_argument("--baseline-json", type=Path, default=config.T1_4_POISSON_BASELINE_JSON,
                    help="Poisson / Mode F T1-4 JSON of the final Table 1 comparison and final_verdicts (D7-3); skipped if missing")
    ap.add_argument("--timeavg-json", type=Path, default=config.T1_4_TIMEAVG_JSON,
                    help="calibrate_timeavg.json (T1-3c offset_report) for the config.T1_4_OFFSET_SOURCE annotation; skipped if missing")
    ap.add_argument("--frame-index", type=int, default=config.T1_4_FRAME_INDEX,
                    help="Mode F truth frame index (default config.T1_4_FRAME_INDEX = 599; D8-2 checks other developed frames)")
    ap.add_argument("--adjoint-K", type=float, default=config.T1_4_ADJOINT_K,
                    help="filter B diffusivity K [m^2/s] (default config.T1_4_ADJOINT_K; D8-1 candidate comparison)")
    ap.add_argument("--adjoint-lam", type=float, default=config.T1_4_ADJOINT_LAM,
                    help="filter B loss rate lambda [1/s] (default config.T1_4_ADJOINT_LAM)")
    args = ap.parse_args(argv)
    if args.nb_r is None:                      # D7-3 defaults by truth mode (D7-2 recommendation)
        args.nb_r = float({"F": config.T1_4_D7_3_NB_R_MODE_F, "T": config.T1_4_D7_3_NB_R_MODE_T,
                           "T2": config.T1_4_D7_3_NB_R_MODE_T2}[args.mode])
    sources = [int(s) for s in args.sources]
    filters = [f for f in FILTERS if f in args.filters]

    t_start = time.perf_counter()
    om = ObstacleMap.load()
    backend = LdmSlabBackend()
    det = Detector()
    analytic_std, analytic_chosen = load_calibration_std(config.CACHE_DIR / "calibrate_forward.json")
    adjoint_std, adjoint_chosen = load_calibration_std(config.CACHE_DIR / "calibrate_adjoint.json")
    library = load_library_summary(config.CACHE_DIR / "validate_library_filter.json")
    baseline = load_baseline(args.baseline_json)                 # read BEFORE the run (--out may overwrite it) (D7-3)
    offset_rep = load_offset_report(args.timeavg_json, config.T1_4_OFFSET_SOURCE)
    nb_label = f"negbin r={args.nb_r:g}" if args.likelihood == "negbin" else args.likelihood
    cal_match = {"analytic": bool(analytic_chosen.get("U") == config.T1_4_ANALYTIC_U and analytic_chosen.get("sigma_v") == config.T1_4_ANALYTIC_SIGMA_V
                                  and analytic_chosen.get("wind_mode", "global") == "global"),
                 "adjoint": bool(adjoint_chosen.get("K") == args.adjoint_K and adjoint_chosen.get("lam") == args.adjoint_lam
                                 and adjoint_chosen.get("wind_band") is None)}
    models: dict[str, object] = {}
    setup: dict[str, float] = {}
    if any(FILTER_MODEL[f] == "analytic" for f in filters):
        models["analytic"] = GaussianPlume(ForwardParams(U=config.T1_4_ANALYTIC_U, sigma_v=config.T1_4_ANALYTIC_SIGMA_V), wind_mode="global")
    if any(FILTER_MODEL[f] == "adjoint" for f in filters):
        t0 = time.perf_counter()
        params = AdjointParams(K=args.adjoint_K, lam=args.adjoint_lam)
        op = AdvectionDiffusionOperator.from_data(params, WindField.load(), om).factorize()
        models["adjoint"] = LbmAdjointModel(op)
        setup["adjoint_seconds"] = time.perf_counter() - t0
        setup["adjoint_n_free"] = op.n_free
    print(f"[setup] filters {filters}, sources {sources}, seeds {args.n_seeds}, steps {args.n_steps}; mode {args.mode}; "
          f"likelihood {args.likelihood}" + (f" (r = {args.nb_r:g})" if args.likelihood == "negbin" else "") + "; "
          f"adjoint K {args.adjoint_K:g} / lambda {args.adjoint_lam:g}; calibration match {cal_match}; setup {time.perf_counter() - t_start:.1f} s", flush=True)

    runs: dict[str, dict[int, list[dict]]] = {f: {} for f in filters}
    meas: dict[str, dict] = {}
    max_counts: dict[int, float] = {}
    path_info: dict[str, list] = {}
    bg = det.background * det.T
    paths_by_source: dict[int, np.ndarray] = {}
    for s in sources:
        paths_by_source[s], path_info[str(s)] = two_drone_paths(om, config.SOURCES_XY[s], args.n_steps)
    t0 = time.perf_counter()
    if args.mode != "F":                       # one ascending pass over the Mode T / T2 frames for all sources (D7-2, D8-2)
        t_start_idx, t_fps = mode_schedule(args.mode)
        densities = mode_t_densities(backend, paths_by_source, start=t_start_idx, files_per_step=t_fps)
        schedule = frame_schedule_mode_t(args.n_steps, t_start_idx, t_fps)
        truth_info: dict | None = {"mode": args.mode, "start": t_start_idx, "files_per_step": t_fps,
                                   "first_frame": int(schedule[0]), "last_frame": int(schedule[-1]),
                                   "n_unique_frames": int(np.unique(schedule).size), "frames": schedule.tolist()}
    else:
        densities = {s: truth_densities(backend, s, paths_by_source[s], "F", args.frame_index) for s in sources}
        truth_info = None
    truth_seconds = time.perf_counter() - t0
    if args.mode != "F":
        print(f"[truth] mode {args.mode} densities for {len(sources)} sources: frames {truth_info['first_frame']}..{truth_info['last_frame']} "
              f"({truth_info['n_unique_frames']} unique) in {truth_seconds:.0f} s", flush=True)
    for s in sources:
        true_xy = config.SOURCES_XY[s]
        paths = paths_by_source[s]
        t_src = time.perf_counter()
        for f in filters:
            runs[f][s] = []
        for i in range(args.n_seeds):
            seed = args.seed + i
            counts, expected = generate_measurements(backend, det, s, paths, seed, mode=args.mode, densities=densities[s])
            max_counts[s] = float(expected.max())
            meas[f"{s}_{seed}"] = {"source": s, "seed": seed, "counts_sum": int(counts.sum()), "counts_max": int(counts.max()),
                                   "n_detections": int(det.is_detection(counts).sum()), "expected_max": float(expected.max())}
            for f in filters:
                pf = RBPF(models[FILTER_MODEL[f]], n_particles=config.PF_N_PARTICLES, mode="grid", obstacles=om,
                          eps_mix=config.T1_4_FILTER_EPS[f], rng=np.random.default_rng([seed, 2]),
                          likelihood=args.likelihood, nb_r=args.nb_r)
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
    truth_label = (f"frame {args.frame_index}" if args.mode == "F"
                   else f"mode {args.mode} frames {truth_info['first_frame']}-{truth_info['last_frame']} "
                        f"({truth_info['files_per_step']:g} files / step)")
    make_figure(agg, runs, sources, unobs, args.fig, truth_label=truth_label, likelihood_label=nb_label,
                adjoint_label=f"K {args.adjoint_K:g} / lambda {args.adjoint_lam:g}")
    final_table: str | None = None
    final_verd: dict | None = None
    if baseline is not None:
        final_table = table1_final_markdown(agg, baseline["aggregates"], sources, analytic_std, adjoint_std, library,
                                            max_counts, unobs, offset_note=offset_annotation(offset_rep), nb_label=nb_label)
        final_verd = final_verdicts(agg, baseline["aggregates"])
    res = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "frame_index": args.frame_index,
        "step": config.index_to_step(args.frame_index), "z": config.DRONE_Z,
        "mode": args.mode, "frame_schedule_mode_t": truth_info, "truth_density_seconds": truth_seconds,
        "mode_note": ("mode F: fixed snapshot frame_index (D6 truth)" if args.mode == "F" else
                      f"mode {args.mode}: time-varying truth (frame_schedule_mode_t, config.T1_4_MODE_SCHEDULES); errors are not directly comparable with mode F"),
        "sensor": {"k0": det.k0, "scale": config.T1_4_SENSOR_SCALE, "background_cps": det.background, "T": det.T,
                   "currie_threshold_cps": det.detection_threshold_cps()},
        "pf": {"n_particles": config.PF_N_PARTICLES, "kappa_ref": config.KAPPA_REF, "grid_decades": config.KAPPA_GRID_DECADES,
               "n_grid": config.KAPPA_G, "map_method": "mode", "jitter_m": config.PF_JITTER_M,
               "likelihood": args.likelihood, "nb_r": args.nb_r},
        "filters": {f: {"model": FILTER_MODEL[f], "eps_mix": config.T1_4_FILTER_EPS[f], "label": FILTER_LABEL[f]} for f in filters},
        "analytic_params": {"wind_mode": "global", "U": config.T1_4_ANALYTIC_U, "sigma_v": config.T1_4_ANALYTIC_SIGMA_V,
                            "calibration_chosen": analytic_chosen, "matches_calibration": cal_match["analytic"]},
        "adjoint_params": {"K": args.adjoint_K, "lam": args.adjoint_lam, "wind_band": None,
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
        "baseline": None if baseline is None else {k: v for k, v in baseline.items() if k != "aggregates"},
        "offset_report": {"source": config.T1_4_OFFSET_SOURCE, "path": str(args.timeavg_json), "report": offset_rep,
                          "annotation": offset_annotation(offset_rep)},
        "table1_final_markdown": final_table, "final_verdicts": final_verd,
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
    if final_table is not None and final_verd is not None:
        print(f"FINAL Table 1 (baseline {baseline['likelihood']} {baseline['path']} vs {nb_label}):")
        print(final_table)
        for f, blk in final_verd["open"].items():
            print(f"final {f}({nb_label}) open sources < {blk['pass_m']:.0f} m: {blk['n_pass']}/{blk['n_sources']} ->",
                  "PASS" if blk["overall_pass"] else "FAIL")
        reg = final_verd["regression"]
        pooled_txt = (f"pooled {reg['pooled']['poisson_median_m']:.1f}->{reg['pooled']['nb_median_m']:.1f} m"
                      if reg["pooled"] else "pooled n/a")
        print("final B regression sources not regressed vs Poisson: "
              + ", ".join(f"{s} {v['poisson_median_m']:.0f}->{v['nb_median_m']:.0f} m ({'ok' if v['not_regressed'] else 'REGRESSED'})"
                          for s, v in reg["per_source"].items())
              + f"; {pooled_txt} ->", "PASS" if reg["overall_pass"] else "FAIL")
    print(f"total {res['total_seconds']:.0f} s; JSON {args.out}; figure {args.fig}")
    return res


if __name__ == "__main__":
    main()