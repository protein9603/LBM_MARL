"""D7-2 / plan S1 T1-4, 4.3 강건화, plan 1 시간 모드: (count likelihood) x (truth mode) comparison of filter B on the
real LDM truth -- Poisson vs negative-binomial with dispersion r in config.D7_2_NB_R_GRID, Mode F (fixed snapshot)
vs Mode T (time-varying frames).

Usage: python -m srcloc_env.scripts.validate_d7_modes [--seed 0] [--n-seeds 3] [--n-steps 150] [--sources 101 ...]
                                                      [--modes F T] [--nb-r 0.3 1 3] [--out ...] [--fig ...]
       (D8-2: --modes F T2 --nb-r 1 3 --out validate_d8_2_modes.json --fig fig_d8_2_modes.png; T2 = developed plume, frames 400 + t)
Writes config.CACHE_DIR / validate_d7_modes.json and config.FIG_DIR / fig_d7_modes.png (config.FIG_DPI_FINAL) +
_preview.png (config.FIG_DPI_PREVIEW).

Filter (B only, the policy-training default of validation_log 전방모델 결정): RBPF(LbmAdjointModel(
AdvectionDiffusionOperator.from_data(AdjointParams(K = config.T1_4_ADJOINT_K, lam = config.T1_4_ADJOINT_LAM))
factorised once and shared), N = config.PF_N_PARTICLES, grid mode, eps_mix = config.T1_4_FILTER_EPS['B'], PF rng
[seed, 2]) with likelihood 'poisson' (R3) or 'negbin' with r in config.D7_2_NB_R_GRID (Gamma-Poisson, R22 / R23;
D7-1).  4 likelihood settings x len(config.T1_4_MODES) modes = 8 configs, keyed ``config_key(label, mode)``.

Truth and measurements: validate_t1_4.generate_measurements with mode 'F' (fixed frame config.T1_4_FRAME_INDEX; the
D6 measurement sequence, rng [seed, source, 1]) and mode 'T' (frame frame_index_mode_t(t) per RL step; the per-step
densities are gathered once per source by validate_t1_4.mode_t_densities and shared by all seeds and configs).
Sources: open config.D7_2_OPEN_SOURCES + regression config.D7_2_REGRESSION_SOURCES, config.D7_2_N_SEEDS seeds,
config.D7_2_N_STEPS RL steps of the validate_pf_adjoint.two_drone_paths lawnmower (plan T1-4).

Metrics per (config, source) (validate_t1_4.run_filter / aggregate): final MAP error median over seeds, MAP error
median at config.D7_2_CHECKPOINT_STEPS (50 / 100), success rate (GMM top sigma < config.SUCCESS_SIGMA_M and MAP
error < config.SUCCESS_ERROR_M at any config.T1_4_GMM_EVERY-step check, plan 4.5) and the final top sigma median.

Recommendation rule (``recommend``, stated in the JSON): among the configs choose the one minimising the median over
the OPEN sources of the per-source final-error median, subject to the REGRESSION sources' median final error not
exceeding config.D7_2_REGRESSION_TOLERANCE x their Poisson-F value; the top config.D7_2_N_TOP admissible configs are
listed.  Mode T changes the truth (the plume grows and moves between frames), so its numbers are not directly
comparable with Mode F on the same axis: the two mode blocks are reported separately and the rule is applied over
all configs AND within each mode (the regression baseline stays Poisson-F in both cases).

Figure fig_d7_modes.png: one panel per mode of per-source final-error bars (median over seeds, p90 whisker) for the
4 likelihood settings, log y, config.D7_2_FIG_REF_LINES_M (30 m / 20 m) reference lines.
References: R3 (Poisson likelihood), R5 (RB-PF), R7 (GMM summary), R17 / R18 (adjoint), R22 / R23 (NB likelihood).
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
from srcloc_env.pf.lbm_adjoint import AdjointParams, AdvectionDiffusionOperator, LbmAdjointModel
from srcloc_env.pf.particle_filter import RBPF
from srcloc_env.scripts.validate_pf_adjoint import two_drone_paths
from srcloc_env.scripts.validate_t1_4 import (aggregate, frame_schedule_mode_t, generate_measurements, mode_schedule,
                                               mode_t_densities, run_filter, source_type, truth_densities)
from srcloc_env.sensor.detector import Detector

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Patch   # noqa: E402

FILTER = "B"
BASELINE_LABEL = "poisson"
# figure colours (dataviz palette: neutral baseline + 3 categorical slots)
SETTING_COLORS = ("#8a8a8a", "#2a78d6", "#1baf7a", "#eb6834")
INK_MUTED = "#8a8a8a"
REGRESSION_HATCH = "///"


# ---------------------------------------------------------------------------------------- configs
def likelihood_settings(nb_r_grid: Sequence[float] = config.D7_2_NB_R_GRID) -> list[dict]:
    """[{label, likelihood, nb_r}]: the Poisson baseline (nb_r None) followed by 'negbin_r{r}' for r in nb_r_grid."""
    out = [{"label": BASELINE_LABEL, "likelihood": "poisson", "nb_r": None}]
    for r in nb_r_grid:
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError("nb_r values must be finite and > 0")
        out.append({"label": f"negbin_r{float(r):g}", "likelihood": "negbin", "nb_r": float(r)})
    return out


def config_key(label: str, mode: str) -> str:
    """Config key '{likelihood label}_{mode}', e.g. 'negbin_r0.3_T'; the Poisson-F baseline is 'poisson_F'."""
    if mode not in config.T1_4_MODES:
        raise ValueError(f"mode must be one of {config.T1_4_MODES}, got {mode!r}")
    return f"{label}_{mode}"


def make_filter(model, setting: dict, seed: int, obstacles: ObstacleMap | None) -> RBPF:
    """Filter B of validate_t1_4 (eps config.T1_4_FILTER_EPS['B'], rng [seed, 2]) with the setting's likelihood."""
    return RBPF(model, n_particles=config.PF_N_PARTICLES, mode="grid", obstacles=obstacles,
                eps_mix=config.T1_4_FILTER_EPS[FILTER], rng=np.random.default_rng([int(seed), 2]),
                likelihood=setting["likelihood"],
                nb_r=config.PF_NB_DISPERSION_R if setting["nb_r"] is None else float(setting["nb_r"]))


# ---------------------------------------------------------------------------------------- tables / rule
def final_error_table(agg: dict[str, dict[str, dict]]) -> dict[str, dict[int, float]]:
    """{config key: {source: final_error_median}} from validate_t1_4.aggregate output."""
    return {key: {int(s): float(a["final_error_median"]) for s, a in by.items()} for key, by in agg.items()}


def _median_over(row: dict[int, float], sources: Sequence[int]) -> float | None:
    vals = [float(row[s]) for s in sources if s in row and row[s] is not None]
    return float(np.median(vals)) if vals else None


def recommend(table: dict[str, dict[int, float]], open_sources: Sequence[int] = config.D7_2_OPEN_SOURCES,
              regression_sources: Sequence[int] = config.D7_2_REGRESSION_SOURCES,
              baseline_key: str = config_key(BASELINE_LABEL, "F"),
              tolerance: float = config.D7_2_REGRESSION_TOLERANCE, candidates: Sequence[str] | None = None,
              n_top: int = config.D7_2_N_TOP) -> dict:
    """The D7-2 recommendation rule on a {config key: {source: final error median}} table.

    Statistic: open_median = median over open_sources of the final error median.  Constraint: regression_median =
    median over regression_sources <= tolerance x the baseline_key's regression_median (vacuous when there are no
    regression sources; a candidate with no regression entries is inadmissible unless the constraint is vacuous).
    The admissible candidates (default: every key of the table) are ranked by open_median (ties by regression
    median, then key); ``recommended`` is the first, ``top`` the first n_top.  The baseline is read from the full
    table, so a within-mode ranking (candidates of one mode) keeps the Poisson-F constraint."""
    if baseline_key not in table:
        raise KeyError(f"baseline {baseline_key!r} missing from the table")
    keys = list(table) if candidates is None else [k for k in candidates if k in table]
    base_reg = _median_over(table[baseline_key], regression_sources)
    limit = None if base_reg is None else tolerance * base_reg
    per: dict[str, dict] = {}
    for k in keys:
        open_med = _median_over(table[k], open_sources)
        reg_med = _median_over(table[k], regression_sources)
        adm = True if limit is None else (reg_med is not None and reg_med <= limit)
        ratio = None if (reg_med is None or base_reg is None or base_reg == 0.0) else reg_med / base_reg
        per[k] = {"open_median_m": open_med, "regression_median_m": reg_med,
                  "admissible": bool(adm and open_med is not None), "regression_ratio_to_baseline": ratio}
    ranked = sorted((k for k in keys if per[k]["admissible"]),
                    key=lambda k: (per[k]["open_median_m"], per[k]["regression_median_m"] or 0.0, k))
    for i, k in enumerate(ranked):
        per[k]["rank"] = i + 1
    for k in keys:
        per[k].setdefault("rank", None)
    return {"rule": ("minimise the median over the open sources of the per-source final MAP error median, subject to "
                     f"the regression sources' median final error <= {tolerance:g} x its value for {baseline_key}"),
            "open_sources": [int(s) for s in open_sources], "regression_sources": [int(s) for s in regression_sources],
            "baseline_key": baseline_key, "baseline_regression_median_m": base_reg, "tolerance": float(tolerance),
            "regression_limit_m": limit, "candidates": keys, "per_config": per,
            "ranking": ranked, "top": ranked[:int(n_top)], "recommended": ranked[0] if ranked else None,
            "n_admissible": len(ranked)}


def per_source_table(agg: dict[str, dict[str, dict]], checkpoints: Sequence[int] = config.D7_2_CHECKPOINT_STEPS) -> dict:
    """{config key: {source: {final_error_median_m, final_error_p90_m, final_error_values_m, map_error_median_at,
    success_rate, n_success, top_sigma_median_m, first_success_step_median}}} (the D7-2 metrics)."""
    out: dict[str, dict[str, dict]] = {}
    for key, by in agg.items():
        out[key] = {}
        for s, a in by.items():
            out[key][str(s)] = {
                "final_error_median_m": a["final_error_median"], "final_error_p90_m": a["final_error_p90"],
                "final_error_values_m": a["final_error_values"],
                "map_error_median_at": {str(c): a["map_error_median"].get(str(c)) for c in checkpoints},
                "success_rate": a["success_rate"], "n_success": a["n_success"], "n_seeds": a["n_seeds"],
                "top_sigma_median_m": a["top_sigma_median"],
                "first_success_step_median": a["first_success_step_median"]}
    return out


def _fmt(x: float | None, nd: int = 0) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def table_markdown(table: dict[str, dict[int, float]], sources: Sequence[int], keys: Sequence[str],
                   per: dict | None = None) -> str:
    """Markdown: source | type | final error median [m] per config key (+ success rate when ``per`` is given)."""
    head = "| source | type | " + " | ".join(keys) + " |"
    rows = [head, "|" + "---|" * (2 + len(keys))]
    for s in sources:
        cells = []
        for k in keys:
            v = table.get(k, {}).get(int(s))
            cell = _fmt(v)
            if per is not None and str(s) in per.get(k, {}):
                cell += f" ({100 * per[k][str(s)]['success_rate']:.0f}%)"
            cells.append(cell)
        rows.append(f"| {s} | {source_type(int(s))} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------- figure
def make_figure(agg: dict[str, dict[str, dict]], sources: Sequence[int], settings: Sequence[dict], modes: Sequence[str],
                path: Path, regression_sources: Sequence[int] = config.D7_2_REGRESSION_SOURCES,
                truth_labels: dict[str, str] | None = None, recommended: dict[str, str | None] | None = None) -> None:
    """One panel per mode: grouped bars (one per likelihood setting) of the median final MAP error per source with
    p90 whiskers, regression sources hatched, config.D7_2_FIG_REF_LINES_M reference lines, log y."""
    n_modes = len(modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(8.0 * n_modes, 6.0), constrained_layout=True, sharey=True, squeeze=False)
    x = np.arange(len(sources))
    n_set = len(settings)
    w = 0.8 / n_set
    for ax, mode in zip(axes[0], modes):
        for j, st in enumerate(settings):
            key = config_key(st["label"], mode)
            col = SETTING_COLORS[j % len(SETTING_COLORS)]
            off = (j - 0.5 * (n_set - 1)) * w
            for xi, s in zip(x + off, sources):
                a = agg.get(key, {}).get(str(s))
                if a is None:
                    continue
                m, p = a["final_error_median"], a["final_error_p90"]
                ax.bar(xi, m, width=w - 0.03, color=col, edgecolor="white", linewidth=0.6,
                       hatch=REGRESSION_HATCH if s in regression_sources else None)
                ax.errorbar(xi, m, yerr=[[0.0], [max(p - m, 0.0)]], color="#333333", linewidth=0.9, capsize=2.5)
        for ref, ls in zip(config.D7_2_FIG_REF_LINES_M, ("--", ":")):
            ax.axhline(ref, color=INK_MUTED, linewidth=1.0, linestyle=ls)
            ax.text(x[-1] + 0.55, ref, f" {ref:.0f} m", color=INK_MUTED, fontsize=8, va="bottom", ha="left")
        ax.set_yscale("log")
        ax.set_ylim(*config.D7_2_FIG_YLIM_M)
        ax.set_xticks(x, [f"{s}\n{source_type(int(s)).replace('(holdout)', '')}" for s in sources], fontsize=8)
        label = (truth_labels or {}).get(mode, f"mode {mode}")
        rec = (recommended or {}).get(mode)
        ax.set_title(f"Mode {mode}: {label}" + (f"\nbest within mode: {rec}" if rec else ""), fontsize=9.5)
        ax.set_xlabel("true source (LDM slab truth, z = %g m); hatched = regression sources" % config.DRONE_Z)
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel(f"final MAP error after {config.D7_2_N_STEPS} RL steps [m] (bar: median over seeds, whisker: p90)")
    handles = [Patch(facecolor=SETTING_COLORS[j % len(SETTING_COLORS)], label=st["label"]) for j, st in enumerate(settings)]
    handles.append(Patch(facecolor="white", edgecolor="#555555", hatch=REGRESSION_HATCH, label="regression source"))
    axes[0][0].legend(handles=handles, loc="upper left", fontsize=8, frameon=False, ncol=2)
    fig.suptitle(f"D7-2: filter B (LBM adjoint K {config.T1_4_ADJOINT_K:g} / lambda {config.T1_4_ADJOINT_LAM:g}), "
                 f"Poisson vs negative-binomial likelihood, N = {config.PF_N_PARTICLES}, {config.D7_2_N_SEEDS} seeds; "
                 "Mode T is a different (time-varying) truth - panels are not comparable on one axis", fontsize=10.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIG_DPI_FINAL)
    fig.savefig(path.with_name(path.stem + "_preview" + path.suffix), dpi=config.FIG_DPI_PREVIEW)
    plt.close(fig)


# ---------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=config.D7_2_N_SEEDS)
    ap.add_argument("--n-steps", type=int, default=config.D7_2_N_STEPS)
    ap.add_argument("--sources", type=int, nargs="*", default=list(config.D7_2_OPEN_SOURCES) + list(config.D7_2_REGRESSION_SOURCES))
    ap.add_argument("--modes", nargs="*", default=list(config.T1_4_MODES), choices=list(config.T1_4_MODES))
    ap.add_argument("--nb-r", type=float, nargs="*", default=list(config.D7_2_NB_R_GRID),
                    help="negative-binomial dispersion candidates next to the Poisson baseline (R22 / R23)")
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_d7_modes.json")
    ap.add_argument("--fig", type=Path, default=config.FIG_DIR / "fig_d7_modes.png")
    args = ap.parse_args(argv)
    sources = [int(s) for s in args.sources]
    modes = [m for m in config.T1_4_MODES if m in args.modes]
    settings = likelihood_settings(args.nb_r)
    open_sources = [s for s in config.D7_2_OPEN_SOURCES if s in sources]
    regression_sources = [s for s in config.D7_2_REGRESSION_SOURCES if s in sources]
    keys = [config_key(st["label"], m) for m in modes for st in settings]

    t_start = time.perf_counter()
    om = ObstacleMap.load()
    backend = LdmSlabBackend()
    det = Detector()
    t0 = time.perf_counter()
    op = AdvectionDiffusionOperator.from_data(AdjointParams(K=config.T1_4_ADJOINT_K, lam=config.T1_4_ADJOINT_LAM), WindField.load(), om).factorize()
    model = LbmAdjointModel(op)
    setup = {"adjoint_seconds": time.perf_counter() - t0, "adjoint_n_free": op.n_free}
    print(f"[setup] configs {keys}; sources {sources} (open {open_sources}, regression {regression_sources}); "
          f"seeds {args.n_seeds}, steps {args.n_steps}; adjoint {setup['adjoint_seconds']:.1f} s", flush=True)

    # truth densities per mode and source (Mode T: one ascending frame pass for all sources, D7-2)
    paths_by_source: dict[int, np.ndarray] = {}
    path_info: dict[str, list] = {}
    for s in sources:
        paths_by_source[s], path_info[str(s)] = two_drone_paths(om, config.SOURCES_XY[s], args.n_steps)
    densities: dict[str, dict[int, np.ndarray]] = {}
    truth_seconds: dict[str, float] = {}
    for mode in modes:
        t0 = time.perf_counter()
        if mode != "F":
            densities[mode] = mode_t_densities(backend, paths_by_source, start=mode_schedule(mode)[0], files_per_step=mode_schedule(mode)[1])
        else:
            densities[mode] = {s: truth_densities(backend, s, paths_by_source[s], "F") for s in sources}
        truth_seconds[mode] = time.perf_counter() - t0
    truth_info = {"F": {"frame_index": config.T1_4_FRAME_INDEX, "step": config.index_to_step(config.T1_4_FRAME_INDEX)}}
    for m in config.T1_4_MODE_SCHEDULES:                       # "T" (young plume) and "T2" (developed plume, D8-2)
        st, fps = mode_schedule(m)
        schedule = frame_schedule_mode_t(args.n_steps, st, fps)
        truth_info[m] = {"start": st, "files_per_step": fps, "first_frame": int(schedule[0]), "last_frame": int(schedule[-1]),
                         "n_unique_frames": int(np.unique(schedule).size), "frames": schedule.tolist()}
    max_counts = {m: {str(s): float(det.expected_counts(densities[m][s], config.T1_4_SENSOR_SCALE).max()) for s in sources} for m in modes}
    print("[truth] densities: " + ", ".join(f"mode {m} {truth_seconds[m]:.0f} s" for m in modes) +
          "".join(f"; mode {m} frames {truth_info[m]['first_frame']}..{truth_info[m]['last_frame']}" for m in modes if m != "F"), flush=True)

    runs: dict[str, dict[int, list[dict]]] = {k: {s: [] for s in sources} for k in keys}
    meas: dict[str, dict] = {}
    for s in sources:
        true_xy = config.SOURCES_XY[s]
        paths = paths_by_source[s]
        for mode in modes:
            t_src = time.perf_counter()
            for i in range(args.n_seeds):
                seed = args.seed + i
                counts, expected = generate_measurements(backend, det, s, paths, seed, mode=mode, densities=densities[mode][s])
                meas[f"{s}_{mode}_{seed}"] = {"source": s, "mode": mode, "seed": seed, "counts_sum": int(counts.sum()),
                                              "counts_max": int(counts.max()), "n_detections": int(det.is_detection(counts).sum()),
                                              "expected_max": float(expected.max())}
                for st in settings:
                    key = config_key(st["label"], mode)
                    pf = make_filter(model, st, seed, om)
                    rec, _ = run_filter(pf, counts, paths, true_xy, checkpoints=tuple(config.D7_2_CHECKPOINT_STEPS))
                    rec.update({"config": key, "likelihood": st["likelihood"], "nb_r": st["nb_r"], "mode": mode,
                                "filter": FILTER, "source": s, "seed": seed, "eps_mix": config.T1_4_FILTER_EPS[FILTER]})
                    runs[key][s].append(rec)
            line = ", ".join(f"{st['label']} {np.median([r['final_map_error_m'] for r in runs[config_key(st['label'], mode)][s]]):.0f} m "
                             f"({sum(r['success'] for r in runs[config_key(st['label'], mode)][s])}/{args.n_seeds})" for st in settings)
            print(f"[source {s} {source_type(s)} mode {mode}] final MAP error median: {line}; max expected count "
                  f"{max_counts[mode][str(s)]:.0f}; {time.perf_counter() - t_src:.0f} s", flush=True)

    agg = aggregate(runs, tuple(config.D7_2_CHECKPOINT_STEPS))
    table = final_error_table(agg)
    per = per_source_table(agg)
    baseline_key = config_key(BASELINE_LABEL, "F")
    rec_all = recommend(table, open_sources, regression_sources, baseline_key=baseline_key) if baseline_key in table else None
    rec_mode = {m: recommend(table, open_sources, regression_sources, baseline_key=baseline_key,
                             candidates=[config_key(st["label"], m) for st in settings])
                for m in modes} if baseline_key in table else {}
    truth_labels = {"F": f"fixed frame {config.T1_4_FRAME_INDEX} (step {truth_info['F']['step']})",
                    **{m: f"time-varying frames {truth_info[m]['first_frame']}-{truth_info[m]['last_frame']} ({truth_info[m]['files_per_step']:g} files / step)"
                       for m in config.T1_4_MODE_SCHEDULES}}
    make_figure(agg, sources, settings, modes, args.fig, regression_sources, truth_labels,
                {m: (rec_mode[m]["recommended"] if m in rec_mode else None) for m in modes})
    md_all = table_markdown(table, sources, keys, per)
    res = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "z": config.DRONE_Z,
        "note_modes": ("Mode T changes the truth (time-varying LDM frames: the plume grows and moves), so its errors are "
                       "not directly comparable with Mode F on the same axis; the blocks are reported separately and the "
                       "recommendation rule is applied over all configs and within each mode (regression baseline: Poisson-F)."),
        "filter": {"label": FILTER, "model": "adjoint", "K": config.T1_4_ADJOINT_K, "lam": config.T1_4_ADJOINT_LAM,
                   "wind_band": None, "eps_mix": config.T1_4_FILTER_EPS[FILTER], **setup},
        "pf": {"n_particles": config.PF_N_PARTICLES, "kappa_ref": config.KAPPA_REF, "grid_decades": config.KAPPA_GRID_DECADES,
               "n_grid": config.KAPPA_G, "map_method": "mode", "jitter_m": config.PF_JITTER_M},
        "sensor": {"k0": det.k0, "scale": config.T1_4_SENSOR_SCALE, "background_cps": det.background, "T": det.T},
        "settings": settings, "modes": modes, "configs": keys, "sources": sources,
        "open_sources": open_sources, "regression_sources": regression_sources,
        "source_types": {str(s): source_type(s) for s in sources},
        "n_seeds": args.n_seeds, "seed0": args.seed, "n_steps": args.n_steps, "n_drones": config.PF_ADJ_N_DRONES,
        "checkpoints": list(config.D7_2_CHECKPOINT_STEPS), "gmm_every": config.T1_4_GMM_EVERY,
        "success_criteria": {"sigma_m": config.SUCCESS_SIGMA_M, "error_m": config.SUCCESS_ERROR_M},
        "truth": {m: truth_info[m] for m in modes}, "truth_density_seconds": truth_seconds,
        "max_expected_counts": max_counts, "measurements": meas,
        "path": {"sweep_width_m": config.PF_ADJ_SWEEP_WIDTH_M, "start_downwind_m": config.PF_ADJ_START_DOWNWIND_M,
                 "step_m": config.DRONE_STEP_M, "info": path_info},
        "aggregates": agg, "final_error_table": {k: {str(s): v for s, v in row.items()} for k, row in table.items()},
        "per_source_table": per,
        "blocks": {m: {"configs": [config_key(st["label"], m) for st in settings],
                       "table_markdown": table_markdown(table, sources, [config_key(st["label"], m) for st in settings], per),
                       "recommendation": rec_mode.get(m)} for m in modes},
        "recommendation_all": rec_all, "recommendation_by_mode": rec_mode, "table_markdown": md_all,
        "runs": {k: {str(s): [{kk: v for kk, v in r.items() if kk != "map_error_trajectory_m"} for r in rr]
                     for s, rr in by.items()} for k, by in runs.items()},
        "error_trajectories": {k: {str(s): [r["map_error_trajectory_m"] for r in rr] for s, rr in by.items()} for k, by in runs.items()},
        "figure": str(args.fig), "total_seconds": time.perf_counter() - t_start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(md_all)
    if rec_all is not None:
        print(f"rule: {rec_all['rule']}; regression limit {_fmt(rec_all['regression_limit_m'], 1)} m")
        print(f"top-{config.D7_2_N_TOP} over all configs: " + ", ".join(
            f"{k} (open {rec_all['per_config'][k]['open_median_m']:.1f} m, regression {_fmt(rec_all['per_config'][k]['regression_median_m'], 1)} m)"
            for k in rec_all["top"]))
        for m in modes:
            r = rec_mode[m]
            print(f"mode {m}: recommended {r['recommended']} ({r['n_admissible']}/{len(r['candidates'])} admissible)")
    print(f"total {res['total_seconds']:.0f} s; JSON {args.out}; figure {args.fig}")
    return res


if __name__ == "__main__":
    main()