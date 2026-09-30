"""T1-2: marginalising kappa (RB-PF, plan 4.3 / R5) avoids the position bias of a wrongly fixed kappa
(plan S1 T1-2; D5-3; figure 4 fig4_kappa_bias.png).

Usage: python -m srcloc_env.scripts.validate_kappa_bias [--seed 0] [--n-repeats 20] [--n-steps 100]
       [--out <CACHE_DIR>/validate_kappa_bias.json] [--fig <FIG_DIR>/fig4_kappa_bias.png]
       [--likelihood poisson|negbin] [--nb-r 0.3]

Design (plan T1-2, followed with the D5-3 choices recorded in config)
---------------------------------------------------------------------
truth   generated from the SAME forward model as the filter (no model mismatch) for config.T1_2_CASES:
        (a) GaussianPlume(wind_mode='global', ForwardParams() = config defaults, plan 4.2, R13 / R15) with the true
            source at config.SOURCES_XY[109] and, as a second case, [101];
        (b) LbmAdjointModel on the real LBM wind at z = config.DRONE_Z with the default AdjointParams (plan 4.2b,
            R17 / R18), true source 109.
        The unit response of the truth is model.unit_response(true_xy, drones) - exactly what a filter particle
        sitting at the true position would compute (for the adjoint model this is the interpolated psi, section 5
        step 4 of docs/lbm_forward_model.md, so the filter's own approximation is part of the truth).
kappa   kappa_true ~ log-uniform config.KAPPA_REF x 10^[-T1_2_KAPPA_TRUE_DECADES, +T1_2_KAPPA_TRUE_DECADES] per
        repeat (config.T1_2_N_REPEATS = 20 repeats, seeded); q_true = kappa_true / k0 so that the Detector draws
        y ~ Poisson((kappa_true g + b) T) (plan 4.1 / 4.3, R3; b = config.SENSOR_BACKGROUND_CPS = 20 cps).
path    the two-drone sawtooth lawnmower of scripts/validate_pf_adjoint.two_drone_paths (plan T1-4 "2대 lawnmower"):
        start config.PF_ADJ_START_DOWNWIND_M downwind, fly -x, adjacent y bands of width PF_ADJ_SWEEP_WIDTH_M;
        config.T1_2_N_STEPS = 100 steps per drone -> 200 measurements per repeat, fused sequentially on one RBPF
        (drone 1, drone 2, drone 1, ...; plan 4.3).  The same path and the same counts are fed to every filter.
filters (all N = config.PF_N_PARTICLES, mode='grid', same PF seed, obstacles = ObstacleMap.load(), eps_mix default;
        count likelihood --likelihood for ALL four filters: 'poisson' (R3) or 'negbin' with dispersion --nb-r
        (R22 / R23, D7-1) - the truth stays Poisson, so 'negbin' measures the cost of the over-dispersed
        likelihood on well-specified counts)
        (i)   kappa_known      kappa fixed at kappa_true            (oracle reference)
        (ii)  rbpf_grid        RB-PF with the config grid: KAPPA_REF x 10^[-KAPPA_GRID_DECADES, +], G = KAPPA_G
        (iii) kappa_fixed_high kappa fixed at T1_2_KAPPA_FIXED_FACTORS[0] x kappa_true (3x)
        (iv)  kappa_fixed_low  kappa fixed at T1_2_KAPPA_FIXED_FACTORS[1] x kappa_true (0.3x)
        A "fixed" kappa is an RBPF with n_grid = 2 and grid_decades = config.T1_2_FIXED_KAPPA_DECADES (1e-6): the
        two nodes differ by 4.6e-6 relative, i.e. the likelihood is the Poisson likelihood at that kappa to
        round-off (RBPF rejects n_grid = 1 / grid_decades = 0; ``make_filter``).
metrics per repeat and filter after the 200 measurements: MAP error |MAP - true| [m] (RBPF.map_estimate, weighted
        mode), the signed downwind bias (MAP - true) . e_x [m] (projection on +x, the domain mean wind, report 3.3;
        the LBM direction at the source is recorded for information), the y offset, the posterior-mean error, the
        posterior spread sqrt(trace cov), the kappa posterior quantiles (median / kappa_true is the kappa-recovery
        statistic of (ii)), N_eff, resample count, and the MAP error after config.T1_2_REPORT_MEASUREMENTS
        measurements.  Per repeat also |MAP_xy(ii) - MAP_xy(i)| (distance between the two estimates) and
        err(ii) - err(i).  ``summarise`` reports median and IQR (q25, q75) over the repeats.
pass    (plan T1-2)  1. median over repeats of |MAP_xy(ii) - MAP_xy(i)| < config.T1_2_MAP_DIFF_PASS_M (10 m);
                     2. the downwind bias of (iii) and of (iv) is larger than that of (ii): median |bias_x| of
                        each fixed-wrong filter > median |bias_x| of the RB-PF.  The SIGN of the median bias is
                        recorded as information; it depends on the plume geometry, not only on the kappa error:
                        a too-high kappa makes the filter look for positions with a WEAKER response at the
                        measured spots.  For the adjoint layer model the response is largest at the source, so
                        the source is pushed upwind (bias_x < 0); for the analytic plume sampled at z = 15 m the
                        response GROWS over the first ~100 m downwind (the vertical spread from z_s = 5.5 m has
                        to reach the drone altitude first), so weaker-response positions lie closer to the
                        measured peak and the source is pushed downwind (bias_x > 0).  A too-low kappa reverses
                        the adjoint sign (0.3x: bias_x > 0), but NOT the analytic one: there the 0.3x filter is
                        also pushed downwind in the median (+41 m / +25 m for 109 / 101, wide IQR; the weaker
                        near-source response of the z = 15 m plume makes a source shifted towards the measured
                        peak fit large counts better than one shifted upwind).  Hence the criterion is on
                        |bias_x| only and the signs are recorded as data (evaluate_pass 'bias_x_median_sign').
        Both criteria are evaluated per case and overall (all cases).

Outputs
-------
config.CACHE_DIR / validate_kappa_bias.json   settings, per-case repeats (all filters), summaries, pass flags, the
        implied kappa_ref of T1-3 / T1-3b (from cache/calibrate_forward.json and calibrate_adjoint.json, if present)
        next to config.KAPPA_REF for context, timings.
config.FIG_DIR / fig4_kappa_bias.png (+ _preview.png)  figure 4: one column per case; row 1 box plots of the MAP
        error of the four filters (points = repeats), row 2 box plots of the signed downwind bias, row 3 kappa
        recovery of the RB-PF (posterior median / kappa_true vs kappa_true / KAPPA_REF, with the 3x / 0.3x lines
        of the fixed-wrong filters).  English labels, Agg backend, config.FIG_DPI_FINAL + FIG_DPI_PREVIEW.

References: R3 (Poisson count likelihood), R5 (Rao-Blackwellised PF: kappa marginalised per particle), R13 / R15
(analytic plume), R16 (systematic resampling), R17 / R18 (adjoint source-receptor model), R22 / R23 (negative-binomial
likelihood, D7-1).  ``sample_kappa_true``,
``make_filter``, ``simulate_counts``, ``run_filter``, ``run_repeat``, ``run_case``, ``summarise`` and
``evaluate_pass`` are data-free and unit-tested on a small GaussianPlume set-up in tests/test_kappa_bias.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.field.wind import WindField
from srcloc_env.pf.forward_model import GaussianPlume
from srcloc_env.pf.lbm_adjoint import AdjointParams, AdvectionDiffusionOperator, LbmAdjointModel
from srcloc_env.pf.particle_filter import ForwardModel, RBPF
from srcloc_env.scripts.validate_pf_adjoint import two_drone_paths
from srcloc_env.sensor.detector import Detector

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

FILTER_NAMES: tuple[str, ...] = ("kappa_known", "rbpf_grid", "kappa_fixed_high", "kappa_fixed_low")
FILTER_LABELS: dict[str, str] = {
    "kappa_known": "(i) kappa known",
    "rbpf_grid": "(ii) RB-PF grid",
    "kappa_fixed_high": f"(iii) kappa fixed {config.T1_2_KAPPA_FIXED_FACTORS[0]:g}x",
    "kappa_fixed_low": f"(iv) kappa fixed {config.T1_2_KAPPA_FIXED_FACTORS[1]:g}x",
}
# categorical colours in fixed filter order: dataviz palette slots 1-4 (validated adjacent quadruple, light mode)
FILTER_COLORS: dict[str, str] = {"kappa_known": "#2a78d6", "rbpf_grid": "#eb6834",
                                 "kappa_fixed_high": "#1baf7a", "kappa_fixed_low": "#eda100"}
INK = "#222222"
INK_MUTED = "#8a8a8a"
GRID_COLOR = "#e6e6e6"
MODEL_LABELS = {"analytic": "Gaussian plume (global wind)", "adjoint": "LBM adjoint (15 m wind)"}


# ---------------------------------------------------------------------------------------- building blocks
def sample_kappa_true(rng: np.random.Generator, n: int, kappa_ref: float = config.KAPPA_REF,
                      decades: float = config.T1_2_KAPPA_TRUE_DECADES) -> np.ndarray:
    """(n,) kappa_true ~ log-uniform kappa_ref x 10^[-decades, +decades] (plan D5-3)."""
    return float(kappa_ref) * 10.0 ** rng.uniform(-float(decades), float(decades), int(n))


def make_filter(name: str, model: ForwardModel, kappa_true: float, rng: np.random.Generator,
                n_particles: int = config.PF_N_PARTICLES, obstacles: ObstacleMap | None = None,
                prior_x: tuple[float, float] = config.PF_PRIOR_X, prior_y: tuple[float, float] = config.PF_PRIOR_Y,
                fixed_factors: tuple[float, float] = config.T1_2_KAPPA_FIXED_FACTORS,
                fixed_decades: float = config.T1_2_FIXED_KAPPA_DECADES, n_grid: int = config.KAPPA_G,
                likelihood: str = config.PF_LIKELIHOOD, nb_r: float = config.PF_NB_DISPERSION_R) -> RBPF:
    """One of the four T1-2 filters (module docstring), all mode='grid' on the same forward model and with the same
    count likelihood (``likelihood`` / ``nb_r``, D7-1).

    'rbpf_grid' uses the config kappa grid (KAPPA_REF, KAPPA_GRID_DECADES; G = `n_grid`, default KAPPA_G - the
    --n-grid diagnostic of main() refines the 0.2-decade spacing); the three fixed-kappa
    filters are degenerate two-node grids of half-width ``fixed_decades`` centred at kappa_true x factor
    (factor 1 for 'kappa_known', fixed_factors[0] / [1] for 'kappa_fixed_high' / 'kappa_fixed_low').
    """
    common = dict(n_particles=int(n_particles), mode="grid", obstacles=obstacles, rng=rng,
                  prior_x=prior_x, prior_y=prior_y, likelihood=likelihood, nb_r=float(nb_r))
    if name == "rbpf_grid":
        return RBPF(model, n_grid=int(n_grid), **common)
    factors = {"kappa_known": 1.0, "kappa_fixed_high": float(fixed_factors[0]), "kappa_fixed_low": float(fixed_factors[1])}
    if name not in factors:
        raise ValueError(f"unknown filter {name!r}; expected one of {FILTER_NAMES}")
    return RBPF(model, kappa_ref=factors[name] * float(kappa_true), grid_decades=float(fixed_decades), n_grid=2,
                **common)


def simulate_counts(model: ForwardModel, true_xy: np.ndarray, drones_xyz: np.ndarray, kappa_true: float,
                    det: Detector, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(counts (M,) int64, expected (M,)) at the M drone positions: truth = the filter's own model at true_xy,
    y ~ Poisson((kappa_true g + b) T) via Detector.measure with q_true = kappa_true / k0 (plan 4.1 / 4.3, R3)."""
    drn = np.asarray(drones_xyz, dtype=np.float64).reshape(-1, 3)
    g = np.asarray(model.unit_response(np.asarray(true_xy, dtype=np.float64).reshape(1, 2), drn), dtype=np.float64)[0]
    density = g * (float(kappa_true) / det.k0)
    return det.measure(density, 1.0, rng), det.expected_counts(density)


def run_filter(pf: RBPF, counts: np.ndarray, drones_xyz: np.ndarray, true_xy: np.ndarray, kappa_true: float,
               report_at: tuple[int, ...] = config.T1_2_REPORT_MEASUREMENTS) -> dict:
    """Feed the M counts sequentially, return the final metrics (module docstring) of one filter."""
    truth = np.asarray(true_xy, dtype=np.float64).reshape(2)
    drn = np.asarray(drones_xyz, dtype=np.float64).reshape(-1, 3)
    m = drn.shape[0]
    err_at: dict[str, float] = {}
    t0 = time.perf_counter()
    for i in range(m):
        pf.update(int(counts[i]), drn[i])
        if (i + 1) in report_at:
            err_at[str(i + 1)] = float(np.hypot(*(pf.map_estimate() - truth)))
    wall = time.perf_counter() - t0
    map_xy = pf.map_estimate()
    d = map_xy - truth
    q = pf.posterior_kappa_quantiles((0.05, 0.5, 0.95))
    return {
        "map_xy": map_xy.tolist(), "map_error_m": float(np.hypot(*d)),
        "bias_x_m": float(d[0]), "bias_y_m": float(d[1]),
        "mean_error_m": float(np.hypot(*(pf.mean() - truth))),
        "posterior_spread_m": float(np.sqrt(np.trace(pf.covariance()))),
        "kappa_q05_q50_q95": q.tolist(), "kappa_median_over_true": float(q[1] / kappa_true),
        "kappa_edge_mass": list(pf.kappa_edge_mass()),
        "neff_final": pf.neff(), "n_resamples": int(pf.n_resamples), "n_updates": int(pf.n_updates),
        "map_error_at": err_at, "wall_seconds": wall,
    }


def run_repeat(model: ForwardModel, true_xy: np.ndarray, drones_xyz: np.ndarray, kappa_true: float, seed: int,
               repeat: int, det: Detector, obstacles: ObstacleMap | None = None,
               n_particles: int = config.PF_N_PARTICLES, prior_x: tuple[float, float] = config.PF_PRIOR_X,
               prior_y: tuple[float, float] = config.PF_PRIOR_Y,
               report_at: tuple[int, ...] = config.T1_2_REPORT_MEASUREMENTS,
               filters: tuple[str, ...] = FILTER_NAMES, n_grid: int = config.KAPPA_G,
               likelihood: str = config.PF_LIKELIHOOD, nb_r: float = config.PF_NB_DISPERSION_R) -> dict:
    """One repeat: one Poisson count set (rng [seed, repeat, 1]) fed to every filter, each started from the same
    PF seed (rng [seed, repeat, 2]); returns the per-filter metrics and the (ii) vs (i) comparison."""
    counts, expected = simulate_counts(model, true_xy, drones_xyz, kappa_true, det, np.random.default_rng([seed, repeat, 1]))
    out: dict = {
        "repeat": int(repeat), "kappa_true": float(kappa_true), "kappa_true_over_ref": float(kappa_true / config.KAPPA_REF),
        "counts_max": int(counts.max()), "counts_mean": float(counts.mean()),
        "expected_counts_max": float(expected.max()), "n_detections": int(det.is_detection(counts).sum()),
        "filters": {},
    }
    for name in filters:
        pf = make_filter(name, model, kappa_true, np.random.default_rng([seed, repeat, 2]), n_particles, obstacles,
                         prior_x, prior_y, n_grid=n_grid, likelihood=likelihood, nb_r=nb_r)
        out["filters"][name] = run_filter(pf, counts, drones_xyz, true_xy, kappa_true, report_at)
    if "rbpf_grid" in out["filters"] and "kappa_known" in out["filters"]:
        a, b = out["filters"]["rbpf_grid"], out["filters"]["kappa_known"]
        out["map_distance_ii_i_m"] = float(np.hypot(*(np.asarray(a["map_xy"]) - np.asarray(b["map_xy"]))))
        out["map_error_diff_ii_minus_i_m"] = float(a["map_error_m"] - b["map_error_m"])
    return out


def _med_iqr(values: list[float] | np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    q25, med, q75 = np.percentile(v, [25.0, 50.0, 75.0])
    return {"median": float(med), "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25), "n": int(v.size)}


def summarise(repeats: list[dict], filters: tuple[str, ...] = FILTER_NAMES) -> dict:
    """Median / IQR over the repeats of every per-filter metric, the (ii) vs (i) distances and kappa recovery."""
    per_filter: dict[str, dict] = {}
    for name in filters:
        rows = [r["filters"][name] for r in repeats]
        abs_bias = [abs(x["bias_x_m"]) for x in rows]
        per_filter[name] = {
            "label": FILTER_LABELS[name],
            "map_error_m": _med_iqr([x["map_error_m"] for x in rows]),
            "bias_x_m": _med_iqr([x["bias_x_m"] for x in rows]),
            "abs_bias_x_m": _med_iqr(abs_bias),
            "bias_y_m": _med_iqr([x["bias_y_m"] for x in rows]),
            "mean_error_m": _med_iqr([x["mean_error_m"] for x in rows]),
            "posterior_spread_m": _med_iqr([x["posterior_spread_m"] for x in rows]),
            "kappa_median_over_true": _med_iqr([x["kappa_median_over_true"] for x in rows]),
            "n_resamples": _med_iqr([x["n_resamples"] for x in rows]),
            "map_error_at": {k: _med_iqr([x["map_error_at"][k] for x in rows]) for k in rows[0]["map_error_at"]},
            "fraction_bias_negative": float(np.mean([x["bias_x_m"] < 0.0 for x in rows])),
            "wall_seconds_total": float(sum(x["wall_seconds"] for x in rows)),
        }
    out = {"n_repeats": len(repeats), "filters": per_filter,
           "kappa_true_over_ref": _med_iqr([r["kappa_true_over_ref"] for r in repeats]),
           "expected_counts_max": _med_iqr([r["expected_counts_max"] for r in repeats]),
           "n_detections": _med_iqr([r["n_detections"] for r in repeats])}
    if all("map_distance_ii_i_m" in r for r in repeats):
        out["map_distance_ii_i_m"] = _med_iqr([r["map_distance_ii_i_m"] for r in repeats])
        out["map_error_diff_ii_minus_i_m"] = _med_iqr([r["map_error_diff_ii_minus_i_m"] for r in repeats])
    return out


def evaluate_pass(summary: dict, map_diff_pass_m: float = config.T1_2_MAP_DIFF_PASS_M) -> dict:
    """Plan T1-2 criteria on one case summary (module docstring 'pass'); sign expectations are informational."""
    f = summary["filters"]
    med_dist = summary["map_distance_ii_i_m"]["median"]
    ii = f["rbpf_grid"]["abs_bias_x_m"]["median"]
    hi = f["kappa_fixed_high"]["abs_bias_x_m"]["median"]
    lo = f["kappa_fixed_low"]["abs_bias_x_m"]["median"]
    return {
        "map_distance_ii_i_median_m": med_dist, "map_diff_pass_m": map_diff_pass_m,
        "pass_map_diff": bool(med_dist < map_diff_pass_m),
        "abs_bias_x_median_m": {"rbpf_grid": ii, "kappa_fixed_high": hi, "kappa_fixed_low": lo,
                                "kappa_known": f["kappa_known"]["abs_bias_x_m"]["median"]},
        "pass_bias_high": bool(hi > ii), "pass_bias_low": bool(lo > ii), "pass_bias": bool(hi > ii and lo > ii),
        "bias_x_median_sign": {nm: ("downwind" if f[nm]["bias_x_m"]["median"] > 0.0 else "upwind") for nm in
                               ("kappa_known", "rbpf_grid", "kappa_fixed_high", "kappa_fixed_low")},
    }


def run_case(model: ForwardModel, true_xy: np.ndarray, paths: np.ndarray, seed: int, n_repeats: int,
             det: Detector, obstacles: ObstacleMap | None = None, n_particles: int = config.PF_N_PARTICLES,
             prior_x: tuple[float, float] = config.PF_PRIOR_X, prior_y: tuple[float, float] = config.PF_PRIOR_Y,
             report_at: tuple[int, ...] = config.T1_2_REPORT_MEASUREMENTS,
             filters: tuple[str, ...] = FILTER_NAMES, n_grid: int = config.KAPPA_G,
             likelihood: str = config.PF_LIKELIHOOD, nb_r: float = config.PF_NB_DISPERSION_R) -> dict:
    """All repeats of one (model, source) case on ``paths`` (n_steps, n_drones, 2): kappa_true drawn once per
    repeat from rng [seed, 0]; measurements interleave the drones per step (drone 1, drone 2, ...)."""
    kappas = sample_kappa_true(np.random.default_rng([seed, 0]), n_repeats)
    flat = np.asarray(paths, dtype=np.float64).reshape(-1, 2)
    drones_xyz = np.column_stack([flat, np.full(flat.shape[0], config.DRONE_Z)])
    repeats = [run_repeat(model, true_xy, drones_xyz, float(kappas[r]), seed, r, det, obstacles, n_particles,
                          prior_x, prior_y, report_at, filters, n_grid, likelihood=likelihood, nb_r=nb_r)
               for r in range(int(n_repeats))]
    summary = summarise(repeats, filters)
    out = {"true_xy": np.asarray(true_xy, dtype=np.float64).tolist(), "n_measurements": int(flat.shape[0]),
           "kappa_true": kappas.tolist(), "repeats": repeats, "summary": summary}
    if "rbpf_grid" in filters and "kappa_known" in filters and "kappa_fixed_high" in filters and "kappa_fixed_low" in filters:
        out["pass"] = evaluate_pass(summary)
    return out


# ---------------------------------------------------------------------------------------- figure 4
def _box(ax: plt.Axes, data: list[np.ndarray], names: tuple[str, ...], log: bool = False) -> None:
    """Thin box plots (no fliers) with the repeats as jittered points, one filter colour each."""
    pos = np.arange(len(names))
    bp = ax.boxplot(data, positions=pos, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.6), whiskerprops=dict(color=INK_MUTED, linewidth=1.0),
                    capprops=dict(color=INK_MUTED, linewidth=1.0), boxprops=dict(linewidth=1.0))
    rng = np.random.default_rng(0)
    for k, (patch, name) in enumerate(zip(bp["boxes"], names)):
        patch.set_facecolor(FILTER_COLORS[name])
        patch.set_alpha(0.25)
        patch.set_edgecolor(FILTER_COLORS[name])
        x = pos[k] + rng.uniform(-0.16, 0.16, len(data[k]))
        ax.scatter(x, data[k], s=11, color=FILTER_COLORS[name], alpha=0.8, linewidths=0.0, zorder=3)
    ax.set_xticks(pos)
    ax.set_xticklabels([FILTER_LABELS[n].replace(" kappa", "\nkappa") for n in names], fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def make_figure(cases: list[dict], path: Path, names: tuple[str, ...] = FILTER_NAMES,
                likelihood: str = config.PF_LIKELIHOOD, nb_r: float = config.PF_NB_DISPERSION_R) -> None:
    """Figure 4 (module docstring): columns = cases; rows = MAP error, downwind bias, RB-PF kappa recovery;
    the count likelihood of the filters is named in the title (D7-1)."""
    lik = f"{likelihood} likelihood" + (f" (r = {nb_r:g})" if likelihood == "negbin" else "")
    n = len(cases)
    fig, axes = plt.subplots(3, n, figsize=(4.6 * n + 0.8, 11.0), constrained_layout=True, squeeze=False)
    for j, c in enumerate(cases):
        reps = c["repeats"]
        title = f"{MODEL_LABELS.get(c['model'], c['model'])}, source {c['source']}"
        # row 1: MAP error
        ax = axes[0, j]
        data = [np.array([r["filters"][nm]["map_error_m"] for r in reps]) for nm in names]
        _box(ax, [np.maximum(d, 1.0) for d in data], names, log=True)
        ax.set_ylim(1.0, 2000.0)
        ax.axhline(config.T1_2_MAP_DIFF_PASS_M, color=INK_MUTED, linewidth=0.8, linestyle="--")
        ax.text(len(names) - 0.5, config.T1_2_MAP_DIFF_PASS_M, f" {config.T1_2_MAP_DIFF_PASS_M:.0f} m", color=INK_MUTED,
                fontsize=8, va="bottom", ha="right")
        p = c["pass"]
        ax.set_title(f"{title}\n|MAP(ii) - MAP(i)| median {p['map_distance_ii_i_median_m']:.1f} m "
                     f"({'PASS' if p['pass_map_diff'] else 'FAIL'}, < {config.T1_2_MAP_DIFF_PASS_M:.0f} m)", fontsize=9.5)
        if j == 0:
            ax.set_ylabel(f"MAP error after {c['n_measurements']} measurements [m]")
        # row 2: signed downwind bias
        ax = axes[1, j]
        data = [np.array([r["filters"][nm]["bias_x_m"] for r in reps]) for nm in names]
        _box(ax, data, names, log=False)
        ax.axhline(0.0, color=INK, linewidth=0.8)
        lim = max(30.0, float(np.percentile(np.abs(np.concatenate(data)), 90)) * 1.3)
        ax.set_ylim(-lim, lim)
        ab = p["abs_bias_x_median_m"]
        ax.set_title(f"median |bias|: (ii) {ab['rbpf_grid']:.1f} m, (iii) {ab['kappa_fixed_high']:.1f} m, "
                     f"(iv) {ab['kappa_fixed_low']:.1f} m ({'PASS' if p['pass_bias'] else 'FAIL'})", fontsize=9.5)
        if j == 0:
            ax.set_ylabel("downwind bias (MAP - true) . e_x [m]")
        ax.text(0.02, 0.97, "points beyond the axis are clipped" if np.any(np.abs(np.concatenate(data)) > lim) else "",
                transform=ax.transAxes, fontsize=7.5, color=INK_MUTED, va="top")
        # row 3: kappa recovery of the RB-PF
        ax = axes[2, j]
        kt = np.array([r["kappa_true_over_ref"] for r in reps])
        km = np.array([r["filters"]["rbpf_grid"]["kappa_median_over_true"] for r in reps])
        q05 = np.array([r["filters"]["rbpf_grid"]["kappa_q05_q50_q95"][0] / r["kappa_true"] for r in reps])
        q95 = np.array([r["filters"]["rbpf_grid"]["kappa_q05_q50_q95"][2] / r["kappa_true"] for r in reps])
        ax.vlines(kt, q05, q95, color=FILTER_COLORS["rbpf_grid"], alpha=0.35, linewidth=1.2)
        ax.scatter(kt, km, s=22, color=FILTER_COLORS["rbpf_grid"], zorder=3, linewidths=0.0)
        for f, nm in zip(config.T1_2_KAPPA_FIXED_FACTORS, ("kappa_fixed_high", "kappa_fixed_low")):
            ax.axhline(f, color=FILTER_COLORS[nm], linewidth=1.0, linestyle=":")
            ax.text(10.0 ** (config.T1_2_KAPPA_TRUE_DECADES * 0.98), f, f"{FILTER_LABELS[nm]} ", color=FILTER_COLORS[nm],
                    fontsize=7.5, ha="right", va="bottom")
        ax.axhline(1.0, color=INK, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10.0 ** -config.T1_2_KAPPA_TRUE_DECADES / 1.3, 10.0 ** config.T1_2_KAPPA_TRUE_DECADES * 1.3)
        ax.set_ylim(0.05, 20.0)
        ax.set_xlabel("kappa_true / KAPPA_REF")
        s = c["summary"]["filters"]["rbpf_grid"]["kappa_median_over_true"]
        ax.set_title(f"RB-PF kappa recovery: median {s['median']:.2f} (IQR {s['q25']:.2f}-{s['q75']:.2f})", fontsize=9.5)
        ax.text(0.02, 0.04, "dot = posterior median, bar = 5-95 % interval", transform=ax.transAxes, fontsize=7.5,
                color=INK_MUTED, va="bottom")
        if j == 0:
            ax.set_ylabel("RB-PF (ii): posterior kappa median / kappa_true")
        ax.grid(True, color=GRID_COLOR, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"T1-2: kappa marginalisation vs fixed kappa (truth = the filter's own forward model, N = {config.PF_N_PARTICLES}, "
                 f"{cases[0]['n_measurements']} measurements per repeat, {len(cases[0]['repeats'])} repeats, "
                 f"kappa_true ~ KAPPA_REF x 10^[-{config.T1_2_KAPPA_TRUE_DECADES:g}, +{config.T1_2_KAPPA_TRUE_DECADES:g}], {lik})", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIG_DPI_FINAL)
    fig.savefig(path.with_name(path.stem + "_preview" + path.suffix), dpi=config.FIG_DPI_PREVIEW)
    plt.close(fig)


# ---------------------------------------------------------------------------------------- main
def _implied_kappa_context() -> dict:
    """kappa_ref implied by T1-3 (analytic) and T1-3b (adjoint), read from the cache JSONs when present."""
    out: dict = {"config_KAPPA_REF": config.KAPPA_REF, "grid_decades": config.KAPPA_GRID_DECADES, "n_grid": config.KAPPA_G}
    for key, fn in (("analytic_T1_3", "calibrate_forward.json"), ("adjoint_T1_3b", "calibrate_adjoint.json")):
        p = config.CACHE_DIR / fn
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out[key] = {"kappa_ref_implied": d["implied_kappa"]["kappa_ref_implied"],
                            "log10_implied_over_current": d["implied_kappa"]["log10_implied_over_current"],
                            "chosen": d.get("chosen")}
            except (KeyError, ValueError, OSError):
                out[key] = None
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-repeats", type=int, default=config.T1_2_N_REPEATS)
    ap.add_argument("--n-steps", type=int, default=config.T1_2_N_STEPS)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_kappa_bias.json")
    ap.add_argument("--fig", type=Path, default=config.FIG_DIR / "fig4_kappa_bias.png")
    ap.add_argument("--n-grid", type=int, default=config.KAPPA_G,
                    help="G of the RB-PF kappa grid (diagnostic: 61 = 0.1-decade spacing instead of the config 0.2)")
    ap.add_argument("--cases", type=str, default=None,
                    help="comma-separated model:source subset of config.T1_2_CASES, e.g. adjoint:109")
    ap.add_argument("--likelihood", choices=list(config.PF_LIKELIHOODS), default=config.PF_LIKELIHOOD,
                    help="count likelihood of every filter: 'poisson' (R3) or 'negbin' (Gamma-Poisson, R22 / R23; D7-1)")
    ap.add_argument("--nb-r", type=float, default=config.PF_NB_DISPERSION_R,
                    help="negative-binomial dispersion r (Var = lam + lam^2 / r) of --likelihood negbin")
    args = ap.parse_args(argv)
    cases_todo = list(config.T1_2_CASES)
    if args.cases:
        wanted = {tuple(c.split(":")) for c in args.cases.split(",")}
        cases_todo = [(m, s) for m, s in cases_todo if (m, str(s)) in wanted]
        if not cases_todo:
            raise SystemExit(f"--cases {args.cases!r} selects none of {config.T1_2_CASES}")

    t0 = time.perf_counter()
    wf = WindField.load()
    om = ObstacleMap.load()
    det = Detector()
    params = AdjointParams()
    op = AdvectionDiffusionOperator.from_data(params, wf, om).factorize()
    models: dict[str, ForwardModel] = {"analytic": GaussianPlume(), "adjoint": LbmAdjointModel(op)}
    setup_s = time.perf_counter() - t0

    cases: list[dict] = []
    for model_name, src in cases_todo:
        true_xy = np.asarray(config.SOURCES_XY[src], dtype=np.float64)
        paths, info = two_drone_paths(om, config.SOURCES_XY[src], args.n_steps)
        t1 = time.perf_counter()
        c = run_case(models[model_name], true_xy, paths, args.seed, args.n_repeats, det, om, n_grid=args.n_grid,
                     likelihood=args.likelihood, nb_r=args.nb_r)
        uv = wf.uv_at(true_xy.reshape(1, 2), config.DRONE_Z)[0]
        c.update({"model": model_name, "source": int(src), "path_info": info, "wall_seconds": time.perf_counter() - t1,
                  "lbm_wind_at_source_15m": {"u": float(uv[0]), "v": float(uv[1]), "speed": float(np.hypot(*uv)),
                                             "dir_deg": float(np.degrees(np.arctan2(uv[1], uv[0])))},
                  "min_path_distance_m": float(np.min(np.hypot(*(paths.reshape(-1, 2) - true_xy).T)))})
        cases.append(c)
        s, p = c["summary"], c["pass"]
        f = s["filters"]
        print(f"[{model_name} {src}] MAP error median (IQR) [m]: " + ", ".join(
            f"{FILTER_LABELS[nm]} {f[nm]['map_error_m']['median']:.1f} ({f[nm]['map_error_m']['q25']:.1f}-{f[nm]['map_error_m']['q75']:.1f})"
            for nm in FILTER_NAMES), flush=True)
        print(f"    bias_x median [m]: " + ", ".join(f"{nm} {f[nm]['bias_x_m']['median']:+.1f}" for nm in FILTER_NAMES)
              + f"; |MAP(ii)-MAP(i)| median {p['map_distance_ii_i_median_m']:.1f} m -> {'PASS' if p['pass_map_diff'] else 'FAIL'}; "
              f"bias (iii)/(iv) > (ii): {p['pass_bias_high']}/{p['pass_bias_low']} -> {'PASS' if p['pass_bias'] else 'FAIL'}; "
              f"RB-PF kappa med/true {f['rbpf_grid']['kappa_median_over_true']['median']:.2f}; "
              f"expected counts max median {s['expected_counts_max']['median']:.0f}; {c['wall_seconds']:.0f} s", flush=True)

    make_figure(cases, args.fig, likelihood=args.likelihood, nb_r=args.nb_r)
    overall_map = all(c["pass"]["pass_map_diff"] for c in cases)
    overall_bias = all(c["pass"]["pass_bias"] for c in cases)
    res = {
        "settings": {"seed": args.seed, "n_repeats": args.n_repeats, "n_steps_per_drone": args.n_steps,
                     "n_drones": config.PF_ADJ_N_DRONES, "n_measurements": int(args.n_steps * config.PF_ADJ_N_DRONES),
                     "n_particles": config.PF_N_PARTICLES, "eps_mix": config.PF_EPS_MIX, "n_grid": args.n_grid,
                     "likelihood": args.likelihood, "nb_r": args.nb_r,
                     "kappa_ref": config.KAPPA_REF, "grid_decades": config.KAPPA_GRID_DECADES,
                     "kappa_grid_spacing_decades": 2.0 * config.KAPPA_GRID_DECADES / (args.n_grid - 1),
                     "kappa_true_decades": config.T1_2_KAPPA_TRUE_DECADES, "kappa_fixed_factors": list(config.T1_2_KAPPA_FIXED_FACTORS),
                     "fixed_kappa_decades": config.T1_2_FIXED_KAPPA_DECADES, "report_measurements": list(config.T1_2_REPORT_MEASUREMENTS),
                     "background_cps": det.background, "T": det.T, "k0": det.k0, "z": config.DRONE_Z,
                     "sweep_width_m": config.PF_ADJ_SWEEP_WIDTH_M, "start_downwind_m": config.PF_ADJ_START_DOWNWIND_M,
                     "adjoint_params": {"K": params.K, "lam": params.lam, "h_layer": params.h_layer, "sigma0": params.sigma0},
                     "analytic_params": {"U": config.FWD_DEFAULT_U, "sigma_v": config.FWD_DEFAULT_SIGMA_V, "wind_mode": "global"},
                     "filters": {nm: FILTER_LABELS[nm] for nm in FILTER_NAMES}},
        "kappa_context": _implied_kappa_context(),
        "cases": cases,
        "overall": {"pass_map_diff": bool(overall_map), "pass_bias": bool(overall_bias), "pass": bool(overall_map and overall_bias),
                    "per_case": {f"{c['model']}_{c['source']}": {"pass_map_diff": c["pass"]["pass_map_diff"], "pass_bias": c["pass"]["pass_bias"]}
                                 for c in cases}},
        "setup_seconds": setup_s, "total_seconds": time.perf_counter() - t0, "figure": str(args.fig),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"T1-2 |MAP(ii) - MAP(i)| median < {config.T1_2_MAP_DIFF_PASS_M:.0f} m (all cases):", "PASS" if overall_map else "FAIL")
    print("T1-2 fixed-wrong kappa bias > RB-PF bias (all cases):", "PASS" if overall_bias else "FAIL")
    print(f"likelihood {args.likelihood}" + (f" (r = {args.nb_r:g})" if args.likelihood == "negbin" else "")
          + f"; total {res['total_seconds']:.0f} s; JSON {args.out}; figure {args.fig}")
    return res


if __name__ == "__main__":
    main()