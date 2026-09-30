"""T1-3b: shape-based calibration of the LBM adjoint forward model (option B-1) against the LDM slabs and
comparison with the analytic Gaussian plume of T1-3 (plan 4.2b 캘리브레이션, S1 T1-3b / T1-4 preparation; D5-1;
docs/lbm_forward_model.md section 6; slide 9, table 1, figure 3).

Usage: python -m srcloc_env.scripts.calibrate_adjoint [--index 599] [--out <CACHE_DIR>/calibrate_adjoint.json]
       [--fig-dir <FIG_DIR>] [--analytic <CACHE_DIR>/calibrate_forward.json]
       [--fields-out <CACHE_DIR>/adjoint_fields_chosen.npz]

Method (mirrors scripts/calibrate_forward.py so that the two models are directly comparable)
-----------------------------------------------------------------------------------------
truth   per-source airborne slab density n_LDM(x, y) [particles/m^3] of frame index config.T1_3_FRAME_INDEX
        (599 = step 30000) at z = config.DRONE_Z (LdmSlabBackend, docs/data_cache.md), 5 m grid (SlabGrid).
model   g(theta_s, p) = AdvectionDiffusionOperator.solve_forward(config.SOURCES_XY[s]) [particles/m^3 per
        particle/s] on the same grid, for every combination of K in config.ADJ_K_CANDIDATES x lam in
        config.ADJ_LAMBDA_CANDIDATES x wind layer in config.T1_3B_WIND_LAYERS (single 15 m wind, wind_band =
        None, vs the per-cell mean over the stored LBM levels in [10, 20) m; lbm_forward_model.md section 6).
        One assembly + LU factorisation per combination, 13 forward solves each.
cells   C_both = {n_LDM > 0 and g > g_min}, g_min = config.T1_3B_G_MIN (= T1_3_G_FLOOR_FACTOR x FWD_G_FLOOR, the
        floor rule of calibrate_forward; the PF floors g at FWD_G_FLOOR anyway).  The adjoint model has NO
        upwind / downwind notion (the wind field decides where the plume goes), so there is no d > 0 rule;
        instead the SUPPORT MISMATCH is reported: the fraction of the source's slab mass (and cells) in cells
        where g == 0 exactly (wall cells whose roof reaches 15 m, and free cells outside the source's connected
        component - "LDM present, model absent") and the fraction of the model mass (sum of g over g > 0) in
        cells where n_LDM == 0 ("model present, LDM absent").  Cells with 0 < g <= g_min are counted
        separately (n_floor_excluded, ldm_mass_frac_below_floor).
residual  rho = log(n_LDM / g), rho' = rho - median_s(rho) over C_both (plan T1-3); std / IQR / p90-p10 of rho',
        std_dense and p90-p10 over the dense cells n_LDM >= config.T1_3_INFO_DENSITY_FRACTION x max_s(n_LDM)
        (same threshold as the analytic run: above the single-particle fringe), the slab-mass-weighted std,
        n_cells and median_s(rho) (implied log kappa offset).
selection  argmin over the 40 rows of the mean over config.TRAIN_SOURCES of std_dense (config.T1_3B_SELECTION_KEY).
        RULE (D5-1 decision, validation_log "T1-3 FAIL의 해석"): the plain std over C_both is dominated by the
        single-particle fringe of the 5 m slab (76-98 % of the cells are below 0.02 particles/m^3) and therefore
        measures the fringe, not the plume shape; the dense-cell std is the shape statistic.  The row that
        minimises the plain std (and the mass-weighted std) is reported next to it.  Holdout sources
        (config.HOLDOUT_SOURCES) are evaluated everywhere but never used for the selection.
comparison  per source: analytic std / std_dense / upwind mass fraction at the analytic chosen combination
        (calibrate_forward.json, per_source_chosen) vs adjoint std / std_dense / support-mismatch fractions at
        the adjoint chosen combination -> table 1 (``table1_markdown``) and per-source verdicts
        "adjoint std_dense < analytic std_dense" with the counts over the open (config.T1_3_OPEN_SOURCES),
        trapped (config.T1_3_TRAPPED_SOURCES) and holdout groups (109 is open AND holdout).
pass    the plan's T1-3 criterion std(rho') < config.T1_3_STD_PASS on the open sources is evaluated on the plain
        std (expected to FAIL for the same fringe reason as the analytic model) and, informationally, on
        std_dense.
kappa   kappa_ref(implied) = config.SENSOR_K0 x exp(mean_{train} median_s rho) (plan 4.3 "κ_ref의 정의"), reported
        with the analytic value and config.KAPPA_REF; config is NOT changed here.  g includes the 1 / h_layer
        (config.ADJ_H_LAYER) conversion, so exp(m_bar) is "particles/s x layer-model scale", not a release rate.

Outputs
-------
config.CACHE_DIR / calibrate_adjoint.json   grid table (40 rows), chosen row, alternative argmins, per-source
        statistics at every row, table 1 markdown, verdicts, implied kappa, timing.
config.CACHE_DIR / adjoint_fields_chosen.npz   float32 fields (13, ny, nx) of the chosen combination (unit
        source, particles/m^3 per particle/s), keys fields, sources, K, lam, wind_band (empty array = None),
        wind_layer, z, h_layer, grid_x0, grid_y0, grid_res, grid_nx, grid_ny, index  (T1-4 / library use).
config.FIG_DIR / fig3_mismatch_adjoint.png (+ _preview)   2 x 2: (a) rho' map of 109, (b) rho' map of 110
        (diverging, clipped to +-config.T1_3_FIG_RHO_CLIP, occupied cells excluded from the statistic in grey),
        (c) / (d) adjoint model field g x exp(median_s rho) of 109 / 110 on the slab's own log10 scale
        (vmax = slab maximum, config.FIG_LOG_DECADES decades).  Building outlines from ObstacleMap.hmap > 0.
config.FIG_DIR / fig3_model_comparison.png (+ _preview)   per-source grouped bars of std_dense, analytic vs
        adjoint, coloured by source type (open / trapped / holdout / other train).

References: R15 (Taylor K), R17 (Keats et al. 2007 source-receptor on a CFD wind field), R21 (finite volumes).
The helpers ``shape_stats_adjoint``, ``evaluate_operator``, ``run_grid``, ``select_row``, ``table1_markdown``
and ``compare_models`` are pure numpy / scipy and unit-tested on synthetic grids in tests/test_calibrate_adjoint.py.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from srcloc_env import config  # noqa: E402
from srcloc_env.env.drone import ObstacleMap  # noqa: E402
from srcloc_env.field.concentration_field import LdmSlabBackend  # noqa: E402
from srcloc_env.field.wind import WindField  # noqa: E402
from srcloc_env.pf.lbm_adjoint import AdjointParams, AdvectionDiffusionOperator  # noqa: E402
from srcloc_env.preprocess.gridder import SlabGrid  # noqa: E402
from srcloc_env.scripts.calibrate_forward import implied_kappa, source_type  # noqa: E402

COLOR_ACCENT = "#d1495b"        # source star (fig_scene / calibrate_forward)
COLOR_EXCLUDED = "#d9d9d9"      # occupied cells excluded from the statistic (g == 0 or below the floor)
CMAP_RHO = "RdBu_r"             # diverging, neutral midpoint: red = slab above model, blue = slab below model
CMAP_DENSITY = LinearSegmentedColormap.from_list("blues_trunc", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
# categorical colours of the source types (fixed order; validated adjacent triple of the dataviz palette)
TYPE_COLORS = {"open": "#2a78d6", "trapped": "#eb6834", "holdout": "#1baf7a", "train": "#8a8984"}
ALT_KEYS = ("mean_train_std", "mean_train_mass_weighted_std", "mean_train_p90_p10_dense")   # informational argmins


# ------------------------------------------------------------------------------------------ statistics
@dataclass(frozen=True)
class AdjointShapeStats:
    """Per-source T1-3b statistics (module docstring).  NaN where no cell qualifies.

    n_cells              C_both cells used: n_LDM > 0 and g > g_min
    n_occupied           cells with n_LDM > 0
    n_model_positive     cells with g > 0 (the source's connected free component)
    n_floor_excluded     occupied cells with 0 < g <= g_min (dropped from the statistic)
    median_rho           median_s(rho) over C_both: implied log kappa offset (plan 4.3)
    std / iqr / p90_p10  of rho' = rho - median_rho over C_both (ddof = 0)
    n_dense / std_dense / p90_p10_dense   C_both cells with n_LDM >= dense_fraction x max_s(n_LDM) (SELECTION)
    mass_weighted_std    std of rho' over C_both weighted by n_LDM
    ldm_mass_frac_model_zero / ldm_cell_frac_model_zero   slab mass / occupied-cell fraction where g == 0
    ldm_mass_frac_below_floor                             slab mass fraction where 0 < g <= g_min
    model_mass_frac_ldm_zero / model_cell_frac_ldm_zero   model mass (sum g, g > 0) / cell fraction where n_LDM == 0
    """

    n_cells: int
    n_occupied: int
    n_model_positive: int
    n_floor_excluded: int
    median_rho: float
    std: float
    iqr: float
    p90_p10: float
    n_dense: int
    std_dense: float
    p90_p10_dense: float
    mass_weighted_std: float
    ldm_mass_frac_model_zero: float
    ldm_cell_frac_model_zero: float
    ldm_mass_frac_below_floor: float
    model_mass_frac_ldm_zero: float
    model_cell_frac_ldm_zero: float


def shape_stats_adjoint(n_ldm: np.ndarray, g: np.ndarray, g_min: float = config.T1_3B_G_MIN,
                        dense_fraction: float = config.T1_3_INFO_DENSITY_FRACTION,
                        ) -> tuple[AdjointShapeStats, np.ndarray]:
    """T1-3b shape residual of one source on M cells: (AdjointShapeStats, rho' (M,) with NaN where not in C_both).

    n_ldm (M,) slab density [particles/m^3] (>= 0); g (M,) unit response [particles/m^3 per particle/s]
    (>= 0, exactly 0 where the model has no support).  A slab that is a pure multiple of g on the common
    support has rho' = 0 everywhere (std = IQR = p90-p10 = std_dense = mass-weighted std = 0) and both
    support-mismatch fractions 0 when the supports coincide.  Shapes must match; negative values are an error.
    """
    n = np.asarray(n_ldm, dtype=np.float64).ravel()
    gg = np.asarray(g, dtype=np.float64).ravel()
    if n.shape != gg.shape:
        raise ValueError(f"n_ldm {n.shape} and g {gg.shape} must have the same length")
    if (n < 0.0).any() or (gg < 0.0).any():
        raise ValueError("n_ldm and g must be non-negative")
    occupied = n > 0.0
    model_pos = gg > 0.0
    above = gg > float(g_min)
    sel = occupied & above
    nan = float("nan")
    n_occ = int(occupied.sum())
    n_pos = int(model_pos.sum())
    mass = float(n[occupied].sum()) if n_occ else 0.0
    model_mass = float(gg[model_pos].sum()) if n_pos else 0.0
    ldm_zero = occupied & ~model_pos
    ldm_floor = occupied & model_pos & ~above
    mod_zero = model_pos & ~occupied
    l_mass0 = float(n[ldm_zero].sum() / mass) if mass > 0.0 else nan
    l_cell0 = float(ldm_zero.sum() / n_occ) if n_occ else nan
    l_floor = float(n[ldm_floor].sum() / mass) if mass > 0.0 else nan
    m_mass0 = float(gg[mod_zero].sum() / model_mass) if model_mass > 0.0 else nan
    m_cell0 = float(mod_zero.sum() / n_pos) if n_pos else nan
    rho_prime = np.full(n.shape, np.nan)
    if not sel.any():
        return AdjointShapeStats(0, n_occ, n_pos, int(ldm_floor.sum()), nan, nan, nan, nan, 0, nan, nan, nan,
                                 l_mass0, l_cell0, l_floor, m_mass0, m_cell0), rho_prime
    ns = n[sel]
    rho = np.log(ns / gg[sel])
    med = float(np.median(rho))
    rp = rho - med
    rho_prime[sel] = rp
    q10, q25, q75, q90 = np.percentile(rp, [10.0, 25.0, 75.0, 90.0])
    dense = ns >= float(dense_fraction) * float(n[occupied].max())          # threshold on the SOURCE maximum
    if dense.any():
        d10, d90 = np.percentile(rp[dense], [10.0, 90.0])
        std_dense, p_dense = float(rp[dense].std()), float(d90 - d10)
    else:
        std_dense, p_dense = nan, nan
    w = ns / ns.sum()
    mu = float((w * rp).sum())
    mw_std = float(np.sqrt((w * (rp - mu) ** 2).sum()))
    return AdjointShapeStats(int(sel.sum()), n_occ, n_pos, int(ldm_floor.sum()), med, float(rp.std()),
                             float(q75 - q25), float(q90 - q10), int(dense.sum()), std_dense, p_dense, mw_std,
                             l_mass0, l_cell0, l_floor, m_mass0, m_cell0), rho_prime

# ------------------------------------------------------------------------------------------ grid search
def evaluate_operator(op: AdvectionDiffusionOperator, src_xy: np.ndarray, slabs: np.ndarray,
                      ) -> tuple[list[AdjointShapeStats], np.ndarray, np.ndarray]:
    """Statistics of n sources on one factorised operator: (stats list, rho' (n, M), g (n, M)), M = ny x nx.

    src_xy (n, 2) source positions (config.SOURCES_XY), slabs (n, M) truth densities in the grid's row-major
    (iy, ix) order; g[i] = op.solve_forward(src_xy[i]).ravel().
    """
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    M = op.grid.ny * op.grid.nx
    sl = np.asarray(slabs, dtype=np.float64).reshape(src.shape[0], -1)
    if sl.shape[1] != M:
        raise ValueError(f"slabs {sl.shape} must be (n_sources, M = {M})")
    g = np.empty((src.shape[0], M))
    rho_prime = np.empty_like(g)
    stats: list[AdjointShapeStats] = []
    for i in range(src.shape[0]):
        g[i] = op.solve_forward(src[i]).ravel()
        st, rp = shape_stats_adjoint(sl[i], g[i])
        stats.append(st)
        rho_prime[i] = rp
    return stats, rho_prime, g


def _mean(stats: list[AdjointShapeStats], attr: str) -> float:
    return float(np.mean([getattr(s, attr) for s in stats]))


def run_grid(grid: SlabGrid, uv_by_layer: dict[str, np.ndarray], blocked: np.ndarray, src_ids: tuple[int, ...],
             src_xy: np.ndarray, slabs: np.ndarray, train_ids: tuple[int, ...] = config.TRAIN_SOURCES,
             layers: dict[str, tuple[float, float] | None] = config.T1_3B_WIND_LAYERS,
             k_candidates: tuple[float, ...] = config.ADJ_K_CANDIDATES,
             lam_candidates: tuple[float, ...] = config.ADJ_LAMBDA_CANDIDATES,
             h_layer: float = config.ADJ_H_LAYER, z: float = config.DRONE_Z) -> list[dict]:
    """One row per (wind layer, K, lam), layer-major then K then lam: per-source stats + mean-over-train summaries.

    uv_by_layer[name] is the (ny, nx, 2) wind of ``layers[name]`` (AdvectionDiffusionOperator.wind_at_cells);
    the operator is assembled with from_arrays so the wind interpolation is done once per layer, not per row.
    """
    train_idx = [i for i, s in enumerate(src_ids) if s in train_ids]
    if not train_idx:
        raise ValueError("no training source among src_ids")
    missing = set(layers) - set(uv_by_layer)
    if missing:
        raise ValueError(f"uv_by_layer lacks the wind of layer(s) {sorted(missing)}")
    rows: list[dict] = []
    for name, K, lam in product(layers, k_candidates, lam_candidates):
        params = AdjointParams(K=float(K), lam=float(lam), h_layer=float(h_layer), z=float(z), wind_band=layers[name])
        t0 = time.perf_counter()
        op = AdvectionDiffusionOperator.from_arrays(grid, uv_by_layer[name], blocked, params).factorize()
        t1 = time.perf_counter()
        stats, _, _ = evaluate_operator(op, src_xy, slabs)
        t2 = time.perf_counter()
        tr = [stats[i] for i in train_idx]
        rows.append({
            "wind_layer": name, "wind_band": None if layers[name] is None else list(layers[name]),
            "K": float(K), "lam": float(lam),
            "mean_train_std_dense": _mean(tr, "std_dense"),
            "mean_train_std": _mean(tr, "std"),
            "mean_train_iqr": _mean(tr, "iqr"),
            "mean_train_p90_p10": _mean(tr, "p90_p10"),
            "mean_train_p90_p10_dense": _mean(tr, "p90_p10_dense"),
            "mean_train_mass_weighted_std": _mean(tr, "mass_weighted_std"),
            "mean_train_median_rho": _mean(tr, "median_rho"),
            "mean_train_ldm_mass_frac_model_zero": _mean(tr, "ldm_mass_frac_model_zero"),
            "mean_train_model_mass_frac_ldm_zero": _mean(tr, "model_mass_frac_ldm_zero"),
            "n_train": len(tr), "n_free": int(op.n_free),
            "seconds_assemble_factorize": t1 - t0, "seconds_solves": t2 - t1,
            "per_source": {str(s): asdict(st) for s, st in zip(src_ids, stats)},
        })
    return rows


def select_row(rows: list[dict], key: str = config.T1_3B_SELECTION_KEY) -> tuple[int, dict]:
    """Index of the row minimising ``key`` (NaN never wins; first minimum on ties) and the spread of ``key``
    over all rows / within the chosen wind layer (discriminability, plan S1 위험·대안)."""
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    if not np.isfinite(vals).any():
        raise ValueError(f"no finite {key} in the grid")
    k = int(np.nanargmin(vals))
    same = np.array([v for r, v in zip(rows, vals) if r["wind_layer"] == rows[k]["wind_layer"]])
    return k, {"key": key, "value": float(vals[k]), "spread_all_rows": float(np.nanmax(vals) - np.nanmin(vals)),
               "spread_chosen_layer": float(np.nanmax(same) - np.nanmin(same)),
               "discriminable": bool(np.nanmax(same) - np.nanmin(same) >= config.T1_3_MIN_DISCRIMINABILITY),
               "threshold": config.T1_3_MIN_DISCRIMINABILITY}


# ------------------------------------------------------------------------------------------ comparison
def compare_models(src_ids: tuple[int, ...], analytic_ps: dict[str, dict], adjoint_ps: dict[str, dict],
                   open_ids: tuple[int, ...] = config.T1_3_OPEN_SOURCES,
                   trapped_ids: tuple[int, ...] = config.T1_3_TRAPPED_SOURCES,
                   holdout_ids: tuple[int, ...] = config.HOLDOUT_SOURCES) -> dict:
    """Per-source verdicts adjoint < analytic on std_dense (and std) and the counts per group.

    analytic_ps / adjoint_ps: per_source dicts keyed by str(source) with 'std' and 'std_dense'.  A NaN on
    either side counts as "not better".  Groups may overlap (109 is open and holdout).
    """
    per = {}
    for s in src_ids:
        a, b = analytic_ps[str(s)], adjoint_ps[str(s)]
        per[str(s)] = {
            "analytic_std_dense": a["std_dense"], "adjoint_std_dense": b["std_dense"],
            "adjoint_better_std_dense": bool(np.isfinite(a["std_dense"]) and np.isfinite(b["std_dense"])
                                             and b["std_dense"] < a["std_dense"]),
            "analytic_std": a["std"], "adjoint_std": b["std"],
            "adjoint_better_std": bool(np.isfinite(a["std"]) and np.isfinite(b["std"]) and b["std"] < a["std"]),
            "delta_std_dense": float(b["std_dense"] - a["std_dense"]),
        }
    groups = {"open": open_ids, "trapped": trapped_ids, "holdout": holdout_ids, "all": src_ids}
    counts = {}
    for name, ids in groups.items():
        ids = tuple(s for s in ids if str(s) in per)
        counts[name] = {"n": len(ids),
                        "adjoint_better_std_dense": int(sum(per[str(s)]["adjoint_better_std_dense"] for s in ids)),
                        "adjoint_better_std": int(sum(per[str(s)]["adjoint_better_std"] for s in ids)),
                        "sources": list(ids)}
    return {"per_source": per, "counts": counts}


def table1_markdown(src_ids: tuple[int, ...], analytic_ps: dict[str, dict], adjoint_ps: dict[str, dict],
                    caption: str = "") -> str:
    """표 1: source | type | analytic std | analytic std_dense | adjoint std | adjoint std_dense |
    adjoint LDM-mass-where-model-0 | adjoint model-mass-where-LDM-0 | n_cells (adjoint C_both)."""
    lines = []
    if caption:
        lines += [caption, ""]
    lines.append("| source | type | analytic std | analytic std_dense | adjoint std | adjoint std_dense | "
                 "adjoint LDM-mass-where-model-0 | adjoint model-mass-where-LDM-0 | n_cells |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in src_ids:
        a, b = analytic_ps[str(s)], adjoint_ps[str(s)]
        lines.append(f"| {s} | {source_type(s)} | {a['std']:.2f} | {a['std_dense']:.2f} | {b['std']:.2f} | "
                     f"{b['std_dense']:.2f} | {b['ldm_mass_frac_model_zero']:.3f} | {b['model_mass_frac_ldm_zero']:.3f} | "
                     f"{b['n_cells']} |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ figures
def _window(xs: float, ys: float, grid: SlabGrid) -> tuple[float, float, float, float]:
    """Same +-window as figure 3 of the analytic run (config.T1_3_FIG_WINDOW_M) for side-by-side comparison."""
    up, down, cross = config.T1_3_FIG_WINDOW_M
    x_lo, x_hi = grid.x0, grid.x0 + grid.nx * grid.res
    y_lo, y_hi = grid.y0, grid.y0 + grid.ny * grid.res
    return max(xs - up, x_lo), min(xs + down, x_hi), max(ys - cross, y_lo), min(ys + cross, y_hi)


def render_figure3_adjoint(grid: SlabGrid, omap: ObstacleMap, panels: dict[int, dict], combo: dict, out_png: Path,
                           out_preview: Path, index: int) -> dict:
    """2 x 2: (a) rho' 109, (b) rho' 110, (c) model field 109, (d) model field 110 (module docstring).

    panels[s] = {"rho_prime", "slab", "model" (= g x exp(median rho)), "occupied" (ny, nx); "stats" dict;
    "wind": {"speed", "dir_deg"} of the chosen layer's wind at the source cell}.
    """
    s_open, s_trap = config.T1_3_FIG_SOURCES
    extent = (grid.x0, grid.x0 + grid.nx * grid.res, grid.y0, grid.y0 + grid.ny * grid.res)
    clip = config.T1_3_FIG_RHO_CLIP
    occ_img = (omap.hmap > config.OCC_HMAP_NO_BUILDING).T.astype(np.float32)
    band = combo["wind_band"]
    layer_txt = (f"{combo['wind_layer']} (z = {config.DRONE_Z:g} m)" if band is None
                 else f"{combo['wind_layer']} ({band[0]:g}-{band[1]:g} m mean)")
    combo_txt = f"K = {combo['K']:g} m$^2$/s, lambda = {combo['lam']:g} 1/s, wind layer {layer_txt}"

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
    numbers: dict = {}

    def base(ax, s: int, title: str) -> None:
        xs, ys = config.SOURCES_XY[s]
        ax.contour(omap.x, omap.y, occ_img, levels=[0.5], colors="black", linewidths=0.6, zorder=4)
        ax.scatter([xs], [ys], marker="*", s=200, c=COLOR_ACCENT, edgecolors="white", linewidths=0.8, zorder=6)
        ax.text(xs - 8, ys + 8, str(s), fontsize=9, ha="right", va="bottom", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        w = panels[s]["wind"]
        a = np.deg2rad(w["dir_deg"])
        L = config.T1_3_FIG_ARROW_M
        ax.annotate("", xy=(xs + L * np.cos(a), ys + L * np.sin(a)), xytext=(xs, ys), zorder=8,
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black", shrinkA=0, shrinkB=0))
        ax.text(xs + 0.5 * L * np.cos(a), ys + 0.5 * L * np.sin(a) - 14,
                f"LBM wind at source {w['speed']:.2f} m/s, {w['dir_deg']:.0f} deg", fontsize=7.5, ha="center",
                va="top", zorder=8, path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        x0, x1, y0, y1 = _window(xs, ys, grid)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title, fontsize=9.5)

    for ax, s, lab in ((axes[0, 0], s_open, "(a)"), (axes[0, 1], s_trap, "(b)")):
        p = panels[s]
        st = p["stats"]
        excl = np.ma.masked_where(~(p["occupied"] & np.isnan(p["rho_prime"])), np.ones_like(p["rho_prime"]))
        ax.imshow(excl, extent=extent, origin="lower", cmap=ListedColormap([COLOR_EXCLUDED]), vmin=0, vmax=1,
                  interpolation="nearest", zorder=1)
        im = ax.imshow(np.ma.masked_invalid(np.clip(p["rho_prime"], -clip, clip)), extent=extent, origin="lower",
                       cmap=CMAP_RHO, vmin=-clip, vmax=clip, interpolation="nearest", zorder=2)
        kind = "open" if s == s_open else "trapped"
        base(ax, s, f"{lab} rho' = log(n_LDM / g) - median, source {s} ({kind}), adjoint model\nstd {st['std']:.2f} "
                    f"(dense {st['std_dense']:.2f}, n {st['n_dense']}), n {st['n_cells']}, LDM mass at g = 0: "
                    f"{st['ldm_mass_frac_model_zero']:.2f}\ngrey = occupied cells excluded (g = 0 or below the floor)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(f"rho' (clipped to +-{clip:g})")
        finite = p["rho_prime"][np.isfinite(p["rho_prime"])]
        numbers[str(s)] = {"n_shown": int(finite.size), "n_clipped": int((np.abs(finite) > clip).sum())}

    for ax, s, lab in ((axes[1, 0], s_open, "(c)"), (axes[1, 1], s_trap, "(d)")):
        slab, model = panels[s]["slab"], panels[s]["model"]
        pos = slab[slab > 0]
        vmax = float(pos.max())
        vmin = max(float(pos.min()), vmax / 10.0 ** config.FIG_LOG_DECADES)
        im = ax.imshow(np.ma.masked_where(model < vmin, model), extent=extent, origin="lower", cmap=CMAP_DENSITY,
                       norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest", zorder=2)
        base(ax, s, f"{lab} source {s}: adjoint model g x exp(median rho)\n(LDM slab colour scale of source {s}: "
                    f"vmax = slab max, {config.FIG_LOG_DECADES:g} decades)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("particles / m$^3$ (log10 scale)")
        numbers[f"density_log_vmin_{s}"] = vmin
        numbers[f"density_log_vmax_{s}"] = vmax
    fig.suptitle(f"Figure 3 (adjoint, draft): LBM adjoint model mismatch at the chosen T1-3b combination ({combo_txt}), "
                 f"frame {index} = step {config.index_to_step(index)}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=config.FIG_DPI_FINAL, bbox_inches="tight")
    fig.savefig(out_preview, dpi=config.FIG_DPI_PREVIEW, bbox_inches="tight")
    plt.close(fig)
    numbers["files"] = {p.name: {"path": str(p), "bytes": p.stat().st_size} for p in (out_png, out_preview)}
    return numbers


def _type_key(s: int) -> str:
    """Colour key of a source: holdout wins over open (109) for the bar colouring; other train sources 'train'."""
    if s in config.HOLDOUT_SOURCES:
        return "holdout"
    if s in config.T1_3_TRAPPED_SOURCES:
        return "trapped"
    if s in config.T1_3_OPEN_SOURCES:
        return "open"
    return "train"


def render_comparison(src_ids: tuple[int, ...], analytic_ps: dict[str, dict], adjoint_ps: dict[str, dict],
                      analytic_combo: dict, adjoint_combo: dict, out_png: Path, out_preview: Path) -> dict:
    """Grouped bars of std_dense per source (analytic hatched / adjoint solid), colour = source type."""
    n = len(src_ids)
    x = np.arange(n)
    wbar = 0.38
    a_vals = np.array([analytic_ps[str(s)]["std_dense"] for s in src_ids], dtype=np.float64)
    b_vals = np.array([adjoint_ps[str(s)]["std_dense"] for s in src_ids], dtype=np.float64)
    cols = [TYPE_COLORS[_type_key(s)] for s in src_ids]
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.bar(x - wbar / 2, np.nan_to_num(a_vals), wbar, color="white", edgecolor=cols, hatch="////", linewidth=1.2,
           label="analytic plume (T1-3)", zorder=3)
    ax.bar(x + wbar / 2, np.nan_to_num(b_vals), wbar, color=cols, edgecolor="white", linewidth=1.0,
           label="LBM adjoint (T1-3b)", zorder=3)
    for xi, (va, vb) in enumerate(zip(a_vals, b_vals)):
        if np.isfinite(va):
            ax.text(xi - wbar / 2, va + 0.04, f"{va:.2f}", ha="center", va="bottom", fontsize=7, color="#333333")
        if np.isfinite(vb):
            ax.text(xi + wbar / 2, vb + 0.04, f"{vb:.2f}", ha="center", va="bottom", fontsize=7, color="#333333")
    ax.axhline(config.T1_3_STD_PASS, color="#777777", lw=1.0, ls="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{source_type(s)}" for s in src_ids], fontsize=8)
    ax.set_ylabel("std(rho') on dense cells (n_LDM >= 1 % of source max)")
    ax.set_xlabel("LDM source")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor="white", edgecolor="#555555", hatch="////", label="analytic plume "
                     f"(U {analytic_combo['U']:g} m/s, sigma_v {analytic_combo['sigma_v']:g} m/s, {analytic_combo['wind_mode']})"),
               Patch(facecolor="#555555", label=f"LBM adjoint (K {adjoint_combo['K']:g} m$^2$/s, lambda {adjoint_combo['lam']:g} 1/s, "
                     f"{adjoint_combo['wind_layer']})")]
    handles += [Patch(facecolor=TYPE_COLORS[k], label=f"{k} source") for k in ("open", "trapped", "holdout", "train")]
    handles.append(Line2D([0], [0], color="#777777", lw=1.0, ls="--", label=f"plan criterion std(rho') < {config.T1_3_STD_PASS:g}"))
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", ncol=3, frameon=False)
    ax.set_title(f"Shape residual per source, frame {config.T1_3_FRAME_INDEX} (step {config.index_to_step(config.T1_3_FRAME_INDEX)}), "
                 f"z = {config.DRONE_Z:g} m: analytic plume vs LBM adjoint model", fontsize=10)
    ax.set_ylim(0, float(np.nanmax(np.r_[a_vals, b_vals])) * 1.4)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=config.FIG_DPI_FINAL, bbox_inches="tight")
    fig.savefig(out_preview, dpi=config.FIG_DPI_PREVIEW, bbox_inches="tight")
    plt.close(fig)
    return {"files": {p.name: {"path": str(p), "bytes": p.stat().st_size} for p in (out_png, out_preview)}}

# ------------------------------------------------------------------------------------------ main
def _wind_at_source(uv: np.ndarray, grid: SlabGrid, s: int) -> dict:
    """Wind (u, v, speed, direction) of a (ny, nx, 2) layer wind at the cell containing source s (annotation)."""
    xs, ys = config.SOURCES_XY[s]
    ix = int(np.clip(np.floor((xs - grid.x0) / grid.res), 0, grid.nx - 1))
    iy = int(np.clip(np.floor((ys - grid.y0) / grid.res), 0, grid.ny - 1))
    u, v = uv[iy, ix]
    return {"u": float(u), "v": float(v), "speed": float(np.hypot(u, v)), "dir_deg": float(np.rad2deg(np.arctan2(v, u)))}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=int, default=config.T1_3_FRAME_INDEX)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "calibrate_adjoint.json")
    ap.add_argument("--fields-out", type=Path, default=config.CACHE_DIR / "adjoint_fields_chosen.npz")
    ap.add_argument("--analytic", type=Path, default=config.CACHE_DIR / "calibrate_forward.json")
    ap.add_argument("--fig-dir", type=Path, default=config.FIG_DIR)
    args = ap.parse_args(argv)
    t_start = time.perf_counter()

    # ---- truth, wind, obstacles -------------------------------------------------------------------
    be = LdmSlabBackend(config.CACHE_DIR, max_cached_frames=config.FIELD_MAX_CACHED_FRAMES)
    sf = be.slab(args.index)
    grid = sf.grid
    zi = sf.z_index(config.DRONE_Z)
    src_ids = tuple(int(s) for s in sf.sources)
    src_xy = np.array([config.SOURCES_XY[s] for s in src_ids], dtype=np.float64)
    slabs = sf.density[:, zi].reshape(len(src_ids), -1).astype(np.float64)     # (n_src, M) row-major (iy, ix)
    t0 = time.perf_counter()
    wf = WindField.load()
    om = ObstacleMap.load()
    uv_by_layer = {name: AdvectionDiffusionOperator.wind_at_cells(wf, grid, AdjointParams(wind_band=band))
                   for name, band in config.T1_3B_WIND_LAYERS.items()}
    blocked = AdvectionDiffusionOperator.blocked_at_cells(om, grid, config.DRONE_Z)
    data_s = time.perf_counter() - t0

    # ---- grid search ------------------------------------------------------------------------------
    t0 = time.perf_counter()
    rows = run_grid(grid, uv_by_layer, blocked, src_ids, src_xy, slabs)
    grid_s = time.perf_counter() - t0
    k, spread = select_row(rows)
    chosen = rows[k]
    combo = {"wind_layer": chosen["wind_layer"], "wind_band": chosen["wind_band"], "K": chosen["K"], "lam": chosen["lam"],
             "row_index": k}
    alternatives = {}
    for key in ALT_KEYS:
        j, _ = select_row(rows, key)
        alternatives[key] = {"row_index": j, "wind_layer": rows[j]["wind_layer"], "K": rows[j]["K"], "lam": rows[j]["lam"],
                             "value": rows[j][key], "same_as_selection": bool(j == k)}
    best_per_layer = {}
    for name in config.T1_3B_WIND_LAYERS:
        sub = [(i, r) for i, r in enumerate(rows) if r["wind_layer"] == name]
        j = min(sub, key=lambda ir: (np.nan_to_num(ir[1][config.T1_3B_SELECTION_KEY], nan=np.inf), ir[0]))[0]
        best_per_layer[name] = {"row_index": j, "K": rows[j]["K"], "lam": rows[j]["lam"],
                                config.T1_3B_SELECTION_KEY: rows[j][config.T1_3B_SELECTION_KEY],
                                "mean_train_std": rows[j]["mean_train_std"]}

    # ---- chosen combination: fields, figure panels, npz -------------------------------------------
    params = AdjointParams(K=combo["K"], lam=combo["lam"],
                           wind_band=None if combo["wind_band"] is None else tuple(combo["wind_band"]))
    op = AdvectionDiffusionOperator.from_arrays(grid, uv_by_layer[combo["wind_layer"]], blocked, params).factorize()
    stats_c, rp_c, g_c = evaluate_operator(op, src_xy, slabs)
    ps = {str(s): asdict(st) for s, st in zip(src_ids, stats_c)}
    balance = {str(s): op.mass_balance(g_c[i].reshape(grid.ny, grid.nx)) for i, s in enumerate(src_ids)}
    fields = g_c.reshape(len(src_ids), grid.ny, grid.nx).astype(np.float32)
    args.fields_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fields_out, fields=fields, sources=np.array(src_ids, dtype=np.int64),
                        K=np.float64(combo["K"]), lam=np.float64(combo["lam"]),
                        wind_band=np.array([] if combo["wind_band"] is None else combo["wind_band"], dtype=np.float64),
                        wind_layer=np.array(combo["wind_layer"]), z=np.float64(config.DRONE_Z),
                        h_layer=np.float64(params.h_layer), grid_x0=np.float64(grid.x0), grid_y0=np.float64(grid.y0),
                        grid_res=np.float64(grid.res), grid_nx=np.int64(grid.nx), grid_ny=np.int64(grid.ny),
                        index=np.int64(args.index))

    # ---- analytic comparison, verdicts, pass, kappa -----------------------------------------------
    ana = json.loads(args.analytic.read_text(encoding="utf-8"))
    ana_ps = ana["per_source_chosen"]
    ana_combo = ana["chosen"]
    comparison = compare_models(src_ids, ana_ps, ps)
    open_std = {str(s): ps[str(s)]["std"] for s in config.T1_3_OPEN_SOURCES}
    open_std_dense = {str(s): ps[str(s)]["std_dense"] for s in config.T1_3_OPEN_SOURCES}
    passed = all(np.isfinite(v) and v < config.T1_3_STD_PASS for v in open_std.values())
    passed_dense = all(np.isfinite(v) and v < config.T1_3_STD_PASS for v in open_std_dense.values())
    kappa = implied_kappa([ps[str(s)]["median_rho"] for s in config.TRAIN_SOURCES])
    kappa.update({
        "train_sources": list(config.TRAIN_SOURCES),
        "analytic_kappa_ref_implied": ana["implied_kappa"]["kappa_ref_implied"],
        "analytic_m_bar": ana["implied_kappa"]["m_bar"],
        "current_config_KAPPA_REF": config.KAPPA_REF,
        "log10_implied_over_current": float(np.log10(kappa["kappa_ref_implied"] / config.KAPPA_REF)),
        "log10_implied_over_analytic": float(np.log10(kappa["kappa_ref_implied"] / ana["implied_kappa"]["kappa_ref_implied"])),
        "h_layer_m": params.h_layer,
        "per_source_kappa_all": {str(s): config.SENSOR_K0 * float(np.exp(ps[str(s)]["median_rho"])) for s in src_ids},
        "mean_train_median_rho_range_over_grid": [float(min(r["mean_train_median_rho"] for r in rows)),
                                                  float(max(r["mean_train_median_rho"] for r in rows))],
        "note": "config.KAPPA_REF is NOT changed by this script; g includes 1/h_layer so exp(m_bar) is a layer-model "
                "scale, not a release rate; m_bar depends on (K, lam, layer) (see mean_train_median_rho_range_over_grid).",
    })
    cnt = comparison["counts"]
    caption = (f"표 1 (T1-3 vs T1-3b, frame {args.index}, z = {config.DRONE_Z:g} m): analytic chosen {ana_combo['wind_mode']} "
               f"U = {ana_combo['U']:g} m/s, sigma_v = {ana_combo['sigma_v']:g} m/s; adjoint chosen K = {combo['K']:g} m^2/s, "
               f"lambda = {combo['lam']:g} 1/s, wind layer {combo['wind_layer']} (selection: min mean_train std_dense = "
               f"{chosen['mean_train_std_dense']:.3f}; plain std {chosen['mean_train_std']:.3f}); implied kappa_ref adjoint "
               f"{kappa['kappa_ref_implied']:.3e} vs analytic {kappa['analytic_kappa_ref_implied']:.3e}; "
               f"adjoint std_dense < analytic: open {cnt['open']['adjoint_better_std_dense']}/{cnt['open']['n']}, "
               f"trapped {cnt['trapped']['adjoint_better_std_dense']}/{cnt['trapped']['n']}, "
               f"holdout {cnt['holdout']['adjoint_better_std_dense']}/{cnt['holdout']['n']}; "
               f"plan PASS (open std < {config.T1_3_STD_PASS:g}) = {passed}")
    table_md = table1_markdown(src_ids, ana_ps, ps, caption)

    # ---- figures ----------------------------------------------------------------------------------
    panels = {}
    for s in config.T1_3_FIG_SOURCES:
        i = src_ids.index(s)
        st = stats_c[i]
        panels[s] = {"rho_prime": rp_c[i].reshape(grid.ny, grid.nx), "slab": slabs[i].reshape(grid.ny, grid.nx),
                     "model": (g_c[i] * np.exp(st.median_rho)).reshape(grid.ny, grid.nx),
                     "occupied": (slabs[i] > 0).reshape(grid.ny, grid.nx), "stats": asdict(st),
                     "wind": _wind_at_source(uv_by_layer[combo["wind_layer"]], grid, s)}
    f3 = args.fig_dir / "fig3_mismatch_adjoint.png"
    f3p = args.fig_dir / "fig3_mismatch_adjoint_preview.png"
    fig3_numbers = render_figure3_adjoint(grid, om, panels, combo, f3, f3p, args.index)
    fc = args.fig_dir / "fig3_model_comparison.png"
    fcp = args.fig_dir / "fig3_model_comparison_preview.png"
    figc_numbers = render_comparison(src_ids, ana_ps, ps, ana_combo, combo, fc, fcp)

    grid_table = [{k2: v for k2, v in r.items() if k2 != "per_source"} for r in rows]
    res = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index": args.index, "step": config.index_to_step(args.index), "z": config.DRONE_Z,
        "grid": grid.to_dict(), "n_cells_grid": int(grid.ny * grid.nx), "n_free": int(op.n_free),
        "sources": list(src_ids), "train_sources": list(config.TRAIN_SOURCES), "holdout_sources": list(config.HOLDOUT_SOURCES),
        "open_sources": list(config.T1_3_OPEN_SOURCES), "trapped_sources": list(config.T1_3_TRAPPED_SOURCES),
        "candidates": {"K": list(config.ADJ_K_CANDIDATES), "lam": list(config.ADJ_LAMBDA_CANDIDATES),
                       "wind_layers": {k2: (None if v is None else list(v)) for k2, v in config.T1_3B_WIND_LAYERS.items()}},
        "selection_rule": f"argmin over rows of {config.T1_3B_SELECTION_KEY} (mean over TRAIN_SOURCES of std(rho') on cells "
                          f"n_LDM >= {config.T1_3_INFO_DENSITY_FRACTION:g} x source max within C_both = n_LDM > 0 & g > "
                          f"{config.T1_3B_G_MIN:g}); plain-std argmin reported in alternative_argmins",
        "g_min": config.T1_3B_G_MIN, "dense_fraction": config.T1_3_INFO_DENSITY_FRACTION,
        "grid_table": grid_table,
        "chosen": combo, "chosen_row_summary": {k2: v for k2, v in chosen.items() if k2 != "per_source"},
        "discriminability": spread,
        "alternative_argmins": alternatives,
        "best_per_wind_layer": best_per_layer,
        "per_source_chosen": ps,
        "per_source_type": {str(s): source_type(s) for s in src_ids},
        "mass_balance_chosen": balance,
        "analytic": {"file": str(args.analytic), "chosen": ana_combo, "per_source_chosen": ana_ps,
                     "mean_train_std": ana["chosen_row"]["mean_train_std"],
                     "mean_train_std_dense": ana["chosen_row"]["mean_train_std_dense"]},
        "comparison": comparison,
        "open_source_std": open_std, "pass_criterion_std": config.T1_3_STD_PASS, "pass": bool(passed),
        "open_source_std_dense": open_std_dense, "pass_dense_informational": bool(passed_dense),
        "implied_kappa": kappa,
        "table1_markdown": table_md,
        "figure3_adjoint": fig3_numbers, "figure_comparison": figc_numbers,
        "fields_npz": {"path": str(args.fields_out), "shape": list(fields.shape), "dtype": "float32",
                       "bytes": args.fields_out.stat().st_size},
        "all_rows": rows,
        "timing": {"data_load_s": data_s, "grid_search_s": grid_s, "n_rows": len(rows),
                   "mean_row_s": grid_s / len(rows), "total_s": time.perf_counter() - t_start},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"T1-3b grid ({len(rows)} rows): mean_train std_dense [std, mass-weighted std, p90-p10 dense]; "
          "LDM mass where g=0; mean median rho")
    for i, r in enumerate(rows):
        mark = " <-- chosen" if i == k else ""
        print(f"  {r['wind_layer']:12s} K {r['K']:4.1f} lam {r['lam']:5.3f}: {r['mean_train_std_dense']:.3f} "
              f"[{r['mean_train_std']:.3f}, {r['mean_train_mass_weighted_std']:.3f}, {r['mean_train_p90_p10_dense']:.3f}]; "
              f"{r['mean_train_ldm_mass_frac_model_zero']:.3f}; {r['mean_train_median_rho']:.2f}{mark}")
    print(table_md)
    print("alternative argmins: " + ", ".join(
        f"{key}: {a['wind_layer']} K {a['K']:g} lam {a['lam']:g} ({'same' if a['same_as_selection'] else 'DIFFERENT'})"
        for key, a in alternatives.items()))
    print(f"discriminability spread (chosen layer) {spread['spread_chosen_layer']:.3f}, all rows {spread['spread_all_rows']:.3f}")
    print("adjoint std_dense < analytic std_dense: " + ", ".join(
        f"{g2} {cnt[g2]['adjoint_better_std_dense']}/{cnt[g2]['n']}" for g2 in ("open", "trapped", "holdout", "all")))
    print(f"implied kappa_ref adjoint {kappa['kappa_ref_implied']:.4e} (m_bar {kappa['m_bar']:.2f}) vs analytic "
          f"{kappa['analytic_kappa_ref_implied']:.4e}; current KAPPA_REF {config.KAPPA_REF:.4e}")
    print(f"T1-3 criterion std(rho') < {config.T1_3_STD_PASS} on open sources: ", "PASS" if passed else "FAIL",
          {k2: round(v, 3) for k2, v in open_std.items()}, "| dense: ", "PASS" if passed_dense else "FAIL",
          {k2: round(v, 3) for k2, v in open_std_dense.items()})
    print(f"elapsed {res['timing']['total_s']:.1f} s (grid {grid_s:.1f} s); wrote {args.out}, {args.fields_out}, {f3}, {fc}")
    return res


if __name__ == "__main__":
    main()