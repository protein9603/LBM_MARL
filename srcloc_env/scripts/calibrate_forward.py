"""T1-3: shape-based calibration of the analytic Gaussian plume against the LDM slabs (plan S1 T1-3 / 4.2
캘리브레이션 규칙, D4-3; slide 9, table 1 draft, figure 3 draft).

Usage: python -m srcloc_env.scripts.calibrate_forward [--index 599] [--out <CACHE_DIR>/calibrate_forward.json]
       [--fig-dir <FIG_DIR>]

Method (plan T1-3, followed exactly)
------------------------------------
truth   per-source airborne slab density n_LDM(x, y) [particles/m^3] of frame index config.T1_3_FRAME_INDEX
        (599 = step 30000) at z = config.DRONE_Z (LdmSlabBackend, docs/data_cache.md).
model   g(theta_s, p) = GaussianPlume.unit_response for the single source position config.SOURCES_XY[s]
        (N = 1) at every slab cell centre (M = 41,400 receptors, drone z = DRONE_Z), for each combination of
        U in config.FWD_U_CANDIDATES x sigma_v in config.FWD_SIGMA_V_CANDIDATES x wind_mode in
        config.FWD_WIND_MODES ('global': params.U along +x; 'local': the LBM wind at the source, plan S1 보강).
        NOTE: with the default local_wind_blend = 1 the 'local' mode takes U_i from the LBM field and ignores
        params.U, so its three U rows are identical by construction (only sigma_v varies); they are kept so
        that the table has the full 18 rows the plan asks for.
cells   DOWNWIND cells only, d > 0 with respect to the model's wind direction at the source (the local
        direction in wind_mode='local'), AND n_LDM > 0, AND g > config.T1_3_G_FLOOR_FACTOR * FWD_G_FLOOR.
        Upwind cells (d <= 0) have g = g_floor (plan 4.2) so rho ~ +20 there would dominate the std; they are
        excluded from the statistic and the fraction of the source's occupied cells (and of its slab mass) that
        lies upwind is reported separately.
        CAVEAT (review D4-3): the floor rule g > 10 g_floor is NOT in the plan text (which selects d > 0 and
        n_LDM > 0 only); it drops far-crosswind cells whose Gaussian response has been clipped to the floor,
        and the number of dropped cells depends on the candidate (a narrow plume drops more: source 113 loses
        779 cells at sigma_v 0.9 but 3,047 at U 3.69 / sigma_v 0.3).  The statistic therefore compares
        candidate-dependent cell sets.  To make the deviation explicit every ShapeStats also carries the
        PLAN-LITERAL statistic (``std_plan``: all occupied downwind cells, floor cells included with their
        clipped g), the grid rows carry ``mean_train_std_plan``, the JSON reports which combination that
        statistic would choose, and the pass criterion is evaluated on both (``pass`` = floor-rule set,
        ``pass_plan_literal`` = plan set).  On frame 599 both statistics rank the same combination first.
residual  rho = log(n_LDM / g); rho' = rho - median_s(rho).  g is the q = 1 unit response while the slab is
        the density for the ACTUAL release rate, so median_s(rho) is dominated by the unknown scale log kappa_eff
        and cannot discriminate (U, sigma_v); only the SHAPE residual rho' can (plan risk R13).
statistics per source: std(rho') (selection), IQR(rho'), p90-p10, n_cells, median_s(rho) = implied log kappa offset.
        Informational extras (NOT used for selection): the same std / p90-p10 over the "dense" cells with
        n_LDM > config.T1_3_INFO_DENSITY_FRACTION x max_s(n_LDM) (the source's slab maximum over all occupied
        cells; above the single-particle fringe), the slab-mass-weighted std of rho', and the plan-literal
        std_plan described above; they show how much of the residual comes from the sparse fringe / the floor rule.
selection  the (U, sigma_v, wind_mode) minimising the mean over config.TRAIN_SOURCES of std(rho'); holdout
        sources (config.HOLDOUT_SOURCES) are evaluated for the table but never used for the selection.
pass    std(rho') < config.T1_3_STD_PASS for the five open sources config.T1_3_OPEN_SOURCES at the chosen
        combination.  Discriminability (plan S1 위험·대안): if the spread of the mean-train std over the 9
        (U, sigma_v) combinations of the chosen wind mode is below config.T1_3_MIN_DISCRIMINABILITY the plan
        prescribes the physical defaults (FWD_DEFAULT_U, FWD_DEFAULT_SIGMA_V) and a "판별력 부족" note in table 1;
        this script REPORTS the flag and the fallback pair and leaves the decision to the orchestrator.
kappa   kappa_ref(implied) = config.SENSOR_K0 * exp(mean_{train} median_s(rho)) (plan 4.3 "κ_ref의 정의");
        exp(m_bar) is reported next to the nominal release rates config.RELEASE_Q_A / RELEASE_Q_B as the
        auxiliary dt-interpretation diagnostic of plan T1-3 [추정, model mismatch may contaminate it].  The
        current config.KAPPA_REF is reported alongside; config is NOT changed here.  NOTE that median_s(rho)
        depends on the combination (mean over train 2.8 at U 1.68 / sigma_v 0.9 vs 4.4 at U 3.69 / sigma_v 0.3
        on frame 599, i.e. 0.7 decade of kappa_ref); every grid row carries ``mean_train_median_rho``.

Outputs
-------
config.CACHE_DIR / calibrate_forward.json  grid table (18 rows), chosen combination, per-source statistics
        (all 13 sources incl. holdout, at every combination), implied kappa, PASS flags, and ``table1_markdown``
        (표 1 초안: source | type | std rho' | p90-p10 | median rho | n_cells | upwind cell frac | upwind mass
        frac; the plan's "풍상 셀에 든 LDM 입자 비율" is the MASS fraction column).
config.FIG_DIR / fig3_mismatch_analytic.png (+ _preview.png at config.FIG_DPI_PREVIEW)  2 x 2 panels at the
        chosen combination: (a) rho' map of the open source 109, (b) rho' map of the trapped source 110
        (config.T1_3_FIG_SOURCES; diverging colour map clipped to +-config.T1_3_FIG_RHO_CLIP; occupied cells
        that are excluded from the statistic - upwind or floor - shown in grey), (c) slab log10 and (d) model
        log10 (g x exp(median_s rho), i.e. the model scaled to the slab's own median; values below the colour-scale
        minimum, including the g_floor background, are masked) of source 110.  Building
        footprint outline from ObstacleMap.load().hmap > 0 (report 4.2-4.3).  English labels.

References: R13 (Gaussian plume; U at the release point), R15 (Taylor lateral spread).  The helper functions
``shape_stats``, ``run_grid``, ``select_combo``, ``implied_kappa`` and ``markdown_table`` are pure numpy and
are unit-tested on synthetic arrays in tests/test_calibrate_forward.py.
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

from srcloc_env import config  # noqa: E402
from srcloc_env.env.drone import ObstacleMap  # noqa: E402
from srcloc_env.field.concentration_field import LdmSlabBackend  # noqa: E402
from srcloc_env.field.wind import WindField  # noqa: E402
from srcloc_env.pf.forward_model import ForwardParams, GaussianPlume, wind_aligned_coords  # noqa: E402
from srcloc_env.preprocess.gridder import SlabGrid  # noqa: E402

COLOR_ACCENT = "#d1495b"        # source star (same accent as fig_scene.py)
COLOR_EXCLUDED = "#d9d9d9"      # occupied cells excluded from the statistic (upwind / floor)
CMAP_RHO = "RdBu_r"             # diverging: red = slab above model, blue = slab below model
CMAP_DENSITY = LinearSegmentedColormap.from_list("blues_trunc", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
SELECTION_KEY = "mean_train_std"
INFO_KEYS = ("mean_train_std_dense", "mean_train_mass_weighted_std", "mean_train_std_plan")   # informational alternatives


# ------------------------------------------------------------------------------------------ statistics
@dataclass(frozen=True)
class ShapeStats:
    """Per-source T1-3 statistics (plan T1-3).  NaN where no cell qualifies.

    n_cells         downwind cells used: n_LDM > 0, d > 0, g > floor_factor * g_floor
    n_occupied      cells with n_LDM > 0 (any direction)
    n_floor_excluded  occupied downwind cells dropped because g <= floor_factor * g_floor
    median_rho      median_s(rho), rho = log(n_LDM / g): implied log kappa offset (plan 4.3)
    std / iqr / p90_p10   of rho' = rho - median_rho (std is the selection statistic, ddof = 0)
    upwind_cell_frac  fraction of the occupied cells with d <= 0 (table 1 column "upwind cell frac")
    upwind_mass_frac  fraction of the slab mass (sum n_LDM) in those upwind cells (plan wording "풍상 셀의 입자 비율",
                    table 1 column "upwind mass frac")
    n_dense / std_dense / p90_p10_dense   informational: the selected cells with n_LDM > dense_fraction x the
                    source's slab maximum (config.T1_3_INFO_DENSITY_FRACTION; above the single-particle fringe)
                    and rho' stats on them (same median as ``std``)
    mass_weighted_std  informational: std of rho' weighted by n_LDM over the selected cells
    n_plan_cells / median_rho_plan / std_plan   informational, PLAN-LITERAL set (module doc CAVEAT): all occupied
                    downwind cells (n_LDM > 0, d > 0), floor-clipped cells included with g = g_floor; own median
    """

    n_cells: int
    n_occupied: int
    n_floor_excluded: int
    median_rho: float
    std: float
    iqr: float
    p90_p10: float
    upwind_cell_frac: float
    upwind_mass_frac: float
    n_dense: int
    std_dense: float
    p90_p10_dense: float
    mass_weighted_std: float
    n_plan_cells: int
    median_rho_plan: float
    std_plan: float


def shape_stats(n_ldm: np.ndarray, g: np.ndarray, d: np.ndarray, g_floor: float = config.FWD_G_FLOOR,
                floor_factor: float = config.T1_3_G_FLOOR_FACTOR,
                dense_fraction: float = config.T1_3_INFO_DENSITY_FRACTION) -> tuple[ShapeStats, np.ndarray]:
    """T1-3 shape residual of one source on M receptors: (ShapeStats, rho' (M,) with NaN where not selected).

    n_ldm  (M,) slab density [particles/m^3];  g (M,) unit response (> 0, floored);  d (M,) downwind distance
    w.r.t. the model's wind direction at the source (wind_aligned_coords).  Selected cells: n_ldm > 0 & d > 0 &
    g > floor_factor * g_floor (plan T1-3 + D4-3 floor rule).  rho' = log(n_ldm / g) - median over the selected
    cells, so a slab that is a pure multiple of g has rho' = 0 everywhere (std = IQR = p90-p10 = 0).  The
    median is always the one over ALL selected cells (the dense / weighted extras reuse it).  The plan-literal
    extras (``std_plan``) use the larger set n_ldm > 0 & d > 0 with its own median.
    """
    n = np.asarray(n_ldm, dtype=np.float64).ravel()
    gg = np.asarray(g, dtype=np.float64).ravel()
    dd = np.asarray(d, dtype=np.float64).ravel()
    if not (n.shape == gg.shape == dd.shape):
        raise ValueError(f"n_ldm {n.shape}, g {gg.shape} and d {dd.shape} must have the same length")
    occupied = n > 0.0
    downwind = dd > 0.0
    above_floor = gg > float(floor_factor) * float(g_floor)
    plan_sel = occupied & downwind
    sel = plan_sel & above_floor
    upwind = occupied & ~downwind
    rho_prime = np.full(n.shape, np.nan)
    n_occ = int(occupied.sum())
    mass = float(n[occupied].sum()) if n_occ else 0.0
    up_cell = float(upwind.sum() / n_occ) if n_occ else float("nan")
    up_mass = float(n[upwind].sum() / mass) if mass > 0.0 else float("nan")
    n_floor = int((plan_sel & ~above_floor).sum())
    nan = float("nan")
    n_plan = int(plan_sel.sum())
    if n_plan:
        rho_plan = np.log(n[plan_sel] / gg[plan_sel])
        med_plan = float(np.median(rho_plan))
        std_plan = float((rho_plan - med_plan).std())
    else:
        med_plan, std_plan = nan, nan
    if not sel.any():
        return ShapeStats(0, n_occ, n_floor, nan, nan, nan, nan, up_cell, up_mass, 0, nan, nan, nan,
                          n_plan, med_plan, std_plan), rho_prime
    ns = n[sel]
    rho = np.log(ns / gg[sel])
    med = float(np.median(rho))
    rp = rho - med
    rho_prime[sel] = rp
    q10, q25, q75, q90 = np.percentile(rp, [10.0, 25.0, 75.0, 90.0])
    dense = ns > float(dense_fraction) * float(n[occupied].max())      # threshold on the SOURCE maximum
    if dense.any():
        d10, d90 = np.percentile(rp[dense], [10.0, 90.0])
        std_dense, p_dense = float(rp[dense].std()), float(d90 - d10)
    else:
        std_dense, p_dense = nan, nan
    w = ns / ns.sum()
    mu = float((w * rp).sum())
    mw_std = float(np.sqrt((w * (rp - mu) ** 2).sum()))
    return ShapeStats(int(sel.sum()), n_occ, n_floor, med, float(rp.std()), float(q75 - q25), float(q90 - q10),
                      up_cell, up_mass, int(dense.sum()), std_dense, p_dense, mw_std,
                      n_plan, med_plan, std_plan), rho_prime

# ------------------------------------------------------------------------------------------ grid search
def make_plume(wind_mode: str, U: float, sigma_v: float, wind_field: WindField | None) -> GaussianPlume:
    """GaussianPlume for one grid combination (ForwardParams defaults otherwise; wind_dir_deg = 0 = +x)."""
    return GaussianPlume(ForwardParams(U=float(U), sigma_v=float(sigma_v)),
                         wind_field=wind_field if wind_mode == "local" else None, wind_mode=wind_mode)


def evaluate_sources(plume: GaussianPlume, src_xy: np.ndarray, cells_xyz: np.ndarray, slabs: np.ndarray,
                     ) -> tuple[list[ShapeStats], np.ndarray, np.ndarray, np.ndarray, dict]:
    """Statistics of n sources at once: (stats list, rho' (n, M), g (n, M), d (n, M), local-wind dict).

    src_xy (n, 2), cells_xyz (M, 3) receptors (cell centres at DRONE_Z), slabs (n, M) truth densities.
    d is measured along the plume's wind direction at each source (plume.local_wind: the global +x in
    wind_mode='global', the LBM direction in 'local'), which is exactly the direction that defines g's d <= 0 floor.
    """
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    cells = np.asarray(cells_xyz, dtype=np.float64).reshape(-1, 3)
    sl = np.asarray(slabs, dtype=np.float64).reshape(src.shape[0], -1)
    if sl.shape[1] != cells.shape[0]:
        raise ValueError(f"slabs {sl.shape} must be (n_sources, M = {cells.shape[0]})")
    g = plume.unit_response(src, cells)                                     # (n, M)
    lw = plume.local_wind(src)
    d, _ = wind_aligned_coords(src, cells[:, :2], lw.dir_deg)               # (n, M)
    stats: list[ShapeStats] = []
    rho_prime = np.empty_like(g)
    for i in range(src.shape[0]):
        st, rp = shape_stats(sl[i], g[i], d[i], plume.params.g_floor)
        stats.append(st)
        rho_prime[i] = rp
    wind = {"U_m_per_s": lw.U.tolist(), "dir_deg": lw.dir_deg.tolist(), "speed_m_per_s": lw.speed.tolist(),
            "fallback": lw.fallback.tolist(), "clipped_to_U_MIN": (lw.speed < config.FWD_U_MIN).tolist()}
    return stats, rho_prime, g, d, wind


def run_grid(src_ids: tuple[int, ...], src_xy: np.ndarray, cells_xyz: np.ndarray, slabs: np.ndarray,
             wind_field: WindField | None, train_ids: tuple[int, ...] = config.TRAIN_SOURCES,
             wind_modes: tuple[str, ...] = config.FWD_WIND_MODES,
             u_candidates: tuple[float, ...] = config.FWD_U_CANDIDATES,
             sigma_v_candidates: tuple[float, ...] = config.FWD_SIGMA_V_CANDIDATES) -> list[dict]:
    """One row per (wind_mode, U, sigma_v): per-source ShapeStats and the mean-over-train summaries.

    Rows are ordered wind_mode-major, then U, then sigma_v (itertools.product).  'local' needs a WindField;
    wind modes are otherwise taken as given so tests can run the global mode alone without data.
    """
    train_idx = [i for i, s in enumerate(src_ids) if s in train_ids]
    if not train_idx:
        raise ValueError("no training source among src_ids")
    rows: list[dict] = []
    for mode, U, sv in product(wind_modes, u_candidates, sigma_v_candidates):
        plume = make_plume(mode, U, sv, wind_field)
        stats, _, _, _, wind = evaluate_sources(plume, src_xy, cells_xyz, slabs)
        tr = [stats[i] for i in train_idx]
        rows.append({
            "wind_mode": mode, "U": float(U), "sigma_v": float(sv),
            "mean_train_std": float(np.mean([s.std for s in tr])),
            "mean_train_iqr": float(np.mean([s.iqr for s in tr])),
            "mean_train_p90_p10": float(np.mean([s.p90_p10 for s in tr])),
            "mean_train_median_rho": float(np.mean([s.median_rho for s in tr])),
            "mean_train_std_dense": float(np.mean([s.std_dense for s in tr])),
            "mean_train_mass_weighted_std": float(np.mean([s.mass_weighted_std for s in tr])),
            "mean_train_std_plan": float(np.mean([s.std_plan for s in tr])),
            "n_train": len(tr),
            "per_source": {str(s): asdict(st) for s, st in zip(src_ids, stats)},
            "source_wind": {str(s): {k: v[i] for k, v in wind.items()} for i, s in enumerate(src_ids)},
        })
    return rows


def select_combo(rows: list[dict], key: str = SELECTION_KEY) -> tuple[int, dict]:
    """Index of the row minimising ``key`` (NaN rows never win; first minimum on ties) and a spread summary.

    The spread summary (plan S1 위험·대안 discriminability rule) gives, for the chosen wind mode, max - min of
    ``key`` over its (U, sigma_v) rows, the same over all rows, and the flag spread_chosen_mode >=
    config.T1_3_MIN_DISCRIMINABILITY.
    """
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    if not np.isfinite(vals).any():
        raise ValueError(f"no finite {key} in the grid")
    k = int(np.nanargmin(vals))
    mode = rows[k]["wind_mode"]
    same = np.array([v for r, v in zip(rows, vals) if r["wind_mode"] == mode])
    spread_mode = float(np.nanmax(same) - np.nanmin(same))
    spread_all = float(np.nanmax(vals) - np.nanmin(vals))
    return k, {"key": key, "spread_chosen_mode": spread_mode, "spread_all_rows": spread_all,
               "n_rows_chosen_mode": int(same.size),
               "discriminable": bool(spread_mode >= config.T1_3_MIN_DISCRIMINABILITY),
               "threshold": config.T1_3_MIN_DISCRIMINABILITY,
               "plan_fallback_if_not_discriminable": {"U": config.FWD_DEFAULT_U, "sigma_v": config.FWD_DEFAULT_SIGMA_V}}


def implied_kappa(median_rhos: np.ndarray, k0: float = config.SENSOR_K0) -> dict:
    """kappa_ref = k0 * exp(mean median_s rho) over the given (train) sources (plan 4.3 "κ_ref의 정의")."""
    m = np.asarray(median_rhos, dtype=np.float64)
    if m.size == 0 or not np.isfinite(m).all():
        raise ValueError("implied_kappa needs finite median rho values")
    m_bar = float(m.mean())
    q_impl = float(np.exp(m_bar))
    return {"m_bar": m_bar, "exp_m_bar_particles_per_s": q_impl, "kappa_ref_implied": float(k0 * q_impl), "k0": float(k0),
            "n_sources": int(m.size), "per_source_kappa": (k0 * np.exp(m)).tolist()}


def source_type(s: int) -> str:
    """Table-1 label: open / trapped / holdout (109 is 'holdout(open)') / train."""
    tag = "open" if s in config.T1_3_OPEN_SOURCES else ("trapped" if s in config.T1_3_TRAPPED_SOURCES else "train")
    if s in config.HOLDOUT_SOURCES:
        return f"holdout({tag})" if tag != "train" else "holdout"
    return tag


def markdown_table(src_ids: tuple[int, ...], per_source: dict[str, dict], caption: str = "") -> str:
    """표 1 초안: source | type | std rho' | p90-p10 | median rho | n_cells | upwind cell frac | upwind mass frac.

    "upwind mass frac" is the plan's "풍상 셀에 든 LDM 입자 비율" (fraction of the slab mass at d <= 0); the
    cell fraction is kept as a second column because it is what figure 3's titles show.
    """
    lines = []
    if caption:
        lines.append(caption)
        lines.append("")
    lines.append("| source | type | std rho' | p90-p10 | median rho | n_cells | upwind cell frac | upwind mass frac |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in src_ids:
        st = per_source[str(s)]
        lines.append(f"| {s} | {source_type(s)} | {st['std']:.2f} | {st['p90_p10']:.2f} | {st['median_rho']:.2f} | "
                     f"{st['n_cells']} | {st['upwind_cell_frac']:.2f} | {st['upwind_mass_frac']:.2f} |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ figure 3
def _window(xs: float, ys: float, grid: SlabGrid) -> tuple[float, float, float, float]:
    up, down, cross = config.T1_3_FIG_WINDOW_M
    x_lo, x_hi = grid.x0, grid.x0 + grid.nx * grid.res
    y_lo, y_hi = grid.y0, grid.y0 + grid.ny * grid.res
    return max(xs - up, x_lo), min(xs + down, x_hi), max(ys - cross, y_lo), min(ys + cross, y_hi)


def render_figure3(grid: SlabGrid, omap: ObstacleMap, panels: dict[int, dict], combo: dict, out_png: Path,
                   out_preview: Path, index: int) -> dict:
    """2 x 2 figure: (a) rho' 109, (b) rho' 110, (c) slab log10 110, (d) model log10 110 (see module doc).

    panels[s] = {"rho_prime": (ny, nx), "slab": (ny, nx), "model": (ny, nx) = g * exp(median rho), "occupied":
    (ny, nx) bool, "stats": ShapeStats dict, "wind": {U, dir_deg}}.
    """
    s_open, s_trap = config.T1_3_FIG_SOURCES
    extent = (grid.x0, grid.x0 + grid.nx * grid.res, grid.y0, grid.y0 + grid.ny * grid.res)
    clip = config.T1_3_FIG_RHO_CLIP
    occ_img = (omap.hmap > config.OCC_HMAP_NO_BUILDING).T.astype(np.float32)      # (ny, nx) for contour(x, y, Z)
    combo_txt = f"{combo['wind_mode']} wind, U = {combo['U']:g} m/s, sigma_v = {combo['sigma_v']:g} m/s"

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
        ax.text(xs + 0.5 * L * np.cos(a), ys + 0.5 * L * np.sin(a) - 14, f"model wind {w['U']:.2f} m/s, {w['dir_deg']:.0f} deg",
                fontsize=7.5, ha="center", va="top", zorder=8, path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
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
        ax.imshow(excl, extent=extent, origin="lower", cmap=ListedColormap([COLOR_EXCLUDED]),
                  vmin=0, vmax=1, interpolation="nearest", zorder=1)
        im = ax.imshow(np.ma.masked_invalid(np.clip(p["rho_prime"], -clip, clip)), extent=extent, origin="lower",
                       cmap=CMAP_RHO, vmin=-clip, vmax=clip, interpolation="nearest", zorder=2)
        kind = "open" if s == s_open else "trapped"
        base(ax, s, f"{lab} rho' = log(n_LDM / g) - median, source {s} ({kind})\nstd {st['std']:.2f}, "
                    f"p90-p10 {st['p90_p10']:.2f}, n = {st['n_cells']}, upwind cell fraction {st['upwind_cell_frac']:.2f} "
                    f"(mass {st['upwind_mass_frac']:.2f}); grey = excluded cells")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(f"rho' (clipped to +-{clip:g})")
        numbers[str(s)] = {"n_shown": int(np.isfinite(p["rho_prime"]).sum()),
                           "n_clipped": int((np.abs(p["rho_prime"][np.isfinite(p["rho_prime"])]) > clip).sum())}

    slab = panels[s_trap]["slab"]
    model = panels[s_trap]["model"]
    pos = slab[slab > 0]
    vmax = float(pos.max())
    vmin = max(float(pos.min()), vmax / 10.0 ** config.FIG_LOG_DECADES)
    for ax, arr, lab, what in ((axes[1, 0], slab, "(c)", f"LDM slab, z = {config.DRONE_Z:g} m"),
                               (axes[1, 1], model, "(d)", "analytic plume g x exp(median rho)")):
        im = ax.imshow(np.ma.masked_where(arr < vmin, arr), extent=extent, origin="lower", cmap=CMAP_DENSITY,
                       norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest", zorder=2)
        base(ax, s_trap, f"{lab} source {s_trap}: {what}\n(same colour scale, {config.FIG_LOG_DECADES:g} decades)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("particles / m$^3$ (log10 scale)")
    numbers["density_log_vmin"] = vmin
    numbers["density_log_vmax"] = vmax
    fig.suptitle(f"Figure 3 (draft): analytic-plume mismatch at the chosen T1-3 combination ({combo_txt}), "
                 f"frame {index} = step {config.index_to_step(index)}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=config.FIG_DPI_FINAL, bbox_inches="tight")
    fig.savefig(out_preview, dpi=config.FIG_DPI_PREVIEW, bbox_inches="tight")
    plt.close(fig)
    numbers["files"] = {p.name: {"path": str(p), "bytes": p.stat().st_size} for p in (out_png, out_preview)}
    return numbers

# ------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=int, default=config.T1_3_FRAME_INDEX)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "calibrate_forward.json")
    ap.add_argument("--fig-dir", type=Path, default=config.FIG_DIR)
    args = ap.parse_args(argv)
    t_start = time.perf_counter()

    # ---- truth: per-source slab at DRONE_Z, cell centres as receptors ------------------------------
    be = LdmSlabBackend(config.CACHE_DIR, max_cached_frames=config.FIELD_MAX_CACHED_FRAMES)
    sf = be.slab(args.index)
    grid = sf.grid
    zi = sf.z_index(config.DRONE_Z)
    src_ids = tuple(int(s) for s in sf.sources)
    src_xy = np.array([config.SOURCES_XY[s] for s in src_ids], dtype=np.float64)
    cells = grid.cell_centres(config.DRONE_Z)                                 # (M, 3) row-major (iy, ix)
    slabs = sf.density[:, zi].reshape(len(src_ids), -1).astype(np.float64)     # (n_src, M)
    t0 = time.perf_counter()
    wf = WindField.load()
    wind_load_s = time.perf_counter() - t0

    # ---- grid search ------------------------------------------------------------------------------
    t0 = time.perf_counter()
    rows = run_grid(src_ids, src_xy, cells, slabs, wf)
    grid_s = time.perf_counter() - t0
    k, spread = select_combo(rows)
    chosen = rows[k]
    combo = {"wind_mode": chosen["wind_mode"], "U": chosen["U"], "sigma_v": chosen["sigma_v"], "row_index": k}
    best_per_mode = {}
    for mode in config.FWD_WIND_MODES:
        sub = [(i, r) for i, r in enumerate(rows) if r["wind_mode"] == mode]
        j = min(sub, key=lambda ir: (np.nan_to_num(ir[1][SELECTION_KEY], nan=np.inf), ir[0]))[0]
        best_per_mode[mode] = {"row_index": j, "U": rows[j]["U"], "sigma_v": rows[j]["sigma_v"],
                               SELECTION_KEY: rows[j][SELECTION_KEY]}
    alternatives = {}
    for key in INFO_KEYS:                                   # informational: would another statistic choose differently?
        j, sp = select_combo(rows, key)
        alternatives[key] = {"row_index": j, "wind_mode": rows[j]["wind_mode"], "U": rows[j]["U"],
                             "sigma_v": rows[j]["sigma_v"], "value": rows[j][key], "spread_chosen_mode": sp["spread_chosen_mode"],
                             "same_as_selection": bool(j == k)}

    # ---- pass criterion, implied kappa ------------------------------------------------------------
    ps = chosen["per_source"]
    open_std = {str(s): ps[str(s)]["std"] for s in config.T1_3_OPEN_SOURCES}
    open_std_plan = {str(s): ps[str(s)]["std_plan"] for s in config.T1_3_OPEN_SOURCES}
    passed = all(np.isfinite(v) and v < config.T1_3_STD_PASS for v in open_std.values())
    passed_plan = all(np.isfinite(v) and v < config.T1_3_STD_PASS for v in open_std_plan.values())
    kappa = implied_kappa([ps[str(s)]["median_rho"] for s in config.TRAIN_SOURCES])
    kappa.update({
        "train_sources": list(config.TRAIN_SOURCES),
        "current_config_KAPPA_REF": config.KAPPA_REF,
        "log10_implied_over_current": float(np.log10(kappa["kappa_ref_implied"] / config.KAPPA_REF)),
        "nominal_q_A_particles_per_s": config.RELEASE_Q_A, "nominal_q_B_particles_per_s": config.RELEASE_Q_B,
        "log10_exp_m_bar_over_q_A": float(np.log10(kappa["exp_m_bar_particles_per_s"] / config.RELEASE_Q_A)),
        "log10_exp_m_bar_over_q_B": float(np.log10(kappa["exp_m_bar_particles_per_s"] / config.RELEASE_Q_B)),
        "kappa_grid_decades": config.KAPPA_GRID_DECADES, "kappa_G": config.KAPPA_G,
        "per_source_kappa_all": {str(s): config.SENSOR_K0 * float(np.exp(ps[str(s)]["median_rho"])) for s in src_ids},
        "mean_train_median_rho_range_over_grid": [float(min(r["mean_train_median_rho"] for r in rows)),
                                                  float(max(r["mean_train_median_rho"] for r in rows))],
        "note": "config.KAPPA_REF is NOT changed by this script (orchestrator decision, plan 4.3 / T1-2b); "
                "m_bar depends on the chosen combination (see mean_train_median_rho_range_over_grid).",
    })
    caption = (f"표 1 초안 (T1-3, frame {args.index}, z = {config.DRONE_Z:g} m): chosen {combo['wind_mode']} wind, "
               f"U = {combo['U']:g} m/s, sigma_v = {combo['sigma_v']:g} m/s; mean_train std(rho') = "
               f"{chosen['mean_train_std']:.3f}; implied kappa_ref = {kappa['kappa_ref_implied']:.3e} "
               f"(exp(m_bar) = {kappa['exp_m_bar_particles_per_s']:.1f} particles/s); "
               f"PASS = {passed}" + ("" if spread["discriminable"] else "; 판별력 부족 (spread < 0.1)"))
    table_md = markdown_table(src_ids, ps, caption)

    # ---- figure 3 at the chosen combination -------------------------------------------------------
    plume = make_plume(combo["wind_mode"], combo["U"], combo["sigma_v"], wf)
    fig_ids = tuple(config.T1_3_FIG_SOURCES)
    fig_idx = [src_ids.index(s) for s in fig_ids]
    stats_f, rp_f, g_f, _, wind_f = evaluate_sources(plume, src_xy[fig_idx], cells, slabs[fig_idx])
    panels = {}
    for j, s in enumerate(fig_ids):
        st = stats_f[j]
        panels[s] = {"rho_prime": rp_f[j].reshape(grid.ny, grid.nx),
                     "slab": slabs[fig_idx[j]].reshape(grid.ny, grid.nx),
                     "model": (g_f[j] * np.exp(st.median_rho)).reshape(grid.ny, grid.nx),
                     "occupied": (slabs[fig_idx[j]] > 0).reshape(grid.ny, grid.nx),
                     "stats": asdict(st), "wind": {"U": wind_f["U_m_per_s"][j], "dir_deg": wind_f["dir_deg"][j]}}
    omap = ObstacleMap.load()
    f3 = args.fig_dir / "fig3_mismatch_analytic.png"
    f3p = args.fig_dir / "fig3_mismatch_analytic_preview.png"
    fig_numbers = render_figure3(grid, omap, panels, combo, f3, f3p, args.index)

    grid_table = [{k2: v for k2, v in r.items() if k2 not in ("per_source", "source_wind")} for r in rows]
    res = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index": args.index, "step": config.index_to_step(args.index), "z": config.DRONE_Z,
        "grid": grid.to_dict(), "n_receptors": int(cells.shape[0]), "sources": list(src_ids),
        "train_sources": list(config.TRAIN_SOURCES), "holdout_sources": list(config.HOLDOUT_SOURCES),
        "open_sources": list(config.T1_3_OPEN_SOURCES), "trapped_sources": list(config.T1_3_TRAPPED_SOURCES),
        "candidates": {"U": list(config.FWD_U_CANDIDATES), "sigma_v": list(config.FWD_SIGMA_V_CANDIDATES),
                       "wind_mode": list(config.FWD_WIND_MODES)},
        "selection_rule": "argmin over rows of mean_{train} std(rho'); downwind cells n_LDM > 0, d > 0, "
                          f"g > {config.T1_3_G_FLOOR_FACTOR:g} * g_floor",
        "notes": ["'local' rows: with local_wind_blend = 1 the per-source U_i comes from the LBM field and params.U "
                  "is ignored, so the three U rows of each sigma_v are identical by construction (3 distinct rows).",
                  f"informational columns *_dense use cells with n_LDM > {config.T1_3_INFO_DENSITY_FRACTION:g} x source max "
                  "(above the single-particle fringe); mass_weighted_std weights rho' by n_LDM. Neither is the plan's selection statistic.",
                  f"the floor rule g > {config.T1_3_G_FLOOR_FACTOR:g} x g_floor is not in the plan text and makes the cell set "
                  "candidate-dependent (per_source[*].n_floor_excluded); *_plan / std_plan / pass_plan_literal use the plan's "
                  "literal set (n_LDM > 0, d > 0, floor cells included with g = g_floor) as a robustness check."],
        "grid_table": grid_table,
        "chosen": combo, "chosen_row": chosen,
        "best_per_wind_mode": best_per_mode,
        "alternative_statistics_informational": alternatives,
        "discriminability": spread,
        "per_source_chosen": ps,
        "per_source_type": {str(s): source_type(s) for s in src_ids},
        "open_source_std": open_std, "pass_criterion_std": config.T1_3_STD_PASS, "pass": bool(passed),
        "open_source_std_plan_literal": open_std_plan, "pass_plan_literal": bool(passed_plan),
        "implied_kappa": kappa,
        "table1_markdown": table_md,
        "figure3": fig_numbers,
        "all_rows": rows,
        "timing": {"wind_load_s": wind_load_s, "grid_search_s": grid_s, "total_s": time.perf_counter() - t_start},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"T1-3 grid ({len(rows)} rows), mean_train std(rho') [iqr, p90-p10] | info: std_dense, mass-weighted std, plan-literal std:")
    for i, r in enumerate(rows):
        mark = " <-- chosen" if i == k else ""
        print(f"  {r['wind_mode']:6s} U {r['U']:4.2f} sv {r['sigma_v']:3.1f}: {r['mean_train_std']:.3f} "
              f"[{r['mean_train_iqr']:.3f}, {r['mean_train_p90_p10']:.3f}] | {r['mean_train_std_dense']:.3f}, "
              f"{r['mean_train_mass_weighted_std']:.3f}, {r['mean_train_std_plan']:.3f}; "
              f"mean median rho {r['mean_train_median_rho']:.2f}{mark}")
    print(table_md)
    print("informational per-source (chosen): " + ", ".join(
        f"{s}: dense std {ps[str(s)]['std_dense']:.2f} (n {ps[str(s)]['n_dense']}), mw std {ps[str(s)]['mass_weighted_std']:.2f}, "
        f"plan std {ps[str(s)]['std_plan']:.2f} (n {ps[str(s)]['n_plan_cells']})"
        for s in src_ids))
    print("alternative statistics choose: " + ", ".join(
        f"{key}: {a['wind_mode']} U {a['U']:g} sv {a['sigma_v']:g} ({'same' if a['same_as_selection'] else 'DIFFERENT'})"
        for key, a in alternatives.items()))
    print(f"discriminability spread (chosen mode) {spread['spread_chosen_mode']:.3f} -> "
          f"{'ok' if spread['discriminable'] else 'INSUFFICIENT (plan: physical defaults)'}")
    print(f"implied kappa_ref {kappa['kappa_ref_implied']:.4e} (exp(m_bar) {kappa['exp_m_bar_particles_per_s']:.2f} particles/s; "
          f"q_A {config.RELEASE_Q_A:.1f}, q_B {config.RELEASE_Q_B:.2f}); current KAPPA_REF {config.KAPPA_REF:.4e}")
    print(f"T1-3 PASS criterion std(rho') < {config.T1_3_STD_PASS} on open sources {list(config.T1_3_OPEN_SOURCES)}:",
          "PASS" if passed else "FAIL", {k2: round(v, 3) for k2, v in open_std.items()},
          "| plan-literal set:", "PASS" if passed_plan else "FAIL", {k2: round(v, 3) for k2, v in open_std_plan.items()})
    print(f"elapsed {res['timing']['total_s']:.1f} s; wrote {args.out}, {f3}")
    return res


if __name__ == "__main__":
    main()