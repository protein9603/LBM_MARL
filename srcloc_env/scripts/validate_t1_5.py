"""D6-2b / plan S1 T1-5: fidelity and ordering stability of the GMM belief summary on the T1-4 belief snapshots.

Usage: python -m srcloc_env.scripts.validate_t1_5 [--sources 109 102] [--snapshot-dir ...] [--out ...] [--fig ...]
Reads config.CACHE_DIR / t1_4_snapshots_{src}.npz (validate_t1_4, filter config.T1_4_SNAPSHOT_FILTER, seed 0, one
snapshot at step 0 and every config.T1_4_GMM_EVERY RL steps: steps (S,), xy (S, N, 2) float32, logw (S, N) float32,
drone_xy (S, n_drones, 2), true_xy (2,), map_xy (S, 2)).  Writes config.CACHE_DIR / validate_t1_5.json and figure 5
config.FIG_DIR / fig5_belief_evolution.png (config.FIG_DPI_FINAL) + _preview.png (config.FIG_DPI_PREVIEW).

Per snapshot (``snapshot_fidelity``, plan 4.4 / T1-5): the K = config.GMM_K weighted-EM GMM (gmm_summary.fit_weighted_gmm
on exp(logw), rng seed config.T1_5_GMM_SEED per snapshot); the criterion TV distance between the weighted particle
histogram and the GMM mass on config.T1_5_TV_CELL_M cells (gmm_summary.total_variation_distance, centre-point rule);
two diagnostics - ``tv_integrated`` (the same cells, GMM mass integrated on config.T1_5_TV_N_SUB^2 sub-cell points,
which removes the centre-point bias once the belief sd is below the cell) and ``tv_noise_floor`` (TV of N i.i.d.
samples from the fitted GMM itself against the GMM mass, i.e. what a perfect summary would score with N particles) -
plus the informational TV on config.T1_5_TV_CELL_INFO_M cells; the top-component sigma (GmmSummary.top_sigma), the
number of valid components, the 27-dim observation vector (GmmSummary.to_vector with the first drone position) and,
against the previous snapshot, the component-order flip rate (gmm_summary.order_flip_rate, match radius
config.T1_5_MATCH_RADIUS_M).
Per source (``analyse_snapshots`` / ``summarise``): TV median / max, mean flip rate (all transitions, and restricted
to the converged phase where both snapshots have top sigma < config.SUCCESS_SIGMA_M), fraction of snapshots with
TV < config.T1_5_TV_MAX, valid components over time, and the top-sigma trajectory next to the MAP error
|map_xy - true_xy| with the first step at which the success criterion (sigma < config.SUCCESS_SIGMA_M and
error < config.SUCCESS_ERROR_M, plan 4.5) holds.
Verdict (``verdict``): T1-5 PASS iff median TV < config.T1_5_TV_MAX and mean flip rate < config.T1_5_FLIP_MAX for
every source (plan S1 T1-5, strict; flips counted over the converged phase, TV by sub-cell integration); the same test on (tv_integrated, all-transition flip rate) is reported as the
informational verdict.

Figure 5 (``make_figure``): rows = sources (109 open, 102 trapped), columns = the snapshots nearest to
config.T1_5_FIG_STEPS; particles coloured by weight, 1- and 2-sigma ellipses of the valid GMM components (line width
by weight), true source (star), MAP (cross), the two drones (triangles) with their paths so far, building outlines
(ObstacleMap.hmap > 0) inside +-config.T1_5_FIG_HALF_WIDTH_M around the true source, and a
+-config.T1_5_FIG_INSET_HALF_WIDTH_M zoom inset around the MAP once the top sigma is below
config.T1_5_FIG_INSET_SIGMA_M; titles with step, MAP error and top sigma.
References: R5 (RB-PF), R7 (GMM summary, Park / Ladosz / Oh 2022).
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
from srcloc_env.pf.gmm_summary import GmmSummary, fit_weighted_gmm, order_flip_rate, total_variation_distance
from srcloc_env.scripts.validate_pf_adjoint import two_drone_paths
from srcloc_env.scripts.validate_t1_4 import source_type

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Ellipse   # noqa: E402

# figure colours (dataviz palette: particles = sequential viridis by weight; marks = accent + ink)
COLOR_TRUE = "#eb6834"
COLOR_MAP = "#1baf7a"
COLOR_ELLIPSE = "#2a78d6"
COLOR_DRONE = ("#7b3fb3", "#b3701e")
COLOR_BUILDING = "#8a8a8a"
SNAPSHOT_KEYS = ("steps", "xy", "logw", "drone_xy", "true_xy", "map_xy")


# ---------------------------------------------------------------------------------------- helpers
def normalised_weights(logw: np.ndarray) -> np.ndarray:
    """exp(logw) normalised to sum 1 (float64, stable shift by the maximum)."""
    lw = np.asarray(logw, dtype=np.float64)
    w = np.exp(lw - lw.max())
    return w / w.sum()


def load_snapshot(path: Path) -> dict[str, np.ndarray]:
    """Load one t1_4_snapshots_{src}.npz and check the array layout (validate_t1_4.run_filter)."""
    with np.load(Path(path)) as z:
        snap = {k: np.asarray(z[k]) for k in SNAPSHOT_KEYS}
    n = snap["steps"].shape[0]
    if snap["xy"].shape[:2] != snap["logw"].shape or snap["xy"].shape[0] != n or snap["map_xy"].shape != (n, 2) \
            or snap["drone_xy"].shape[0] != n or snap["true_xy"].shape != (2,):
        raise ValueError(f"inconsistent snapshot arrays in {path}: { {k: v.shape for k, v in snap.items()} }")
    return snap


def _tv_bounds(xy: np.ndarray, cell: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """The default bounds of gmm_summary.total_variation_distance (particle box padded by 3 cells)."""
    pad = 3.0 * cell
    return ((float(xy[:, 0].min()) - pad, float(xy[:, 0].max()) + pad), (float(xy[:, 1].min()) - pad, float(xy[:, 1].max()) + pad))


def tv_distance_integrated(xy: np.ndarray, w: np.ndarray, gmm: GmmSummary, cell: float = config.T1_5_TV_CELL_M,
                           n_sub: int = config.T1_5_TV_N_SUB) -> float:
    """TV between the weighted particle histogram and the GMM mass with the cell mass integrated on an n_sub x n_sub
    sub-cell grid (diagnostic for the centre-point rule of total_variation_distance, plan D6-2b)."""
    xy = np.asarray(xy, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    (x0, x1), (y0, y1) = _tv_bounds(xy, cell)
    nx = int(np.ceil((x1 - x0) / cell)); ny = int(np.ceil((y1 - y0) / cell))
    ix = np.clip(((xy[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    hist = np.zeros((nx, ny)); np.add.at(hist, (ix, iy), w)
    off = (np.arange(n_sub) + 0.5) / n_sub * cell
    cx = (x0 + cell * np.arange(nx)[:, None] + off[None, :]).ravel()
    cy = (y0 + cell * np.arange(ny)[:, None] + off[None, :]).ravel()
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    dens = gmm.density(np.column_stack([gx.ravel(), gy.ravel()])).reshape(nx, n_sub, ny, n_sub).mean(axis=(1, 3)) * cell * cell
    dens = dens / max(dens.sum(), 1e-300)
    return float(0.5 * np.abs(hist - dens).sum())


def sample_gmm(gmm: GmmSummary, n: int, rng: np.random.Generator) -> np.ndarray:
    """n i.i.d. samples (n, 2) from the valid components of the mixture."""
    idx = np.where(gmm.mask)[0]
    p = gmm.weights[idx] / gmm.weights[idx].sum()
    comp = rng.choice(idx, size=n, p=p)
    out = np.empty((n, 2))
    for j in idx:
        m = comp == j
        if m.any():
            out[m] = rng.multivariate_normal(gmm.means[j], gmm.covs[j], size=int(m.sum()))
    return out


def tv_noise_floor(gmm: GmmSummary, n: int, cell: float = config.T1_5_TV_CELL_M,
                   n_repeats: int = config.T1_5_TV_NOISE_N_REPEATS, seed: int = config.T1_5_GMM_SEED) -> float:
    """Mean TV (centre rule, same cells) between n i.i.d. samples of the GMM itself and the GMM mass: the score a
    perfect summary would get with n equally weighted particles (finite-N sampling noise, plan D6-2b)."""
    rng = np.random.default_rng([int(seed), 5])
    vals = [total_variation_distance(sample_gmm(gmm, n, rng), np.ones(n), gmm, cell=cell) for _ in range(n_repeats)]
    return float(np.mean(vals))


def snapshot_fidelity(xy: np.ndarray, logw: np.ndarray, drone_xy: np.ndarray | None = None,
                      prev: GmmSummary | None = None, cell: float = config.T1_5_TV_CELL_M,
                      cell_info: float | None = config.T1_5_TV_CELL_INFO_M, k: int = config.GMM_K,
                      match_radius: float = config.T1_5_MATCH_RADIUS_M, n_sub: int = config.T1_5_TV_N_SUB,
                      noise_floor: bool = True, seed: int = config.T1_5_GMM_SEED) -> tuple[dict, GmmSummary]:
    """One snapshot -> (stats, gmm): criterion TV on `cell`, tv_integrated, tv_noise_floor, TV on `cell_info`, top
    sigma, n valid components, the 27-dim vector and the flip rate against `prev` (None without a previous
    snapshot) (plan 4.4 / T1-5)."""
    w = normalised_weights(logw)
    xy = np.asarray(xy, dtype=np.float64)
    gmm = fit_weighted_gmm(xy, w, k=k, rng=np.random.default_rng(seed))
    stats = {
        "tv": total_variation_distance(xy, w, gmm, cell=cell),
        "tv_integrated": tv_distance_integrated(xy, w, gmm, cell=cell, n_sub=n_sub),
        "tv_noise_floor": tv_noise_floor(gmm, xy.shape[0], cell=cell, seed=seed) if noise_floor else None,
        "tv_info": None if cell_info is None else total_variation_distance(xy, w, gmm, cell=cell_info),
        "top_sigma_m": gmm.top_sigma(),
        "n_valid": int(gmm.mask.sum()),
        "weights": gmm.weights.tolist(),
        "flip_rate": None if prev is None else order_flip_rate(prev, gmm, match_radius=match_radius),
        "vector": gmm.to_vector(None if drone_xy is None else np.asarray(drone_xy, dtype=np.float64)).tolist(),
        "neff": float(1.0 / np.sum(w * w)),
    }
    return stats, gmm


PER_STEP_KEYS = ("tv", "tv_integrated", "tv_noise_floor", "tv_info", "top_sigma_m", "n_valid", "weights", "flip_rate", "neff")


def analyse_snapshots(snap: dict[str, np.ndarray], cell: float = config.T1_5_TV_CELL_M,
                      cell_info: float | None = config.T1_5_TV_CELL_INFO_M,
                      sigma_pass: float = config.SUCCESS_SIGMA_M, error_pass: float = config.SUCCESS_ERROR_M,
                      seed: int = config.T1_5_GMM_SEED) -> tuple[dict, list[GmmSummary]]:
    """All snapshots of one source -> (per-step record dict of lists, list of GmmSummary).

    Adds the MAP error |map_xy - true_xy| per snapshot and the success flag (sigma < sigma_pass and error <
    error_pass, plan 4.5); first_success_step is the first snapshot step where it holds (None if never).
    """
    steps = snap["steps"].astype(int)
    err = np.hypot(*(snap["map_xy"].astype(np.float64) - snap["true_xy"].astype(np.float64)[None, :]).T)
    prev: GmmSummary | None = None
    gmms: list[GmmSummary] = []
    rows: list[dict] = []
    for i in range(steps.shape[0]):
        st, g = snapshot_fidelity(snap["xy"][i], snap["logw"][i], drone_xy=snap["drone_xy"][i, 0], prev=prev,
                                  cell=cell, cell_info=cell_info, seed=seed)
        prev = g
        gmms.append(g)
        st["step"] = int(steps[i])
        st["map_error_m"] = float(err[i])
        st["success"] = bool(st["top_sigma_m"] < sigma_pass and err[i] < error_pass)
        rows.append(st)
    record = {"steps": [r["step"] for r in rows], "map_error_m": [r["map_error_m"] for r in rows],
              "success": [r["success"] for r in rows],
              "first_success_step": next((r["step"] for r in rows if r["success"]), None),
              "vectors": [r["vector"] for r in rows]}
    record.update({k: [r[k] for r in rows] for k in PER_STEP_KEYS})
    return record, gmms


def _first(steps: Sequence[int], flags: Sequence[bool]) -> int | None:
    return next((int(s) for s, f in zip(steps, flags) if f), None)


def summarise(record: dict, tv_max: float = config.T1_5_TV_MAX, flip_max: float = config.T1_5_FLIP_MAX,
              sigma_pass: float = config.SUCCESS_SIGMA_M) -> dict:
    """Per-source T1-5 statistics: TV median / max / fraction below tv_max (criterion, integrated, noise floor,
    info cell), mean flip rate over all transitions and over the converged phase (both snapshots with top sigma <
    sigma_pass), valid components, top sigma and MAP error at the last snapshot, the first success step."""
    tv = np.asarray(record["tv"], dtype=np.float64)
    tvi = np.asarray(record["tv_integrated"], dtype=np.float64)
    floor = [t for t in record["tv_noise_floor"] if t is not None]
    tv_info = [t for t in record["tv_info"] if t is not None]
    sig = np.asarray(record["top_sigma_m"], dtype=np.float64)
    conv = sig < sigma_pass
    flips_all = np.asarray([f for f in record["flip_rate"][1:] if f is not None], dtype=np.float64)
    flips_conv = np.asarray([f for f, a, b in zip(record["flip_rate"][1:], conv[:-1], conv[1:]) if f is not None and a and b], dtype=np.float64)
    n_valid = np.asarray(record["n_valid"], dtype=int)
    steps = record["steps"]
    mean = lambda a: float(a.mean()) if a.size else 0.0   # noqa: E731
    return {
        "n_snapshots": int(tv.shape[0]),
        "tv_median": float(np.median(tv)), "tv_max": float(tv.max()), "tv_min": float(tv.min()), "tv_final": float(tv[-1]),
        "tv_fraction_below_max": float(np.mean(tv < tv_max)),
        "tv_integrated_median": float(np.median(tvi)), "tv_integrated_max": float(tvi.max()), "tv_integrated_final": float(tvi[-1]),
        "tv_integrated_fraction_below_max": float(np.mean(tvi < tv_max)),
        "tv_noise_floor_median": None if not floor else float(np.median(floor)),
        "tv_noise_floor_final": None if not floor else float(floor[-1]),
        "tv_converged_median": float(np.median(tv[conv])) if conv.any() else None,
        "tv_integrated_converged_median": float(np.median(tvi[conv])) if conv.any() else None,
        "tv_info_median": None if not tv_info else float(np.median(tv_info)),
        "tv_info_max": None if not tv_info else float(np.max(tv_info)),
        "flip_rate_mean": mean(flips_all), "flip_rate_max": float(flips_all.max()) if flips_all.size else 0.0,
        "n_flip_events": int(np.sum(flips_all > 0.0)), "n_transitions": int(flips_all.size),
        "flip_rate_mean_converged": mean(flips_conv), "n_flip_events_converged": int(np.sum(flips_conv > 0.0)),
        "n_transitions_converged": int(flips_conv.size),
        "n_valid_median": float(np.median(n_valid)), "n_valid_min": int(n_valid.min()), "n_valid_max": int(n_valid.max()),
        "n_valid_final": int(n_valid[-1]), "n_valid_by_step": [[int(s), int(n)] for s, n in zip(steps, n_valid)],
        "first_step_single_component": _first(steps, n_valid == 1),
        "top_sigma_final_m": float(sig[-1]), "map_error_final_m": float(record["map_error_m"][-1]),
        "first_success_step": record["first_success_step"],
        "first_step_sigma_below": _first(steps, conv),
        "first_step_error_below": _first(steps, np.asarray(record["map_error_m"]) < config.SUCCESS_ERROR_M),
        "n_converged_snapshots": int(conv.sum()),
        "pass_tv": bool(np.median(tv) < tv_max), "pass_flip": bool(mean(flips_all) < flip_max),
        "pass_tv_integrated": bool(np.median(tvi) < tv_max), "pass_flip_converged": bool(mean(flips_conv) < flip_max),
    }


def verdict(summaries: dict[str, dict], tv_max: float = config.T1_5_TV_MAX,
            flip_max: float = config.T1_5_FLIP_MAX) -> dict:
    """Strict T1-5 verdict (plan S1 T1-5): PASS iff median TV < tv_max and mean flip rate < flip_max for every
    source; 'informational' applies the same thresholds to tv_integrated and the converged-phase flip rate."""
    def block(tv_key: str, flip_key: str) -> dict:
        per = {s: {"tv_median": v[tv_key], "flip_rate_mean": v[flip_key],
                   "pass": bool(v[tv_key] < tv_max and v[flip_key] < flip_max)} for s, v in summaries.items()}
        return {"tv_key": tv_key, "flip_key": flip_key, "per_source": per,
                "n_pass": int(sum(p["pass"] for p in per.values())), "n_sources": len(per),
                "overall_pass": bool(per) and all(p["pass"] for p in per.values())}
    # strict (plan T1-5): TV on the integrated cell mass and order flips over the CONVERGED phase only - on the
    # near-uniform prior the component order is arbitrary, so pre-convergence flips are reported as informational
    strict = block("tv_median", "flip_rate_mean_converged")
    info = block("tv_integrated_median", "flip_rate_mean")
    return {"tv_max": tv_max, "flip_max": flip_max, **strict, "informational": info}


def nearest_snapshot_indices(steps: Sequence[int], wanted: Sequence[int] = config.T1_5_FIG_STEPS) -> list[int]:
    """Index of the saved snapshot nearest to each wanted RL step (ties -> the earlier snapshot)."""
    st = np.asarray(steps, dtype=np.int64)
    return [int(np.argmin(np.abs(st - int(w)))) for w in wanted]


# ---------------------------------------------------------------------------------------- figure
def _ellipse_patch(mean: np.ndarray, cov: np.ndarray, n_sigma: float, **kw) -> Ellipse:
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    vals = np.maximum(vals, 1e-12)
    angle = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))     # major axis = largest eigenvalue (last)
    return Ellipse(xy=(float(mean[0]), float(mean[1])), width=2.0 * n_sigma * np.sqrt(vals[1]),
                   height=2.0 * n_sigma * np.sqrt(vals[0]), angle=angle, fill=False, **kw)


def _draw_core(ax, xy: np.ndarray, w: np.ndarray, gmm: GmmSummary, true_xy: np.ndarray, map_xy: np.ndarray,
               sigmas: Sequence[float], marker_scale: float = 1.0) -> None:
    """Particles (coloured by weight), GMM ellipses, true source and MAP on `ax` (shared by panel and inset)."""
    order = np.argsort(w)                                             # heavy particles drawn last (on top)
    wn = w / w.max()
    ax.scatter(xy[order, 0], xy[order, 1], c=wn[order], cmap="viridis", vmin=0.0, vmax=1.0,
               s=config.T1_5_FIG_PARTICLE_SIZE * marker_scale, linewidths=0, alpha=0.8, rasterized=True)
    lw0, lw1 = config.T1_5_FIG_ELLIPSE_LW_M
    for wgt, m, c, ok in zip(gmm.weights, gmm.means, gmm.covs, gmm.mask):
        if not ok:
            continue
        for j, ns in enumerate(sigmas):
            ax.add_patch(_ellipse_patch(m, c, ns, edgecolor=COLOR_ELLIPSE, linewidth=lw0 + (lw1 - lw0) * float(wgt),
                                        linestyle="-" if j == 0 else "--", alpha=0.95))
    ax.plot(true_xy[0], true_xy[1], marker="*", markersize=13, color=COLOR_TRUE, markeredgecolor="white", linestyle="none")
    ax.plot(map_xy[0], map_xy[1], marker="x", markersize=9, markeredgewidth=2.0, color=COLOR_MAP, linestyle="none")


def draw_belief_panel(ax, xy: np.ndarray, logw: np.ndarray, gmm: GmmSummary, true_xy: np.ndarray, map_xy: np.ndarray,
                      drone_xy: np.ndarray, paths_so_far: np.ndarray | None, om: ObstacleMap | None,
                      half_width: float = config.T1_5_FIG_HALF_WIDTH_M,
                      sigmas: Sequence[float] = config.T1_5_FIG_ELLIPSE_SIGMAS,
                      inset_half_width: float = config.T1_5_FIG_INSET_HALF_WIDTH_M,
                      inset_sigma: float = config.T1_5_FIG_INSET_SIGMA_M) -> bool:
    """One figure-5 panel (see the module docstring); paths_so_far (t, n_drones, 2) or None.  Returns whether the
    zoom inset (top sigma < inset_sigma) was drawn."""
    true_xy = np.asarray(true_xy, dtype=np.float64); map_xy = np.asarray(map_xy, dtype=np.float64)
    x0, x1 = float(true_xy[0]) - half_width, float(true_xy[0]) + half_width
    y0, y1 = float(true_xy[1]) - half_width, float(true_xy[1]) + half_width
    if om is not None:
        ix = np.where((om.x >= x0 - om.res) & (om.x <= x1 + om.res))[0]
        iy = np.where((om.y >= y0 - om.res) & (om.y <= y1 + om.res))[0]
        if ix.size > 1 and iy.size > 1:
            sub = (om.hmap[ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1] > config.OCC_HMAP_NO_BUILDING).T.astype(float)
            ax.contourf(om.x[ix], om.y[iy], sub, levels=[0.5, 1.5], colors=[COLOR_BUILDING], alpha=0.18)
            ax.contour(om.x[ix], om.y[iy], sub, levels=[0.5], colors=[COLOR_BUILDING], linewidths=0.5)
    xy = np.asarray(xy, dtype=np.float64)
    w = normalised_weights(logw)
    _draw_core(ax, xy, w, gmm, true_xy, map_xy, sigmas)
    if paths_so_far is not None and paths_so_far.shape[0] > 0:
        for d in range(paths_so_far.shape[1]):
            ax.plot(paths_so_far[:, d, 0], paths_so_far[:, d, 1], color=COLOR_DRONE[d % len(COLOR_DRONE)], linewidth=0.8, alpha=0.7)
    for d in range(drone_xy.shape[0]):
        ax.plot(drone_xy[d, 0], drone_xy[d, 1], marker="^", markersize=7, color=COLOR_DRONE[d % len(COLOR_DRONE)],
                markeredgecolor="white", linestyle="none")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#e6e6e6", linewidth=0.5)
    drew_inset = bool(gmm.top_sigma() < inset_sigma)
    if drew_inset:
        # place the inset in the corner farthest from the MAP
        left = map_xy[0] > true_xy[0]
        low = map_xy[1] > true_xy[1]
        ins = ax.inset_axes([0.03 if left else 0.55, 0.03 if low else 0.55, 0.42, 0.42])
        ins.set_facecolor("white")
        _draw_core(ins, xy, w, gmm, true_xy, map_xy, sigmas, marker_scale=2.0)
        ins.set_xlim(map_xy[0] - inset_half_width, map_xy[0] + inset_half_width)
        ins.set_ylim(map_xy[1] - inset_half_width, map_xy[1] + inset_half_width)
        ins.set_aspect("equal")
        ins.set_xticks([]); ins.set_yticks([])
        ins.set_title(f"zoom +-{inset_half_width:.0f} m around MAP", fontsize=6.5, pad=2)
        for sp in ins.spines.values():
            sp.set_edgecolor(COLOR_ELLIPSE)
        ax.indicate_inset_zoom(ins, edgecolor=COLOR_ELLIPSE, linewidth=0.8)
    return drew_inset


def make_figure(snaps: dict[int, dict[str, np.ndarray]], records: dict[int, dict], gmms: dict[int, list[GmmSummary]],
                path: Path, om: ObstacleMap | None = None, paths: dict[int, np.ndarray] | None = None,
                fig_steps: Sequence[int] = config.T1_5_FIG_STEPS,
                half_width: float = config.T1_5_FIG_HALF_WIDTH_M) -> list[list[int]]:
    """Figure 5: rows = sources, columns = snapshots nearest fig_steps; returns the chosen snapshot steps per row."""
    sources = list(snaps)
    n_rows, n_cols = len(sources), len(fig_steps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols + 0.6, 4.9 * n_rows + 1.2), squeeze=False)
    chosen: list[list[int]] = []
    for r, s in enumerate(sources):
        snap, rec = snaps[s], records[s]
        steps = snap["steps"].astype(int)
        idx = nearest_snapshot_indices(steps, fig_steps)
        chosen.append([int(steps[i]) for i in idx])
        full = None if paths is None else paths.get(s)
        for c, i in enumerate(idx):
            ax = axes[r, c]
            step = int(steps[i])
            so_far = full[:step] if full is not None else snap["drone_xy"][: i + 1]
            draw_belief_panel(ax, snap["xy"][i], snap["logw"][i], gmms[s][i], snap["true_xy"], snap["map_xy"][i],
                              snap["drone_xy"][i], so_far, om, half_width=half_width)
            ax.set_title(f"source {s} ({source_type(s)}), RL step {step}\nMAP error {rec['map_error_m'][i]:.0f} m, "
                         f"top sigma {rec['top_sigma_m'][i]:.1f} m, TV {rec['tv'][i]:.2f}, K valid {rec['n_valid'][i]}", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("x [m]", fontsize=8)
            if c == 0:
                ax.set_ylabel("y [m]", fontsize=8)
    handles = [Line2D([], [], marker="*", color=COLOR_TRUE, markersize=11, linestyle="none", label="true source"),
               Line2D([], [], marker="x", color=COLOR_MAP, markersize=8, markeredgewidth=2, linestyle="none", label="MAP (weighted mode)"),
               Line2D([], [], color=COLOR_ELLIPSE, linewidth=1.5, label="GMM 1 sigma (solid) / 2 sigma (dashed); line width = weight"),
               Line2D([], [], marker="^", color=COLOR_DRONE[0], markersize=7, linestyle="-", linewidth=0.8, label="drone 1 + path so far"),
               Line2D([], [], marker="^", color=COLOR_DRONE[1], markersize=7, linestyle="-", linewidth=0.8, label="drone 2 + path so far"),
               Line2D([], [], marker="o", color="#440154", markersize=4, linestyle="none", label="particles (viridis: weight / max weight)"),
               Line2D([], [], color=COLOR_BUILDING, linewidth=1.0, label="building outline (hmap > 0)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.suptitle(f"Figure 5 - belief evolution of the RB-PF with the SPH adjoint model (filter {config.T1_4_SNAPSHOT_FILTER}, "
                 f"seed 0, N = {config.PF_N_PARTICLES}, 2-drone lawnmower); window +-{half_width:.0f} m around the true source",
                 fontsize=10)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIG_DPI_FINAL)
    fig.savefig(path.with_name(path.stem + "_preview" + path.suffix), dpi=config.FIG_DPI_PREVIEW)
    plt.close(fig)
    return chosen


# ---------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, nargs="*", default=list(config.T1_5_SNAPSHOT_SOURCES))
    ap.add_argument("--snapshot-dir", type=Path, default=config.CACHE_DIR)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_t1_5.json")
    ap.add_argument("--fig", type=Path, default=config.FIG_DIR / "fig5_belief_evolution.png")
    ap.add_argument("--no-obstacles", action="store_true", help="skip ObstacleMap (no building outlines / full paths)")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()

    om = None if args.no_obstacles else ObstacleMap.load()
    snaps: dict[int, dict[str, np.ndarray]] = {}
    records: dict[int, dict] = {}
    gmms: dict[int, list[GmmSummary]] = {}
    summaries: dict[str, dict] = {}
    paths: dict[int, np.ndarray] = {}
    for s in [int(x) for x in args.sources]:
        f = args.snapshot_dir / f"t1_4_snapshots_{s}.npz"
        t0 = time.perf_counter()
        snaps[s] = load_snapshot(f)
        records[s], gmms[s] = analyse_snapshots(snaps[s])
        sm = summarise(records[s])
        sm["seconds"] = time.perf_counter() - t0
        sm["snapshot_file"] = str(f)
        summaries[str(s)] = sm
        if om is not None:
            paths[s], _ = two_drone_paths(om, tuple(config.SOURCES_XY[s]), int(snaps[s]["steps"].max()))
        print(f"[source {s} {source_type(s)}] {sm['n_snapshots']} snapshots: TV median {sm['tv_median']:.3f} max {sm['tv_max']:.3f} "
              f"final {sm['tv_final']:.3f} (frac < {config.T1_5_TV_MAX} = {sm['tv_fraction_below_max']:.2f}); integrated median "
              f"{sm['tv_integrated_median']:.3f} final {sm['tv_integrated_final']:.3f}; noise floor median {sm['tv_noise_floor_median']:.3f} "
              f"final {sm['tv_noise_floor_final']:.3f}; {config.T1_5_TV_CELL_INFO_M:.0f} m cells median {sm['tv_info_median']:.3f}; "
              f"flip rate mean {sm['flip_rate_mean']:.3f} ({sm['n_flip_events']}/{sm['n_transitions']} transitions), converged phase "
              f"{sm['flip_rate_mean_converged']:.3f} ({sm['n_flip_events_converged']}/{sm['n_transitions_converged']}); K valid median "
              f"{sm['n_valid_median']:.0f} final {sm['n_valid_final']} (single from step {sm['first_step_single_component']}); top sigma final "
              f"{sm['top_sigma_final_m']:.1f} m, MAP error final {sm['map_error_final_m']:.1f} m, first success {sm['first_success_step']} "
              f"(sigma < {config.SUCCESS_SIGMA_M:.0f} m from step {sm['first_step_sigma_below']}, error < {config.SUCCESS_ERROR_M:.0f} m from "
              f"step {sm['first_step_error_below']}); {sm['seconds']:.1f} s", flush=True)
    verd = verdict(summaries)
    chosen = make_figure(snaps, records, gmms, args.fig, om=om, paths=paths or None)
    res = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": list(snaps), "source_types": {str(s): source_type(s) for s in snaps},
        "snapshot_filter": config.T1_4_SNAPSHOT_FILTER, "snapshot_seed": 0, "gmm_every": config.T1_4_GMM_EVERY,
        "gmm": {"k": config.GMM_K, "em_iters": config.GMM_EM_ITERS, "min_weight": config.GMM_MIN_WEIGHT,
                "merge_bhat": config.GMM_MERGE_BHAT, "seed": config.T1_5_GMM_SEED, "vector_dim": config.GMM_VECTOR_DIM},
        "criteria": {"tv_cell_m": config.T1_5_TV_CELL_M, "tv_cell_info_m": config.T1_5_TV_CELL_INFO_M, "tv_n_sub": config.T1_5_TV_N_SUB,
                     "tv_max": config.T1_5_TV_MAX, "flip_max": config.T1_5_FLIP_MAX, "match_radius_m": config.T1_5_MATCH_RADIUS_M,
                     "success_sigma_m": config.SUCCESS_SIGMA_M, "success_error_m": config.SUCCESS_ERROR_M},
        "n_particles": int(snaps[next(iter(snaps))]["xy"].shape[1]) if snaps else None,
        "summaries": summaries, "verdict": verd,
        "figure": {"path": str(args.fig), "wanted_steps": list(config.T1_5_FIG_STEPS), "chosen_steps": {str(s): c for s, c in zip(snaps, chosen)},
                   "half_width_m": config.T1_5_FIG_HALF_WIDTH_M},
        "per_step": {str(s): {k: v for k, v in r.items() if k != "vectors"} for s, r in records.items()},
        "vectors": {str(s): r["vectors"] for s, r in records.items()},
        "total_seconds": time.perf_counter() - t_start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    info = verd["informational"]
    print(f"T1-5 strict verdict: {verd['n_pass']}/{verd['n_sources']} sources pass (median TV < {verd['tv_max']}, mean flip < {verd['flip_max']}) -> "
          f"{'PASS' if verd['overall_pass'] else 'FAIL'}; informational (integrated TV, converged-phase flips): {info['n_pass']}/{info['n_sources']} -> "
          f"{'PASS' if info['overall_pass'] else 'FAIL'}; total {res['total_seconds']:.0f} s; JSON {args.out}; figure {args.fig}")
    return res


if __name__ == "__main__":
    main()