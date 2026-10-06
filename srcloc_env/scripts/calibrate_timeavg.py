"""T1-3c / D7-2b: time-averaged 15 m slab (frames config.T1_3C_FRAME_RANGE = 400..599) -> concentration-fluctuation
statistics (implied negative-binomial dispersion r) and re-evaluation of both forward models against the MEAN field
(plan S1 T1-3 / T1-4, 4.3 강건화; validation_log '결정·주의 사항' "G1 FAIL 원인과 치료"; R22 Yee & Chan 1997, R23 Hilbe 2011).

Usage: python -m srcloc_env.scripts.calibrate_timeavg [--lo 400 --hi 599] [--reuse-npz]
       [--npz-out <CACHE_DIR>/slab_timeavg_400_599.npz] [--out <CACHE_DIR>/calibrate_timeavg.json]
       [--adjoint <CACHE_DIR>/calibrate_adjoint.json] [--analytic <CACHE_DIR>/calibrate_forward.json] [--fig-dir <FIG_DIR>]

Why (validation_log G1 FAIL 원인과 치료)
--------------------------------------
The T1-4 filters collapse because the frozen LDM snapshot carries turbulent clumps (dense-cell residual std ~1.0-1.6)
that enter the Poisson likelihood as independent evidence every step.  The D7-1 remedy is a Gamma-Poisson (negative
binomial) count likelihood whose dispersion r was set from the SPATIAL residual of one frame (CV ~1.8 -> r ~0.3).
This script measures the TEMPORAL fluctuation of the field itself: per cell, over the 200 Mode F frames, and asks
(1) what CV the field really has (with and without the ~40 % growth trend of T0-4), (2) which r that implies
(R23: Var = mu + mu^2 / r, r = 1 / CV^2 for gamma-type fluctuations, R22), and (3) how much of the T1-3 / T1-3b
"shape mismatch" of the forward models is fluctuation (disappears against the time mean) versus structure (stays).

Method
------
(1) time average   per source, over frames lo..hi (inclusive), the z = config.DRONE_Z slab (z_levels[0] of these
    frames; LdmSlabBackend.slab(i).z_index(DRONE_Z)); streaming float64 sums (TimeAverageAccumulator) ->
    mean, std (population, ddof 0), CV = std / mean where mean > 0 (NaN elsewhere).  Trend removal: each frame is
    divided by its own source total mass (sum over the slab cells) before averaging -> mean_norm, std_norm,
    CV_norm.  A constant field gives CV = CV_norm = 0; frames that are pure scalings of one field give
    CV > 0 but CV_norm = 0 (tests/test_calibrate_timeavg.py).  Saved to config.CACHE_DIR /
    T1_3C_TIMEAVG_NPZ_TEMPLATE (keys mean, std, cv, cv_norm, mean_norm, std_norm (13, ny, nx) float32; sources,
    grid [x0, y0, nx, ny, res], frames [lo, hi], totals (T, 13), z).
(2) fluctuation    dense cells = mean >= config.T1_3C_DENSE_FRACTION x max(mean) of the source (the T1-3 dense rule);
    median / IQR of CV and CV_norm over them; implied r = 1 / median(CV_norm)^2 per source (also 1 / mean(CV_norm^2))
    and pooled over config.T1_3C_POOLED_SOURCES (all dense cells of those sources together); compared with
    config.PF_NB_DISPERSION_R.
(3) forward models vs the MEAN field   with the T1-3b / T1-3 dense-cell shape statistic (calibrate_adjoint.
    shape_stats_adjoint via evaluate_operator; calibrate_forward.shape_stats via evaluate_sources):
    adjoint at every (K, lam) in config.T1_3C_ADJ_K_CANDIDATES x T1_3C_ADJ_LAMBDA_CANDIDATES (single 15 m wind,
    one factorisation per combination, evaluated against BOTH the instantaneous frame hi and the mean field);
    analytic plume at the chosen T1-3 combination (config.T1_4_ANALYTIC_U / _SIGMA_V, global wind).  Per source:
    std_dense instantaneous (calibrate_adjoint.json / calibrate_forward.json per_source_chosen, cross-checked
    against the recomputation) vs mean-field; implied kappa (median rho -> calibrate_forward.implied_kappa over
    TRAIN_SOURCES) for both models; peak / centroid offset of the mean-field maximum from the source position
    (distance, direction) for config.T1_3C_OFFSET_SOURCES (the '109 systematic offset' hypothesis).

Outputs
-------
config.CACHE_DIR / calibrate_timeavg.json, the npz above, config.FIG_DIR / fig_timeavg_calibration.png (+ _preview):
(a) source T1_3C_FIG_SOURCE instantaneous frame hi, (b) time mean, (c) adjoint model g x exp(median rho | mean)
at the chosen (K, lam) (log10, one colour scale, building outlines, source star, window config.T1_3_FIG_WINDOW_M);
(d) per-source bars of adjoint std_dense instantaneous vs mean (analytic mean-field value as a marker).

References: R22 (gamma-type concentration fluctuations), R23 (NB dispersion), R15 / R17 / R21 (adjoint model).
Helpers ``TimeAverageAccumulator``, ``timeavg_stats``, ``dense_fluctuation_stats``, ``pooled_dispersion``,
``peak_offset`` are pure numpy and unit-tested on synthetic frames in tests/test_calibrate_timeavg.py.
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
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from srcloc_env import config  # noqa: E402
from srcloc_env.env.drone import ObstacleMap  # noqa: E402
from srcloc_env.field.concentration_field import LdmSlabBackend  # noqa: E402
from srcloc_env.field.wind import WindField  # noqa: E402
from srcloc_env.pf.sph_adjoint import AdjointParams, AdvectionDiffusionOperator  # noqa: E402
from srcloc_env.preprocess.gridder import SlabGrid  # noqa: E402
from srcloc_env.scripts import calibrate_adjoint as ca  # noqa: E402
from srcloc_env.scripts import calibrate_forward as cf  # noqa: E402

COLOR_ACCENT = ca.COLOR_ACCENT
COLOR_ANALYTIC_MARK = "#333333"
CMAP_DENSITY = ca.CMAP_DENSITY


# ------------------------------------------------------------------------------------------ (1) time average
@dataclass(frozen=True)
class TimeAverage:
    """Per-cell statistics over T frames of shape (n_src, ny, nx) (module docstring (1)).

    mean / std / cv          raw frames (population std; cv = std / mean where mean > 0, NaN elsewhere)
    mean_norm / std_norm / cv_norm   frames divided by their own per-source total mass (trend removed)
    totals (T, n_src)        per-frame source totals (sum over cells) in the frame units
    n_frames                 T
    """

    mean: np.ndarray
    std: np.ndarray
    cv: np.ndarray
    mean_norm: np.ndarray
    std_norm: np.ndarray
    cv_norm: np.ndarray
    totals: np.ndarray
    n_frames: int


def _cv(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = np.full(mean.shape, np.nan)
    pos = mean > 0.0
    out[pos] = std[pos] / mean[pos]
    return out


class TimeAverageAccumulator:
    """Streaming float64 moments of frames (n_src, ...) so that 200 frames never sit in memory at once.

    add(frame) updates Welford running mean / M2 (numerically stable: identical frames give exactly std = 0) of
    the raw frame and of the frame normalised by its per-source total (sum over all non-source axes); a source
    with zero total contributes a zero normalised frame.  result() returns the TimeAverage (population variance
    M2 / n).
    """

    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(int(v) for v in shape)
        if len(self.shape) < 2 or any(v <= 0 for v in self.shape):
            raise ValueError(f"shape must be (n_src, ...) with positive sizes, got {self.shape}")
        self.n = 0
        self._mean = np.zeros(self.shape, dtype=np.float64)
        self._m2 = np.zeros(self.shape, dtype=np.float64)
        self._mean_n = np.zeros(self.shape, dtype=np.float64)
        self._m2_n = np.zeros(self.shape, dtype=np.float64)
        self._totals: list[np.ndarray] = []

    @staticmethod
    def _welford(mean: np.ndarray, m2: np.ndarray, x: np.ndarray, n_new: int) -> None:
        delta = x - mean
        mean += delta / n_new
        m2 += delta * (x - mean)

    def add(self, frame: np.ndarray) -> None:
        f = np.asarray(frame, dtype=np.float64)
        if f.shape != self.shape:
            raise ValueError(f"frame shape {f.shape} != accumulator shape {self.shape}")
        if (f < 0.0).any():
            raise ValueError("frames must be non-negative densities")
        tot = f.reshape(self.shape[0], -1).sum(axis=1)                        # (n_src,)
        safe = np.where(tot > 0.0, tot, 1.0).reshape((-1,) + (1,) * (f.ndim - 1))
        self.n += 1
        self._welford(self._mean, self._m2, f, self.n)
        self._welford(self._mean_n, self._m2_n, f / safe, self.n)
        self._totals.append(tot)

    def result(self) -> TimeAverage:
        if self.n == 0:
            raise ValueError("no frames accumulated")
        T = float(self.n)
        mean = self._mean.copy()
        std = np.sqrt(np.maximum(self._m2 / T, 0.0))
        mean_n = self._mean_n.copy()
        std_n = np.sqrt(np.maximum(self._m2_n / T, 0.0))
        return TimeAverage(mean, std, _cv(mean, std), mean_n, std_n, _cv(mean_n, std_n),
                           np.array(self._totals, dtype=np.float64), int(self.n))


def timeavg_stats(frames: np.ndarray) -> TimeAverage:
    """TimeAverage of an in-memory stack (T, n_src, ...) (convenience wrapper over the accumulator)."""
    fr = np.asarray(frames, dtype=np.float64)
    if fr.ndim < 3:
        raise ValueError(f"frames must be (T, n_src, ...), got {fr.shape}")
    acc = TimeAverageAccumulator(fr.shape[1:])
    for f in fr:
        acc.add(f)
    return acc.result()


# ------------------------------------------------------------------------------------------ (2) fluctuation
def _iqr_summary(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": float("nan"), "q25": float("nan"), "q75": float("nan"), "iqr": float("nan"),
                "mean": float("nan")}
    q25, med, q75 = np.percentile(v, [25.0, 50.0, 75.0])
    return {"n": int(v.size), "median": float(med), "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25),
            "mean": float(v.mean())}


def implied_dispersion(cv: np.ndarray) -> dict:
    """NB dispersion implied by a CV sample (R23, Var = mu + mu^2 / r; gamma-type fluctuation R22): r = 1 / median(CV)^2
    and, as a second estimate, 1 / mean(CV^2).  NaN when no finite CV is available; inf when the CV is 0."""
    v = np.asarray(cv, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"r_median": float("nan"), "r_mean_cv2": float("nan"), "n": 0}
    med = float(np.median(v))
    m2 = float(np.mean(v * v))
    return {"r_median": float(1.0 / med ** 2) if med > 0.0 else float("inf"),
            "r_mean_cv2": float(1.0 / m2) if m2 > 0.0 else float("inf"), "n": int(v.size)}


def dense_mask(mean: np.ndarray, dense_fraction: float = config.T1_3C_DENSE_FRACTION) -> np.ndarray:
    """Cells with mean >= dense_fraction x max(mean) (and mean > 0) of one source (ny, nx) -> bool (ny, nx)."""
    m = np.asarray(mean, dtype=np.float64)
    mx = float(m.max()) if m.size else 0.0
    if mx <= 0.0:
        return np.zeros(m.shape, dtype=bool)
    return (m >= float(dense_fraction) * mx) & (m > 0.0)


def dense_fluctuation_stats(mean: np.ndarray, cv: np.ndarray, cv_norm: np.ndarray,
                            dense_fraction: float = config.T1_3C_DENSE_FRACTION) -> dict:
    """Module docstring (2) for one source: dense-cell summaries of CV and CV_norm and the implied r."""
    d = dense_mask(mean, dense_fraction)
    cvd = np.asarray(cv, dtype=np.float64)[d]
    cvn = np.asarray(cv_norm, dtype=np.float64)[d]
    return {"n_dense": int(d.sum()), "max_mean": float(np.asarray(mean).max()) if np.asarray(mean).size else 0.0,
            "cv": _iqr_summary(cvd), "cv_norm": _iqr_summary(cvn),
            "r_implied": implied_dispersion(cvn), "r_implied_raw_cv": implied_dispersion(cvd)}


def pooled_dispersion(ta: TimeAverage, src_ids: tuple[int, ...], pool: tuple[int, ...],
                      dense_fraction: float = config.T1_3C_DENSE_FRACTION) -> dict:
    """All dense cells of the sources in ``pool`` together: CV / CV_norm summaries and the implied r."""
    cvs, cvns = [], []
    for s in pool:
        i = src_ids.index(s)
        d = dense_mask(ta.mean[i], dense_fraction)
        cvs.append(ta.cv[i][d])
        cvns.append(ta.cv_norm[i][d])
    cvd = np.concatenate(cvs) if cvs else np.array([])
    cvn = np.concatenate(cvns) if cvns else np.array([])
    return {"sources": list(pool), "n_dense": int(cvn.size), "cv": _iqr_summary(cvd), "cv_norm": _iqr_summary(cvn),
            "r_implied": implied_dispersion(cvn), "r_implied_raw_cv": implied_dispersion(cvd)}


# ------------------------------------------------------------------------------------------ (3) offsets
def peak_offset(field: np.ndarray, grid: SlabGrid, src_xy: tuple[float, float]) -> dict:
    """Offset of the field maximum and of the mass centroid from the source (ny, nx field on ``grid``).

    distance [m] and direction [deg, atan2(dy, dx), 0 = +x] of (peak - source) and (centroid - source);
    NaN when the field is all zero.
    """
    f = np.asarray(field, dtype=np.float64)
    if f.shape != (grid.ny, grid.nx):
        raise ValueError(f"field {f.shape} must be (ny, nx) = ({grid.ny}, {grid.nx})")
    xs, ys = float(src_xy[0]), float(src_xy[1])
    nan = float("nan")
    if not (f > 0.0).any():
        return {"peak_xy": [nan, nan], "peak_value": 0.0, "peak_dx": nan, "peak_dy": nan, "peak_distance_m": nan,
                "peak_direction_deg": nan, "centroid_xy": [nan, nan], "centroid_distance_m": nan,
                "centroid_direction_deg": nan}
    iy, ix = np.unravel_index(int(np.argmax(f)), f.shape)
    px, py = float(grid.x_centres[ix]), float(grid.y_centres[iy])
    xx, yy = np.meshgrid(grid.x_centres, grid.y_centres)
    w = f / f.sum()
    cx, cy = float((w * xx).sum()), float((w * yy).sum())
    return {"peak_xy": [px, py], "peak_value": float(f[iy, ix]), "peak_dx": px - xs, "peak_dy": py - ys,
            "peak_distance_m": float(np.hypot(px - xs, py - ys)),
            "peak_direction_deg": float(np.rad2deg(np.arctan2(py - ys, px - xs))),
            "centroid_xy": [cx, cy], "centroid_distance_m": float(np.hypot(cx - xs, cy - ys)),
            "centroid_direction_deg": float(np.rad2deg(np.arctan2(cy - ys, cx - xs)))}

# ------------------------------------------------------------------------------------------ (3) adjoint grid
def evaluate_adjoint_neighbourhood(grid: SlabGrid, uv: np.ndarray, blocked: np.ndarray, src_ids: tuple[int, ...],
                                   src_xy: np.ndarray, truths: dict[str, np.ndarray],
                                   k_candidates: tuple[float, ...] = config.T1_3C_ADJ_K_CANDIDATES,
                                   lam_candidates: tuple[float, ...] = config.T1_3C_ADJ_LAMBDA_CANDIDATES,
                                   train_ids: tuple[int, ...] = config.TRAIN_SOURCES,
                                   z: float = config.DRONE_Z, h_layer: float = config.ADJ_H_LAYER,
                                   keep_fields: tuple[float, float] | None = None) -> tuple[list[dict], np.ndarray | None]:
    """One factorisation per (K, lam) (single wind ``uv``), evaluated against every truth in ``truths`` (name ->
    (n_src, M) slabs).  Rows: K, lam, seconds, per truth: mean_train_std_dense / std / median_rho and per_source
    AdjointShapeStats.  ``keep_fields`` = (K, lam) whose unit fields g (n_src, M) are returned (figure panel)."""
    train_idx = [i for i, s in enumerate(src_ids) if s in train_ids]
    rows: list[dict] = []
    kept = None
    for K, lam in product(k_candidates, lam_candidates):
        params = AdjointParams(K=float(K), lam=float(lam), h_layer=float(h_layer), z=float(z), wind_band=None)
        t0 = time.perf_counter()
        op = AdvectionDiffusionOperator.from_arrays(grid, uv, blocked, params).factorize()
        t1 = time.perf_counter()
        row = {"K": float(K), "lam": float(lam), "seconds_assemble_factorize": t1 - t0, "per_truth": {}}
        for name, slabs in truths.items():
            stats, _, g = ca.evaluate_operator(op, src_xy, slabs)
            tr = [stats[i] for i in train_idx]
            row["per_truth"][name] = {
                "mean_train_std_dense": ca._mean(tr, "std_dense"), "mean_train_std": ca._mean(tr, "std"),
                "mean_train_median_rho": ca._mean(tr, "median_rho"),
                "per_source": {str(s): asdict(st) for s, st in zip(src_ids, stats)}}
            if keep_fields is not None and (float(K), float(lam)) == tuple(map(float, keep_fields)) and kept is None:
                kept = g
        row["seconds_solves"] = time.perf_counter() - t1
        rows.append(row)
    return rows, kept


# ------------------------------------------------------------------------------------------ figure
def render_figure(grid: SlabGrid, omap: ObstacleMap, s: int, inst: np.ndarray, mean: np.ndarray, model: np.ndarray,
                  bars: dict[str, dict], src_ids: tuple[int, ...], combo: dict, frames: tuple[int, int],
                  wind: dict, out_png: Path, out_preview: Path) -> dict:
    """(a) instantaneous, (b) time mean, (c) adjoint model of source s on one log10 scale; (d) std_dense bars."""
    extent = (grid.x0, grid.x0 + grid.nx * grid.res, grid.y0, grid.y0 + grid.ny * grid.res)
    occ_img = (omap.hmap > config.OCC_HMAP_NO_BUILDING).T.astype(np.float32)
    xs, ys = config.SOURCES_XY[s]
    vmax = float(max(inst.max(), mean.max()))
    vmin = vmax / 10.0 ** config.FIG_LOG_DECADES
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
    panels = ((axes[0, 0], inst, f"(a) source {s}: instantaneous LDM slab, frame {frames[1]} (step {config.index_to_step(frames[1])})"),
              (axes[0, 1], mean, f"(b) source {s}: time mean over frames {frames[0]}-{frames[1]} ({frames[1] - frames[0] + 1} frames)"),
              (axes[1, 0], model, f"(c) source {s}: adjoint model g x exp(median rho | mean), K {combo['K']:g} m$^2$/s, "
                                  f"lambda {combo['lam']:g} 1/s"))
    a = np.deg2rad(wind["dir_deg"])
    L = config.T1_3_FIG_ARROW_M
    for ax, arr, title in panels:
        im = ax.imshow(np.ma.masked_where(arr < vmin, arr), extent=extent, origin="lower", cmap=CMAP_DENSITY,
                       norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest", zorder=2)
        ax.contour(omap.x, omap.y, occ_img, levels=[0.5], colors="black", linewidths=0.6, zorder=4)
        ax.scatter([xs], [ys], marker="*", s=200, c=COLOR_ACCENT, edgecolors="white", linewidths=0.8, zorder=6)
        ax.text(xs - 8, ys + 8, str(s), fontsize=9, ha="right", va="bottom", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        ax.annotate("", xy=(xs + L * np.cos(a), ys + L * np.sin(a)), xytext=(xs, ys), zorder=8,
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black", shrinkA=0, shrinkB=0))
        ax.text(xs + 0.5 * L * np.cos(a), ys + 0.5 * L * np.sin(a) - 14,
                f"SPH wind at source {wind['speed']:.2f} m/s, {wind['dir_deg']:.0f} deg", fontsize=7.5, ha="center",
                va="top", zorder=8, path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        x0, x1, y0, y1 = ca._window(xs, ys, grid)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title, fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(f"particles / m$^3$ (log10, {config.FIG_LOG_DECADES:g} decades, one scale)")
    # (d) bars
    ax = axes[1, 1]
    n = len(src_ids)
    x = np.arange(n)
    wbar = 0.38
    a_inst = np.array([bars[str(t)]["adjoint_std_dense_instantaneous"] for t in src_ids], dtype=np.float64)
    a_mean = np.array([bars[str(t)]["adjoint_std_dense_mean"] for t in src_ids], dtype=np.float64)
    an_mean = np.array([bars[str(t)]["analytic_std_dense_mean"] for t in src_ids], dtype=np.float64)
    cols = [ca.TYPE_COLORS[ca._type_key(t)] for t in src_ids]
    ax.bar(x - wbar / 2, np.nan_to_num(a_inst), wbar, color="white", edgecolor=cols, hatch="////", linewidth=1.2, zorder=3)
    ax.bar(x + wbar / 2, np.nan_to_num(a_mean), wbar, color=cols, edgecolor="white", linewidth=1.0, zorder=3)
    ax.scatter(x, np.nan_to_num(an_mean), marker="x", s=28, c=COLOR_ANALYTIC_MARK, zorder=5)
    for xi, (vi, vm) in enumerate(zip(a_inst, a_mean)):
        if np.isfinite(vi):
            ax.text(xi - wbar / 2, vi + 0.03, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.5, color="#333333")
        if np.isfinite(vm):
            ax.text(xi + wbar / 2, vm + 0.03, f"{vm:.2f}", ha="center", va="bottom", fontsize=6.5, color="#333333")
    ax.axhline(config.T1_3_STD_PASS, color="#777777", lw=1.0, ls="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n" + cf.source_type(t).replace("(", "\n(") for t in src_ids], fontsize=7)
    ax.set_ylabel("adjoint std(rho') on dense cells")
    ax.set_xlabel("LDM source")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor="white", edgecolor="#555555", hatch="////", label=f"vs instantaneous frame {frames[1]}"),
               Patch(facecolor="#555555", label=f"vs time mean {frames[0]}-{frames[1]}"),
               Line2D([0], [0], marker="x", color=COLOR_ANALYTIC_MARK, lw=0, label="analytic plume vs time mean"),
               Line2D([0], [0], color="#777777", lw=1.0, ls="--", label=f"plan criterion std(rho') < {config.T1_3_STD_PASS:g}")]
    handles += [Patch(facecolor=ca.TYPE_COLORS[k], label=f"{k} source") for k in ("open", "trapped", "holdout", "train")]
    ax.legend(handles=handles, fontsize=7, loc="upper left", ncol=2, frameon=False)
    ax.set_ylim(0, float(np.nanmax(np.r_[a_inst, a_mean, an_mean])) * 1.45)
    ax.set_title(f"(d) adjoint shape residual per source (K {combo['K']:g}, lambda {combo['lam']:g}): instantaneous vs time mean",
                 fontsize=9.5)
    fig.suptitle(f"Time-averaged calibration (T1-3c / D7-2b): source {s} at z = {config.DRONE_Z:g} m, frames {frames[0]}-{frames[1]}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=config.FIG_DPI_FINAL, bbox_inches="tight")
    fig.savefig(out_preview, dpi=config.FIG_DPI_PREVIEW, bbox_inches="tight")
    plt.close(fig)
    return {"log_vmin": vmin, "log_vmax": vmax,
            "files": {p.name: {"path": str(p), "bytes": p.stat().st_size} for p in (out_png, out_preview)}}


# ------------------------------------------------------------------------------------------ npz i/o
def build_time_average(be: LdmSlabBackend, lo: int, hi: int, z: float = config.DRONE_Z) -> tuple[TimeAverage, SlabGrid, tuple[int, ...]]:
    """Stream the z slab of frames lo..hi through the accumulator; returns (TimeAverage, grid, source ids)."""
    if hi < lo:
        raise ValueError("hi must be >= lo")
    acc = None
    grid = None
    src_ids: tuple[int, ...] = ()
    for i in range(lo, hi + 1):
        sf = be.slab(i)
        zi = sf.z_index(z)
        if acc is None:
            grid = sf.grid
            src_ids = tuple(int(s) for s in sf.sources)
            acc = TimeAverageAccumulator((len(src_ids), grid.ny, grid.nx))
        elif tuple(int(s) for s in sf.sources) != src_ids or sf.grid != grid:
            raise ValueError(f"frame {i}: sources / grid differ from frame {lo}")
        acc.add(sf.density[:, zi])
    assert acc is not None and grid is not None
    return acc.result(), grid, src_ids


def save_time_average(path: Path, ta: TimeAverage, grid: SlabGrid, src_ids: tuple[int, ...], frames: tuple[int, int],
                      z: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=ta.mean.astype(np.float32), std=ta.std.astype(np.float32),
                        cv=ta.cv.astype(np.float32), mean_norm=ta.mean_norm.astype(np.float32),
                        std_norm=ta.std_norm.astype(np.float32), cv_norm=ta.cv_norm.astype(np.float32),
                        sources=np.array(src_ids, dtype=np.int16),
                        grid=np.array([grid.x0, grid.y0, grid.nx, grid.ny, grid.res], dtype=np.float64),
                        frames=np.array(frames, dtype=np.int64), totals=ta.totals.astype(np.float64), z=np.float64(z))


def load_time_average(path: Path) -> tuple[TimeAverage, SlabGrid, tuple[int, ...], tuple[int, int]]:
    with np.load(path) as npz:
        ta = TimeAverage(npz["mean"].astype(np.float64), npz["std"].astype(np.float64), npz["cv"].astype(np.float64),
                         npz["mean_norm"].astype(np.float64), npz["std_norm"].astype(np.float64),
                         npz["cv_norm"].astype(np.float64), npz["totals"].astype(np.float64), int(npz["totals"].shape[0]))
        g = npz["grid"]
        src_ids = tuple(int(v) for v in npz["sources"])
        frames = (int(npz["frames"][0]), int(npz["frames"][1]))
    grid = SlabGrid(x0=float(g[0]), y0=float(g[1]), nx=int(round(g[2])), ny=int(round(g[3])), res=float(g[4]))
    return ta, grid, src_ids, frames

# ------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lo", type=int, default=config.T1_3C_FRAME_RANGE[0])
    ap.add_argument("--hi", type=int, default=config.T1_3C_FRAME_RANGE[1])
    ap.add_argument("--npz-out", type=Path, default=None)
    ap.add_argument("--reuse-npz", action="store_true", help="skip the frame pass when the npz already exists")
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "calibrate_timeavg.json")
    ap.add_argument("--adjoint", type=Path, default=config.CACHE_DIR / "calibrate_adjoint.json")
    ap.add_argument("--analytic", type=Path, default=config.CACHE_DIR / "calibrate_forward.json")
    ap.add_argument("--fig-dir", type=Path, default=config.FIG_DIR)
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    lo, hi = int(args.lo), int(args.hi)
    npz_path = args.npz_out or (config.CACHE_DIR / config.T1_3C_TIMEAVG_NPZ_TEMPLATE.format(lo=lo, hi=hi))
    z = config.DRONE_Z

    # ---- (1) time average ----------------------------------------------------------------------------
    be = LdmSlabBackend(config.CACHE_DIR, max_cached_frames=1)
    t0 = time.perf_counter()
    if args.reuse_npz and npz_path.exists():
        ta, grid, src_ids, frames = load_time_average(npz_path)
        if frames != (lo, hi):
            raise ValueError(f"{npz_path} holds frames {frames}, requested ({lo}, {hi})")
        built = False
    else:
        ta, grid, src_ids = build_time_average(be, lo, hi, z)
        save_time_average(npz_path, ta, grid, src_ids, (lo, hi), z)
        built = True
    avg_s = time.perf_counter() - t0
    n_src = len(src_ids)
    src_xy = np.array([config.SOURCES_XY[s] for s in src_ids], dtype=np.float64)
    M = grid.ny * grid.nx
    growth = {str(s): float(ta.totals[-1, i] / ta.totals[0, i]) if ta.totals[0, i] > 0 else float("nan")
              for i, s in enumerate(src_ids)}
    growth["total"] = float(ta.totals[-1].sum() / ta.totals[0].sum())

    # ---- (2) fluctuation statistics ------------------------------------------------------------------
    fluct = {str(s): dense_fluctuation_stats(ta.mean[i], ta.cv[i], ta.cv_norm[i]) for i, s in enumerate(src_ids)}
    pooled = pooled_dispersion(ta, src_ids, config.T1_3C_POOLED_SOURCES)
    pooled_all = pooled_dispersion(ta, src_ids, src_ids)
    r_default = config.PF_NB_DISPERSION_R
    dispersion = {
        "default_PF_NB_DISPERSION_R": r_default,
        "per_source_r_median": {s: v["r_implied"]["r_median"] for s, v in fluct.items()},
        "per_source_cv_norm_median": {s: v["cv_norm"]["median"] for s, v in fluct.items()},
        "per_source_cv_median": {s: v["cv"]["median"] for s, v in fluct.items()},
        "pooled_open": pooled, "pooled_all": pooled_all,
        "log10_pooled_open_r_over_default": float(np.log10(pooled["r_implied"]["r_median"] / r_default)),
        "note": "r = 1 / CV^2 (R23) from the TEMPORAL per-cell fluctuation over the Mode F frames; the D7-1 default 0.3 "
                "came from the SPATIAL dense-cell residual of one frame (validation_log G1 FAIL 원인과 치료). CV includes the "
                "T0-4 growth trend, CV_norm removes it by normalising each frame with its own source total.",
    }

    # ---- (3) forward models vs the mean field --------------------------------------------------------
    sf_hi = be.slab(hi)
    inst = sf_hi.density[:, sf_hi.z_index(z)].reshape(n_src, M).astype(np.float64)
    mean_flat = ta.mean.reshape(n_src, M)
    truths = {"instantaneous": inst, "mean": mean_flat}
    t0 = time.perf_counter()
    wf = WindField.load()
    om = ObstacleMap.load()
    uv = AdvectionDiffusionOperator.wind_at_cells(wf, grid, AdjointParams(wind_band=config.T1_3C_ADJ_WIND_LAYERS["single_15m"]))
    blocked = AdvectionDiffusionOperator.blocked_at_cells(om, grid, z)
    data_s = time.perf_counter() - t0
    chosen = {"K": config.T1_4_ADJOINT_K, "lam": config.T1_4_ADJOINT_LAM, "wind_layer": "single_15m"}
    t0 = time.perf_counter()
    adj_rows, g_chosen = evaluate_adjoint_neighbourhood(grid, uv, blocked, src_ids, src_xy, truths,
                                                        keep_fields=(chosen["K"], chosen["lam"]))
    adj_s = time.perf_counter() - t0
    if g_chosen is None:
        raise RuntimeError("chosen (K, lam) not in the neighbourhood grid")
    chosen_row = next(r for r in adj_rows if (r["K"], r["lam"]) == (chosen["K"], chosen["lam"]))
    adj_ps_mean = chosen_row["per_truth"]["mean"]["per_source"]
    adj_ps_inst_recomputed = chosen_row["per_truth"]["instantaneous"]["per_source"]
    best_mean = min(adj_rows, key=lambda r: np.nan_to_num(r["per_truth"]["mean"]["mean_train_std_dense"], nan=np.inf))
    best_inst = min(adj_rows, key=lambda r: np.nan_to_num(r["per_truth"]["instantaneous"]["mean_train_std_dense"], nan=np.inf))

    # analytic plume at the chosen T1-3 combination (global wind)
    cells = grid.cell_centres(z)
    plume = cf.make_plume("global", config.T1_4_ANALYTIC_U, config.T1_4_ANALYTIC_SIGMA_V, None)
    t0 = time.perf_counter()
    an_stats_mean, _, _, _, an_wind = cf.evaluate_sources(plume, src_xy, cells, mean_flat)
    an_stats_inst, _, _, _, _ = cf.evaluate_sources(plume, src_xy, cells, inst)
    an_s = time.perf_counter() - t0
    an_ps_mean = {str(s): asdict(st) for s, st in zip(src_ids, an_stats_mean)}
    an_ps_inst_recomputed = {str(s): asdict(st) for s, st in zip(src_ids, an_stats_inst)}

    # stored instantaneous values (frame hi must be the calibration frame for the cross-check to be exact)
    adj_json = json.loads(args.adjoint.read_text(encoding="utf-8"))
    ana_json = json.loads(args.analytic.read_text(encoding="utf-8"))
    adj_ps_inst = adj_json["per_source_chosen"]
    an_ps_inst = ana_json["per_source_chosen"]
    stored_ok = {"adjoint_index": adj_json["index"], "analytic_index": ana_json["index"], "frame_hi": hi,
                 "adjoint_chosen": adj_json["chosen"], "analytic_chosen": ana_json["chosen"],
                 "max_abs_diff_std_dense_adjoint": float(np.nanmax([abs(adj_ps_inst[str(s)]["std_dense"] - adj_ps_inst_recomputed[str(s)]["std_dense"]) for s in src_ids])),
                 "max_abs_diff_std_dense_analytic": float(np.nanmax([abs(an_ps_inst[str(s)]["std_dense"] - an_ps_inst_recomputed[str(s)]["std_dense"]) for s in src_ids]))}

    # implied kappa (train sources) for both models, both truths
    def _kappa(ps: dict[str, dict]) -> dict:
        k = cf.implied_kappa([ps[str(s)]["median_rho"] for s in config.TRAIN_SOURCES])
        k["log10_over_current_KAPPA_REF"] = float(np.log10(k["kappa_ref_implied"] / config.KAPPA_REF))
        return k
    kappa = {"adjoint_instantaneous": _kappa(adj_ps_inst), "adjoint_mean": _kappa(adj_ps_mean),
             "analytic_instantaneous": _kappa(an_ps_inst), "analytic_mean": _kappa(an_ps_mean),
             "current_config_KAPPA_REF": config.KAPPA_REF, "train_sources": list(config.TRAIN_SOURCES),
             "expected_shift_log_mean_over_inst_total": float(np.log(ta.totals.mean(axis=0).sum() / ta.totals[-1].sum())),
             "note": "kappa_ref = SENSOR_K0 x exp(mean_train median rho) (plan 4.3); the mean field is lighter than frame hi "
                     "(growth), so log kappa shifts by about expected_shift_log_mean_over_inst_total."}

    # offsets
    offsets = {}
    for i, s in enumerate(src_ids):
        offsets[str(s)] = {"mean": peak_offset(ta.mean[i], grid, config.SOURCES_XY[s]),
                           "instantaneous": peak_offset(inst[i].reshape(grid.ny, grid.nx), grid, config.SOURCES_XY[s]),
                           "adjoint_model_mean": peak_offset(g_chosen[i].reshape(grid.ny, grid.nx), grid, config.SOURCES_XY[s]),
                           "wind_at_source": ca._wind_at_source(uv, grid, s)}
    offset_report = {str(s): {"mean_peak_distance_m": offsets[str(s)]["mean"]["peak_distance_m"],
                              "mean_peak_direction_deg": offsets[str(s)]["mean"]["peak_direction_deg"],
                              "mean_centroid_distance_m": offsets[str(s)]["mean"]["centroid_distance_m"],
                              "mean_centroid_direction_deg": offsets[str(s)]["mean"]["centroid_direction_deg"],
                              "instantaneous_peak_distance_m": offsets[str(s)]["instantaneous"]["peak_distance_m"],
                              "instantaneous_peak_direction_deg": offsets[str(s)]["instantaneous"]["peak_direction_deg"],
                              "adjoint_model_peak_distance_m": offsets[str(s)]["adjoint_model_mean"]["peak_distance_m"],
                              "adjoint_model_peak_direction_deg": offsets[str(s)]["adjoint_model_mean"]["peak_direction_deg"],
                              "wind_dir_deg": offsets[str(s)]["wind_at_source"]["dir_deg"],
                              "wind_speed": offsets[str(s)]["wind_at_source"]["speed"]}
                     for s in config.T1_3C_OFFSET_SOURCES}

    # per-source table
    per_source = {}
    for s in src_ids:
        k = str(s)
        per_source[k] = {
            "type": cf.source_type(s),
            "adjoint_std_dense_instantaneous": adj_ps_inst[k]["std_dense"],
            "adjoint_std_dense_mean": adj_ps_mean[k]["std_dense"],
            "adjoint_std_instantaneous": adj_ps_inst[k]["std"], "adjoint_std_mean": adj_ps_mean[k]["std"],
            "adjoint_n_dense_mean": adj_ps_mean[k]["n_dense"], "adjoint_n_dense_instantaneous": adj_ps_inst[k]["n_dense"],
            "adjoint_median_rho_instantaneous": adj_ps_inst[k]["median_rho"], "adjoint_median_rho_mean": adj_ps_mean[k]["median_rho"],
            "adjoint_ldm_mass_frac_model_zero_mean": adj_ps_mean[k]["ldm_mass_frac_model_zero"],
            "analytic_std_dense_instantaneous": an_ps_inst[k]["std_dense"],
            "analytic_std_dense_mean": an_ps_mean[k]["std_dense"],
            "analytic_std_instantaneous": an_ps_inst[k]["std"], "analytic_std_mean": an_ps_mean[k]["std"],
            "analytic_median_rho_instantaneous": an_ps_inst[k]["median_rho"], "analytic_median_rho_mean": an_ps_mean[k]["median_rho"],
            "analytic_upwind_mass_frac_mean": an_ps_mean[k]["upwind_mass_frac"],
            "cv_median_dense": fluct[k]["cv"]["median"], "cv_norm_median_dense": fluct[k]["cv_norm"]["median"],
            "r_implied": fluct[k]["r_implied"]["r_median"], "n_dense_fluct": fluct[k]["n_dense"],
            "growth_hi_over_lo": growth[k],
            "mean_peak_offset_m": offsets[k]["mean"]["peak_distance_m"],
            "mean_peak_direction_deg": offsets[k]["mean"]["peak_direction_deg"],
        }
    open_ids = config.T1_3_OPEN_SOURCES
    summary = {
        "adjoint_std_dense_open_mean_of_instantaneous": float(np.mean([per_source[str(s)]["adjoint_std_dense_instantaneous"] for s in open_ids])),
        "adjoint_std_dense_open_mean_of_mean": float(np.mean([per_source[str(s)]["adjoint_std_dense_mean"] for s in open_ids])),
        "analytic_std_dense_open_mean_of_instantaneous": float(np.mean([per_source[str(s)]["analytic_std_dense_instantaneous"] for s in open_ids])),
        "analytic_std_dense_open_mean_of_mean": float(np.mean([per_source[str(s)]["analytic_std_dense_mean"] for s in open_ids])),
        "adjoint_mean_train_std_dense_instantaneous": chosen_row["per_truth"]["instantaneous"]["mean_train_std_dense"],
        "adjoint_mean_train_std_dense_mean": chosen_row["per_truth"]["mean"]["mean_train_std_dense"],
        "n_sources_adjoint_improved_by_mean": int(sum(per_source[str(s)]["adjoint_std_dense_mean"] < per_source[str(s)]["adjoint_std_dense_instantaneous"] for s in src_ids)),
        "n_sources_analytic_improved_by_mean": int(sum(per_source[str(s)]["analytic_std_dense_mean"] < per_source[str(s)]["analytic_std_dense_instantaneous"] for s in src_ids)),
        "open_pass_dense_mean_adjoint": bool(all(np.isfinite(per_source[str(s)]["adjoint_std_dense_mean"]) and per_source[str(s)]["adjoint_std_dense_mean"] < config.T1_3_STD_PASS for s in open_ids)),
        "open_pass_dense_mean_analytic": bool(all(np.isfinite(per_source[str(s)]["analytic_std_dense_mean"]) and per_source[str(s)]["analytic_std_dense_mean"] < config.T1_3_STD_PASS for s in open_ids)),
        "neighbourhood_best_vs_mean": {"K": best_mean["K"], "lam": best_mean["lam"], "mean_train_std_dense": best_mean["per_truth"]["mean"]["mean_train_std_dense"]},
        "neighbourhood_best_vs_instantaneous": {"K": best_inst["K"], "lam": best_inst["lam"], "mean_train_std_dense": best_inst["per_truth"]["instantaneous"]["mean_train_std_dense"]},
        "chosen_is_best_vs_mean": bool((best_mean["K"], best_mean["lam"]) == (chosen["K"], chosen["lam"])),
    }

    # ---- figure --------------------------------------------------------------------------------------
    s_fig = config.T1_3C_FIG_SOURCE
    i_fig = src_ids.index(s_fig)
    model_fig = (g_chosen[i_fig] * np.exp(adj_ps_mean[str(s_fig)]["median_rho"])).reshape(grid.ny, grid.nx)
    fig_png = args.fig_dir / "fig_timeavg_calibration.png"
    fig_prev = args.fig_dir / "fig_timeavg_calibration_preview.png"
    fig_numbers = render_figure(grid, om, s_fig, inst[i_fig].reshape(grid.ny, grid.nx), ta.mean[i_fig], model_fig,
                                per_source, src_ids, chosen, (lo, hi), ca._wind_at_source(uv, grid, s_fig), fig_png, fig_prev)

    neighbourhood = [{"K": r["K"], "lam": r["lam"], "seconds_assemble_factorize": r["seconds_assemble_factorize"],
                      "seconds_solves": r["seconds_solves"],
                      "mean_train_std_dense_instantaneous": r["per_truth"]["instantaneous"]["mean_train_std_dense"],
                      "mean_train_std_dense_mean": r["per_truth"]["mean"]["mean_train_std_dense"],
                      "mean_train_std_instantaneous": r["per_truth"]["instantaneous"]["mean_train_std"],
                      "mean_train_std_mean": r["per_truth"]["mean"]["mean_train_std"],
                      "mean_train_median_rho_mean": r["per_truth"]["mean"]["mean_train_median_rho"],
                      "open_std_dense_mean": {str(s): r["per_truth"]["mean"]["per_source"][str(s)]["std_dense"] for s in open_ids}}
                     for r in adj_rows]
    res = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames": [lo, hi], "n_frames": ta.n_frames, "steps": [config.index_to_step(lo), config.index_to_step(hi)],
        "z": z, "grid": grid.to_dict(), "sources": list(src_ids),
        "npz": {"path": str(npz_path), "built": built, "bytes": npz_path.stat().st_size},
        "dense_fraction": config.T1_3C_DENSE_FRACTION,
        "growth_hi_over_lo": growth, "expected_growth_T0_4": config.T0_4_EXPECTED_GROWTH_TOTAL,
        "fluctuation_per_source": fluct, "dispersion": dispersion,
        "adjoint": {"chosen": chosen, "candidates": {"K": list(config.T1_3C_ADJ_K_CANDIDATES), "lam": list(config.T1_3C_ADJ_LAMBDA_CANDIDATES)},
                    "neighbourhood": neighbourhood, "per_source_mean": adj_ps_mean,
                    "per_source_instantaneous_stored": adj_ps_inst, "per_source_instantaneous_recomputed": adj_ps_inst_recomputed,
                    "all_rows": adj_rows},
        "analytic": {"chosen": {"wind_mode": "global", "U": config.T1_4_ANALYTIC_U, "sigma_v": config.T1_4_ANALYTIC_SIGMA_V},
                     "per_source_mean": an_ps_mean, "per_source_instantaneous_stored": an_ps_inst,
                     "per_source_instantaneous_recomputed": an_ps_inst_recomputed, "source_wind": an_wind},
        "stored_cross_check": stored_ok,
        "implied_kappa": kappa,
        "offsets_all": offsets, "offset_report": offset_report,
        "per_source": per_source, "summary": summary,
        "figure": fig_numbers,
        "timing": {"time_average_s": avg_s, "data_load_s": data_s, "adjoint_neighbourhood_s": adj_s, "analytic_s": an_s,
                   "total_s": time.perf_counter() - t_start},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"time average frames {lo}-{hi} ({ta.n_frames}): growth total {growth['total']:.3f} (T0-4 expected {config.T0_4_EXPECTED_GROWTH_TOTAL})")
    print("source type       CV med [IQR]      CV_norm med [IQR]   r_impl  n_dense | adj std_dense inst -> mean | ana inst -> mean | peak off m (deg)")
    for s in src_ids:
        p, f = per_source[str(s)], fluct[str(s)]
        print(f"  {s} {p['type']:14s} {f['cv']['median']:.2f} [{f['cv']['q25']:.2f},{f['cv']['q75']:.2f}]  "
              f"{f['cv_norm']['median']:.2f} [{f['cv_norm']['q25']:.2f},{f['cv_norm']['q75']:.2f}]  {p['r_implied']:.2f}  {f['n_dense']:5d} | "
              f"{p['adjoint_std_dense_instantaneous']:.2f} -> {p['adjoint_std_dense_mean']:.2f} | "
              f"{p['analytic_std_dense_instantaneous']:.2f} -> {p['analytic_std_dense_mean']:.2f} | "
              f"{p['mean_peak_offset_m']:.0f} ({p['mean_peak_direction_deg']:.0f})")
    print(f"pooled open {list(config.T1_3C_POOLED_SOURCES)}: CV_norm median {pooled['cv_norm']['median']:.3f} -> r {pooled['r_implied']['r_median']:.3f} "
          f"(1/mean CV^2 {pooled['r_implied']['r_mean_cv2']:.3f}); raw CV median {pooled['cv']['median']:.3f}; default r {r_default}")
    print("adjoint neighbourhood (mean_train std_dense inst / mean): " + ", ".join(
        f"K {r['K']:g} lam {r['lam']:g}: {r['mean_train_std_dense_instantaneous']:.3f} / {r['mean_train_std_dense_mean']:.3f}" for r in neighbourhood))
    print(f"implied kappa_ref adjoint inst {kappa['adjoint_instantaneous']['kappa_ref_implied']:.3e} -> mean {kappa['adjoint_mean']['kappa_ref_implied']:.3e}; "
          f"analytic inst {kappa['analytic_instantaneous']['kappa_ref_implied']:.3e} -> mean {kappa['analytic_mean']['kappa_ref_implied']:.3e}")
    for s, o in offset_report.items():
        print(f"offset {s}: mean peak {o['mean_peak_distance_m']:.0f} m @ {o['mean_peak_direction_deg']:.0f} deg, centroid "
              f"{o['mean_centroid_distance_m']:.0f} m @ {o['mean_centroid_direction_deg']:.0f} deg; inst peak {o['instantaneous_peak_distance_m']:.0f} m; "
              f"adjoint model peak {o['adjoint_model_peak_distance_m']:.0f} m; wind {o['wind_speed']:.2f} m/s @ {o['wind_dir_deg']:.0f} deg")
    print(f"stored cross-check max |d std_dense|: adjoint {stored_ok['max_abs_diff_std_dense_adjoint']:.2e}, analytic {stored_ok['max_abs_diff_std_dense_analytic']:.2e}")
    print(f"elapsed {res['timing']['total_s']:.1f} s (average {avg_s:.1f} s, adjoint {adj_s:.1f} s); wrote {args.out}, {npz_path}, {fig_png}")
    return res


if __name__ == "__main__":
    main()