"""Plan S0 deliverable figures 1-2: LBM/LDM scene overview and per-source slab statistics.

Usage:
    python -m srcloc_env.scripts.fig_scene [--frame 599] [--out-dir <FIG_DIR>] [--json <CACHE_DIR>/fig_scene_numbers.json]

Figure 1 (fig1_scene.png 300 dpi, fig1_scene_preview.png 120 dpi)
    background  building height map of occupancy_2m_flowframe.npz (report 4.2-4.3; occ/hmap[ix, iy] per
                config.OCC_AXIS_ORDER, cells without a building masked) on a grey ramp;
    overlay     airborne particle density [particles/m^3] at z = config.DRONE_Z of the cached slab frame
                (cache/slabs/slab_XXX.npz, docs/data_cache.md; units of report 2.1) summed over the 13 sources,
                log10 colour scale masked where 0 (deposited and outflow particles are excluded upstream, report 2.4);
    symbols     the 13 source centres config.SOURCES_XY (report 2.6) as stars with their p_type ids, a mean-wind
                arrow along +x (report 2.5: 11 of 13 plume drift directions within +-15 deg of +x; report 3.3),
                the LDM particle envelope of step 30000 (report 2.5 / 5) as a dashed rectangle.
    axes        fluid/LDM frame, x 0-1315 m, y -657.5..657.5 m (report 3.1 / 5), equal aspect.
Figure 2 (fig2_slab_stats.png 300 dpi)
    (a) per-source max and mean-of-non-zero-cells density of the z = 15 m slab (log y), source 110 highlighted;
    (b) zoom map (+-FIG_ZOOM_HALF_WIDTH_M) of the source 110 slab with the building footprint outline.
Every plotted number (per-source max / mean-nonzero / occupied cells for all stored z levels, the summed-slab
statistics, the colour-scale limits, the wind-arrow value, the 115 m tower location used as the axis-order check)
is written to the JSON file.

Only matplotlib (Agg backend) is used; labels are English.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

from srcloc_env import config  # noqa: E402
from srcloc_env.field.concentration_field import SlabFrame, load_slab_frame  # noqa: E402
from srcloc_env.preprocess.gridder import SlabGrid, slab_stats  # noqa: E402
from srcloc_env.scripts.convert_ldm import frame_paths  # noqa: E402

# Colours (neutral greys for context, one blue hue light->dark for the sequential density ramp, one warm accent
# for the source markers so that they stay visible on both the grey buildings and the blue plume).
COLOR_ACCENT = "#d1495b"        # source stars / highlighted bars
COLOR_BAR_MAX = "#8c8c8c"       # figure 2(a) 'max' bars
COLOR_BAR_MEAN = "#c9c9c9"      # figure 2(a) 'mean non-zero' bars
COLOR_HL_MAX = "#1f4e9c"        # highlighted source 'max'
COLOR_HL_MEAN = "#7fa8e0"       # highlighted source 'mean non-zero'
CMAP_DENSITY = LinearSegmentedColormap.from_list("blues_trunc", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
CMAP_BUILDING = LinearSegmentedColormap.from_list("greys_trunc", plt.get_cmap("Greys")(np.linspace(0.3, 1.0, 256)))
FIG1_TITLE = "LBM urban wind field + LDM dispersion, step {step} (airborne particles, x < {xout:.0f} m)"


# ------------------------------------------------------------------------------------------ building map
@dataclass(frozen=True)
class BuildingMap:
    """2 m occupancy / height map in the fluid-LDM frame (report 4.2-4.3), stored as hmap[ix, iy].

    x[ix], y[iy] are the cell coordinates written by stl_tools (x = x0 + ix*res); the cell is drawn as the
    square [x, x + res) x [y, y + res) (the possible half-cell offset of that convention is below the 2.5 m
    lattice spacing and irrelevant for the figures).
    """

    x: np.ndarray            # (nx,) [m]
    y: np.ndarray            # (ny,) [m]
    hmap: np.ndarray         # (nx, ny) building height [m]; config.OCC_HMAP_NO_BUILDING where free
    res: float               # cell size [m]

    @property
    def occ(self) -> np.ndarray:
        return self.hmap > config.OCC_HMAP_NO_BUILDING

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """imshow extent (left, right, bottom, top) for the image() array with origin='lower'."""
        return (float(self.x[0]), float(self.x[-1] + self.res), float(self.y[0]), float(self.y[-1] + self.res))

    def image(self) -> np.ma.MaskedArray:
        """(ny, nx) masked height image: row iy, column ix (matplotlib image convention)."""
        return np.ma.masked_where(~self.occ.T, self.hmap.T)

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x + 0.5 * self.res, self.y + 0.5 * self.res

    def tallest_cell_xy(self) -> tuple[float, float, float]:
        """(x, y, height) of the tallest cell: the 115 m tower of report 4.1, expected next to source 110."""
        ix, iy = np.unravel_index(int(np.argmax(self.hmap)), self.hmap.shape)
        return float(self.x[ix]), float(self.y[iy]), float(self.hmap[ix, iy])


def load_building_map(path: Path = config.OCCUPANCY_2M_NPZ) -> BuildingMap:
    """Read occupancy_2m_flowframe.npz (keys occ, hmap, x, y, meta) honouring config.OCC_AXIS_ORDER."""
    with np.load(path) as z:
        hmap = np.asarray(z["hmap"], dtype=np.float32)
        occ = np.asarray(z["occ"], dtype=bool)
        x = np.asarray(z["x"], dtype=np.float64)
        y = np.asarray(z["y"], dtype=np.float64)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if config.OCC_AXIS_ORDER == "iy,ix":
        hmap, occ = hmap.T, occ.T
    elif config.OCC_AXIS_ORDER != "ix,iy":
        raise ValueError(f"unknown OCC_AXIS_ORDER {config.OCC_AXIS_ORDER!r}")
    if hmap.shape != (x.size, y.size):
        raise ValueError(f"{path}: hmap shape {hmap.shape} != (nx={x.size}, ny={y.size}) under order {config.OCC_AXIS_ORDER}")
    if not np.array_equal(occ, hmap > config.OCC_HMAP_NO_BUILDING):
        raise ValueError(f"{path}: occ differs from hmap > {config.OCC_HMAP_NO_BUILDING}")
    res = float(meta.get("res", np.diff(x).mean() if x.size > 1 else config.DX))
    return BuildingMap(x=x, y=y, hmap=hmap, res=res)


# ------------------------------------------------------------------------------------------ slab helpers
def slab_extent(grid: SlabGrid) -> tuple[float, float, float, float]:
    """imshow extent of a (ny, nx) slab: cell edges x0..x0+nx*res, y0..y0+ny*res (cell centres at +res/2)."""
    return (grid.x0, grid.x0 + grid.nx * grid.res, grid.y0, grid.y0 + grid.ny * grid.res)


def sum_sources_at_z(frame: SlabFrame, z: float = config.DRONE_Z) -> np.ndarray:
    """(ny, nx) float32 density summed over all sources at the stored level z."""
    return frame.density[:, frame.z_index(z)].sum(axis=0, dtype=np.float32)


def source_slab_at_z(frame: SlabFrame, source: int, z: float = config.DRONE_Z) -> np.ndarray:
    return frame.density[frame.source_indices(source)[0], frame.z_index(z)]


def log_limits(values: np.ndarray, decades: float = config.FIG_LOG_DECADES) -> tuple[float, float]:
    """(vmin, vmax) of the log10 colour scale: vmax = max, vmin = max(min positive, vmax / 10**decades)."""
    pos = values[values > 0]
    if pos.size == 0:
        return 1.0, 10.0
    vmax = float(pos.max())
    vmin = max(float(pos.min()), vmax / 10.0**decades)
    if vmin >= vmax:
        vmin = vmax / 10.0
    return vmin, vmax


def masked_log_image(values: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(values <= 0, values)


# ------------------------------------------------------------------------------------------ wind arrow value
def wind_arrow_numbers(z: float = config.DRONE_Z) -> dict:
    """Mean wind used for the arrow annotation: plume-region mean (u, v) at z from field/wind.py (report 3.3).

    Falls back to the report's 10-15 m band value config.WIND_REF_PLUME_U_10_15 when the wind files are absent.
    """
    try:
        from srcloc_env.field.wind import WindField

        wf = WindField.load()
        u, v = wf.region_mean(z, exclude_buildings=True)
        u_all, v_all = wf.region_mean(z, exclude_buildings=False)
        u_lvl, v_lvl = wf.mean_profile(z)
        return {"source": "WindField.region_mean (levels_uvw.npz), plume region x 330-1315, |y| < 500",
                "z_m": float(z), "u": u, "v": v, "speed": float(np.hypot(u, v)),
                "direction_deg_from_plus_x": float(np.degrees(np.arctan2(v, u))),
                "u_all_nodes": u_all, "v_all_nodes": v_all, "u_level_mean": u_lvl, "v_level_mean": v_lvl,
                "report_3_3_plume_u_10_15": config.WIND_REF_PLUME_U_10_15}
    except Exception as exc:  # noqa: BLE001 - figure must still be produced without the wind cache
        return {"source": f"config.WIND_REF_PLUME_U_10_15 (fallback: {type(exc).__name__}: {exc})",
                "z_m": float(z), "u": config.WIND_REF_PLUME_U_10_15, "v": 0.0,
                "speed": config.WIND_REF_PLUME_U_10_15, "direction_deg_from_plus_x": 0.0}


# ------------------------------------------------------------------------------------------ figure 1
def render_figure1(bmap: BuildingMap, dens_sum: np.ndarray, grid: SlabGrid, step: int, wind: dict,
                   out_png: Path, out_preview: Path | None = None,
                   sources_xy: dict[int, tuple[float, float]] = config.SOURCES_XY, z: float = config.DRONE_Z) -> dict:
    """Draw figure 1 and return the numbers shown on it (colour limits, arrow geometry)."""
    vmin, vmax = log_limits(dens_sum)
    fig, ax = plt.subplots(figsize=(10.0, 10.2))
    ax.set_facecolor("white")

    im_b = ax.imshow(bmap.image(), extent=bmap.extent, origin="lower", cmap=CMAP_BUILDING,
                     vmin=0.0, vmax=float(bmap.hmap.max()), interpolation="nearest", zorder=1)
    im_d = ax.imshow(masked_log_image(dens_sum), extent=slab_extent(grid), origin="lower", cmap=CMAP_DENSITY,
                     norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest", alpha=0.92, zorder=2)

    # LDM particle envelope at step 30000 (report 2.5)
    ex, ey = config.LDM_EXTENT_X_30000, config.LDM_EXTENT_Y_30000
    ax.add_patch(Rectangle((ex[0], ey[0]), ex[1] - ex[0], ey[1] - ey[0], fill=False, ls="--", lw=1.3,
                           ec="black", zorder=4, label=f"LDM particle extent, step {config.STEP_MAX} (report 2.5)"))

    # sources
    ids = sorted(sources_xy)
    sx = np.array([sources_xy[s][0] for s in ids])
    sy = np.array([sources_xy[s][1] for s in ids])
    ax.scatter(sx, sy, marker="*", s=170, c=COLOR_ACCENT, edgecolors="white", linewidths=0.8, zorder=6,
               label="source centres (13, report 2.6)")
    for s, xx, yy in zip(ids, sx, sy):
        ax.text(xx + 9.0, yy + 9.0, str(s), fontsize=8.5, color="black", zorder=7, ha="left", va="bottom",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    # mean-wind arrow in the free strip north of the buildings
    ax0, ay0 = 40.0, float(config.DOMAIN_Y[1]) - 200.0      # free strip x < 200 m, below the extent box top
    L = config.FIG_WIND_ARROW_LENGTH_M
    ax.annotate("", xy=(ax0 + L, ay0), xytext=(ax0, ay0), zorder=8,
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="black", shrinkA=0, shrinkB=0))
    ax.text(ax0, ay0 + 16.0, f"mean wind (+x)\n{wind['speed']:.2f} m/s at z = {z:g} m", fontsize=9.5,
            ha="left", va="bottom", zorder=8)

    ax.set_xlim(*config.DOMAIN_X)
    ax.set_ylim(*config.DOMAIN_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]  (fluid / LDM frame)")
    ax.set_ylabel("y [m]")
    ax.set_title(FIG1_TITLE.format(step=step, xout=config.X_OUTFLOW), fontsize=11.5)
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    divider = make_axes_locatable(ax)          # colorbars that follow the equal-aspect axes exactly
    cb_d = fig.colorbar(im_d, cax=divider.append_axes("right", size="3.5%", pad=0.12))
    cb_d.set_label(f"particles / m$^3$ (z = {z:g} m)")
    cb_b = fig.colorbar(im_b, cax=divider.append_axes("bottom", size="3%", pad=0.65), orientation="horizontal")
    cb_b.set_label("building height [m]")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=config.FIG_DPI_FINAL, bbox_inches="tight")
    if out_preview is not None:
        fig.savefig(out_preview, dpi=config.FIG_DPI_PREVIEW, bbox_inches="tight")
    plt.close(fig)
    return {"log_vmin": vmin, "log_vmax": vmax, "arrow_start_xy": [ax0, ay0], "arrow_length_m": L,
            "arrow_speed_label_m_s": float(wind["speed"])}


# ------------------------------------------------------------------------------------------ figure 2
def render_figure2(frame: SlabFrame, bmap: BuildingMap, out_png: Path, z: float = config.DRONE_Z,
                   highlight: int = config.FIG_HIGHLIGHT_SOURCE,
                   sources_xy: dict[int, tuple[float, float]] = config.SOURCES_XY) -> dict:
    """Draw figure 2 (a: per-source bars, b: zoom on the highlighted source) and return its numbers."""
    zi = frame.z_index(z)
    stats = slab_stats(frame.density[:, zi:zi + 1])
    ids = list(frame.sources)
    smax = np.array([row[0] for row in stats["max"]], dtype=np.float64)
    smean = np.array([row[0] for row in stats["mean_nonzero"]], dtype=np.float64)
    ncell = [int(row[0]) for row in stats["occupied_cells"]]

    fig = plt.figure(figsize=(13.0, 5.4))
    gs = GridSpec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28, figure=fig)

    # (a) bars
    ax = fig.add_subplot(gs[0, 0])
    pos = np.arange(len(ids))
    w = 0.38
    c_max = [COLOR_HL_MAX if s == highlight else COLOR_BAR_MAX for s in ids]
    c_mean = [COLOR_HL_MEAN if s == highlight else COLOR_BAR_MEAN for s in ids]
    ax.bar(pos - w / 2, np.where(smax > 0, smax, np.nan), w, color=c_max, zorder=3)
    ax.bar(pos + w / 2, np.where(smean > 0, smean, np.nan), w, color=c_mean, zorder=3)
    for p, n, m in zip(pos, ncell, smax):
        if m > 0:
            ax.text(p, m * 1.25, f"{n}", ha="center", va="bottom", fontsize=7, color="#444444", zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(pos)
    ax.set_xticklabels([str(s) for s in ids], fontsize=9)
    ax.set_xlabel("source (p_type)")
    ax.set_ylabel(f"particles / m$^3$ at z = {z:g} m")
    ax.set_title(f"(a) per-source slab statistics, step {config.index_to_step(frame.index)}", fontsize=10.5, pad=34)
    ax.grid(axis="y", which="major", color="#e3e3e3", zorder=0)
    ax.set_axisbelow(True)
    handles = [Patch(color=COLOR_BAR_MAX, label="max cell"), Patch(color=COLOR_BAR_MEAN, label="mean of non-zero cells"),
               Patch(color=COLOR_HL_MAX, label=f"source {highlight} (highlighted)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False,
              title="number above each bar = occupied (non-zero) cells", title_fontsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # (b) zoom on the highlighted source
    axb = fig.add_subplot(gs[0, 1])
    hx, hy = sources_xy[highlight]
    half = config.FIG_ZOOM_HALF_WIDTH_M
    slab = source_slab_at_z(frame, highlight, z)
    vmin, vmax = log_limits(slab)
    axb.imshow(bmap.image(), extent=bmap.extent, origin="lower", cmap=CMAP_BUILDING, vmin=0.0,
               vmax=float(bmap.hmap.max()), interpolation="nearest", alpha=0.55, zorder=1)
    im = axb.imshow(masked_log_image(slab), extent=slab_extent(frame.grid), origin="lower", cmap=CMAP_DENSITY,
                    norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest", alpha=0.9, zorder=2)
    cx, cy = bmap.cell_centres()
    axb.contour(cx, cy, bmap.occ.T.astype(np.float32), levels=[0.5], colors="black", linewidths=0.7, zorder=3)
    axb.scatter([hx], [hy], marker="*", s=220, c=COLOR_ACCENT, edgecolors="white", linewidths=0.8, zorder=5)
    axb.text(hx + 6, hy + 6, str(highlight), fontsize=9, zorder=6,
             path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    axb.set_xlim(hx - half, hx + half)
    axb.set_ylim(hy - half, hy + half)
    axb.set_aspect("equal")
    axb.set_xlabel("x [m]")
    axb.set_ylabel("y [m]")
    axb.set_title(f"(b) source {highlight} slab, z = {z:g} m ($\\pm${half:g} m), building outline", fontsize=10.5)
    cb = fig.colorbar(im, ax=axb, fraction=0.046, pad=0.03)
    cb.set_label(f"particles / m$^3$ (source {highlight})")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=config.FIG_DPI_FINAL, bbox_inches="tight")
    plt.close(fig)

    win = ((cx >= hx - half) & (cx <= hx + half))[:, None] & ((cy >= hy - half) & (cy <= hy + half))[None, :]
    return {"z_m": float(z), "highlight_source": int(highlight), "zoom_centre_xy": [float(hx), float(hy)],
            "zoom_half_width_m": float(half), "zoom_log_vmin": vmin, "zoom_log_vmax": vmax,
            "zoom_building_cells": int((bmap.occ & win).sum()),
            "zoom_building_height_max_m": float(bmap.hmap[win].max()) if np.any(win) else 0.0,
            "per_source": {str(s): {"max": float(smax[i]), "mean_nonzero": float(smean[i]), "occupied_cells": ncell[i]}
                           for i, s in enumerate(ids)}}


# ------------------------------------------------------------------------------------------ numbers
def summed_slab_numbers(dens_sum: np.ndarray, grid: SlabGrid) -> dict:
    pos = dens_sum[dens_sum > 0]
    return {"max": float(dens_sum.max()), "mean_nonzero": float(pos.mean()) if pos.size else 0.0,
            "occupied_cells": int(pos.size), "total_cells": int(dens_sum.size),
            "occupied_fraction": float(pos.size / dens_sum.size),
            "cell_area_m2": float(grid.res**2)}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frame", type=int, default=config.FIG_SCENE_FRAME_INDEX, help="LDM frame index (0..599)")
    ap.add_argument("--cache", type=Path, default=config.CACHE_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.FIG_DIR)
    ap.add_argument("--json", type=Path, default=config.CACHE_DIR / "fig_scene_numbers.json")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    step = config.index_to_step(args.frame)
    slab_path = frame_paths(args.cache, args.frame)[2]
    frame = load_slab_frame(slab_path, args.frame)
    bmap = load_building_map()
    dens_sum = sum_sources_at_z(frame, config.DRONE_Z)
    wind = wind_arrow_numbers(config.DRONE_Z)

    out_dir = args.out_dir
    f1 = out_dir / "fig1_scene.png"
    f1p = out_dir / "fig1_scene_preview.png"
    f2 = out_dir / "fig2_slab_stats.png"
    fig1_numbers = render_figure1(bmap, dens_sum, frame.grid, step, wind, f1, f1p)
    fig2_numbers = render_figure2(frame, bmap, f2)

    tx, ty, th = bmap.tallest_cell_xy()
    hx, hy = config.SOURCES_XY[config.FIG_HIGHLIGHT_SOURCE]
    all_levels = slab_stats(frame.density)
    numbers = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_index": int(args.frame), "step": int(step), "z_m": float(config.DRONE_Z),
        "slab_file": str(slab_path), "occupancy_file": str(config.OCCUPANCY_2M_NPZ),
        "grid": frame.grid.to_dict(), "z_levels": list(frame.z_levels), "sources": list(frame.sources),
        "occupancy": {"axis_order": config.OCC_AXIS_ORDER, "res_m": bmap.res, "shape_ix_iy": list(bmap.hmap.shape),
                      "extent": list(bmap.extent), "building_cells": int(bmap.occ.sum()),
                      "height_max_m": float(bmap.hmap.max()),
                      "tallest_cell_xy": [tx, ty], "tallest_cell_height_m": th,
                      "tallest_cell_distance_to_source_110_m": float(np.hypot(tx - hx, ty - hy)),
                      "tower_within_search_radius": bool(np.hypot(tx - hx, ty - hy) <= config.TOWER_115_SEARCH_RADIUS_M)},
        "per_source_z15": fig2_numbers.pop("per_source"),
        "per_source_all_levels": {str(s): {"max": all_levels["max"][i], "mean_nonzero": all_levels["mean_nonzero"][i],
                                           "occupied_cells": all_levels["occupied_cells"][i]}
                                  for i, s in enumerate(frame.sources)},
        "summed_slab_z15": summed_slab_numbers(dens_sum, frame.grid),
        "figure1": fig1_numbers, "figure2": fig2_numbers, "wind_arrow": wind,
        "ldm_extent_box": {"x": list(config.LDM_EXTENT_X_30000), "y": list(config.LDM_EXTENT_Y_30000)},
        "files": {p.name: {"path": str(p), "bytes": p.stat().st_size} for p in (f1, f1p, f2)},
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: numbers[k] for k in ("frame_index", "step", "summed_slab_z15", "wind_arrow", "files", "elapsed_s")},
                     indent=2, ensure_ascii=False))
    return numbers


if __name__ == "__main__":
    main()