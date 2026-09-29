"""Plan S0 figure 2(a): near-ground concentration WITH vs WITHOUT deposited particles (slide 5 of the talk).

Usage:
    python -m srcloc_env.scripts.fig_data [--frame 599] [--out-dir <FIG_DIR>] [--json <CACHE_DIR>/fig_data_numbers.json]

Question answered (report 2.4, plan S0 그림 2(a), slide 5): deposited particles sit at z = config.DEPOSIT_Z
(1e-4 m) with zero velocity and are never removed, so a sensor that gathers ALL particles of a source would see
a spurious ground "concentration" that keeps growing.  With the 7.5 m C6 kernel (report 2.1, R1) they cannot
influence any height above config.KERNEL_H, which is why the drone slabs (z >= 12.5 m) are unaffected; this
script quantifies the effect at the heights config.FIG_DATA_Z_LEVELS = (0.5, 1, 2.5, 5, 7.5, 10, 15) m.

Method (frame index config.FIG_DATA_FRAME_INDEX = 599, step 30000, loaded from the raw VTK because the
deposited mask needs the velocity array): for every source s and height z the C6 density
    n(x, y, z) = c6_gather(...) / config.LATTICE_CELL_VOLUME   [particles / m^3]
is evaluated at the 5 m slab cell centres (preprocess.gridder.SlabGrid) over two particle sets of source s
    (i)  "all"       every particle except the outflow pile-up (x >= config.X_OUTFLOW), i.e. airborne + deposited;
    (ii) "airborne"  frame.airborne (deposited and outflow excluded, CLAUDE.md rule / report 2.4).
Only particles with |z_p - z| <= KERNEL_H can contribute; one cKDTree per source per particle set is built over
the band z_p <= max(z levels) + KERNEL_H and c6_gather's radius search enforces the 3-D support.
Per source and z the script reports: mean over the cells where (i) > 0 of (i) and of (ii), the maxima, the
ratios mean_i / mean_ii and max_i / max_ii, and the fraction of cells where (i) > 0 but (ii) == 0 (cells whose
"concentration" is entirely deposited particles).  The pooled profile uses all (source, cell) pairs with (i) > 0.

Outputs
    <CACHE_DIR>/fig_data_numbers.json         every number, plus a markdown table string ("markdown_table")
    <FIG_DIR>/fig2a_deposited_ratio.png       heatmap sources x heights of mean_i / mean_ii (log colour, annotated)
    <FIG_DIR>/fig2a_profile.png               pooled mean density vs height, with and without deposited particles
Only matplotlib (Agg backend) is used; labels are English.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from srcloc_env import config  # noqa: E402
from srcloc_env.io.ldm_reader import LdmFrame, load_frame  # noqa: E402
from srcloc_env.preprocess.gridder import SlabGrid  # noqa: E402
from srcloc_env.sensor.kernel import c6_gather  # noqa: E402

SET_ALL = "all"              # (i)  airborne + deposited, outflow pile-up removed
SET_AIRBORNE = "airborne"    # (ii) frame.airborne
COLOR_WITH = "#d1495b"       # with deposited particles (accent, as the source markers of fig_scene)
COLOR_WITHOUT = "#1f4e9c"    # airborne only (blue, as the density ramp of fig_scene)
CMAP_RATIO = LinearSegmentedColormap.from_list("oranges_trunc", plt.get_cmap("Oranges")(np.linspace(0.05, 1.0, 256)))


# ------------------------------------------------------------------------------------------ particle sets
def particle_sets(frame: LdmFrame, source: int) -> dict[str, np.ndarray]:
    """The two particle-position sets of one source used by figure 2(a).

    Returns {"all": (n_i, 3), "airborne": (n_ii, 3)} float64 positions of source ``source``:
    "all" keeps every particle of the source except the outflow pile-up (x >= config.X_OUTFLOW, report 2.4),
    "airborne" additionally removes the deposited particles (z == config.DEPOSIT_Z and zero velocity).
    n_i - n_ii is therefore the number of deposited particles of the source inside the lattice.
    """
    of_source = frame.p_type == source
    keep_all = of_source & ~frame.outflow
    keep_air = of_source & frame.airborne
    return {SET_ALL: frame.xyz[keep_all].astype(np.float64), SET_AIRBORNE: frame.xyz[keep_air].astype(np.float64)}


def band_tree(xyz: np.ndarray, z_levels: tuple[float, ...]) -> tuple[np.ndarray, cKDTree | None]:
    """Restrict to the vertical band that can reach any of the query heights and build one cKDTree."""
    z = np.asarray(z_levels, dtype=np.float64)
    band = (xyz[:, 2] >= z.min() - config.KERNEL_H) & (xyz[:, 2] <= z.max() + config.KERNEL_H)
    pts = xyz[band]
    return pts, (cKDTree(pts) if pts.shape[0] else None)


def density_on_grid(pts: np.ndarray, tree: cKDTree | None, grid: SlabGrid, z: float) -> np.ndarray:
    """(ny, nx) C6 density [particles/m^3] at the cell centres of ``grid`` at height z (exact gather)."""
    if tree is None:
        return np.zeros((grid.ny, grid.nx), dtype=np.float64)
    dens = c6_gather(grid.cell_centres(z), pts, tree) / config.LATTICE_CELL_VOLUME
    return dens.reshape(grid.ny, grid.nx)


# ------------------------------------------------------------------------------------------ statistics
def compare_sets(dens_all: np.ndarray, dens_air: np.ndarray) -> dict:
    """Per (source, z) comparison of the two density maps on the cells where the 'all' set is non-zero."""
    occ = dens_all > 0
    n_occ = int(occ.sum())
    if n_occ == 0:
        return {"cells_all_pos": 0, "mean_all": 0.0, "mean_airborne": 0.0, "max_all": 0.0, "max_airborne": 0.0,
                "ratio_mean": float("nan"), "ratio_max": float("nan"), "frac_all_pos_airborne_zero": float("nan"),
                "sum_all": 0.0, "sum_airborne": 0.0}
    a, b = dens_all[occ], dens_air[occ]
    mean_a, mean_b = float(a.mean()), float(b.mean())
    max_a, max_b = float(dens_all.max()), float(dens_air.max())
    return {"cells_all_pos": n_occ, "mean_all": mean_a, "mean_airborne": mean_b, "max_all": max_a, "max_airborne": max_b,
            "ratio_mean": mean_a / mean_b if mean_b > 0 else float("inf"),
            "ratio_max": max_a / max_b if max_b > 0 else float("inf"),
            "frac_all_pos_airborne_zero": float((b == 0).mean()),
            "sum_all": float(a.sum()), "sum_airborne": float(b.sum())}


def compute_numbers(frame: LdmFrame, z_levels: tuple[float, ...] = config.FIG_DATA_Z_LEVELS,
                    sources: tuple[int, ...] = config.ALL_SOURCES, grid: SlabGrid = SlabGrid(),
                    verbose: bool = True) -> dict:
    """Run the gathers for every source / z / particle set and return the per-source and pooled numbers."""
    per_source: dict[str, dict] = {}
    pooled_sum = {z: {"sum_all": 0.0, "sum_airborne": 0.0, "cells": 0, "max_all": 0.0, "max_airborne": 0.0,
                      "cells_airborne_zero": 0} for z in z_levels}
    timing = {}
    for s in sources:
        t0 = time.perf_counter()
        sets = particle_sets(frame, s)
        trees = {name: band_tree(xyz, z_levels) for name, xyz in sets.items()}
        entry = {"n_all": int(sets[SET_ALL].shape[0]), "n_airborne": int(sets[SET_AIRBORNE].shape[0]),
                 "n_deposited": int(sets[SET_ALL].shape[0] - sets[SET_AIRBORNE].shape[0]),
                 "n_in_band": {name: int(pts.shape[0]) for name, (pts, _) in trees.items()}, "by_z": {}}
        for z in z_levels:
            d_all = density_on_grid(*trees[SET_ALL], grid, z)
            d_air = density_on_grid(*trees[SET_AIRBORNE], grid, z)
            cmp_ = compare_sets(d_all, d_air)
            entry["by_z"][str(z)] = cmp_
            p = pooled_sum[z]
            p["sum_all"] += cmp_["sum_all"]
            p["sum_airborne"] += cmp_["sum_airborne"]
            p["cells"] += cmp_["cells_all_pos"]
            p["max_all"] = max(p["max_all"], cmp_["max_all"])
            p["max_airborne"] = max(p["max_airborne"], cmp_["max_airborne"])
            p["cells_airborne_zero"] += int(round(cmp_["frac_all_pos_airborne_zero"] * cmp_["cells_all_pos"])) if cmp_["cells_all_pos"] else 0
        timing[str(s)] = round(time.perf_counter() - t0, 2)
        per_source[str(s)] = entry
        if verbose:
            r1 = entry["by_z"][str(z_levels[0])]["ratio_mean"]
            print(f"source {s}: n_all {entry['n_all']:,} n_airborne {entry['n_airborne']:,} deposited {entry['n_deposited']:,} "
                  f"ratio_mean(z={z_levels[0]:g}) {r1:.2f}  [{timing[str(s)]} s]", flush=True)
    pooled = {}
    for z in z_levels:
        p = pooled_sum[z]
        mean_a = p["sum_all"] / p["cells"] if p["cells"] else 0.0
        mean_b = p["sum_airborne"] / p["cells"] if p["cells"] else 0.0
        pooled[str(z)] = {"cells_all_pos": p["cells"], "mean_all": mean_a, "mean_airborne": mean_b,
                          "ratio_mean": mean_a / mean_b if mean_b > 0 else float("inf"),
                          "max_all": p["max_all"], "max_airborne": p["max_airborne"],
                          "ratio_max": p["max_all"] / p["max_airborne"] if p["max_airborne"] > 0 else float("inf"),
                          "frac_all_pos_airborne_zero": p["cells_airborne_zero"] / p["cells"] if p["cells"] else float("nan")}
    return {"per_source": per_source, "pooled": pooled, "timing_per_source_s": timing}


def ratio_matrix(numbers: dict, sources: tuple[int, ...], z_levels: tuple[float, ...], key: str = "ratio_mean") -> np.ndarray:
    """(n_sources, n_z) matrix of one per-(source, z) statistic."""
    return np.array([[numbers["per_source"][str(s)]["by_z"][str(z)][key] for z in z_levels] for s in sources], dtype=np.float64)


def markdown_table(numbers: dict, sources: tuple[int, ...], z_levels: tuple[float, ...]) -> str:
    """Markdown table: rows = sources (+ pooled), columns = z, cell = mean ratio (with/without deposited)."""
    head = "| source | " + " | ".join(f"z={z:g} m" for z in z_levels) + " |"
    sep = "|---|" + "|".join("---:" for _ in z_levels) + "|"
    rows = [head, sep]
    for s in sources:
        vals = [numbers["per_source"][str(s)]["by_z"][str(z)]["ratio_mean"] for z in z_levels]
        rows.append(f"| {s} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    vals = [numbers["pooled"][str(z)]["ratio_mean"] for z in z_levels]
    rows.append("| pooled | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    return "\n".join(rows)


# ------------------------------------------------------------------------------------------ figures
def render_ratio_heatmap(numbers: dict, sources: tuple[int, ...], z_levels: tuple[float, ...], step: int, out_png: Path,
                         decades: float = config.FIG_DATA_RATIO_LOG_DECADES) -> dict:
    """Heatmap (sources x heights) of mean_all / mean_airborne with a log colour scale and annotated values."""
    m = ratio_matrix(numbers, sources, z_levels, "ratio_mean")
    finite = m[np.isfinite(m) & (m > 0)]
    vmax = max(float(finite.max()) if finite.size else 10.0, 10.0**0.5)
    vmax = min(vmax, 10.0**decades)
    shown = np.where(np.isfinite(m), np.clip(m, 1.0, vmax), vmax)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    im = ax.imshow(shown, cmap=CMAP_RATIO, norm=LogNorm(vmin=1.0, vmax=vmax), aspect="auto", interpolation="nearest")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            txt = "inf" if np.isinf(v) else ("n/a" if np.isnan(v) else (f"{v:.2f}" if v < 100 else f"{v:.0f}"))
            dark = np.isfinite(v) and v < vmax**0.55
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="black" if dark else "white")
    ax.set_xticks(np.arange(len(z_levels)))
    ax.set_xticklabels([f"{z:g}" for z in z_levels])
    ax.set_yticks(np.arange(len(sources)))
    ax.set_yticklabels([str(s) for s in sources])
    ax.set_xlabel("query height z [m]")
    ax.set_ylabel("source (p_type)")
    ax.set_title(f"(a) mean density WITH / WITHOUT deposited particles, step {step}\n"
                 f"(C6 kernel H = {config.KERNEL_H:g} m; deposited particles at z = {config.DEPOSIT_Z:g} m cannot reach z > {config.KERNEL_H:g} m)",
                 fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("ratio of means over cells with any particle within H  (log scale)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=config.FIG_DPI_FINAL, bbox_inches="tight")
    plt.close(fig)
    return {"log_vmin": 1.0, "log_vmax": vmax, "ratio_min": float(finite.min()) if finite.size else None,
            "ratio_max": float(finite.max()) if finite.size else None}


def render_profile(numbers: dict, z_levels: tuple[float, ...], step: int, out_png: Path) -> dict:
    """Pooled mean density vs height for the two particle sets (log x, height on y)."""
    z = np.array(z_levels, dtype=np.float64)
    mean_all = np.array([numbers["pooled"][str(v)]["mean_all"] for v in z_levels])
    mean_air = np.array([numbers["pooled"][str(v)]["mean_airborne"] for v in z_levels])
    max_all = np.array([numbers["pooled"][str(v)]["max_all"] for v in z_levels])
    max_air = np.array([numbers["pooled"][str(v)]["max_airborne"] for v in z_levels])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot(mean_all, z, "-o", color=COLOR_WITH, lw=2.0, ms=6, label="with deposited particles (all, x < 1315 m)")
    ax.plot(mean_air, z, "-s", color=COLOR_WITHOUT, lw=2.0, ms=6, label="airborne only (sensor definition)")
    ax.plot(max_all, z, ":o", color=COLOR_WITH, lw=1.2, ms=4, alpha=0.7, label="max cell, with deposited")
    ax.plot(max_air, z, ":s", color=COLOR_WITHOUT, lw=1.2, ms=4, alpha=0.7, label="max cell, airborne only")
    ax.axhline(config.KERNEL_H, color="#777777", ls="--", lw=1.0)
    ax.text(1.0, config.KERNEL_H + 0.25, f"kernel support H = {config.KERNEL_H:g} m",
            fontsize=8.5, color="#555555", va="bottom", transform=ax.get_yaxis_transform(), ha="right")
    ax.axhline(config.DRONE_Z, color="#777777", ls=":", lw=1.0)
    ax.text(1.0, config.DRONE_Z + 0.25, f"drone altitude {config.DRONE_Z:g} m", fontsize=8.5, color="#555555",
            va="bottom", transform=ax.get_yaxis_transform(), ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("pooled mean density over cells with any particle within H  [particles / m$^3$]")
    ax.set_ylabel("query height z [m]")
    ax.set_ylim(0.0, max(z) + 1.5)
    ax.set_title(f"Near-ground density profile, 13 sources pooled, step {step}", fontsize=10.5)
    ax.grid(which="major", color="#e3e3e3")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.86), frameon=True, framealpha=0.95)   # upper middle of the log axis is empty (means ~1e-2..1e-1, maxima ~1e1) and clear of the H / drone-altitude labels
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=config.FIG_DPI_FINAL, bbox_inches="tight")
    plt.close(fig)
    return {"z_m": z.tolist(), "pooled_mean_all": mean_all.tolist(), "pooled_mean_airborne": mean_air.tolist(),
            "pooled_max_all": max_all.tolist(), "pooled_max_airborne": max_air.tolist()}


# ------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frame", type=int, default=config.FIG_DATA_FRAME_INDEX, help="LDM frame index (0..599)")
    ap.add_argument("--out-dir", type=Path, default=config.FIG_DIR)
    ap.add_argument("--json", type=Path, default=config.CACHE_DIR / "fig_data_numbers.json")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    step = config.index_to_step(args.frame)
    frame = load_frame(args.frame)
    t_load = time.perf_counter() - t0
    print(f"loaded frame {args.frame} (step {step}): {frame.n:,} particles, deposited {int(frame.deposited.sum()):,}, "
          f"outflow {int(frame.outflow.sum()):,}, airborne {int(frame.airborne.sum()):,}  [{t_load:.1f} s]", flush=True)

    z_levels, sources, grid = config.FIG_DATA_Z_LEVELS, config.ALL_SOURCES, SlabGrid()
    t1 = time.perf_counter()
    numbers = compute_numbers(frame, z_levels, sources, grid)
    t_gather = time.perf_counter() - t1

    out_dir = args.out_dir
    f_heat = out_dir / "fig2a_deposited_ratio.png"
    f_prof = out_dir / "fig2a_profile.png"
    heat = render_ratio_heatmap(numbers, sources, z_levels, step, f_heat)
    prof = render_profile(numbers, z_levels, step, f_prof)
    table = markdown_table(numbers, sources, z_levels)

    out = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_index": int(args.frame), "step": int(step), "ldm_file": str(config.ldm_path(args.frame)),
        "z_levels_m": list(z_levels), "sources": list(sources), "grid": grid.to_dict(),
        "kernel_h_m": config.KERNEL_H, "deposit_z_m": config.DEPOSIT_Z, "x_outflow_m": config.X_OUTFLOW,
        "definitions": {"all": "particles of the source with x < X_OUTFLOW (airborne + deposited)",
                        "airborne": "frame.airborne: not deposited (z == DEPOSIT_Z & v == 0) and x < X_OUTFLOW",
                        "mean": "mean over the slab cells where the 'all' density > 0 (same cells for both sets)",
                        "ratio_mean": "mean_all / mean_airborne", "ratio_max": "max_all / max_airborne",
                        "frac_all_pos_airborne_zero": "fraction of those cells whose airborne density is exactly 0",
                        "pooled": "all (source, cell) pairs with all-density > 0, 13 sources"},
        "frame_summary": {"n_total": frame.n, "n_deposited": int(frame.deposited.sum()),
                          "n_outflow": int(frame.outflow.sum()), "n_airborne": int(frame.airborne.sum())},
        "per_source": numbers["per_source"], "pooled": numbers["pooled"],
        "figure_heatmap": heat, "figure_profile": prof, "markdown_table": table,
        "files": {p.name: {"path": str(p), "bytes": p.stat().st_size} for p in (f_heat, f_prof)},
        "timing_s": {"load_frame": round(t_load, 2), "gathers_total": round(t_gather, 2),
                     "per_source": numbers["timing_per_source_s"], "total": round(time.perf_counter() - t0, 2)},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table)
    print(json.dumps({"pooled": out["pooled"], "files": out["files"], "timing_s": out["timing_s"]}, indent=2))
    return out


if __name__ == "__main__":
    main()