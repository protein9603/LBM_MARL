"""Independent alignment check: STL-derived occupancy grid vs the SPH solver's own building evidence.

Ground truth for "where the buildings are in the SPH/LDM frame" comes from the fluid file itself:
  (1) p_type == 1000 immersed-boundary surface points (offlattice_points.npz: foot_cnt per 2.5 m cell),
  (2) near-ground lattice cells with |u| < 0.1 m/s at z = 1.25 m (fluid_slices_2p5m.npz).
The environment uses occupancy_2m_flowframe.npz (STL shifted by config.STL_SHIFT, rasterised at 2 m).
We resample the 2 m occupancy onto the 2.5 m lattice and report Jaccard overlap, the best integer
cross-correlation shift (expected (0, 0) cells), the low-speed-cell overlap, the fraction of low-altitude
LDM particles (frame 599, z < 10 m) that fall inside occupied cells, and the shifted-STL bbox vs the IB bbox.
Output: cache/validate_alignment.json and 분석그림/icrs15/check_alignment.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from srcloc_env import config
from srcloc_env.field.concentration_field import LdmSlabBackend

ART = config.ARTIFACT_DIR
OUT_JSON = config.CACHE_DIR / "validate_alignment.json"
OUT_PNG = config.FIG_DIR / "check_alignment.png"


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def best_shift(ref: np.ndarray, test: np.ndarray, max_cells: int = 8) -> tuple[tuple[int, int], np.ndarray]:
    """Return the integer (dix, diy) that maximises overlap of test shifted onto ref, and the score map."""
    r = np.arange(-max_cells, max_cells + 1)
    score = np.zeros((r.size, r.size))
    for a, dix in enumerate(r):
        for b, diy in enumerate(r):
            shifted = np.roll(np.roll(test, dix, axis=0), diy, axis=1)
            score[a, b] = jaccard(ref, shifted)
    a, b = np.unravel_index(np.argmax(score), score.shape)
    return (int(r[a]), int(r[b])), score


def main() -> None:
    ib = np.load(ART / "fluid" / "offlattice_points.npz")
    x25, y25 = ib["x"], ib["y"]                      # 2.5 m lattice, x[ix], y[iy]
    ib_foot = ib["foot_cnt"] > 0                     # [ix, iy]
    ib_sample = ib["sample"]                         # (n, 4): x, y, z, ? sample of IB points
    sl = np.load(ART / "fluid" / "fluid_slices_2p5m.npz")
    uvw = sl["uvw"]; zl = sl["z_levels"]
    speed125 = np.linalg.norm(uvw[int(np.argmin(np.abs(zl - 1.25)))], axis=-1)   # [ix, iy]
    low = speed125 < 0.1
    occ_npz = np.load(ART / "stl" / "occupancy_2m_flowframe.npz")
    occ, hmap = occ_npz["occ"], occ_npz["hmap"]      # [ix, iy] 2 m cells, x = ix*2, y = -657.5 + iy*2
    meta = json.loads(str(occ_npz["meta"]))
    x0, y0, res = meta["x0"], meta["y0"], meta["res"]

    # resample 2 m occupancy onto the 2.5 m lattice nodes (nearest 2 m cell)
    ix2 = np.clip(np.rint((x25 - x0) / res).astype(int), 0, occ.shape[0] - 1)
    iy2 = np.clip(np.rint((y25 - y0) / res).astype(int), 0, occ.shape[1] - 1)
    occ25 = occ[np.ix_(ix2, iy2)]

    (dix, diy), score = best_shift(ib_foot, occ25)
    j0 = jaccard(ib_foot, occ25)
    j_low = jaccard(low, occ25)
    frac_low_in_occ = float(low[occ25].mean())
    frac_low_out_occ = float(low[~occ25].mean())

    # shifted STL bbox vs IB bbox
    tri = np.load(ART / "stl" / "stl_triangles.npz")["tri"].reshape(-1, 3) + config.STL_SHIFT
    stl_bbox = [tri[:, 0].min(), tri[:, 0].max(), tri[:, 1].min(), tri[:, 1].max()]
    ib_xy = ib_sample[:, :2]
    ib_bbox = [ib_xy[:, 0].min(), ib_xy[:, 0].max(), ib_xy[:, 1].min(), ib_xy[:, 1].max()]

    # low-altitude LDM particles inside occupied cells (roof above the particle)
    be = LdmSlabBackend()
    pts = be.airborne_particles(config.N_FILES - 1)
    lowp = pts[pts[:, 2] < 10.0]
    i2 = np.clip(np.floor((lowp[:, 0] - x0) / res + 0.5).astype(int), 0, occ.shape[0] - 1)
    j2 = np.clip(np.floor((lowp[:, 1] - y0) / res + 0.5).astype(int), 0, occ.shape[1] - 1)
    inside = occ[i2, j2] & (hmap[i2, j2] > lowp[:, 2])
    frac_particles_inside = float(inside.mean())

    result = {
        "jaccard_occ_vs_ib_footprint_2p5m": round(j0, 4),
        "best_integer_shift_cells_(dix,diy)": [dix, diy], "best_shift_metres": [dix * 2.5, diy * 2.5],
        "jaccard_at_best_shift": round(float(score.max()), 4),
        "jaccard_lowspeed_z1.25_vs_occ": round(j_low, 4),
        "frac_lowspeed_cells_inside_occ": round(frac_low_in_occ, 4),
        "frac_lowspeed_cells_outside_occ": round(frac_low_out_occ, 4),
        "shifted_stl_bbox_xmin_xmax_ymin_ymax": [round(float(v), 2) for v in stl_bbox],
        "ib_points_bbox_xmin_xmax_ymin_ymax": [round(float(v), 2) for v in ib_bbox],
        "ldm_particles_z_lt_10m_frame599": int(lowp.shape[0]),
        "frac_ldm_particles_inside_occupied_cells": round(frac_particles_inside, 5),
        "stl_shift_used": config.STL_SHIFT.tolist(),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax = axes[0, 0]
    xor = np.zeros(ib_foot.shape, dtype=int); xor[ib_foot & occ25] = 1; xor[ib_foot & ~occ25] = 2; xor[~ib_foot & occ25] = 3
    cmap = matplotlib.colors.ListedColormap(["white", "0.6", "tab:blue", "tab:red"])
    ax.imshow(xor.T, origin="lower", extent=[x25[0] - 1.25, x25[-1] + 1.25, y25[0] - 1.25, y25[-1] + 1.25], cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_title(f"(a) SPH building points vs STL occupancy (2.5 m lattice)\ngrey = both, blue = SPH only, red = STL only; Jaccard {j0:.3f}, best shift {(dix*2.5, diy*2.5)} m")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    for s, (sx, sy) in config.SOURCES_XY.items():
        ax.plot(sx, sy, "*", color="gold", mec="k", ms=9)

    def zoom(ax, cx, cy, half, title):
        sel = (np.abs(ib_sample[:, 0] - cx) < half) & (np.abs(ib_sample[:, 1] - cy) < half)
        ax.imshow(low.T, origin="lower", extent=[x25[0] - 1.25, x25[-1] + 1.25, y25[0] - 1.25, y25[-1] + 1.25], cmap="Greys", alpha=0.35, interpolation="nearest")
        ax.contour(x0 + np.arange(occ.shape[0]) * res, y0 + np.arange(occ.shape[1]) * res, occ.T.astype(float), levels=[0.5], colors="red", linewidths=1.2)
        ax.scatter(ib_sample[sel, 0], ib_sample[sel, 1], s=3, c="tab:blue", label="SPH p_type 1000 points")
        selp = (np.abs(lowp[:, 0] - cx) < half) & (np.abs(lowp[:, 1] - cy) < half)
        ax.scatter(lowp[selp, 0], lowp[selp, 1], s=1, c="k", alpha=0.5, label="LDM particles z<10 m")
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_aspect("equal")
        ax.set_title(title); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    for ax, s in ((axes[0, 1], 110), (axes[1, 0], 106), (axes[1, 1], 103)):
        sx, sy = config.SOURCES_XY[s]
        zoom(ax, sx, sy, 120, f"zoom source {s}: grey = |u|<0.1 m/s at z=1.25 m, red = STL occupancy outline")
        ax.plot(sx, sy, "*", color="gold", mec="k", ms=14, label=f"source {s}")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print("saved", OUT_PNG)


if __name__ == "__main__":
    main()