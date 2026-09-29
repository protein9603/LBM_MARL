"""T0-3 / T0-4: slab bilinear interpolation vs the exact Wendland C6 gather, and the snapshot growth table
(plan 3 S0, module row `scripts/validate_slabs.py`; data_cache.md "사용 규칙").

Usage (repo root, venv): python -m srcloc_env.scripts.validate_slabs [--index 599] [--seed 0]
Writes config.CACHE_DIR / validate_slabs.json and prints T0-3 PASS/FAIL.

T0-3 (plan S0 item 3): at frame index config.FIG_SCENE_FRAME_INDEX (599 = step 30000) and z = config.DRONE_Z
the drone altitude density returned by LdmSlabBackend.density (bilinear on the 5 m float16 slab,
field/concentration_field.py) is compared with the exact gather that the gridder used to build the slab
(preprocess/gridder.py: C6 sum over the AIRBORNE particles of one source within |z - z_k| <= KERNEL_H, divided
by LATTICE_CELL_VOLUME -> particles/m^3, report 2.1 units).  For each of the 13 sources (report 2.6)
config.T0_3_N_QUERIES_PER_SOURCE query positions are drawn inside that source's occupied slab cells (density
> 0) and jittered uniformly within the cell, so the queries are not at cell centres where the two agree by
construction.  Reported per source and pooled: Pearson r of the raw values, Pearson r of
log10(value + T0_3_LOG_OFFSET), median / p90 relative error where the exact value > T0_3_REL_ERR_MIN_EXACT,
and the two zero-disagreement fractions.  PASS if the pooled raw r >= config.T0_3_MIN_R.  A second set of
config.T0_3_N_UNIFORM positions uniform over the whole slab grid compares the 13-source sum with the exact
13-source sum (zero-agreement rate and r).  The exact values come from frames/frame_XXX.npz (airborne
particles, report 2.4 masks) through one cKDTree per source.

T0-4 (plan S0 item 4, slide 6 footnote): per-source and total airborne particle counts at
config.T0_4_FRAME_INDICES from frames/meta_XXX.json (keys counts_by_source_airborne, n_airborne) and the growth
ratio last/first; the plan expects about +46% overall (config.T0_4_EXPECTED_GROWTH_TOTAL).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from srcloc_env import config
from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.sensor.kernel import c6_gather


# ------------------------------------------------------------------------------------------ sampling
def sample_in_occupied_cells(dens2d: np.ndarray, grid: SlabGrid, n: int, rng: np.random.Generator) -> np.ndarray:
    """n random (x, y) positions uniformly inside randomly chosen (with replacement) cells where dens2d > 0.

    dens2d is the (ny, nx) slab of one source at one z level (row-major iy, ix as in SlabGrid.cell_centres);
    the jitter is uniform over the full 5 m cell so the points are generally not at cell centres.
    """
    iy, ix = np.nonzero(np.asarray(dens2d) > 0)
    if iy.size == 0:
        raise ValueError("no occupied cells to sample from")
    k = rng.integers(0, iy.size, n)
    x = grid.x0 + grid.res * (ix[k] + rng.uniform(0.0, 1.0, n))
    y = grid.y0 + grid.res * (iy[k] + rng.uniform(0.0, 1.0, n))
    return np.column_stack([x, y])


def sample_uniform(grid: SlabGrid, n: int, rng: np.random.Generator) -> np.ndarray:
    """n random (x, y) positions uniform over the whole slab grid extent."""
    x = rng.uniform(grid.x0, grid.x0 + grid.res * grid.nx, n)
    y = rng.uniform(grid.y0, grid.y0 + grid.res * grid.ny, n)
    return np.column_stack([x, y])


# ------------------------------------------------------------------------------------------ exact gather
def band_tree(particles: np.ndarray, z: float) -> tuple[np.ndarray, cKDTree | None]:
    """Particles with |z_p - z| <= KERNEL_H (the only ones that can contribute at height z) and their cKDTree."""
    xyz = np.asarray(particles, dtype=np.float64)
    if xyz.shape[0] == 0:
        return xyz.reshape(0, 3), None
    pts = xyz[np.abs(xyz[:, 2] - float(z)) <= config.KERNEL_H]
    return pts, (cKDTree(pts) if pts.shape[0] else None)


def exact_density(xy: np.ndarray, z: float, pts: np.ndarray, tree: cKDTree | None) -> np.ndarray:
    """Exact C6 gather [particles/m^3] at (x, y, z) over the pre-banded particles (same maths as the gridder)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if tree is None or pts.shape[0] == 0:
        return np.zeros(xy.shape[0], dtype=np.float64)
    q = np.column_stack([xy, np.full(xy.shape[0], float(z))])
    return c6_gather(q, pts, tree) / config.LATTICE_CELL_VOLUME


# ------------------------------------------------------------------------------------------ statistics
def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; NaN when either vector is constant (fewer than 2 points or zero variance)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_values(slab: np.ndarray, exact: np.ndarray) -> dict:
    """T0-3 agreement numbers between slab-interpolated and exact densities (both particles/m^3, (n,))."""
    slab = np.asarray(slab, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    n = int(slab.size)
    sel = exact > config.T0_3_REL_ERR_MIN_EXACT
    rel = np.abs(slab[sel] - exact[sel]) / exact[sel]
    return {
        "n": n,
        "r_raw": pearson_r(slab, exact),
        "r_log10": pearson_r(np.log10(slab + config.T0_3_LOG_OFFSET), np.log10(exact + config.T0_3_LOG_OFFSET)),
        "n_rel_err": int(sel.sum()),
        "rel_err_median": float(np.median(rel)) if rel.size else float("nan"),
        "rel_err_p90": float(np.percentile(rel, config.T0_3_REL_ERR_PERCENTILE)) if rel.size else float("nan"),
        "frac_slab_zero_exact_pos": float(np.mean((slab == 0.0) & (exact > 0.0))) if n else float("nan"),
        "frac_slab_pos_exact_zero": float(np.mean((slab > 0.0) & (exact == 0.0))) if n else float("nan"),
        "zero_agreement": float(np.mean((slab == 0.0) == (exact == 0.0))) if n else float("nan"),
        "mean_slab": float(slab.mean()) if n else float("nan"),
        "mean_exact": float(exact.mean()) if n else float("nan"),
        "max_slab": float(slab.max()) if n else float("nan"),
        "max_exact": float(exact.max()) if n else float("nan"),
    }


# ------------------------------------------------------------------------------------------ T0-4 table
def growth_table(metas: dict[int, dict]) -> tuple[dict, str]:
    """Per-source / total airborne counts at the given frame indices and the last/first growth ratios.

    metas maps frame index -> parsed frames/meta_XXX.json (keys counts_by_source_airborne, n_airborne).
    Returns (numbers, Markdown table string).
    """
    indices = sorted(int(i) for i in metas)
    first, last = indices[0], indices[-1]
    sources = sorted({int(s) for i in indices for s in metas[i]["counts_by_source_airborne"]})
    counts = {s: {i: int(metas[i]["counts_by_source_airborne"].get(str(s), metas[i]["counts_by_source_airborne"].get(s, 0)))
                  for i in indices} for s in sources}
    totals = {i: int(metas[i]["n_airborne"]) for i in indices}

    def ratio(a: int, b: int) -> float:
        return float(b) / float(a) if a else float("nan")

    per_source_ratio = {s: ratio(counts[s][first], counts[s][last]) for s in sources}
    total_ratio = ratio(totals[first], totals[last])
    header = ["source"] + [f"index {i} (step {config.index_to_step(i)})" for i in indices] + [f"{last}/{first}"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for s in sources:
        row = [str(s)] + [f"{counts[s][i]:,}" for i in indices] + [f"{per_source_ratio[s]:.3f}"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| " + " | ".join(["**total**"] + [f"**{totals[i]:,}**" for i in indices] + [f"**{total_ratio:.3f}**"]) + " |")
    numbers = {
        "indices": indices, "steps": {i: config.index_to_step(i) for i in indices},
        "counts_by_source_airborne": {s: {i: counts[s][i] for i in indices} for s in sources},
        "n_airborne": totals,
        "growth_ratio_by_source": per_source_ratio, "growth_ratio_total": total_ratio,
        "growth_pct_total": 100.0 * (total_ratio - 1.0),
        "expected_growth_ratio_total": config.T0_4_EXPECTED_GROWTH_TOTAL,
    }
    return numbers, "\n".join(lines)


# ------------------------------------------------------------------------------------------ main
def run_t0_3(be: LdmSlabBackend, index: int, z: float, rng: np.random.Generator) -> dict:
    sf = be.slab(index)
    zi = sf.z_index(z)
    grid = sf.grid
    per_source: dict[int, dict] = {}
    all_slab, all_exact = [], []
    timings = {}
    for si, s in enumerate(sf.sources):
        t0 = time.perf_counter()
        xy = sample_in_occupied_cells(sf.density[si, zi], grid, config.T0_3_N_QUERIES_PER_SOURCE, rng)
        slab_v = be.density([s], xy, index, z=z)
        part = be.airborne_particles(index, s)
        pts, tree = band_tree(part, z)
        exact_v = exact_density(xy, z, pts, tree)
        stats = compare_values(slab_v, exact_v)
        stats.update({"n_airborne": int(part.shape[0]), "n_in_band": int(pts.shape[0]),
                      "n_occupied_cells": int((sf.density[si, zi] > 0).sum()), "seconds": time.perf_counter() - t0})
        per_source[int(s)] = stats
        all_slab.append(slab_v)
        all_exact.append(exact_v)
        timings[int(s)] = stats["seconds"]
    pooled = compare_values(np.concatenate(all_slab), np.concatenate(all_exact))

    # uniform positions over the whole grid, all sources summed
    t0 = time.perf_counter()
    xy_u = sample_uniform(grid, config.T0_3_N_UNIFORM, rng)
    slab_u = be.density(sf.sources, xy_u, index, z=z)
    exact_u = np.zeros(xy_u.shape[0], dtype=np.float64)
    for s in sf.sources:
        pts, tree = band_tree(be.airborne_particles(index, s), z)
        exact_u += exact_density(xy_u, z, pts, tree)
    uniform = compare_values(slab_u, exact_u)
    uniform["frac_exact_nonzero"] = float(np.mean(exact_u > 0.0))
    uniform["seconds"] = time.perf_counter() - t0

    return {
        "index": index, "step": config.index_to_step(index), "z": z, "z_levels": list(sf.z_levels),
        "grid": grid.to_dict(), "n_queries_per_source": config.T0_3_N_QUERIES_PER_SOURCE,
        "n_uniform": config.T0_3_N_UNIFORM, "criterion_min_r": config.T0_3_MIN_R,
        "rel_err_min_exact": config.T0_3_REL_ERR_MIN_EXACT, "log_offset": config.T0_3_LOG_OFFSET,
        "per_source": per_source, "pooled": pooled, "uniform_all_sources": uniform,
        "pass": bool(np.isfinite(pooled["r_raw"]) and pooled["r_raw"] >= config.T0_3_MIN_R),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=config.FIG_SCENE_FRAME_INDEX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_slabs.json")
    args = ap.parse_args()

    be = LdmSlabBackend(config.CACHE_DIR, max_cached_frames=config.FIELD_MAX_CACHED_FRAMES)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    t0_3 = run_t0_3(be, args.index, config.DRONE_Z, rng)
    t0_3["seconds_total"] = time.perf_counter() - t0

    metas = {i: be.frame_meta(i) for i in config.T0_4_FRAME_INDICES}
    t0_4, table = growth_table(metas)
    t0_4["markdown"] = table

    res = {"seed": args.seed, "t0_3": t0_3, "t0_4": t0_4}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(table)
    p = t0_3["pooled"]
    print(f"T0-3 pooled r_raw {p['r_raw']:.4f} (r_log10 {p['r_log10']:.4f}, rel err median {p['rel_err_median']:.4f}, "
          f"p90 {p['rel_err_p90']:.4f}), criterion >= {config.T0_3_MIN_R}:", "PASS" if t0_3["pass"] else "FAIL")
    print(f"T0-4 total airborne growth {t0_4['indices'][0]} -> {t0_4['indices'][-1]}: x{t0_4['growth_ratio_total']:.3f} "
          f"({t0_4['growth_pct_total']:+.1f}%, expected about x{config.T0_4_EXPECTED_GROWTH_TOTAL})")
    print("VALIDATE_SLABS", "PASS" if t0_3["pass"] else "FAIL")


if __name__ == "__main__":
    main()