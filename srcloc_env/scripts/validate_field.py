"""T0-5: slab query latency and sanity numbers of LdmSlabBackend on the real cache (plan 3 S0).

Usage: python -m srcloc_env.scripts.validate_field [--index 599] [--n-single 10000] [--n-batch 1000]
Writes config.CACHE_DIR / validate_field.json and prints T0-5 PASS/FAIL
(single cached query mean < config.T0_5_QUERY_LATENCY_S = 0.1 ms).
Also reports: per-source slab maximum at z = DRONE_Z (must equal the gridder stats in frames/meta_XXX.json,
e.g. 110 -> 12.0, 108 -> 0.368, 113 -> 0.363 at index 599), the density at the 13 source positions
(config.SOURCES_XY, report 2.6) and the LRU footprint after loading FIELD_MAX_CACHED_FRAMES frames.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.field.concentration_field import LdmSlabBackend


def time_calls(fn, n: int) -> dict:
    """Mean / median / p99 wall time [s] of n repeated calls (time.perf_counter around each call)."""
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        t[i] = time.perf_counter() - t0
    return {"n": int(n), "mean_s": float(t.mean()), "median_s": float(np.median(t)),
            "p99_s": float(np.percentile(t, 99)), "max_s": float(t.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=config.N_FILES - 1)
    ap.add_argument("--n-single", type=int, default=10000)
    ap.add_argument("--n-batch", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_field.json")
    args = ap.parse_args()

    be = LdmSlabBackend(config.CACHE_DIR, max_cached_frames=config.FIELD_MAX_CACHED_FRAMES)
    idx, z = args.index, config.DRONE_Z

    t0 = time.perf_counter()
    sf = be.slab(idx)
    load_s = time.perf_counter() - t0
    grid = sf.grid
    rng = np.random.default_rng(args.seed)

    # ---- T0-5 timing (frame already cached) ------------------------------------------------------
    src = config.SOURCES_XY[110]
    one = np.array([[src[0] + 20.0, src[1]]])
    be.density(110, one, idx, z)                                            # warm up
    single = time_calls(lambda: be.density(110, one, idx, z), args.n_single)
    single_all = time_calls(lambda: be.density(config.ALL_SOURCES, one, idx, z), args.n_single // 10)
    batch_xy = np.column_stack([rng.uniform(grid.x0, grid.x0 + grid.res * grid.nx, args.n_batch),
                                rng.uniform(grid.y0, grid.y0 + grid.res * grid.ny, args.n_batch)])
    batch = time_calls(lambda: be.density(110, batch_xy, idx, z), 200)
    batch_all = time_calls(lambda: be.density(config.ALL_SOURCES, batch_xy, idx, z), 200)
    t0_5_pass = single["mean_s"] < config.T0_5_QUERY_LATENCY_S

    # ---- per-source maxima vs the gridder numbers in meta_XXX.json ------------------------------
    zi = sf.z_index(z)
    per_source_max = {int(s): float(sf.density[si, zi].max()) for si, s in enumerate(sf.sources)}
    meta = be.frame_meta(idx)
    meta_z = [float(v) for v in meta["slab"]["z_levels"]]
    meta_max = {int(s): float(meta["slab"]["stats"]["max"][si][meta_z.index(z)]) for si, s in enumerate(sf.sources)}
    max_match = all(abs(per_source_max[s] - meta_max[s]) <= 1e-6 * max(1.0, meta_max[s]) for s in per_source_max)
    # bilinear maximum over cell centres must reproduce the array maximum exactly
    xc, yc = grid.x_centres, grid.y_centres
    xx, yy = np.meshgrid(xc, yc)
    centres = np.column_stack([xx.ravel(), yy.ravel()])
    interp_max = {int(s): float(be.density(s, centres, idx, z).max()) for s in sf.sources}
    interp_match = all(abs(interp_max[s] - per_source_max[s]) <= 1e-6 * max(1.0, per_source_max[s]) for s in interp_max)

    # ---- density at the 13 source positions (own source and all sources) -------------------------
    src_xy = np.array([config.SOURCES_XY[s] for s in config.ALL_SOURCES])
    own = {int(s): float(be.density(s, src_xy[i], idx, z)[0]) for i, s in enumerate(config.ALL_SOURCES)}
    total = be.density(config.ALL_SOURCES, src_xy, idx, z)
    flipped = be.density(config.ALL_SOURCES, src_xy, idx, z, flip_y=True)
    mirrored = be.density(config.ALL_SOURCES, src_xy * np.array([1.0, -1.0]), idx, z)
    at_sources = {int(s): {"xy": [float(v) for v in src_xy[i]], "own": own[int(s)], "all_sources": float(total[i]),
                           "flip_y": float(flipped[i])} for i, s in enumerate(config.ALL_SOURCES)}
    outside = be.density(config.ALL_SOURCES, np.array([[grid.x0 - 1.0, 0.0], [0.0, 0.0], [grid.x0 + 10.0, 1000.0]]), idx, z)

    # ---- LRU footprint for FIELD_MAX_CACHED_FRAMES frames (3-level frames from 400..599) ---------
    lru_indices = list(range(config.FRAME_RANGE_MODE_F[1], config.FRAME_RANGE_MODE_F[1] - config.FIELD_MAX_CACHED_FRAMES, -1))
    t0 = time.perf_counter()
    for i in lru_indices:
        be.slab(i)
    lru_load_s = time.perf_counter() - t0
    lru = {"indices": be.cached_frames, "n_frames": len(be.cached_frames), "bytes": be.cache_nbytes(),
           "megabytes": be.cache_nbytes() / 1e6, "bytes_per_frame": be.cache_nbytes() // max(1, len(be.cached_frames)),
           "load_seconds_total": lru_load_s}
    # one more frame evicts the oldest
    be.slab(0)
    lru["after_extra_load"] = be.cached_frames
    lru["max_respected"] = len(be.cached_frames) == config.FIELD_MAX_CACHED_FRAMES

    res = {
        "index": idx, "z": z, "z_levels": list(sf.z_levels), "sources": list(sf.sources), "grid": grid.to_dict(),
        "slab_shape": list(sf.density.shape), "slab_load_seconds": load_s,
        "t0_5": {"criterion_s": config.T0_5_QUERY_LATENCY_S, "single_query_one_source": single,
                 "single_query_13_sources": single_all, "batch_1000_one_source": batch,
                 "batch_1000_13_sources": batch_all, "pass": bool(t0_5_pass)},
        "per_source_max_slab": per_source_max, "per_source_max_meta": meta_max, "max_matches_meta": bool(max_match),
        "per_source_max_bilinear_at_centres": interp_max, "bilinear_matches_slab_max": bool(interp_match),
        "density_at_sources": at_sources, "flip_y_equals_mirror": bool(np.allclose(flipped, mirrored)),
        "outside_grid_zero": bool(np.all(outside == 0.0)),
        "lru": lru,
        "pass": bool(t0_5_pass and max_match and interp_match and np.allclose(flipped, mirrored) and np.all(outside == 0.0)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"T0-5 single query mean {single['mean_s'] * 1e3:.4f} ms (median {single['median_s'] * 1e3:.4f} ms), "
          f"criterion < {config.T0_5_QUERY_LATENCY_S * 1e3:.2f} ms:", "PASS" if t0_5_pass else "FAIL")
    print("VALIDATE_FIELD", "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()