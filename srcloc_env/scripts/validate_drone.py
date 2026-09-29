"""D2 obstacle-map validation: check env/drone.py against the real 2 m occupancy raster and stl_tools.

Usage: python -m srcloc_env.scripts.validate_drone [--out cache/validate_drone.json]
Checks (report 2.6, 4.1-4.3; plan 4.5):
  0. keys / shapes / dtypes of occupancy_2m_flowframe.npz and the axis-order evidence (bbox under both readings
     vs config.BUILDING_BBOX_*, 115 m tower near source 110, hmap no-building value);
  1. all 13 sources (config.SOURCES_XY) free at DRONE_Z with the 2 m margin, and a point inside the 115 m tower
     footprint (hmap maximum within TOWER_115_SEARCH_RADIUS_M of source 110) blocked;
  2. agreement with stl_tools.is_inside_building (margin 0, imported from config.ARTIFACT_DIR/stl/stl_tools.py)
     on VALIDATE_DRONE_N_RANDOM uniform points in the building bbox at DRONE_Z: ours (2 m margin) must be a
     superset of the reference (ref-blocked-but-ours-free == 0); with margin 0 the two must agree except at
     cells whose roof is exactly at z (ours blocks hmap >= z, stl_tools blocks z < hmap strictly);
  3. fraction of the 5 m slab grid cell centres (config.SLAB_*) that are no-fly at DRONE_Z;
  4. timing of is_free and ray_distances on VALIDATE_DRONE_N_TIMING points.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.preprocess.gridder import SlabGrid


def load_stl_tools(path: Path = config.ARTIFACT_DIR / "stl" / "stl_tools.py"):
    """Import the reference implementation from the analysis folder without copying it into the repo."""
    spec = importlib.util.spec_from_file_location("stl_tools_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def axis_order_evidence(om: ObstacleMap, meta: dict) -> dict:
    """Occupied-cell bbox under both readings vs the report 4.2 building bbox; tower heights near sources."""
    r, c = np.nonzero(om.occ)
    x, y = om.x, om.y
    bbox_ix_iy = {"x": [float(x[r.min()]), float(x[r.max()])], "y": [float(y[c.min()]), float(y[c.max()])]}
    bbox_iy_ix = {"x": [float(x[c.min()]) if c.max() < x.size else None, float(x[c.max()]) if c.max() < x.size else None],
                  "y": [float(y[r.min()]) if r.max() < y.size else None, float(y[r.max()]) if r.max() < y.size else None]}

    def err(b):
        if any(v is None for v in b["x"] + b["y"]):
            return float("inf")
        return abs(b["x"][0] - config.BUILDING_BBOX_X[0]) + abs(b["x"][1] - config.BUILDING_BBOX_X[1]) + \
            abs(b["y"][0] - config.BUILDING_BBOX_Y[0]) + abs(b["y"][1] - config.BUILDING_BBOX_Y[1])

    chosen = "ix,iy" if err(bbox_ix_iy) <= err(bbox_iy_ix) else "iy,ix"
    towers = {}
    w = int(np.ceil(config.TOWER_115_SEARCH_RADIUS_M / om.res))
    for sid in (110, 106):
        sx, sy = config.SOURCES_XY[sid]
        ix, iy = om.cell_index(np.array([sx, sy]))
        sub = om.hmap[max(ix - w, 0):ix + w + 1, max(iy - w, 0):iy + w + 1]
        subT = om.hmap[max(iy - w, 0):iy + w + 1, max(ix - w, 0):ix + w + 1]
        towers[str(sid)] = {"hmax_ix_iy": float(sub.max()), "hmax_iy_ix": float(subT.max()),
                            "hmap_at_source_cell": float(om.hmap[ix, iy])}
    return {"meta": meta, "bbox_ix_iy": bbox_ix_iy, "bbox_iy_ix": bbox_iy_ix,
            "ref_bbox": {"x": list(config.BUILDING_BBOX_X), "y": list(config.BUILDING_BBOX_Y)},
            "tower_hmax_within_80m": towers, "chosen": chosen, "matches_config": chosen == config.OCC_AXIS_ORDER,
            "occ_equals_hmap_gt0": bool(np.array_equal(om.occ, om.hmap > config.OCC_HMAP_NO_BUILDING)),
            "hmap_min": float(om.hmap.min()), "hmap_max": float(om.hmap.max()),
            "n_occupied": int(om.occ.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_drone.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    om = ObstacleMap.load()
    load_s = time.perf_counter() - t0
    with np.load(config.OCCUPANCY_2M_NPZ) as z:
        keys = {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}
        meta = json.loads(str(z["meta"]))
    res: dict = {"path": str(config.OCCUPANCY_2M_NPZ), "keys": keys, "load_seconds": round(load_s, 3),
                 "z": config.DRONE_Z, "margin_m": config.BUILDING_MARGIN_M,
                 "axis_order": axis_order_evidence(om, meta)}
    checks: dict[str, bool] = {"axis_order_matches_config": res["axis_order"]["matches_config"],
                               "occ_equals_hmap_gt0": res["axis_order"]["occ_equals_hmap_gt0"]}
    z = config.DRONE_Z

    # 1. sources free, tower blocked
    ids = list(config.ALL_SOURCES)
    xy = np.array([config.SOURCES_XY[s] for s in ids])
    free = om.is_free(xy, z)
    rays = om.ray_distances(xy, z)
    res["sources_15m"] = {str(s): {"xy": xy[i].tolist(), "free": bool(free[i]),
                                   "nearest_ray_m": float(rays[i].min()),
                                   "ray_m": np.round(rays[i], 1).tolist()} for i, s in enumerate(ids)}
    res["sources_all_free"] = bool(free.all())
    checks["sources_all_free"] = res["sources_all_free"]
    sx, sy = config.SOURCES_XY[110]
    ix0, iy0 = om.cell_index(np.array([sx, sy]))
    w = int(np.ceil(config.TOWER_115_SEARCH_RADIUS_M / om.res))
    sub = om.hmap[ix0 - w:ix0 + w + 1, iy0 - w:iy0 + w + 1]
    di, dj = np.unravel_index(int(sub.argmax()), sub.shape)
    tix, tiy = ix0 - w + di, iy0 - w + dj
    tower_xy = np.array([om.x[tix], om.y[tiy]])
    tower = {"xy": tower_xy.tolist(), "height_m": float(om.hmap[tix, tiy]),
             "distance_to_source_110_m": float(np.hypot(*(tower_xy - np.array([sx, sy])))),
             "blocked_at_15m": bool(not om.is_free(tower_xy, z)),
             "blocked_at_15m_margin0": bool(not om.is_free(tower_xy, z, margin=0.0)),
             "free_at_120m": bool(om.is_free(tower_xy, 120.0))}
    res["tower_115"] = tower
    checks["tower_blocked"] = tower["blocked_at_15m"] and tower["height_m"] >= 100.0

    # 2. agreement with stl_tools.is_inside_building
    stl_tools = load_stl_tools()
    rng = np.random.default_rng(args.seed)
    n = config.VALIDATE_DRONE_N_RANDOM
    pts = np.column_stack([rng.uniform(*config.BUILDING_BBOX_X, n), rng.uniform(*config.BUILDING_BBOX_Y, n)])
    ref_blocked = stl_tools.is_inside_building(pts[:, 0], pts[:, 1], z, om.hmap, meta, margin=0.0)
    ours_blocked = ~om.is_free(pts, z)
    ours0_blocked = ~om.is_free(pts, z, margin=0.0)
    pix, piy = om.cell_index(pts)
    roof_eq_z = om.hmap[pix, piy] == np.float32(z)
    m0_diff = ref_blocked != ours0_blocked
    agree = {"n_points": n, "ref_blocked_frac": float(ref_blocked.mean()), "ours_blocked_frac": float(ours_blocked.mean()),
             "agreement_pct": float(100.0 * (ref_blocked == ours_blocked).mean()),
             "ours_blocked_ref_free_pct": float(100.0 * (ours_blocked & ~ref_blocked).mean()),
             "ref_blocked_ours_free_pct": float(100.0 * (ref_blocked & ~ours_blocked).mean()),
             "margin0_agreement_pct": float(100.0 * (~m0_diff).mean()),
             "margin0_n_diff": int(m0_diff.sum()),
             "margin0_ref_blocked_ours_free": int((ref_blocked & ~ours0_blocked).sum()),
             "margin0_diff_all_at_roof_eq_z": bool(np.all(roof_eq_z[m0_diff])),
             "n_raster_cells_with_roof_eq_z": int((om.hmap == np.float32(z)).sum())}
    res["stl_tools_agreement"] = agree
    checks["ours_is_superset_of_ref"] = agree["ref_blocked_ours_free_pct"] == 0.0
    checks["margin0_matches_ref_except_roof_eq_z"] = (agree["margin0_ref_blocked_ours_free"] == 0
                                                       and agree["margin0_diff_all_at_roof_eq_z"])

    # 3. slab grid no-fly fraction
    grid = SlabGrid()
    centres = grid.cell_centres(z)[:, :2]
    c_free = om.is_free(centres, z)
    in_dom = om.in_domain(centres)
    mask = om.no_fly_mask(z)
    res["slab_grid_15m"] = {"n_cells": int(centres.shape[0]), "grid": grid.to_dict(),
                            "not_free_frac": float(1.0 - c_free.mean()),
                            "outside_domain_frac": float(1.0 - in_dom.mean()),
                            "no_fly_building_frac_in_domain": float((~c_free[in_dom]).mean()),
                            "raster_no_fly_cell_frac": float(mask.mean()),
                            "raster_no_fly_cell_frac_margin0": float(om.no_fly_mask(z, 0.0).mean()),
                            "raster_occupied_cell_frac": float(om.occ.mean())}

    # 4. timing
    m = config.VALIDATE_DRONE_N_TIMING
    q = np.column_stack([rng.uniform(*config.PF_PRIOR_X, m), rng.uniform(*config.PF_PRIOR_Y, m)])
    om.is_free(q, z)                      # warm the (z, margin) cache
    t_free, t_ray = [], []
    for _ in range(5):
        t1 = time.perf_counter(); om.is_free(q, z); t_free.append(time.perf_counter() - t1)
        t1 = time.perf_counter(); om.ray_distances(q, z); t_ray.append(time.perf_counter() - t1)
    t1 = time.perf_counter(); om.no_fly_mask(z + 1.0); t_mask = time.perf_counter() - t1
    res["timing"] = {"n_points": m, "is_free_median_s": float(np.median(t_free)),
                     "ray_distances_median_s": float(np.median(t_ray)), "no_fly_mask_build_s": t_mask}

    res["checks"] = checks
    res["pass"] = bool(all(checks.values()))
    res["seconds"] = round(time.perf_counter() - t0, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    ao = res["axis_order"]
    print(f"axis order: {ao['chosen']} (config {config.OCC_AXIS_ORDER}); bbox ix,iy x {ao['bbox_ix_iy']['x']} y {ao['bbox_ix_iy']['y']}")
    print(f"sources free at {z} m: {int(free.sum())}/13 -> {'PASS' if checks['sources_all_free'] else 'FAIL'}")
    print(f"tower {tower['height_m']:.0f} m at {tower['xy']} ({tower['distance_to_source_110_m']:.1f} m from source 110) "
          f"blocked -> {'PASS' if checks['tower_blocked'] else 'FAIL'}")
    print(f"stl_tools agreement {agree['agreement_pct']:.2f} %, ours-blocked/ref-free {agree['ours_blocked_ref_free_pct']:.2f} %, "
          f"ref-blocked/ours-free {agree['ref_blocked_ours_free_pct']:.2f} % -> {'PASS' if checks['ours_is_superset_of_ref'] else 'FAIL'}; "
          f"margin 0 agreement {agree['margin0_agreement_pct']:.2f} % ({agree['margin0_n_diff']} points, all at roof == z: "
          f"{agree['margin0_diff_all_at_roof_eq_z']}) -> {'PASS' if checks['margin0_matches_ref_except_roof_eq_z'] else 'FAIL'}")
    print(f"slab grid no-fly at {z} m: {100 * res['slab_grid_15m']['not_free_frac']:.2f} % of cells not free "
          f"({100 * res['slab_grid_15m']['no_fly_building_frac_in_domain']:.2f} % of in-domain cells)")
    print(f"timing ({m} pts): is_free {1e3 * res['timing']['is_free_median_s']:.2f} ms, "
          f"ray_distances {1e3 * res['timing']['ray_distances_median_s']:.1f} ms")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("VALIDATE_DRONE", "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()