"""D2 wind validation: reproduce the report 3.3 / 2.6 measurements from levels_uvw.npz through WindField.

Usage: python -m srcloc_env.scripts.validate_wind [--out cache/validate_wind.json]
Hard checks (report 3.3, 2.6; the report numbers were produced by
분석스크립트/verify_ldm-timeseries/v11_fluid_ib_and_profile.py and v14_fluid_src_velocity_k.py):
  0. axis order of uvw: the reading whose x = 0 column reproduces the inflow u (3.39, v = 0) must equal
     config.WIND_AXIS_ORDER (see field/wind.py docstring);
  1. whole-level horizontal mean u at z = 1.25 / 11.25 / 21.25 / 48.75 within WIND_TOL_MEAN_U
     (363.75 m is not a stored level of levels_uvw.npz -> reported as N/A with the interpolated value);
  2. inflow column x = 0 (mean over all lattice y) u at z = 1.25 within WIND_TOL_INFLOW_U of 3.39;
  3. plume-region band means (open window 330 < x < 1315, |y| < 500, all nodes, stored levels in [lo, hi))
     for 0-5 / 10-15 / 20-25 m within WIND_TOL_PLUME_U of config.WIND_REF_PLUME_U_BANDS (0.82 / 1.68 / 2.96);
  4. near-source "근방 풍속" of report 2.6 (103: 0.33, 107: 0.63, 109: 3.79, 110: 0.43 m/s) = mean |(u,v,w)| over
     all lattice nodes with |dx|, |dy| < WIND_REF_SOURCE_BOX_HALF_M and z in WIND_REF_SOURCE_Z_RANGE (open), within
     WIND_TOL_SOURCE_SPEED.
Informational: 15 m plume-region means with/without building nodes, the bilinear point wind at all 13 sources at
WIND_REF_SOURCE_Z and at DRONE_Z (the 15 m speed range vs WIND_REF_SOURCE_SPEED_RANGE), query timing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.field.wind import WindField


def axis_order_evidence(wf: WindField) -> dict:
    """Compare both readings of the raw array against the measured inflow column and building bbox."""
    u0 = wf.uvw[0]
    as_ix_iy = {"x0_column_u_mean": float(u0[0, :, 0].mean()), "x0_column_v_absmax": float(np.abs(u0[0, :, 1]).max())}
    as_iy_ix = {"x0_column_u_mean": float(u0[:, 0, 0].mean()), "x0_column_v_absmax": float(np.abs(u0[:, 0, 1]).max())}
    r, c = np.nonzero(wf.building_mask)
    bbox_ix_iy = {"x": [float(wf.x[r.min()]), float(wf.x[r.max()])], "y": [float(wf.y[c.min()]), float(wf.y[c.max()])]}
    bbox_iy_ix = {"x": [float(wf.x[c.min()]), float(wf.x[c.max()])], "y": [float(wf.y[r.min()]), float(wf.y[r.max()])]}
    ref = config.WIND_REF_INFLOW_U[1.25]
    chosen = "level,ix,iy,comp" if abs(as_ix_iy["x0_column_u_mean"] - ref) < abs(as_iy_ix["x0_column_u_mean"] - ref) \
        else "level,iy,ix,comp"
    return {"reading_ix_iy": as_ix_iy, "reading_iy_ix": as_iy_ix, "mask_bbox_ix_iy": bbox_ix_iy,
            "mask_bbox_iy_ix": bbox_iy_ix, "chosen": chosen, "matches_config": chosen == config.WIND_AXIS_ORDER}


def source_table(wf: WindField, z: float) -> dict:
    """Bilinear point wind (u, v, speed) at the 13 source centres at height z, plus the number of dead corner nodes."""
    ids = list(config.ALL_SOURCES)
    xy = np.array([config.SOURCES_XY[s] for s in ids])
    uv = wf.uv_at(xy, z)
    spd = np.hypot(uv[:, 0], uv[:, 1])
    dead = wf.dead_mask(z)
    ix0 = np.clip(np.floor((xy[:, 0] - wf.x[0]) / wf.dx).astype(int), 0, wf.x.size - 2)
    iy0 = np.clip(np.floor((xy[:, 1] - wf.y[0]) / wf.dy).astype(int), 0, wf.y.size - 2)
    out = {}
    for i, s in enumerate(ids):
        n_dead = int(dead[ix0[i]:ix0[i] + 2, iy0[i]:iy0[i] + 2].sum())
        out[str(s)] = {"xy": xy[i].tolist(), "u": float(uv[i, 0]), "v": float(uv[i, 1]), "speed": float(spd[i]),
                       "dead_corners": n_dead}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_wind.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    wf = WindField.load()
    load_s = time.perf_counter() - t0
    res: dict = {"levels_path": str(config.LEVELS_UVW_NPZ), "slices_path": str(config.FLUID_SLICES_NPZ),
                 "uvw_shape": list(wf.uvw.shape), "uvw_dtype": str(wf.uvw.dtype),
                 "z_levels": wf.z_levels.tolist(), "load_seconds": round(load_s, 2),
                 "n_nan": int(np.isnan(wf.uvw).sum()),
                 "n_building_nodes": int(wf.building_mask.sum()),
                 "n_dead_nodes_15m": int(wf.dead_mask(config.DRONE_Z).sum()),
                 "axis_order": axis_order_evidence(wf)}
    checks: dict[str, bool] = {"axis_order": res["axis_order"]["matches_config"], "no_nan": res["n_nan"] == 0}

    # 1. whole-level means
    stored = set(np.round(wf.z_levels, 4).tolist())
    level_means = {}
    for z, ref in config.WIND_REF_MEAN_U.items():
        u, v = wf.mean_profile(z)
        entry = {"ref_u": ref, "u": u, "v": v, "abs_err": abs(u - ref)}
        if round(z, 4) in stored:
            entry["status"] = "PASS" if entry["abs_err"] <= config.WIND_TOL_MEAN_U else "FAIL"
            checks[f"level_mean_u_{z}"] = entry["status"] == "PASS"
        else:
            entry["status"] = "N/A (level not stored; value interpolated between neighbouring levels)"
        level_means[str(z)] = entry
    res["level_mean_u"] = level_means

    # 2. inflow column x = 0 (mean over all lattice y)
    inflow = {}
    xy0 = np.column_stack([np.zeros_like(wf.y), wf.y])
    for z, ref in config.WIND_REF_INFLOW_U.items():
        uv = wf.uv_at(xy0, z)
        entry = {"ref_u": ref, "u_col_mean": float(uv[:, 0].mean()), "v_col_absmax": float(np.abs(uv[:, 1]).max()),
                 "u_at_y0": float(wf.uv_at(np.array([[0.0, 0.0]]), z)[0, 0])}
        entry["status"] = "PASS" if abs(entry["u_col_mean"] - ref) <= config.WIND_TOL_INFLOW_U else "FAIL"
        if z == 1.25:
            checks["inflow_u_1.25"] = entry["status"] == "PASS"
        inflow[str(z)] = entry
    res["inflow_x0"] = inflow

    # 3. plume-region band means (hard) and 15 m region means (informational)
    bands = {}
    for (lo, hi), ref in config.WIND_REF_PLUME_U_BANDS.items():
        u, v = wf.band_mean((lo, hi), exclude_buildings=False)
        u_ex, v_ex = wf.band_mean((lo, hi), exclude_buildings=True)
        entry = {"ref_u": ref, "u_all_nodes": u, "v_all_nodes": v, "u_excl_buildings": u_ex, "v_excl_buildings": v_ex,
                 "levels": wf.z_levels[(wf.z_levels >= lo) & (wf.z_levels < hi)].tolist(), "abs_err": abs(u - ref)}
        entry["status"] = "PASS" if entry["abs_err"] <= config.WIND_TOL_PLUME_U else "FAIL"
        checks[f"plume_band_u_{lo}_{hi}"] = entry["status"] == "PASS"
        bands[f"{lo}-{hi}"] = entry
    res["plume_band_means"] = bands
    region = {}
    for z in (config.DRONE_Z,) + config.SLAB_Z_AUX:
        u_ex, v_ex = wf.region_mean(z, exclude_buildings=True)
        u_all, v_all = wf.region_mean(z, exclude_buildings=False)
        region[str(z)] = {"u_excl_buildings": u_ex, "v_excl_buildings": v_ex, "u_all_nodes": u_all, "v_all_nodes": v_all,
                          "direction_deg_excl": wf.direction_deg(z, exclude_buildings=True)}
    res["plume_region_15m"] = region
    res["mean_profile_15m"] = dict(zip(("u", "v"), wf.mean_profile(config.DRONE_Z)))
    res["direction_deg_15m_all_nodes"] = wf.direction_deg(config.DRONE_Z)

    # 4. report 2.6 near-source box-mean speeds (hard) + point winds at the release level and at 15 m (informational)
    box = {}
    for s in config.ALL_SOURCES:
        box[str(s)] = {"box_mean_speed": wf.box_mean_speed(config.SOURCES_XY[s])}
    near_checks = {}
    for s, ref in config.WIND_REF_SOURCE_SPEED.items():
        got = box[str(s)]["box_mean_speed"]
        near_checks[str(s)] = {"ref_speed": ref, "speed": got, "abs_err": abs(got - ref),
                               "status": "PASS" if abs(got - ref) <= config.WIND_TOL_SOURCE_SPEED else "FAIL"}
        checks[f"source_{s}_box_speed"] = near_checks[str(s)]["status"] == "PASS"
    res["sources_box_mean_speed"] = {"half_width_m": config.WIND_REF_SOURCE_BOX_HALF_M,
                                     "z_range": list(config.WIND_REF_SOURCE_Z_RANGE), "checks": near_checks, "all": box}
    res["sources_point_wind_release_z"] = {"z": config.WIND_REF_SOURCE_Z, "table": source_table(wf, config.WIND_REF_SOURCE_Z)}
    src15 = source_table(wf, config.DRONE_Z)
    spd15 = np.array([e["speed"] for e in src15.values()])
    lo, hi = config.WIND_REF_SOURCE_SPEED_RANGE
    res["sources_15m"] = src15
    res["sources_15m_speed_range"] = [float(spd15.min()), float(spd15.max())]
    res["sources_15m_in_expected_range"] = bool(np.all((spd15 >= lo) & (spd15 <= hi)))   # informational
    res["source_110_speed_15m"] = src15["110"]["speed"]

    # timing of a batched query (env usage)
    rng = np.random.default_rng(0)
    q = np.column_stack([rng.uniform(*config.PF_PRIOR_X, 2000), rng.uniform(*config.PF_PRIOR_Y, 2000)])
    t1 = time.perf_counter()
    wf.uv_at(q, config.DRONE_Z)
    res["query_2000_seconds"] = round(time.perf_counter() - t1, 5)

    res["checks"] = checks
    res["pass"] = bool(all(checks.values()))
    res["seconds"] = round(time.perf_counter() - t0, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print("axis order:", res["axis_order"]["chosen"], "(config:", config.WIND_AXIS_ORDER + ")", "| NaN count:", res["n_nan"])
    for z, e in level_means.items():
        print(f"level mean u z={z}: {e['u']:.4f} (ref {e['ref_u']}) -> {e['status']}")
    for z, e in inflow.items():
        print(f"inflow x=0 z={z}: {e['u_col_mean']:.3f} (ref {e['ref_u']}) -> {e['status']}")
    for b, e in bands.items():
        print(f"plume band {b} m mean u (all nodes, levels {e['levels']}): {e['u_all_nodes']:.4f} (ref {e['ref_u']}) "
              f"-> {e['status']}; excl. buildings {e['u_excl_buildings']:.4f}")
    r15 = region[str(config.DRONE_Z)]
    print(f"15 m plume-region mean u: excl buildings {r15['u_excl_buildings']:.3f}, all nodes {r15['u_all_nodes']:.3f}")
    for s, e in near_checks.items():
        print(f"source {s} box-mean speed (report 2.6): {e['speed']:.3f} (ref {e['ref_speed']}) -> {e['status']}")
    print(f"sources 15 m point speed range {spd15.min():.2f}-{spd15.max():.2f} m/s (expected {lo}-{hi}, informational), "
          f"source 110: {res['source_110_speed_15m']:.2f} m/s")
    print("VALIDATE_WIND", "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()