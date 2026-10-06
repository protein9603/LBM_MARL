"""D4-2 validation of GaussianPlume(wind_mode='local') on the real SPH wind field and slab cache (plan S1 보강 / 4.2).

Usage: python -m srcloc_env.scripts.validate_forward_local [--index 599] [--n-calls 100]
Writes config.CACHE_DIR / validate_forward_local.json.  Reported only (calibration is D4-3 / T1-3):
  * for the sources config.FWD_VALIDATE_SOURCES (109 open, 110 trapped; report 2.6): the local SPH wind
    (u, v) at the source position and z = config.FWD_LOCAL_WIND_Z (= 15 m drone slab), its speed, the clipped
    U_i = max(|uv|, FWD_U_MIN), the direction, and for reference the report-2.6 near-source box mean speed
    (config.WIND_REF_SOURCE_SPEED) and the global (U, dir) of ForwardParams;
  * the +x centreline comparison rho10 = log10(n_LDM / (g * q_A)) at d in config.FWD_LOCAL_VALIDATE_D_M for
    wind_mode 'local' vs 'global' (same ForwardParams), q_A = config.Q_RELEASE_A (interpretation A);
  * extra: the same comparison along the LOCAL downwind direction e_i (the line the local plume actually
    follows), so the two modes can be judged where each puts its mass;
  * timing of the local mode for N = config.FWD_TIMING_N_HYPOTHESES x M in config.FWD_TIMING_N_DRONES
    (informational against config.FWD_TIMING_TARGET_S; the extra cost is one bilinear lookup of N points).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.field.wind import WindField
from srcloc_env.pf.forward_model import ForwardParams, GaussianPlume
from srcloc_env.scripts.validate_forward import time_calls, _finite_or_none


def _centreline(plume: GaussianPlume, be: LdmSlabBackend, s: int, xs: float, ys: float, e: np.ndarray,
                d_list: np.ndarray, index: int, q_a: float) -> dict:
    """rho10 = log10(n_LDM / (g q_A)) at the points source + d * e (z = DRONE_Z) for one plume / one source."""
    pts = np.column_stack([xs + d_list * e[0], ys + d_list * e[1], np.full_like(d_list, config.DRONE_Z)])
    g = plume.unit_response(np.array([[xs, ys]]), pts)[0]
    n_ldm = be.density([s], pts[:, :2], index, config.DRONE_Z)
    with np.errstate(divide="ignore"):
        rho10 = np.log10(n_ldm / (g * q_a))
    finite = np.isfinite(rho10)
    return {
        "points_xy": pts[:, :2].tolist(),
        "g_unit": g.tolist(),
        "g_times_qA": (g * q_a).tolist(),
        "n_ldm_slab": n_ldm.tolist(),
        "log10_ratio_slab_over_g_qA": [_finite_or_none(v) for v in rho10],
        "log10_ratio_median_finite": _finite_or_none(np.median(rho10[finite])) if finite.any() else None,
        "n_finite": int(finite.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=config.FWD_VALIDATE_FRAME_INDEX)
    ap.add_argument("--n-calls", type=int, default=config.FWD_TIMING_N_CALLS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_forward_local.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    wf = WindField.load()
    wind_load_s = time.perf_counter() - t0
    params = ForwardParams()
    glob = GaussianPlume(params)
    loc = GaussianPlume(params, wind_field=wf, wind_mode="local")

    be = LdmSlabBackend(config.CACHE_DIR, max_cached_frames=config.FIELD_MAX_CACHED_FRAMES)
    be.slab(args.index)
    q_a = config.Q_RELEASE_A
    d_list = np.asarray(config.FWD_LOCAL_VALIDATE_D_M, dtype=np.float64)
    e_x = np.array([1.0, 0.0])

    sources: dict[str, dict] = {}
    for s in config.FWD_VALIDATE_SOURCES:
        xs, ys = config.SOURCES_XY[s]
        lw = loc.local_wind(np.array([[xs, ys]]))
        e_loc = lw.e[0]
        sources[str(s)] = {
            "source_xy": [xs, ys],
            "local_wind": {
                "z_m": loc.wind_z,
                "uv_m_per_s": lw.uv[0].tolist(),
                "speed_m_per_s": float(lw.speed[0]),
                "U_used_m_per_s": float(lw.U[0]),
                "clipped_to_U_MIN": bool(lw.speed[0] < config.FWD_U_MIN),
                "direction_deg_ccw_from_x": float(lw.dir_deg[0]),
                "direction_fallback_to_global": bool(lw.fallback[0]),
                "report_2_6_box_mean_speed_m_per_s": wf.box_mean_speed((xs, ys)),
                "report_2_6_reference_m_per_s": config.WIND_REF_SOURCE_SPEED.get(s),
            },
            "global_wind": {"U_m_per_s": params.U, "direction_deg": params.wind_dir_deg},
            "d_m": d_list.tolist(),
            "centreline_plus_x": {
                "local": _centreline(loc, be, s, xs, ys, e_x, d_list, args.index, q_a),
                "global": _centreline(glob, be, s, xs, ys, e_x, d_list, args.index, q_a),
            },
            "centreline_local_downwind": {
                "unit_vector": e_loc.tolist(),
                "local": _centreline(loc, be, s, xs, ys, e_loc, d_list, args.index, q_a),
                "global": _centreline(glob, be, s, xs, ys, e_loc, d_list, args.index, q_a),
            },
        }

    # ---- timing of the local mode ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    n_hyp = config.FWD_TIMING_N_HYPOTHESES
    src = np.column_stack([rng.uniform(*config.PF_PRIOR_X, n_hyp), rng.uniform(*config.PF_PRIOR_Y, n_hyp)])
    timing: dict[str, dict] = {}
    for m in config.FWD_TIMING_N_DRONES:
        drn = np.column_stack([rng.uniform(*config.PF_PRIOR_X, m), rng.uniform(*config.PF_PRIOR_Y, m),
                               np.full(m, config.DRONE_Z)])
        loc.unit_response(src, drn)
        timing[f"local_unit_response_N{n_hyp}_M{m}"] = time_calls(lambda: loc.unit_response(src, drn), args.n_calls)
        timing[f"global_unit_response_N{n_hyp}_M{m}"] = time_calls(lambda: glob.unit_response(src, drn), args.n_calls)
    timing[f"local_wind_lookup_N{n_hyp}"] = time_calls(lambda: loc.local_wind(src), args.n_calls)
    lw_all = loc.local_wind(src)
    hyp_stats = {
        "n": int(n_hyp),
        "frac_speed_below_U_MIN": float(np.mean(lw_all.speed < config.FWD_U_MIN)),
        "frac_direction_fallback": float(np.mean(lw_all.fallback)),
        "speed_percentiles_10_50_90": np.percentile(lw_all.speed, [10, 50, 90]).tolist(),
    }
    key_2 = f"local_unit_response_N{n_hyp}_M{config.FWD_TIMING_N_DRONES[0]}"

    res = {
        "index": args.index, "z": config.DRONE_Z, "wind_load_seconds": wind_load_s,
        "params": {k: float(v) for k, v in params.__dict__.items()},
        "U_MIN": config.FWD_U_MIN, "q_A_particles_per_s": q_a,
        "sources": sources,
        "uniform_prior_hypotheses": hyp_stats,
        "timing": timing, "timing_key": key_2, "timing_criterion_s": config.FWD_TIMING_TARGET_S,
        "timing_local_within_target": bool(timing[key_2]["median_s"] < config.FWD_TIMING_TARGET_S),
        "note": "numbers only; (U, sigma_v) x wind_mode calibration and kappa offset are D4-3 / T1-3 (plan 4.2).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()