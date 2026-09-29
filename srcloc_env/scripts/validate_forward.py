"""D3 validation of pf/forward_model.py on the real slab cache (plan 4.2 / D3).

Usage: python -m srcloc_env.scripts.validate_forward [--index 599] [--n-calls 100]
Writes config.CACHE_DIR / validate_forward.json and prints the timing PASS/FAIL:
  * timing: GaussianPlume.unit_response for N = config.FWD_TIMING_N_HYPOTHESES (2000) hypotheses x M drones,
    M in config.FWD_TIMING_N_DRONES (2, 1), median of config.FWD_TIMING_N_CALLS calls; PASS if the (2000 x 2)
    median < config.FWD_TIMING_TARGET_S (1 ms, plan D3).  log_unit_response and LibraryModel are timed too.
  * analytic vs library: for sources config.FWD_VALIDATE_SOURCES (109 open, 110 trapped; report 2.6) along
    the +x centreline at z = config.DRONE_Z, the slab density n_LDM and the analytic g (U = FWD_DEFAULT_U,
    sigma_v = FWD_DEFAULT_SIGMA_V) are compared as rho10 = log10(n_LDM / (g * q_A)) at d in FWD_VALIDATE_D_M,
    q_A = config.Q_RELEASE_A = 266.6 particles/s (interpretation A).  Reported only: the (U, sigma_v)
    calibration and kappa_ref offset are T1-3 (plan 4.2), not part of this gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.pf.forward_model import ForwardParams, GaussianPlume, LibraryModel


def time_calls(fn, n: int) -> dict:
    """Median / mean / p99 wall time [s] of n repeated calls (time.perf_counter around each call)."""
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        t[i] = time.perf_counter() - t0
    return {"n": int(n), "median_s": float(np.median(t)), "mean_s": float(t.mean()),
            "p99_s": float(np.percentile(t, 99)), "max_s": float(t.max())}


def _finite_or_none(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=config.FWD_VALIDATE_FRAME_INDEX)
    ap.add_argument("--n-calls", type=int, default=config.FWD_TIMING_N_CALLS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_forward.json")
    args = ap.parse_args()

    params = ForwardParams()
    plume = GaussianPlume(params)
    rng = np.random.default_rng(args.seed)
    n_hyp = config.FWD_TIMING_N_HYPOTHESES
    src = np.column_stack([rng.uniform(*config.PF_PRIOR_X, n_hyp), rng.uniform(*config.PF_PRIOR_Y, n_hyp)])

    # ---- timing (analytic, log-analytic, library) ------------------------------------------------
    timing: dict[str, dict] = {}
    for m in config.FWD_TIMING_N_DRONES:
        drn = np.column_stack([rng.uniform(*config.PF_PRIOR_X, m), rng.uniform(*config.PF_PRIOR_Y, m),
                               np.full(m, config.DRONE_Z)])
        plume.unit_response(src, drn)                                            # warm up
        timing[f"unit_response_N{n_hyp}_M{m}"] = time_calls(lambda: plume.unit_response(src, drn), args.n_calls)
        timing[f"log_unit_response_N{n_hyp}_M{m}"] = time_calls(lambda: plume.log_unit_response(src, drn), args.n_calls)
    key_2 = f"unit_response_N{n_hyp}_M{config.FWD_TIMING_N_DRONES[0]}"
    timing_pass = timing[key_2]["median_s"] < config.FWD_TIMING_TARGET_S

    be = LdmSlabBackend(config.CACHE_DIR, max_cached_frames=config.FIELD_MAX_CACHED_FRAMES)
    t0 = time.perf_counter()
    be.slab(args.index)
    load_s = time.perf_counter() - t0
    lib = LibraryModel(be, args.index, config.DRONE_Z)
    for m in config.FWD_TIMING_N_DRONES:
        drn = np.column_stack([rng.uniform(*config.PF_PRIOR_X, m), rng.uniform(*config.PF_PRIOR_Y, m),
                               np.full(m, config.DRONE_Z)])
        lib.unit_response(drn)
        timing[f"library_unit_response_13_M{m}"] = time_calls(lambda: lib.unit_response(drn), args.n_calls)

    # ---- analytic vs slab along the +x centreline --------------------------------------------------
    q_a = config.Q_RELEASE_A
    d_list = np.asarray(config.FWD_VALIDATE_D_M, dtype=np.float64)
    comparison: dict[str, dict] = {}
    for s in config.FWD_VALIDATE_SOURCES:
        xs, ys = config.SOURCES_XY[s]
        drn = np.column_stack([xs + d_list, np.full_like(d_list, ys), np.full_like(d_list, config.DRONE_Z)])
        g = plume.unit_response(np.array([[xs, ys]]), drn)[0]
        n_ldm = be.density([s], drn[:, :2], args.index, config.DRONE_Z)
        with np.errstate(divide="ignore"):
            rho10 = np.log10(n_ldm / (g * q_a))
        finite = np.isfinite(rho10)
        comparison[str(s)] = {
            "source_xy": [xs, ys],
            "d_m": d_list.tolist(),
            "sigma_y_m": plume.sigma_y(d_list).tolist(),
            "g_unit": g.tolist(),
            "g_times_qA": (g * q_a).tolist(),
            "n_ldm_slab": n_ldm.tolist(),
            "log10_ratio_slab_over_g_qA": [_finite_or_none(v) for v in rho10],
            "log10_ratio_median_finite": _finite_or_none(np.median(rho10[finite])) if finite.any() else None,
            "n_finite": int(finite.sum()),
            "implied_q_particles_per_s_median": _finite_or_none(q_a * 10 ** np.median(rho10[finite])) if finite.any() else None,
        }

    res = {
        "index": args.index, "z": config.DRONE_Z, "slab_load_seconds": load_s,
        "params": {k: float(v) for k, v in params.__dict__.items()},
        "q_A_particles_per_s": q_a,
        "timing": timing,
        "timing_criterion_s": config.FWD_TIMING_TARGET_S, "timing_key": key_2, "timing_pass": bool(timing_pass),
        "comparison_centreline": comparison,
        "note": "log10 ratio is reported only; (U, sigma_v) calibration and kappa_ref offset are T1-3 (plan 4.2).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"D3 timing {key_2} median {timing[key_2]['median_s'] * 1e3:.4f} ms "
          f"(criterion < {config.FWD_TIMING_TARGET_S * 1e3:.2f} ms):", "PASS" if timing_pass else "FAIL")


if __name__ == "__main__":
    main()