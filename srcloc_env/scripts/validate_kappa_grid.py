"""T1-2b preparation: kappa log-grid range check + RB-PF update timing (plan 4.3 / T1-2b, D3).

Usage: python -m srcloc_env.scripts.validate_kappa_grid [--kappa-ref 7011.58] [--grid-decades 2.5] [--n-updates 1000]
Writes config.CACHE_DIR / validate_kappa_grid.json and prints 'T1-2b ... PASS/FAIL' and 'D3 PF timing ... PASS/FAIL'.

Range check (plan T1-2b (ii), nominal ranges; the measured per-source range (i) needs the T1-3 offsets):
    kappa_true = SENSOR_K0 * scale * q,  scale in SENSOR_SCALE_RANGE,
    q_A = PARTICLES_PER_INDEX_STEP_PER_SOURCE / SEC_PER_INDEX_STEP  (interpretation A, 266.6 particles/s)
    q_B = PARTICLES_PER_INDEX_STEP_PER_SOURCE / DT_LDM_SECONDS      (interpretation B, 26.66 particles/s)
    grid = kappa_ref * 10^(+-grid_decades); margin_low = log10(range_lo / grid_lo), margin_high = log10(grid_hi / range_hi)
PASS if both margins of the A-B union are >= config.T1_2B_MIN_MARGIN_DECADES (1 decade); otherwise the script
recommends (a) the grid_decades needed with the current kappa_ref and (b) kappa_ref centred on the union
(geometric mean) with the grid_decades then needed, both rounded up to config.T1_2B_GRID_ROUND_DECADES.
The trapped-source mismatch (config.T1_2B_MISMATCH_DECADES, report 2.6) is reported as an extra margin
requirement for information (plan 4.3 "격자 폭의 근거").

Timing: RBPF.update in grid mode with GaussianPlume, N = PF_N_PARTICLES, G = KAPPA_G, config.PF_TIMING_N_UPDATES
updates at random drone positions with Poisson(background T) counts; resampling is included when it triggers.
PASS if the median float64 update < config.PF_UPDATE_TIME_TARGET_S; the float32 grid block (RBPF(dtype=np.float32))
is timed as well.  The forward-model share is timed separately.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.pf.forward_model import ForwardParams, GaussianPlume
from srcloc_env.pf.particle_filter import RBPF


def _round_up(x: float, step: float) -> float:
    return float(np.ceil(x / step - 1e-12) * step)


def nominal_kappa_ranges() -> dict[str, tuple[float, float]]:
    """Nominal true-kappa ranges [lo, hi] for interpretations A and B and their union (plan T1-2b (ii))."""
    lo_s, hi_s = config.SENSOR_SCALE_RANGE
    out = {}
    for name, q in (("A", config.Q_RELEASE_A), ("B", config.Q_RELEASE_B)):
        out[name] = (config.SENSOR_K0 * lo_s * q, config.SENSOR_K0 * hi_s * q)
    out["union"] = (min(r[0] for r in out.values()), max(r[1] for r in out.values()))
    return out


def kappa_grid_margins(kappa_ref: float, grid_decades: float,
                       min_margin: float = config.T1_2B_MIN_MARGIN_DECADES,
                       round_step: float = config.T1_2B_GRID_ROUND_DECADES,
                       mismatch_decades: float = config.T1_2B_MISMATCH_DECADES) -> dict:
    """Margins in decades between the nominal kappa ranges and the grid ends, PASS flag and recommendations."""
    ranges = nominal_kappa_ranges()
    grid_lo, grid_hi = kappa_ref * 10.0 ** -grid_decades, kappa_ref * 10.0 ** grid_decades
    margins = {}
    for name, (lo, hi) in ranges.items():
        margins[name] = {"lo": float(np.log10(lo / grid_lo)), "hi": float(np.log10(grid_hi / hi))}
    u_lo, u_hi = ranges["union"]
    m = margins["union"]
    passed = m["lo"] >= min_margin and m["hi"] >= min_margin
    # (a) keep kappa_ref: half-width needed = max distance of the union ends from log10(kappa_ref) + margin
    need_current = max(abs(np.log10(u_lo / kappa_ref)), abs(np.log10(u_hi / kappa_ref))) + min_margin
    # (b) centre kappa_ref on the union (geometric mean): half-width = half union width + margin
    kappa_centred = float(np.sqrt(u_lo * u_hi))
    need_centred = 0.5 * np.log10(u_hi / u_lo) + min_margin
    return {
        "kappa_ref": float(kappa_ref), "grid_decades": float(grid_decades), "n_grid": int(config.KAPPA_G),
        "grid_lo": float(grid_lo), "grid_hi": float(grid_hi),
        "q_A_particles_per_s": float(config.Q_RELEASE_A), "q_B_particles_per_s": float(config.Q_RELEASE_B),
        "ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
        "margins_decades": margins, "min_margin_decades": float(min_margin), "pass": bool(passed),
        "recommend_grid_decades_current_kappa_ref": _round_up(need_current, round_step),
        "recommend_kappa_ref_centred": kappa_centred,
        "recommend_grid_decades_centred": _round_up(need_centred, round_step),
        "mismatch_decades_info": float(mismatch_decades),
        "recommend_grid_decades_centred_with_mismatch": _round_up(need_centred + mismatch_decades, round_step),
    }


def time_updates(n_updates: int, seed: int = 0, dtype: type = np.float64) -> dict:
    """Median / mean / p99 wall time of RBPF.update (grid mode, N = PF_N_PARTICLES, G = KAPPA_G, given dtype)."""
    rng = np.random.default_rng(seed)
    plume = GaussianPlume(ForwardParams())
    pf = RBPF(plume, rng=np.random.default_rng(seed + 1), dtype=dtype)
    drones = np.column_stack([rng.uniform(*config.PF_PRIOR_X, n_updates), rng.uniform(*config.PF_PRIOR_Y, n_updates),
                              np.full(n_updates, config.DRONE_Z)])
    counts = rng.poisson(config.SENSOR_BACKGROUND_CPS * config.SENSOR_T, n_updates)
    pf.update(int(counts[0]), drones[0])                                       # warm up
    pf.reset()
    t = np.empty(n_updates)
    for i in range(n_updates):
        t0 = time.perf_counter()
        pf.update(int(counts[i]), drones[i])
        t[i] = time.perf_counter() - t0
    tf = np.empty(n_updates)
    for i in range(n_updates):
        t0 = time.perf_counter()
        plume.unit_response(pf.xy, drones[i][None, :])
        tf[i] = time.perf_counter() - t0
    return {"n_updates": int(n_updates), "n_particles": int(pf.N), "n_grid": int(pf.G), "dtype": np.dtype(dtype).name,
            "update_median_s": float(np.median(t)), "update_mean_s": float(t.mean()),
            "update_p99_s": float(np.percentile(t, 99)), "update_max_s": float(t.max()),
            "forward_share_median_s": float(np.median(tf)), "n_resamples": int(pf.n_resamples),
            "target_s": float(config.PF_UPDATE_TIME_TARGET_S),
            "pass": bool(np.median(t) < config.PF_UPDATE_TIME_TARGET_S)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa-ref", type=float, default=config.KAPPA_REF)
    ap.add_argument("--grid-decades", type=float, default=config.KAPPA_GRID_DECADES)
    ap.add_argument("--n-updates", type=int, default=config.PF_TIMING_N_UPDATES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_kappa_grid.json")
    args = ap.parse_args()

    grid = kappa_grid_margins(args.kappa_ref, args.grid_decades)
    timing = time_updates(args.n_updates, args.seed, np.float64)
    timing32 = time_updates(args.n_updates, args.seed, np.float32)
    result = {"t1_2b": grid, "timing": timing, "timing_float32": timing32}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m = grid["margins_decades"]
    print(f"kappa_ref {grid['kappa_ref']:.4g}, grid 10^+-{grid['grid_decades']} -> [{grid['grid_lo']:.4g}, {grid['grid_hi']:.4g}]")
    for k in ("A", "B", "union"):
        lo, hi = grid["ranges"][k]
        print(f"  {k:5s} kappa_true [{lo:.4g}, {hi:.4g}]  margin low {m[k]['lo']:.3f} / high {m[k]['hi']:.3f} decades")
    print(f"T1-2b range check (union margins >= {grid['min_margin_decades']} decade): {'PASS' if grid['pass'] else 'FAIL'}")
    if not grid["pass"]:
        print(f"  recommend: grid_decades {grid['recommend_grid_decades_current_kappa_ref']} with the current kappa_ref, or "
              f"kappa_ref {grid['recommend_kappa_ref_centred']:.4g} (union geometric mean) with grid_decades "
              f"{grid['recommend_grid_decades_centred']} (+ mismatch {grid['mismatch_decades_info']} decade -> "
              f"{grid['recommend_grid_decades_centred_with_mismatch']})")
    print(f"D3 PF timing: N={timing['n_particles']} G={timing['n_grid']} median {timing['update_median_s'] * 1e3:.3f} ms "
          f"(forward share {timing['forward_share_median_s'] * 1e3:.3f} ms, p99 {timing['update_p99_s'] * 1e3:.3f} ms, "
          f"{timing['n_resamples']} resamples) target {timing['target_s'] * 1e3:.1f} ms: {'PASS' if timing['pass'] else 'FAIL'}")
    print(f"   float32 grid block: median {timing32['update_median_s'] * 1e3:.3f} ms, p99 {timing32['update_p99_s'] * 1e3:.3f} ms: "
          f"{'PASS' if timing32['pass'] else 'FAIL'} (option RBPF(dtype=np.float32))")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()