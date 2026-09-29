"""D3: sensor model numbers on the real cached slabs (plan 4.1; R3 Poisson sensor, R4 Currie threshold).

Usage: python -m srcloc_env.scripts.validate_detector [--index 599] [--z 15] [--seed 0]
Writes config.CACHE_DIR / validate_detector.json and prints PASS/FAIL.

Reported per source (frame index 599 = step 30000, z = DRONE_Z, scale = 1 and the scale range ends):
  * expected counts at the source's own (x, y) (config.SOURCES_XY), and at +50 m / +200 m downwind (+x,
    config.VALIDATE_DETECTOR_DOWNWIND_M) - the density there is the bilinear slab value of that source only;
  * "detectable area" = fraction of the source's occupied slab cells (density > 0) whose expected rate at
    scale 1 exceeds the Currie threshold (config.SENSOR_CURRIE_K), plus the absolute cell count / area;
  * the same fraction at the low and high ends of config.SENSOR_SCALE_RANGE (informational).
Pass criterion: the empirical mean of config.VALIDATE_DETECTOR_N_POISSON Poisson draws at three rates
(background only, source-110 own-position rate, y_max) is within config.VALIDATE_DETECTOR_MEAN_TOL (1%)
of lambda, and the measure_per_source total draw matches the summed rate the same way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.sensor.detector import Detector


def poisson_mean_check(det: Detector, density: float, scale: float, rng: np.random.Generator, n: int) -> dict:
    lam = float(det.expected_counts(density, scale))
    y = det.measure(np.full(n, density), scale, rng)
    rel = abs(float(y.mean()) - lam) / lam
    return {"lambda": lam, "empirical_mean": float(y.mean()), "empirical_var": float(y.var()),
            "rel_err_mean": rel, "n": int(n), "pass": bool(rel < config.VALIDATE_DETECTOR_MEAN_TOL)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=config.N_FILES - 1)
    ap.add_argument("--z", type=float, default=config.DRONE_Z)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-poisson", type=int, default=config.VALIDATE_DETECTOR_N_POISSON)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_detector.json")
    args = ap.parse_args()

    det = Detector()
    be = LdmSlabBackend(config.CACHE_DIR, max_cached_frames=2)
    idx, z = args.index, args.z
    sf = be.slab(idx)
    zi = sf.z_index(z)
    grid = sf.grid
    thr = det.detection_threshold_cps()
    lo, hi = config.SENSOR_SCALE_RANGE
    offsets = [0.0] + [float(v) for v in config.VALIDATE_DETECTOR_DOWNWIND_M]
    cell_area = grid.res ** 2

    per_source: dict[int, dict] = {}
    for s in config.ALL_SOURCES:
        xs, ys = config.SOURCES_XY[s]
        pts = np.array([[xs + dx, ys] for dx in offsets])
        dens = be.density(s, pts, idx, z)                                   # own source only, (3,)
        dens_all = be.density(config.ALL_SOURCES, pts, idx, z)              # all 13 sources summed
        counts = {f"dx_{int(dx)}m": {"xy": [float(xs + dx), float(ys)], "density_own": float(dens[i]),
                                     "density_all_sources": float(dens_all[i]),
                                     "expected_counts_scale1": float(det.expected_counts(dens[i], 1.0)),
                                     "expected_counts_scale_lo": float(det.expected_counts(dens[i], lo)),
                                     "expected_counts_scale_hi": float(det.expected_counts(dens[i], hi)),
                                     "detectable_scale1": bool(det.expected_rate(dens[i], 1.0) > thr)}
                  for i, dx in enumerate(offsets)}
        slab = np.asarray(sf.density[sf.source_indices(s)[0], zi], dtype=np.float64)   # (ny, nx)
        occ = slab > 0.0
        n_occ = int(occ.sum())
        rate1 = det.expected_rate(slab, 1.0)
        det1 = occ & (rate1 > thr)
        frac = {sc_name: float((occ & (det.expected_rate(slab, sc) > thr)).sum() / max(1, n_occ))
                for sc_name, sc in (("scale1", 1.0), ("scale_lo", lo), ("scale_hi", hi))}
        per_source[int(s)] = {
            "source_xy": [float(xs), float(ys)], **counts,
            "occupied_cells": n_occ, "occupied_area_m2": n_occ * cell_area,
            "detectable_cells_scale1": int(det1.sum()), "detectable_area_m2_scale1": float(det1.sum() * cell_area),
            "detectable_fraction_scale1": frac["scale1"], "detectable_fraction_scale_lo": frac["scale_lo"],
            "detectable_fraction_scale_hi": frac["scale_hi"],
            "slab_max_density": float(slab.max()),
            "slab_max_expected_counts_scale1": float(det.expected_counts(slab.max(), 1.0)),
            "min_detectable_density_scale1": float((thr - det.background) / det.k0),
        }

    # ---- empirical Poisson mean check (1e5 samples) ---------------------------------------------
    rng = np.random.default_rng(args.seed)
    d110 = per_source[110]["dx_0m"]["density_own"]
    checks = {
        "background_only": poisson_mean_check(det, 0.0, 1.0, rng, args.n_poisson),
        "source_110_own_position_scale1": poisson_mean_check(det, d110, 1.0, rng, args.n_poisson),
        "y_max_rate": poisson_mean_check(det, config.SENSOR_REF_DENSITY, hi, rng, args.n_poisson),
    }
    # measure_per_source: one draw from the summed 13-source rate at the source-110 position
    p110 = np.array(config.SOURCES_XY[110])
    d_vec = np.array([be.density(s, p110, idx, z)[0] for s in config.ALL_SOURCES])       # (13,)
    n = args.n_poisson
    counts, truth = det.measure_per_source(np.tile(d_vec, (n, 1)), 1.0, rng)
    lam_tot = float(truth[0].sum() + det.background * det.T)
    rel = abs(float(counts.mean()) - lam_tot) / lam_tot
    checks["per_source_total_draw_at_110"] = {
        "lambda_total": lam_tot, "empirical_mean": float(counts.mean()), "rel_err_mean": rel,
        "truth_vector_expected_counts": [float(v) for v in truth[0]],
        "truth_sum_equals_total_minus_background": bool(abs(truth[0].sum() - (lam_tot - det.background * det.T)) < 1e-9),
        "n": int(n), "pass": bool(rel < config.VALIDATE_DETECTOR_MEAN_TOL)}
    mean_pass = all(c["pass"] for c in checks.values())

    res = {
        "index": idx, "z": z, "grid": grid.to_dict(),
        "sensor": {"k0": det.k0, "background_cps": det.background, "T": det.T, "scale_range": [lo, hi],
                   "y_max": det.y_max, "currie_k": det.currie_k, "threshold_cps": float(thr),
                   "min_detectable_density_scale1": float((thr - det.background) / det.k0),
                   "ref_density": config.SENSOR_REF_DENSITY},
        "per_source": per_source,
        "detectable_fraction_scale1_mean": float(np.mean([v["detectable_fraction_scale1"] for v in per_source.values()])),
        "poisson_checks": checks,
        "pass": bool(mean_pass),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"Currie threshold {thr:.2f} cps; y_max {det.y_max:.1f}; "
          f"mean detectable fraction (scale 1) {res['detectable_fraction_scale1_mean']:.3f}")
    for k, c in checks.items():
        print(f"  {k}: lambda {c.get('lambda', c.get('lambda_total')):.2f} mean {c['empirical_mean']:.2f} "
              f"rel {c['rel_err_mean']:.2e} ->", "PASS" if c["pass"] else "FAIL")
    print("VALIDATE_DETECTOR", "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()