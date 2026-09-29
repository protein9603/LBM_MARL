"""Data-dependent validation of io.ldm_reader against the values measured in the analysis report.

Usage (from the repo root, venv active):
  python -m srcloc_env.scripts.validate_reader [--vtk] [--out F:/.../cache/validate_reader.json]

Expected for step 30000 (report 2.2, 2.4; verifier v04/v08): n_total 1,299,844; each of the 13 sources
99,988; z == 1e-4 particles 296,404 of which 6 have non-zero velocity -> deposited 296,398;
x > 1322.0 particles 11,697.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.io.ldm_reader import load_frame, load_frame_from_npz

EXPECTED_30000 = {"n_total": 1_299_844, "per_source": 99_988, "n_z_eq_deposit": 296_404,
                  "n_deposited": 296_398, "n_x_gt_1322": 11_697}


def check(summary: dict) -> dict[str, bool]:
    per_source_ok = all(v == EXPECTED_30000["per_source"] for v in summary["counts_by_source"].values()) \
        and len(summary["counts_by_source"]) == 13
    return {
        "n_total": summary["n_total"] == EXPECTED_30000["n_total"],
        "per_source_99988_x13": per_source_ok,
        "n_z_eq_deposit": summary["n_z_eq_deposit"] == EXPECTED_30000["n_z_eq_deposit"],
        "n_deposited": summary["n_deposited"] == EXPECTED_30000["n_deposited"],
        "n_x_gt_1322": summary["n_x_gt_1322"] == EXPECTED_30000["n_x_gt_1322"],
        "airborne_is_intersection": summary["n_airborne"]
        == summary["n_total"] - summary["n_deposited"] - summary["n_outflow"] + summary.get("n_dep_and_out", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk", action="store_true", help="also read LDM_30000stp.vtk and compare with the NPZ")
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_reader.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    f_npz = load_frame_from_npz()
    t_npz = time.perf_counter() - t0
    s = f_npz.summary()
    s["n_dep_and_out"] = int((f_npz.deposited & f_npz.outflow).sum())
    result = {"npz": {"summary": s, "load_seconds": round(t_npz, 3), "checks": check(s)}}

    if args.vtk:
        t0 = time.perf_counter()
        f_vtk = load_frame(config.N_FILES - 1)
        t_vtk = time.perf_counter() - t0
        same = {
            "xyz": bool(np.array_equal(f_vtk.xyz, f_npz.xyz)),
            "p_type": bool(np.array_equal(f_vtk.p_type, f_npz.p_type)),
            "velocity": bool(np.array_equal(f_vtk.velocity, f_npz.velocity)),
            "concentration": bool(np.array_equal(f_vtk.concentration, f_npz.concentration)),
            "concn": bool(np.array_equal(f_vtk.concn, f_npz.concn)),
        }
        result["vtk"] = {"load_seconds": round(t_vtk, 3), "identical_to_npz": same,
                         "k_turb_range": [float(f_vtk.k_turb.min()), float(f_vtk.k_turb.max())],
                         "e_turb_range": [float(f_vtk.e_turb.min()), float(f_vtk.e_turb.max())]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    all_ok = all(result["npz"]["checks"].values()) and (not args.vtk or all(result["vtk"]["identical_to_npz"].values()))
    print(json.dumps(result, indent=2))
    print("VALIDATE_READER", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()