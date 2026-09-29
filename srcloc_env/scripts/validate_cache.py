"""T0-2: verify the LDM cache (frames/*.npz + meta_*.json) against the raw files and the mask definitions.

Checks (plan S0 T0-2):
  1. every meta: n_airborne == n_total - n_deposited - n_outflow + n_dep_and_out (intersection accounting) and
     sum(counts_by_source_airborne) == n_airborne == particles stored in frame_XXX.npz
  2. cached particles: none with x >= X_OUTFLOW; count of z == DEPOSIT_Z (only ground-touching MOVING particles may remain)
  3. spot frames: reload the raw VTK, recompute the airborne mask and require byte-identical xyz / p_type arrays
Usage: python -m srcloc_env.scripts.validate_cache [--spot 0 100 300 450 599] -> cache/validate_cache.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.io.ldm_reader import load_frame

EXPECTED_DEPOSITED_599 = 296_398   # report 2.4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=config.CACHE_DIR)
    ap.add_argument("--spot", type=int, nargs="*", default=[0, 100, 300, 450, 599])
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_cache.json")
    args = ap.parse_args()
    t0 = time.perf_counter()

    meta_ok = npz_ok = 0
    bad: list[dict] = []
    n_x_outflow_cached = 0
    n_z_deposit_cached = 0
    series = []
    for index in range(config.N_FILES):
        meta = json.loads((args.cache / "frames" / f"meta_{index:03d}.json").read_text(encoding="utf-8"))
        expect_air = meta["n_total"] - meta["n_deposited"] - meta["n_outflow"] + meta["n_dep_and_out"]
        by_src = sum(meta["counts_by_source_airborne"].values())
        with np.load(args.cache / "frames" / f"frame_{index:03d}.npz") as z:
            xyz, pt = z["xyz"], z["p_type"]
        n_x_outflow_cached += int((xyz[:, 0] >= config.X_OUTFLOW).sum())
        n_z_deposit_cached += int((xyz[:, 2] == config.DEPOSIT_Z).sum())
        ok_meta = meta["n_airborne"] == expect_air == by_src
        ok_npz = xyz.shape[0] == meta["n_airborne"] == pt.shape[0] and pt.min() >= 1 and pt.max() <= 13
        meta_ok += int(ok_meta)
        npz_ok += int(ok_npz)
        if not (ok_meta and ok_npz):
            bad.append({"index": index, "meta": meta["n_airborne"], "expect": expect_air, "by_src": by_src, "npz": int(xyz.shape[0])})
        series.append((index, meta["n_total"], meta["n_deposited"], meta["n_outflow"], meta["n_airborne"]))

    spot = {}
    for index in args.spot:
        f = load_frame(index)
        air = f.airborne
        with np.load(args.cache / "frames" / f"frame_{index:03d}.npz") as z:
            xyz, pt = z["xyz"], z["p_type"]
        spot[str(index)] = {
            "step": f.step,
            "xyz_identical": bool(np.array_equal(xyz, f.xyz[air])),
            "p_type_identical": bool(np.array_equal(pt.astype(np.int32) + 100, f.p_type[air])),
            "n_airborne": int(air.sum()), "n_deposited_raw": int(f.deposited.sum()),
            "n_outflow_raw": int(f.outflow.sum()),
            "deposited_in_cache": int((f.deposited[air]).sum()),   # must be 0 by construction
            "outflow_in_cache": int((f.outflow[air]).sum()),
        }

    s = np.array(series)
    result = {
        "n_files": config.N_FILES, "meta_consistent": meta_ok, "npz_consistent": npz_ok, "inconsistent": bad,
        "cached_x_ge_outflow_total": n_x_outflow_cached,
        "cached_z_eq_deposit_total": n_z_deposit_cached,
        "airborne_monotone": bool(np.all(np.diff(s[:, 4]) >= 0)),
        "totals_last": {"n_total": int(s[-1, 1]), "n_deposited": int(s[-1, 2]), "n_outflow": int(s[-1, 3]), "n_airborne": int(s[-1, 4])},
        "deposited_599_matches_report": int(s[-1, 2]) == EXPECTED_DEPOSITED_599,
        "spot_frames": spot,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    result["pass"] = (meta_ok == npz_ok == config.N_FILES and n_x_outflow_cached == 0 and result["deposited_599_matches_report"]
                      and all(v["xyz_identical"] and v["p_type_identical"] and v["deposited_in_cache"] == 0 and v["outflow_in_cache"] == 0
                              for v in spot.values()))
    text = json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print("VALIDATE_CACHE", "PASS" if result["pass"] else "FAIL")


if __name__ == "__main__":
    main()