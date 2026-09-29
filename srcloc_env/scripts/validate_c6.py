"""T0-1: reproduce the files' concn with our Wendland C6 gather (NO masks: all same-p_type particles).

Usage: python -m srcloc_env.scripts.validate_c6 [--n 2000] [--seed 0] [--index 599] [--source vtk|npz]
Pass criterion (plan S0 T0-1): median relative error < 1e-5 against concn stored in the file.
The file's concn includes deposited and outflow particles, so this test must NOT apply the airborne mask;
the mask is only applied when building slabs/sensor readings (T0-2/T0-3).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from srcloc_env import config
from srcloc_env.io.ldm_reader import load_frame, load_frame_from_npz
from srcloc_env.sensor.kernel import c6_gather


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", type=int, default=config.N_FILES - 1)
    ap.add_argument("--source", choices=("npz", "vtk"), default="npz")
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "t0_1_c6.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    frame = load_frame_from_npz() if (args.source == "npz" and args.index == config.N_FILES - 1) else load_frame(args.index)
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(frame.n, size=min(args.n, frame.n), replace=False)

    recon = np.zeros(sample.size)
    for s in np.unique(frame.p_type[sample]):
        same = frame.p_type == s                       # ALL particles of this source, no masks
        tree = cKDTree(frame.xyz[same].astype(np.float64))
        sel = frame.p_type[sample] == s
        recon[sel] = c6_gather(frame.xyz[sample[sel]], frame.xyz[same], tree)

    stored = frame.concn[sample].astype(np.float64)
    rel = np.abs(recon - stored) / np.maximum(stored, 1e-12)
    absd = np.abs(recon - stored)
    res = {
        "index": frame.index, "step": frame.step, "n_sample": int(sample.size), "seed": args.seed,
        "median_rel_err": float(np.median(rel)), "p99_rel_err": float(np.percentile(rel, 99)),
        "max_rel_err": float(rel.max()), "max_abs_err": float(absd.max()),
        "frac_rel_err_lt_1e-5": float(np.mean(rel < 1e-5)), "frac_rel_err_lt_1e-3": float(np.mean(rel < 1e-3)),
        "n_deposited_in_sample": int(frame.deposited[sample].sum()),
        "n_outflow_in_sample": int(frame.outflow[sample].sum()),
        "seconds": round(time.perf_counter() - t0, 2),
        "pass": bool(np.median(rel) < 1e-5),
    }
    worst = np.argsort(rel)[-5:][::-1]
    res["worst_cases"] = [{"idx": int(sample[i]), "stored": float(stored[i]), "recon": float(recon[i]),
                           "rel": float(rel[i]), "p_type": int(frame.p_type[sample[i]]),
                           "xyz": frame.xyz[sample[i]].tolist(), "|v|": float(np.linalg.norm(frame.velocity[sample[i]]))}
                          for i in worst]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print("VALIDATE_C6", "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()