"""Convert LDM VTK frames to (a) an airborne-particle NPZ cache and (b) per-source density slabs.

Usage (repo root, venv active):
  python -m srcloc_env.scripts.convert_ldm --index 400 599 --slab-z 15 12.5 17.5 --log F:/.../cache/convert.log
Outputs under config.CACHE_DIR:
  frames/frame_{index:03d}.npz   xyz float32 (airborne & x<1315 only), p_type uint8 (= p_type - 100)
  frames/meta_{index:03d}.json   counts (total / deposited / outflow / airborne / per source), timings
  slabs/slab_{index:03d}.npz     density float16 (n_src, n_z, ny, nx) [particles/m^3], z_levels, sources, grid
Frames whose outputs already exist are skipped unless --overwrite is given.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.io.ldm_reader import load_frame
from srcloc_env.preprocess.gridder import SlabGrid, build_slabs, slab_stats


def frame_paths(out: Path, index: int) -> tuple[Path, Path, Path]:
    return (out / "frames" / f"frame_{index:03d}.npz", out / "frames" / f"meta_{index:03d}.json",
            out / "slabs" / f"slab_{index:03d}.npz")


def convert_one(index: int, z_levels: tuple[float, ...], out: Path, overwrite: bool = False) -> dict:
    f_npz, f_meta, f_slab = frame_paths(out, index)
    if not overwrite and f_npz.exists() and f_meta.exists() and f_slab.exists():
        return {"index": index, "skipped": True}
    t0 = time.perf_counter()
    frame = load_frame(index)
    t_read = time.perf_counter() - t0
    air = frame.airborne
    meta = frame.summary()
    meta["n_dep_and_out"] = int((frame.deposited & frame.outflow).sum())
    meta["counts_by_source_airborne"] = frame.counts_by_source(air)
    t1 = time.perf_counter()
    np.savez(f_npz, xyz=frame.xyz[air], p_type=(frame.p_type[air] - config.FRAME_P_TYPE_OFFSET).astype(np.uint8))
    t2 = time.perf_counter()
    grid = SlabGrid()
    slabs = build_slabs(frame, z_levels, grid=grid)
    t3 = time.perf_counter()
    np.savez(f_slab, density=slabs, z_levels=np.asarray(z_levels, dtype=np.float32),
             sources=np.asarray(config.ALL_SOURCES, dtype=np.int16),
             grid=np.asarray([grid.x0, grid.y0, grid.nx, grid.ny, grid.res], dtype=np.float64))
    meta["slab"] = {"z_levels": list(z_levels), "grid": grid.to_dict(), "stats": slab_stats(slabs)}
    meta["timing_s"] = {"read": round(t_read, 2), "cache": round(t2 - t1, 2), "slabs": round(t3 - t2, 2),
                        "total": round(time.perf_counter() - t0, 2)}
    f_meta.write_text(json.dumps(meta), encoding="utf-8")
    return {"index": index, "skipped": False, "n_airborne": meta["n_airborne"], "timing_s": meta["timing_s"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, nargs=2, metavar=("A", "B"), default=(0, config.N_FILES - 1),
                    help="inclusive file-index range (0 = step 15025, 599 = step 30000)")
    ap.add_argument("--slab-z", type=float, nargs="+", default=list(config.SLAB_Z_MAIN))
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    (args.out / "slabs").mkdir(parents=True, exist_ok=True)
    log = open(args.log, "a", encoding="utf-8") if args.log else None

    def emit(msg: str) -> None:
        print(msg, flush=True)
        if log:
            log.write(msg + "\n"); log.flush()

    a, b = args.index
    emit(f"START convert index {a}..{b} slab_z={args.slab_z} out={args.out} python={sys.executable}")
    t_start = time.perf_counter()
    n_done = n_skip = 0
    for index in range(a, b + 1):
        try:
            r = convert_one(index, tuple(args.slab_z), args.out, args.overwrite)
        except Exception as exc:  # keep the batch going; the frame can be redone later
            emit(f"ERROR index {index} step {config.index_to_step(index)}: {exc!r}")
            continue
        if r["skipped"]:
            n_skip += 1
        else:
            n_done += 1
            emit(f"index {index} step {config.index_to_step(index)} airborne {r['n_airborne']} timing {r['timing_s']}")
    emit(f"DONE {n_done} converted, {n_skip} skipped, {b - a + 1} files, {time.perf_counter() - t_start:.0f} s")
    if log:
        log.close()


if __name__ == "__main__":
    main()