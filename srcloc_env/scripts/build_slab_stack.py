"""Build (and verify) the memory-mapped 15 m slab stack used by the time-varying truth mode T2 (D11).

Usage: python -m srcloc_env.scripts.build_slab_stack [--verify N] [--rebuild | --no-build]
The stack is built only when it does not exist (or with --rebuild; this fails on Windows while a training / evaluation process has it mapped);
--no-build never builds.
Reads cache/slabs/slab_000..599.npz (about 35 ms each), writes cache/slab_stack_z15.npy (float16, about 0.65 GB) and its JSON meta file; --verify
compares StackedSlabBackend with LdmSlabBackend on N random frames (bit-identical density queries at random points, all 13 sources).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from srcloc_env import config
from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.field.slab_stack import StackedSlabBackend, build_slab_stack


def verify(n_frames: int) -> int:
    a, b = LdmSlabBackend(), StackedSlabBackend()
    rng = np.random.default_rng(0)
    frames = sorted(int(f) for f in rng.choice(config.N_FILES, size=n_frames, replace=False)) + [0, config.N_FILES - 1]
    bad = 0
    for f in frames:
        pts = np.column_stack([rng.uniform(330.0, 1330.0, 400), rng.uniform(-487.5, 547.5, 400)])
        for src in config.ALL_SOURCES:
            bad += int(not np.array_equal(a.density([src], pts, f, config.DRONE_Z), b.density([src], pts, f, config.DRONE_Z)))
        bad += int(not np.array_equal(a.density([101], pts[0], f, config.DRONE_Z), b.density([101], pts[0], f, config.DRONE_Z)))
    print(f"[slab_stack] verified {len(frames)} frames x {len(config.ALL_SOURCES)} sources: {bad} mismatching queries")
    return bad


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", type=int, default=25, help="random frames to compare against LdmSlabBackend (0 = skip)")
    ap.add_argument("--no-build", action="store_true", help="only verify an existing stack")
    ap.add_argument("--rebuild", action="store_true", help="rebuild even if the stack exists")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    exists = config.SLAB_STACK_PATH.exists() and config.SLAB_STACK_META_PATH.exists()
    meta = None if (args.no_build or (exists and not args.rebuild)) else build_slab_stack(verbose=True)
    if meta is not None:
        print(f"[slab_stack] built {meta['shape']} {meta['dtype']} in {time.perf_counter() - t0:.0f} s -> {config.SLAB_STACK_PATH}")
    bad = verify(args.verify) if args.verify > 0 else 0
    if bad:
        raise SystemExit(1)
    return {"meta": meta, "mismatches": bad}


if __name__ == "__main__":
    main()
