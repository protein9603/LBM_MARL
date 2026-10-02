"""All frames of the 15 m slab in one memory-mapped array, for time-varying truth episodes (Mode T2; D11).

Mode F reads ONE frame per episode (the LdmSlabBackend LRU pays the disk read once).  Mode T2 needs another frame at every step; reading and
converting a slab_XXX.npz costs about 35 ms, twice the environment step.  The stack holds the z = DRONE_Z level of every cached frame as
float16 (the dtype of the cached slabs, so no information is lost) in a single .npy file; every training / evaluation process opens it with
mmap_mode="r", so the operating system keeps ONE copy in its page cache (about 0.65 GB) and a frame costs a float32 conversion of 2 MB (about 1 ms).

StackedSlabBackend is an LdmSlabBackend whose slab() reads from the stack; density() / sources / grid semantics are inherited, and its values equal the
LdmSlabBackend values bit for bit (tests/test_slab_stack.py, scripts/build_slab_stack.py --verify).  Only the z = DRONE_Z level exists in the stack.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.field.concentration_field import Z_KEY_DECIMALS, LdmSlabBackend, SlabFrame, _bilinear_one, frame_paths
from srcloc_env.preprocess.gridder import SlabGrid


def build_slab_stack(cache_dir: Path = config.CACHE_DIR, out_path: Path = config.SLAB_STACK_PATH,
                     meta_path: Path = config.SLAB_STACK_META_PATH, n_files: int = config.N_FILES, z: float = config.DRONE_Z,
                     verbose: bool = False) -> dict:
    """Read slab_000 .. slab_<n_files-1>.npz and write the stacked (n_files, n_src, ny, nx) float16 array plus a JSON meta file.
    Every frame must have the same sources and grid and contain the level z.  The array is written to a temporary name and renamed."""
    out_path, meta_path = Path(out_path), Path(meta_path)
    stack = None
    sources = grid = None
    for f in range(int(n_files)):
        with np.load(frame_paths(cache_dir, f)[2]) as npz:
            zl = np.asarray(npz["z_levels"], dtype=np.float64).ravel()
            hit = np.flatnonzero(np.abs(zl - float(z)) < 1e-6)
            if hit.size != 1:
                raise ValueError(f"frame {f}: level z={z} not stored (levels {zl.tolist()})")
            dens = np.asarray(npz["density"])[:, int(hit[0])]
            src = tuple(int(v) for v in np.asarray(npz["sources"]).ravel())
            g = tuple(float(v) for v in np.asarray(npz["grid"], dtype=np.float64).ravel())
        if stack is None:
            sources, grid = src, g
            tmp = out_path.with_suffix(".tmp.npy")
            stack = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(int(n_files),) + dens.shape)
        elif src != sources or g != grid or dens.shape != stack.shape[1:]:
            raise ValueError(f"frame {f}: sources / grid / shape differ from frame 0")
        stack[f] = dens.astype(np.float16)
        if verbose and f % 100 == 0:
            print(f"[slab_stack] frame {f}", flush=True)
    stack.flush()
    del stack
    try:
        tmp.replace(out_path)
    except PermissionError as exc:                           # Windows: another process (training / evaluation worker) has the old stack mapped
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{out_path} is memory-mapped by another process; stop the running training / evaluation and rebuild, or verify only (--no-build)") from exc
    meta = {"n_files": int(n_files), "z": float(z), "sources": list(sources), "grid": list(grid), "shape": [int(n_files), *dens.shape], "dtype": "float16"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class StackedSlabBackend(LdmSlabBackend):
    """LdmSlabBackend reading from the memory-mapped stack (module docstring)."""

    def __init__(self, stack_path: Path = config.SLAB_STACK_PATH, meta_path: Path = config.SLAB_STACK_META_PATH,
                 cache_dir: Path = config.CACHE_DIR, max_cached_frames: int = config.FIELD_MAX_CACHED_FRAMES) -> None:
        super().__init__(cache_dir=cache_dir, max_cached_frames=max_cached_frames)
        self.stack_path, self.meta_path = Path(stack_path), Path(meta_path)
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._stack = np.load(self.stack_path, mmap_mode="r")
        if list(self._stack.shape) != meta["shape"] or self._stack.dtype != np.float16:
            raise ValueError(f"{self.stack_path}: shape / dtype do not match {self.meta_path}")
        g = meta["grid"]
        self._grid = SlabGrid(x0=float(g[0]), y0=float(g[1]), nx=int(round(g[2])), ny=int(round(g[3])), res=float(g[4]))
        self._sources = tuple(int(s) for s in meta["sources"])
        self._z_levels = (float(meta["z"]),)
        self._src_map = {int(s): i for i, s in enumerate(self._sources)}
        self.n_files = int(meta["n_files"])

    def slab(self, frame_index: int) -> SlabFrame:
        key = int(frame_index)
        sf = self._frames.get(key)
        if sf is not None:
            self._frames.move_to_end(key)
            return sf
        if not 0 <= key < self.n_files:
            raise IndexError(f"frame {key} outside the stack (0..{self.n_files - 1})")
        dens = np.asarray(self._stack[key], dtype=np.float32)[:, None]            # (n_src, 1, ny, nx), exact float16 to float32
        sf = SlabFrame(index=key, density=np.ascontiguousarray(dens), z_levels=self._z_levels, sources=self._sources, grid=self._grid)
        self._frames[key] = sf
        while len(self._frames) > self.max_cached_frames:
            self._frames.popitem(last=False)
        return sf

    # ------------------------------------------------------------------ queries without converting the whole frame
    def density(self, src_ids, xy, frame_index: int, z: float = config.DRONE_Z, scale: float = 1.0, flip_y: bool = False) -> np.ndarray:   # noqa: ANN001
        """Same contract and bit-identical values as LdmSlabBackend.density, but the bilinear corners are read straight from the float16
        memory map (4 values per point and source instead of a 2.5 ms float32 conversion of the whole frame)."""
        key = int(frame_index)
        if not 0 <= key < self.n_files:
            raise IndexError(f"frame {key} outside the stack (0..{self.n_files - 1})")
        if round(float(z), Z_KEY_DECIMALS) != round(self._z_levels[0], Z_KEY_DECIMALS):
            raise ValueError(f"z={z} not stored in the stack; available z levels: {list(self._z_levels)}")
        ids = [int(src_ids)] if isinstance(src_ids, (int, np.integer)) else [int(s) for s in np.asarray(src_ids).ravel()]
        try:
            si = [self._src_map[s] for s in ids]
        except KeyError as e:
            raise KeyError(f"unknown source id {e.args[0]}; slab sources are {list(self._sources)}") from None
        g = self._grid
        view = self._stack[key]                                                     # (n_src, ny, nx) float16 memmap view
        pts = np.asarray(xy, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"xy must have shape (n, 2) or (2,), got {pts.shape}")
        if pts.shape[0] == 1:
            x, y = pts[0].tolist()
            if flip_y:
                y = -y
            val = _bilinear_one(view[:, None], si, 0, (x - g.x0) / g.res - 0.5, (y - g.y0) / g.res - 0.5, g.nx, g.ny)
            return np.array([val * float(scale)])
        x = pts[:, 0]
        y = -pts[:, 1] if flip_y else pts[:, 1]
        fx = (x - g.x0) / g.res - 0.5
        fy = (y - g.y0) / g.res - 0.5
        inside = (fx >= -0.5) & (fx <= g.nx - 0.5) & (fy >= -0.5) & (fy <= g.ny - 0.5)
        fx = np.clip(fx, 0.0, g.nx - 1.0)
        fy = np.clip(fy, 0.0, g.ny - 1.0)
        ix0 = np.minimum(fx.astype(np.intp), g.nx - 2)
        iy0 = np.minimum(fy.astype(np.intp), g.ny - 2)
        tx = fx - ix0
        ty = fy - iy0
        sk = np.asarray(si, dtype=np.intp)[:, None]
        iy0n, ix0n, iy1n, ix1n = iy0[None, :], ix0[None, :], (iy0 + 1)[None, :], (ix0 + 1)[None, :]
        c00 = view[sk, iy0n, ix0n].astype(np.float32).sum(axis=0)
        c01 = view[sk, iy0n, ix1n].astype(np.float32).sum(axis=0)
        c10 = view[sk, iy1n, ix0n].astype(np.float32).sum(axis=0)
        c11 = view[sk, iy1n, ix1n].astype(np.float32).sum(axis=0)
        out = (1.0 - ty) * ((1.0 - tx) * c00 + tx * c01) + ty * ((1.0 - tx) * c10 + tx * c11)
        out = out * float(scale)
        out[~inside] = 0.0
        return out
