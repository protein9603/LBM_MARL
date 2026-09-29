"""Query interface over the cached per-source concentration slabs (plan 3 S0, module table row
`field/concentration_field.py`; T0-5 latency criterion).

Cache layout (docs/data_cache.md, produced by scripts/convert_ldm.py):
  slabs/slab_XXX.npz  density float16 (n_src=13, n_z, ny=207, nx=200) [particles/m^3]   (report 2.1 units)
                      z_levels float32 (n_z,)   index 400-599: [15, 12.5, 17.5], 0-399: [15]
                      sources  int16   (13,)    p_type ids 101..113 in slab order (report 2.6)
                      grid     float64 [x0, y0, nx, ny, res] = [330, -487.5, 200, 207, 5]   (report 2.5 extent)
  frames/frame_XXX.npz  xyz float32 (N,3) airborne particles only (not deposited, x < X_OUTFLOW; report 2.4),
                        p_type uint8 = p_type - config.FRAME_P_TYPE_OFFSET
  frames/meta_XXX.json  counts, slab statistics, timings.

Slab cell (iy, ix) is centred at x = x0 + res*(ix+0.5), y = y0 + res*(iy+0.5) (preprocess/gridder.py
SlabGrid.cell_centres, row-major iy, ix).  The density at an arbitrary (x, y) is obtained by bilinear
interpolation between the four surrounding cell centres.  In the outermost half cell (between the last
centre and the grid edge) the fractional index is clamped, i.e. the edge value is held constant instead of
extrapolated; queries outside the grid edges return 0 (the slab already covers the whole LDM extent).

y-reflection augmentation (plan 1, "대칭 증강"): flip_y=True returns the ORIGINAL slab evaluated at (x, -y).
This is only physically consistent if the environment mirrors everything else at the same time: the source
positions (y -> -y), the wind field (v -> -v, u and w unchanged), the buildings / occupancy mask and the drone
start positions.  The backend does not know about those; the environment is responsible for that bookkeeping.

Latency (T0-5): a single query goes through a scalar fast path (plain Python arithmetic + integer indexing,
no numpy temporaries) because on this machine every numpy call costs several microseconds; batches of n >= 2
points use the vectorised path.  Both paths compute the same bilinear formula on the same float32 corners.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from srcloc_env import config
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.scripts.convert_ldm import frame_paths

SrcIds = int | Sequence[int] | np.ndarray
Z_KEY_DECIMALS = 6            # z levels are matched after rounding to this many decimals (exact level, no interpolation)


class FieldBackend(Protocol):
    """Concentration-field source of truth for the environment, sensor and PF (plan 3 S0).

    Implementations: LdmSlabBackend (current cached slabs), GridNpzBackend (future 1 s-cadence data, plan 8).
    """

    def density(self, src_ids: SrcIds, xy: np.ndarray, frame_index: int, z: float = config.DRONE_Z,
                scale: float = 1.0, flip_y: bool = False) -> np.ndarray:
        """Number density [particles/m^3] at n query points (n, 2) for the summed sources, times scale."""
        ...

    def available_z(self, frame_index: int) -> tuple[float, ...]:
        """Slab heights [m] stored for that frame."""
        ...


@dataclass(frozen=True)
class SlabFrame:
    """One cached slab file converted to float32 (kept immutable so LRU entries can be shared)."""

    index: int
    density: np.ndarray            # (n_src, n_z, ny, nx) float32 [particles/m^3]
    z_levels: tuple[float, ...]    # slab heights [m]
    sources: tuple[int, ...]       # p_type ids in slab order (101..113)
    grid: SlabGrid
    _z_map: dict[float, int] = field(default_factory=dict, repr=False, compare=False)
    _src_map: dict[int, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._z_map.update({round(float(z), Z_KEY_DECIMALS): i for i, z in enumerate(self.z_levels)})
        self._src_map.update({int(s): i for i, s in enumerate(self.sources)})

    @property
    def nbytes(self) -> int:
        return int(self.density.nbytes)

    def z_index(self, z: float) -> int:
        """Index of the exact z level, else ValueError listing the available levels (no vertical interpolation)."""
        try:
            return self._z_map[round(float(z), Z_KEY_DECIMALS)]
        except KeyError:
            raise ValueError(f"z={z} not stored for frame {self.index}; available z levels: {list(self.z_levels)}") from None

    def source_indices(self, src_ids: SrcIds) -> list[int]:
        """Slab indices for p_type ids (int or sequence) via the stored 'sources' array."""
        try:
            if isinstance(src_ids, (int, np.integer)):
                return [self._src_map[int(src_ids)]]
            return [self._src_map[int(s)] for s in np.asarray(src_ids).ravel()]
        except KeyError as e:
            raise KeyError(f"unknown source id {e.args[0]}; slab sources are {list(self.sources)}") from None


def load_slab_frame(path: Path, index: int) -> SlabFrame:
    """Read one slab_XXX.npz and return its arrays as a float32 SlabFrame."""
    with np.load(path) as npz:
        dens = np.ascontiguousarray(npz["density"], dtype=np.float32)
        z_levels = tuple(float(v) for v in np.asarray(npz["z_levels"]).ravel())
        sources = tuple(int(v) for v in np.asarray(npz["sources"]).ravel())
        g = np.asarray(npz["grid"], dtype=np.float64).ravel()
    if dens.ndim != 4 or dens.shape[0] != len(sources) or dens.shape[1] != len(z_levels):
        raise ValueError(f"{path}: density shape {dens.shape} inconsistent with sources {len(sources)} / z {len(z_levels)}")
    grid = SlabGrid(x0=float(g[0]), y0=float(g[1]), nx=int(round(g[2])), ny=int(round(g[3])), res=float(g[4]))
    if dens.shape[2] != grid.ny or dens.shape[3] != grid.nx:
        raise ValueError(f"{path}: density shape {dens.shape} does not match grid (ny={grid.ny}, nx={grid.nx})")
    if grid.nx < 2 or grid.ny < 2:
        raise ValueError(f"{path}: bilinear interpolation needs nx, ny >= 2, got ({grid.nx}, {grid.ny})")
    return SlabFrame(index=index, density=dens, z_levels=z_levels, sources=sources, grid=grid)


def _bilinear_one(d: np.ndarray, si: list[int], zi: int, fx: float, fy: float, nx: int, ny: int) -> float:
    """Scalar bilinear sample at fractional centre index (fx, fy), summed over slab indices si (fast path)."""
    if fx < -0.5 or fx > nx - 0.5 or fy < -0.5 or fy > ny - 0.5:
        return 0.0
    fx = 0.0 if fx < 0.0 else (nx - 1.0 if fx > nx - 1.0 else fx)
    fy = 0.0 if fy < 0.0 else (ny - 1.0 if fy > ny - 1.0 else fy)
    ix0 = int(fx)
    iy0 = int(fy)
    if ix0 > nx - 2:
        ix0 = nx - 2
    if iy0 > ny - 2:
        iy0 = ny - 2
    tx = fx - ix0
    ty = fy - iy0
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    acc = 0.0
    for k in si:
        acc += ((1.0 - ty) * ((1.0 - tx) * float(d[k, zi, iy0, ix0]) + tx * float(d[k, zi, iy0, ix1]))
                + ty * ((1.0 - tx) * float(d[k, zi, iy1, ix0]) + tx * float(d[k, zi, iy1, ix1])))
    return acc


class LdmSlabBackend:
    """FieldBackend over the cached float16 slabs with lazy per-frame loading into an LRU (plan 3 S0).

    Each frame is converted to float32 on load ((13, n_z, 207, 200) -> 2.15 MB per z level, 6.5 MB for 3
    levels) and kept in an OrderedDict LRU of at most max_cached_frames entries; the most recently queried
    frame stays resident so a fixed-snapshot episode (Mode F, plan 1) pays the disk read once.  Queries are
    vectorised over n points and a single cached query is required to take < config.T0_5_QUERY_LATENCY_S (T0-5).
    """

    def __init__(self, cache_dir: Path = config.CACHE_DIR, max_cached_frames: int = config.FIELD_MAX_CACHED_FRAMES):
        if max_cached_frames < 1:
            raise ValueError("max_cached_frames must be >= 1")
        self.cache_dir = Path(cache_dir)
        self.max_cached_frames = int(max_cached_frames)
        self._frames: OrderedDict[int, SlabFrame] = OrderedDict()

    # ------------------------------------------------------------------ cache management
    def _paths(self, frame_index: int) -> tuple[Path, Path, Path]:
        return frame_paths(self.cache_dir, int(frame_index))

    def slab(self, frame_index: int) -> SlabFrame:
        """Return the (float32) slab frame, loading it from disk and evicting the least recently used one."""
        key = int(frame_index)
        sf = self._frames.get(key)
        if sf is not None:
            self._frames.move_to_end(key)
            return sf
        sf = load_slab_frame(self._paths(key)[2], key)
        self._frames[key] = sf
        while len(self._frames) > self.max_cached_frames:
            self._frames.popitem(last=False)
        return sf

    @property
    def cached_frames(self) -> tuple[int, ...]:
        """Frame indices currently resident, least recently used first."""
        return tuple(self._frames.keys())

    def cache_nbytes(self) -> int:
        """Bytes held by the LRU (float32 density arrays only)."""
        return sum(sf.nbytes for sf in self._frames.values())

    def clear(self) -> None:
        self._frames.clear()

    # ------------------------------------------------------------------ queries
    def available_z(self, frame_index: int) -> tuple[float, ...]:
        return self.slab(frame_index).z_levels

    def sources(self, frame_index: int) -> tuple[int, ...]:
        return self.slab(frame_index).sources

    def density(self, src_ids: SrcIds, xy: np.ndarray, frame_index: int, z: float = config.DRONE_Z,
                scale: float = 1.0, flip_y: bool = False) -> np.ndarray:
        """Bilinear density [particles/m^3] at xy (n, 2) or (2,), summed over src_ids, times scale.

        Returns float64 (n,).  Points outside the grid give 0; inside the outermost half cell the edge value
        is held (fractional index clamped).  flip_y=True samples the original slab at (x, -y) (plan 1
        y-reflection; see the module docstring for what the environment must mirror alongside).
        """
        sf = self.slab(frame_index)
        zi = sf.z_index(z)
        si = sf.source_indices(src_ids)
        g = sf.grid
        pts = np.asarray(xy, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"xy must have shape (n, 2) or (2,), got {pts.shape}")
        if pts.shape[0] == 1:                                  # scalar fast path (T0-5)
            x, y = pts[0].tolist()
            if flip_y:
                y = -y
            val = _bilinear_one(sf.density, si, zi, (x - g.x0) / g.res - 0.5, (y - g.y0) / g.res - 0.5, g.nx, g.ny)
            return np.array([val * float(scale)])
        x = pts[:, 0]
        y = -pts[:, 1] if flip_y else pts[:, 1]
        # fractional cell-centre index: 0 at the first centre, nx-1 at the last centre
        fx = (x - g.x0) / g.res - 0.5
        fy = (y - g.y0) / g.res - 0.5
        inside = (fx >= -0.5) & (fx <= g.nx - 0.5) & (fy >= -0.5) & (fy <= g.ny - 0.5)
        fx = np.clip(fx, 0.0, g.nx - 1.0)
        fy = np.clip(fy, 0.0, g.ny - 1.0)
        ix0 = np.minimum(fx.astype(np.intp), g.nx - 2)
        iy0 = np.minimum(fy.astype(np.intp), g.ny - 2)
        tx = fx - ix0
        ty = fy - iy0
        # (k, n) corner gathers (no copy of the (k, ny, nx) field) summed over the selected sources -> (n,)
        d = sf.density
        sk = np.asarray(si, dtype=np.intp)[:, None]
        iy0n, ix0n, iy1n, ix1n = iy0[None, :], ix0[None, :], (iy0 + 1)[None, :], (ix0 + 1)[None, :]
        c00 = d[sk, zi, iy0n, ix0n].sum(axis=0)
        c01 = d[sk, zi, iy0n, ix1n].sum(axis=0)
        c10 = d[sk, zi, iy1n, ix0n].sum(axis=0)
        c11 = d[sk, zi, iy1n, ix1n].sum(axis=0)
        out = (1.0 - ty) * ((1.0 - tx) * c00 + tx * c01) + ty * ((1.0 - tx) * c10 + tx * c11)
        out = out * float(scale)
        out[~inside] = 0.0
        return out

    # ------------------------------------------------------------------ particle-level access (validation)
    def airborne_particles(self, frame_index: int, src_id: int | None = None) -> np.ndarray:
        """(n, 3) float32 airborne particle positions from frames/, optionally one source (p_type id 101..113)."""
        f_npz = self._paths(frame_index)[0]
        with np.load(f_npz) as npz:
            xyz = np.asarray(npz["xyz"], dtype=np.float32)
            if src_id is None:
                return xyz
            p_type = np.asarray(npz["p_type"])
        code = int(src_id) - config.FRAME_P_TYPE_OFFSET
        return xyz[p_type == code]

    def frame_meta(self, frame_index: int) -> dict:
        """Contents of frames/meta_XXX.json (counts, slab stats, timings written by convert_ldm.py)."""
        f_meta = self._paths(frame_index)[1]
        return json.loads(f_meta.read_text(encoding="utf-8"))


class GridNpzBackend:
    """Placeholder FieldBackend for the requested new data set (plan 8, report 8장).

    Expected format (data request, plan 8 / 표 "새 데이터"): one NPZ per output time at 1 s cadence
    (= 4 LDM steps), 10-15 min runs, outflow particles deleted in the simulator, 30-50 sources, holding a
    per-source 3-D density grid (n_src, nz, ny, nx) plus axis vectors, and a meta.json validated by
    scripts/validate_meta.py.  density() will trilinearly interpolate in (x, y, z) instead of picking an exact
    slab level, and z will no longer need to be one of a few stored heights.  Swapping backends is a config
    one-liner once the schema is known; until then every method raises NotImplementedError.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GridNpzBackend: new-data format not yet received (plan 8)")

    def density(self, src_ids: SrcIds, xy: np.ndarray, frame_index: int, z: float = config.DRONE_Z,
                scale: float = 1.0, flip_y: bool = False) -> np.ndarray:
        raise NotImplementedError

    def available_z(self, frame_index: int) -> tuple[float, ...]:
        raise NotImplementedError