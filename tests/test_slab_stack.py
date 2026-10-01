"""D11: memory-mapped slab stack (field/slab_stack.py) on synthetic slab NPZs: identical values, mmap, error handling."""
from __future__ import annotations

import numpy as np
import pytest

from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.field.slab_stack import StackedSlabBackend, build_slab_stack
from tests.test_concentration_field import NX, NY, RES, SOURCES, X0, Y0, write_slab

N = 4


@pytest.fixture
def cache(tmp_path):
    for i in range(N):
        write_slab(tmp_path / "slabs" / f"slab_{i:03d}.npz", i, value_offset=0.37 * i)
    return tmp_path


def _build(cache, z=15.0):
    return build_slab_stack(cache, cache / "stack.npy", cache / "stack.json", n_files=N, z=z)


def test_stack_values_equal_the_per_frame_backend_bit_for_bit(cache):
    meta = _build(cache)
    assert meta["shape"] == [N, len(SOURCES), NY, NX] and meta["dtype"] == "float16"
    a = LdmSlabBackend(cache)
    b = StackedSlabBackend(cache / "stack.npy", cache / "stack.json", cache)
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(X0 - 3, X0 + NX * RES + 3, 60), rng.uniform(Y0 - 3, Y0 + NY * RES + 3, 60)])
    for f in range(N):
        for src in (101, 105, 113):
            assert np.array_equal(a.density([src], pts, f, 15.0), b.density([src], pts, f, 15.0))
            assert np.array_equal(a.density([src], pts, f, 15.0, 2.5, flip_y=True), b.density([src], pts, f, 15.0, 2.5, flip_y=True))
        assert np.array_equal(a.density([101, 102], pts[:1], f, 15.0), b.density([101, 102], pts[:1], f, 15.0))     # scalar fast path, summed sources


def test_stack_is_memory_mapped_and_slab_frames_are_float32_views_of_one_level(cache):
    _build(cache)
    b = StackedSlabBackend(cache / "stack.npy", cache / "stack.json", cache, max_cached_frames=2)
    assert isinstance(b._stack, np.memmap) and b.n_files == N
    sf = b.slab(2)
    assert sf.density.dtype == np.float32 and sf.density.shape == (len(SOURCES), 1, NY, NX) and sf.z_levels == (15.0,)
    assert sf.z_index(15.0) == 0 and sf.sources == SOURCES and sf.grid.nx == NX
    with pytest.raises(ValueError):
        sf.z_index(12.5)                                                       # only the stacked level exists
    for f in range(N):
        b.slab(f)
    assert len(b.cached_frames) == 2                                           # LRU bound is kept


def test_other_level_can_be_stacked_and_errors_are_explicit(cache):
    _build(cache, z=12.5)
    a = LdmSlabBackend(cache)
    b = StackedSlabBackend(cache / "stack.npy", cache / "stack.json", cache)
    pt = np.array([[X0 + 11.0, Y0 + 7.0]])
    assert np.array_equal(a.density([103], pt, 1, 12.5), b.density([103], pt, 1, 12.5))
    with pytest.raises(IndexError):
        b.slab(N)
    with pytest.raises(ValueError):
        build_slab_stack(cache, cache / "x.npy", cache / "x.json", n_files=N, z=20.0)
    assert not (cache / "x.npy").exists()
