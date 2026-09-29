"""Unit tests for field/concentration_field.py on synthetic slab / frame NPZs (no raw data)."""
import json

import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.field.concentration_field import GridNpzBackend, LdmSlabBackend

NX, NY, RES, X0, Y0 = 6, 5, 5.0, 100.0, -20.0
Z_LEVELS = (15.0, 12.5)
SOURCES = tuple(range(101, 114))
B, C = 0.02, -0.01           # slope of the linear fields (per m)


def a_s(src: int) -> float:
    return 1.0 + 0.1 * (src - 101)


def lin(src: int, x, y, zi: int = 0):
    """Synthetic density d = a_s + b*x + c*y (+ zi) used for every source; exactly bilinear."""
    return a_s(src) + B * np.asarray(x, float) + C * np.asarray(y, float) + zi


def write_slab(path, index: int, value_offset: float = 0.0, sources=SOURCES):
    xc = X0 + RES * (np.arange(NX) + 0.5)
    yc = Y0 + RES * (np.arange(NY) + 0.5)
    xx, yy = np.meshgrid(xc, yc)                                  # (NY, NX)
    dens = np.zeros((len(sources), len(Z_LEVELS), NY, NX), np.float32)
    for si, s in enumerate(sources):
        for zi in range(len(Z_LEVELS)):
            dens[si, zi] = lin(s, xx, yy, zi) + value_offset
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, density=dens.astype(np.float16), z_levels=np.array(Z_LEVELS, np.float32),
             sources=np.array(sources, np.int16), grid=np.array([X0, Y0, NX, NY, RES], np.float64))
    return dens


@pytest.fixture
def cache(tmp_path):
    for i in range(3):
        write_slab(tmp_path / "slabs" / f"slab_{i:03d}.npz", i, value_offset=float(i))
    return tmp_path


def test_bilinear_exact_at_cell_centres_and_edge_midpoints(cache):
    be = LdmSlabBackend(cache, max_cached_frames=2)
    xc = X0 + RES * (np.arange(NX) + 0.5)
    yc = Y0 + RES * (np.arange(NY) + 0.5)
    xx, yy = np.meshgrid(xc, yc)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    got = be.density(101, pts, 0)
    np.testing.assert_allclose(got, lin(101, pts[:, 0], pts[:, 1]), rtol=2e-3)       # float16 storage
    # midpoints between neighbouring centres (cell edges) in x and in y, and a cell corner
    mid = np.array([[xc[1] + RES / 2, yc[2]], [xc[3], yc[1] + RES / 2], [xc[2] + RES / 2, yc[3] + RES / 2]])
    got = be.density(105, mid, 0)
    np.testing.assert_allclose(got, lin(105, mid[:, 0], mid[:, 1]), rtol=2e-3)
    assert got.shape == (3,)
    # single (2,) query is accepted and gives a (1,) array
    assert be.density(105, mid[0], 0).shape == (1,)


def test_outside_grid_is_zero_and_edge_half_cell_is_clamped(cache):
    be = LdmSlabBackend(cache)
    x_end, y_end = X0 + RES * NX, Y0 + RES * NY
    outside = np.array([[X0 - 0.01, 0.0], [x_end + 0.01, 0.0], [110.0, Y0 - 1.0], [110.0, y_end + 1.0]])
    assert np.all(be.density(101, outside, 0) == 0.0)
    # inside the outermost half cell the edge-centre value is held (no extrapolation)
    edge = be.density(101, np.array([[X0 + 0.1, 0.0]]), 0)
    centre = be.density(101, np.array([[X0 + RES / 2, 0.0]]), 0)
    np.testing.assert_allclose(edge, centre)


def test_sum_over_sources_and_scale(cache):
    be = LdmSlabBackend(cache)
    p = np.array([[112.0, -3.0]])
    ids = (101, 108, 113)
    expected = sum(lin(s, p[:, 0], p[:, 1]) for s in ids)
    np.testing.assert_allclose(be.density(ids, p, 0), expected, rtol=2e-3)
    np.testing.assert_allclose(be.density(ids, p, 0, scale=2.5), 2.5 * expected, rtol=2e-3)
    np.testing.assert_allclose(be.density(np.array(ids), p, 0), be.density(list(ids), p, 0))
    with pytest.raises(KeyError):
        be.density(99, p, 0)


def test_flip_y_equals_query_at_minus_y(cache):
    be = LdmSlabBackend(cache)
    p = np.array([[113.0, 4.0], [121.0, -7.5]])
    flipped = be.density(104, p, 0, flip_y=True)
    mirrored = be.density(104, p * np.array([1.0, -1.0]), 0)
    np.testing.assert_allclose(flipped, mirrored)
    assert not np.allclose(flipped, be.density(104, p, 0))      # C != 0 so the field is not symmetric


def test_z_level_selection_and_error(cache):
    be = LdmSlabBackend(cache)
    p = np.array([[112.5, 2.5]])
    assert be.available_z(0) == Z_LEVELS
    d15 = be.density(101, p, 0, z=15.0)
    d12 = be.density(101, p, 0, z=12.5)
    np.testing.assert_allclose(d12 - d15, 1.0, rtol=2e-3)          # zi offset baked into the synthetic field
    with pytest.raises(ValueError) as exc:
        be.density(101, p, 0, z=17.5)
    assert "12.5" in str(exc.value) and "15.0" in str(exc.value)


def test_lru_eviction(cache):
    be = LdmSlabBackend(cache, max_cached_frames=2)
    p = np.array([[112.5, 2.5]])
    be.density(101, p, 0)
    be.density(101, p, 1)
    assert be.cached_frames == (0, 1)
    be.density(101, p, 0)                       # touch 0 -> becomes most recent
    assert be.cached_frames == (1, 0)
    be.density(101, p, 2)                       # evicts 1
    assert be.cached_frames == (0, 2)
    assert be.cache_nbytes() == 2 * len(SOURCES) * len(Z_LEVELS) * NY * NX * 4
    # per-frame values are distinct (offset = index), so the reloaded frame is the right one
    np.testing.assert_allclose(be.density(101, p, 2) - be.density(101, p, 0), 2.0, rtol=2e-3)
    assert be.cached_frames == (2, 0)


def test_airborne_particles_filter_and_meta(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    xyz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], np.float32)
    p_type = np.array([101, 105, 101, 113], np.int32) - config.FRAME_P_TYPE_OFFSET
    np.savez(frames / "frame_007.npz", xyz=xyz, p_type=p_type.astype(np.uint8))
    (frames / "meta_007.json").write_text(json.dumps({"index": 7, "n_airborne": 4}), encoding="utf-8")
    be = LdmSlabBackend(tmp_path)
    np.testing.assert_array_equal(be.airborne_particles(7), xyz)
    np.testing.assert_array_equal(be.airborne_particles(7, src_id=101), xyz[[0, 2]])
    assert be.airborne_particles(7, src_id=102).shape == (0, 3)
    assert be.frame_meta(7)["n_airborne"] == 4


def test_grid_npz_backend_placeholder():
    with pytest.raises(NotImplementedError):
        GridNpzBackend()

def test_single_query_fast_path_matches_vectorised_path(cache):
    be = LdmSlabBackend(cache)
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(X0 - 3, X0 + RES * NX + 3, 40), rng.uniform(Y0 - 3, Y0 + RES * NY + 3, 40)])
    ids = (102, 107, 111)
    for flip in (False, True):
        batch = be.density(ids, pts, 1, z=12.5, scale=1.7, flip_y=flip)
        single = np.concatenate([be.density(ids, p, 1, z=12.5, scale=1.7, flip_y=flip) for p in pts])
        np.testing.assert_allclose(single, batch, rtol=1e-12, atol=1e-12)
    assert be.density(101, (112.5, 2.5), 0).shape == (1,)        # tuple input works on the fast path


# ---------------------------------------------------------------- reviewer-added tests (D2 field review)
def test_source_lookup_uses_stored_sources_array_not_position(tmp_path):
    """The slab's 'sources' array is the only id -> slab-index map: a permuted array must still resolve
    p_type ids correctly (a src_id - 101 shortcut would silently return another source's field)."""
    perm = (110, 101, 113, 105, 102, 108, 111, 104, 107, 112, 103, 109, 106)
    write_slab(tmp_path / "slabs" / "slab_004.npz", 4, sources=perm)
    be = LdmSlabBackend(tmp_path)
    assert be.sources(4) == perm
    p = np.array([[117.0, -3.0], [108.0, 1.5]])
    for s in (101, 110, 106):
        np.testing.assert_allclose(be.density(s, p, 4), lin(s, p[:, 0], p[:, 1]), rtol=2e-3)
        np.testing.assert_allclose(be.density(s, p[0], 4), lin(s, p[0, 0], p[0, 1]), rtol=2e-3)   # fast path
    ids = (113, 102)
    np.testing.assert_allclose(be.density(ids, p, 4), sum(lin(s, p[:, 0], p[:, 1]) for s in ids), rtol=2e-3)


def test_random_interior_points_match_linear_truth_both_paths(cache):
    """Bilinear interpolation of an exactly linear field reproduces it at arbitrary points inside the hull of
    cell centres (checks weights, corner offsets and the y sign of the fractional index in one go)."""
    be = LdmSlabBackend(cache)
    rng = np.random.default_rng(7)
    xc_lo, xc_hi = X0 + RES / 2, X0 + RES * (NX - 0.5)
    yc_lo, yc_hi = Y0 + RES / 2, Y0 + RES * (NY - 0.5)
    pts = np.column_stack([rng.uniform(xc_lo, xc_hi, 200), rng.uniform(yc_lo, yc_hi, 200)])
    for zi, z in enumerate(Z_LEVELS):
        truth = lin(109, pts[:, 0], pts[:, 1], zi) + 2.0          # frame 2 has value_offset 2
        np.testing.assert_allclose(be.density(109, pts, 2, z=z), truth, rtol=2e-3)
        single = np.concatenate([be.density(109, p, 2, z=z) for p in pts[:25]])
        np.testing.assert_allclose(single, truth[:25], rtol=2e-3)
    # flip_y: the mirrored world at (x, y) equals the original field at (x, -y) wherever that is inside
    yin = pts[:, 1][(-pts[:, 1] >= yc_lo) & (-pts[:, 1] <= yc_hi)]
    xin = pts[: len(yin), 0]
    q = np.column_stack([xin, yin])
    np.testing.assert_allclose(be.density(109, q, 2, flip_y=True), lin(109, xin, -yin) + 2.0, rtol=2e-3)