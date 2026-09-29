"""Unit tests for the obstacle map and drone kinematics (synthetic rasters only, no raw data)."""
import json

import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.env.drone import ACTION_STAY, HEADING_NAMES, DroneKinematics, ObstacleMap, heading_unit_vectors

RES = 2.0
N = 30                                   # 30 x 30 cells, centres 0..58 m
DOM = (0.0, RES * (N - 1))               # synthetic domain = raster centre extent (0..58)


def _grid(blocks=(), n=N):
    """blocks: iterable of (ix0, ix1, iy0, iy1, height) inclusive index ranges -> ObstacleMap (nx, ny) = (n, n)."""
    occ = np.zeros((n, n), dtype=bool)
    hmap = np.zeros((n, n), dtype=np.float32)
    for ix0, ix1, iy0, iy1, h in blocks:
        occ[ix0:ix1 + 1, iy0:iy1 + 1] = True
        hmap[ix0:ix1 + 1, iy0:iy1 + 1] = h
    return ObstacleMap.from_arrays(0.0, 0.0, RES, occ, hmap, domain_x=DOM, domain_y=DOM)


def test_headings_and_action_layout():
    v = heading_unit_vectors(8)
    np.testing.assert_allclose(v[0], [1, 0], atol=1e-12)        # E
    np.testing.assert_allclose(v[2], [0, 1], atol=1e-12)        # N
    np.testing.assert_allclose(v[4], [-1, 0], atol=1e-12)       # W
    np.testing.assert_allclose(v[6], [0, -1], atol=1e-12)       # S
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0)
    assert HEADING_NAMES == ("E", "NE", "N", "NW", "W", "SW", "S", "SE") and ACTION_STAY == 8


def test_altitude_dependence_with_margin():
    centre = np.array([22.0, 22.0])                          # inside cells 10..12 (x 19..25)
    low = _grid([(10, 12, 10, 12, 10.0)])                    # 10 m roof
    high = _grid([(10, 12, 10, 12, 14.0)])                   # 14 m roof
    assert low.is_free(centre, 15.0, margin=2.0)             # 10 < 15 - 2 -> free
    assert not high.is_free(centre, 15.0, margin=2.0)        # 14 >= 13 -> blocked
    assert high.is_free(centre, 15.0, margin=0.0)            # 14 < 15 -> free without margin
    assert not low.is_free(centre, 10.0, margin=0.0)         # 10 >= 10 -> blocked at roof height
    assert not low.no_fly_mask(15.0, 2.0).any()


def test_dilation_adds_exactly_one_ring():
    om = _grid([(10, 10, 10, 10, 50.0)])                     # single tall cell
    m0 = om.no_fly_mask(15.0, margin=0.0)
    m2 = om.no_fly_mask(15.0, margin=2.0)                    # ceil(2/2) = 1 iteration
    assert m0.sum() == 1 and m0[10, 10]
    assert m2.sum() == 9
    assert m2[9:12, 9:12].all() and not m2[8, :].any() and not m2[:, 12].any()
    assert om.no_fly_mask(15.0, margin=3.0).sum() == 25       # ceil(3/2) = 2 rings
    assert m2.shape == (N, N) and not m2.flags.writeable
    assert om.no_fly_mask(15.0, 2.0) is m2                   # cached per (z, margin)


def test_ray_distances_against_a_wall():
    om = _grid([(10, 10, 0, N - 1, 50.0)])                   # wall column at x = 20 (cell 10 covers 19..21)
    origin = np.array([[0.0, 30.0]])
    d0 = om.ray_distances(origin, 15.0, margin=0.0)
    assert d0.shape == (1, 8)
    assert d0[0, 0] == pytest.approx(20.0)                   # E: first sample inside the wall cell
    d2 = om.ray_distances(origin, 15.0, margin=2.0)
    assert d2[0, 0] == pytest.approx(18.0)                   # dilated ring blocks cell 9 (17..19)
    assert d0[0, 2] == pytest.approx(30.0)                   # N: domain edge y = 58 -> first sample outside at 60
    assert d0[0, 4] == pytest.approx(2.0)                    # W: x = -2 is outside the domain
    assert abs(d0[0, 0] - 20.0) <= RES
    # cap at max_range when nothing is hit
    free = _grid()
    d = free.ray_distances(np.array([[30.0, 30.0], [30.0, 30.0]]), 15.0, max_range=10.0)
    assert d.shape == (2, 8) and np.all(d == 10.0)
    # n_dirs = 4 -> E, N, W, S
    d4 = om.ray_distances(origin, 15.0, n_dirs=4, margin=0.0)
    assert d4.shape == (1, 4)
    np.testing.assert_allclose(d4[0], [20.0, 30.0, 2.0, 32.0])   # E wall, N edge, W edge (x = -2), S edge (y = -2)


def test_domain_edge_and_raster_edge_handling():
    om = _grid()
    assert om.is_free(np.array([30.0, 30.0]), 15.0)
    assert not om.is_free(np.array([-1.0, 30.0]), 15.0)     # outside the synthetic domain
    assert not om.is_free(np.array([30.0, 58.5]), 15.0)     # outside the synthetic domain (y > 58)
    xy = np.array([[0.0, 0.0], [58.0, 58.0], [59.0, 10.0], [10.0, -0.5]])
    np.testing.assert_array_equal(om.is_free(xy, 15.0), [True, True, False, False])
    # default domain = config.DOMAIN_X/Y; a raster that does not cover the domain blocks outside the raster
    small = ObstacleMap.from_arrays(0.0, 0.0, RES, np.zeros((5, 5), bool), np.zeros((5, 5), np.float32))
    assert small.domain_x == tuple(map(float, config.DOMAIN_X))
    assert not small.is_free(np.array([config.DOMAIN_X[1] + 1.0, 0.0]), 15.0)
    assert not small.is_free(np.array([50.0, 0.0]), 15.0)   # inside the domain but outside the raster
    assert small.is_free(np.array([4.0, 4.0]), 15.0)


def test_action_mask_next_to_wall_and_step():
    om = _grid([(10, 10, 0, N - 1, 50.0)])                   # wall at x = 20
    kin = DroneKinematics(om, step_m=5.0, z=15.0, margin=0.0)
    xy = np.array([14.0, 30.0])
    mask = kin.action_mask(xy)
    assert mask.shape == (9,) and mask.dtype == bool
    assert not mask[0]                                       # E lands at x = 19 -> wall cell 10
    assert mask[1] and mask[7]                               # NE / SE land at x = 17.54 -> cell 9, free
    assert mask[2:7].all() and mask[ACTION_STAY]
    new, applied = kin.step(xy, 0)
    assert not applied and np.array_equal(new, xy)
    new, applied = kin.step(xy, 4)
    assert applied and np.allclose(new, [9.0, 30.0])
    new, applied = kin.step(xy, ACTION_STAY)
    assert applied and np.array_equal(new, xy)
    with pytest.raises(ValueError):
        kin.step(xy, 9)
    # with the 2 m margin the ring blocks cell 9 (x 17..19): NE / SE now masked too
    kin2 = DroneKinematics(om, step_m=5.0, z=15.0, margin=2.0)
    m2 = kin2.action_mask(xy)
    assert not m2[0] and not m2[1] and not m2[7] and m2[ACTION_STAY]


def test_action_mask_at_domain_corner_vectorised():
    om = _grid()
    kin = DroneKinematics(om, step_m=5.0, z=15.0)
    xy = np.array([[0.0, 0.0], [30.0, 30.0], [58.0, 58.0]])
    mask = kin.action_mask(xy)
    assert mask.shape == (3, 9)
    np.testing.assert_array_equal(mask[0], [True, True, True, False, False, False, False, False, True])
    assert mask[1].all()
    # top-right corner: NW / SE diagonals also leave the domain (y > 58 / x > 58)
    np.testing.assert_array_equal(mask[2], [False, False, False, False, True, True, True, False, True])
    new, applied = kin.step(xy, np.array([4, 0, 3]))
    np.testing.assert_array_equal(applied, [False, True, False])
    np.testing.assert_allclose(new[0], [0.0, 0.0])
    np.testing.assert_allclose(new[1], [35.0, 30.0])
    np.testing.assert_allclose(new[2], [58.0, 58.0])
    new, applied = kin.step(xy, np.array([1, 8, 5]))
    np.testing.assert_array_equal(applied, [True, True, True])
    np.testing.assert_allclose(new[0], [5.0 / np.sqrt(2.0), 5.0 / np.sqrt(2.0)])
    np.testing.assert_allclose(new[2], [58.0 - 5.0 / np.sqrt(2.0), 58.0 - 5.0 / np.sqrt(2.0)])
    assert applied.shape == (3,) and new.shape == (3, 2)


def test_stay_always_allowed_even_when_surrounded():
    om = _grid([(5, 25, 5, 25, 50.0)])                       # big block; drone inside it
    kin = DroneKinematics(om, step_m=5.0, z=15.0)
    mask = kin.action_mask(np.array([30.0, 30.0]))
    assert not mask[:ACTION_STAY].any() and mask[ACTION_STAY]
    new, applied = kin.step(np.array([30.0, 30.0]), ACTION_STAY)
    assert applied and np.array_equal(new, [30.0, 30.0])
    assert kin.displacements.shape == (9, 2) and np.allclose(np.linalg.norm(kin.displacements[:8], axis=1), 5.0)


def test_load_from_npz_with_stl_tools_keys(tmp_path):
    nx, ny = 6, 4
    occ = np.zeros((nx, ny), bool)
    hmap = np.zeros((nx, ny), np.float32)
    occ[5, 0] = True
    hmap[5, 0] = 30.0                                        # cell at x = x[5], y = y[0]
    x0, y0, res = 100.0, -20.0, 2.0
    meta = {"res": res, "x0": x0, "y0": y0, "nx": nx, "ny": ny}
    path = tmp_path / "occ.npz"
    np.savez_compressed(path, occ=occ, hmap=hmap, x=x0 + np.arange(nx) * res, y=y0 + np.arange(ny) * res,
                        meta=json.dumps(meta), shift=np.array([0.0, 0.0]))
    om = ObstacleMap.load(path)
    assert (om.nx, om.ny, om.res, om.x0, om.y0) == (nx, ny, res, x0, y0)
    assert not om.is_free(np.array([x0 + 5 * res, y0]), 15.0, margin=0.0)
    assert om.is_free(np.array([x0, y0 + 3 * res]), 15.0, margin=0.0)
    # transposed arrays contradict meta -> refused
    np.savez_compressed(tmp_path / "bad.npz", occ=occ.T, hmap=hmap.T, x=x0 + np.arange(nx) * res,
                        y=y0 + np.arange(ny) * res, meta=json.dumps(meta), shift=np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        ObstacleMap.load(tmp_path / "bad.npz")


def test_half_open_cell_convention_axis_order_and_diagonal_ray():
    """Reviewer test (D2): stl_tools cell convention cell i covers [x0 + (i - 0.5) res, x0 + (i + 0.5) res),
    occ[ix, iy] orientation on a non-square raster where a transposed lookup would NOT raise, and the
    NE ray distance (a rounding-based or transposed implementation fails each of these)."""
    om = _grid([(10, 10, 0, N - 1, 50.0)])                   # wall cell 10 covers x in [19, 21)
    xs = np.array([[18.999, 30.0], [19.0, 30.0], [20.999, 30.0], [21.0, 30.0]])
    ix, _ = om.cell_index(xs)
    np.testing.assert_array_equal(ix, [9, 10, 10, 11])       # floor((p - x0)/res + 0.5), not round()
    np.testing.assert_array_equal(om.is_free(xs, 15.0, margin=0.0), [True, False, False, True])
    # NE ray from the origin: sample k sits at x = 2k/sqrt2; first sample in cell 10 (x >= 19) is k = 14 -> 28 m,
    # with the 2 m ring cell 9 (x >= 17) is blocked at k = 13 -> 26 m; SE leaves the domain at the first sample.
    d0 = om.ray_distances(np.array([[0.0, 0.0]]), 15.0, margin=0.0)
    d2 = om.ray_distances(np.array([[0.0, 0.0]]), 15.0, margin=2.0)
    assert d0[0, 1] == pytest.approx(28.0) and d2[0, 1] == pytest.approx(26.0)
    assert d0[0, 7] == pytest.approx(2.0) and d0[0, 5] == pytest.approx(2.0)
    # non-square raster (nx = 6, ny = 4): only cell (ix = 1, iy = 3) -> (x = 2, y = 6) is blocked; (x = 6, y = 2)
    # (the transposed cell, in bounds) must stay free
    occ = np.zeros((6, 4), bool)
    occ[1, 3] = True
    hmap = np.where(occ, np.float32(50.0), np.float32(0.0))
    om2 = ObstacleMap.from_arrays(0.0, 0.0, RES, occ, hmap, domain_x=(0.0, 10.0), domain_y=(0.0, 6.0))
    assert om2.no_fly_mask(15.0, margin=0.0).shape == (6, 4)
    np.testing.assert_array_equal(om2.is_free(np.array([[2.0, 6.0], [6.0, 2.0], [2.0, 4.0]]), 15.0, margin=0.0),
                                  [False, True, True])
    kin = DroneKinematics(om2, step_m=2.0, z=15.0, margin=0.0)
    assert not kin.action_mask(np.array([2.0, 4.0]))[2]      # N from (2, 4) lands in the blocked cell
    assert kin.action_mask(np.array([4.0, 2.0]))[2]          # N from (4, 2) -> (4, 4) free