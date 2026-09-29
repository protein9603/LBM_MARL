"""Unit tests for the LBM wind lookup (synthetic lattices only, no raw data)."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.field.wind import WindField


def _lattice(nx=6, ny=5, dx=2.5, x0=0.0, y0=-5.0):
    x = x0 + dx * np.arange(nx)
    y = y0 + dx * np.arange(ny)
    return x, y


def _linear_field(x, y, z_levels, a=1.0, b=0.2, c=-0.3, dz=0.1):
    """u = a + b*x + c*y + dz*z, v = 2 - b*x, w = 0.5*z; stacked as (nz, nx, ny, 3)."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    uvw = np.zeros((len(z_levels), len(x), len(y), 3), np.float32)
    for k, z in enumerate(z_levels):
        uvw[k, :, :, 0] = a + b * xx + c * yy + dz * z
        uvw[k, :, :, 1] = 2.0 - b * xx
        uvw[k, :, :, 2] = 0.5 * z
    return uvw


def test_axis_order_is_level_ix_iy_comp():
    x, y = _lattice(nx=6, ny=4)
    uvw = np.zeros((2, 6, 4, 3), np.float32)
    uvw[:, 5, 0, 0] = 7.0            # node at x = x[5], y = y[0]
    wf = WindField.from_arrays(x, y, [1.0, 2.0], uvw)
    assert wf.uv_at(np.array([[x[5], y[0]]]), 1.0)[0, 0] == pytest.approx(7.0)
    assert wf.uv_at(np.array([[x[0], y[3]]]), 1.0)[0, 0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        WindField.from_arrays(x, y, [1.0, 2.0], np.zeros((2, 4, 6, 3), np.float32))
    with pytest.raises(ValueError):   # height map without a mask is a caller error, not silently ignored
        WindField.from_arrays(x, y, [1.0, 2.0], uvw, None, np.zeros((6, 4), np.float32))


def test_bilinear_exact_on_linear_field():
    x, y = _lattice()
    zl = np.array([10.0, 20.0])
    wf = WindField.from_arrays(x, y, zl, _linear_field(x, y, zl))
    rng = np.random.default_rng(0)
    xy = np.column_stack([rng.uniform(x[0], x[-1], 50), rng.uniform(y[0], y[-1], 50)])
    uv = wf.uv_at(xy, 10.0)
    np.testing.assert_allclose(uv[:, 0], 1.0 + 0.2 * xy[:, 0] - 0.3 * xy[:, 1] + 0.1 * 10.0, rtol=1e-6)
    np.testing.assert_allclose(uv[:, 1], 2.0 - 0.2 * xy[:, 0], rtol=1e-6)
    # exact at the nodes too
    uv_nodes = wf.uv_at(np.array([[x[2], y[3]]]), 10.0)
    assert uv_nodes[0, 0] == pytest.approx(1.0 + 0.2 * x[2] - 0.3 * y[3] + 1.0)


def test_bilinear_corner_weights_on_nonlinear_field():
    """A single non-zero node must fade with the product weight (1-tx)(1-ty) etc. - catches corner permutations."""
    x, y = _lattice(nx=4, ny=4, dx=2.0, x0=10.0, y0=-4.0)
    zl = np.array([5.0])
    uvw = np.zeros((1, 4, 4, 3), np.float32)
    uvw[0, 2, 1, 0] = 8.0              # node (ix=2, iy=1) -> x = 14, y = -2
    wf = WindField.from_arrays(x, y, zl, uvw)
    # query in cell ix 1..2, iy 1..2 at tx = 0.75, ty = 0.25: the node is corner (1, 0) -> weight tx*(1-ty)
    q = np.array([[10.0 + 2.0 * 1.75, -4.0 + 2.0 * 1.25]])
    assert wf.uv_at(q, 5.0)[0, 0] == pytest.approx(8.0 * 0.75 * 0.75)
    # query in cell ix 2..3, iy 0..1 at tx = 0.25, ty = 0.5: the node is corner (0, 1) -> weight (1-tx)*ty
    q2 = np.array([[10.0 + 2.0 * 2.25, -4.0 + 2.0 * 0.5]])
    assert wf.uv_at(q2, 5.0)[0, 0] == pytest.approx(8.0 * 0.75 * 0.5)
    # no leakage into v / w
    assert wf.uvw_at(q, 5.0)[0, 1] == 0.0 and wf.w_at(q, 5.0)[0] == 0.0


def test_linear_in_z_between_bracketing_levels():
    x, y = _lattice()
    zl = np.array([1.25, 13.75, 16.25, 30.0])
    wf = WindField.from_arrays(x, y, zl, _linear_field(x, y, zl))
    xy = np.array([[3.3, -1.1]])
    u15 = wf.uv_at(xy, 15.0)[0, 0]
    u1375 = wf.uv_at(xy, 13.75)[0, 0]
    u1625 = wf.uv_at(xy, 16.25)[0, 0]
    assert u15 == pytest.approx(0.5 * (u1375 + u1625), rel=1e-6)
    assert u15 == pytest.approx(1.0 + 0.2 * 3.3 - 0.3 * (-1.1) + 0.1 * 15.0, rel=1e-6)
    assert wf.w_at(xy, 15.0)[0] == pytest.approx(7.5, rel=1e-6)
    # outside the vertical range: clamped to the end levels
    assert wf.uv_at(xy, 0.0)[0, 0] == pytest.approx(wf.uv_at(xy, 1.25)[0, 0])
    assert wf.uv_at(xy, 100.0)[0, 0] == pytest.approx(wf.uv_at(xy, 30.0)[0, 0])
    assert wf.uv_at(xy, config.DRONE_Z).shape == (1, 2)


def test_positions_outside_lattice_are_clipped():
    x, y = _lattice()
    zl = np.array([10.0, 20.0])
    wf = WindField.from_arrays(x, y, zl, _linear_field(x, y, zl))
    far = np.array([[x[-1] + 100.0, y[-1] + 50.0], [x[0] - 30.0, y[0] - 30.0]])
    edge = np.array([[x[-1], y[-1]], [x[0], y[0]]])
    np.testing.assert_allclose(wf.uv_at(far, 12.0), wf.uv_at(edge, 12.0))
    assert np.all(np.isfinite(wf.uv_at(far, 12.0)))


def test_dead_node_renormalisation():
    x, y = _lattice(nx=4, ny=4, dx=1.0, x0=0.0, y0=0.0)
    zl = np.array([10.0, 20.0])
    uvw = np.zeros((2, 4, 4, 3), np.float32)
    uvw[:, :, :, 0] = 1.0
    uvw[:, 2, 2, 0] = 100.0           # a "building" node with garbage inside
    mask = np.zeros((4, 4), bool)
    mask[2, 2] = True
    height = np.zeros((4, 4), np.float32)
    height[2, 2] = 15.0
    wf = WindField.from_arrays(x, y, zl, uvw, mask, height)
    # query inside the cell (1..2, 1..2): corner (2,2) dead at z=10 -> remaining corners all 1.0 -> exactly 1.0
    q = np.array([[1.75, 1.75]])
    assert wf.uv_at(q, 10.0)[0, 0] == pytest.approx(1.0)
    # at z = 20 the building (height 15) is below -> node alive -> garbage leaks in as expected
    assert wf.uv_at(q, 20.0)[0, 0] > 1.0
    # renormalisation: weights of the three alive corners sum to one
    uvw2 = uvw.copy()
    uvw2[:, 1, 1, 0], uvw2[:, 2, 1, 0], uvw2[:, 1, 2, 0] = 2.0, 4.0, 8.0
    wf2 = WindField.from_arrays(x, y, zl, uvw2, mask, height)
    tx = ty = 0.75
    w = np.array([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty])
    expected = (w * np.array([2.0, 4.0, 8.0])).sum() / w.sum()
    assert wf2.uv_at(q, 10.0)[0, 0] == pytest.approx(expected, rel=1e-6)
    # all four dead -> (0, 0)
    mask4 = np.ones((4, 4), bool)
    height4 = np.full((4, 4), 50.0, np.float32)
    wf4 = WindField.from_arrays(x, y, zl, uvw, mask4, height4)
    np.testing.assert_array_equal(wf4.uv_at(q, 10.0), np.zeros((1, 2)))
    assert wf4.w_at(q, 10.0)[0] == 0.0
    # mask without height map blocks at every height
    wf5 = WindField.from_arrays(x, y, zl, uvw, mask)
    assert wf5.uv_at(q, 20.0)[0, 0] == pytest.approx(1.0)


def test_mean_profile_and_region_mean_on_constant_field():
    x, y = _lattice(nx=8, ny=6)
    zl = np.array([1.25, 11.25, 21.25])
    uvw = np.zeros((3, 8, 6, 3), np.float32)
    uvw[0, :, :, 0], uvw[1, :, :, 0], uvw[2, :, :, 0] = 1.0, 3.0, 5.0
    uvw[:, :, :, 1] = 0.5
    mask = np.zeros((8, 6), bool)
    mask[0, 0] = True
    height = np.zeros((8, 6), np.float32)
    height[0, 0] = 100.0
    uvw[:, 0, 0, 0] = 99.0            # dead node value
    wf = WindField.from_arrays(x, y, zl, uvw, mask, height)
    u, v = wf.mean_profile(11.25)
    assert u == pytest.approx((3.0 * 47 + 99.0) / 48) and v == pytest.approx(0.5)
    u16, _ = wf.mean_profile(16.25)
    assert u16 == pytest.approx(0.5 * ((3.0 * 47 + 99.0) / 48 + (5.0 * 47 + 99.0) / 48))
    whole = dict(x_range=(x[0] - 1.0, x[-1] + 1.0), y_range=(y[0] - 1.0, y[-1] + 1.0))   # open window holding every node
    ur, vr = wf.region_mean(11.25, exclude_buildings=True, **whole)
    assert ur == pytest.approx(3.0) and vr == pytest.approx(0.5)
    ur_all, _ = wf.region_mean(11.25, exclude_buildings=False, **whole)
    assert ur_all == pytest.approx(u)
    assert wf.direction_deg(11.25, **whole) == pytest.approx(np.degrees(np.arctan2(0.5, 3.0)))
    assert wf.region_mean(11.25, x_range=(1e6, 2e6), y_range=(0, 1)) == (0.0, 0.0)


def test_region_window_bounds_are_open():
    """Report 3.3 / v14: x > 330, x < 1315, |y| < 500 - nodes exactly on the bounds are excluded."""
    x, y = _lattice(nx=5, ny=5, dx=2.5, x0=0.0, y0=-5.0)      # x 0..10, y -5..5
    zl = np.array([11.25, 13.75])
    uvw = np.zeros((2, 5, 5, 3), np.float32)
    uvw[:, 0, :, 0] = uvw[:, 4, :, 0] = 50.0                   # nodes on x = 0 and x = 10
    uvw[:, :, 0, 0] = uvw[:, :, 4, 0] = 50.0                   # nodes on y = -5 and y = 5
    uvw[:, 1:4, 1:4, 0] = 2.0                                  # interior 3 x 3 nodes
    wf = WindField.from_arrays(x, y, zl, uvw)
    u, _ = wf.region_mean(11.25, x_range=(0.0, 10.0), y_range=(-5.0, 5.0), exclude_buildings=False)
    assert u == pytest.approx(2.0)                             # boundary nodes (50) must not be counted
    u_in, _ = wf.region_mean(11.25, x_range=(-0.1, 10.1), y_range=(-5.1, 5.1), exclude_buildings=False)
    assert u_in == pytest.approx((2.0 * 9 + 50.0 * 16) / 25)


def test_band_mean_pools_stored_levels_half_open():
    x, y = _lattice(nx=5, ny=5, dx=2.5, x0=0.0, y0=-5.0)
    zl = np.array([1.25, 3.75, 6.25, 11.25, 13.75, 16.25])
    uvw = np.zeros((6, 5, 5, 3), np.float32)
    uvw[:, :, :, 0] = zl[:, None, None]                        # u = z at every node
    uvw[:, :, :, 1] = 0.25
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    height = np.zeros((5, 5), np.float32)
    height[2, 2] = 12.0                                        # dead at 11.25, alive at 13.75
    uvw[:, 2, 2, 0] = -100.0
    wf = WindField.from_arrays(x, y, zl, uvw, mask, height)
    win = dict(x_range=(-1.0, 11.0), y_range=(-6.0, 6.0))
    # [10, 15) picks 11.25 and 13.75 only; all nodes -> pooled mean over 2 x 25 nodes
    u, v = wf.band_mean((10.0, 15.0), exclude_buildings=False, **win)
    assert u == pytest.approx((11.25 * 24 - 100.0 + 13.75 * 24 - 100.0) / 50) and v == pytest.approx(0.25)
    # exclude_buildings: dead only on the 11.25 level -> pooled over 24 + 25 nodes (not a mean of level means)
    u_ex, _ = wf.band_mean((10.0, 15.0), exclude_buildings=True, **win)
    assert u_ex == pytest.approx((11.25 * 24 + 13.75 * 24 - 100.0) / 49)
    # [0, 5) -> 1.25 and 3.75; empty band -> (0, 0)
    assert wf.band_mean((0.0, 5.0), exclude_buildings=False, **win)[0] == pytest.approx((1.25 * 25 - 101.25 + 3.75 * 25 - 103.75) / 50)
    assert wf.band_mean((7.0, 10.0), **win) == (0.0, 0.0)


def test_box_mean_speed_matches_report_2_6_definition():
    """mean |(u,v,w)| over ALL nodes with |dx| < h, |dy| < h, z_lo < z < z_hi (open bounds, buildings included)."""
    x, y = _lattice(nx=9, ny=9, dx=2.5, x0=0.0, y0=-10.0)     # x 0..20, y -10..10
    zl = np.array([1.25, 3.75, 6.25, 13.75, 16.25])
    uvw = np.zeros((5, 9, 9, 3), np.float32)
    uvw[:, :, :, 0] = 3.0
    uvw[:, :, :, 1] = 4.0                                      # speed 5 everywhere ...
    uvw[1:4, 4, 4, :] = 0.0                                    # ... except the centre node on the 3 inner levels
    uvw[0, :, :, :] = 100.0                                    # level 1.25 must be excluded by z_lo = 2
    uvw[4, :, :, :] = 100.0                                    # level 16.25 excluded by z_hi = 15
    uvw[1:4, 0, :, :] = uvw[1:4, 8, :, :] = 100.0              # nodes at |dx| = 10 excluded by "< 10"
    uvw[1:4, :, 0, :] = uvw[1:4, :, 8, :] = 100.0
    mask = np.zeros((9, 9), bool)
    mask[4, 4] = True
    wf = WindField.from_arrays(x, y, zl, uvw, mask, np.full((9, 9), 50.0, np.float32))
    got = wf.box_mean_speed((10.0, 0.0), half_width=10.0, z_range=(2.0, 15.0))
    assert got == pytest.approx((5.0 * (49 * 3 - 3)) / (49 * 3))     # 7x7 nodes x 3 levels, centre node zero, mask ignored
    assert np.isnan(wf.box_mean_speed((10.0, 0.0), half_width=10.0, z_range=(7.0, 10.0)))


def test_load_from_npz(tmp_path):
    x, y = _lattice(nx=5, ny=5)
    zl = np.array([1.25, 3.75])
    uvw = _linear_field(x, y, zl)
    mask = np.zeros((5, 5), bool)
    height = np.zeros((5, 5), np.float32)
    lv = tmp_path / "levels.npz"
    sl = tmp_path / "slices.npz"
    np.savez(lv, uvw=uvw, z_levels=zl, x=x, y=y)
    np.savez(sl, x=x, y=y, building_mask=mask, building_height_max=height)
    wf = WindField.load(lv, sl)
    assert wf.uvw.dtype == np.float32 and wf.shape == (2, 5, 5)
    np.testing.assert_allclose(wf.z_levels, zl)
    assert wf.uv_at(np.array([[x[1], y[2]]]), 1.25)[0, 0] == pytest.approx(uvw[0, 1, 2, 0])
    wf2 = WindField.load(lv, None)
    assert not wf2.building_mask.any()