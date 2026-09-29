"""Unit tests for pf/forward_model.py (plan 4.2): Gaussian plume with ground reflection, wind-aligned
coordinates and the LibraryModel wrapper on a synthetic FieldBackend (no raw data)."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.pf.forward_model import ForwardParams, GaussianPlume, LibraryModel, wind_aligned_coords

# numerical-integration grid of the mass-conservation test (test-local, not physics constants)
D_TEST = 100.0
C_RANGE, Z_RANGE = (-200.0, 200.0), (0.0, 200.0)
DC, DZ = 0.5, 0.25
MASS_TOL = 0.02


@pytest.fixture
def plume() -> GaussianPlume:
    return GaussianPlume(ForwardParams())


def test_params_defaults_come_from_config():
    p = ForwardParams()
    assert p.U == config.FWD_DEFAULT_U and p.sigma_v == config.FWD_DEFAULT_SIGMA_V
    assert p.z_s == config.SOURCE_Z and p.sigma0 == config.RELEASE_SIGMA_XY and p.T_L == config.FWD_T_L
    assert p.sigma_z_ratio == config.FWD_SIGMA_Z_RATIO and p.g_floor == config.FWD_G_FLOOR
    assert p.wind_dir_deg == 0.0
    with pytest.raises(Exception):
        p.U = 3.0                                      # frozen
    with pytest.raises(ValueError):
        ForwardParams(U=0.0)


def test_wind_aligned_coords_default_and_rotation():
    src = np.array([[10.0, 20.0], [0.0, 0.0]])
    drn = np.array([[110.0, 50.0], [-40.0, 20.0], [0.0, 100.0]])
    d, c = wind_aligned_coords(src, drn, 0.0)
    assert d.shape == (2, 3) and c.shape == (2, 3)
    assert np.allclose(d, drn[None, :, 0] - src[:, None, 0])
    assert np.allclose(c, drn[None, :, 1] - src[:, None, 1])
    d90, c90 = wind_aligned_coords(np.zeros((1, 2)), np.array([[0.0, 100.0]]), 90.0)
    assert np.isclose(d90[0, 0], 100.0) and np.isclose(c90[0, 0], 0.0, atol=1e-9)
    d180, _ = wind_aligned_coords(np.zeros((1, 2)), np.array([[100.0, 0.0]]), 180.0)
    assert np.isclose(d180[0, 0], -100.0)
    # rotation preserves the norm
    for ang in (0.0, 37.0, 90.0, 213.0):
        dd, cc = wind_aligned_coords(src, drn, ang)
        r = np.hypot(drn[None, :, 0] - src[:, None, 0], drn[None, :, 1] - src[:, None, 1])
        assert np.allclose(np.hypot(dd, cc), r)


def test_shapes_and_1d_inputs(plume):
    src = np.random.default_rng(0).uniform(0, 100, (7, 2))
    drn = np.column_stack([np.linspace(150, 250, 3), np.zeros(3), np.full(3, config.DRONE_Z)])
    g = plume.unit_response(src, drn)
    assert g.shape == (7, 3) and g.dtype == np.float64 and np.all(np.isfinite(g))
    assert plume.unit_response(src[0], drn[0]).shape == (1, 1)
    with pytest.raises(ValueError):
        plume.unit_response(src, drn[:, :2])


def test_crosswind_symmetry(plume):
    src = np.array([[0.0, 0.0]])
    c = np.array([5.0, 20.0, 60.0])
    z = config.DRONE_Z
    gp = plume.unit_response(src, np.column_stack([np.full(3, D_TEST), c, np.full(3, z)]))
    gm = plume.unit_response(src, np.column_stack([np.full(3, D_TEST), -c, np.full(3, z)]))
    assert np.allclose(gp, gm, rtol=1e-12)
    g0 = plume.unit_response(src, np.array([[D_TEST, 0.0, z]]))
    assert np.all(gp < g0)                       # centreline is the maximum


def test_centreline_monotone_decay(plume):
    p = plume.params
    d = np.linspace(5.0 * p.sigma0, 1000.0, 400)
    drn = np.column_stack([d, np.zeros_like(d), np.full_like(d, p.z_s)])   # at release height: no vertical growth
    g = plume.unit_response(np.zeros((1, 2)), drn)[0]
    assert np.all(np.diff(g) < 0.0)
    # sigma_y grows monotonically and matches the closed form
    sy = plume.sigma_y(d)
    ref = np.sqrt(p.sigma0 ** 2 + (p.sigma_v * d / p.U) ** 2 / (1.0 + d / (2.0 * p.U * p.T_L)))
    assert np.allclose(sy, ref) and np.all(np.diff(sy) > 0.0)
    assert np.isclose(plume.sigma_y(np.array([0.0]))[0], p.sigma0)


def test_upwind_returns_floor(plume):
    src = np.array([[100.0, 0.0]])
    drn = np.array([[100.0, 0.0, 15.0], [50.0, 0.0, 15.0], [99.999, 0.0, 15.0], [100.0, 30.0, 5.5]])
    g = plume.unit_response(src, drn)
    assert np.all(g == plume.params.g_floor)
    assert np.all(plume.log_unit_response(src, drn) == np.log(plume.params.g_floor))
    # just downwind the response is far above the floor
    assert plume.unit_response(src, np.array([[100.5, 0.0, 5.5]]))[0, 0] > 1e3 * plume.params.g_floor


def test_mass_conservation_half_space(plume):
    """Integral of g over c in C_RANGE and z in Z_RANGE at fixed d equals 1/U (ground reflection)."""
    c = np.arange(C_RANGE[0], C_RANGE[1] + DC / 2, DC)
    z = np.arange(Z_RANGE[0], Z_RANGE[1] + DZ / 2, DZ)
    cc, zz = np.meshgrid(c, z, indexing="ij")
    drn = np.column_stack([np.full(cc.size, D_TEST), cc.ravel(), zz.ravel()])
    g = plume.unit_response(np.zeros((1, 2)), drn)[0].reshape(cc.shape)
    integral = np.trapezoid(np.trapezoid(g, dx=DZ, axis=1), dx=DC)
    assert abs(integral * plume.params.U - 1.0) < MASS_TOL
    # without the image term the half-space integral would be well below 1/U
    p = plume.params
    sy = plume.sigma_y(np.array([D_TEST]))[0]
    sz = p.sigma_z_ratio * sy
    direct = np.exp(-(zz - p.z_s) ** 2 / (2 * sz ** 2)) * np.exp(-cc ** 2 / (2 * sy ** 2)) / (2 * np.pi * p.U * sy * sz)
    assert np.trapezoid(np.trapezoid(direct, dx=DZ, axis=1), dx=DC) * p.U < 1.0 - MASS_TOL


def test_wind_rotation_moves_plume(plume):
    rot = GaussianPlume(ForwardParams(wind_dir_deg=90.0))
    src = np.zeros((1, 2))
    g_rot = rot.unit_response(src, np.array([[0.0, 100.0, 15.0], [100.0, 0.0, 15.0]]))[0]
    g_ref = plume.unit_response(src, np.array([[100.0, 0.0, 15.0]]))[0, 0]
    assert np.isclose(g_rot[0], g_ref)                 # (0, 100) is downwind under a +y wind
    assert g_rot[1] == rot.params.g_floor              # (100, 0) is crosswind at d = 0 -> floor
    # y-reflection with +x wind: mirrored source and drone give the same response
    s, p = np.array([[300.0, 40.0]]), np.array([[420.0, 70.0, 15.0]])
    assert np.isclose(plume.unit_response(s, p)[0, 0],
                      plume.unit_response(s * [1, -1], p * [1, -1, 1])[0, 0])


def test_log_unit_response_matches_log(plume):
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 500, (50, 2))
    drn = np.column_stack([rng.uniform(0, 700, 6), rng.uniform(-300, 300, 6), rng.uniform(0, 30, 6)])
    g = plume.unit_response(src, drn)
    lg = plume.log_unit_response(src, drn)
    above = g > plume.params.g_floor
    assert above.any()
    assert np.allclose(lg[above], np.log(g[above]), rtol=1e-12)
    assert np.all(lg[~above] == np.log(plume.params.g_floor))
    # far crosswind: log path stays finite and floored instead of log(0)
    far = plume.log_unit_response(np.zeros((1, 2)), np.array([[10.0, 5000.0, 15.0]]))
    assert np.isfinite(far).all() and far[0, 0] == np.log(plume.params.g_floor)


def test_explicit_formula(plume):
    p = plume.params
    src = np.array([[100.0, -50.0]])
    drn = np.array([[250.0, -20.0, 15.0]])
    d, c = 150.0, 30.0
    sy = np.sqrt(p.sigma0 ** 2 + (p.sigma_v * d / p.U) ** 2 / (1 + d / (2 * p.U * p.T_L)))
    sz = p.sigma_z_ratio * sy
    z = 15.0
    ref = (np.exp(-c ** 2 / (2 * sy ** 2)) * (np.exp(-(z - p.z_s) ** 2 / (2 * sz ** 2)) + np.exp(-(z + p.z_s) ** 2 / (2 * sz ** 2)))
           / (2 * np.pi * p.U * sy * sz))
    assert np.isclose(plume.unit_response(src, drn)[0, 0], ref, rtol=1e-12)


class _FakeBackend:
    """FieldBackend stub: density = source id + 0.001 * x - 0.002 * y (flip_y mirrors y), times scale."""

    def __init__(self):
        self.calls = []

    def density(self, src_ids, xy, frame_index, z=config.DRONE_Z, scale=1.0, flip_y=False):
        self.calls.append((tuple(int(s) for s in np.atleast_1d(src_ids)), frame_index, z, scale, flip_y))
        xy = np.asarray(xy, float).reshape(-1, 2)
        y = -xy[:, 1] if flip_y else xy[:, 1]
        return sum(float(s) for s in np.atleast_1d(src_ids)) + 0.001 * xy[:, 0] - 0.002 * y * scale

    def available_z(self, frame_index):
        return (config.DRONE_Z,)


def test_library_model_shapes_and_per_source_calls():
    be = _FakeBackend()
    lib = LibraryModel(be, frame_index=599, z=config.DRONE_Z, scale=2.0)
    drn = np.array([[400.0, 10.0, 15.0], [600.0, -20.0, 15.0]])
    g = lib.unit_response(drn)
    assert g.shape == (len(config.ALL_SOURCES), 2) and lib.n_candidates == 13
    assert lib.candidates == config.ALL_SOURCES
    assert len(be.calls) == 13 and all(len(c[0]) == 1 for c in be.calls)
    assert [c[0][0] for c in be.calls] == list(config.ALL_SOURCES)
    assert all(c[1] == 599 and c[2] == config.DRONE_Z and c[3] == 2.0 and c[4] is False for c in be.calls)
    assert np.allclose(g[:, 0], np.array(config.ALL_SOURCES) + 0.4 - 0.04)
    assert np.allclose(lib.log_unit_response(drn), np.log(g))
    xy = lib.candidates_xy()
    assert xy.shape == (13, 2) and np.allclose(xy[0], config.SOURCES_XY[101])
    assert np.allclose(lib.candidates_xy(flip_y=True)[:, 1], -xy[:, 1])
    # flip_y is forwarded to the backend and mirrors y
    g_flip = lib.unit_response(drn, flip_y=True)
    assert be.calls[-1][4] is True
    assert np.allclose(g_flip[:, 0], np.array(config.ALL_SOURCES) + 0.4 + 0.04)

def test_vectorised_grid_matches_pairwise_and_rotation_equivariance():
    """Reviewer gap test: the (N, M) broadcast equals per-pair scalar evaluation for several wind angles, and
    rotating source, drone and wind together by the same angle leaves g unchanged (CCW convention, plan 4.2)."""
    rng = np.random.default_rng(3)
    src = rng.uniform(0, 500, (9, 2))
    drn = np.column_stack([rng.uniform(0, 700, 5), rng.uniform(-200, 200, 5), rng.uniform(0, 30, 5)])
    for ang in (0.0, 37.0, 250.0):
        pr = GaussianPlume(ForwardParams(wind_dir_deg=ang))
        g = pr.unit_response(src, drn)
        lg = pr.log_unit_response(src, drn)
        ref = np.array([[pr.unit_response(src[i], drn[j])[0, 0] for j in range(drn.shape[0])]
                        for i in range(src.shape[0])])
        assert np.allclose(g, ref, rtol=1e-13) and np.allclose(lg, np.log(ref), rtol=1e-13)
    ang = 37.0
    a = np.deg2rad(ang)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    g0 = GaussianPlume(ForwardParams()).unit_response(src, drn)
    g1 = GaussianPlume(ForwardParams(wind_dir_deg=ang)).unit_response(
        src @ rot.T, np.column_stack([drn[:, :2] @ rot.T, drn[:, 2]]))
    assert np.allclose(g0, g1, rtol=1e-10)