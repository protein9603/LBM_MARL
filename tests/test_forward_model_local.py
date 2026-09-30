"""Unit tests for GaussianPlume(wind_mode='local') (plan S1 보강 / 4.2 option A, D4-2) on synthetic
WindField.from_arrays lattices only (no raw data).  Test numbering follows the D4-2 task list:
(1) uniform +x field -> local == global, (2) rotated uniform field -> local == global with wind_dir_deg=90,
(3) speed below FWD_U_MIN -> U_MIN, (4) dead node -> U_MIN and global-direction fallback, (5) shapes,
(6) log_unit_response consistency, (7) independent hand-written formula with distinct per-source (U_i, dir_i);
plus constructor errors, blend, array-angle wind_aligned_coords."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.field.wind import WindField
from srcloc_env.pf.forward_model import ForwardParams, GaussianPlume, LocalWind, wind_aligned_coords

# synthetic lattice (test-local layout, not physics constants): x 0..20 m, y -10..10 m, 2.5 m spacing
NX, NY, DX, X0, Y0 = 9, 9, 2.5, 0.0, -10.0
Z_LEVELS = (10.0, 20.0)                      # brackets config.FWD_LOCAL_WIND_Z = 15 m
N_SRC, N_DRN = 7, 3


def _lattice():
    return X0 + DX * np.arange(NX), Y0 + DX * np.arange(NY)


def _uniform_field(u: float, v: float, mask=None, height=None) -> WindField:
    x, y = _lattice()
    uvw = np.zeros((len(Z_LEVELS), NX, NY, 3), np.float32)
    uvw[..., 0] = u
    uvw[..., 1] = v
    return WindField.from_arrays(x, y, np.array(Z_LEVELS), uvw, mask, height)


def _split_field(u_left: float, u_right: float, ix_split: int, mask=None) -> WindField:
    """u = u_left for ix < ix_split, u_right for ix >= ix_split; v = 0."""
    x, y = _lattice()
    uvw = np.zeros((len(Z_LEVELS), NX, NY, 3), np.float32)
    uvw[:, :ix_split, :, 0] = u_left
    uvw[:, ix_split:, :, 0] = u_right
    return WindField.from_arrays(x, y, np.array(Z_LEVELS), uvw, mask)


def _random_sources_and_drones(seed: int = 0):
    rng = np.random.default_rng(seed)
    x, y = _lattice()
    src = np.column_stack([rng.uniform(x[0], x[-1], N_SRC), rng.uniform(y[0], y[-1], N_SRC)])
    drn = np.column_stack([rng.uniform(-100, 300, N_DRN), rng.uniform(-200, 200, N_DRN), rng.uniform(0, 30, N_DRN)])
    return src, drn


def test_config_constants_of_option_a():
    assert config.FWD_U_MIN == 0.3 and config.PF_EPS_MIX_TRAPPED == 0.1
    assert config.FWD_WIND_MODES == ("global", "local") and config.FWD_LOCAL_WIND_Z == config.DRONE_Z
    assert ForwardParams().local_wind_blend == config.FWD_LOCAL_WIND_BLEND == 1.0


def test_1_uniform_plus_x_field_local_equals_global():
    """(1) uniform +x wind with speed params.U: the local mode reproduces the global response."""
    src, drn = _random_sources_and_drones(0)
    params = ForwardParams(U=2.0)                       # float32-exact speed -> equality up to bilinear rounding
    wf = _uniform_field(params.U, 0.0)
    loc = GaussianPlume(params, wind_field=wf, wind_mode="local")
    glob = GaussianPlume(params)
    np.testing.assert_allclose(loc.unit_response(src, drn), glob.unit_response(src, drn), rtol=1e-12)
    np.testing.assert_allclose(loc.log_unit_response(src, drn), glob.log_unit_response(src, drn), rtol=1e-12)
    lw = loc.local_wind(src)
    assert isinstance(lw, LocalWind)
    np.testing.assert_allclose(lw.U, params.U, rtol=1e-12)
    assert np.all(lw.dir_deg == 0.0) and not lw.fallback.any()
    np.testing.assert_allclose(lw.e, [[1.0, 0.0]] * N_SRC)
    # the config default U (1.68, not float32-exact) still agrees to float32 precision
    p_def = ForwardParams()
    wf_def = _uniform_field(p_def.U, 0.0)
    np.testing.assert_allclose(GaussianPlume(p_def, wf_def, "local").unit_response(src, drn),
                               GaussianPlume(p_def).unit_response(src, drn), rtol=1e-6)


def test_2_rotated_uniform_field_equals_global_wind_dir_90():
    """(2) uniform +y wind of 2.0 m/s: local == global with wind_dir_deg = 90 and U = 2.0 (params.U ignored)."""
    src, drn = _random_sources_and_drones(1)
    wf = _uniform_field(0.0, 2.0)
    loc = GaussianPlume(ForwardParams(U=1.0, wind_dir_deg=0.0), wind_field=wf, wind_mode="local")
    glob = GaussianPlume(ForwardParams(U=2.0, wind_dir_deg=90.0))
    np.testing.assert_allclose(loc.unit_response(src, drn), glob.unit_response(src, drn), rtol=1e-10)
    lw = loc.local_wind(src)
    np.testing.assert_allclose(lw.U, 2.0, rtol=1e-12)
    np.testing.assert_allclose(lw.dir_deg, 90.0)
    np.testing.assert_allclose(lw.uv, [[0.0, 2.0]] * N_SRC, rtol=1e-12)


def test_3_speed_below_u_min_is_clipped():
    """(3) a hypothesis where |(u, v)| < FWD_U_MIN uses U = FWD_U_MIN; the other keeps its own speed."""
    x, y = _lattice()
    wf = _split_field(2.0, 0.1, ix_split=4)                     # x < 10 m: 2.0 m/s, x >= 10 m: 0.1 m/s
    src = np.array([[x[1], y[4]], [x[6], y[4]]])                # on lattice nodes: no bilinear mixing
    drn = np.column_stack([np.linspace(50, 250, N_DRN), np.full(N_DRN, 3.0), np.full(N_DRN, config.DRONE_Z)])
    loc = GaussianPlume(ForwardParams(U=1.0), wind_field=wf, wind_mode="local")
    lw = loc.local_wind(src)
    np.testing.assert_allclose(lw.speed, [2.0, 0.1], rtol=1e-6)
    np.testing.assert_allclose(lw.U, [2.0, config.FWD_U_MIN], rtol=1e-6)
    assert not lw.fallback.any()                                # 0.1 m/s is calm but not zero: direction kept
    g = loc.unit_response(src, drn)
    np.testing.assert_allclose(g[0], GaussianPlume(ForwardParams(U=2.0)).unit_response(src[0], drn)[0], rtol=1e-6)
    np.testing.assert_allclose(g[1], GaussianPlume(ForwardParams(U=config.FWD_U_MIN)).unit_response(src[1], drn)[0],
                               rtol=1e-6)


def test_4_dead_node_falls_back_to_u_min_and_global_direction():
    """(4) all four bracketing nodes inside a building -> (u, v) = (0, 0) -> U_MIN and the GLOBAL direction."""
    x, y = _lattice()
    mask = np.zeros((NX, NY), dtype=bool)
    mask[4:, :] = True                                          # x >= 10 m blocked at every height (no height map)
    wf = _uniform_field(0.0, 2.0, mask=mask)                    # open part: +y wind 2 m/s
    src = np.array([[x[1] + 0.7, y[3] + 1.1], [x[6] + 0.4, y[4] + 0.9]])   # open cell / fully dead cell
    drn = np.column_stack([np.linspace(30, 120, N_DRN), np.linspace(40, 160, N_DRN), np.full(N_DRN, config.DRONE_Z)])
    params = ForwardParams(U=1.0, wind_dir_deg=30.0)
    loc = GaussianPlume(params, wind_field=wf, wind_mode="local")
    lw = loc.local_wind(src)
    np.testing.assert_allclose(lw.uv[1], [0.0, 0.0])
    assert lw.fallback.tolist() == [False, True]
    np.testing.assert_allclose(lw.U, [2.0, config.FWD_U_MIN], rtol=1e-12)
    np.testing.assert_allclose(lw.dir_deg, [90.0, params.wind_dir_deg])
    g = loc.unit_response(src, drn)
    ref_open = GaussianPlume(ForwardParams(U=2.0, wind_dir_deg=90.0)).unit_response(src[0], drn)[0]
    ref_dead = GaussianPlume(ForwardParams(U=config.FWD_U_MIN, wind_dir_deg=params.wind_dir_deg)).unit_response(src[1], drn)[0]
    np.testing.assert_allclose(g[0], ref_open, rtol=1e-10)
    np.testing.assert_allclose(g[1], ref_dead, rtol=1e-12)
    assert np.any(g[1] > params.g_floor)                        # the dead-node hypothesis is still evaluated


def test_5_shapes_unchanged_in_local_mode():
    src, drn = _random_sources_and_drones(2)
    loc = GaussianPlume(ForwardParams(), wind_field=_uniform_field(1.0, 0.5), wind_mode="local")
    g = loc.unit_response(src, drn)
    assert g.shape == (N_SRC, N_DRN) and g.dtype == np.float64 and np.all(np.isfinite(g))
    assert loc.log_unit_response(src, drn).shape == (N_SRC, N_DRN)
    assert loc.unit_response(src[0], drn[0]).shape == (1, 1)
    lw = loc.local_wind(src)
    assert lw.uv.shape == (N_SRC, 2) and lw.U.shape == (N_SRC,) and lw.dir_deg.shape == (N_SRC,)
    with pytest.raises(ValueError):
        loc.unit_response(src, drn[:, :2])


def test_6_log_unit_response_consistent_in_local_mode():
    src, drn = _random_sources_and_drones(3)
    wf = _split_field(1.5, 0.2, ix_split=5)
    loc = GaussianPlume(ForwardParams(sigma_v=0.9), wind_field=wf, wind_mode="local")
    g = loc.unit_response(src, drn)
    lg = loc.log_unit_response(src, drn)
    above = g > loc.params.g_floor
    assert above.any() and (~above).any()
    np.testing.assert_allclose(lg[above], np.log(g[above]), rtol=1e-12)
    assert np.all(lg[~above] == np.log(loc.params.g_floor))
    far = loc.log_unit_response(src[:1], np.array([[10.0, 5000.0, 15.0]]))
    assert np.isfinite(far).all()


def _hand_plume(p: ForwardParams, sxy: np.ndarray, U: float, dir_deg: float, drn: np.ndarray) -> np.ndarray:
    """Scalar re-implementation of the plan 4.2 formulas for ONE hypothesis (independent of GaussianPlume)."""
    a = np.deg2rad(dir_deg)
    e = np.array([np.cos(a), np.sin(a)])
    out = []
    for q in drn:
        r = q[:2] - sxy
        d = r @ e
        c = -r[0] * e[1] + r[1] * e[0]
        if d <= 0.0:
            out.append(p.g_floor)
            continue
        sy = np.sqrt(p.sigma0 ** 2 + (p.sigma_v * d / U) ** 2 / (1.0 + d / (2.0 * U * p.T_L)))
        sz = p.sigma_z_ratio * sy
        vert = np.exp(-(q[2] - p.z_s) ** 2 / (2 * sz * sz)) + np.exp(-(q[2] + p.z_s) ** 2 / (2 * sz * sz))
        out.append(max(np.exp(-c * c / (2 * sy * sy)) * vert / (2.0 * np.pi * U * sy * sz), p.g_floor))
    return np.array(out)


def test_7_local_mode_matches_hand_formula_with_distinct_u_and_direction():
    """(7) two hypotheses with DIFFERENT local winds (3 m/s +x vs 1.2 m/s -y): each row of the (N, M) response
    equals an independent scalar evaluation with its own U_i in BOTH sigma_y and the 1/(2 pi U sy sz)
    prefactor and its own unit vector (direction sign: the -y hypothesis sees the drone at -y downwind)."""
    x, y = _lattice()
    uvw = np.zeros((len(Z_LEVELS), NX, NY, 3), np.float32)
    uvw[:, :, : NY // 2, 0] = 3.0                               # y < 0: +x 3 m/s
    uvw[:, :, NY // 2:, 1] = -1.2                               # y >= 0: -y 1.2 m/s
    wf = WindField.from_arrays(x, y, np.array(Z_LEVELS), uvw)
    p = ForwardParams(U=1.68, sigma_v=0.5, wind_dir_deg=0.0)
    loc = GaussianPlume(p, wind_field=wf, wind_mode="local")
    src = np.array([[x[2] + 0.3, y[1] + 0.4], [x[2] + 0.3, y[6] + 0.4]])
    drn = np.array([[src[0, 0] + 100.0, src[0, 1], 15.0],       # downwind of A (+x), crosswind of B
                    [src[1, 0], src[1, 1] - 70.0, 15.0],        # downwind of B (-y), off-plume for A
                    [src[0, 0] + 50.0, src[0, 1] + 4.0, 15.0]])
    lw = loc.local_wind(src)
    u_b = float(np.float32(1.2))                                # the field is float32
    np.testing.assert_allclose(lw.U, [3.0, u_b], rtol=1e-12)
    np.testing.assert_allclose(lw.dir_deg, [0.0, -90.0])
    g = loc.unit_response(src, drn)
    ref = np.array([_hand_plume(p, src[0], 3.0, 0.0, drn), _hand_plume(p, src[1], u_b, -90.0, drn)])
    np.testing.assert_allclose(g, ref, rtol=1e-10)
    assert g[0, 0] > p.g_floor and g[1, 1] > p.g_floor and g[1, 0] == p.g_floor
    # using the global U (1.68) in the prefactor / sigma_y would be off by a factor ~ 3 / 1.68 on row A
    wrong = GaussianPlume(p).unit_response(src[0], drn)[0]
    assert not np.allclose(g[0, 0], wrong[0], rtol=0.05)


def test_constructor_errors_and_global_mode_local_wind():
    with pytest.raises(ValueError):
        GaussianPlume(ForwardParams(), wind_field=None, wind_mode="local")
    with pytest.raises(ValueError):
        GaussianPlume(ForwardParams(), wind_field=_uniform_field(1.0, 0.0), wind_mode="sideways")
    with pytest.raises(ValueError):
        ForwardParams(local_wind_blend=1.5)
    with pytest.raises(ValueError):
        ForwardParams(local_wind_blend=-0.1)
    # global mode: local_wind reports the global wind for every hypothesis (uniform helper API)
    src, _ = _random_sources_and_drones(4)
    glob = GaussianPlume(ForwardParams(U=2.5, wind_dir_deg=45.0))
    lw = glob.local_wind(src)
    np.testing.assert_allclose(lw.U, 2.5)
    np.testing.assert_allclose(lw.dir_deg, 45.0)
    np.testing.assert_allclose(lw.speed, 2.5)
    np.testing.assert_allclose(lw.uv, np.full((N_SRC, 2), 2.5 / np.sqrt(2.0)))
    assert not lw.fallback.any()


def test_blend_between_local_and_global():
    src, drn = _random_sources_and_drones(5)
    wf = _uniform_field(0.0, 2.0)                                # local +y 2 m/s; global +x 2 m/s
    glob = GaussianPlume(ForwardParams(U=2.0, wind_dir_deg=0.0))
    blend0 = GaussianPlume(ForwardParams(U=2.0, wind_dir_deg=0.0, local_wind_blend=0.0), wf, "local")
    np.testing.assert_allclose(blend0.unit_response(src, drn), glob.unit_response(src, drn), rtol=1e-12)
    half = GaussianPlume(ForwardParams(U=2.0, wind_dir_deg=0.0, local_wind_blend=0.5), wf, "local")
    lw = half.local_wind(src)
    np.testing.assert_allclose(lw.vec, [[1.0, 1.0]] * N_SRC, rtol=1e-12)
    np.testing.assert_allclose(lw.U, np.sqrt(2.0), rtol=1e-12)
    np.testing.assert_allclose(lw.dir_deg, 45.0)
    ref = GaussianPlume(ForwardParams(U=np.sqrt(2.0), wind_dir_deg=45.0))
    np.testing.assert_allclose(half.unit_response(src, drn), ref.unit_response(src, drn), rtol=1e-10)


def test_wind_aligned_coords_accepts_per_source_angles():
    rng = np.random.default_rng(6)
    src = rng.uniform(0, 100, (5, 2))
    drn = rng.uniform(-50, 150, (4, 2))
    ang = rng.uniform(0, 360, 5)
    d, c = wind_aligned_coords(src, drn, ang)
    for i in range(5):
        di, ci = wind_aligned_coords(src[i], drn, float(ang[i]))
        np.testing.assert_allclose(d[i], di[0])
        np.testing.assert_allclose(c[i], ci[0])
    # rows must NOT all share one angle: row i with row j's angle differs (guards a broadcast that ignores the array)
    d_swapped, _ = wind_aligned_coords(src, drn, np.roll(ang, 1))
    assert not np.allclose(d, d_swapped)
    with pytest.raises(ValueError):
        wind_aligned_coords(src, drn, ang[:3])
    with pytest.raises(ValueError):
        wind_aligned_coords(src, drn, np.zeros((5, 1)))


def test_local_grid_matches_pairwise_scalar_evaluation():
    """The (N, M) local-mode broadcast equals evaluating each hypothesis alone with its own (U_i, dir_i)."""
    src, drn = _random_sources_and_drones(7)
    x, y = _lattice()
    uvw = np.zeros((len(Z_LEVELS), NX, NY, 3), np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    uvw[..., 0] = 1.0 + 0.05 * xx                                 # spatially varying, all speeds > FWD_U_MIN
    uvw[..., 1] = 0.5 - 0.04 * yy
    wf = WindField.from_arrays(x, y, np.array(Z_LEVELS), uvw)
    loc = GaussianPlume(ForwardParams(), wind_field=wf, wind_mode="local")
    g = loc.unit_response(src, drn)
    lw = loc.local_wind(src)
    assert np.all(lw.speed > config.FWD_U_MIN)
    ref = np.array([GaussianPlume(ForwardParams(U=float(lw.U[i]), wind_dir_deg=float(lw.dir_deg[i])))
                    .unit_response(src[i], drn)[0] for i in range(N_SRC)])
    np.testing.assert_allclose(g, ref, rtol=1e-12)
    np.testing.assert_allclose(loc.sigma_y(np.array([[100.0]] * N_SRC), lw.U[:, None])[:, 0],
                               [GaussianPlume(ForwardParams(U=float(u))).sigma_y(np.array([100.0]))[0] for u in lw.U])