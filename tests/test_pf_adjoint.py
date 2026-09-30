"""Unit tests for the PF <-> LBM adjoint connection (plan 4.2b option B-3, docs/lbm_forward_model.md section 5
step 4; D5-2) on a synthetic 40 x 30 operator only (AdvectionDiffusionOperator.from_arrays, uniform wind + one
wall): RBPF with LbmAdjointModel converges to a synthetic source (MAP error < config.PF_ADJ_TEST_MAP_ERROR_CELLS
cells after config.PF_ADJ_TEST_N_MEASUREMENTS measurements), unit_response shape / LRU behaviour along a path,
RBPF unchanged (duck-typed forward model, grid and nb paths), and the lawnmower path generator of
scripts/validate_pf_adjoint.py on a synthetic ObstacleMap.  References R3, R5, R17."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.pf.lbm_adjoint import AdjointParams, AdvectionDiffusionOperator, LbmAdjointModel
from srcloc_env.pf.particle_filter import RBPF
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.scripts.validate_pf_adjoint import SyntheticTruth, lawnmower_path, two_drone_paths
from srcloc_env.sensor.detector import Detector

# test-local grid / wind values (not physics constants)
NX, NY, RES = 40, 30, 5.0                     # 200 m x 150 m
U_TEST = 1.68
TRUE_XY = (62.0, 78.0)                        # upwind of the wall column at x 100-110 m
N_PARTICLES_TEST = 400
COUNTS_MAX_TEST = 300.0                       # kappa_true chosen so the largest expected count is ~300


def make_grid() -> SlabGrid:
    return SlabGrid(x0=0.0, y0=0.0, nx=NX, ny=NY, res=RES)


def make_operator() -> AdvectionDiffusionOperator:
    grid = make_grid()
    uv = np.zeros((NY, NX, 2))
    uv[..., 0] = U_TEST
    blocked = np.zeros((NY, NX), dtype=bool)
    blocked[8:22, 20:22] = True                      # one wall: x 100-110 m, y 40-110 m
    return AdvectionDiffusionOperator.from_arrays(grid, uv, blocked, AdjointParams()).factorize()


def sweep_path(op: AdvectionDiffusionOperator, n: int) -> np.ndarray:
    """(n, 2) distinct free cell centres: a boustrophedon over the grid downwind of the source (x >= 40 m)."""
    g = op.grid
    pts = []
    for j, ix in enumerate(range(8, g.nx, 2)):
        ys = range(1, g.ny, 3) if j % 2 == 0 else range(g.ny - 2, 0, -3)
        for iy in ys:
            if not op.blocked[iy, ix]:
                pts.append((g.x_centres[ix], g.y_centres[iy]))
    pts = np.asarray(pts)
    assert pts.shape[0] >= n
    return pts[:n]


def test_rbpf_converges_with_lbm_adjoint_model():
    """MAP error < PF_ADJ_TEST_MAP_ERROR_CELLS cells after PF_ADJ_TEST_N_MEASUREMENTS measurements (truth =
    solve_forward(true) x kappa_true, counts via Detector; no model mismatch)."""
    op = make_operator()
    model = LbmAdjointModel(op)
    det = Detector()
    field = op.solve_forward(np.array(TRUE_XY))
    kappa_true = COUNTS_MAX_TEST / field.max()
    truth = SyntheticTruth(op, TRUE_XY, kappa_true, det)
    rng = np.random.default_rng(11)
    path = sweep_path(op, config.PF_ADJ_TEST_N_MEASUREMENTS)
    counts = truth.counts(path, rng)
    assert counts.max() > 5 * det.background * det.T          # the sweep sees the plume
    pf = RBPF(model, n_particles=N_PARTICLES_TEST, kappa_ref=kappa_true, mode="grid", obstacles=None,
              prior_x=(0.0, NX * RES), prior_y=(0.0, NY * RES), rng=np.random.default_rng(12))
    e0 = pf.entropy_xy()
    for k in range(path.shape[0]):
        m = pf.update(int(counts[k]), np.array([path[k, 0], path[k, 1], config.DRONE_Z]))
        assert np.isfinite(m)
    err = np.hypot(*(pf.map_estimate() - np.asarray(TRUE_XY)))
    assert err < config.PF_ADJ_TEST_MAP_ERROR_CELLS * RES, err
    assert np.hypot(*(pf.mean() - np.asarray(TRUE_XY))) < 2.0 * config.PF_ADJ_TEST_MAP_ERROR_CELLS * RES
    assert pf.entropy_xy() < 0.5 * e0
    q = pf.posterior_kappa_quantiles((0.05, 0.5, 0.95))
    assert q[0] < kappa_true < q[2]                              # kappa_true inside the 90 % interval
    assert 0.5 < q[1] / kappa_true < 2.0
    assert model.misses == path.shape[0] and model.hits == 0     # every measurement at a new receptor


def test_unit_response_shape_and_lru_along_path():
    """(N, M) shapes; a path of distinct receptors misses once each, revisiting the cached tail hits."""
    op = make_operator()
    model = LbmAdjointModel(op, max_cached=8)
    rng = np.random.default_rng(5)
    src = np.column_stack([rng.uniform(5.0, 195.0, 25), rng.uniform(5.0, 145.0, 25)])
    path = sweep_path(op, 12)
    drn = np.column_stack([path, np.full(path.shape[0], config.DRONE_Z)])
    g = model.unit_response(src, drn)
    assert g.shape == (25, 12) and np.all(np.isfinite(g)) and np.all(g >= config.FWD_G_FLOOR)
    assert model.misses == 12 and model.hits == 0 and model.n_cached == 8
    g2 = model.unit_response(src, drn[-8:])                     # cached tail: all hits
    assert model.hits == 8 and model.misses == 12
    assert np.array_equal(g2, g[:, -8:])
    model.unit_response(src, drn[:1])                           # evicted head: a miss again
    assert model.misses == 13
    # the response of a hypothesis at a cell centre equals the forward solution there (reciprocity)
    cc = op.free_cell_centres()
    s = 333
    fwd = op.solve_forward(cc[s])
    gs = model.unit_response(cc[s], drn)[0]
    expect = np.maximum(op.interpolate(fwd, path), config.FWD_G_FLOOR)
    assert np.allclose(gs, expect, rtol=1e-9)
    # one update of the RBPF uses exactly unit_response(xy, drone) (duck-typed forward, particle_filter unchanged)
    pf = RBPF(model, n_particles=30, prior_x=(0.0, 200.0), prior_y=(0.0, 150.0), rng=np.random.default_rng(1))
    calls = []
    orig = model.unit_response

    def spy(source_xy, drone_xyz):
        out = orig(source_xy, drone_xyz)
        calls.append((np.array(source_xy).shape, np.array(drone_xyz).shape, out.shape))
        return out

    model.unit_response = spy
    pf.update(25, np.array([150.0, 40.0, config.DRONE_Z]))
    assert calls == [((30, 2), (1, 3), (30, 1))]
    assert np.isclose(np.exp(pf.logw).sum(), 1.0)


def test_rbpf_paths_unchanged_grid_and_nb():
    """RBPF grid and nb (b = 0) paths both accept LbmAdjointModel; the grid path with a fine Gamma prior matches
    the NB closed form on one update (T1-1 style check, R6) - no change in particle_filter.py needed."""
    op = make_operator()
    model = LbmAdjointModel(op)
    rng = np.random.default_rng(3)
    xy = np.column_stack([rng.uniform(5.0, 195.0, 40), rng.uniform(5.0, 145.0, 40)])
    kw = dict(n_particles=40, prior_x=(0.0, 200.0), prior_y=(0.0, 150.0), eps_mix=0.0, background=0.0,
              kappa_ref=1e4, resample_frac=0.0)
    nb = RBPF(model, mode="nb", rng=np.random.default_rng(0), **kw)
    gr = RBPF(model, mode="grid", kappa_prior="gamma", grid_decades=6.0, n_grid=2000,
              rng=np.random.default_rng(0), **kw)
    nb.xy = xy.copy()
    gr.xy = xy.copy()
    drone = np.array([130.0, 70.0, config.DRONE_Z])
    m_nb = nb.update(7, drone)
    m_gr = gr.update(7, drone)
    assert np.isfinite(m_nb) and np.isfinite(m_gr)
    assert np.allclose(nb.last_loglik, gr.last_loglik, atol=1e-3)
    assert abs(m_nb - m_gr) < 1e-3


def test_lawnmower_path_on_synthetic_obstacles():
    """Waypoints are free, distinct, inside the prior box, start >= the requested distance, advance DRONE_STEP_M
    in x per step, bounce at the prior edge, and blocked candidates are skipped (not visited)."""
    res = 2.0
    nx = ny = 120                                             # 240 m x 240 m raster
    occ = np.zeros((nx, ny), dtype=bool)
    hmap = np.zeros((nx, ny), dtype=np.float32)
    occ[50:60, 40:80] = True                                  # building x 100-118 m, y 80-158 m, 30 m tall
    hmap[occ] = 30.0
    om = ObstacleMap.from_arrays(0.0, 0.0, res, occ, hmap, domain_x=(0.0, 238.0), domain_y=(0.0, 238.0))
    prior_x, prior_y = (10.0, 230.0), (10.0, 230.0)
    n = 60
    path, info = lawnmower_path(om, 200.0, 120.0, n, step_m=config.DRONE_STEP_M, sweep_width=60.0,
                                direction=-1.0, prior_x=prior_x, prior_y=prior_y)
    assert path.shape == (n, 2)
    assert np.all(om.is_free(path, config.DRONE_Z))
    assert np.all((path[:, 0] >= prior_x[0]) & (path[:, 0] <= prior_x[1]))
    assert np.all((path[:, 1] >= 90.0 - 1e-9) & (path[:, 1] <= 150.0 + 1e-9))
    assert np.all(np.any(np.diff(path, axis=0) != 0.0, axis=1))   # consecutive waypoints differ
    assert len({tuple(np.round(p, 6)) for p in path}) >= 0.9 * n  # a few coincidences after the bounce are fine
    assert info["n_skipped"] > 0 and info["n_hover"] == 0 and info["n_bounces"] == 1
    assert np.allclose(np.abs(np.diff(path[:, 0]))[np.abs(np.diff(path[:, 0])) > 0] % config.DRONE_STEP_M, 0.0)
    assert path[:, 0].min() >= prior_x[0] and path[:, 0].max() <= 200.0
    # the blocked candidates were dropped: no waypoint inside the building at 15 m
    inside = (path[:, 0] >= 100.0) & (path[:, 0] <= 118.0) & (path[:, 1] >= 80.0) & (path[:, 1] <= 158.0)
    assert not inside.any()
    # two drones: adjacent bands centred on y_s +- width/2, start PF_ADJ_START_DOWNWIND_M downwind
    with pytest.raises(ValueError):
        lawnmower_path(om, 110.0, 120.0, 5, sweep_width=1.0, prior_x=(105.0, 115.0), prior_y=prior_y)
    paths, infos = two_drone_paths(None, (60.0, 120.0), 30, prior_x=(0.0, 400.0), prior_y=(0.0, 400.0))
    assert paths.shape == (30, 2, 2) and len(infos) == 2
    assert np.allclose(paths[0, :, 0], 60.0 + config.PF_ADJ_START_DOWNWIND_M)
    # bands [y_s - w, y_s] and [y_s, y_s + w]; each drone starts at its outer band edge and both reach y_s together
    assert paths[0, 0, 1] == pytest.approx(20.0) and paths[0, 1, 1] == pytest.approx(220.0)
    assert paths[5, 0, 1] > 20.0 and paths[5, 1, 1] < 220.0
    k_meet = int(config.PF_ADJ_SWEEP_WIDTH_M / config.DRONE_STEP_M)
    assert paths[k_meet, 0, 1] == pytest.approx(120.0) and paths[k_meet, 1, 1] == pytest.approx(120.0)