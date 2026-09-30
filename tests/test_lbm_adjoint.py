"""Unit tests for pf/lbm_adjoint.py (plan 4.2b option B-1, docs/lbm_forward_model.md section 7) on synthetic
grids only (no raw data): reciprocity, mass balance, walls, Gaussian far field, rotation invariance, the PF
model wrapper, from_data on synthetic WindField / ObstacleMap, and timing on the real-size grid."""
import time

import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.field.wind import WindField
from srcloc_env.pf.lbm_adjoint import AdjointParams, AdvectionDiffusionOperator, LbmAdjointModel
from srcloc_env.pf.particle_filter import RBPF
from srcloc_env.preprocess.gridder import SlabGrid

# test-local grid / wind values (not physics constants)
U_TEST = 1.68
K_TEST = config.ADJ_SIGMA_V_REF ** 2 * config.FWD_T_L      # Taylor far-field K = sigma_v^2 T_L (R15)
RECIP_RTOL = 1e-9
MASS_ATOL = 1e-8
GAUSS_D_M = 150.0
GAUSS_SIGMA_RTOL = 0.20


def uniform_wind(ny: int, nx: int, u: float, v: float = 0.0) -> np.ndarray:
    uv = np.zeros((ny, nx, 2))
    uv[..., 0], uv[..., 1] = u, v
    return uv


def sheared_wind(grid: SlabGrid) -> np.ndarray:
    """Smooth, non-uniform (and slightly divergent) wind for the algebraic identity tests."""
    xx, yy = np.meshgrid(grid.x_centres, grid.y_centres)
    return np.stack([1.0 + 0.5 * np.sin(yy / 50.0), 0.3 * np.cos(xx / 70.0)], axis=-1)


def build(grid: SlabGrid, uv: np.ndarray, blocked: np.ndarray | None = None, **kw) -> AdvectionDiffusionOperator:
    blocked = np.zeros((grid.ny, grid.nx), dtype=bool) if blocked is None else blocked
    return AdvectionDiffusionOperator.from_arrays(grid, uv, blocked, AdjointParams(**kw)).factorize()


def test_params_defaults_and_validation():
    p = AdjointParams()
    assert (p.K, p.lam, p.h_layer, p.sigma0, p.z) == (config.ADJ_K_DEFAULT, config.ADJ_LAMBDA_DEFAULT,
                                                     config.ADJ_H_LAYER, config.RELEASE_SIGMA_XY, config.DRONE_Z)
    assert p.wind_band is None
    with pytest.raises(ValueError):
        AdjointParams(K=-1.0)
    with pytest.raises(ValueError):
        AdjointParams(wind_band=(20.0, 10.0))
    with pytest.raises(Exception):
        p.K = 1.0                                        # frozen


def test_index_maps_and_footprint_columns():
    grid = SlabGrid(x0=0.0, y0=0.0, nx=40, ny=30, res=5.0)
    blocked = np.zeros((30, 40), dtype=bool)
    blocked[10:14, 20:23] = True
    op = AdvectionDiffusionOperator.from_arrays(grid, uniform_wind(30, 40, U_TEST), blocked)
    assert op.n_free == 30 * 40 - 12
    assert np.all(op.cell_to_unk[op.unk_to_cell] == np.arange(op.n_free))
    assert np.all(op.cell_to_unk.reshape(30, 40)[blocked] == -1)
    # footprint columns sum to 1, the forward source vector at a cell centre equals the F column
    colsum = np.asarray(op.F.sum(axis=0)).ravel()
    assert np.allclose(colsum, 1.0)
    s = 517
    b = op.source_vector(op.free_cell_centres()[s])
    assert np.allclose(b, op.F[:, s].toarray().ravel(), atol=1e-15)
    assert op.footprint_radius_cells == int(np.ceil(config.ADJ_FOOTPRINT_TRUNC_SIGMA * config.RELEASE_SIGMA_XY / 5.0))
    # A: positive diagonal, non-positive off-diagonal, column sums = lam area + outflow (M-matrix structure)
    d = op.A.diagonal()
    assert np.all(d > 0.0)
    off = op.A - op.A.multiply(np.eye(op.n_free))
    assert off.max() <= 0.0
    colsum_a = np.asarray(op.A.sum(axis=0)).ravel()
    assert np.allclose(colsum_a, op.params.lam * 25.0 + op.outflow_coef)
    # separable footprint correlation == sparse F^T product
    op.factorize()
    phi = op.luT.solve(op.receptor_vector(np.array([101.0, 52.0])))
    assert np.allclose(op.apply_footprint_transpose(phi), op.FT @ phi, rtol=1e-12, atol=0.0)
    p = np.array([77.0, 33.0])
    assert np.allclose(op.solve_adjoint(p), op.solve_adjoint(p, sparse_footprint=True), rtol=1e-12, atol=0.0)
    assert op.n_components == 1


def test_reciprocity_forward_vs_adjoint():
    """(1) solve_forward(s)[p] == solve_adjoint(p)[s] for random free cells s, p; sheared wind + a wall."""
    rng = np.random.default_rng(1)
    grid = SlabGrid(x0=0.0, y0=0.0, nx=40, ny=30, res=5.0)
    blocked = np.zeros((30, 40), dtype=bool)
    blocked[8:12, 18:21] = True
    op = build(grid, sheared_wind(grid), blocked)
    cc = op.free_cell_centres()
    ny, nx = grid.ny, grid.nx
    for _ in range(6):
        s, p = rng.integers(0, op.n_free, 2)
        fwd = op.solve_forward(cc[s])
        adj = op.solve_adjoint(cc[p])
        a = fwd.reshape(-1)[op.unk_to_cell[p]]
        b = adj.reshape(-1)[op.unk_to_cell[s]]
        assert a > 0.0 and abs(a - b) <= RECIP_RTOL * max(a, b)
        # continuous receptor position: exact too (same bilinear weights on both sides)
        pc = cc[p] + rng.uniform(-2.0, 2.0, 2)
        a2 = op.interpolate(fwd, pc)[0]
        b2 = op.solve_adjoint(pc).reshape(-1)[op.unk_to_cell[s]]
        assert abs(a2 - b2) <= RECIP_RTOL * max(a2, b2)
    assert fwd.shape == (ny, nx) and adj.shape == (ny, nx)
    assert np.all(fwd[blocked] == 0.0) and np.all(adj[blocked] == 0.0)
    assert fwd.min() >= 0.0 and adj.min() >= 0.0


def test_mass_balance_uniform_and_sheared():
    """(2) lam sum(C) area + outflow == 1 (uniform wind / sheared wind, no walls)."""
    grid = SlabGrid(x0=0.0, y0=0.0, nx=40, ny=30, res=5.0)
    for uv in (uniform_wind(30, 40, U_TEST), sheared_wind(grid)):
        op = build(grid, uv, lam=0.02)
        mb = op.mass_balance(op.solve_forward(np.array([60.0, 77.0])))
        assert abs(mb["loss"] + mb["outflow"] - 1.0) < MASS_ATOL
        assert mb["source"] == 1.0 and mb["loss"] > 0.0 and mb["outflow"] > 0.0
        assert abs(mb["residual"]) < MASS_ATOL


def test_wall_column_blocks_transport():
    """(3) a wall across the whole y-extent at x = 100 m, source at x = 50, wind +x: downstream C == 0."""
    grid = SlabGrid(x0=0.0, y0=0.0, nx=40, ny=30, res=5.0)
    blocked = np.zeros((30, 40), dtype=bool)
    blocked[:, 20] = True                                    # cells covering x in [100, 105)
    op = build(grid, uniform_wind(30, 40, U_TEST), blocked, lam=0.02)
    fwd = op.solve_forward(np.array([50.0, 75.0]))
    assert np.all(fwd[:, 21:] == 0.0)
    assert np.all(fwd[:, 20] == 0.0)
    assert fwd[:, :20].max() > 0.0
    mb = op.mass_balance(fwd)
    assert mb["outflow"] == 0.0 and abs(mb["loss"] - 1.0) < MASS_ATOL   # everything is lost in the enclosure
    # adjoint from the downstream side sees nothing upstream (the footprint does not cross the wall)
    assert op.n_components == 2
    adj = op.solve_adjoint(np.array([150.0, 75.0]))
    assert np.all(adj[:, :21] == 0.0) and adj[:, 21:].max() > 0.0
    # a source right next to the wall (within the footprint box) still injects nothing behind it
    fwd2 = op.solve_forward(np.array([97.5, 75.0]))
    assert np.all(fwd2[:, 20:] == 0.0) and abs(op.mass_balance(fwd2)["loss"] - 1.0) < MASS_ATOL
    assert np.isclose(op.source_vector(np.array([97.5, 72.5])).sum(), 1.0)


def test_uniform_wind_far_field_is_gaussian():
    """(4) uniform wind U along +x, K = sigma_v^2 T_L, lam = 0, no walls: the crosswind profile at d = 150 m is
    Gaussian with sigma^2 = sigma0^2 + 2 K d / U.  The upwind numerical diffusion ~ U res / 2 acts along the
    flow only (the crosswind advection is zero), so no correction to K is needed for the crosswind spread."""
    grid = SlabGrid(x0=0.0, y0=-150.0, nx=60, ny=60, res=5.0)
    op = build(grid, uniform_wind(60, 60, U_TEST), K=K_TEST, lam=0.0)
    xs = 50.0
    fwd = op.solve_forward(np.array([xs, 0.0]))
    ix = int(np.argmin(np.abs(grid.x_centres - (xs + GAUSS_D_M))))
    d = grid.x_centres[ix] - xs
    prof = fwd[:, ix]
    y = grid.y_centres
    var_fit = float((prof * y ** 2).sum() / prof.sum())
    var_exp = config.RELEASE_SIGMA_XY ** 2 + 2.0 * K_TEST * d / U_TEST
    assert abs(np.sqrt(var_fit) / np.sqrt(var_exp) - 1.0) < GAUSS_SIGMA_RTOL
    # shape: normalised profile vs the Gaussian with the fitted variance
    gauss = np.exp(-0.5 * y ** 2 / var_fit)
    assert np.max(np.abs(prof / prof.max() - gauss)) < 0.05
    # centreline decay 1/sqrt(d): C(d1)/C(d2) = sqrt(var(d2)/var(d1)) (far field, slender plume)
    ix2 = int(np.argmin(np.abs(grid.x_centres - (xs + 2 * GAUSS_D_M))))
    d2 = grid.x_centres[ix2] - xs
    ratio = fwd[30, ix] / fwd[30, ix2]
    assert abs(ratio / np.sqrt((config.RELEASE_SIGMA_XY ** 2 + 2 * K_TEST * d2 / U_TEST) / var_exp) - 1.0) < 0.1
    mb = op.mass_balance(fwd)
    assert mb["loss"] == 0.0 and abs(mb["outflow"] - 1.0) < MASS_ATOL


def test_rotation_invariance_transposed_field():
    """(5) wind along +y with the source coordinates swapped gives the transposed field (square grid)."""
    grid = SlabGrid(x0=0.0, y0=0.0, nx=40, ny=40, res=5.0)
    blocked = np.zeros((40, 40), dtype=bool)
    blocked[12:15, 25:28] = True
    op_x = build(grid, uniform_wind(40, 40, U_TEST, 0.0), blocked)
    op_y = build(grid, uniform_wind(40, 40, 0.0, U_TEST), blocked.T.copy())
    f_x = op_x.solve_forward(np.array([60.0, 100.0]))
    f_y = op_y.solve_forward(np.array([100.0, 60.0]))
    assert np.allclose(f_y, f_x.T, rtol=1e-12, atol=1e-20)
    a_x = op_x.solve_adjoint(np.array([160.0, 110.0]))
    a_y = op_y.solve_adjoint(np.array([110.0, 160.0]))
    assert np.allclose(a_y, a_x.T, rtol=1e-12, atol=1e-20)


def test_model_unit_response_shapes_cache_and_forward_agreement():
    """(6) LbmAdjointModel: (N, M) shapes, agreement with solve_forward for sources at cell centres, LRU."""
    rng = np.random.default_rng(3)
    grid = SlabGrid(x0=0.0, y0=0.0, nx=40, ny=30, res=5.0)
    blocked = np.zeros((30, 40), dtype=bool)
    blocked[5:9, 10:13] = True
    op = build(grid, sheared_wind(grid), blocked)
    model = LbmAdjointModel(op, max_cached=2)
    cc = op.free_cell_centres()
    src_idx = rng.integers(0, op.n_free, 5)
    src = cc[src_idx]
    drn_idx = rng.integers(0, op.n_free, 3)
    drn = np.column_stack([cc[drn_idx], np.full(3, config.DRONE_Z)])
    g = model.unit_response(src, drn)
    assert g.shape == (5, 3) and np.all(np.isfinite(g)) and np.all(g >= config.FWD_G_FLOOR)
    for n in range(5):
        fwd = op.solve_forward(src[n])
        for m in range(3):
            expect = max(fwd.reshape(-1)[op.unk_to_cell[drn_idx[m]]], config.FWD_G_FLOOR)
            assert np.isclose(g[n, m], expect, rtol=1e-9)
    assert model.misses == 3 and model.n_cached == 2 and model.hits == 0   # LRU of 2 psi fields
    model.unit_response(src, drn[-1:])                              # most recent drone: cache hit
    assert model.hits == 1 and model.misses == 3
    model.unit_response(src, drn[:1])                               # evicted (capacity 2): miss again
    assert model.misses == 4
    assert np.allclose(model.log_unit_response(src, drn), np.log(g))
    assert model.unit_response(src[0], drn[0]).shape == (1, 1)
    with pytest.raises(ValueError):
        model.unit_response(src, drn[:, :2])
    # off-centre source: interpolation between the four neighbouring centred footprints (approximation)
    g_off = model.unit_response(src[:1] + np.array([[1.0, -1.5]]), drn)
    assert np.all(g_off > 0.0) and np.allclose(g_off, g[:1], rtol=0.5)
    # RBPF accepts the model unchanged (grid mode, one update)
    pf = RBPF(model, n_particles=50, prior_x=(2.5, 197.5), prior_y=(2.5, 147.5), rng=rng)
    ll = pf.update(5, drn[0])
    assert np.isfinite(ll) and np.isfinite(pf.logw).all()


def test_from_data_synthetic_wind_and_obstacles():
    """from_data: wind sampled at the cell centres (single level and band mean) and walls from the roof map."""
    x = np.arange(0.0, 102.5, 2.5)
    y = np.arange(0.0, 102.5, 2.5)
    z_levels = np.array([11.25, 13.75, 16.25])
    uvw = np.zeros((3, x.size, y.size, 3), dtype=np.float32)
    uvw[0, ..., 0], uvw[1, ..., 0], uvw[2, ..., 0] = 1.0, 3.0, 5.0
    wf = WindField.from_arrays(x, y, z_levels, uvw)
    occ = np.zeros((51, 51), dtype=bool)
    hmap = np.zeros((51, 51), dtype=np.float32)
    occ[20:30, 10:20] = True                                       # x 40-58, y 20-38 (2 m raster [ix, iy])
    hmap[20:30, 10:20] = 30.0
    occ[40:45, 40:45] = True
    hmap[40:45, 40:45] = 10.0                                      # low roof: free at 15 m
    om = ObstacleMap.from_arrays(0.0, 0.0, 2.0, occ, hmap, domain_x=(0.0, 100.0), domain_y=(0.0, 100.0))
    grid = SlabGrid(x0=0.0, y0=0.0, nx=20, ny=20, res=5.0)
    op = AdvectionDiffusionOperator.from_data(AdjointParams(z=15.0), wf, om, grid)
    assert op.blocked.shape == (20, 20)
    cx, cy = np.meshgrid(grid.x_centres, grid.y_centres)
    expect = (cx >= 39.0) & (cx < 59.0) & (cy >= 19.0) & (cy < 39.0)
    assert np.array_equal(op.blocked, expect)
    assert np.allclose(op.uv[~op.blocked][:, 0], 4.0)               # z = 15 between 13.75 (3) and 16.25 (5)
    assert np.allclose(op.uv[..., 1], 0.0)
    op_band = AdvectionDiffusionOperator.from_data(AdjointParams(z=15.0, wind_band=(10.0, 15.0)), wf, om, grid)
    assert np.allclose(op_band.uv[~op_band.blocked][:, 0], 2.0)     # mean of levels 11.25 (1) and 13.75 (3)
    with pytest.raises(ValueError):
        AdvectionDiffusionOperator.from_data(AdjointParams(z=15.0, wind_band=(30.0, 40.0)), wf, om, grid)
    fwd = op.factorize().solve_forward(np.array([20.0, 30.0]))
    assert np.all(fwd[op.blocked] == 0.0) and fwd.max() > 0.0


def test_real_size_grid_timing():
    """(7) 200 x 207 grid with ~20 % blocked cells, uniform wind: factorisation < ADJ_FACTORIZE_TIME_TARGET_S,
    adjoint solve median < ADJ_TEST_SOLVE_TIME_LOOSE_S (the 20 ms target is reported by validate_adjoint)."""
    rng = np.random.default_rng(7)
    grid = SlabGrid()
    blocked = np.zeros((grid.ny, grid.nx), dtype=bool)
    while blocked.mean() < 0.2:
        iy, ix = rng.integers(0, grid.ny), rng.integers(0, grid.nx)
        h, w = rng.integers(2, 12, 2)
        blocked[iy:iy + h, ix:ix + w] = True
    uv = uniform_wind(grid.ny, grid.nx, U_TEST)
    t0 = time.perf_counter()
    op = AdvectionDiffusionOperator.from_arrays(grid, uv, blocked, AdjointParams()).factorize()
    build_s = time.perf_counter() - t0
    assert grid.nx == config.SLAB_NX and grid.ny == config.SLAB_NY and 0.15 < blocked.mean() < 0.3
    assert build_s < config.ADJ_FACTORIZE_TIME_TARGET_S
    timing = op.time_adjoint_solves(20, rng)
    print(f"real-size grid: n_free {op.n_free}, assembly {op.assembly_seconds:.2f} s, factorize "
          f"{op.factorize_seconds:.2f} s, adjoint solve median {timing['median_s'] * 1e3:.1f} ms "
          f"(target {config.ADJ_SOLVE_TIME_TARGET_S * 1e3:.0f} ms), p99 {timing['p99_s'] * 1e3:.1f} ms")
    assert timing["median_s"] < config.ADJ_TEST_SOLVE_TIME_LOOSE_S
    mb = op.mass_balance(op.solve_forward(op.free_cell_centres()[rng.integers(0, op.n_free)]))
    assert abs(mb["residual"]) < MASS_ATOL

def test_receptor_straddling_two_components_and_singular_guard():
    """(review B1 gap test) a diagonal wall splits the grid into two 4-connected components; a receptor at a
    grid vertex has bilinear weight on one free cell of each component, so A^-T e_p is nonzero in BOTH
    components and F^T must be applied per component: separable path == sparse FT, reciprocity holds for a
    source in either component.  Also: lam = 0 on an enclosed component raises a clear ValueError."""
    grid = SlabGrid(x0=0.0, y0=0.0, nx=24, ny=24, res=5.0)
    blocked = np.zeros((24, 24), dtype=bool)
    blocked[np.arange(24), np.arange(24)] = True                  # diagonal (iy == ix): two triangles
    op = build(grid, sheared_wind(grid), blocked, lam=0.02)
    assert op.n_components == 2
    corner = np.array([grid.x0 + 10 * grid.res, grid.y0 + 10 * grid.res])   # vertex shared by cells (9,9)..(10,10)
    unk, w = op.bilinear_weights(corner)
    assert np.count_nonzero(w) == 2 and np.allclose(w[w > 0], 0.5)
    comps = op.component.ravel()[op.unk_to_cell[unk[0][unk[0] >= 0]]]
    assert set(comps.tolist()) == {1, 2}
    psi = op.solve_adjoint(corner)
    psi_ref = op.solve_adjoint(corner, sparse_footprint=True)
    assert np.allclose(psi, psi_ref, rtol=1e-12, atol=0.0)
    assert psi[op.component == 1].max() > 0.0 and psi[op.component == 2].max() > 0.0
    cc = op.free_cell_centres()
    for s in (int(np.flatnonzero(op.component.ravel()[op.unk_to_cell] == c)[7]) for c in (1, 2)):
        a = op.interpolate(op.solve_forward(cc[s]), corner)[0]
        b = psi.reshape(-1)[op.unk_to_cell[s]]
        assert a > 0.0 and abs(a - b) <= RECIP_RTOL * max(a, b)
    # lam = 0 with a fully enclosed pocket: singular operator -> ValueError (not SuperLU's RuntimeError)
    ring = np.zeros((24, 24), dtype=bool)
    ring[4, 4:12] = ring[11, 4:12] = True
    ring[4:12, 4] = ring[4:12, 11] = True
    with pytest.raises(ValueError, match="lam = 0"):
        build(grid, uniform_wind(24, 24, U_TEST), ring, lam=0.0)
    build(grid, uniform_wind(24, 24, U_TEST), ring, lam=0.02)   # lam > 0: fine
