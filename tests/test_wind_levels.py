"""D13: wind-knowledge levels W0 (uniform wind + Gaussian plume) and W1 (potential flow + adjoint) on the synthetic scene."""
from __future__ import annotations

import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.baselines.policies import RandomPolicy
from srcloc_env.env.source_env import Scene, SourceLocEnv
from srcloc_env.eval.episodes import make_episode_list
from srcloc_env.eval.run_eval import run_episode
from srcloc_env.field.wind_models import (build_wind_field, potential_flow_diagnostics, potential_flow_field, potential_flow_uv,
                                          uniform_wind_field)
from srcloc_env.pf.forward_model import GaussianPlume
from srcloc_env.pf.lbm_adjoint import AdjointParams, AdvectionDiffusionOperator, LbmAdjointModel
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.rl.rollout import make_train_env
from tests.test_source_env import DOMAIN_X, DOMAIN_Y, SOURCES, SyntheticBackend, _obstacles

GRID = SlabGrid(0.0, -75.0, 40, 28, 5.0)
KW = dict(sources=(1, 2), n_particles=300, prior_x=DOMAIN_X, prior_y=DOMAIN_Y, start_min_dist=50.0, frame_range=(400, 599))
PARAMS = AdjointParams(K=16.0, lam=0.005)


def test_uniform_field_is_constant_everywhere():
    wf = uniform_wind_field(2.0, 30.0, x_range=DOMAIN_X, y_range=DOMAIN_Y)
    pts = np.array([[DOMAIN_X[0], DOMAIN_Y[0]], [100.0, 0.0], [DOMAIN_X[1], DOMAIN_Y[1]], [5000.0, -5000.0]])
    uv = wf.uv_at(pts, 15.0)
    assert np.allclose(uv, [2.0 * np.cos(np.deg2rad(30.0)), 2.0 * np.sin(np.deg2rad(30.0))])


def test_potential_flow_divergence_free_symmetric_and_uniform_without_obstacles():
    ny, nx, res = 30, 50, 5.0
    blocked = np.zeros((ny, nx), bool)
    blocked[10:18, 20:28] = True
    x, y = res * (np.arange(nx) + 0.5), -75.0 + res * (np.arange(ny) + 0.5)
    uv = potential_flow_uv(blocked, res, 1.68, 0.0, x, y)
    d = potential_flow_diagnostics(uv, blocked, res, 1.68, 0.0)
    assert d["div_rms"] < 0.05 * d["div_scale"]                      # mass consistent away from the obstacle corners
    assert np.all(uv[blocked] == 0.0)
    assert abs(d["inflow_edge_mean_speed"] - 1.68) < 0.2             # far field ~ U
    assert d["speed_max_free"] > 1.68                                 # flow accelerates around the block
    uv_m = potential_flow_uv(blocked[::-1], res, 1.68, 0.0, x, y)   # y-mirror symmetry (reflected scenes)
    assert np.allclose(uv_m[::-1, :, 0], uv[..., 0]) and np.allclose(uv_m[::-1, :, 1], -uv[..., 1])
    uv0 = potential_flow_uv(np.zeros((ny, nx), bool), res, 1.68, 45.0, x, y)
    assert np.allclose(uv0[..., 0], 1.68 * np.cos(np.pi / 4)) and np.allclose(uv0[..., 1], 1.68 * np.sin(np.pi / 4))


def test_w1_field_reproduces_its_array_at_the_cell_centres_and_blocks_buildings():
    om = _obstacles()
    wf = potential_flow_field(om, GRID, 1.5, 0.0)
    blocked = AdvectionDiffusionOperator.blocked_at_cells(om, GRID, config.DRONE_Z)
    uv = AdvectionDiffusionOperator.wind_at_cells(wf, GRID, AdjointParams(z=config.DRONE_Z))
    expect = np.stack([wf.uvw[0, :, :, 0].T, wf.uvw[0, :, :, 1].T], axis=-1)
    assert np.allclose(uv[~blocked], expect[~blocked], atol=1e-5)
    assert np.all(uv[blocked] == 0.0)
    assert blocked.any()


def test_scene_w0_builds_a_plume_that_mirrors_with_the_wind():
    wf = uniform_wind_field(1.5, 20.0, x_range=DOMAIN_X, y_range=DOMAIN_Y)
    sc = Scene.build(wf, _obstacles(), SyntheticBackend(), SOURCES, PARAMS, GRID, wind_level="W0")
    assert isinstance(sc.model, GaussianPlume) and sc.wind_level == "W0"
    assert np.isclose(sc.model.params.U, 1.5) and np.isclose(sc.model.params.wind_dir_deg, 20.0)
    assert np.isclose(sc.model.params.sigma_v, config.WIND_PLUME_SIGMA_V)
    g = sc.model.unit_response(np.array([[50.0, 0.0], [60.0, 10.0]]), np.array([[150.0, 0.0, 15.0]]))
    assert g.shape == (2, 1) and np.all(np.isfinite(g)) and np.all(g > 0)
    r = sc.reflected_scene()
    assert r.reflected and r.wind_level == "W0" and isinstance(r.model, GaussianPlume)
    assert np.isclose(r.model.params.wind_dir_deg, -20.0) and np.isclose(r.model.params.U, 1.5)


def test_scene_w1_uses_the_adjoint_on_the_potential_flow_and_mirrors_consistently():
    om = _obstacles()
    wf = potential_flow_field(om, GRID, 1.5, 0.0)
    sc = Scene.build(wf, om, SyntheticBackend(), SOURCES, PARAMS, GRID, wind_level="W1")
    assert isinstance(sc.model, LbmAdjointModel) and sc.wind_level == "W1"
    r = sc.reflected_scene()
    assert r.wind_level == "W1" and isinstance(r.model, LbmAdjointModel)
    uv, uv_r = sc.model.operator.uv, r.model.operator.uv                 # mirrored operator wind = mirrored potential flow
    assert np.allclose(uv_r[::-1, :, 0], uv[..., 0], atol=1e-5) and np.allclose(uv_r[::-1, :, 1], -uv[..., 1], atol=1e-5)


def test_bad_wind_level_rejected():
    with pytest.raises(ValueError):
        Scene.build(uniform_wind_field(1.0, 0.0, x_range=DOMAIN_X, y_range=DOMAIN_Y), _obstacles(), SyntheticBackend(), SOURCES, PARAMS, GRID, wind_level="W9")
    with pytest.raises(ValueError):
        build_wind_field("W9")


@pytest.mark.parametrize("level", ["W0", "W1"])
def test_environment_runs_and_records_carry_the_wind_level(level):
    om = _obstacles()
    wf = uniform_wind_field(1.5, 0.0, x_range=DOMAIN_X, y_range=DOMAIN_Y) if level == "W0" else potential_flow_field(om, GRID, 1.5, 0.0)
    sc = Scene.build(wf, om, SyntheticBackend(), SOURCES, PARAMS, GRID, wind_level=level)
    env = SourceLocEnv(sc, truth_mode="F", reflect_prob=0.0, terminate_on_success=False, max_steps=12, **KW)
    obs, info = env.reset(seed=3)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(int(np.flatnonzero(env.action_mask())[0]))
    assert np.all(np.isfinite(obs)) and np.isfinite(r)
    spec = make_episode_list([1], 1, 7, "F")[0]
    rec, _ = run_episode(env, RandomPolicy(seed=1), spec, log_steps=False)
    assert rec["wind_level"] == level
    tr = make_train_env(sc, sc.reflected_scene(), 1, truth_mode="F", wind_level=level, max_steps=12, **KW)   # scene option is dropped
    assert tr.scene.wind_level == level


def test_enclosed_courtyard_is_calm_and_the_speed_cap_applies(monkeypatch):
    ny, nx, res = 30, 40, 5.0
    blocked = np.zeros((ny, nx), bool)
    blocked[8:20, 12:24] = True                                   # ring with a free courtyard inside
    blocked[11:17, 15:21] = False
    x, y = res * (np.arange(nx) + 0.5), -75.0 + res * (np.arange(ny) + 0.5)
    uv = potential_flow_uv(blocked, res, 1.68, 0.0, x, y)
    inside = np.zeros_like(blocked); inside[11:17, 15:21] = True
    assert np.all(uv[inside] == 0.0)                              # pure-Neumann component: calm, no singular solve
    assert np.hypot(*uv[2, 2]) > 0.5                              # the outer flow still exists
    om = _obstacles()
    monkeypatch.setattr(config, "WIND_POTENTIAL_SPEED_CAP", 1.0)
    wf = potential_flow_field(om, GRID, 1.5, 0.0)
    speed = np.hypot(wf.uvw[0, :, :, 0], wf.uvw[0, :, :, 1])
    assert speed.max() <= 1.5 * 1.0 + 1e-5                        # every cell at or below the cap


def test_w1_default_lattice_covers_the_whole_domain_and_the_slab_grid():
    from srcloc_env.field.wind_models import domain_grid
    g = domain_grid()
    slab = SlabGrid()
    assert g.x0 <= config.DOMAIN_X[0] and g.y0 <= config.DOMAIN_Y[0]
    assert g.x0 + g.nx * g.res >= slab.x0 + slab.nx * slab.res and g.y0 + g.ny * g.res >= slab.y0 + slab.ny * slab.res
    ix = (slab.x_centres - g.x_centres[0]) / g.res
    iy = (slab.y_centres - g.y_centres[0]) / g.res
    assert np.allclose(ix, np.round(ix)) and np.allclose(iy, np.round(iy))     # slab centres are lattice nodes


def test_observation_wind_can_be_uniform_while_the_estimator_keeps_the_potential_flow():
    om = _obstacles()
    wf = potential_flow_field(om, GRID, 1.5, 0.0)
    sc = Scene.build(wf, om, SyntheticBackend(), SOURCES, PARAMS, GRID, wind_level="W1", wind_u=1.5, wind_dir_deg=0.0, obs_wind="uniform")
    assert isinstance(sc.model, LbmAdjointModel) and sc.obs_wind == "uniform"
    pts = np.array([[20.0, -50.0], [150.0, 40.0]])
    assert np.allclose(sc.observation_wind().uv_at(pts, config.DRONE_Z), [[1.5, 0.0], [1.5, 0.0]])    # constant for the policy
    assert not np.allclose(sc.wind.uv_at(pts, config.DRONE_Z), [[1.5, 0.0], [1.5, 0.0]])                # the estimator still sees the potential flow
    r = sc.reflected_scene()
    assert r.obs_wind == "uniform" and np.allclose(r.observation_wind().uv_at(pts, config.DRONE_Z), [[1.5, 0.0], [1.5, 0.0]])
    env = SourceLocEnv(sc, truth_mode="F", reflect_prob=0.0, terminate_on_success=False, max_steps=6, **KW)
    obs, info = env.reset(seed=1)
    assert np.all(np.isfinite(obs))
    with pytest.raises(ValueError):
        Scene.build(wf, om, SyntheticBackend(), SOURCES, PARAMS, GRID, wind_level="W1", obs_wind="bogus")
