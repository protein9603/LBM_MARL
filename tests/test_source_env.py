"""T2-1 / T2-2 unit tests of the Gymnasium environment on a synthetic scene (D8-3; plan 4.5):
env_checker, determinism, random-policy safety (no building / domain violation, masked action = stay), observation
layout, reward / termination branches, Mode T2 frame schedule, option overrides and the y-reflected scene."""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from srcloc_env import config
from srcloc_env.env.drone import ACTION_STAY, ObstacleMap
from srcloc_env.env.source_env import Scene, SourceLocEnv
from srcloc_env.field.wind import WindField
from srcloc_env.pf.gmm_summary import GmmSummary
from srcloc_env.pf.lbm_adjoint import AdjointParams
from srcloc_env.preprocess.gridder import SlabGrid

DOMAIN_X, DOMAIN_Y = (0.0, 200.0), (-75.0, 75.0)
SOURCES = {1: (60.0, 10.0), 2: (150.0, -30.0)}


class SyntheticBackend:
    """Frame-independent Gaussian bumps (sigma 20 m) around the sources; flip_y samples (x, -y) like LdmSlabBackend."""

    def __init__(self, sources=SOURCES, amp=5.0, sigma=20.0):
        self.sources, self.amp, self.sigma = dict(sources), amp, sigma
        self.calls = []

    def density(self, src_ids, xy, frame_index, z=config.DRONE_Z, scale=1.0, flip_y=False):
        self.calls.append(int(frame_index))
        ids = [int(src_ids)] if np.isscalar(src_ids) else [int(s) for s in src_ids]
        p = np.atleast_2d(np.asarray(xy, dtype=np.float64)).copy()
        if flip_y:
            p[:, 1] *= -1.0
        out = np.zeros(p.shape[0])
        for s in ids:
            sx, sy = self.sources[s]
            out += self.amp * np.exp(-((p[:, 0] - sx) ** 2 + (p[:, 1] - sy) ** 2) / (2 * self.sigma ** 2))
        return out * scale


def _obstacles() -> ObstacleMap:
    res = 2.0
    nx, ny = 101, 76                                              # x 0..200, y -75..75 (centres y0 + k res)
    occ = np.zeros((nx, ny), dtype=bool)
    hmap = np.zeros((nx, ny), dtype=np.float32)
    occ[45:55, 40:50] = True                                      # block x 90..108, y 5..23 (asymmetric in y)
    hmap[occ] = 30.0
    return ObstacleMap.from_arrays(0.0, -75.0, res, occ, hmap, DOMAIN_X, DOMAIN_Y)


def _wind() -> WindField:
    x = np.linspace(0.0, 200.0, 9)
    y = np.linspace(-75.0, 75.0, 7)
    z = np.array([13.75, 16.25])
    uvw = np.zeros((2, x.size, y.size, 3), dtype=np.float32)
    uvw[..., 0] = 1.5
    uvw[..., 1] = 0.3 + 0.002 * y[None, None, :]                   # asymmetric v(y)
    return WindField.from_arrays(x, y, z, uvw)


@pytest.fixture(scope="module")
def scene() -> Scene:
    grid = SlabGrid(0.0, -75.0, 40, 28, 5.0)                          # y -75..65: NOT symmetric (mirror test)
    return Scene.build(_wind(), _obstacles(), SyntheticBackend(), SOURCES, AdjointParams(K=16.0, lam=0.005), grid)


def _env(scene: Scene, **kw) -> SourceLocEnv:
    base = dict(sources=(1, 2), n_particles=300, prior_x=DOMAIN_X, prior_y=DOMAIN_Y, start_min_dist=50.0,
                frame_range=(400, 599))
    base.update(kw)
    return SourceLocEnv(scene, **base)


def test_spaces_and_observation_layout(scene):
    env = _env(scene)
    assert env.observation_space.shape == (config.ENV_OBS_DIM,) == (56,)
    assert env.action_space.n == 9
    obs, info = env.reset(seed=0, options={"source": 1, "frame": 450, "scale": 1.7, "start_xy": (20.0, -60.0)})
    assert obs.shape == (56,) and obs.dtype == np.float32 and env.observation_space.contains(obs)
    assert info["source"] == 1 and info["frame"] == 450 and info["scale"] == 1.7 and np.allclose(info["drone_xy"], (20, -60))
    assert np.allclose(obs[27:30], (20 / 1315, -60 / 657.5, 1.0))
    assert np.all(obs[30:45] == 0.0) and obs[45] == 0.0                # no measurement yet
    uv = scene.wind.uv_at(np.array([20.0, -60.0]), config.DRONE_Z)[0]
    assert np.allclose(obs[46:48], uv / config.ENV_WIND_NORM, atol=1e-6)
    rays = scene.obstacles.ray_distances(np.array([20.0, -60.0]), config.DRONE_Z)[0]
    assert np.allclose(obs[48:56], rays / config.RAY_MAX_RANGE_M, atol=1e-6)
    assert info["action_mask"].shape == (9,) and info["action_mask"][ACTION_STAY]
    obs2, r, term, trunc, info2 = env.step(0)                           # move east 5 m
    assert np.allclose(info2["drone_xy"], (25, -60)) and info2["applied"]
    assert obs2[30] == pytest.approx(env.det.normalise(info2["y"])) and obs2[31] == 0.0 and obs2[32] == 0.0
    assert obs2[29] == pytest.approx(1 - 1 / env.max_steps)


def test_env_checker(scene):
    check_env(_env(scene), skip_render_check=True)


def test_determinism_same_seed_same_rollout(scene):
    e1, e2 = _env(scene), _env(scene)
    o1, _ = e1.reset(seed=11)
    o2, _ = e2.reset(seed=11)
    assert np.array_equal(o1, o2)
    rng = np.random.default_rng(5)
    for _ in range(60):
        a = int(rng.integers(9))
        s1 = e1.step(a)
        s2 = e2.step(a)
        assert np.array_equal(s1[0], s2[0]) and s1[1] == s2[1] and s1[2] == s2[2] and s1[3] == s2[3]
        if s1[2] or s1[3]:
            break
    o3, _ = e1.reset(seed=12)
    assert not np.array_equal(o1, o3)


def test_random_policy_is_safe(scene):
    env = _env(scene, max_steps=40)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=1)
    n_steps = n_masked = n_episodes = 0
    for _ in range(600):
        mask = env.action_mask()
        a = int(rng.integers(9))
        xy_before = env.xy.copy()
        obs, r, term, trunc, info = env.step(a)
        n_steps += 1
        assert env.observation_space.contains(obs) and np.isfinite(r)
        assert scene.obstacles.is_free(env.xy, config.DRONE_Z) and scene.obstacles.in_domain(env.xy)
        if not mask[a]:
            n_masked += 1
            assert np.array_equal(env.xy, xy_before) and not info["applied"]
        if term or trunc:
            n_episodes += 1
            obs, _ = env.reset()
    assert n_steps == 600 and n_episodes >= 10 and n_masked > 0


def test_truncation_failure_term_and_success_bonus(scene):
    env = _env(scene, max_steps=3)
    env.reset(seed=2, options={"source": 2, "start_xy": (30.0, 60.0)})
    rewards = []
    for _ in range(3):
        obs, r, term, trunc, info = env.step(ACTION_STAY)
        rewards.append(r)
    assert trunc and not term and info["t"] == 3
    err = info["map_error_m"]
    expected_tail = config.ENV_REWARD_TIME + config.ENV_REWARD_INFO * info["info_gain"] \
        - min(err, config.ENV_FAIL_ERROR_CAP_M) / config.ENV_FAIL_ERROR_SCALE_M
    assert rewards[-1] == pytest.approx(expected_tail)
    # success branch: force a tight, correct belief summary
    env = _env(scene, max_steps=50)
    env.reset(seed=3, options={"source": 1, "start_xy": (60.0, 40.0)})
    truth = env.truth_xy.copy()
    env.pf.map_estimate = lambda *a, **k: truth.copy()
    fake = GmmSummary(np.array([1.0, 0.0, 0.0]), np.array([truth, [0, 0], [0, 0]], dtype=float),
                      np.array([np.eye(2) * 4.0, np.eye(2), np.eye(2)]), np.array([True, False, False]))
    env._refresh_gmm = lambda: setattr(env, "gmm", fake)
    obs, r, term, trunc, info = env.step(ACTION_STAY)
    assert term and info["success"] and not trunc
    assert r == pytest.approx(config.ENV_REWARD_TIME + config.ENV_REWARD_INFO * info["info_gain"] + config.ENV_REWARD_SUCCESS)


def test_mode_t2_frame_schedule(scene):
    env = _env(scene, truth_mode="T2", t2_start_range=(560, 560), max_steps=60)
    env.reset(seed=4)
    assert env.frame_at(0) == 560 and env.frame_at(39) == 599 and env.frame_at(100) == 599
    scene.backend.calls.clear()
    for t in range(45):
        _, _, _, _, info = env.step(ACTION_STAY)
        assert info["frame"] == scene.backend.calls[-1] == min(599, 560 + t) and info["t"] == t + 1   # frame of info["y"]
    assert scene.backend.calls[:3] == [560, 561, 562] and scene.backend.calls[-1] == 599
    with pytest.raises(ValueError):
        _env(scene, truth_mode="X")


def test_reflected_scene_mirrors_wind_obstacles_sources_and_model(scene):
    sr = scene.reflected_scene()
    assert sr.reflected and sr.sources_xy == {1: (60.0, -10.0), 2: (150.0, 30.0)}
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 200, 300), rng.uniform(-75, 75, 300)])
    mir = pts * np.array([1.0, -1.0])
    uv = scene.wind.uv_at(pts, config.DRONE_Z)
    uv_r = sr.wind.uv_at(mir, config.DRONE_Z)
    assert np.allclose(uv_r[:, 0], uv[:, 0], atol=1e-5) and np.allclose(uv_r[:, 1], -uv[:, 1], atol=1e-5)
    assert np.array_equal(sr.obstacles.is_free(mir, config.DRONE_Z), scene.obstacles.is_free(pts, config.DRONE_Z))
    d = scene.backend.density([1], pts, 400)
    d_r = sr.backend.density([1], mir, 400, flip_y=True)
    assert np.allclose(d, d_r)
    g = scene.model.unit_response(np.array(SOURCES[1]), np.array([120.0, 20.0, 15.0]))
    g_r = sr.model.unit_response(np.array(sr.sources_xy[1]), np.array([120.0, -20.0, 15.0]))
    assert np.allclose(g, g_r, rtol=1e-6)
    assert np.allclose(sr.grid.y_centres, np.sort(-scene.grid.y_centres))        # adjoint grid mirrored (review D8-3)
    H = np.column_stack([rng.uniform(0, 200, 200), rng.uniform(-75, 75, 200)])
    D = np.array([[100.0, -70.0, 15.0], [120.0, 20.0, 15.0]])
    assert np.allclose(scene.model.unit_response(H, D), sr.model.unit_response(H * [1, -1], D * [1, -1, 1]), rtol=1e-6)
    env = SourceLocEnv(scene, sources=(1,), n_particles=200, prior_x=DOMAIN_X, prior_y=DOMAIN_Y, start_min_dist=50.0,
                       scene_reflected=sr, reflect_prob=0.5)
    o, info = env.reset(seed=7, options={"reflect": False, "start_xy": (20.0, 40.0)})
    o_r, info_r = env.reset(seed=7, options={"reflect": True, "start_xy": (20.0, -40.0)})
    assert info_r["reflected"] and np.allclose(info_r["truth_xy"], (60.0, -10.0))
    assert o_r[27] == pytest.approx(o[27]) and o_r[28] == pytest.approx(-o[28])
    assert o_r[46] == pytest.approx(o[46], abs=1e-6) and o_r[47] == pytest.approx(-o[47], abs=1e-6)
    perm = [0, 7, 6, 5, 4, 3, 2, 1]                                   # heading k -> -k mod 8 under y -> -y
    assert np.allclose(o_r[48:56], o[48:56][perm])
    with pytest.raises(ValueError):
        SourceLocEnv(scene, sources=(1,), reflect_prob=0.5)


PERM = np.array([0, 7, 6, 5, 4, 3, 2, 1, ACTION_STAY])             # heading k -> -k mod 8 under y -> -y


def test_reflected_rollout_mirrors_positions_masks_and_measurements(scene):
    """Stepping a reflected episode with mirrored actions mirrors the drone path, permutes the action mask and
    reproduces the same truth densities / counts (flip_y wiring, mirrored kinematics; review D8-3)."""
    sr = scene.reflected_scene()
    kw = dict(sources=(1,), n_particles=200, prior_x=DOMAIN_X, prior_y=DOMAIN_Y, start_min_dist=50.0,
              frame_range=(400, 599), scene_reflected=sr, reflect_prob=0.5, max_steps=80)
    env, env_r = SourceLocEnv(scene, **kw), SourceLocEnv(scene, **kw)
    # start y 13.3: positions on exact 2 m raster cell boundaries (even y) round towards +y in both orientations
    # (ObstacleMap.cell_index floor(.. + 0.5)), which breaks exact mirror symmetry of ray samples by one cell
    o, info = env.reset(seed=7, options={"reflect": False, "start_xy": (80.0, 13.3)})
    o_r, info_r = env_r.reset(seed=7, options={"reflect": True, "start_xy": (80.0, -13.3)})
    rng = np.random.default_rng(5)
    actions = [0, 0, 0] + [int(a) for a in rng.integers(9, size=70)]
    n_masked = 0
    for a in actions:
        o, r, term, trunc, info = env.step(a)
        o_r, r_r, term_r, trunc_r, info_r = env_r.step(int(PERM[a]))
        assert np.allclose(info_r["drone_xy"], info["drone_xy"] * [1, -1])
        assert info_r["applied"] == info["applied"]
        assert np.array_equal(info_r["action_mask"], info["action_mask"][PERM])
        assert info_r["density"] == pytest.approx(info["density"], rel=1e-9) and info_r["y"] == info["y"]
        assert sr.obstacles.is_free(env_r.xy, config.DRONE_Z) and sr.obstacles.in_domain(env_r.xy)
        assert o_r[27] == pytest.approx(o[27]) and o_r[28] == pytest.approx(-o[28])
        assert o_r[46] == pytest.approx(o[46], abs=1e-5) and o_r[47] == pytest.approx(-o[47], abs=1e-5)
        assert np.allclose(o_r[48:56], o[48:56][PERM[:8]], atol=1e-6)
        n_masked += int(not info["applied"])
        if term or trunc:
            break
    assert n_masked > 0


def test_pf_uses_the_environment_detector(scene):
    from srcloc_env.sensor.detector import Detector
    env = _env(scene, detector=Detector(background=0.0, T=2.0))
    env.reset(seed=0)
    assert env.pf.background == 0.0 and env.pf.T == 2.0
    env2 = _env(scene, detector=Detector(k0=500.0))
    env2.reset(seed=0)
    assert env2.pf.kgrid[len(env2.pf.kgrid) // 2] == pytest.approx(config.KAPPA_REF * 0.5)
    with pytest.raises(ValueError):
        _env(scene, detector=Detector(k0=0.0))


def test_validate_env_random_policy_checks_the_current_orientation(scene):
    from srcloc_env.scripts.validate_env import run_random_policy
    sr = scene.reflected_scene()
    env = _env(scene, scene_reflected=sr, reflect_prob=1.0, max_steps=50)
    rp = run_random_policy(env, 400, seed=1)
    assert rp["n_exceptions"] == 0 and rp["n_violations"] == 0 and rp["n_outside_domain"] == 0 and rp["n_episodes"] >= 7


def test_gmm_iters_and_warm_start_options(scene):
    env = _env(scene, gmm_iters=3, gmm_warm_start=True, max_steps=10)
    o1, _ = env.reset(seed=9)
    for _ in range(5):
        o1, *_ = env.step(ACTION_STAY)
    env2 = _env(scene, gmm_iters=3, gmm_warm_start=True, max_steps=10)
    o2, _ = env2.reset(seed=9)
    for _ in range(5):
        o2, *_ = env2.step(ACTION_STAY)
    assert np.array_equal(o1, o2)                                      # still deterministic
    with pytest.raises(ValueError):
        _env(scene, gmm_iters=0)


def test_plume_start_rule(scene):
    """D9-4: start_plume_frac = 1 puts the drone where the episode's source is detectable (expected counts >= Currie
    threshold), >= start_min_dist from the source; frac 0 never reports 'plume'; a given start reports 'given'."""
    env = _env(scene, start_plume_frac=1.0)
    thr = config.ENV_START_PLUME_MIN_COUNTS_FACTOR * env.det.detection_threshold_cps() * env.det.T
    n_plume = 0
    for seed in range(8):
        _, info = env.reset(seed=seed, options={"source": 1, "scale": 3.0})
        dens = scene.backend.density([1], info["drone_xy"], info["frame"], config.DRONE_Z, 1.0)
        d_src = np.hypot(*(info["drone_xy"] - info["truth_xy"]))
        assert d_src >= 50.0
        if info["start_type"] == "plume":
            n_plume += 1
            assert env.det.expected_counts(dens, 3.0)[0] >= thr
    assert n_plume >= 6
    env0 = _env(scene, start_plume_frac=0.0)
    assert all(env0.reset(seed=s)[1]["start_type"] == "random" for s in range(4))
    assert env0.reset(seed=1, options={"start_xy": (20.0, -60.0)})[1]["start_type"] == "given"



class SlabSyntheticBackend(SyntheticBackend):
    """SyntheticBackend that also exposes slab() (like LdmSlabBackend), so the direct plume-start draw is used."""

    def slab(self, frame_index):
        from srcloc_env.field.concentration_field import SlabFrame
        grid = SlabGrid(0.0, -75.0, 40, 30, 5.0)
        xx, yy = np.meshgrid(grid.x_centres, grid.y_centres)
        dens = np.zeros((len(self.sources), 1, grid.ny, grid.nx), np.float32)
        for k, (sid, (sx, sy)) in enumerate(sorted(self.sources.items())):
            dens[k, 0] = self.amp * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * self.sigma ** 2))
        return SlabFrame(index=int(frame_index), density=dens, z_levels=(config.DRONE_Z,), sources=tuple(sorted(self.sources)), grid=grid)


def test_reflected_scene_draws_plume_starts_like_the_original(scene):
    """The direct plume-start draw must also work in a y-reflected scene (mirrored cell centres): starts are detectable under the
    mirrored density and the plume share equals the original scene's (review D11: reflected episodes used to fall back to random)."""
    sc = Scene.build(_wind(), _obstacles(), SlabSyntheticBackend(), SOURCES, AdjointParams(K=16.0, lam=0.005), SlabGrid(0.0, -75.0, 40, 28, 5.0))
    scr = sc.reflected_scene()
    kw = dict(start_plume_frac=1.0, start_min_dist=30.0, scene_reflected=scr)
    n_plume = {False: 0, True: 0}
    for refl in (False, True):
        env = _env(sc, **kw)
        for seed in range(25):
            _, info = env.reset(seed=seed, options={"reflect": refl, "source": 1})
            if info["start_type"] == "plume":
                n_plume[refl] += 1
                thr = config.ENV_START_PLUME_MIN_COUNTS_FACTOR * env.det.detection_threshold_cps() * env.det.T
                dens = sc.backend.density([1], info["drone_xy"], env.frame_at(0), config.DRONE_Z, 1.0, flip_y=refl)
                assert float(env.det.expected_counts(dens, env.scale)[0]) >= thr * 0.999        # detectable at the START (mirrored density for reflected)
    assert n_plume[True] >= 20 and n_plume[False] >= 20


def test_start_max_dist_keeps_every_start_inside_the_distance_ring(scene):
    """Curriculum stage A: starts (plume and random) lie within [start_min_dist, start_max_dist] of the source."""
    for frac in (0.0, 1.0):
        env = _env(scene, start_min_dist=20.0, start_max_dist=60.0, start_plume_frac=frac)
        for seed in range(15):
            _, info = env.reset(seed=seed, options={"source": 1})
            d = float(np.hypot(*(info["drone_xy"] - info["truth_xy"])))
            assert 20.0 - 1e-6 <= d <= 60.0 + 1e-6, (frac, seed, d)
