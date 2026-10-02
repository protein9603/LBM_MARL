"""D9-1: parameter-shared multi-drone environment on the synthetic scene of tests/test_source_env.py:
n = 1 regression against SourceLocEnv, observation layout (59 / 62), teammate features, determinism, team reward,
random-policy safety for 2 drones, start rules."""
from __future__ import annotations

import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.env.drone import ACTION_STAY
from srcloc_env.env.multi_agent import MultiDroneEnv
from srcloc_env.env.source_env import Scene, SourceLocEnv
from srcloc_env.pf.lbm_adjoint import AdjointParams
from srcloc_env.preprocess.gridder import SlabGrid
from tests.test_source_env import DOMAIN_X, DOMAIN_Y, SOURCES, SyntheticBackend, _obstacles, _wind

KW = dict(sources=(1, 2), n_particles=300, prior_x=DOMAIN_X, prior_y=DOMAIN_Y, start_min_dist=50.0, frame_range=(400, 599))


@pytest.fixture(scope="module")
def scene() -> Scene:
    grid = SlabGrid(0.0, -75.0, 40, 28, 5.0)
    return Scene.build(_wind(), _obstacles(), SyntheticBackend(), SOURCES, AdjointParams(K=16.0, lam=0.005), grid)


def test_single_drone_regression_bit_identical(scene):
    single = SourceLocEnv(scene, **KW)
    multi = MultiDroneEnv(scene, n_drones=1, **KW)
    o1, i1 = single.reset(seed=21)
    om, im = multi.reset(seed=21)
    assert om.shape == (1, 56) and np.array_equal(om[0], o1)
    assert np.allclose(im["drone_xy"][0], i1["drone_xy"]) and im["source"] == i1["source"] and im["frame"] == i1["frame"]
    rng = np.random.default_rng(3)
    for _ in range(40):
        a = int(rng.integers(9))
        s1 = single.step(a)
        sm = multi.step([a])
        assert np.array_equal(sm[0][0], s1[0]) and sm[1] == s1[1] and sm[2] == s1[2] and sm[3] == s1[3]
        assert sm[4]["y"][0] == s1[4]["y"] and sm[4]["applied"][0] == s1[4]["applied"]
        if s1[2] or s1[3]:
            break


def test_observation_layout_and_teammate_features(scene):
    env2 = MultiDroneEnv(scene, n_drones=2, **KW)
    env3 = MultiDroneEnv(scene, n_drones=3, **KW)
    assert env2.observation_space.shape == (2, 59) and env3.observation_space.shape == (3, 62)
    assert env2.action_space.nvec.tolist() == [9, 9]
    starts = np.array([[20.0, -60.0], [40.0, 50.0]])
    o, info = env2.reset(seed=1, options={"source": 1, "start_xy": starts})
    assert np.allclose(info["drone_xy"], starts) and info["action_mask"].shape == (2, 9)
    assert np.allclose(o[0, 56:58], (starts[1] - starts[0]) / 1000) and o[0, 58] == 0.0
    assert np.allclose(o[1, 56:58], (starts[0] - starts[1]) / 1000) and o[1, 58] == 0.0
    assert np.allclose(o[0, 27:29], (20 / 1315, -60 / 657.5)) and np.allclose(o[1, 27:29], (40 / 1315, 50 / 657.5))
    o, r, term, trunc, info = env2.step([0, 2])                      # drone 0 east, drone 1 north
    assert np.allclose(info["drone_xy"], [[25.0, -60.0], [40.0, 55.0]]) and info["applied"] == [True, True]
    assert o[0, 58] == pytest.approx(env2.det.normalise(info["y"][1])) and o[1, 58] == pytest.approx(env2.det.normalise(info["y"][0]))
    assert o[0, 30] == pytest.approx(env2.det.normalise(info["y"][0])) and o[1, 30] == pytest.approx(env2.det.normalise(info["y"][1]))
    assert np.allclose(o[0, 56:58], (info["drone_xy"][1] - info["drone_xy"][0]) / 1000)
    assert isinstance(r, float) and o.shape == (2, 59) and env2.observation_space.contains(o)
    with pytest.raises(ValueError):
        env2.step([0])
    with pytest.raises(ValueError):
        MultiDroneEnv(scene, n_drones=0, **KW)


def test_team_reward_uses_one_shared_belief_update(scene):
    env = MultiDroneEnv(scene, n_drones=2, **KW)
    env.reset(seed=4, options={"source": 2})
    n_upd = env.pf.n_updates
    h_before, h0 = env.h_prev, env.h0
    o, r, term, trunc, info = env.step([ACTION_STAY, ACTION_STAY])
    assert env.pf.n_updates == n_upd + 2                              # both measurements fused sequentially
    assert info["info_gain"] == pytest.approx((h_before - env.h_prev) / h0)
    assert r == pytest.approx(config.ENV_REWARD_TIME + config.ENV_REWARD_INFO * info["info_gain"])
    assert env.t == 1 and len(info["y"]) == 2 and len(info["density"]) == 2


def test_determinism_two_drones(scene):
    e1, e2 = MultiDroneEnv(scene, n_drones=2, **KW), MultiDroneEnv(scene, n_drones=2, **KW)
    o1, _ = e1.reset(seed=11)
    o2, _ = e2.reset(seed=11)
    assert np.array_equal(o1, o2)
    rng = np.random.default_rng(5)
    for _ in range(50):
        a = rng.integers(9, size=2)
        s1, s2 = e1.step(a), e2.step(a)
        assert np.array_equal(s1[0], s2[0]) and s1[1] == s2[1] and s1[2] == s2[2]
        if s1[2] or s1[3]:
            break


def test_random_policy_two_drones_is_safe_and_masks_are_stays(scene):
    env = MultiDroneEnv(scene, n_drones=2, max_steps=40, **KW)
    rng = np.random.default_rng(0)
    env.reset(seed=1)
    n_masked = n_episodes = 0
    for _ in range(400):
        masks = env.action_masks()
        a = rng.integers(9, size=2)
        before = env.xys.copy()
        o, r, term, trunc, info = env.step(a)
        assert env.observation_space.contains(o) and np.isfinite(r)
        assert np.all(scene.obstacles.is_free(env.xys, config.DRONE_Z)) and np.all(scene.obstacles.in_domain(env.xys))
        for i in range(2):
            if not masks[i, a[i]]:
                n_masked += 1
                assert np.array_equal(env.xys[i], before[i]) and not info["applied"][i]
        if term or trunc:
            n_episodes += 1
            env.reset()
    assert n_masked > 0 and n_episodes >= 8


def test_start_rules_distance_and_separation(scene):
    env = MultiDroneEnv(scene, n_drones=3, **KW)
    for seed in range(6):
        _, info = env.reset(seed=seed)
        xys = info["drone_xy"]
        d_src = np.hypot(*(xys - info["truth_xy"]).T)
        assert np.all(d_src >= 50.0)
        for i in range(3):
            for j in range(i + 1, 3):
                assert np.hypot(*(xys[i] - xys[j])) >= config.ENV_START_MIN_SEPARATION_M
    single = SourceLocEnv(scene, **KW)
    _, i1 = single.reset(seed=3)
    _, im = env.reset(seed=3)
    assert np.allclose(im["drone_xy"][0], i1["drone_xy"])          # drone 0 identical to the single-drone draw


def test_second_drone_start_falls_back_when_the_plume_start_set_is_too_small(scene):
    """A narrow ring can leave only a few admissible plume cells: if every plume draw is within the minimum separation of drone 0, the
    second drone falls back to a plain point of the distance ring instead of raising (found by the D12 ring probe)."""
    env = MultiDroneEnv(scene, n_drones=2, start_plume_frac=1.0, start_min_dist=30.0, start_max_dist=90.0, **{k: v for k, v in KW.items() if k not in ("start_min_dist",)})
    orig = env._sample_start
    state = {"first": None, "calls": 0}

    def stuck(rng):
        state["calls"] += 1
        if env.start_plume_frac > 0.0:               # plume pass: always the same cell as drone 0 (separation 0 m)
            if state["first"] is None:
                state["first"] = orig(rng)
            env.start_type = "plume"
            return state["first"].copy()
        return orig(rng)                              # fallback pass: ordinary ring draws

    env._sample_start = stuck
    obs, info = env.reset(seed=5)
    assert env.start_plume_frac == 1.0 and info["drone_xy"].shape == (2, 2)
    assert float(np.hypot(*(info["drone_xy"][0] - info["drone_xy"][1]))) >= env.min_separation
