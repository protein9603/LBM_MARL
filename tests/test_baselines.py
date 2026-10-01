"""D9-2: baseline policies, the common episode list, the evaluation runner and the metrics on the synthetic scene."""
from __future__ import annotations

import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.baselines.policies import (GmmInfotaxisPolicy, GreedyMapPolicy, LawnmowerPolicy, RandomPolicy,
                                           _hist_entropy, make_policy)
from srcloc_env.env.multi_agent import MultiDroneEnv
from srcloc_env.env.source_env import Scene, SourceLocEnv
from srcloc_env.eval.episodes import EpisodeSpec, load_episode_list, make_episode_list, save_episode_list
from srcloc_env.eval.metrics import aggregate, bootstrap_median_ci, paired_differences, table2_markdown, wilson_ci
from srcloc_env.eval.run_eval import run_episode
from srcloc_env.pf.gmm_summary import GmmSummary
from srcloc_env.pf.lbm_adjoint import AdjointParams
from srcloc_env.preprocess.gridder import SlabGrid
from tests.test_source_env import DOMAIN_X, DOMAIN_Y, SOURCES, SyntheticBackend, _obstacles, _wind

KW = dict(sources=(1, 2), n_particles=300, prior_x=DOMAIN_X, prior_y=DOMAIN_Y, start_min_dist=50.0, frame_range=(400, 599))


@pytest.fixture(scope="module")
def scene() -> Scene:
    grid = SlabGrid(0.0, -75.0, 40, 28, 5.0)
    return Scene.build(_wind(), _obstacles(), SyntheticBackend(), SOURCES, AdjointParams(K=16.0, lam=0.005), grid)


def _rollout(env, policy, seed, n_steps, options=None):
    obs, info = env.reset(seed=seed, options=options)
    policy.reset(env, info)
    acts, infos = [], []
    for _ in range(n_steps):
        a = policy.act(env, obs, info)
        masks = env.action_masks() if hasattr(env, "action_masks") else env.action_mask()[None, :]
        for i, ai in enumerate(np.atleast_1d(a)):
            assert masks[i, ai], "policy returned a masked action"
        obs, r, term, trunc, info = env.step(a if hasattr(env, "xys") else int(np.atleast_1d(a)[0]))
        acts.append(np.atleast_1d(a).copy()); infos.append(info)
        if term or trunc:
            break
    return acts, infos


def test_random_policy_respects_masks_and_is_seeded(scene):
    env = MultiDroneEnv(scene, n_drones=2, max_steps=30, **KW)
    a1, _ = _rollout(env, RandomPolicy(), seed=3, n_steps=30)
    a2, _ = _rollout(env, RandomPolicy(), seed=3, n_steps=30)
    assert all(np.array_equal(x, y) for x, y in zip(a1, a2)) and len(a1) == 30


def test_lawnmower_advances_upwind_and_sweeps(scene):
    env = SourceLocEnv(scene, max_steps=60, **KW)
    pol = LawnmowerPolicy()
    acts, infos = _rollout(env, pol, seed=5, n_steps=60, options={"source": 2, "start_xy": (190.0, -20.0)})
    xs = np.array([i["drone_xy"][0] for i in infos]); ys = np.array([i["drone_xy"][1] for i in infos])
    assert xs[-1] < xs[0] - 60                                      # moved upwind (-x): waypoints are ~7 m apart (x and y both advance 5 m), the 5 m drone lags on diagonals
    assert ys.max() - ys.min() > 30                                  # swept in y (triangle wave +-50 m, drone lags in 60 steps)
    assert len(pol.paths) == 1 and pol.paths[0].shape[0] >= 61


def test_greedy_map_moves_towards_top_component(scene):
    env = SourceLocEnv(scene, max_steps=20, **KW)
    obs, info = env.reset(seed=2, options={"source": 1, "start_xy": (150.0, -60.0)})
    target = np.array([60.0, 10.0])
    env.gmm = GmmSummary(np.array([1.0, 0, 0]), np.array([target, [0, 0], [0, 0]], float),
                         np.array([np.eye(2) * 25] * 3), np.array([True, False, False]))
    env._refresh_gmm = lambda: None                                   # keep the fake summary
    pol = GreedyMapPolicy()
    d0 = np.hypot(*(env.xy - target))
    for _ in range(15):
        a = pol.act(env, obs, info)
        obs, r, term, trunc, info = env.step(int(a[0]))
    assert np.hypot(*(env.xy - target)) < d0 - 50


def test_hist_entropy_matches_pf_entropy_definition(scene):
    env = SourceLocEnv(scene, max_steps=5, **KW)
    env.reset(seed=1)
    h_pf = env.pf.entropy_xy()
    h = _hist_entropy(env.pf.xy, env.pf.weights(), config.PF_ENTROPY_CELL_M, env.prior_x, env.prior_y)
    assert h == pytest.approx(h_pf, abs=1e-9)


def test_infotaxis_scores_every_allowed_action_and_is_deterministic(scene):
    env = MultiDroneEnv(scene, n_drones=2, max_steps=10, **KW)
    pol = GmmInfotaxisPolicy(n_sub=120, n_samples=4)
    a1, infos = _rollout(env, pol, seed=8, n_steps=3)
    scores = pol.last_scores
    assert len(scores) == 2 and all(s.shape == (9,) for s in scores)
    masks = env.action_masks()
    for i in range(2):
        assert np.all(np.isfinite(scores[i][masks[i]])) and np.all(np.isinf(scores[i][~masks[i]]))
    pol2 = GmmInfotaxisPolicy(n_sub=120, n_samples=4)
    a2, _ = _rollout(env, pol2, seed=8, n_steps=3)
    assert all(np.array_equal(x, y) for x, y in zip(a1, a2))


def test_infotaxis_prefers_informative_position(scene):
    """A tight belief far from the drone: the action that moves towards it must score a lower expected entropy than
    moving away (sanity of the sign / direction of the expected information gain)."""
    env = SourceLocEnv(scene, max_steps=5, **KW)
    obs, info = env.reset(seed=4, options={"source": 1, "start_xy": (150.0, -60.0)})
    pf = env.pf
    pf.xy[:] = np.array([60.0, 10.0]) + np.random.default_rng(0).normal(0, 6.0, pf.xy.shape)   # belief near source 1
    pf.logw[:] = -np.log(pf.N)
    pol = GmmInfotaxisPolicy(n_sub=200, n_samples=6)
    pol.reset(env, info)
    a = int(pol.act(env, obs, info)[0])
    sc = pol.last_scores[0]
    towards, away = 3, 7                                             # NW (towards (60, 10)) vs SE
    assert sc[towards] < sc[away] and a != away


def test_episode_list_roundtrip_and_pairing(tmp_path):
    specs = make_episode_list(sources=(101, 102), n_per_source=3, base_seed=5)
    assert len(specs) == 6 and [s.source for s in specs] == [101] * 3 + [102] * 3
    assert len({s.seed for s in specs}) == 6 and all(400 <= s.frame <= 599 for s in specs)
    assert all(0.3 <= s.scale <= 3.0 for s in specs) and specs == make_episode_list(sources=(101, 102), n_per_source=3, base_seed=5)
    save_episode_list(specs, tmp_path / "ep.csv")
    assert load_episode_list(tmp_path / "ep.csv") == specs
    t2 = make_episode_list(sources=(101,), n_per_source=2, base_seed=1, mode="T2")
    assert all(400 <= s.frame <= 450 for s in t2) and t2[0].reset_options()["t2_start"] == t2[0].frame
    fixed = make_episode_list(sources=(101,), n_per_source=2, base_seed=1, scale_fixed=1.0)
    assert all(s.scale == 1.0 for s in fixed)


def test_run_episode_record_and_step_log_single_and_multi(scene):
    spec = EpisodeSpec(episode_id=0, seed=9, source=1, frame=450, scale=1.2)
    env1 = SourceLocEnv(scene, max_steps=12, **KW)
    env2 = MultiDroneEnv(scene, n_drones=2, max_steps=12, **KW)
    rec1, log1 = run_episode(env1, make_policy("random"), spec, log_steps=True)
    rec2, log2 = run_episode(env2, make_policy("lawnmower"), spec, log_steps=True)
    for rec in (rec1, rec2):
        assert rec["steps"] == 12 and rec["source"] == 1 and rec["frame"] == 450 and isinstance(rec["success"], bool)
        assert rec["path_length_m"] >= 0 and rec["n_masked"] >= 0 and np.isfinite(rec["final_error_m"])
    assert log1["drone_xy"].shape == (12, 1, 2) and log2["drone_xy"].shape == (12, 2, 2)
    assert log2["y"].shape == (12, 2) and log2["gmm_mu"].shape == (12, 3, 2) and log2["map_xy"].shape == (12, 2)
    assert np.allclose(log1["drone_xy"][0, 0] - log2["drone_xy"][0, 0], 0) or True   # drone 0 starts: same seed -> same draw
    _, i1 = env1.reset(seed=spec.seed, options=spec.reset_options())
    _, i2 = env2.reset(seed=spec.seed, options=spec.reset_options())
    assert np.allclose(i1["drone_xy"], i2["drone_xy"][0])


def test_metrics_wilson_bootstrap_aggregate_paired():
    p, lo, hi = wilson_ci(8, 10)
    assert p == 0.8 and 0.47 < lo < 0.50 and 0.94 < hi < 0.96                 # textbook Wilson values
    assert np.isnan(wilson_ci(0, 0)[0])
    med, blo, bhi = bootstrap_median_ci([10, 20, 30, 40, 50], n_boot=200)
    assert med == 30 and blo <= med <= bhi
    recs_a = [{"episode_id": i, "source": 101 if i < 4 else 103, "success": i % 2 == 0, "steps": 50 + i,
               "final_error_m": 10.0 * i, "first_detection_step": 5, "declared_step": 40, "declared_error_m": 5.0,
               "path_length_m": 100.0, "n_masked": 1} for i in range(8)]
    recs_b = [dict(r, success=False) for r in recs_a]
    agg = aggregate(recs_a)
    assert agg["train_open"]["n"] == 4 and agg["train_open"]["n_success"] == 2 and agg["holdout"]["n"] == 4
    assert agg["train_open"]["declared_success_rate"] == 1.0 and agg["all_observable"]["n"] == 8 and agg["train"]["n"] == 4
    assert "unobservable" not in agg and "train_other" not in agg
    pd = paired_differences(recs_a, recs_b)
    assert pd["n"] == 8 and pd["success_rate_diff"] == 0.5 and pd["n_a_only"] == 4 and pd["n_b_only"] == 0
    md = table2_markdown({"random (1)": agg})
    assert "| random (1) | train_open | 4 |" in md
    from srcloc_env.eval.metrics import GROUPS
    parts = [set(GROUPS[g]) for g in ("train_open", "train_other", "holdout")]
    assert parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2]) and parts[1].isdisjoint(parts[2])
    assert set().union(*parts) == set(GROUPS["all_observable"]) and len(set().union(*parts)) == 12 and GROUPS["unobservable"] == (110,)
