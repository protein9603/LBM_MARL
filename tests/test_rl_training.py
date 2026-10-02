"""D10: rollout collection, training loop, checkpoint round trip and the evaluation adapter on the synthetic scene."""
from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from srcloc_env import config
from srcloc_env.env.source_env import Scene
from srcloc_env.eval.episodes import EpisodeSpec
from srcloc_env.eval.run_eval import run_episode
from srcloc_env.pf.lbm_adjoint import AdjointParams
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.rl import train as tr
from srcloc_env.rl.ppo import ActorCritic
from srcloc_env.rl.ppo_policy import PPOPolicy
from srcloc_env.rl.rollout import RolloutCollector, episode_seed, make_train_env
from tests.test_source_env import DOMAIN_X, DOMAIN_Y, SOURCES, SyntheticBackend, _obstacles, _wind

KW = dict(sources=(1, 2), n_particles=300, prior_x=DOMAIN_X, prior_y=DOMAIN_Y, start_min_dist=50.0, frame_range=(400, 599), max_steps=15, terminate_on_success=False)


@pytest.fixture(scope="module")
def scene() -> Scene:
    grid = SlabGrid(0.0, -75.0, 40, 28, 5.0)
    return Scene.build(_wind(), _obstacles(), SyntheticBackend(), SOURCES, AdjointParams(K=16.0, lam=0.005), grid)


def _collector(scene: Scene, n: int, run_seed: int = 1, proc: int = 0, **kw) -> RolloutCollector:
    return RolloutCollector(make_train_env(scene, scene.reflected_scene(), n, **{**KW, **kw}), run_seed, proc)


def test_episode_seed_spaces_are_disjoint_from_the_evaluation_lists():
    assert episode_seed(1, 0, 0) == 1_000_000 and episode_seed(3, 2, 99) == 3_200_099
    assert episode_seed(3, 9, config.TRAIN_SEED_PROC_STRIDE - 1) < config.EVAL_BASE_SEED
    assert episode_seed(3, 9, 99_999) < config.EVAL_CKPT_BASE_SEED
    with pytest.raises(ValueError):
        episode_seed(1, 0, config.TRAIN_SEED_PROC_STRIDE)


@pytest.mark.parametrize("n", [1, 2])
def test_collector_shapes_masks_episode_records_and_no_masked_action_is_applied(scene, n):
    col = _collector(scene, n)
    net = ActorCritic(config.ENV_OBS_DIM + config.ENV_TEAMMATE_DIM * (n - 1))
    ro = col.collect(net, 40, iteration=1)
    D = net.obs_dim
    assert ro["obs"].shape == (40, n, D) and ro["act"].shape == (40, n) and ro["mask"].shape == (40, n, config.DRONE_N_ACTIONS)
    assert ro["rew"].shape == (40,) and ro["last_val"].shape == (n,)
    assert ro["applied_masked"] == 0                                      # logit masking: the policy never samples a masked action
    assert ro["mask"][np.arange(40)[:, None], np.arange(n)[None, :], ro["act"]].all()
    assert ro["done"].sum() == 2 and ro["done"][14] and ro["done"][29]    # max_steps 15 -> truncation at steps 15 and 30
    assert [e["length"] for e in ro["episodes"]] == [15, 15] and all(e["truncated"] for e in ro["episodes"])
    assert [e["episode_idx"] for e in ro["episodes"]] == [0, 1] and ro["episode_idx"] == 3     # episode 2 is running
    assert np.isfinite(ro["val"]).all() and 0.0 <= ro["masked_share"] < 1.0
    # the team return of an episode is the sum of its rewards
    assert ro["episodes"][0]["ret"] == pytest.approx(float(ro["rew"][:15].sum()))


def test_collection_is_deterministic_given_weights_and_seeds(scene):
    torch.manual_seed(0)
    net = ActorCritic(59)
    a = _collector(scene, 2).collect(net, 20, 3)
    b = _collector(scene, 2).collect(net, 20, 3)
    c = _collector(scene, 2, run_seed=2).collect(net, 20, 3)
    assert np.array_equal(a["obs"], b["obs"]) and np.array_equal(a["act"], b["act"]) and np.array_equal(a["rew"], b["rew"])
    assert not np.array_equal(a["obs"], c["obs"])
    d = _collector(scene, 2).collect(net, 20, 4)                           # another iteration: another sampling stream
    assert not np.array_equal(a["act"], d["act"])


def test_success_terminates_the_training_episode_and_is_recorded(scene):
    col = _collector(scene, 1, terminate_on_success=True, max_steps=60)
    ro = col.collect(ActorCritic(config.ENV_OBS_DIM), 60, 1)
    ended = [e for e in ro["episodes"] if e["success"]]
    assert ended and all(not e["truncated"] and e["length"] < 60 for e in ended)


class _FakePool:
    """In-process stand-in for RolloutPool on the synthetic scene."""
    scene: Scene = None
    last_env_kw: dict | None = None

    def __init__(self, n_drones, run_seed, n_procs, episode_idx=None, env_kw=None):   # noqa: ANN001
        _FakePool.last_env_kw = env_kw
        extra = {"obs_version": env_kw["obs_version"]} if env_kw and "obs_version" in env_kw else {}
        self.col = RolloutCollector(make_train_env(self.scene, self.scene.reflected_scene(), n_drones, **{**KW, **extra}), run_seed, 0, (episode_idx or [0])[0])

    def collect(self, net, n_steps, iteration):   # noqa: ANN001
        return [self.col.collect(net, n_steps, iteration)]

    def close(self) -> None:
        pass


def _run(args_list, scene, monkeypatch):
    _FakePool.scene = scene
    monkeypatch.setattr(tr, "RolloutPool", _FakePool)
    return tr.train(tr.build_parser().parse_args(args_list))


def _rows(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_training_loop_logs_checkpoints_resumes_and_widens(scene, tmp_path, monkeypatch):
    base = ["--run-name", "m1", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--log-every", "1"]
    out = _run(base + ["--n-drones", "1", "--total-steps", "96"], scene, monkeypatch)
    d = tmp_path / "m1"
    rows = _rows(d / "train_log.csv")
    assert out["iterations"] == 3 and [int(r["iteration"]) for r in rows] == [1, 2, 3] and int(rows[-1]["env_steps"]) == 96
    assert all(int(r["applied_masked"]) == 0 and np.isfinite(float(r["policy_loss"])) for r in rows)
    assert (d / "final.pt").exists() and (d / "latest.pt").exists() and (d / "config.json").exists()
    assert _rows(d / "episodes.csv")                                           # 15-step episodes finished within 96 steps
    # resume continues the iteration and step counters and appends to the logs
    _run(base + ["--n-drones", "1", "--total-steps", "160", "--resume"], scene, monkeypatch)
    rows = _rows(d / "train_log.csv")
    assert [int(r["iteration"]) for r in rows] == [1, 2, 3, 4, 5] and int(rows[-1]["env_steps"]) == 160
    # M2 from M1 weights: the widened network starts as the same function on the 56 shared inputs
    _run(["--run-name", "m2", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "2",
          "--total-steps", "64", "--init-from", str(d / "final.pt")], scene, monkeypatch)
    cfg = (tmp_path / "m2" / "config.json").read_text(encoding="utf-8")
    assert "widened 56 -> 59" in cfg
    assert len(_rows(tmp_path / "m2" / "train_log.csv")) == 2


def test_ppo_policy_adapter_runs_an_episode_and_checks_the_input_size(scene, tmp_path, monkeypatch):
    _run(["--run-name", "p", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "1",
          "--total-steps", "32"], scene, monkeypatch)
    ck = tmp_path / "p" / "final.pt"
    env = make_train_env(scene, scene.reflected_scene(), 1, **{**KW, "terminate_on_success": False})
    spec = EpisodeSpec(episode_id=0, seed=7, source=1, frame=450, scale=1.0)
    pol = PPOPolicy(ck, alias="ppo_test")
    rec, _ = run_episode(env, pol, spec)
    rec2, _ = run_episode(env, PPOPolicy(ck, alias="ppo_test"), spec)
    assert rec["method"] == "ppo_test" and rec["steps"] == 15 and rec["n_masked"] == 0
    assert rec["final_error_m"] == pytest.approx(rec2["final_error_m"]) and rec["path_length_m"] == pytest.approx(rec2["path_length_m"])   # reproducible
    env2 = make_train_env(scene, scene.reflected_scene(), 2, **KW)
    with pytest.raises(ValueError):
        run_episode(env2, PPOPolicy(ck), spec)                                     # 1-drone checkpoint on the 2-drone environment


def test_t2_environment_advances_the_frame_every_step_and_uses_the_t2_horizon(scene):
    kw = {k: v for k, v in KW.items() if k not in ("max_steps", "frame_range", "terminate_on_success")}
    env = make_train_env(scene, scene.reflected_scene(), 1, truth_mode="T2", terminate_on_success=False, **kw)
    assert env.max_steps == config.T2_MAX_STEPS == 150
    env.reset(seed=3)
    f0 = env.frame_at(0)
    assert config.T2_TRAIN_START_RANGE[0] <= f0 <= config.T2_TRAIN_START_RANGE[1]
    frames, n, done = [], 0, False
    while not done:
        _, _, term, trunc, info = env.step(8)
        frames.append(info["frame"])
        n += 1
        done = term or trunc
    assert n == 150 and frames == [f0 + t for t in range(150)] and max(frames) <= 599       # no zero-order hold inside the episode


def test_train_passes_the_truth_mode_and_horizon_to_the_workers_and_the_checkpoint(scene, tmp_path, monkeypatch):
    _run(["--run-name", "t2", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "1",
          "--total-steps", "32", "--truth-mode", "T2"], scene, monkeypatch)
    assert _FakePool.last_env_kw == {"truth_mode": "T2", "max_steps": config.T2_MAX_STEPS}
    ck = torch.load(tmp_path / "t2" / "final.pt", map_location="cpu", weights_only=False)
    assert ck["truth_mode"] == "T2" and ck["max_steps"] == 150
    _run(["--run-name", "f", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "1",
          "--total-steps", "32"], scene, monkeypatch)
    assert _FakePool.last_env_kw == {"truth_mode": "F", "max_steps": config.MAX_EPISODE_STEPS}


def test_resume_refuses_a_different_truth_mode_and_ppo_policy_warns_on_a_mode_mismatch(scene, tmp_path, monkeypatch):
    base = ["--run-name", "g", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "1", "--truth-mode", "T2"]
    _run(base + ["--total-steps", "32"], scene, monkeypatch)
    with pytest.raises(ValueError, match="--resume"):
        _run(["--run-name", "g", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "1",
              "--total-steps", "64", "--resume"], scene, monkeypatch)                      # Mode F by default: must not silently continue a T2 run
    _run(base + ["--total-steps", "64", "--resume"], scene, monkeypatch)                    # same mode: fine
    ck = tmp_path / "g" / "final.pt"
    env_f = make_train_env(scene, scene.reflected_scene(), 1, **{**KW, "terminate_on_success": False})              # Mode F environment
    spec = EpisodeSpec(episode_id=0, seed=7, source=1, frame=450, scale=1.0)
    with pytest.warns(UserWarning, match="truth_mode"):
        run_episode(env_f, PPOPolicy(ck), spec)


def test_run_eval_refuses_an_episode_list_of_another_mode(tmp_path):
    from srcloc_env.eval.episodes import make_episode_list, save_episode_list
    from srcloc_env.eval.run_eval import main as eval_main
    save_episode_list(make_episode_list([101], 2, 123, "T2"), tmp_path / "ep.csv")
    with pytest.raises(SystemExit, match="--mode"):
        eval_main(["--episodes", str(tmp_path / "ep.csv"), "--mode", "F", "--methods", "random", "--n-drones", "1", "--out-dir", str(tmp_path / "o"), "--processes", "1"])


def test_belief_alignment_is_the_cosine_between_the_heading_and_the_direction_to_the_belief_mean():
    from srcloc_env.rl.rollout import belief_alignment
    pos = np.array([[0.0, 0.0], [0.0, 0.0], [50.0, 50.0]])
    # drone 0 flies east towards a mean at (100, 0): +1; drone 1 flies west: -1; drone 2 stays: skipped
    total, n = belief_alignment(pos, np.array([100.0, 0.0]), np.array([0, 4, config.DRONE_N_ACTIONS - 1]))
    assert n == 2 and total == pytest.approx(0.0)
    total, n = belief_alignment(pos[:1], np.array([100.0, 0.0]), np.array([0]))
    assert (total, n) == (pytest.approx(1.0), 1)
    total, n = belief_alignment(pos[:1], np.array([100.0, 100.0]), np.array([1]))              # north-east heading, mean to the north-east
    assert total / n == pytest.approx(1.0)
    assert belief_alignment(pos[:1], np.array([0.4, 0.0]), np.array([0])) == (0.0, 0)          # already within 1 m of the mean


def test_collector_reports_belief_alignment_contacts_and_the_closest_approach(scene):
    col = _collector(scene, 2)
    ro = col.collect(ActorCritic(config.ENV_OBS_DIM + config.ENV_TEAMMATE_DIM), 40, 1)
    assert ro["n_team_steps"] == 40 and 0 <= ro["contact_steps"] <= 40 and ro["belief_cos_n"] > 0
    assert -1.0 <= ro["belief_cos_sum"] / ro["belief_cos_n"] <= 1.0
    assert all(e["closest_m"] >= 0.0 and 0 <= e["contact_steps"] <= e["length"] for e in ro["episodes"])


def test_train_passes_the_source_start_and_ppo_options_and_logs_the_diagnostics(scene, tmp_path, monkeypatch):
    out = _run(["--run-name", "opt", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "1",
                "--total-steps", "64", "--sources", "1", "2", "--start-plume-frac", "1.0", "--start-min-dist", "60", "--start-max-dist", "120", "--scale-range", "1", "2",
                "--minibatch", "16", "--epochs", "2", "--gamma", "1.0", "--gae-lambda", "0.97", "--target-kl", "0.02", "--log-every", "1"], scene, monkeypatch)
    assert _FakePool.last_env_kw == {"truth_mode": "F", "max_steps": config.MAX_EPISODE_STEPS, "sources": (1, 2), "start_plume_frac": 1.0,
                                     "start_min_dist": 60.0, "start_max_dist": 120.0, "scale_range": (1.0, 2.0)}
    cfgj = (tmp_path / "opt" / "config.json").read_text(encoding="utf-8")
    assert "\"gamma\": 1.0" in cfgj and "\"target_kl\": 0.02" in cfgj and "\"minibatch\": 16" in cfgj
    rows = _rows(tmp_path / "opt" / "train_log.csv")
    assert out["iterations"] == 2 and all(k in rows[0] for k in ("belief_cos", "contact_frac", "closest_m_ma", "n_success", "adv_std", "kl_stopped"))
    assert float(rows[0]["adv_std"]) > 0.0


def test_observation_v2_layout_bounds_and_the_two_drone_dimension(scene):
    env = make_train_env(scene, scene.reflected_scene(), 1, obs_version="v2", terminate_on_success=False, **{k: v for k, v in KW.items() if k not in ("terminate_on_success",)})
    obs, _ = env.reset(seed=11)
    assert env.obs_version == "v2" and obs.shape == (config.ENV_OBS_DIM_V2,) == (64,) and np.isfinite(obs).all()
    assert obs[:3].sum() == pytest.approx(1.0, abs=1e-5) and set(np.unique(obs[3:6])) <= {0.0, 1.0}          # GMM weights and valid mask
    assert obs[45] == 1.0 and obs[26] == pytest.approx(1.0, abs=1e-6)                                         # nothing detected yet, belief entropy H/H0 = 1
    assert float(obs[18:26].max()) >= np.cos(np.pi / 8) - 1e-6                                              # some heading points within 22.5 degrees of the belief mean
    rng = np.random.default_rng(0)
    for _ in range(12):
        obs, *_ = env.step(int(rng.choice(np.flatnonzero(env.action_mask()))))
        assert np.abs(obs).max() <= config.ENV_OBS_BOUND and np.abs(obs[30:45]).max() <= 1.5                 # recent-measurement offsets are O(1) now (v1: 0.004)
    env2 = make_train_env(scene, scene.reflected_scene(), 2, obs_version="v2", **KW)
    o2, _ = env2.reset(seed=5)
    assert o2.shape == (2, 67) == (2, config.agent_obs_dim("v2", 2))
    env1 = make_train_env(scene, scene.reflected_scene(), 1, **KW)
    assert env1.reset(seed=11)[0].shape == (config.ENV_OBS_DIM,) == (56,)                                      # v1 is unchanged


def test_obs_version_is_trained_saved_and_checked_at_evaluation(scene, tmp_path, monkeypatch):
    _run(["--run-name", "v2", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "2", "--total-steps", "32",
          "--obs-version", "v2"], scene, monkeypatch)
    assert _FakePool.last_env_kw["obs_version"] == "v2"
    ck = torch.load(tmp_path / "v2" / "final.pt", map_location="cpu", weights_only=False)
    assert ck["obs_version"] == "v2" and ck["learner"]["obs_dim"] == 67
    spec = EpisodeSpec(episode_id=0, seed=7, source=1, frame=450, scale=1.0)
    env_v2 = make_train_env(scene, scene.reflected_scene(), 2, obs_version="v2", **{**KW, "terminate_on_success": False})
    rec, _ = run_episode(env_v2, PPOPolicy(tmp_path / "v2" / "final.pt"), spec)
    assert 1 <= rec["steps"] <= 15 and rec["n_masked"] == 0                                                  # steps = first success step or the horizon
    env_v1 = make_train_env(scene, scene.reflected_scene(), 2, **KW)
    with pytest.raises(ValueError):
        run_episode(env_v1, PPOPolicy(tmp_path / "v2" / "final.pt"), spec)                                     # v2 policy on a v1 observation


def test_privileged_features_are_critic_only_and_train_saves_and_checks_the_options(scene, tmp_path, monkeypatch):
    env = make_train_env(scene, scene.reflected_scene(), 2, **KW)
    env.reset(seed=3, options={"source": 1, "reflect": False})
    pv = env.privileged()
    assert pv.shape == (2, config.ENV_PRIV_DIM) and pv.dtype == np.float32
    rel = env.truth_xy - env.xys[0]
    assert pv[0, 0] == pytest.approx(rel[0] / 1000.0, abs=1e-6) and pv[0, 2] == pytest.approx(np.log10(1.0 + np.hypot(*rel) / 50.0), abs=1e-5)
    assert env.reset(seed=3, options={"source": 1, "reflect": False})[0].shape == (2, config.agent_obs_dim("v1", 2))      # the observation does not contain them
    base = ["--run-name", "pv", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "2", "--priv-critic", "--value-norm"]
    _run(base + ["--total-steps", "64"], scene, monkeypatch)
    ck = torch.load(tmp_path / "pv" / "final.pt", map_location="cpu", weights_only=False)
    assert ck["learner"]["priv_dim"] == config.ENV_PRIV_DIM and ck["learner"]["value_norm"] is True
    rows = _rows(tmp_path / "pv" / "train_log.csv")
    assert len(rows) == 2 and np.isfinite(float(rows[-1]["value_loss"]))
    _run(base + ["--total-steps", "96", "--resume"], scene, monkeypatch)                                                   # resume keeps the critic type
    with pytest.raises(ValueError, match="priv-critic"):
        _run(["--run-name", "pv2", "--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "2",
              "--total-steps", "32", "--init-from", str(tmp_path / "pv" / "final.pt")], scene, monkeypatch)                # init from a privileged critic without the flag


def test_init_from_checks_the_obs_version_and_resume_checks_the_environment_options(scene, tmp_path, monkeypatch):
    base = ["--procs", "0", "--n-steps", "32", "--out-root", str(tmp_path), "--no-ckpt-eval", "--n-drones", "1"]
    _run(base + ["--run-name", "a", "--total-steps", "32"], scene, monkeypatch)                                               # v1
    with pytest.raises(ValueError, match="observation"):
        _run(base + ["--run-name", "b", "--total-steps", "32", "--obs-version", "v2", "--init-from", str(tmp_path / "a" / "final.pt")], scene, monkeypatch)
    _run(base + ["--run-name", "c", "--total-steps", "32", "--sources", "1"], scene, monkeypatch)
    with pytest.raises(ValueError, match="--resume"):
        _run(base + ["--run-name", "c", "--total-steps", "64", "--resume"], scene, monkeypatch)                              # forgot --sources 1
    _run(base + ["--run-name", "c", "--total-steps", "64", "--resume", "--sources", "1", "--lr", "1e-3"], scene, monkeypatch)     # new lr is honoured
    ck = torch.load(tmp_path / "c" / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["learner"]["opt_actor"]["param_groups"][0]["lr"] == pytest.approx(1e-3)
    assert '"train_sources": [\n    1\n  ]' in (tmp_path / "c" / "config.json").read_text(encoding="utf-8") or "\"train_sources\": [1]" in (tmp_path / "c" / "config.json").read_text(encoding="utf-8").replace("\n", "").replace(" ", "").replace("\"train_sources\":[1]", "\"train_sources\": [1]")
