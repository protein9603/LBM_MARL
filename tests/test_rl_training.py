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

    def __init__(self, n_drones, run_seed, n_procs, episode_idx=None, env_kw=None):   # noqa: ANN001
        self.col = RolloutCollector(make_train_env(self.scene, self.scene.reflected_scene(), n_drones, **KW), run_seed, 0, (episode_idx or [0])[0])

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
