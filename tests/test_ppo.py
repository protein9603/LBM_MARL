"""D10: parameter-shared PPO (rl/ppo.py): masked policy, GAE, update direction, input widening (M1 to M2)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from srcloc_env import config
from srcloc_env.rl.ppo import ActorCritic, PPOConfig, PPOLearner, compute_gae, make_batch, net_from_state


def test_gae_matches_hand_computation():
    gamma, lam = 0.9, 0.8
    rew = np.array([1.0, 0.0, 2.0])
    val = np.array([[0.5], [0.4], [0.3]])
    done = np.array([False, False, False])
    last = np.array([0.2])
    adv, ret = compute_gae(rew, val, done, last, gamma, lam)
    d2 = 2.0 + gamma * 0.2 - 0.3
    d1 = 0.0 + gamma * 0.3 - 0.4
    d0 = 1.0 + gamma * 0.4 - 0.5
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    assert adv[:, 0] == pytest.approx([a0, a1, a2])
    assert ret[:, 0] == pytest.approx(adv[:, 0] + val[:, 0])


def test_gae_does_not_bootstrap_across_an_episode_end():
    gamma, lam = 0.99, 0.95
    rew = np.array([1.0, 5.0])
    val = np.array([[0.0], [0.0]])
    adv, _ = compute_gae(rew, val, np.array([True, False]), np.array([100.0]), gamma, lam)
    assert adv[0, 0] == pytest.approx(1.0)                       # the terminal step ignores everything after it
    assert adv[1, 0] == pytest.approx(5.0 + gamma * 100.0)       # the last step bootstraps from last_val


def test_gae_team_reward_is_shared_by_two_agents():
    rew = np.array([1.0, 1.0])
    val = np.array([[0.0, 1.0], [0.0, 1.0]])
    adv, ret = compute_gae(rew, val, np.array([False, True]), np.zeros(2), 0.9, 0.9)
    assert adv.shape == (2, 2)
    assert ret[1, 0] == pytest.approx(1.0) and ret[1, 1] == pytest.approx(1.0)      # both agents see the same terminal reward


def test_masked_actions_have_probability_zero_and_are_never_sampled():
    torch.manual_seed(0)
    net = ActorCritic(10)
    obs = torch.randn(64, 10)
    mask = torch.ones(64, config.DRONE_N_ACTIONS, dtype=torch.bool)
    mask[:, 2] = False
    mask[::2, 5] = False
    p = net.masked_logits(obs, mask).exp().detach()
    assert float(p[:, 2].max()) == 0.0 and float(p[::2, 5].max()) == 0.0
    assert p.sum(-1).tolist() == pytest.approx([1.0] * 64, abs=1e-5)
    gen = torch.Generator().manual_seed(1)
    for _ in range(20):
        a, logp, _ = net.act(obs, mask, gen)
        assert bool(mask[torch.arange(64), a].all())
        assert torch.isfinite(logp).all()


def test_entropy_of_a_single_allowed_action_is_zero_and_initial_policy_is_near_uniform():
    net = ActorCritic(8)
    obs = torch.randn(5, 8)
    one = torch.zeros(5, config.DRONE_N_ACTIONS, dtype=torch.bool)
    one[:, 8] = True
    with torch.no_grad():
        _, ent, _ = net.evaluate(obs, one, torch.full((5,), 8))
    assert float(ent.abs().max()) < 1e-6
    full = torch.ones(5, config.DRONE_N_ACTIONS, dtype=torch.bool)
    with torch.no_grad():
        _, ent, _ = net.evaluate(obs, full, torch.zeros(5, dtype=torch.long))
    assert float(ent.min()) > 0.98 * np.log(config.DRONE_N_ACTIONS)


def test_deterministic_act_is_the_argmax_of_the_masked_policy():
    net = ActorCritic(6)
    obs = torch.randn(7, 6)
    mask = torch.ones(7, config.DRONE_N_ACTIONS, dtype=torch.bool)
    mask[:, 0] = False
    a, _, _ = net.act(obs, mask, deterministic=True)
    assert a.tolist() == net.masked_logits(obs, mask).argmax(-1).tolist()


def test_widened_network_is_the_same_function_and_ignores_the_new_inputs():
    torch.manual_seed(3)
    net = ActorCritic(56)
    wide = net.widened(59)
    obs = torch.randn(9, 56)
    extra = torch.randn(9, 3) * 10.0
    mask = torch.ones(9, config.DRONE_N_ACTIONS, dtype=torch.bool)
    lp0 = net.masked_logits(obs, mask)
    lp1 = wide.masked_logits(torch.cat([obs, extra], dim=1), mask)
    assert torch.allclose(lp0, lp1, atol=1e-6)
    assert torch.allclose(net.value(obs), wide.value(torch.cat([obs, extra], dim=1)), atol=1e-6)
    with pytest.raises(ValueError):
        wide.widened(56)


def test_state_round_trip():
    net = ActorCritic(12)
    learner = PPOLearner(net, PPOConfig(), seed=0)
    net2 = net_from_state(learner.state())
    obs = torch.randn(4, 12)
    mask = torch.ones(4, config.DRONE_N_ACTIONS, dtype=torch.bool)
    assert torch.equal(net.masked_logits(obs, mask), net2.masked_logits(obs, mask))


def _bandit_batch(net: ActorCritic, n: int, gen: torch.Generator):
    ctx = torch.randint(0, 4, (n,), generator=gen)
    obs = torch.nn.functional.one_hot(ctx, 4).float()
    mask = torch.ones(n, config.DRONE_N_ACTIONS, dtype=torch.bool)
    mask[:, 3:] = False                                              # three allowed actions
    a, logp, v = net.act(obs, mask, gen)
    rew = (a == ctx % 3).float()                                      # the rewarded action depends on the context
    from srcloc_env.rl.ppo import Batch
    return Batch(obs=obs, act=a, logp=logp, mask=mask, adv=rew - v, ret=rew, val=v), float(rew.mean())


def test_ppo_solves_a_contextual_bandit_and_diagnostics_are_sane():
    torch.manual_seed(0)
    net = ActorCritic(4, hidden=(32, 32))
    learner = PPOLearner(net, PPOConfig(minibatch=128, epochs=4, ent_coef=0.0), seed=0)
    gen = torch.Generator().manual_seed(5)
    first = None
    for it in range(80):
        b, rate = _bandit_batch(net, 512, gen)
        st = learner.update(b)
        first = rate if first is None else first
        assert 0.0 <= st["clip_frac"] <= 1.0 and np.isfinite(st["value_loss"]) and st["approx_kl"] > -1e-3
    assert first < 0.5 and rate > 0.9                                # about 1/3 at the start, solved afterwards


def test_update_raises_the_probability_of_positive_advantage_actions():
    torch.manual_seed(1)
    net = ActorCritic(4, hidden=(16, 16))
    learner = PPOLearner(net, PPOConfig(minibatch=64, epochs=1, ent_coef=0.0, lr=1e-2), seed=0)
    gen = torch.Generator().manual_seed(2)
    b, _ = _bandit_batch(net, 64, gen)
    b.adv = torch.where(b.act == 0, torch.tensor(1.0), torch.tensor(-1.0))
    before = net.evaluate(b.obs, b.mask, b.act)[0].detach()
    learner.update(b)
    after = net.evaluate(b.obs, b.mask, b.act)[0].detach()
    assert float((after - before)[b.act == 0].mean()) > 0.0 > float((after - before)[b.act != 0].mean())


def test_make_batch_flattens_agents_and_workers():
    T, n, D, A = 6, 2, 5, config.DRONE_N_ACTIONS

    def ro(seed: int) -> dict:
        r = np.random.default_rng(seed)
        return {"obs": r.normal(size=(T, n, D)), "act": r.integers(0, A, (T, n)), "logp": -np.abs(r.normal(size=(T, n))),
                "val": r.normal(size=(T, n)), "mask": np.ones((T, n, A), bool), "rew": r.normal(size=T),
                "done": np.array([0, 0, 1, 0, 0, 0], bool), "last_val": r.normal(size=n)}

    b = make_batch([ro(0), ro(1), ro(2)], 0.99, 0.95)
    assert b.obs.shape == (3 * T * n, D) and b.act.shape == (3 * T * n,) and b.mask.shape == (3 * T * n, A)
    assert torch.isfinite(b.adv).all() and torch.isfinite(b.ret).all()


def test_target_kl_stops_the_remaining_epochs_early():
    torch.manual_seed(2)
    net = ActorCritic(4, hidden=(16, 16))
    learner = PPOLearner(net, PPOConfig(minibatch=32, epochs=6, ent_coef=0.0, lr=5e-2, target_kl=1e-4), seed=0)
    gen = torch.Generator().manual_seed(3)
    b, _ = _bandit_batch(net, 256, gen)
    st = learner.update(b)
    assert st["kl_stopped"] == 1.0
    free = PPOLearner(ActorCritic(4, hidden=(16, 16)), PPOConfig(minibatch=32, epochs=6, ent_coef=0.0, lr=5e-2), seed=0).update(b)
    assert free["kl_stopped"] == 0.0


def test_privileged_critic_is_critic_only_and_widening_keeps_the_function():
    torch.manual_seed(4)
    net = ActorCritic(10, hidden=(16, 16), priv_dim=5)
    obs, priv = torch.randn(6, 10), torch.randn(6, 5)
    mask = torch.ones(6, config.DRONE_N_ACTIONS, dtype=torch.bool)
    with pytest.raises(ValueError):
        net.value(obs)                                                       # the critic needs its privileged features
    _, _, v0 = net.act(obs, mask)                                            # the actor path (evaluation) works without them; the value is a placeholder
    assert torch.equal(v0, torch.zeros(6))
    a0 = net.masked_logits(obs, mask)
    assert not torch.allclose(net.value(obs, priv), net.value(obs, priv * 0.0))
    wide = net.widened(13)
    extra = torch.randn(6, 3) * 5.0
    assert torch.allclose(wide.masked_logits(torch.cat([obs, extra], 1), mask), a0, atol=1e-6)
    assert torch.allclose(wide.value(torch.cat([obs, extra], 1), priv), net.value(obs, priv), atol=1e-6)    # privileged columns kept behind the widened observation
    st = PPOLearner(net, PPOConfig(), seed=0).state()
    assert st["priv_dim"] == 5 and net_from_state(st).priv_dim == 5


def test_value_norm_tracks_the_return_scale_and_update_uses_normalised_targets():
    torch.manual_seed(5)
    net = ActorCritic(4, hidden=(16, 16), value_norm=True)
    learner = PPOLearner(net, PPOConfig(minibatch=128, epochs=2, ent_coef=0.0, lr=3e-3), seed=0)
    gen = torch.Generator().manual_seed(6)
    for _ in range(25):
        b, _ = _bandit_batch(net, 256, gen)
        b.ret = b.ret * 5.0 + 100.0                                          # returns far from zero and with a larger scale
        st = learner.update(b)
    assert float(net.vn_mean) == pytest.approx(102.0, abs=3.0) and 1.0 < float(net.vn_std) < 4.0
    obs = torch.nn.functional.one_hot(torch.tensor([0, 1, 2]), 4).float()
    assert 90.0 < float(net.value(obs).mean()) < 112.0                       # the critic reports values in return units
    assert np.isfinite(st["value_loss"]) and st["value_loss"] < 50.0
