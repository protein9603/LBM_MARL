"""Parameter-shared PPO for the multi-drone source-localisation task (plan 4.6, S3; D10-1).

One policy is shared by all drones: every drone contributes its own (observation, action) transitions to the same batch and the
team reward is the reward of each of its transitions (the shared-belief entropy reduction is a team quantity, plan 4.5).  This is the
parameter-shared IPPO / MAPPO-without-central-critic form of Yu et al. 2022 (value function of the own observation of the agent, which for
n >= 2 already contains the relative positions and latest counts of the teammates).

Implementation choices (each one is a standard PPO ingredient; see docs/references.md):
- clipped surrogate objective, GAE(lambda), several epochs of minibatch updates (Schulman et al. 2017, 2016);
- actor and critic are separate tanh MLPs with orthogonal initialisation (sqrt(2) hidden, 0.01 policy head, 1 value head) and their own
  Adam optimiser and gradient clipping, because the critic gradient (returns of order 10) would otherwise dominate a joint clip
  (Andrychowicz et al. 2021 "What matters in on-policy RL"; Yu et al. 2022);
- invalid actions are removed by a logit mask (large negative logit -> probability exactly 0), the same mask is stored with the
  transition and re-applied in the update, so the ratio pi_new / pi_old is computed over the same support;
- per-iteration advantage standardisation; no value clipping, no learning-rate annealing, no observation / reward normalisation
  (the observation is already O(1) and the reward is a normalised information gain; plan 4.6);
- the time fraction is part of the observation, so the 300-step truncation is a terminal state of a time-augmented MDP and is NOT
  bootstrapped (done = terminated or truncated).

Pure PyTorch (cpu); no environment code is imported here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from srcloc_env import config


def _mlp(sizes: Sequence[int], gain_hidden: float, gain_out: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(lin.weight, gain_out if last else gain_hidden)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if not last:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor (obs -> n_actions logits) and critic (obs -> value) MLPs."""

    def __init__(self, obs_dim: int, n_actions: int = config.DRONE_N_ACTIONS, hidden: Sequence[int] = config.PPO_HIDDEN) -> None:
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = int(obs_dim), int(n_actions), tuple(int(h) for h in hidden)
        self.actor = _mlp([self.obs_dim, *self.hidden, self.n_actions], config.PPO_ORTHO_GAIN_HIDDEN, config.PPO_ORTHO_GAIN_POLICY)
        self.critic = _mlp([self.obs_dim, *self.hidden, 1], config.PPO_ORTHO_GAIN_HIDDEN, config.PPO_ORTHO_GAIN_VALUE)

    # ------------------------------------------------------------------ distribution
    def masked_logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalised log-probabilities over the allowed actions (masked entries -> about -1e9, probability 0)."""
        logits = self.actor(obs).masked_fill(~mask, config.PPO_LOGIT_MASK_VALUE)
        return torch.log_softmax(logits, dim=-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, mask: torch.Tensor, generator: torch.Generator | None = None,
            deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(action, log-prob of the action, value) for a batch of observations (no gradient)."""
        with torch.no_grad():
            logp = self.masked_logits(obs, mask)
            if deterministic:
                a = logp.argmax(dim=-1)
            else:
                a = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
            return a, logp.gather(-1, a.unsqueeze(-1)).squeeze(-1), self.value(obs)

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(log-prob of ``act``, entropy of the masked policy, value) with gradients."""
        logp = self.masked_logits(obs, mask)
        ent = -(logp.exp() * logp).sum(-1)
        return logp.gather(-1, act.unsqueeze(-1)).squeeze(-1), ent, self.value(obs)

    # ------------------------------------------------------------------ curriculum: widen the input (1 -> n drones)
    def widened(self, new_obs_dim: int) -> "ActorCritic":
        """Copy of this network that accepts ``new_obs_dim`` inputs.  The extra inputs (teammate features) get zero weights, so the
        widened network computes exactly the same function while those inputs are ignored (M1 -> M2 initialisation)."""
        if new_obs_dim < self.obs_dim:
            raise ValueError("cannot narrow the input")
        new = ActorCritic(new_obs_dim, self.n_actions, self.hidden)
        sd_old, sd_new = self.state_dict(), new.state_dict()
        for key, w in sd_old.items():
            if w.shape == sd_new[key].shape:
                sd_new[key] = w.clone()
            else:                                                   # first layer weight (hidden, obs_dim) of actor and critic
                pad = torch.zeros_like(sd_new[key])
                pad[:, :self.obs_dim] = w
                sd_new[key] = pad
        new.load_state_dict(sd_new)
        return new


@dataclass
class PPOConfig:
    n_steps: int = config.PPO_N_STEPS
    minibatch: int = config.PPO_MINIBATCH
    epochs: int = config.PPO_EPOCHS
    lr: float = config.PPO_LR
    gamma: float = config.PPO_GAMMA
    gae_lambda: float = config.PPO_GAE_LAMBDA
    clip: float = config.PPO_CLIP
    vf_coef: float = config.PPO_VF_COEF
    ent_coef: float = config.PPO_ENT_COEF
    max_grad_norm: float = config.PPO_MAX_GRAD_NORM
    adv_normalise: bool = config.PPO_ADV_NORMALISE
    target_kl: float | None = None              # stop the remaining epochs when the minibatch approx KL exceeds 1.5 x this value (None = never)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------ GAE
def compute_gae(rew: np.ndarray, val: np.ndarray, done: np.ndarray, last_val: np.ndarray, gamma: float, lam: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation for one worker batch.

    rew (T,) team reward, val (T, n) value of each agent before step t, done (T,) the step ended the episode, last_val (n,) value of
    the observation after the last step (used only when the last step did not end an episode).  Returns (advantage (T, n), return (T, n))
    with return = advantage + value.  The reward is the same for all n agents; episode ends are not bootstrapped (module docstring).
    """
    T, n = val.shape
    adv = np.zeros((T, n), dtype=np.float64)
    last = np.zeros(n)
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(done[t])
        next_val = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * next_val * nonterminal - val[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + val


# ------------------------------------------------------------------------------------------ batch + update
@dataclass
class Batch:
    """Flattened transitions of all workers: every row is one (agent, step) pair."""
    obs: torch.Tensor           # (N, D) float32
    act: torch.Tensor           # (N,) int64
    logp: torch.Tensor          # (N,) float32 behaviour log-probability
    mask: torch.Tensor          # (N, A) bool
    adv: torch.Tensor           # (N,) float32 (not yet standardised)
    ret: torch.Tensor           # (N,) float32
    val: torch.Tensor           # (N,) float32 behaviour value (diagnostic: explained variance)


def make_batch(rollouts: Sequence[dict[str, np.ndarray]], gamma: float, lam: float) -> Batch:
    """Rollouts (one dict per worker: obs (T, n, D), act (T, n), logp (T, n), val (T, n), mask (T, n, A), rew (T,), done (T,),
    last_val (n,)) -> GAE per worker, then one flat batch."""
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("obs", "act", "logp", "mask", "adv", "ret", "val")}
    for r in rollouts:
        adv, ret = compute_gae(r["rew"], r["val"], r["done"], r["last_val"], gamma, lam)
        D, A = r["obs"].shape[-1], r["mask"].shape[-1]
        cols["obs"].append(r["obs"].reshape(-1, D))
        cols["act"].append(r["act"].reshape(-1))
        cols["logp"].append(r["logp"].reshape(-1))
        cols["mask"].append(r["mask"].reshape(-1, A))
        cols["adv"].append(adv.reshape(-1))
        cols["ret"].append(ret.reshape(-1))
        cols["val"].append(r["val"].reshape(-1))
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    return Batch(obs=torch.from_numpy(cat["obs"].astype(np.float32)), act=torch.from_numpy(cat["act"].astype(np.int64)),
                 logp=torch.from_numpy(cat["logp"].astype(np.float32)), mask=torch.from_numpy(cat["mask"].astype(bool)),
                 adv=torch.from_numpy(cat["adv"].astype(np.float32)), ret=torch.from_numpy(cat["ret"].astype(np.float32)),
                 val=torch.from_numpy(cat["val"].astype(np.float32)))


class PPOLearner:
    """Holds the network and the two optimisers and performs one PPO update (``update``) on a Batch."""

    def __init__(self, net: ActorCritic, cfg: PPOConfig | None = None, seed: int = 0) -> None:
        self.net = net
        self.cfg = cfg if cfg is not None else PPOConfig()
        self.opt_actor = torch.optim.Adam(net.actor.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.opt_critic = torch.optim.Adam(net.critic.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.gen = torch.Generator().manual_seed(int(seed))             # minibatch shuffling

    def update(self, b: Batch) -> dict[str, float]:
        c = self.cfg
        n = b.obs.shape[0]
        adv = b.adv
        if c.adv_normalise:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats: dict[str, list[float]] = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")}
        self.net.train()
        stopped = False
        for _ in range(c.epochs):
            if stopped:
                break
            perm = torch.randperm(n, generator=self.gen)
            for s in range(0, n, c.minibatch):
                idx = perm[s:s + c.minibatch]
                logp, ent, val = self.net.evaluate(b.obs[idx], b.mask[idx], b.act[idx])
                log_ratio = logp - b.logp[idx]
                ratio = log_ratio.exp()
                a = adv[idx]
                pg = torch.max(-a * ratio, -a * ratio.clamp(1.0 - c.clip, 1.0 + c.clip)).mean()
                mse = (val - b.ret[idx]).pow(2).mean()
                actor_loss = pg - c.ent_coef * ent.mean()
                critic_loss = c.vf_coef * mse
                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                actor_loss.backward()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.net.actor.parameters(), c.max_grad_norm)
                nn.utils.clip_grad_norm_(self.net.critic.parameters(), c.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()
                with torch.no_grad():
                    stats["policy_loss"].append(float(pg))
                    stats["value_loss"].append(float(mse))
                    stats["entropy"].append(float(ent.mean()))
                    stats["approx_kl"].append(float(((ratio - 1.0) - log_ratio).mean()))
                    stats["clip_frac"].append(float(((ratio - 1.0).abs() > c.clip).float().mean()))
                    if c.target_kl is not None and stats["approx_kl"][-1] > 1.5 * c.target_kl:
                        stopped = True
                        break
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        var = float(b.ret.var())
        out["kl_stopped"] = float(stopped)
        out["explained_var"] = float(1.0 - (b.ret - b.val).var() / var) if var > 1e-12 else float("nan")
        return out

    # ------------------------------------------------------------------ persistence
    def state(self) -> dict[str, Any]:
        return {"net": self.net.state_dict(), "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
                "obs_dim": self.net.obs_dim, "n_actions": self.net.n_actions, "hidden": list(self.net.hidden)}

    def load_optimizers(self, st: dict[str, Any]) -> None:
        self.opt_actor.load_state_dict(st["opt_actor"])
        self.opt_critic.load_state_dict(st["opt_critic"])


def net_from_state(st: dict[str, Any]) -> ActorCritic:
    net = ActorCritic(int(st["obs_dim"]), int(st["n_actions"]), tuple(st["hidden"]))
    net.load_state_dict(st["net"])
    return net
