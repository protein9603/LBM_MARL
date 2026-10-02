"""Rollout collection for PPO: training environments, a per-worker collector and a pool of worker processes (D10-2).

RolloutCollector drives ONE training environment (make_train_env: Mode F, the 9 TRAIN_SOURCES, y-reflection with probability
config.ENV_REFLECT_PROB_TRAIN, termination at the primary success) with the current network and returns the trajectory arrays that
rl.ppo.make_batch expects.  Episodes run on across rollouts (the policy changes between iterations, as in any on-policy
algorithm); a new episode starts right after a terminal step.

Seeds (spec 1.3): the environment seed of episode k of worker p in run r is 1e6 r + 1e5 p + k; the action-sampling generator of one
rollout is seeded from (r, p, iteration), so a rollout is a deterministic function of the weights, the episode counter and the seeds.

RolloutPool runs the collectors in n_procs spawned processes (torch single-threaded in each) and sends them the network
weights every iteration; n_procs = 0 runs one collector in the calling process (tests, smoke runs).
"""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

import numpy as np
import torch

from srcloc_env import config
from srcloc_env.env.drone import ACTION_STAY, heading_unit_vectors
from srcloc_env.env.multi_agent import MultiDroneEnv
from srcloc_env.env.source_env import Scene, SourceLocEnv, default_max_steps, load_scene
from srcloc_env.rl.ppo import ActorCritic


def make_train_env(scene: Scene, scene_reflected: Scene, n_drones: int, **kw: Any) -> SourceLocEnv:
    opts = dict(sources=config.TRAIN_SOURCES, truth_mode="F", scene_reflected=scene_reflected,
                reflect_prob=config.ENV_REFLECT_PROB_TRAIN, terminate_on_success=True)
    opts.update(kw)
    opts.setdefault("max_steps", default_max_steps(opts["truth_mode"]))    # Mode T2: config.T2_MAX_STEPS (frames 400 + t stay inside the cache)
    return MultiDroneEnv(scene, n_drones=n_drones, **opts) if n_drones > 1 else SourceLocEnv(scene, **opts)


def episode_seed(run_seed: int, proc: int, episode_idx: int) -> int:
    if not (0 <= proc < 10 and 0 <= episode_idx < config.TRAIN_SEED_PROC_STRIDE):
        raise ValueError("proc must be 0..9 and episode_idx below the process stride")
    return config.TRAIN_SEED_RUN_STRIDE * int(run_seed) + config.TRAIN_SEED_PROC_STRIDE * int(proc) + int(episode_idx)


_HEADINGS = heading_unit_vectors()


def belief_alignment(positions: np.ndarray, belief_mean: np.ndarray, actions: np.ndarray, min_dist_m: float = 1.0) -> tuple[float, int]:
    """(sum of cosines, count): cosine between the heading of every moving drone's action and the direction from the drone to the top GMM
    component mean (stay actions and drones within min_dist_m of the mean are skipped).  Random policy: expected 0, greedy-MAP: about 0.7
    (D11 diagnostic: the PPO policy of the 2-drone T2 run had -0.05 .. +0.08)."""
    total, count = 0.0, 0
    for xy, a in zip(np.asarray(positions, dtype=float).reshape(-1, 2), np.asarray(actions).reshape(-1)):
        if int(a) == ACTION_STAY:
            continue
        u = np.asarray(belief_mean, dtype=float) - xy
        d = float(np.hypot(*u))
        if d < min_dist_m:
            continue
        total += float(_HEADINGS[int(a)] @ u) / d
        count += 1
    return total, count


def _rollout_generator(run_seed: int, proc: int, iteration: int) -> torch.Generator:
    seed = int(np.random.SeedSequence([int(run_seed), int(proc), int(iteration), 11]).generate_state(1)[0])
    return torch.Generator().manual_seed(seed)


class RolloutCollector:
    def __init__(self, env: SourceLocEnv, run_seed: int, proc: int, episode_idx: int = 0) -> None:
        self.env, self.run_seed, self.proc = env, int(run_seed), int(proc)
        self.n = int(getattr(env, "n_drones", 1))
        self.episode_idx = int(episode_idx)              # index of the NEXT episode to start
        self.obs: np.ndarray | None = None
        self.ep_return = 0.0
        self.ep_info = 0.0
        self.ep_shaping = 0.0
        self.ep_len = 0
        self.ep_closest = float("inf")
        self.ep_contact = 0

    def _masks(self) -> np.ndarray:
        return self.env.action_masks() if self.n > 1 else self.env.action_mask()[None]

    def _reset(self) -> None:
        obs, _ = self.env.reset(seed=episode_seed(self.run_seed, self.proc, self.episode_idx))
        self.episode_idx += 1
        self.obs = np.asarray(obs, dtype=np.float32).reshape(self.n, -1)
        self.ep_return, self.ep_info, self.ep_shaping, self.ep_len = 0.0, 0.0, 0.0, 0
        self.ep_closest, self.ep_contact = float("inf"), 0

    def collect(self, net: ActorCritic, n_steps: int, iteration: int) -> dict[str, Any]:
        gen = _rollout_generator(self.run_seed, self.proc, iteration)
        if self.obs is None:
            self._reset()
        D, A = self.obs.shape[1], config.DRONE_N_ACTIONS
        buf: dict[str, Any] = {"obs": np.zeros((n_steps, self.n, D), np.float32), "act": np.zeros((n_steps, self.n), np.int64),
                               "logp": np.zeros((n_steps, self.n), np.float32), "val": np.zeros((n_steps, self.n), np.float32),
                               "mask": np.zeros((n_steps, self.n, A), bool), "rew": np.zeros(n_steps), "done": np.zeros(n_steps, bool)}
        episodes: list[dict[str, Any]] = []
        masked_share = 0.0
        n_applied_masked = 0
        cos_sum, cos_n, contact_steps = 0.0, 0, 0
        t0 = time.perf_counter()
        net.eval()
        for t in range(n_steps):
            mask = self._masks()
            priv = torch.from_numpy(self.env.privileged()) if net.priv_dim else None          # critic-only features (asymmetric critic)
            if priv is not None:
                buf.setdefault("priv", np.zeros((n_steps, self.n, net.priv_dim), np.float32))[t] = priv.numpy()
            a, logp, v = net.act(torch.from_numpy(self.obs), torch.from_numpy(mask), gen, priv=priv)
            buf["obs"][t], buf["mask"][t] = self.obs, mask
            buf["act"][t], buf["logp"][t], buf["val"][t] = a.numpy(), logp.numpy(), v.numpy()
            masked_share += float(1.0 - mask.mean())
            acts = a.numpy()
            pos = self.env.xys if self.n > 1 else self.env.xy[None]
            c_s, c_n = belief_alignment(pos, self.env.gmm.means[0], acts)       # diagnostic: does the policy steer towards the belief mean?
            cos_sum, cos_n = cos_sum + c_s, cos_n + c_n
            obs, r, term, trunc, info = self.env.step(acts if self.n > 1 else int(acts[0]))
            ys = info["y"] if self.n > 1 else [info["y"]]
            hit = any(int(y) >= config.DIAG_CONTACT_COUNTS for y in ys)         # real plume contact (false-alarm probability about 1e-8)
            contact_steps, self.ep_contact = contact_steps + int(hit), self.ep_contact + int(hit)
            self.ep_closest = min(self.ep_closest, float(np.hypot(*(np.asarray(info["drone_xy"]).reshape(-1, 2) - info["truth_xy"]).T).min()))
            applied = info["applied"] if self.n > 1 else [info["applied"]]
            n_applied_masked += sum(1 for x in applied if not x)          # must stay 0: the policy never samples a masked action
            self.ep_return += float(r)
            self.ep_info += config.ENV_REWARD_INFO * float(info["info_gain"])
            self.ep_shaping += float(info.get("shaping_reward", 0.0))
            self.ep_len += 1
            buf["rew"][t] = r
            done = bool(term or trunc)
            buf["done"][t] = done
            if done:
                episodes.append({"proc": self.proc, "episode_idx": self.episode_idx - 1, "ret": self.ep_return, "ret_info": self.ep_info, "ret_time": config.ENV_REWARD_TIME * self.ep_len,
                                 "ret_shaping": self.ep_shaping, "ret_terminal": self.ep_return - self.ep_info - self.ep_shaping - config.ENV_REWARD_TIME * self.ep_len, "length": self.ep_len,
                                 "success": bool(info["success"]), "success_strict": bool(info.get("success_strict", False)),
                                 "source": int(info["source"]), "reflected": bool(info["reflected"]), "start_type": info.get("start_type", ""),
                                 "final_error_m": float(info["map_error_m"]), "top_sigma_m": float(info["top_sigma_m"]),
                                 "entropy_drop": float(1.0 - info["entropy"] / max(self.env.h0, 1e-9)), "truncated": bool(trunc),
                                 "closest_m": self.ep_closest, "contact_steps": self.ep_contact})
                self._reset()
            else:
                self.obs = np.asarray(obs, dtype=np.float32).reshape(self.n, -1)
        with torch.no_grad():
            priv_last = torch.from_numpy(self.env.privileged()) if net.priv_dim else None
            buf["last_val"] = net.value(torch.from_numpy(self.obs), priv_last).numpy().astype(np.float64)
        buf["val"] = buf["val"].astype(np.float64)
        buf["masked_share"] = masked_share / n_steps
        buf["applied_masked"] = n_applied_masked
        buf["belief_cos_sum"], buf["belief_cos_n"], buf["contact_steps"], buf["n_team_steps"] = cos_sum, cos_n, contact_steps, n_steps
        buf["wall_s"] = time.perf_counter() - t0
        buf["episodes"] = episodes
        buf["episode_idx"] = self.episode_idx
        return buf


# ------------------------------------------------------------------------------------------ worker processes
def _net_from_payload(p: dict[str, Any]) -> ActorCritic:
    net = ActorCritic(int(p["obs_dim"]), int(p["n_actions"]), tuple(p["hidden"]), int(p.get("priv_dim", 0)), bool(p.get("value_norm", False)))
    net.load_state_dict({k: torch.from_numpy(v) for k, v in p["state"].items()})
    return net


def net_payload(net: ActorCritic) -> dict[str, Any]:
    return {"obs_dim": net.obs_dim, "n_actions": net.n_actions, "hidden": list(net.hidden), "priv_dim": net.priv_dim, "value_norm": net.value_norm,
            "state": {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}}


def _worker_main(conn, n_drones: int, run_seed: int, proc: int, episode_idx: int, env_kw: dict[str, Any]) -> None:   # noqa: ANN001
    torch.set_num_threads(1)
    try:
        scene = load_scene(env_kw.get("truth_mode", "F"))
        env = make_train_env(scene, scene.reflected_scene(), n_drones, **env_kw)
        col = RolloutCollector(env, run_seed, proc, episode_idx)
        conn.send(("ready", proc))
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                break
            net = _net_from_payload(payload["net"])
            conn.send(("rollout", col.collect(net, int(payload["n_steps"]), int(payload["iteration"]))))
    except BaseException as exc:                                           # noqa: BLE001 - report to the parent, then die
        import traceback
        conn.send(("error", traceback.format_exc()))
        raise exc
    finally:
        conn.close()


class RolloutPool:
    """n_procs worker processes (or one in-process collector when n_procs == 0)."""

    def __init__(self, n_drones: int, run_seed: int, n_procs: int = config.PPO_N_PROCS, episode_idx: list[int] | None = None,
                 env_kw: dict[str, Any] | None = None) -> None:
        self.n_drones, self.run_seed, self.n_procs = int(n_drones), int(run_seed), int(n_procs)
        self.env_kw = dict(env_kw or {})
        idx0 = list(episode_idx) if episode_idx is not None else [0] * max(self.n_procs, 1)
        self.procs: list[Any] = []
        self.conns: list[Any] = []
        self.local: RolloutCollector | None = None
        if self.n_procs == 0:
            scene = load_scene(self.env_kw.get("truth_mode", "F"))
            env = make_train_env(scene, scene.reflected_scene(), self.n_drones, **self.env_kw)
            self.local = RolloutCollector(env, self.run_seed, 0, idx0[0])
        else:
            ctx = mp.get_context("spawn")
            for p in range(self.n_procs):
                parent, child = ctx.Pipe()
                pr = ctx.Process(target=_worker_main, args=(child, self.n_drones, self.run_seed, p, idx0[p], self.env_kw), daemon=True)
                pr.start()
                child.close()
                self.procs.append(pr)
                self.conns.append(parent)
            for c in self.conns:
                tag, payload = c.recv()
                if tag != "ready":
                    raise RuntimeError(f"worker failed to start:\n{payload}")

    def collect(self, net: ActorCritic, n_steps: int, iteration: int) -> list[dict[str, Any]]:
        if self.local is not None:
            return [self.local.collect(net, n_steps, iteration)]
        payload = {"net": net_payload(net), "n_steps": int(n_steps), "iteration": int(iteration)}
        for c in self.conns:
            c.send(("rollout", payload))
        out = []
        for c in self.conns:
            tag, res = c.recv()
            if tag != "rollout":
                raise RuntimeError(f"rollout worker failed:\n{res}")
            out.append(res)
        return out

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self.procs, self.conns = [], []
