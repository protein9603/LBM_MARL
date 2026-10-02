"""PPO training driver: parameter-shared policy, 1 or n drones (plan 4.6, S3; D10-3).

Usage (M1: one drone, 1 M steps; M2: two drones from the M1 weights, 0.5 M team steps):
    python -m srcloc_env.rl.train --n-drones 1 --run-seed 1 --total-steps 1000000 --run-name m1_s1
    python -m srcloc_env.rl.train --n-drones 2 --run-seed 1 --total-steps 500000 --run-name m2_s1 --init-from <run>/final.pt

Per iteration: every worker collects n_steps team steps with the current weights, GAE per worker, one PPO update on the pooled batch.
Run directory (config.TRAIN_ROOT / run-name): config.json, train_log.csv (one row per iteration), episodes.csv (one row per finished
training episode), ckpt_<steps>.pt (every --ckpt-every-steps and every config.TRAIN_CKPT_INTERVAL_S seconds), latest.pt, final.pt,
ckpt_eval/<ckpt>.csv (background quick evaluation of every checkpoint; curves only).  --resume continues from latest.pt.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from srcloc_env import config
from srcloc_env.rl.ppo import ActorCritic, PPOConfig, PPOLearner, make_batch, net_from_state
from srcloc_env.env.source_env import default_max_steps
from srcloc_env.rl.rollout import RolloutPool

LOG_FIELDS = ["iteration", "env_steps", "wall_s", "rollout_s", "update_s", "steps_per_s", "episodes_total", "success_ma", "strict_ma",
              "return_ma", "length_ma", "entropy_drop_ma", "final_error_ma", "policy_loss", "value_loss", "entropy", "approx_kl",
              "clip_frac", "explained_var", "masked_share", "applied_masked"]
EP_FIELDS = ["env_steps", "proc", "episode_idx", "source", "reflected", "start_type", "success", "success_strict", "truncated", "length",
             "ret", "ret_info", "ret_time", "ret_terminal", "final_error_m", "top_sigma_m", "entropy_drop"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:   # noqa: BLE001
        return "unknown"


def save_checkpoint(path: Path, learner: PPOLearner, extra: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"learner": learner.state(), **extra}, tmp)
    tmp.replace(path)


def _append_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def _launch_ckpt_eval(ckpt: Path, out_dir: Path, n_per_source: int, limit: int | None, log_per_source: int = 0) -> subprocess.Popen:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "srcloc_env.rl.ckpt_eval", "--ckpt", str(ckpt), "--out", str(out_dir / (ckpt.stem + ".csv")),
           "--n-per-source", str(n_per_source)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if log_per_source:
        cmd += ["--log-per-source", str(log_per_source)]
    log = (out_dir / (ckpt.stem + ".log")).open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def _ma(buf: collections.deque, key: str) -> float:
    return float(np.mean([e[key] for e in buf])) if buf else float("nan")


def train(a: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(a.out_root) / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    n = int(a.n_drones)
    mode = str(a.truth_mode)
    max_steps = int(a.max_steps) if a.max_steps else default_max_steps(mode)
    obs_dim = int(config.ENV_OBS_DIM + config.ENV_TEAMMATE_DIM * (n - 1))
    cfg = PPOConfig(n_steps=int(a.n_steps))
    torch.set_num_threads(1)
    torch.manual_seed(int(a.run_seed))
    latest = run_dir / "latest.pt"
    resume = bool(a.resume) and latest.exists()
    it, steps, ep_idx = 0, 0, [0] * max(int(a.procs), 1)
    init_note = "random orthogonal initialisation"
    if resume:
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        net = net_from_state(ck["learner"])
        ck_mode = str(ck.get("truth_mode", "F"))
        ck_steps = int(ck.get("max_steps") or default_max_steps(ck_mode))
        if (ck_mode, ck_steps) != (mode, max_steps):
            raise ValueError(f"--resume: {latest} was trained with truth mode {ck_mode} and {ck_steps}-step episodes, this call asks for {mode} / {max_steps}; pass the same --truth-mode / --max-steps")
        it, steps, ep_idx = int(ck["iteration"]), int(ck["env_steps"]), list(ck["episode_idx"])
        init_note = f"resumed from {latest} at iteration {it}"
    elif a.init_from:
        ck0 = torch.load(Path(a.init_from), map_location="cpu", weights_only=False)
        net0 = net_from_state(ck0["learner"])
        net = net0.widened(obs_dim)
        ck0_mode = str(ck0.get("truth_mode", "F"))
        if ck0_mode != mode:
            print(f"[train] WARNING: --init-from checkpoint was trained in truth mode {ck0_mode}, this run uses {mode}", flush=True)
        init_note = f"initialised from {a.init_from} (input widened {net0.obs_dim} -> {obs_dim})"
    else:
        net = ActorCritic(obs_dim)
    if net.obs_dim != obs_dim:
        raise ValueError(f"network inputs {net.obs_dim} do not match {n} drone(s) ({obs_dim})")
    learner = PPOLearner(net, cfg, seed=int(a.run_seed) * 100_003 + it)
    if resume:
        learner.load_optimizers(ck["learner"])
    info = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "ppo": cfg.to_dict(), "obs_dim": obs_dim,
            "n_drones": n, "init": init_note, "git_commit": _git_commit(), "torch": torch.__version__,
            "train_sources": list(config.TRAIN_SOURCES), "reflect_prob": config.ENV_REFLECT_PROB_TRAIN,
            "success_sigma_m": config.ENV_SUCCESS_SIGMA_M, "success_error_m": config.ENV_SUCCESS_ERROR_M, "truth_mode": mode, "max_steps": max_steps}
    cfg_name = "config.json" if not resume else f"config_resume_{it}.json"
    (run_dir / cfg_name).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[train] {a.run_name}: {n} drone(s), truth {mode}, {max_steps}-step episodes, obs {obs_dim}, {a.procs} worker(s), {cfg.n_steps} steps each, total {a.total_steps} team steps; {init_note}", flush=True)

    pool = RolloutPool(n, int(a.run_seed), int(a.procs), episode_idx=ep_idx, env_kw={"truth_mode": mode, "max_steps": max_steps})
    ma: collections.deque = collections.deque(maxlen=int(a.ma_window))
    every = int(a.ckpt_every_steps) if a.ckpt_every_steps else max(int(a.total_steps) // 10, 1)
    next_step_ckpt = (steps // every + 1) * every
    t_start = t_ckpt = time.perf_counter()
    pending: list[Path] = []
    running: list[subprocess.Popen | None] = [None]
    all_evals: list[subprocess.Popen] = []

    def checkpoint(final: bool = False) -> None:
        extra = {"iteration": it, "env_steps": steps, "episode_idx": list(ep_idx), "n_drones": n, "run_name": a.run_name, "ppo": cfg.to_dict(),
                 "truth_mode": mode, "max_steps": max_steps}
        path = run_dir / ("final.pt" if final else f"ckpt_{steps:08d}.pt")
        save_checkpoint(path, learner, extra)
        save_checkpoint(latest, learner, extra)
        if a.ckpt_eval:
            pending.append(path)
        print(f"[train] checkpoint {path.name} (iteration {it}, {steps} team steps)", flush=True)

    def pump_evals() -> None:
        if pending and (running[0] is None or running[0].poll() is not None):
            running[0] = _launch_ckpt_eval(pending.pop(0), run_dir / "ckpt_eval", int(a.ckpt_eval_per_source), a.ckpt_eval_limit, int(a.ckpt_eval_log_per_source))
            all_evals.append(running[0])

    try:
        while steps < int(a.total_steps):
            it += 1
            t0 = time.perf_counter()
            ro = pool.collect(net, cfg.n_steps, it)
            t1 = time.perf_counter()
            stats = learner.update(make_batch(ro, cfg.gamma, cfg.gae_lambda))
            t2 = time.perf_counter()
            steps += cfg.n_steps * len(ro)
            ep_idx = [r["episode_idx"] for r in ro]
            eps = [e for r in ro for e in r["episodes"]]
            ma.extend(eps)
            _append_rows(run_dir / "episodes.csv", EP_FIELDS, [{**e, "env_steps": steps} for e in eps])
            s_ma, st_ma, r_ma, l_ma, d_ma = (_ma(ma, k) for k in ("success", "success_strict", "ret", "length", "entropy_drop"))
            sps = cfg.n_steps * len(ro) / (t2 - t0)
            row = {"iteration": it, "env_steps": steps, "wall_s": t2 - t_start, "rollout_s": t1 - t0, "update_s": t2 - t1,
                   "steps_per_s": sps, "episodes_total": len(eps), "success_ma": s_ma, "strict_ma": st_ma, "return_ma": r_ma,
                   "length_ma": l_ma, "entropy_drop_ma": d_ma, "final_error_ma": _ma(ma, "final_error_m"),
                   "masked_share": float(np.mean([r["masked_share"] for r in ro])), "applied_masked": int(sum(r["applied_masked"] for r in ro)), **stats}
            _append_rows(run_dir / "train_log.csv", LOG_FIELDS, [row])
            if it % int(a.log_every) == 0:
                ent, kl, clip, ev = stats["entropy"], stats["approx_kl"], stats["clip_frac"], stats["explained_var"]
                print(f"[train] it {it:4d} steps {steps:8d} | success {s_ma:.3f} strict {st_ma:.3f} ret {r_ma:+.2f} len {l_ma:.0f} drop {d_ma:.2f}"
                      f" | ent {ent:.3f} kl {kl:.4f} clip {clip:.2f} ev {ev:+.2f} | {t2 - t0:.1f} s/it ({sps:.0f} steps/s)", flush=True)
            now = time.perf_counter()
            if steps >= next_step_ckpt or now - t_ckpt >= config.TRAIN_CKPT_INTERVAL_S:
                if steps < int(a.total_steps):
                    checkpoint()
                next_step_ckpt = (steps // every + 1) * every
                t_ckpt = now
            pump_evals()
        checkpoint(final=True)
    finally:
        pool.close()
    while pending or (running[0] is not None and running[0].poll() is None):
        pump_evals()
        time.sleep(2.0)
    for pr in all_evals:
        pr.wait()
    wall = time.perf_counter() - t_start
    print(f"[train] done: {steps} team steps in {wall / 60:.1f} min; run directory {run_dir}", flush=True)
    return {"run_dir": str(run_dir), "env_steps": steps, "iterations": it, "wall_s": wall}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-drones", type=int, default=1)
    ap.add_argument("--run-seed", type=int, default=1)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--total-steps", type=int, default=config.TRAIN_M1_STEPS, help="team (environment) steps")
    ap.add_argument("--procs", type=int, default=config.PPO_N_PROCS, help="rollout worker processes (0 = in this process)")
    ap.add_argument("--n-steps", type=int, default=config.PPO_N_STEPS, help="team steps per worker per iteration")
    ap.add_argument("--truth-mode", choices=list(config.ENV_MODES), default=config.ENV_TRUTH_MODE_DEFAULT,
                    help="F = one frozen frame per episode, T2 = time-varying truth (frame 400 + t, D11)")
    ap.add_argument("--max-steps", type=int, default=0, help="episode horizon (0 = 300 in Mode F, config.T2_MAX_STEPS in Mode T2)")
    ap.add_argument("--init-from", type=Path, default=None, help="checkpoint whose weights initialise the network (M1 to M2)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-root", type=Path, default=config.TRAIN_ROOT)
    ap.add_argument("--ma-window", type=int, default=100, help="episodes of the moving averages")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--ckpt-every-steps", type=int, default=0, help="checkpoint every N team steps (0 = total / 10); a checkpoint is also written every 30 min")
    ap.add_argument("--no-ckpt-eval", dest="ckpt_eval", action="store_false", help="skip the background quick evaluation of checkpoints")
    ap.add_argument("--ckpt-eval-per-source", type=int, default=config.EVAL_CKPT_EPISODES_PER_SOURCE)
    ap.add_argument("--ckpt-eval-log-per-source", type=int, default=1, help="step logs of the first N quick-evaluation episodes of every source (sample trajectories)")
    ap.add_argument("--ckpt-eval-limit", type=int, default=None, help="evaluate only the first N episodes of the list (smoke tests)")
    return ap


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    return train(args)


if __name__ == "__main__":
    main()
