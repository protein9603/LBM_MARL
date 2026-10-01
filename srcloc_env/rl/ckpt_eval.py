"""Quick evaluation of one training checkpoint (curves only; never used to select a checkpoint or a seed; spec 6, D10-4).

12 observable sources x config.EVAL_CKPT_EPISODES_PER_SOURCE episodes of Mode F from the fixed list
make_episode_list(..., base_seed=config.EVAL_CKPT_BASE_SEED) (disjoint from the final evaluation list), full 300 steps without early
stop, the same environment / metrics as eval/run_eval.py, in ONE process (it runs next to the training workers).
Output: <out>.csv (one record per episode, run_eval format) and a one-line summary on stdout.

Usage: python -m srcloc_env.rl.ckpt_eval --ckpt PATH --out PATH.csv [--n-per-source 5] [--limit N]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.env.source_env import Scene
from srcloc_env.eval.episodes import make_episode_list
from srcloc_env.eval.run_eval import make_env, run_episode, write_records
from srcloc_env.rl.ppo_policy import PPOPolicy


def checkpoint_sources() -> list[int]:
    return [s for s in config.ALL_SOURCES if s not in config.EXCLUDED_SOURCES]


def evaluate_checkpoint(ckpt: str | Path, out_csv: str | Path, n_per_source: int = config.EVAL_CKPT_EPISODES_PER_SOURCE,
                        limit: int | None = None, deterministic: bool = config.PPO_EVAL_DETERMINISTIC) -> dict:
    pol = PPOPolicy(ckpt, alias="ppo", deterministic=deterministic)
    n_drones = int(round((pol.net.obs_dim - config.ENV_OBS_DIM) / config.ENV_TEAMMATE_DIM)) + 1
    specs = make_episode_list(checkpoint_sources(), n_per_source, config.EVAL_CKPT_BASE_SEED, "F")
    if limit is not None:
        specs = specs[:limit]
    env = make_env(Scene.load(), n_drones, "F")
    t0 = time.perf_counter()
    recs = []
    for sp in specs:
        rec, _ = run_episode(env, pol, sp)
        rec.update({"ckpt": str(ckpt), "env_steps": pol.meta.get("env_steps"), "group": "holdout" if sp.source in config.HOLDOUT_SOURCES else "train"})
        recs.append(rec)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    write_records(Path(out_csv), recs)
    out = {"n": len(recs), "wall_s": time.perf_counter() - t0, "n_drones": n_drones, "env_steps": pol.meta.get("env_steps")}
    for g in ("train", "holdout"):
        sel = [r for r in recs if r["group"] == g]
        out[f"success_{g}"] = float(np.mean([r["success"] for r in sel])) if sel else float("nan")
        out[f"strict_{g}"] = float(np.mean([r["success_strict"] for r in sel])) if sel else float("nan")
    out["success_all"] = float(np.mean([r["success"] for r in recs]))
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-source", type=int, default=config.EVAL_CKPT_EPISODES_PER_SOURCE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args(argv)
    res = evaluate_checkpoint(args.ckpt, args.out, args.n_per_source, args.limit, args.deterministic or config.PPO_EVAL_DETERMINISTIC)
    print("[ckpt_eval] " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()), flush=True)
    return res


if __name__ == "__main__":
    main()
