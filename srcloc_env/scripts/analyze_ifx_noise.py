"""Is the one-step expected-entropy score of GMM-Infotaxis signal or Monte-Carlo noise? (D9-4)

For a set of states of real 1-drone GMM-Infotaxis episodes (belief rebuilt by replaying the logged actions) the nine action
scores are computed repeatedly with independent random draws, for several numbers of predictive samples S (config default 10).
Reported per S: the spread of the MEAN scores between actions (the information difference the policy hopes to exploit), the
noise (standard deviation of one action's score over the repetitions), their ratio, the share of repetitions that choose the
same action (argmin) as the majority, and the mean rank correlation between repetitions.

Usage: python -m srcloc_env.scripts.analyze_ifx_noise [--tag t2_4_v2] [--reps 8] [--samples 10 40 160]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from srcloc_env import config
from srcloc_env.baselines.policies import GmmInfotaxisPolicy
from srcloc_env.env.source_env import Scene, SourceLocEnv
from srcloc_env.eval.episodes import load_episode_list


def pick_states(tag: str, n_episodes: int = 4) -> list[tuple[int, int, str]]:
    """(episode_id, step, label): per chosen plume-start episode with a clear signal three states - early, just before and
    after the first clear signal (y > 100) - so that flat and informative situations are both represented."""
    out = []
    d = config.CACHE_DIR / "eval" / tag
    seen_src = set()
    with (d / "records_gmm_infotaxis_1drones.csv").open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["start_type"] != "plume" or int(r["source"]) in seen_src or int(r["source"]) in config.EXCLUDED_SOURCES:
                continue
            z = np.load(r["step_log"], allow_pickle=False)
            hit = np.flatnonzero(z["y"][:, 0] > 100)
            if hit.size == 0:
                continue
            t1 = int(hit[0])
            ep = int(r["episode_id"])
            out += [(ep, 8, "early"), (ep, max(t1 - 3, 1), "before first signal"), (ep, min(t1 + 25, 250), "after first signal")]
            seen_src.add(int(r["source"]))
            if len(seen_src) >= n_episodes:
                break
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="t2_4_v2")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--samples", type=int, nargs="+", default=[10, 40, 160])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    d = config.CACHE_DIR / "eval" / args.tag
    specs = {sp.episode_id: sp for sp in load_episode_list(d / "episodes.csv")}
    scene = Scene.load()
    states = pick_states(args.tag)
    rows = []
    for ep, step, label in states:
        z = np.load(d / "steps" / f"gmm_infotaxis_1drones_ep{ep:04d}.npz", allow_pickle=False)
        env = SourceLocEnv(scene, sources=config.ALL_SOURCES, truth_mode="F", reflect_prob=0.0, terminate_on_success=False)
        sp = specs[ep]
        obs, info = env.reset(seed=sp.seed, options=sp.reset_options())
        for t in range(step):
            obs, r, term, trunc, info = env.step(int(z["action"][t, 0]))
        mask = env.action_mask()
        for S in args.samples:
            sc = []
            for rep in range(args.reps):
                pol = GmmInfotaxisPolicy(n_samples=S, tie_tol=0.0, seed=1000 * S + rep)
                pol.reset(env, info)
                pol.rng = np.random.default_rng([ep, step, S, rep])
                pol.act(env, obs, info)
                sc.append(pol.last_scores[0][mask])
            sc = np.array(sc)                                                   # (reps, n_allowed)
            mean = sc.mean(axis=0)
            spread = float(mean.max() - mean.min())
            noise = float(sc.std(axis=0, ddof=1).mean())
            argmins = sc.argmin(axis=1)
            maj = float(np.bincount(argmins).max() / len(argmins))
            cors = [spearmanr(sc[i], sc[j])[0] for i in range(len(sc)) for j in range(i + 1, len(sc))]
            rows.append({"episode": ep, "step": step, "label": label, "S": S, "n_allowed": int(mask.sum()), "mean_spread_nats": spread,
                         "noise_sd_nats": noise, "snr": spread / noise if noise > 0 else float("inf"), "argmin_majority_share": maj,
                         "rank_corr": float(np.nanmean(cors))})
            print(f"ep {ep:3d} step {step:3d} {label:20s} S {S:4d}: spread {spread:.4f} noise {noise:.4f} ratio {spread / noise:5.2f} | same-argmin share {maj:.2f} | rank corr {np.nanmean(cors):+.2f}", flush=True)
    summ = {}
    for S in args.samples:
        sel = [r for r in rows if r["S"] == S]
        summ[S] = {k: float(np.median([r[k] for r in sel])) for k in ("mean_spread_nats", "noise_sd_nats", "snr", "argmin_majority_share", "rank_corr")}
        print(f"S {S:4d} median over {len(sel)} states: spread {summ[S]['mean_spread_nats']:.4f} nats, noise {summ[S]['noise_sd_nats']:.4f} nats, ratio {summ[S]['snr']:.2f}, "
              f"same-argmin share {summ[S]['argmin_majority_share']:.2f}, rank corr {summ[S]['rank_corr']:+.2f}")
    out = {"states": [{"episode": e, "step": t, "label": l} for e, t, l in states], "rows": rows, "median_by_S": summ}
    path = args.out if args.out is not None else config.CACHE_DIR / "eval" / "ifx_noise.json"
    Path(path).write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
