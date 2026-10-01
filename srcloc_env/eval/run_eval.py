"""Evaluation runner: methods x drone counts over a common episode list (plan 5장 실험 매트릭스, S4; D9-2).

Usage: python -m srcloc_env.eval.run_eval --methods random lawnmower greedy_map gmm_infotaxis --n-drones 1 2
           [--n-per-source 10] [--sources 101 ...] [--episodes <csv>] [--processes 3] [--log-steps]
           [--out-dir <CACHE_DIR>/eval/<tag>] [--tag prelim] [--mode F|T2] [--scale-fixed 1.0] [--max-episodes N]
Outputs (out-dir): episodes.csv (the common list), records_<method>_<n>drones.csv (one row per episode),
summary.json (aggregates per configuration, paired differences vs the first method, Table 2 markdown, timing), and with
--log-steps steps/<method>_<n>drones_ep<id>.npz (per-step drone xy, counts, actions, applied, GMM parameters, MAP,
entropy, top sigma, success flag; input of figure 7 and the 3-D episode video).

Every configuration is evaluated on exactly the same EpisodeSpec list (source, frame, scale, reset seed -> drone
starts), so comparisons are paired.  Episodes are distributed over processes one by one; each worker builds the Scene
once (initializer) and keeps one environment per drone count.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from srcloc_env import config
from srcloc_env.baselines.policies import POLICIES, VERIFICATION_POLICIES, make_policy
from srcloc_env.env.multi_agent import MultiDroneEnv
from srcloc_env.env.source_env import Scene, SourceLocEnv
from srcloc_env.eval.episodes import EpisodeSpec, load_episode_list, make_episode_list, save_episode_list
from srcloc_env.eval.metrics import aggregate, paired_differences, table2_markdown

RECORD_FIELDS = ["method", "n_drones", "episode_id", "seed", "source", "frame", "scale", "mode", "start_type", "success", "steps",
                 "success_strict", "steps_strict", "min_error_m",
                 "final_error_m", "first_detection_step", "declared_step", "declared_error_m", "path_length_m", "n_masked",
                 "entropy_final", "top_sigma_final_m", "wall_s", "step_ms_median"]


def make_env(scene: Scene, n_drones: int, mode: str) -> SourceLocEnv:
    kw = dict(sources=config.ALL_SOURCES, truth_mode=mode, reflect_prob=0.0, terminate_on_success=not config.EVAL_NO_EARLY_STOP)
    return MultiDroneEnv(scene, n_drones=n_drones, **kw) if n_drones > 1 else SourceLocEnv(scene, **kw)


def run_episode(env: SourceLocEnv, policy, spec: EpisodeSpec, log_steps: bool = False) -> tuple[dict, dict | None]:
    """One episode of ``policy`` on ``env`` for ``spec``; returns (record, step log or None)."""
    t_wall = time.perf_counter()
    obs, info = env.reset(seed=spec.seed, options=spec.reset_options())
    policy.reset(env, info)
    n = int(getattr(env, "n_drones", 1))
    pos_prev = info["drone_xy"].reshape(n, 2).copy()
    path_len = 0.0
    n_masked = 0
    first_det = declared_step = None
    first_ok = first_strict = None
    min_err = float("inf")
    start_type = info.get("start_type", "")
    declared_err = None
    step_ms = []
    log: dict[str, list] = {k: [] for k in ("drone_xy", "y", "action", "applied", "gmm_w", "gmm_mu", "gmm_cov", "gmm_mask",
                                              "map_xy", "entropy", "top_sigma", "map_error")} if log_steps else None
    terminated = truncated = False
    while not (terminated or truncated):
        acts = policy.act(env, obs, info)
        t0 = time.perf_counter()
        obs, r, terminated, truncated, info = env.step(acts if n > 1 else int(np.atleast_1d(acts)[0]))
        step_ms.append((time.perf_counter() - t0) * 1e3)
        pos = info["drone_xy"].reshape(n, 2)
        path_len += float(np.sum(np.hypot(*(pos - pos_prev).T)))
        pos_prev = pos.copy()
        applied = info["applied"] if n > 1 else [info["applied"]]
        n_masked += sum(1 for a in applied if not a)
        ys = info["y"] if n > 1 else [info["y"]]
        if first_det is None and any(bool(env.det.is_detection(y)) for y in ys):
            first_det = info["t"]
        if first_ok is None and info["success"]:
            first_ok = info["t"]
        if first_strict is None and info["success_strict"]:
            first_strict = info["t"]
        min_err = min(min_err, float(info["map_error_m"]))
        if declared_step is None and info["top_sigma_m"] < env.success_sigma:
            declared_step, declared_err = info["t"], float(info["map_error_m"])
        if log is not None:
            g = env.gmm
            log["drone_xy"].append(pos.copy()); log["y"].append(np.array(ys)); log["action"].append(np.atleast_1d(acts).copy())
            log["applied"].append(np.array(applied)); log["gmm_w"].append(g.weights.copy()); log["gmm_mu"].append(g.means.copy())
            log["gmm_cov"].append(g.covs.copy()); log["gmm_mask"].append(g.mask.copy()); log["map_xy"].append(env.pf.map_estimate().copy())
            log["entropy"].append(env.h_prev); log["top_sigma"].append(g.top_sigma()); log["map_error"].append(float(info["map_error_m"]))
    rec = {"method": policy.name, "n_drones": n, "episode_id": spec.episode_id, "seed": spec.seed, "source": spec.source,
           "frame": spec.frame, "scale": spec.scale, "mode": spec.mode, "start_type": start_type, "success": first_ok is not None,
           "steps": int(first_ok if first_ok is not None else info["t"]), "success_strict": first_strict is not None,
           "steps_strict": int(first_strict if first_strict is not None else info["t"]), "min_error_m": min_err,
           "final_error_m": float(info["map_error_m"]), "first_detection_step": first_det, "declared_step": declared_step,
           "declared_error_m": declared_err, "path_length_m": path_len, "n_masked": n_masked, "entropy_final": float(env.h_prev),
           "top_sigma_final_m": float(info["top_sigma_m"]), "wall_s": time.perf_counter() - t_wall,
           "step_ms_median": float(np.median(step_ms)) if step_ms else float("nan")}
    if log is not None:
        log = {k: np.asarray(v) for k, v in log.items()}
        log.update({"truth_xy": info["truth_xy"], "source": spec.source, "frame": spec.frame, "scale": spec.scale,
                    "seed": spec.seed, "success": first_ok is not None, "method": policy.name, "n_drones": n, "start_type": start_type})
    return rec, log


# ------------------------------------------------------------------------------------------ workers
_SCENE: Scene | None = None
_ENVS: dict[tuple[int, str], SourceLocEnv] = {}


def _init_worker() -> None:
    global _SCENE
    _SCENE = Scene.load()


def _worker(task: dict) -> dict:
    global _SCENE
    if _SCENE is None:
        _SCENE = Scene.load()
    key = (int(task["n_drones"]), task["mode"])
    if key not in _ENVS:
        _ENVS[key] = make_env(_SCENE, key[0], key[1])
    env = _ENVS[key]
    spec = EpisodeSpec(**task["spec"])
    policy = make_policy(task["method"])
    rec, log = run_episode(env, policy, spec, log_steps=task["log_steps"])
    if log is not None:
        p = Path(task["step_dir"]) / f"{task['method']}_{key[0]}drones_ep{spec.episode_id:04d}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **{k: v for k, v in log.items() if isinstance(v, np.ndarray)},
                            meta=json.dumps({k: v for k, v in log.items() if not isinstance(v, np.ndarray)}))
        rec["step_log"] = str(p)
    return rec


def write_records(path: Path, recs: list[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RECORD_FIELDS + ["step_log"], extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="+", default=list(POLICIES), choices=list(POLICIES) + list(VERIFICATION_POLICIES))
    ap.add_argument("--n-drones", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-per-source", type=int, default=config.EVAL_PRELIM_EPISODES_PER_SOURCE)
    ap.add_argument("--sources", type=int, nargs="*", default=list(config.ALL_SOURCES))
    ap.add_argument("--episodes", type=Path, default=None, help="existing episodes.csv (overrides --n-per-source / --sources)")
    ap.add_argument("--base-seed", type=int, default=config.EVAL_BASE_SEED)
    ap.add_argument("--mode", choices=list(config.ENV_MODES), default="F")
    ap.add_argument("--scale-fixed", type=float, default=None)
    ap.add_argument("--processes", type=int, default=config.EVAL_PROCESSES)
    ap.add_argument("--log-steps", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None, help="truncate the list (smoke tests)")
    ap.add_argument("--tag", default="prelim")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")            # Windows cp949 console: never crash on a non-ASCII table
    out_dir = args.out_dir if args.out_dir is not None else config.CACHE_DIR / "eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.episodes is not None:
        specs = load_episode_list(args.episodes)
    else:
        specs = make_episode_list(args.sources, args.n_per_source, args.base_seed, args.mode, scale_fixed=args.scale_fixed)
    if args.max_episodes is not None:
        specs = specs[:args.max_episodes]
    save_episode_list(specs, out_dir / "episodes.csv")
    tasks = [{"method": m, "n_drones": n, "mode": args.mode, "spec": spec.__dict__, "log_steps": args.log_steps,
              "step_dir": str(out_dir / "steps")}
             for m in args.methods for n in args.n_drones for spec in specs]
    print(f"[eval] {len(specs)} episodes x {len(args.methods)} methods x {args.n_drones} drones = {len(tasks)} runs on "
          f"{args.processes} processes -> {out_dir}", flush=True)
    t0 = time.perf_counter()
    if args.processes > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.processes, initializer=_init_worker) as pool:
            recs = []
            for i, rec in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1)):
                recs.append(rec)
                if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                    print(f"[eval] {i + 1}/{len(tasks)} done, {time.perf_counter() - t0:.0f} s", flush=True)
    else:
        recs = [_worker(t) for t in tasks]
    total_s = time.perf_counter() - t0
    by_cfg: dict[str, list[dict]] = {}
    for r in recs:
        by_cfg.setdefault(f"{r['method']} ({r['n_drones']})", []).append(r)
    for label, rs in by_cfg.items():
        m, n = rs[0]["method"], rs[0]["n_drones"]
        write_records(out_dir / f"records_{m}_{n}drones.csv", sorted(rs, key=lambda r: r["episode_id"]))
    agg = {label: aggregate(rs) for label, rs in by_cfg.items()}
    agg_by_start = {st: {label: aggregate(rs, start_type=st) for label, rs in by_cfg.items()} for st in ("plume", "random")}
    first = args.methods[0]
    paired = {label: paired_differences(rs, by_cfg[f"{first} ({rs[0]['n_drones']})"])
              for label, rs in by_cfg.items() if f"{first} ({rs[0]['n_drones']})" in by_cfg and rs[0]["method"] != first}
    timing = {label: {"step_ms_median": float(np.median([r["step_ms_median"] for r in rs])),
                      "episode_wall_s_median": float(np.median([r["wall_s"] for r in rs]))} for label, rs in by_cfg.items()}
    table = table2_markdown(agg)
    summary = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": args.tag, "mode": args.mode, "methods": args.methods,
               "n_drones": args.n_drones, "n_episodes": len(specs), "sources": sorted({s.source for s in specs}),
               "aggregates": agg, "aggregates_by_start_type": agg_by_start, "paired_vs_first_method": paired, "timing": timing, "table2_markdown": table,
               "table2_markdown_by_start_type": {st: table2_markdown(a) for st, a in agg_by_start.items()},
               "total_seconds": total_s, "processes": args.processes}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                          encoding="utf-8")
    print(table)
    for st, a in agg_by_start.items():
        print(f"--- start type: {st} ---")
        print(table2_markdown(a, groups=("train", "holdout", "all_observable")))
    print(f"[eval] total {total_s:.0f} s; summary {out_dir / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
