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
from srcloc_env.env.source_env import Scene, SourceLocEnv, default_max_steps, load_scene
from srcloc_env.eval.episodes import EpisodeSpec, load_episode_list, make_episode_list, save_episode_list
from srcloc_env.eval.metrics import aggregate, paired_differences, table2_markdown

RECORD_FIELDS = ["method", "n_drones", "episode_id", "seed", "source", "frame", "scale", "mode", "start_type", "success", "steps",
                 "success_strict", "steps_strict", "min_error_m",
                 "final_error_m", "first_detection_step", "declared_step", "declared_error_m", "path_length_m", "n_masked",
                 "entropy_final", "top_sigma_final_m", "wall_s", "step_ms_median", "tie_tol", "tie_frac", "all_tied_frac"]


def make_env(scene: Scene, n_drones: int, mode: str, max_steps: int | None = None) -> SourceLocEnv:
    kw = dict(sources=config.ALL_SOURCES, truth_mode=mode, reflect_prob=0.0, terminate_on_success=not config.EVAL_NO_EARLY_STOP,
              max_steps=int(max_steps) if max_steps is not None else default_max_steps(mode))
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
    if hasattr(policy, "stats"):
        rec.update(policy.stats())                                         # e.g. GMM-Infotaxis tie statistics
    if log is not None:
        log = {k: np.asarray(v) for k, v in log.items()}
        log.update({"truth_xy": info["truth_xy"], "source": spec.source, "frame": spec.frame, "scale": spec.scale,
                    "seed": spec.seed, "success": first_ok is not None, "method": policy.name, "n_drones": n, "start_type": start_type})
    return rec, log


# ------------------------------------------------------------------------------------------ workers
_SCENES: dict[str, Scene] = {}
_ENVS: dict[tuple[int, str, int], SourceLocEnv] = {}


def _init_worker() -> None:
    pass                                                       # scenes are loaded lazily per truth mode


def _worker(task: dict) -> dict:
    mode = task["mode"]
    if mode not in _SCENES:
        _SCENES[mode] = load_scene(mode)
    max_steps = int(task.get("max_steps") or default_max_steps(mode))
    key = (int(task["n_drones"]), mode, max_steps)
    if key not in _ENVS:
        _ENVS[key] = make_env(_SCENES[mode], key[0], mode, max_steps)
    env = _ENVS[key]
    spec = EpisodeSpec(**task["spec"])
    kw = task.get("policy_kw", {}).get(task["method"], {})
    if "checkpoint" in kw:                                              # trained PPO checkpoint under an alias (rl/ppo_policy.py)
        from srcloc_env.rl.ppo_policy import PPOPolicy
        policy = PPOPolicy(kw["checkpoint"], alias=task["method"], deterministic=bool(kw.get("deterministic", config.PPO_EVAL_DETERMINISTIC)))
    else:
        policy = make_policy(task["method"], **kw)
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


_INT = ("n_drones", "episode_id", "seed", "source", "frame", "steps", "steps_strict", "n_masked")
_FLOAT = ("scale", "final_error_m", "min_error_m", "path_length_m", "entropy_final", "top_sigma_final_m", "wall_s", "step_ms_median",
          "tie_tol", "tie_frac", "all_tied_frac")
_OPT_INT = ("first_detection_step", "declared_step")
_BOOL = ("success", "success_strict")


def _cast(row: dict) -> dict:
    """CSV strings -> typed record (inverse of write_records)."""
    out = dict(row)
    for k in _INT:
        out[k] = int(float(row[k]))
    for k in _FLOAT:
        out[k] = float(row[k]) if row.get(k) not in (None, "") else float("nan")
    for k in _OPT_INT:
        out[k] = int(float(row[k])) if row.get(k) not in (None, "", "None") else None
    out["declared_error_m"] = float(row["declared_error_m"]) if row.get("declared_error_m") not in (None, "", "None") else None
    for k in _BOOL:
        out[k] = str(row[k]) == "True"
    return out


def load_records(out_dir: Path) -> list[dict]:
    recs = []
    for f in sorted(Path(out_dir).glob("records_*.csv")):
        with f.open("r", encoding="utf-8", newline="") as fh:
            recs += [_cast(r) for r in csv.DictReader(fh)]
    return recs


def import_records(out_dir: Path, spec: str) -> int:
    """spec = 'TAG:method[:newname]': copy records_<method>_<n>drones.csv of <CACHE_DIR>/eval/<TAG> into out_dir, renaming the method."""
    parts = spec.split(":")
    tag, method = parts[0], parts[1]
    new = parts[2] if len(parts) > 2 else method
    n = 0
    for f in sorted((config.CACHE_DIR / "eval" / tag).glob(f"records_{method}_[0-9]drones.csv")):      # [0-9]: "lawnmower" must not match "lawnmower_alongwind"
        with f.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["method"] = new
        nd = rows[0]["n_drones"]
        with (Path(out_dir) / f"records_{new}_{nd}drones.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        n += len(rows)
    return n


def build_summary(out_dir: Path, recs: list[dict], tag: str, mode: str, total_s: float, processes: int, methods: list[str],
                  max_steps: int | None = None) -> dict:
    max_steps = int(max_steps) if max_steps is not None else default_max_steps(mode)
    by_cfg: dict[str, list[dict]] = {}
    for r in recs:
        by_cfg.setdefault(f"{r['method']} ({r['n_drones']})", []).append(r)
    for label, rs in by_cfg.items():
        m, n = rs[0]["method"], rs[0]["n_drones"]
        write_records(out_dir / f"records_{m}_{n}drones.csv", sorted(rs, key=lambda r: r["episode_id"]))
    agg = {label: aggregate(rs, max_steps=max_steps) for label, rs in by_cfg.items()}
    agg_by_start = {st: {label: aggregate(rs, max_steps=max_steps, start_type=st) for label, rs in by_cfg.items()} for st in ("plume", "random")}
    first = methods[0]
    paired = {label: paired_differences(rs, by_cfg[f"{first} ({rs[0]['n_drones']})"], max_steps)
              for label, rs in by_cfg.items() if f"{first} ({rs[0]['n_drones']})" in by_cfg and rs[0]["method"] != first}
    timing = {label: {"step_ms_median": float(np.median([r["step_ms_median"] for r in rs])),
                      "episode_wall_s_median": float(np.median([r["wall_s"] for r in rs]))} for label, rs in by_cfg.items()}
    table = table2_markdown(agg)
    summary = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "mode": mode, "max_steps": max_steps, "methods": methods,
               "n_drones": sorted({r["n_drones"] for r in recs}), "n_episodes": len({r["episode_id"] for r in recs}),
               "sources": sorted({r["source"] for r in recs}),
               "aggregates": agg, "aggregates_by_start_type": agg_by_start, "paired_vs_first_method": paired, "timing": timing, "table2_markdown": table,
               "table2_markdown_by_start_type": {st: table2_markdown(a) for st, a in agg_by_start.items()},
               "total_seconds": total_s, "processes": processes}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                          encoding="utf-8")
    print(table)
    for st, a in agg_by_start.items():
        print(f"--- start type: {st} ---")
        print(table2_markdown(a, groups=("train", "holdout", "all_observable")))
    print(f"[eval] total {total_s:.0f} s; summary {out_dir / 'summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="*", default=list(POLICIES), choices=list(POLICIES) + list(VERIFICATION_POLICIES))
    ap.add_argument("--ppo", nargs="*", default=[], metavar="ALIAS=CHECKPOINT", help="trained PPO checkpoints evaluated as extra methods under ALIAS (e.g. ppo_m1=final.pt)")
    ap.add_argument("--ppo-deterministic", action="store_true", help="PPO methods take the arg-max action (reference rows only)")
    ap.add_argument("--n-drones", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-per-source", type=int, default=config.EVAL_PRELIM_EPISODES_PER_SOURCE)
    ap.add_argument("--sources", type=int, nargs="*", default=list(config.ALL_SOURCES))
    ap.add_argument("--episodes", type=Path, default=None, help="existing episodes.csv (overrides --n-per-source / --sources)")
    ap.add_argument("--base-seed", type=int, default=config.EVAL_BASE_SEED)
    ap.add_argument("--mode", choices=list(config.ENV_MODES), default="F")
    ap.add_argument("--scale-fixed", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="episode horizon (default 300 in Mode F, config.T2_MAX_STEPS = 150 in Mode T2)")
    ap.add_argument("--processes", type=int, default=config.EVAL_PROCESSES)
    ap.add_argument("--log-steps", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None, help="truncate the list (smoke tests)")
    ap.add_argument("--tag", default="prelim")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tie-tol", type=float, default=None, help="GMM-Infotaxis tie width [nats] (default config.INFOTAXIS_TIE_TOL_NATS)")
    ap.add_argument("--summarize-only", action="store_true", help="rebuild summary.json from the records already in the tag directory")
    ap.add_argument("--merge-from", nargs="*", default=[], metavar="TAG:method[:newname]",
                    help="with --summarize-only: import records of other tags first (method renamed to newname)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")            # Windows cp949 console: never crash on a non-ASCII table
    out_dir = args.out_dir if args.out_dir is not None else config.CACHE_DIR / "eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        for spec in args.merge_from:
            print(f"[eval] imported {import_records(out_dir, spec)} records from {spec}")
        recs = load_records(out_dir)
        methods = []
        for r in recs:
            if r["method"] not in methods:
                methods.append(r["method"])
        return build_summary(out_dir, recs, args.tag, args.mode, 0.0, 0, methods, args.max_steps)
    if args.episodes is not None:
        specs = load_episode_list(args.episodes)
    else:
        specs = make_episode_list(args.sources, args.n_per_source, args.base_seed, args.mode, scale_fixed=args.scale_fixed)
    if args.max_episodes is not None:
        specs = specs[:args.max_episodes]
    save_episode_list(specs, out_dir / "episodes.csv")
    policy_kw = {"gmm_infotaxis": {"tie_tol": args.tie_tol}} if args.tie_tol is not None else {}
    for spec_s in args.ppo:
        alias, _, ckpt = spec_s.partition("=")
        if not alias or not ckpt or alias in POLICIES or alias in VERIFICATION_POLICIES:
            raise SystemExit(f"--ppo expects ALIAS=CHECKPOINT with a new alias, got {spec_s}")
        policy_kw[alias] = {"checkpoint": ckpt, "deterministic": args.ppo_deterministic}
        args.methods = list(args.methods) + [alias]
    tasks = [{"method": m, "n_drones": n, "mode": args.mode, "max_steps": args.max_steps, "spec": spec.__dict__, "log_steps": args.log_steps,
              "step_dir": str(out_dir / "steps"), "policy_kw": policy_kw}
             for m in args.methods for n in args.n_drones for spec in specs]
    print(f"[eval] {len(specs)} episodes x {len(args.methods)} methods x {args.n_drones} drones = {len(tasks)} runs on "
          f"{args.processes} processes -> {out_dir}" + (f"; Infotaxis tie_tol {args.tie_tol}" if args.tie_tol is not None else ""), flush=True)
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
    return build_summary(out_dir, recs, args.tag, args.mode, time.perf_counter() - t0, args.processes, args.methods, args.max_steps)


if __name__ == "__main__":
    main()
