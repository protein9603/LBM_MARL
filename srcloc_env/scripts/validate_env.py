"""T2-1 / T2-3 on the real scene (D8-3, D8-4; plan S2): Gymnasium env_checker, a random policy for
config.ENV_CHECK_RANDOM_STEPS steps (no exception, no building / domain violation) and the per-step timing of the
environment (PF negative-binomial update + GMM summary + observation) -> steps per second per core.

Usage: python -m srcloc_env.scripts.validate_env [--steps 10000] [--seed 0] [--gmm-every 1] [--n-particles 2000]
                                                 [--mode F|T2] [--reflect] [--out <CACHE_DIR>/validate_env.json]
Writes config.CACHE_DIR / validate_env.json.

T2-1 pass: check_env passes; 0 exceptions; every visited position is free (ObstacleMap.is_free at DRONE_Z) and
inside the domain; masked actions leave the drone in place.
T2-3 record: median / p90 of total step time, PF update, GMM refresh and observation assembly; steps/s = 1 / median;
PASS iff median <= config.ENV_STEP_TIME_TARGET_S (20 ms, plan T2-3 [추정 목표]); the plan's fallback (1,000 particles,
EM 10 iterations) is measured with --n-particles 1000 --em-iters 10 when the target is missed; --warm-start measures
the D8-4 warm-started EM.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from srcloc_env import config
from srcloc_env.env.source_env import Scene, SourceLocEnv


def run_random_policy(env: SourceLocEnv, n_steps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    t_step, t_pf, t_gmm, t_obs, t_reset = [], [], [], [], []
    n_masked = n_violation = n_outside = n_success = n_trunc = n_exc = 0
    ep_len, lengths, successes, errors = 0, [], [], []
    sources, frames = [], []
    for i in range(n_steps):
        a = int(rng.integers(env.action_space.n))
        mask = env.action_mask()
        before = env.xy.copy()
        try:
            obs, r, term, trunc, info = env.step(a)
        except Exception as exc:                                        # noqa: BLE001 - T2-1 counts exceptions
            n_exc += 1
            print(f"[exception] step {i}: {exc!r}")
            obs, info = env.reset()
            ep_len = 0
            continue
        ep_len += 1
        tm = env.last_timing
        t_step.append(tm["total_s"]); t_pf.append(tm["pf_s"]); t_gmm.append(tm["gmm_s"]); t_obs.append(tm["obs_s"])
        if not mask[a]:
            n_masked += 1
            if not np.array_equal(env.xy, before):
                n_violation += 1
        om = env.scene.obstacles                                        # the CURRENT orientation (reflected episodes)
        if not om.is_free(env.xy, env.z):
            n_violation += 1
        if not om.in_domain(env.xy):
            n_outside += 1
        if not env.observation_space.contains(obs):
            n_violation += 1
        if term or trunc:
            lengths.append(ep_len); successes.append(bool(info["success"])); errors.append(float(info["map_error_m"]))
            sources.append(env.source); frames.append(env.frame0)
            n_success += int(info["success"]); n_trunc += int(trunc)
            t0 = time.perf_counter()
            obs, info = env.reset()
            t_reset.append(time.perf_counter() - t0)
            ep_len = 0
    q = lambda v, p: float(np.percentile(v, p)) if v else float("nan")   # noqa: E731
    med = q(t_step, 50)
    return {
        "n_steps": n_steps, "n_exceptions": n_exc, "n_masked_actions": n_masked, "n_violations": n_violation,
        "n_outside_domain": n_outside, "n_episodes": len(lengths), "n_success": n_success, "n_truncated": n_trunc,
        "episode_length_median": q(lengths, 50), "final_map_error_median_m": q(errors, 50),
        "sources_seen": sorted(set(sources)), "frames_seen_min_max": [min(frames), max(frames)] if frames else None,
        "timing": {"step_median_s": med, "step_p90_s": q(t_step, 90), "step_p99_s": q(t_step, 99),
                   "pf_median_s": q(t_pf, 50), "pf_p90_s": q(t_pf, 90), "gmm_median_s": q(t_gmm, 50), "gmm_p90_s": q(t_gmm, 90),
                   "obs_median_s": q(t_obs, 50), "reset_median_s": q(t_reset, 50), "reset_p90_s": q(t_reset, 90),
                   "steps_per_s_median": (1.0 / med) if med > 0 else float("nan"),
                   "target_s": config.ENV_STEP_TIME_TARGET_S, "pass": bool(med <= config.ENV_STEP_TIME_TARGET_S)},
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=config.ENV_CHECK_RANDOM_STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gmm-every", type=int, default=config.ENV_GMM_EVERY)
    ap.add_argument("--n-particles", type=int, default=config.PF_N_PARTICLES)
    ap.add_argument("--em-iters", type=int, default=config.ENV_GMM_EM_ITERS, help="GMM EM iterations per refresh (plan fallback 10)")
    ap.add_argument("--warm-start", action="store_true", help="EM warm start from the previous step's summary (D8-4)")
    ap.add_argument("--mode", choices=list(config.ENV_MODES), default=config.ENV_TRUTH_MODE_DEFAULT)
    ap.add_argument("--reflect", action="store_true", help="also build the reflected scene and reflect 50 %% of the episodes")
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_env.json")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    scene = Scene.load()
    scene_r = scene.reflected_scene() if args.reflect else None
    load_s = time.perf_counter() - t0
    env = SourceLocEnv(scene, truth_mode=args.mode, n_particles=args.n_particles, gmm_every=args.gmm_every,
                       gmm_iters=args.em_iters, gmm_warm_start=args.warm_start,
                       scene_reflected=scene_r, reflect_prob=config.ENV_REFLECT_PROB_TRAIN if args.reflect else 0.0)
    print(f"[setup] scene {load_s:.1f} s (adjoint build {scene.seconds_build:.2f} s, free cells {scene.model.op.n_free if hasattr(scene.model, 'op') else 'n/a'}); "
          f"mode {args.mode}, particles {args.n_particles}, gmm_every {args.gmm_every}, EM iters {args.em_iters}, warm start {args.warm_start}, reflect {args.reflect}", flush=True)
    t0 = time.perf_counter()
    check_env(env, skip_render_check=True)
    check_s = time.perf_counter() - t0
    print(f"[T2-1] gymnasium env_checker passed in {check_s:.1f} s", flush=True)
    t0 = time.perf_counter()
    rp = run_random_policy(env, args.steps, args.seed)
    rp_s = time.perf_counter() - t0
    tm = rp["timing"]
    t21 = rp["n_exceptions"] == 0 and rp["n_violations"] == 0 and rp["n_outside_domain"] == 0
    print(f"[T2-1] random policy {args.steps} steps in {rp_s:.0f} s: exceptions {rp['n_exceptions']}, violations "
          f"{rp['n_violations']}, outside {rp['n_outside_domain']}, masked {rp['n_masked_actions']}, episodes {rp['n_episodes']} "
          f"(success {rp['n_success']}, truncated {rp['n_truncated']}, median length {rp['episode_length_median']:.0f}) -> "
          f"{'PASS' if t21 else 'FAIL'}", flush=True)
    print(f"[T2-3] step median {tm['step_median_s'] * 1e3:.1f} ms (p90 {tm['step_p90_s'] * 1e3:.1f}, p99 {tm['step_p99_s'] * 1e3:.1f}); "
          f"PF {tm['pf_median_s'] * 1e3:.1f} ms, GMM {tm['gmm_median_s'] * 1e3:.1f} ms, obs {tm['obs_median_s'] * 1e3:.2f} ms; "
          f"reset {tm['reset_median_s'] * 1e3:.0f} ms; {tm['steps_per_s_median']:.0f} steps/s/core -> "
          f"{'PASS' if tm['pass'] else 'FAIL'} (target {config.ENV_STEP_TIME_TARGET_S * 1e3:.0f} ms)", flush=True)
    res = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": args.mode, "n_particles": args.n_particles,
           "gmm_every": args.gmm_every, "em_iters": args.em_iters, "warm_start": args.warm_start, "reflect": args.reflect, "seed": args.seed, "scene_load_s": load_s,
           "adjoint_build_s": scene.seconds_build, "env_checker": {"pass": True, "seconds": check_s},
           "random_policy": rp, "t2_1_pass": bool(t21), "t2_3_pass": bool(tm["pass"]), "random_policy_seconds": rp_s,
           "obs_dim": config.ENV_OBS_DIM, "n_actions": config.DRONE_N_ACTIONS, "adjoint": {"K": scene.params.K, "lam": scene.params.lam},
           "likelihood": config.PF_LIKELIHOOD, "nb_r": config.PF_NB_DISPERSION_R, "sources": list(env.sources)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(f"wrote {args.out}")
    return res


if __name__ == "__main__":
    main()
