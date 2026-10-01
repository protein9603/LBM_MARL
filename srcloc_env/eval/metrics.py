"""Evaluation metrics and Table 2 aggregation (plan 5장 지표; D9-2).

Per episode record (eval/run_eval.py): success (true success: GMM top sigma < SUCCESS_SIGMA_M and MAP error <
SUCCESS_ERROR_M), steps, final MAP error [m], first detection step (Currie threshold), path length [m], masked actions,
declared-success step (first step with top sigma < SUCCESS_SIGMA_M) and the MAP error at that step (deployment view:
a stopping rule that only uses the belief), wall time.
Aggregates per (method, n_drones, group): success rate with Wilson 95 % CI, success-step median (successes only) with
bootstrap CI, censored median (failures counted as max_steps), final error median / p90, first-detection median,
declared-success rate = fraction of episodes whose declared stop is within SUCCESS_ERROR_M.
Paired differences on the common episode list: per episode success difference and success-step difference.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from srcloc_env import config

GROUPS: dict[str, tuple[int, ...]] = {
    "open": tuple(config.T1_3_OPEN_SOURCES),
    "trapped": tuple(config.T1_3_TRAPPED_SOURCES + (104, 106)),
    "holdout": tuple(config.HOLDOUT_SOURCES),
    "train": tuple(config.TRAIN_SOURCES),
    "all_observable": tuple(s for s in config.ALL_SOURCES if s not in config.EXCLUDED_SOURCES),
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(rate, lo, hi) Wilson score interval; (nan, nan, nan) for n = 0."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def bootstrap_median_ci(x: Sequence[float], n_boot: int = config.EVAL_N_BOOTSTRAP, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(list(x), dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(meds, 100 * alpha / 2)), float(np.percentile(meds, 100 * (1 - alpha / 2)))


def aggregate(records: Iterable[dict], max_steps: int = config.MAX_EPISODE_STEPS) -> dict:
    """Aggregate a list of episode records of ONE (method, n_drones) configuration into group statistics."""
    recs = list(records)
    out: dict[str, dict] = {}
    for gname, srcs in GROUPS.items():
        rs = [r for r in recs if int(r["source"]) in srcs]
        if not rs:
            continue
        n = len(rs)
        k = sum(int(bool(r["success"])) for r in rs)
        rate, lo, hi = wilson_ci(k, n)
        succ_steps = [r["steps"] for r in rs if r["success"]]
        med, mlo, mhi = bootstrap_median_ci(succ_steps)
        censored = [r["steps"] if r["success"] else max_steps for r in rs]
        errs = [r["final_error_m"] for r in rs]
        first = [r["first_detection_step"] for r in rs if r.get("first_detection_step") is not None]
        declared = [r for r in rs if r.get("declared_step") is not None]
        decl_ok = sum(1 for r in declared if r["declared_error_m"] < config.SUCCESS_ERROR_M)
        out[gname] = {"n": n, "n_success": k, "success_rate": rate, "success_ci": [lo, hi],
                      "success_step_median": med, "success_step_ci": [mlo, mhi],
                      "censored_step_median": float(np.median(censored)),
                      "final_error_median_m": float(np.median(errs)), "final_error_p90_m": float(np.percentile(errs, 90)),
                      "first_detection_median": (float(np.median(first)) if first else None),
                      "declared_n": len(declared), "declared_success_rate": (decl_ok / len(declared) if declared else None),
                      "path_length_median_m": float(np.median([r["path_length_m"] for r in rs])),
                      "masked_actions_median": float(np.median([r["n_masked"] for r in rs]))}
    return out


def paired_differences(rec_a: Iterable[dict], rec_b: Iterable[dict], max_steps: int = config.MAX_EPISODE_STEPS) -> dict:
    """A minus B on common episode_ids: success-rate difference, censored success-step difference (median + bootstrap CI)."""
    a = {int(r["episode_id"]): r for r in rec_a}
    b = {int(r["episode_id"]): r for r in rec_b}
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    ds = [int(bool(a[i]["success"])) - int(bool(b[i]["success"])) for i in ids]
    dsteps = [(a[i]["steps"] if a[i]["success"] else max_steps) - (b[i]["steps"] if b[i]["success"] else max_steps) for i in ids]
    med, lo, hi = bootstrap_median_ci(dsteps)
    return {"n": len(ids), "success_rate_diff": float(np.mean(ds)), "n_a_only": int(sum(d > 0 for d in ds)),
            "n_b_only": int(sum(d < 0 for d in ds)), "censored_step_diff_median": med, "censored_step_diff_ci": [lo, hi]}


def table2_markdown(agg_by_config: dict[str, dict], groups: Sequence[str] = ("open", "trapped", "holdout")) -> str:
    """agg_by_config[label] = aggregate(...) -> markdown rows label x group."""
    lines = ["| 방법 (드론 수) | 그룹 | n | 성공률 [95% CI] | 성공 스텝 중앙값 [CI] | 검열 중앙값 | 최종 오차 중앙값 / p90 [m] | 선언 성공률 |",
             "|---|---|---|---|---|---|---|---|"]
    for label, agg in agg_by_config.items():
        for g in groups:
            if g not in agg:
                continue
            a = agg[g]
            ss = (f"{a['success_step_median']:.0f} [{a['success_step_ci'][0]:.0f}, {a['success_step_ci'][1]:.0f}]"
                  if np.isfinite(a["success_step_median"]) else "-")
            decl = f"{a['declared_success_rate']:.0%}" if a["declared_success_rate"] is not None else "-"
            lines.append(f"| {label} | {g} | {a['n']} | {a['success_rate']:.0%} [{a['success_ci'][0]:.0%}, {a['success_ci'][1]:.0%}] | {ss} | "
                         f"{a['censored_step_median']:.0f} | {a['final_error_median_m']:.0f} / {a['final_error_p90_m']:.0f} | {decl} |")
    return "\n".join(lines)
