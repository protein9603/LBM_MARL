"""Evaluation metrics and Table 2 aggregation (plan 5장 지표; D9-2).

Per episode record (eval/run_eval.py): success (primary criterion: GMM top sigma < ENV_SUCCESS_SIGMA_M and MAP error <
ENV_SUCCESS_ERROR_M at ANY step; episodes run all MAX_EPISODE_STEPS), steps (first success step, else the episode length),
success_strict / steps_strict (SUCCESS_SIGMA_M / SUCCESS_ERROR_M = 15 m / 20 m), min_error_m, start_type (plume / random), final MAP error [m], first detection step (Currie threshold), path length [m], masked actions,
declared-success step (first step with top sigma < ENV_SUCCESS_SIGMA_M) and the MAP error at that step (deployment view:
a stopping rule that only uses the belief), wall time.
Aggregates per (method, n_drones, group; groups = GROUPS: mutually exclusive train_open / train_other / holdout, summaries train / all_observable, and the unobservable source 110 apart): success rate with Wilson 95 % CI, success-step median (successes only) with
bootstrap CI, censored median (failures counted as max_steps), final error median / p90, first-detection median,
declared-success rate = fraction of episodes whose declared stop is within ENV_SUCCESS_ERROR_M.
Paired differences on the common episode list: per episode success difference and success-step difference.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from srcloc_env import config

_TRAIN_OPEN = tuple(x for x in config.T1_3_OPEN_SOURCES if x in config.TRAIN_SOURCES)          # (101, 108, 111, 113): 109 is a holdout source
GROUPS: dict[str, tuple[int, ...]] = {
    # mutually exclusive partition of the 12 observable sources (Table 2 rows; docs/training_evaluation_spec.md section 8):
    "train_open": _TRAIN_OPEN,
    "train_other": tuple(x for x in config.TRAIN_SOURCES if x not in _TRAIN_OPEN),                # (102, 104, 106, 107, 112): trapped / regression 102, 104, 106 + 107, 112
    "holdout": tuple(config.HOLDOUT_SOURCES),                                                      # (103, 105, 109)
    # summary rows
    "train": tuple(config.TRAIN_SOURCES),
    "all_observable": tuple(x for x in config.ALL_SOURCES if x not in config.EXCLUDED_SOURCES),
    # reported separately, never part of a success-rate aggregate of the rows above
    "unobservable": tuple(config.EXCLUDED_SOURCES),                                                # (110): not observable at the fixed 15 m altitude (D7-4)
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


def aggregate(records: Iterable[dict], max_steps: int = config.MAX_EPISODE_STEPS, start_type: str | None = None) -> dict:
    """Aggregate a list of episode records of ONE (method, n_drones) configuration into group statistics
    (``start_type`` = 'plume' / 'random' restricts to one start stratum)."""
    recs = [r for r in records if start_type is None or r.get("start_type") == start_type]
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
        decl_ok = sum(1 for r in declared if r["declared_error_m"] < config.ENV_SUCCESS_ERROR_M)
        ks = sum(int(bool(r.get("success_strict", False))) for r in rs)
        srate, slo, shi = wilson_ci(ks, n)
        out[gname] = {"n": n, "n_success": k, "success_rate": rate, "success_ci": [lo, hi],
                      "n_success_strict": ks, "success_strict_rate": srate, "success_strict_ci": [slo, shi],
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


def table2_markdown(agg_by_config: dict[str, dict],
                    groups: Sequence[str] = ("train_open", "train_other", "holdout", "train", "all_observable", "unobservable")) -> str:
    """agg_by_config[label] = aggregate(...) -> markdown rows label x group."""
    lines = ["| 방법 (드론 수) | 그룹 | n | 성공률 [95% CI] | 엄격 성공률 | 성공 스텝 중앙값 [CI] | 검열 중앙값 | 최종 오차 중앙값 / p90 [m] | 선언 성공률 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, agg in agg_by_config.items():
        for g in groups:
            if g not in agg:
                continue
            a = agg[g]
            ss = (f"{a['success_step_median']:.0f} [{a['success_step_ci'][0]:.0f}, {a['success_step_ci'][1]:.0f}]"
                  if np.isfinite(a["success_step_median"]) else "-")
            decl = f"{a['declared_success_rate']:.0%}" if a["declared_success_rate"] is not None else "-"
            lines.append(f"| {label} | {g} | {a['n']} | {a['success_rate']:.0%} [{a['success_ci'][0]:.0%}, {a['success_ci'][1]:.0%}] | {a['success_strict_rate']:.0%} | {ss} | "
                         f"{a['censored_step_median']:.0f} | {a['final_error_median_m']:.0f} / {a['final_error_p90_m']:.0f} | {decl} |")
    return "\n".join(lines)
