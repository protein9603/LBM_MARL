"""Sensitivity of GMM-Infotaxis to the tie width (D9-4): compare tie_tol in {0, 0.005, 0.05} (and greedy-MAP) on the SAME episodes.

Usage: python -m srcloc_env.scripts.analyze_ifx_sensitivity [--tags ifx_tol0 ifx_tol0.005 ifx_tol0.05] [--n-drones 1] [--out docs/ifx_tie_sensitivity.md]
Reads cache/eval/<tag>/records_gmm_infotaxis_<n>drones.csv (+ greedy_map from t2_4_v2 as the tol -> infinity reference).
Per tie width: primary / strict success rate (Wilson 95 % CI) over the 12 observable sources and per start type, final MAP error
(median, p90), closest approach to the source (from the step logs), path length, the share of decisions that fell in the tie band
(tie_frac) and of decisions in which EVERY allowed action was tied (all_tied_frac = pure greedy-MAP behaviour); paired
comparison against the 0.05 run: number of episodes whose primary success differs and the median difference of the closest approach.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from srcloc_env import config
from srcloc_env.eval.metrics import wilson_ci

OBS = set(config.ALL_SOURCES) - set(config.EXCLUDED_SOURCES)


def _load(tag: str, method: str, n: int) -> list[dict]:
    f = config.CACHE_DIR / "eval" / tag / f"records_{method}_{n}drones.csv"
    with f.open("r", encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if int(r["source"]) in OBS]


def _closest(r: dict) -> float:
    z = np.load(r["step_log"], allow_pickle=False)
    return float(np.hypot(*(z["drone_xy"] - z["truth_xy"]).transpose(2, 0, 1)).min())


def summarize(rows: list[dict]) -> dict:
    ok = np.array([r["success"] == "True" for r in rows]); okS = np.array([r["success_strict"] == "True" for r in rows])
    err = np.array([float(r["final_error_m"]) for r in rows])
    out = {"n": len(rows), "success": float(ok.mean()), "success_ci": wilson_ci(int(ok.sum()), len(rows))[1:], "n_success": int(ok.sum()),
           "strict": float(okS.mean()), "n_strict": int(okS.sum()), "err_median": float(np.median(err)), "err_p90": float(np.percentile(err, 90)),
           "path_median": float(np.median([float(r["path_length_m"]) for r in rows]))}
    for st in ("plume", "random"):
        sel = [r for r in rows if r["start_type"] == st]
        k = sum(r["success"] == "True" for r in sel)
        out[f"success_{st}"] = (k, len(sel))
    tf = [float(r["tie_frac"]) for r in rows if r.get("tie_frac") not in (None, "", "nan")]
    at = [float(r["all_tied_frac"]) for r in rows if r.get("all_tied_frac") not in (None, "", "nan")]
    out["tie_frac"] = float(np.nanmean(tf)) if tf else float("nan")
    out["all_tied_frac"] = float(np.nanmean(at)) if at else float("nan")
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", default=["ifx_tol0", "ifx_tol0.005", "ifx_tol0.05"])
    ap.add_argument("--n-drones", type=int, default=1)
    ap.add_argument("--reference-tag", default="t2_4_v2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    n = args.n_drones
    runs = {tag: _load(tag, "gmm_infotaxis", n) for tag in args.tags}
    runs["greedy_map (reference)"] = _load(args.reference_tag, "greedy_map", n)
    base_tag = [t for t in args.tags if t.endswith("0.05")][0]
    base = {int(r["episode_id"]): r for r in runs[base_tag]}
    res, rows_md = {}, []
    for tag, rows in runs.items():
        s = summarize(rows)
        cl = {int(r["episode_id"]): _closest(r) for r in rows}
        s["closest_median"] = float(np.median(list(cl.values())))
        s["within_100m"] = float(np.mean(np.array(list(cl.values())) < 100.0))
        common = sorted(set(cl) & set(base))
        rows_by = {int(r["episode_id"]): r for r in rows}
        s["paired_vs_0.05"] = {"n_success_differs": int(sum(rows_by[i]["success"] != base[i]["success"] for i in common)),
                                "n_identical_final_error": int(sum(abs(float(rows_by[i]["final_error_m"]) - float(base[i]["final_error_m"])) < 1e-6 for i in common)),
                                "closest_diff_median_m": float(np.median([cl[i] - _closest(base[i]) for i in common]))}
        res[tag] = s
        sp, sr = s["success_plume"], s["success_random"]
        rows_md.append(f"| {tag} | {s['n']} | {100 * s['success']:.1f} % [{100 * s['success_ci'][0]:.0f}, {100 * s['success_ci'][1]:.0f}] ({s['n_success']}) | "
                       f"{100 * s['strict']:.1f} % | {sp[0]}/{sp[1]} · {sr[0]}/{sr[1]} | {s['err_median']:.0f} / {s['err_p90']:.0f} | {s['closest_median']:.0f} ({100 * s['within_100m']:.0f} % < 100 m) | "
                       f"{s['path_median']:.0f} | {s['tie_frac']:.2f} | {s['all_tied_frac']:.2f} | {s['paired_vs_0.05']['n_success_differs']} / {s['paired_vs_0.05']['n_identical_final_error']} |")
    md = ("| 방법 / 동률 폭 | n | 주 성공률 [95 % CI] (성공 수) | 엄격 성공률 | plume 시작 · random 시작 성공 | 최종 오차 중앙값 / p90 [m] | 소스 최근접 거리 중앙값 [m] | 경로 중앙값 [m] | 동률 해소 결정 비율 | 전 행동 동률 비율 | 0.05 대비 성공 달라진 에피소드 / 최종오차 동일 에피소드 |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows_md))
    print(md)
    out = {"n_drones": n, "runs": res, "table_markdown": md}
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    (config.CACHE_DIR / "eval" / f"ifx_sensitivity_{n}drone.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
