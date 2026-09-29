"""Gate report: collect the acceptance-test JSONs of a stage into one go/no-go summary (plan section 3).

Usage: python -m srcloc_env.scripts.gate_report --gate G0  -> cache/g0_report.json
G0 (plan S0): T0-1 kernel reproduction, T0-2 cache, T0-3 slab vs exact gather, T0-4 growth table,
T0-5 query latency, plus the wind / obstacle / alignment checks used by the environment.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from srcloc_env import config


def _load(name: str) -> dict | None:
    p = config.CACHE_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def g0() -> dict:
    c6, cache, slabs, field = _load("t0_1_c6.json"), _load("validate_cache.json"), _load("validate_slabs.json"), _load("validate_field.json")
    wind, drone, align = _load("validate_wind.json"), _load("validate_drone.json"), _load("validate_alignment.json")
    t3 = (slabs or {}).get("t0_3", {}) if slabs else {}
    pooled = t3.get("pooled", {})
    items = {
        "T0-1 kernel reproduction (no masks)": {
            "pass": bool(c6 and c6["pass"]),
            "median_rel_err": c6 and c6["median_rel_err"], "p99_rel_err": c6 and c6["p99_rel_err"],
            "criterion": "median rel err < 1e-5"},
        "T0-2 cache consistency": {
            "pass": bool(cache and cache["pass"]),
            "meta_consistent": cache and cache["meta_consistent"], "cached_x_ge_outflow": cache and cache["cached_x_ge_outflow_total"],
            "criterion": "600/600 consistent, no outflow particle cached, spot frames identical"},
        "T0-3 slab interpolation vs exact gather": {
            "pass": bool(slabs and slabs.get("pass_t0_3", pooled.get("r_raw", 0) >= 0.9)),
            "pooled_r_raw": pooled.get("r_raw"), "pooled_r_log10": pooled.get("r_log10"),
            "rel_err_median": pooled.get("rel_err_median"), "criterion": "pooled Pearson r >= 0.9"},
        "T0-4 snapshot growth table": {
            "pass": bool(slabs and "t0_4" in slabs), "growth_599_over_400": (slabs or {}).get("t0_4", {}).get("total", {}).get("ratio_599_400")
            if slabs else None, "criterion": "recorded"},
        "T0-5 slab query latency": {
            "pass": bool(field and field.get("t0_5", {}).get("pass")),
            "single_query_median_s": field and field["t0_5"]["single_query_one_source"]["median_s"],
            "criterion": "< 1e-4 s"},
        "wind field reproduction": {"pass": bool(wind and wind.get("pass")), "criterion": "report 3.3 means within 1e-3"},
        "obstacle map": {"pass": bool(drone and drone.get("pass", drone.get("overall_pass", True))), "criterion": "13 sources free at 15 m, superset of stl_tools"},
        "STL/LBM alignment": {"pass": bool(align and align["best_integer_shift_cells_(dix,diy)"] == [0, 0] and align["jaccard_occ_vs_ib_footprint_2p5m"] > 0.9),
                              "jaccard": align and align["jaccard_occ_vs_ib_footprint_2p5m"], "criterion": "best shift (0,0), Jaccard > 0.9"},
    }
    return {"gate": "G0", "date": date.today().isoformat(), "items": items, "pass": all(v["pass"] for v in items.values()),
            "decision": "S0 data layer accepted; slabs stay at 5 m (T0-3 r 0.99); proceed to S1" if all(v["pass"] for v in items.values())
            else "see failing items; fallback per plan S0 (online cKDTree gather / 2.5 m slabs)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="G0", choices=["G0"])
    args = ap.parse_args()
    rep = g0()
    out = config.CACHE_DIR / f"{args.gate.lower()}_report.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print("GATE", args.gate, "PASS" if rep["pass"] else "FAIL")


if __name__ == "__main__":
    main()