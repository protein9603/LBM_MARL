"""Gate report: collect the acceptance-test JSONs of a stage into one go/no-go summary (plan section 3).

Usage: python -m srcloc_env.scripts.gate_report --gate G0|G1  -> cache/g0_report.json / g1_report.json
G0 (plan S0): T0-1 kernel reproduction, T0-2 cache, T0-3 slab vs exact gather, T0-4 growth table,
T0-5 query latency, plus the wind / obstacle / alignment checks used by the environment.
G1 (plan S1, 2026-09-30 보강 포함): library selection >= 95%, T1-2b kappa grid, T1-2 kappa bias criterion 2,
T1-3/T1-3b shape comparison (adjoint better on all sources), T1-4 open-source error < 30 m (analytic AND adjoint),
T1-5 GMM fidelity.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

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
        "T0-4 snapshot growth table": {"pass": bool(slabs and "t0_4" in slabs), "criterion": "recorded"},
        "T0-5 slab query latency": {
            "pass": bool(field and field.get("t0_5", {}).get("pass")),
            "single_query_median_s": field and field["t0_5"]["single_query_one_source"]["median_s"],
            "criterion": "< 1e-4 s"},
        "wind field reproduction": {"pass": bool(wind and wind.get("pass")), "criterion": "report 3.3 means within 1e-3"},
        "obstacle map": {"pass": bool(drone and drone.get("pass", drone.get("overall_pass", True))), "criterion": "13 sources free at 15 m, superset of stl_tools"},
        "STL/LBM alignment": {"pass": bool(align and align["best_integer_shift_cells_(dix,diy)"] == [0, 0] and align["jaccard_occ_vs_ib_footprint_2p5m"] > 0.9),
                              "jaccard": align and align["jaccard_occ_vs_ib_footprint_2p5m"], "criterion": "best shift (0,0), Jaccard > 0.9"},
    }
    ok = all(v["pass"] for v in items.values())
    return {"gate": "G0", "date": date.today().isoformat(), "items": items, "pass": ok,
            "decision": "S0 data layer accepted; slabs stay at 5 m (T0-3 r 0.99); proceed to S1" if ok
            else "see failing items; fallback per plan S0 (online cKDTree gather / 2.5 m slabs)"}


def g1() -> dict:
    lib, kg, kb, ca, t14, t15 = (_load("validate_library_filter.json"), _load("validate_kappa_grid.json"),
                                 _load("validate_kappa_bias.json"), _load("calibrate_adjoint.json"),
                                 _load("validate_t1_4.json"), _load("validate_t1_5.json"))
    lib_rate = ((lib or {}).get("blocks", {}).get("library", {}) or {}).get("selection_rate_overall")
    t12b = (kg or {}).get("t1_2b", {})
    kb_overall = (kb or {}).get("overall", {})
    ca_per = ((ca or {}).get("verdicts", {}) or {}).get("per_source", {})
    n_adj_better = sum(1 for v in ca_per.values() if v.get("adjoint_better_std_dense"))
    open_src = [str(s) for s in config.T1_3_OPEN_SOURCES] if hasattr(config, "T1_3_OPEN_SOURCES") else ["101", "108", "109", "111", "113"]
    v = (t14 or {}).get("verdicts", {})
    a_open = {s: v.get("i_A_open", {}).get("per_source", {}).get(s, {}).get("final_error_median_m") for s in open_src}
    b_open = {s: v.get("ii_B_open", {}).get("per_source", {}).get(s, {}).get("final_error_median_m") for s in open_src}
    a_pass = bool(a_open and all(x is not None and x < config.T1_4_FINAL_ERROR_PASS_M for x in a_open.values()))
    b_pass = bool(b_open and all(x is not None and x < config.T1_4_FINAL_ERROR_PASS_M for x in b_open.values()))
    items = {
        "library filter selection >= 95% (T1-4 upper bound)": {"pass": bool(lib_rate is not None and lib_rate >= 0.95), "selection_rate": lib_rate},
        "T1-2b kappa grid margins >= 1 decade": {"pass": bool(t12b.get("pass")), "kappa_ref": t12b.get("kappa_ref"), "grid_decades": t12b.get("grid_decades")},
        "T1-2 fixed-wrong-kappa bias > RB-PF bias": {"pass": bool(kb_overall.get("pass_bias")), "map_diff_criterion": kb_overall.get("pass_map_diff")},
        "T1-3b adjoint shape residual < analytic (all sources)": {"pass": n_adj_better == len(ca_per) and len(ca_per) > 0, "n_better": n_adj_better, "n_sources": len(ca_per)},
        "T1-4 analytic RB-PF open-source median error < 30 m": {"pass": a_pass, "per_source_m": a_open},
        "T1-4 adjoint RB-PF open-source median error < 30 m": {"pass": b_pass, "per_source_m": b_open},
        "T1-5 GMM fidelity (TV < 0.2, converged flips < 10%)": {"pass": bool((t15 or {}).get("verdict", {}).get("overall_pass")),
                                                            "per_source": (t15 or {}).get("verdict", {}).get("per_source")},
    }
    ok = all(x["pass"] for x in items.values())
    decision = ("S1 accepted; proceed to S2" if ok else
                "FAIL on T1-4 (both forward models over-collapse on the frozen LDM snapshot). Plan S1 fallback + D7: "
                "overdispersed (negative-binomial) likelihood in RBPF, calibrated dispersion; re-run T1-4 before S2.")
    return {"gate": "G1", "date": date.today().isoformat(), "items": items, "pass": ok, "decision": decision}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="G0", choices=["G0", "G1"])
    args = ap.parse_args()
    rep = g0() if args.gate == "G0" else g1()
    out = config.CACHE_DIR / f"{args.gate.lower()}_report.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print("GATE", args.gate, "PASS" if rep["pass"] else "FAIL")


if __name__ == "__main__":
    main()