"""D5-4: library-filter upper bound (plan 4.2 "라이브러리 모델", S1 "라이브러리 필터 상한", T1-4 library upper bound).

Usage: python -m srcloc_env.scripts.validate_library_filter [--seed 0] [--n-seeds 5] [--n-steps 150]
Writes config.CACHE_DIR / validate_library_filter.json and config.FIG_DIR / fig_library_filter.png
(config.FIG_DPI_FINAL) + _preview.png (config.FIG_DPI_PREVIEW).

For each true source s of config.ALL_SOURCES (13): the truth is the cached LDM slab of s at frame
config.LIB_FRAME_INDEX (599 = step 30000), z = config.DRONE_Z, sensor scale config.LIB_SENSOR_SCALE; counts
y ~ Poisson((k0 scale n_s(p) + b) T) via Detector.measure (R3, plan 4.1).  Two drones fly the sawtooth
lawnmower of validate_pf_adjoint.two_drone_paths (config.LIB_N_STEPS = 150 steps, adjacent 100 m bands, start
config.PF_ADJ_START_DOWNWIND_M = 250 m downwind >= config.LIB_START_MIN_DISTANCE_M, drone-free cells only,
ObstacleMap 2 m margin).  The CandidateFilter (pf/library_filter.py) with the LibraryModel responses
(library_unit_response_fn: the same slabs as the truth -> no model mismatch, Poisson noise only) is updated
sequentially with both drones' counts (plan 4.3 fusion); config.LIB_N_SEEDS seeds per source.

Reported per source: correct-selection rate (MAP candidate after the last step == true source), the step at
which the posterior of the true candidate first exceeds config.LIB_POSTERIOR_PASS and the step from which it
stays above it, the posterior after config.LIB_REPORT_STEPS, the number of Currie detections, the maximum
expected count along the path; overall: selection rate, the 13 x 13 confusion matrix of the final MAP
candidate and the confusion pattern (which wrong candidate wins when the selection fails, with the runner-up
posterior when it succeeds).  If config.LIB_ADJOINT_FIELDS_NPZ exists the same runs are repeated with the
calibrated adjoint fields as the candidate responses (adjoint_unit_response_fn; real model mismatch, kappa
absorbs k scale q) for comparison ("adjoint" block; not the upper bound).

CAVEAT (plan 4.2): the library responses are densities for the ACTUAL release of the simulation, so kappa
absorbs only k0 x scale = 1000 here (recorded as kappa_true_library), 1.93 decades below config.KAPPA_REF
but inside the +-3 decade grid; the adjoint block's kappa is k0 x scale x q (unknown q, T1-3 implied ~17-270
particles/s).  Figure: posterior of the true candidate vs RL step per source (seeds thin, median bold) and the
confusion matrix.  References R3 (Poisson likelihood), R5 (kappa marginalisation).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.pf.library_filter import (CandidateFilter, UnitResponseFn, adjoint_unit_response_fn,
                                          library_unit_response_fn)
from srcloc_env.scripts.validate_pf_adjoint import two_drone_paths
from srcloc_env.sensor.detector import Detector

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

SERIES_COLOR = "#2a78d6"          # one series per panel (dataviz palette slot 1)
FAIL_COLOR = "#eb6834"            # seeds whose final MAP is wrong (slot 2)
INK_MUTED = "#8a8a8a"
SEQ_CMAP = "Blues"


# ---------------------------------------------------------------------------------------- one run
def run_filter(fn: UnitResponseFn, backend: LdmSlabBackend, true_source: int, paths: np.ndarray, det: Detector,
               seed: int, frame_index: int = config.LIB_FRAME_INDEX, z: float = config.DRONE_Z,
               scale: float = config.LIB_SENSOR_SCALE, eps_mix: float = config.PF_EPS_MIX) -> dict:
    """One episode: slab-truth counts along paths (n_steps, n_drones, 2), sequential CandidateFilter updates;
    returns the per-step posterior of the true candidate, the final posterior and selection statistics."""
    n_steps, n_drones = paths.shape[:2]
    flat = paths.reshape(-1, 2)
    dens = backend.density([true_source], flat, frame_index, z, 1.0)
    rng = np.random.default_rng([seed, true_source, 1])
    counts = det.measure(dens, scale, rng).reshape(n_steps, n_drones)
    expected = det.expected_counts(dens, scale).reshape(n_steps, n_drones)
    cf = CandidateFilter(fn, candidates=config.ALL_SOURCES, eps_mix=eps_mix)
    j_true = cf.index_of(true_source)
    p_true = np.empty(n_steps)
    ent = np.empty(n_steps)
    t_upd = []
    for k in range(n_steps):
        for d in range(n_drones):
            t0 = time.perf_counter()
            cf.update(int(counts[k, d]), np.array([paths[k, d, 0], paths[k, d, 1], z]))
            t_upd.append(time.perf_counter() - t0)
        p_true[k] = cf.posterior()[j_true]
        ent[k] = cf.entropy()
    post = cf.posterior()
    order = np.argsort(-post)
    above = p_true > config.LIB_POSTERIOR_PASS
    first = int(np.argmax(above)) + 1 if above.any() else None
    if above[-1]:
        k = n_steps - 1
        while k > 0 and above[k - 1]:
            k -= 1
        sustained = k + 1
    else:
        sustained = None
    return {
        "seed": int(seed), "true_source": int(true_source), "map_candidate": int(cf.map_candidate()),
        "correct": bool(cf.map_candidate() == true_source),
        "posterior_true_final": float(post[j_true]), "posterior_final": post.tolist(),
        "runner_up": int(config.ALL_SOURCES[order[1]]) if int(config.ALL_SOURCES[order[0]]) == true_source else int(config.ALL_SOURCES[order[0]]),
        "runner_up_posterior": float(post[order[1]] if int(config.ALL_SOURCES[order[0]]) == true_source else post[order[0]]),
        "posterior_true_at": {str(s): float(p_true[s - 1]) for s in config.LIB_REPORT_STEPS if s <= n_steps},
        "first_step_above": first, "sustained_step_above": sustained,
        "posterior_true_trajectory": p_true.tolist(), "entropy_trajectory_nats": ent.tolist(),
        "entropy_final_nats": float(ent[-1]),
        "kappa_q05_q50_q95_true": cf.posterior_kappa_quantiles((0.05, 0.5, 0.95), candidate=true_source).tolist(),
        "counts_max": int(counts.max()), "counts_mean": float(counts.mean()),
        "expected_counts_max": float(expected.max()),
        "n_detections": int(det.is_detection(counts).sum()),
        "update_time_median_s": float(np.median(t_upd)), "update_time_p99_s": float(np.percentile(t_upd, 99)),
    }


def summarise(runs: dict[int, list[dict]], label: str, n_steps: int) -> dict:
    """Per-source and overall statistics of a block of runs (one list of seeds per true source)."""
    srcs = list(runs)
    K = len(config.ALL_SOURCES)
    idx = {s: i for i, s in enumerate(config.ALL_SOURCES)}
    conf = np.zeros((K, K), dtype=int)
    per_source = {}
    failures = []
    for s in srcs:
        rr = runs[s]
        for r in rr:
            conf[idx[s], idx[r["map_candidate"]]] += 1
            if not r["correct"]:
                failures.append({"true": s, "seed": r["seed"], "winner": r["map_candidate"],
                                 "winner_posterior": float(max(r["posterior_final"])),
                                 "posterior_true_final": r["posterior_true_final"]})
        firsts = [r["first_step_above"] for r in rr]
        sust = [r["sustained_step_above"] for r in rr]
        per_source[str(s)] = {
            "kind": "trapped" if s in config.T1_3_TRAPPED_SOURCES else ("open" if s in config.T1_3_OPEN_SOURCES else "other"),
            "selection_rate": float(np.mean([r["correct"] for r in rr])),
            "n_correct": int(sum(r["correct"] for r in rr)), "n_seeds": len(rr),
            "posterior_true_final": [r["posterior_true_final"] for r in rr],
            "posterior_true_at": {k: [r["posterior_true_at"][k] for r in rr] for k in rr[0]["posterior_true_at"]},
            "first_step_above_0p9": firsts, "sustained_step_above_0p9": sust,
            "first_step_median": float(np.median([f for f in firsts if f is not None])) if any(f is not None for f in firsts) else None,
            "sustained_step_median": float(np.median([f for f in sust if f is not None])) if any(f is not None for f in sust) else None,
            "n_reached_0p9": int(sum(f is not None for f in firsts)),
            "runner_up": [r["runner_up"] for r in rr], "runner_up_posterior": [r["runner_up_posterior"] for r in rr],
            "entropy_final_nats": [r["entropy_final_nats"] for r in rr],
            "kappa_q50_true_candidate": [r["kappa_q05_q50_q95_true"][1] for r in rr],
            "counts_max": [r["counts_max"] for r in rr], "expected_counts_max": float(rr[0]["expected_counts_max"]),
            "n_detections": [r["n_detections"] for r in rr],
            "update_time_median_s": float(np.median([r["update_time_median_s"] for r in rr])),
        }
    n_runs = sum(len(v) for v in runs.values())
    # confusion pattern: for every true source, the most frequent wrong winner (or the most frequent runner-up)
    pattern = {}
    for s in srcs:
        rr = runs[s]
        wrong = [r["map_candidate"] for r in rr if not r["correct"]]
        ru = [r["runner_up"] for r in rr]
        pattern[str(s)] = {
            "wrong_winners": {str(w): int(wrong.count(w)) for w in sorted(set(wrong))},
            "runner_up_mode": int(max(set(ru), key=ru.count)),
            "runner_up_mode_count": int(max(ru.count(x) for x in set(ru))),
            "runner_up_posterior_max": float(max(r["runner_up_posterior"] for r in rr)),
        }
    return {
        "label": label, "n_steps": n_steps, "n_sources": len(srcs), "n_runs": n_runs,
        "selection_rate_overall": float(sum(r["correct"] for v in runs.values() for r in v) / n_runs),
        "n_sources_all_seeds_correct": int(sum(per_source[str(s)]["selection_rate"] == 1.0 for s in srcs)),
        "sources_with_failures": [s for s in srcs if per_source[str(s)]["selection_rate"] < 1.0],
        "first_step_median_over_sources": float(np.median([v["first_step_median"] for v in per_source.values() if v["first_step_median"] is not None])),
        "sustained_step_median_over_sources": float(np.median([v["sustained_step_median"] for v in per_source.values() if v["sustained_step_median"] is not None])),
        "n_runs_reached_0p9": int(sum(r["first_step_above"] is not None for v in runs.values() for r in v)),
        "confusion_matrix": {"rows_true_cols_map": [int(s) for s in config.ALL_SOURCES], "counts": conf.tolist()},
        "confusion_pattern": pattern, "failures": failures, "per_source": per_source,
        "update_time_median_s": float(np.median([v["update_time_median_s"] for v in per_source.values()])),
    }


# ---------------------------------------------------------------------------------------- figure
def make_figure(blocks: dict[str, dict], runs_by_block: dict[str, dict[int, list[dict]]], path: Path) -> None:
    """Left: 13 small multiples of the true-candidate posterior vs RL step (library block; seeds thin, median
    bold; wrong final MAP in orange); right: confusion matrix per block (rows true, columns MAP)."""
    lib = runs_by_block["library"]
    srcs = list(lib)
    n_blocks = len(blocks)
    fig = plt.figure(figsize=(17.0, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 4 + n_blocks, width_ratios=[1] * 4 + [1.6] * n_blocks)
    n_steps = blocks["library"]["n_steps"]
    for i, s in enumerate(srcs):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        P = np.array([r["posterior_true_trajectory"] for r in lib[s]])
        steps = np.arange(1, P.shape[1] + 1)
        for r, row in zip(lib[s], P):
            ax.plot(steps, row, color=SERIES_COLOR if r["correct"] else FAIL_COLOR, alpha=0.35, linewidth=0.9)
        ax.plot(steps, np.median(P, axis=0), color=SERIES_COLOR, linewidth=2.0)
        ax.axhline(config.LIB_POSTERIOR_PASS, color=INK_MUTED, linewidth=0.8, linestyle="--")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(0, n_steps)
        ps = blocks["library"]["per_source"][str(s)]
        med = ps["first_step_median"]
        ax.set_title(f"true {s} ({ps['kind']}): {ps['selection_rate']:.0%} correct\n"
                     f"P > {config.LIB_POSTERIOR_PASS} first at step {'n/a' if med is None else f'{med:.0f}'} (median)", fontsize=7.5)
        if i % 4 == 0:
            ax.set_ylabel("P(true candidate)", fontsize=8)
        if i // 4 == 3 or i + 4 >= len(srcs):
            ax.set_xlabel("RL step (2 drones)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, color="#e6e6e6", linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ids = [str(s) for s in config.ALL_SOURCES]
    for b, (name, blk) in enumerate(blocks.items()):
        ax = fig.add_subplot(gs[:, 4 + b])
        C = np.asarray(blk["confusion_matrix"]["counts"], dtype=float)
        im = ax.imshow(C, cmap=SEQ_CMAP, vmin=0.0, vmax=C.max())
        for (i, j), v in np.ndenumerate(C):
            if v > 0:
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 * C.max() else "#222222")
        ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=7)
        ax.set_yticks(range(len(ids)), ids, fontsize=7)
        ax.set_xlabel("MAP candidate after the last step")
        ax.set_ylabel("true source")
        ax.set_title(f"{name} responses: selection rate {blk['selection_rate_overall']:.0%} ({blk['n_runs']} runs)\n"
                     f"median first step with P > {config.LIB_POSTERIOR_PASS}: {blk['first_step_median_over_sources']:.0f}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.5, label="runs")
    fig.suptitle(f"Candidate filter over the 13 sources: LDM slab truth (frame {config.LIB_FRAME_INDEX}, z = "
                 f"{config.DRONE_Z:.0f} m, scale {config.LIB_SENSOR_SCALE}), 2-drone lawnmower {n_steps} steps, "
                 f"{config.LIB_N_SEEDS} seeds; blue = correct final MAP, orange = wrong", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIG_DPI_FINAL)
    fig.savefig(path.with_name(path.stem + "_preview" + path.suffix), dpi=config.FIG_DPI_PREVIEW)
    plt.close(fig)


# ---------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=config.LIB_N_SEEDS)
    ap.add_argument("--n-steps", type=int, default=config.LIB_N_STEPS)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_library_filter.json")
    ap.add_argument("--fig", type=Path, default=config.FIG_DIR / "fig_library_filter.png")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    om = ObstacleMap.load()
    backend = LdmSlabBackend()
    det = Detector()
    lib_fn, lib_model = library_unit_response_fn(backend, config.LIB_FRAME_INDEX, config.DRONE_Z, 1.0)
    adj = adjoint_unit_response_fn(config.LIB_ADJOINT_FIELDS_NPZ)
    fns: dict[str, UnitResponseFn] = {"library": lib_fn}
    if adj is not None:
        fns["adjoint"] = adj[0]

    paths_by_src, path_info, start_dist = {}, {}, {}
    for s in config.ALL_SOURCES:
        paths_by_src[s], path_info[s] = two_drone_paths(om, config.SOURCES_XY[s], args.n_steps)
        xy = np.asarray(config.SOURCES_XY[s])
        start_dist[s] = [float(np.hypot(*(paths_by_src[s][0, d] - xy))) for d in range(paths_by_src[s].shape[1])]
    start_ok = all(min(v) >= config.LIB_START_MIN_DISTANCE_M for v in start_dist.values())

    runs_by_block: dict[str, dict[int, list[dict]]] = {}
    blocks: dict[str, dict] = {}
    for name, fn in fns.items():
        runs: dict[int, list[dict]] = {}
        for s in config.ALL_SOURCES:
            runs[s] = [run_filter(fn, backend, s, paths_by_src[s], det, args.seed + i) for i in range(args.n_seeds)]
            ps = runs[s]
            print(f"[{name}] true {s}: correct {sum(r['correct'] for r in ps)}/{len(ps)}, "
                  f"first step > {config.LIB_POSTERIOR_PASS}: {[r['first_step_above'] for r in ps]}, "
                  f"P(true) final {np.median([r['posterior_true_final'] for r in ps]):.3f}, "
                  f"max expected counts {ps[0]['expected_counts_max']:.0f}, detections {[r['n_detections'] for r in ps]}",
                  flush=True)
        runs_by_block[name] = runs
        blocks[name] = summarise(runs, name, args.n_steps)
        print(f"[{name}] overall selection rate {blocks[name]['selection_rate_overall']:.3f}; "
              f"median first step > 0.9 over sources {blocks[name]['first_step_median_over_sources']:.0f}; "
              f"failures {[(f['true'], f['winner']) for f in blocks[name]['failures']]}", flush=True)

    make_figure(blocks, runs_by_block, args.fig)
    res = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "frame_index": config.LIB_FRAME_INDEX,
        "step": config.index_to_step(config.LIB_FRAME_INDEX), "z": config.DRONE_Z,
        "sensor": {"k0": det.k0, "scale": config.LIB_SENSOR_SCALE, "background_cps": det.background, "T": det.T,
                   "currie_threshold_cps": det.detection_threshold_cps()},
        "kappa_true_library": det.k0 * config.LIB_SENSOR_SCALE,
        "kappa_grid": {"kappa_ref": config.KAPPA_REF, "grid_decades": config.KAPPA_GRID_DECADES, "n_grid": config.KAPPA_G,
                       "log10_kappa_true_library_over_ref": float(np.log10(det.k0 * config.LIB_SENSOR_SCALE / config.KAPPA_REF))},
        "eps_mix": config.PF_EPS_MIX, "n_seeds": args.n_seeds, "n_steps": args.n_steps, "n_drones": config.PF_ADJ_N_DRONES,
        "n_measurements_per_run": int(args.n_steps * config.PF_ADJ_N_DRONES),
        "path": {"sweep_width_m": config.PF_ADJ_SWEEP_WIDTH_M, "start_downwind_m": config.PF_ADJ_START_DOWNWIND_M,
                 "step_m": config.DRONE_STEP_M, "start_distance_m": {str(s): v for s, v in start_dist.items()},
                 "min_start_distance_required_m": config.LIB_START_MIN_DISTANCE_M, "start_distance_ok": bool(start_ok),
                 "info": {str(s): v for s, v in path_info.items()}},
        "candidates": list(config.ALL_SOURCES), "posterior_pass": config.LIB_POSTERIOR_PASS,
        "caveat": ("plan 4.2: the library responses are the slab densities for the ACTUAL release of the simulation "
                   "(6.665 particles per index step per source), not per-unit-q responses; kappa therefore absorbs "
                   "only k0 x scale (= kappa_true_library) and the library filter is an upper bound (truth == model, "
                   "Poisson noise only), not a deployable forward model."),
        "adjoint_fields": None if adj is None else adj[1],
        "blocks": blocks, "figure": str(args.fig), "total_seconds": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"start distances >= {config.LIB_START_MIN_DISTANCE_M:.0f} m: {'OK' if start_ok else 'VIOLATED'}; "
          f"total {res['total_seconds']:.0f} s; JSON {args.out}; figure {args.fig}")
    return res


if __name__ == "__main__":
    main()