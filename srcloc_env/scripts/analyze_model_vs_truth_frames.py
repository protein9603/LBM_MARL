"""Per-frame similarity of the three forward models (W0 plume, W1 potential-flow adjoint, W2 CFD adjoint) to the ACTUAL 15 m
concentration field of every observable source (D15 question: how close is each wind-knowledge level to reality, frame by frame).

The wind we hold is one steady snapshot, so the per-frame comparison is made on the quantity the PF uses: the steady model's
unit concentration response from the source over the slab cells versus the time-varying truth slab (frames 400..599).
Metrics on the truth's dense free cells (n >= T1_3_INFO_DENSITY_FRACTION x max): std of the log10 residual after a fitted scale
(the T1-3 shape statistic; 0 = perfect shape), Spearman rank correlation of the two fields, the distance between the mass
centroids, and the overlap (Jaccard) of the dense supports (model dense = top fraction of its own max).

Usage: python -m srcloc_env.scripts.analyze_model_vs_truth_frames [--frames 400 599 10] [--sources ...] [--out-dir 분석그림/icrs15]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from srcloc_env import config
from srcloc_env.env.source_env import Scene
from srcloc_env.eval.metrics import GROUPS
from srcloc_env.field.slab_stack import StackedSlabBackend
from srcloc_env.pf.sph_adjoint import AdvectionDiffusionOperator

LEVELS = ("W0", "W1", "W2")
COL = {"W0": "#2a78d6", "W1": "#eb6834", "W2": "#1baf7a"}


def model_field(scene: Scene, src: int, centres: np.ndarray) -> np.ndarray:
    xy = np.array(scene.sources_xy[src])
    if hasattr(scene.model, "operator"):
        return scene.model.operator.solve_forward(xy).ravel()
    return scene.model.unit_response(xy[None, :], centres)[0]


def metrics(truth: np.ndarray, g: np.ndarray, free: np.ndarray, xy: np.ndarray, frac: float) -> dict:
    dense = free & (truth >= frac * truth[free].max()) & (g > 0)
    if dense.sum() < 20:
        return {"n_dense": int(dense.sum())}
    lt, lg = np.log10(truth[dense]), np.log10(g[dense])
    res = lt - lg
    res -= np.median(res)                                                  # fitted scale (kappa) removed
    rho = spearmanr(lt, lg)[0]
    ct = (xy[dense] * truth[dense, None]).sum(0) / truth[dense].sum()
    gm = free & (g >= frac * g[free].max())
    cg = (xy[gm] * g[gm, None]).sum(0) / g[gm].sum()
    jac = (dense & gm).sum() / max((dense | gm).sum(), 1)
    return {"n_dense": int(dense.sum()), "std_log_res": float(res.std()), "spearman": float(rho),
            "centroid_offset_m": float(np.hypot(*(ct - cg))), "support_jaccard": float(jac)}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, nargs=3, default=[400, 599, 10], metavar=("FIRST", "LAST", "STEP"))
    ap.add_argument("--sources", type=int, nargs="*", default=list(GROUPS["all_observable"]))
    ap.add_argument("--out-dir", type=Path, default=config.FIG_DIR)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)
    backend = StackedSlabBackend()
    scenes = {lv: Scene.load(backend=backend, wind_level=lv) for lv in LEVELS}
    grid = scenes["W2"].grid
    centres = grid.cell_centres(config.DRONE_Z)
    free = ~AdvectionDiffusionOperator.blocked_at_cells(scenes["W2"].obstacles, grid, config.DRONE_Z).ravel()
    frac = config.T1_3_INFO_DENSITY_FRACTION
    frames = list(range(args.frames[0], args.frames[1] + 1, args.frames[2]))
    rows = []
    for src in args.sources:
        fields = {lv: model_field(scenes[lv], src, centres) for lv in LEVELS}
        for f in frames:
            truth = backend.density([src], centres[:, :2], f, config.DRONE_Z)[0] if backend.density([src], centres[:2, :2], f).ndim == 2 else backend.density([src], centres[:, :2], f)
            truth = np.asarray(truth, dtype=np.float64).reshape(-1)
            if truth[free].max() <= 0:
                continue
            for lv in LEVELS:
                m = metrics(truth, fields[lv], free, centres[:, :2], frac)
                rows.append({"source": src, "frame": f, "level": lv, **m})
        print(f"source {src}: " + " ".join(f"{lv} std {np.median([r['std_log_res'] for r in rows if r['source']==src and r['level']==lv and 'std_log_res' in r]):.2f}" for lv in LEVELS), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "model_vs_truth_frames.csv"
    keys = ["source", "frame", "level", "n_dense", "std_log_res", "spearman", "centroid_offset_m", "support_jaccard"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows([{k: r.get(k, "") for k in keys} for r in rows])
    summ = {}
    for lv in LEVELS:
        sel = [r for r in rows if r["level"] == lv and "std_log_res" in r]
        summ[lv] = {k: float(np.median([r[k] for r in sel])) for k in ("std_log_res", "spearman", "centroid_offset_m", "support_jaccard")}
        summ[lv]["n"] = len(sel)
    wins = {lv: 0 for lv in LEVELS}
    for src in args.sources:
        for f in frames:
            trio = {r["level"]: r for r in rows if r["source"] == src and r["frame"] == f and "std_log_res" in r}
            if len(trio) == 3:
                wins[min(trio, key=lambda lv: trio[lv]["std_log_res"])] += 1
    summ["best_shape_counts"] = wins
    (args.out_dir / "model_vs_truth_frames.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    # figure: shape residual by frame (median over sources) and per-source medians
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    for lv in LEVELS:
        med = [np.median([r["std_log_res"] for r in rows if r["level"] == lv and r["frame"] == f and "std_log_res" in r] or [np.nan]) for f in frames]
        axes[0].plot(frames, med, "-o", ms=3, lw=1.6, color=COL[lv], label=lv)
    axes[0].set_xlabel("truth frame (time-varying plume)"); axes[0].set_ylabel("std of log10 residual on dense cells (median over sources)"); axes[0].legend(frameon=False)
    axes[0].set_title("Shape mismatch of the steady model vs the truth, frame by frame", fontsize=10)
    srcs = args.sources; w = 0.27
    for i, lv in enumerate(LEVELS):
        vals = [np.median([r["std_log_res"] for r in rows if r["level"] == lv and r["source"] == s and "std_log_res" in r] or [np.nan]) for s in srcs]
        axes[1].bar(np.arange(len(srcs)) + (i - 1) * w, vals, w * 0.92, color=COL[lv], label=lv)
    axes[1].set_xticks(range(len(srcs))); axes[1].set_xticklabels([str(s) for s in srcs], fontsize=8); axes[1].set_xlabel("source"); axes[1].set_ylabel("median over frames")
    axes[1].set_title("Per source (lower = closer to reality)", fontsize=10); axes[1].legend(frameon=False)
    for ax in axes:
        ax.yaxis.grid(True, color="#e6e5e0"); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    fig.savefig(args.out_dir / "fig_model_vs_truth_frames.png", dpi=args.dpi, facecolor="white"); plt.close(fig)
    print(json.dumps(summ, indent=1))
    return summ


if __name__ == "__main__":
    main()
