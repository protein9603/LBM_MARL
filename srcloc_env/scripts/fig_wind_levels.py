"""Figure: success rate of every method under the three wind-knowledge levels of the estimator (D13, plan D13).

Grouped bars (x = method, colour = wind level W0 / W1 / W2 in fixed order) with Wilson 95 % intervals, one panel per
episode group (default all_observable and holdout), values written on the bars; the numbers are also saved as CSV.

Usage (tags hold the records of ONE level each; the PPO alias names the method inside its tag):
  python -m srcloc_env.scripts.fig_wind_levels --n-drones 2 \
      --level W0=base_W0 --level W1=base_W1 --level W2=t2_compare5 \
      --ppo W0=ppo_W0:ppo_b2_w0 --ppo W1=ppo_W1:ppo_b2_w1 --ppo W2=t2_compare5:ppo_b2 [--groups all_observable holdout] [--dpi 150]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from srcloc_env import config
from srcloc_env.eval.metrics import GROUPS, wilson_ci

LEVELS = ("W0", "W1", "W2")
LEVEL_LABEL = {"W0": "W0: mean wind + Gaussian plume", "W1": "W1: mean wind + building map", "W2": "W2: CFD wind field"}
LEVEL_COLOR = {"W0": "#2a78d6", "W1": "#eb6834", "W2": "#1baf7a"}        # categorical slots 1-3 of the reference palette (validated)
METHODS = (("random", "random"), ("lawnmower", "lawnmower"), ("greedy_map", "greedy-MAP"), ("gmm_infotaxis", "Infotaxis"),
           ("ppo", "PPO (ours," + chr(10) + "3 seeds pooled)"), ("oracle_loiter", "oracle" + chr(10) + "(knows source)"))
GROUP_LABEL = {"all_observable": "12 observable sources (n = 120; PPO 3 seeds, 360)", "holdout": "3 held-out sources (n = 30; PPO 90)", "train": "9 training sources (n = 90)",
               "train_open": "4 open training sources (n = 40)"}


def _records(tag: str, method: str, n_drones: int) -> list[dict]:
    f = config.CACHE_DIR / "eval" / tag / f"records_{method}_{n_drones}drones.csv"
    if not f.exists():
        return []
    with f.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def collect(levels: dict[str, str], ppo: dict[str, list[tuple[str, str]]], n_drones: int, groups: list[str]) -> list[dict]:
    rows = []
    for lv in LEVELS:
        if lv not in levels:
            continue
        for key, label in METHODS:
            if key == "ppo":
                if lv not in ppo:
                    continue
                recs = [r for tag, method in ppo[lv] for r in _records(tag, method, n_drones)]     # several seeds pool
            else:
                recs = _records(levels[lv], key, n_drones)
            for g in groups:
                sel = [r for r in recs if int(r["source"]) in GROUPS[g]]
                k = sum(r["success"] == "True" for r in sel)
                if sel:
                    p, lo, hi = wilson_ci(k, len(sel))
                    rows.append({"level": lv, "method": key, "method_label": label, "group": g, "n": len(sel), "k": k, "success": p, "ci_lo": lo, "ci_hi": hi})
    return rows


def draw(rows: list[dict], groups: list[str], n_drones: int, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, len(groups), figsize=(6.0 * len(groups) + 0.8, 4.6), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    levels = [lv for lv in LEVELS if any(r["level"] == lv for r in rows)]
    width = 0.8 / max(len(levels), 1)
    for ax, g in zip(axes, groups):
        ax.set_facecolor("#fcfcfb")
        for j, (key, label) in enumerate(METHODS):
            for i, lv in enumerate(levels):
                r = next((r for r in rows if r["level"] == lv and r["method"] == key and r["group"] == g), None)
                if r is None:
                    continue
                x = j + (i - (len(levels) - 1) / 2) * width
                ax.bar(x, 100 * r["success"], width * 0.92, color=LEVEL_COLOR[lv], edgecolor="none", label=LEVEL_LABEL[lv] if j == 0 else None, zorder=3)
                ax.errorbar(x, 100 * r["success"], yerr=[[100 * (r["success"] - r["ci_lo"])], [100 * (r["ci_hi"] - r["success"])]],
                            fmt="none", ecolor="#52514e", elinewidth=1.0, capsize=2, zorder=4)
                ax.text(x, 100 * r["ci_hi"] + 1.5, f"{100 * r['success']:.0f}", ha="center", va="bottom", fontsize=7, color="#0b0b0b", zorder=5)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([label for _, label in METHODS], fontsize=8.5)
        ax.set_title(GROUP_LABEL.get(g, g), fontsize=10)
        ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("success rate [%] (Wilson 95 % interval)", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle(f"Source localisation success by the estimator's wind knowledge ({n_drones} drones, time-varying truth, identical episodes per panel)", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", action="append", default=[], metavar="LEVEL=TAG", help="records tag of the baselines of one wind level")
    ap.add_argument("--ppo", action="append", default=[], metavar="LEVEL=TAG:METHOD", help="records tag and method name of the PPO policy of one level")
    ap.add_argument("--n-drones", type=int, default=2)
    ap.add_argument("--groups", nargs="+", default=["all_observable", "holdout"])
    ap.add_argument("--out", type=Path, default=config.FIG_DIR / "fig_wind_levels.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)
    levels = dict(s.split("=", 1) for s in args.level)
    ppo: dict[str, list[tuple[str, str]]] = {}
    for s in args.ppo:                                                   # repeat --ppo LEVEL=... to pool several seeds of one level
        lv, _, rest = s.partition("=")
        tag, _, method = rest.partition(":")
        ppo.setdefault(lv, []).append((tag, method))
    bad = [lv for lv in list(levels) + list(ppo) if lv not in LEVELS]
    if bad:
        raise SystemExit(f"unknown wind level(s) {bad}; use {LEVELS}")
    rows = collect(levels, ppo, args.n_drones, args.groups)
    if not rows:
        raise SystemExit("no records found for the given tags")
    draw(rows, args.groups, args.n_drones, args.out, args.dpi)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[fig_wind_levels] {len(rows)} rows -> {args.out} and {csv_path}")
    return {"rows": rows, "png": str(args.out), "csv": str(csv_path)}


if __name__ == "__main__":
    main()
