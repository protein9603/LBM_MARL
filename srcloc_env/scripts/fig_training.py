"""Training-result figures from the run directories of rl/train.py (spec section 7, items 1, 2, 3 and 5; D10).

Usage: python -m srcloc_env.scripts.fig_training --runs m1_s1 m1_s2 m1_s3 --prefix fig6_m1 [--baseline-tag t2_4_v3 --n-drones 1]
Writes <FIG_DIR>/<prefix>_curves.png (success, strict success, return, length vs team steps; one thin line per seed, mean and
sd band over seeds; horizontal lines = baselines on the 9 training sources of the preliminary batch), <prefix>_diagnostics.png (PPO
losses, entropy, KL, clip fraction, explained variance, masked share), <prefix>_ckpt_eval.png (quick evaluation of the checkpoints:
training sources vs held-out sources) and the numbers behind them (<prefix>_curves.csv, <prefix>_ckpt_eval.csv).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402

from srcloc_env import config      # noqa: E402

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
BASE_COLORS = {"random": "0.55", "lawnmower": "#8c564b", "greedy_map": "#17becf", "gmm_infotaxis": "#e377c2"}


def read_log(run: Path) -> dict[str, np.ndarray]:
    with (run / "train_log.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]) for k in rows[0]}


def baseline_rates(tag: str, n_drones: int) -> dict[str, float]:
    out = {}
    d = config.CACHE_DIR / "eval" / tag
    for m in BASE_COLORS:
        f = d / f"records_{m}_{n_drones}drones.csv"
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            sel = [r for r in csv.DictReader(fh) if int(r["source"]) in config.TRAIN_SOURCES]
        if sel:
            out[m] = float(np.mean([r["success"] == "True" for r in sel]))
    return out


def _band(ax, logs: list[dict[str, np.ndarray]], key: str, label: str, grid: np.ndarray) -> np.ndarray:
    ys = []
    for i, lg in enumerate(logs):
        ax.plot(lg["env_steps"] / 1e6, lg[key], color=COLORS[i % 5], lw=0.8, alpha=0.55)
        ys.append(np.interp(grid, lg["env_steps"], lg[key]))
    ys = np.array(ys)
    if len(logs) > 1:
        m, s = np.nanmean(ys, 0), np.nanstd(ys, 0)
        ax.plot(grid / 1e6, m, color="k", lw=1.8, label="mean of seeds")
        ax.fill_between(grid / 1e6, m - s, m + s, color="k", alpha=0.12)
    ax.set_xlabel("team steps [M]")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    return ys


def fig_curves(runs: list[Path], prefix: str, out_dir: Path, base: dict[str, float]) -> dict:
    logs = [read_log(r) for r in runs]
    top = min(lg["env_steps"][-1] for lg in logs)
    grid = np.linspace(0.0, top, 200)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    table = {"env_steps": grid}
    for ax, key, lab in zip(axes.flat, ("success_ma", "strict_ma", "return_ma", "length_ma"),
                            ("success rate (primary, 100-episode mean)", "strict success rate (sigma<15 m, error<20 m)",
                             "team return per episode", "episode length [steps]")):
        ys = _band(ax, logs, key, lab, grid)
        table[key + "_mean"] = np.nanmean(ys, 0)
        if key == "success_ma":
            for m, v in base.items():
                ax.axhline(v, color=BASE_COLORS[m], ls="--", lw=1.0, label=f"{m} ({100 * v:.1f} %)")
            ax.legend(fontsize=7, loc="upper left")
        ax.set_title(lab, fontsize=9)
    fig.suptitle(f"{prefix}: training curves of {len(runs)} run(s): " + ", ".join(r.name for r in runs), fontsize=10)
    fig.savefig(out_dir / f"{prefix}_curves.png", dpi=150)
    plt.close(fig)
    with (out_dir / f"{prefix}_curves.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(table))
        w.writerows(zip(*table.values()))
    return {"final_success_ma": [float(lg["success_ma"][-1]) for lg in logs], "final_return_ma": [float(lg["return_ma"][-1]) for lg in logs]}


def fig_diagnostics(runs: list[Path], prefix: str, out_dir: Path) -> None:
    logs = [read_log(r) for r in runs]
    keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var", "masked_share", "steps_per_s")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for ax, key in zip(axes.flat, keys):
        for i, lg in enumerate(logs):
            ax.plot(lg["iteration"], lg[key], color=COLORS[i % 5], lw=0.8, label=runs[i].name)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"{prefix}: PPO diagnostics (policy entropy starts near ln 9 = 2.2; a collapse or a KL spike marks instability)", fontsize=10)
    fig.savefig(out_dir / f"{prefix}_diagnostics.png", dpi=150)
    plt.close(fig)


def read_ckpt_eval(run: Path) -> list[dict]:
    """One row per evaluated checkpoint: team steps, primary / strict success and median final error on training and held-out sources."""
    log = read_log(run)
    out = []
    for f in sorted((run / "ckpt_eval").glob("*.csv")):
        steps = float(log["env_steps"][-1]) if f.stem == "final" else float(f.stem.split("_")[1])
        with f.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        row = {"run": run.name, "env_steps": steps, "n": len(rows)}
        for g, sel in (("train", [r for r in rows if int(r["source"]) not in config.HOLDOUT_SOURCES]),
                       ("holdout", [r for r in rows if int(r["source"]) in config.HOLDOUT_SOURCES])):
            row[f"success_{g}"] = float(np.mean([r["success"] == "True" for r in sel])) if sel else float("nan")
            row[f"strict_{g}"] = float(np.mean([r["success_strict"] == "True" for r in sel])) if sel else float("nan")
            row[f"error_{g}"] = float(np.median([float(r["final_error_m"]) for r in sel])) if sel else float("nan")
            row[f"n_{g}"] = len(sel)
        out.append(row)
    return sorted(out, key=lambda r: r["env_steps"])


def fig_ckpt_eval(runs: list[Path], prefix: str, out_dir: Path) -> list[dict]:
    allrows = [r for run in runs for r in read_ckpt_eval(run)]
    if not allrows:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for i, run in enumerate(runs):
        rows = [r for r in allrows if r["run"] == run.name]
        x = [r["env_steps"] / 1e6 for r in rows]
        for ax, k, lab in zip(axes, ("success", "strict", "error"), ("primary success rate", "strict success rate", "median final error [m]")):
            ax.plot(x, [r[f"{k}_train"] for r in rows], "o-", color=COLORS[i % 5], lw=1.2, label=f"{run.name} training sources")
            ax.plot(x, [r[f"{k}_holdout"] for r in rows], "s--", color=COLORS[i % 5], lw=1.0, alpha=0.7, label=f"{run.name} held-out sources")
            ax.set_xlabel("team steps [M]")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{prefix}: checkpoint quick evaluation (12 sources x {config.EVAL_CKPT_EPISODES_PER_SOURCE} episodes, 300 steps, sampled actions; small samples, curves only)", fontsize=10)
    fig.savefig(out_dir / f"{prefix}_ckpt_eval.png", dpi=150)
    plt.close(fig)
    with (out_dir / f"{prefix}_ckpt_eval.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    return allrows


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run names under config.TRAIN_ROOT (or paths)")
    ap.add_argument("--prefix", default="fig6_m1")
    ap.add_argument("--n-drones", type=int, default=1)
    ap.add_argument("--baseline-tag", default="t2_4_v3")
    ap.add_argument("--out-dir", type=Path, default=config.FIG_DIR)
    args = ap.parse_args(argv)
    runs = [Path(r) if Path(r).exists() else config.TRAIN_ROOT / r for r in args.runs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res = fig_curves(runs, args.prefix, args.out_dir, baseline_rates(args.baseline_tag, args.n_drones))
    fig_diagnostics(runs, args.prefix, args.out_dir)
    res["ckpt_eval"] = fig_ckpt_eval(runs, args.prefix, args.out_dir)
    print(f"[fig_training] wrote {args.prefix}_* to {args.out_dir}")
    return res


if __name__ == "__main__":
    main()
