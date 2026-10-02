"""End-to-end smoke test of the training / evaluation / visualisation pipeline (spec sections 6-11; D11).

Usage: python -m srcloc_env.scripts.smoke_pipeline [--truth-mode T2] [--n-drones 2] [--tag smoke_t2_2d] [--procs 3] [--skip-train] [--skip-eval]
One call exercises EVERY result the pipeline is supposed to produce, on tiny settings, and checks each output file:
  1. training  (rl/train.py): config.json, train_log.csv, episodes.csv (with the reward decomposition), checkpoints, final.pt, background checkpoint
     quick evaluation (ckpt_eval/*.csv and sample-trajectory step logs); plus a tiny 1-drone run so that the 1-versus-2-drone comparison can be drawn;
  2. evaluation (eval/run_eval.py): the trained policy and the reference methods on a common episode list (episodes.csv, records_*.csv, summary.json with
     Table 2, step logs);
  3. training figures (scripts/fig_training.py make_all) and evaluation figures / tables (scripts/fig_eval.py make_all);
  4. the 3-D episode video (scripts/render_episode_3d.py) of a trained-policy episode;
  5. validation of every file (PNG not blank, CSV columns, MP4 decodable with moving content, NPZ keys, JSON parse) and a report listing each result with its
     spec item, status and size: <out>/SMOKE_REPORT.md and <out>/smoke_manifest.json.  Exit status 1 when a required result is missing or invalid.
Outputs go to <CACHE_DIR>/smoke/<tag>/ (train/, eval/, figs/, video/); nothing is written into the repository.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from srcloc_env import config

PY = sys.executable


# ------------------------------------------------------------------------------------------ file validators
def check_png(path: Path, min_w: int = 400, min_h: int = 200) -> str | None:
    from PIL import Image
    im = np.asarray(Image.open(path).convert("L"), dtype=float)
    if im.shape[1] < min_w or im.shape[0] < min_h:
        return f"image too small {im.shape[1]}x{im.shape[0]}"
    if im.std() < 3.0:
        return "image is blank (grey-level std < 3)"
    return None


def check_csv(path: Path, cols: tuple[str, ...] = (), min_rows: int = 1) -> str | None:
    import csv
    with path.open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        head = rd.fieldnames or []
    miss = [c for c in cols if c not in head]
    if miss:
        return f"missing columns {miss}"
    if len(rows) < min_rows:
        return f"only {len(rows)} data rows (< {min_rows})"
    return None


def check_json(path: Path, keys: tuple[str, ...] = ()) -> str | None:
    d = json.loads(path.read_text(encoding="utf-8"))
    miss = [k for k in keys if k not in d]
    return f"missing keys {miss}" if miss else None


def check_mp4(path: Path, min_frames: int = 5) -> str | None:
    import imageio.v3 as iio
    frames = [f for f in iio.imiter(path)]
    if len(frames) < min_frames:
        return f"only {len(frames)} frames (< {min_frames})"
    if frames[0].std() < 3.0 or frames[-1].std() < 3.0:
        return "first or last frame is blank"
    if float(np.abs(frames[0].astype(float) - frames[-1].astype(float)).mean()) < 0.5:
        return "first and last frame are identical (nothing moves)"
    return None


def check_npz(path: Path, keys: tuple[str, ...]) -> str | None:
    with np.load(path, allow_pickle=False) as z:
        miss = [k for k in keys if k not in z.files]
    return f"missing arrays {miss}" if miss else None


def validate(path: Path | str | None, kind: str, **kw: Any) -> str | None:
    """None when the file is fine, else the reason it is not."""
    if path is None or not Path(path).exists():
        return "file does not exist"
    p = Path(path)
    if p.stat().st_size == 0:
        return "file is empty"
    try:
        return {"png": check_png, "csv": check_csv, "json": check_json, "mp4": check_mp4, "npz": check_npz}[kind](p, **kw) if kind != "any" else None
    except Exception as exc:   # noqa: BLE001 - an unreadable file is an invalid file
        return f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------------------------------------------ report
class Report:
    """Collects every produced result with its validation outcome."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []

    def add(self, section: str, item: str, path: Path | str | None, kind: str, required: bool = True, **kw: Any) -> dict[str, Any]:
        problem = validate(path, kind, **kw)
        row = {"section": section, "item": item, "path": str(path) if path else "", "kind": kind, "required": bool(required),
               "status": "ok" if problem is None else ("FAILED" if required else "warning"), "detail": problem or "",
               "size_kb": round(Path(path).stat().st_size / 1024, 1) if path and Path(path).exists() else 0.0}
        self.rows.append(row)
        return row

    def skipped(self, section: str, item: str, reason: str, allowed: bool) -> None:
        self.rows.append({"section": section, "item": item, "path": "", "kind": "-", "required": not allowed, "status": "skipped" if allowed else "FAILED",
                          "detail": reason, "size_kb": 0.0})

    def step(self, name: str, seconds: float, ok: bool, note: str = "") -> None:
        self.steps.append({"step": name, "seconds": round(seconds, 1), "ok": bool(ok), "note": note})

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["status"] == "FAILED"] + [{"step": s["step"], "detail": s["note"]} for s in self.steps if not s["ok"]]

    def write(self, out: Path, meta: dict[str, Any]) -> None:
        (out / "smoke_manifest.json").write_text(json.dumps({"meta": meta, "steps": self.steps, "results": self.rows,
                                                             "n_failed": len(self.failed)}, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = [f"# Smoke test of the training / evaluation / visualisation pipeline", "",
                 f"- settings: {json.dumps(meta, ensure_ascii=False)}", f"- result: **{'PASS' if not self.failed else 'FAIL'}**, {len(self.rows)} results checked, "
                 f"{sum(r['status'] == 'ok' for r in self.rows)} ok, {sum(r['status'] == 'skipped' for r in self.rows)} skipped, "
                 f"{sum(r['status'] == 'warning' for r in self.rows)} warnings, {len(self.failed)} failed", "", "## Steps", "", "| step | seconds | ok | note |", "|---|---|---|---|"]
        lines += [f"| {s['step']} | {s['seconds']} | {s['ok']} | {s['note']} |" for s in self.steps]
        lines += ["", "## Results", "", "| section | item | status | size [kB] | file | detail |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['section']} | {r['item']} | {r['status']} | {r['size_kb']} | {Path(r['path']).name if r['path'] else ''} | {r['detail']} |" for r in self.rows]
        (out / "SMOKE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)


# ------------------------------------------------------------------------------------------ steps 1-2: training and evaluation
STEP_LOG_KEYS = ("drone_xy", "y", "action", "applied", "gmm_w", "gmm_mu", "gmm_cov", "gmm_mask", "map_xy", "entropy", "top_sigma", "map_error",
                 "truth_xy", "frame", "reward", "calib_inside", "start_xy", "meta")


def step_train(a: argparse.Namespace, out: Path, rep: Report) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    main_name = f"smoke_{a.n_drones}d"
    plans = [(main_name, a.n_drones, a.steps, True)]
    if a.n_drones != 1 and not a.no_compare_run:
        plans.append(("smoke_1d", 1, max(a.steps // 2, 512), False))
    for name, n, steps, ckpt_eval in plans:
        t0 = time.perf_counter()
        cmd = [PY, "-u", "-m", "srcloc_env.rl.train", "--n-drones", str(n), "--run-seed", "1", "--run-name", name, "--out-root", str(out / "train"),
               "--truth-mode", a.truth_mode, "--total-steps", str(steps), "--n-steps", str(a.n_steps), "--procs", str(a.procs),
               "--ckpt-every-steps", str(a.ckpt_every), "--log-every", "1", "--ckpt-eval-per-source", "1", "--ckpt-eval-log-per-source", "1"]
        if not ckpt_eval:
            cmd.append("--no-ckpt-eval")
        rc = run_cmd(cmd, out / "logs" / f"train_{name}.log")
        rep.step(f"train {name} ({n} drone(s), {a.truth_mode}, {steps} team steps)", time.perf_counter() - t0, rc == 0, "" if rc == 0 else f"exit {rc}, see logs/train_{name}.log")
        d = out / "train" / name
        runs[name] = d
        if rc != 0:
            continue
        sec = f"1 training ({name})"
        rep.add(sec, "run configuration", d / "config.json", "json", keys=("args", "ppo", "truth_mode", "max_steps"))
        rep.add(sec, "per-iteration log (spec 6 CSV)", d / "train_log.csv", "csv", min_rows=2,
                cols=("iteration", "env_steps", "success_ma", "return_ma", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var", "masked_share"))
        rep.add(sec, "per-episode log with reward decomposition", d / "episodes.csv", "csv", cols=("env_steps", "source", "success", "ret", "ret_info", "ret_time", "ret_terminal", "final_error_m"))
        ckpts = sorted(d.glob("ckpt_*.pt"))
        rep.add(sec, f"checkpoints ({len(ckpts)} periodic) + final + latest", d / "final.pt", "any", required=len(ckpts) >= 2)
        rep.add(sec, "latest checkpoint (resume)", d / "latest.pt", "any")
        if ckpt_eval:
            evs = sorted((d / "ckpt_eval").glob("*.csv"))
            rep.add(sec, f"checkpoint quick evaluation ({len(evs)} CSV for {len(ckpts) + 1} checkpoints)", evs[-1] if evs else None, "csv", required=len(evs) == len(ckpts) + 1,
                    cols=("method", "episode_id", "source", "success", "final_error_m"))
            logs = sorted((d / "ckpt_eval").glob("*_steps/*.npz"))
            rep.add(sec, f"sample-trajectory step logs of the quick evaluation ({len(logs)})", logs[0] if logs else None, "npz", keys=STEP_LOG_KEYS)
    return runs


def step_eval(a: argparse.Namespace, out: Path, rep: Report, train_dir: Path) -> Path:
    t0 = time.perf_counter()
    ed = out / "eval"
    methods = ["random", "lawnmower", "greedy_map"] + (["gmm_infotaxis"] if a.with_infotaxis else [])
    cmd = [PY, "-u", "-m", "srcloc_env.eval.run_eval", "--mode", a.truth_mode, "--methods", *methods, "--ppo", f"ppo_smoke={train_dir / 'final.pt'}",
           "--n-drones", str(a.n_drones), "--n-per-source", str(a.eval_per_source), "--processes", str(a.procs), "--log-steps", "--tag", "smoke", "--out-dir", str(ed)]
    rc = run_cmd(cmd, out / "logs" / "eval.log")
    rep.step(f"evaluation ({len(methods) + 1} methods, {a.n_drones} drone(s), 13 sources x {a.eval_per_source})", time.perf_counter() - t0, rc == 0,
             "" if rc == 0 else f"exit {rc}, see logs/eval.log")
    if rc != 0:
        return ed
    sec = "2 evaluation"
    rep.add(sec, "common episode list", ed / "episodes.csv", "csv", cols=("episode_id", "seed", "source", "frame", "scale", "mode"))
    for m in methods + ["ppo_smoke"]:
        rep.add(sec, f"records of {m}", ed / f"records_{m}_{a.n_drones}drones.csv", "csv",
                cols=("method", "success", "steps", "final_error_m", "min_error_m", "declared_step", "path_length_m", "calib_2sigma_frac", "step_log"))
    rep.add(sec, "summary with Table 2 and paired differences", ed / "summary.json", "json", keys=("aggregates", "table2_markdown", "paired_vs_first_method", "mode", "max_steps"))
    logs = sorted((ed / "steps").glob("ppo_smoke_*drones_ep*.npz"))
    rep.add(sec, f"step logs of the trained policy ({len(logs)})", logs[0] if logs else None, "npz", keys=STEP_LOG_KEYS)
    return ed


# ------------------------------------------------------------------------------------------ steps 3-4: figures and video
ALLOWED_SKIPS = ("sensitivity",)          # outputs that need several evaluation tags: not available in a single-tag smoke run


def add_manifest(rep: Report, section: str, manifest: dict[str, dict], allow: tuple[str, ...] = ALLOWED_SKIPS) -> None:
    for name, ent in manifest.items():
        path = ent.get("path")
        if ent.get("status") == "ok":
            suf = Path(path).suffix.lower() if path else ""
            kind = {".png": "png", ".csv": "csv", ".json": "json", ".mp4": "mp4", ".md": "any"}.get(suf, "any")
            rep.add(section, name, path, kind)
        else:
            rep.skipped(section, name, ent.get("reason", ""), allowed=any(k in name for k in allow))


def step_figures(a: argparse.Namespace, out: Path, rep: Report, runs: dict[str, Path], eval_dir: Path) -> None:
    from srcloc_env.scripts import fig_eval, fig_training
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    main_run = runs[f"smoke_{a.n_drones}d"]
    t0 = time.perf_counter()
    try:
        compare = None
        if f"smoke_{a.n_drones}d" in runs and "smoke_1d" in runs:
            compare = [[str(runs["smoke_1d"])], [str(main_run)]]
        man = fig_training.make_all([str(main_run)], "smoke_train", figs, baseline_tag=str(eval_dir), n_drones=a.n_drones, compare=compare)
        add_manifest(rep, "3 training figures (spec 7)", man)
        rep.step("training figures", time.perf_counter() - t0, True, f"{sum(m['status'] == 'ok' for m in man.values())} ok / {len(man)}")
    except Exception as exc:   # noqa: BLE001
        rep.step("training figures", time.perf_counter() - t0, False, f"{type(exc).__name__}: {exc}")
    t0 = time.perf_counter()
    try:
        man = fig_eval.make_all([str(eval_dir)], figs, "smoke_eval", reference_method="random")
        add_manifest(rep, "3 evaluation figures and tables (spec 9-10)", man)
        rep.step("evaluation figures", time.perf_counter() - t0, True, f"{sum(m['status'] == 'ok' for m in man.values())} ok / {len(man)}")
    except Exception as exc:   # noqa: BLE001
        rep.step("evaluation figures", time.perf_counter() - t0, False, f"{type(exc).__name__}: {exc}")


def step_video(a: argparse.Namespace, out: Path, rep: Report, eval_dir: Path) -> None:
    t0 = time.perf_counter()
    vdir = out / "video"
    vdir.mkdir(parents=True, exist_ok=True)
    logs = sorted((eval_dir / "steps").glob(f"ppo_smoke_{a.n_drones}drones_ep*.npz"))
    if not logs:
        rep.step("3-D episode video", time.perf_counter() - t0, False, "no step log of the trained policy")
        return
    try:
        from srcloc_env.scripts.render_episode_3d import render_episode
        res = render_episode(logs[0], vdir / "episode_ppo.mp4", width=a.video_width, height=a.video_height, fps=8, stride=a.video_stride,
                             max_frames=a.video_frames, poster=vdir / "episode_ppo_poster.png")
        rep.step("3-D episode video", time.perf_counter() - t0, True, f"{res.get('frames')} frames in {res.get('seconds', 0):.0f} s")
        rep.add("4 episode video (spec 11)", "3-D animation of a trained-policy episode (MP4)", vdir / "episode_ppo.mp4", "mp4")
        rep.add("4 episode video (spec 11)", "last-frame poster", vdir / "episode_ppo_poster.png", "png", min_w=300, min_h=150)
    except Exception as exc:   # noqa: BLE001
        rep.step("3-D episode video", time.perf_counter() - t0, False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth-mode", choices=list(config.ENV_MODES), default="T2")
    ap.add_argument("--n-drones", type=int, default=2)
    ap.add_argument("--tag", default=None, help="output directory name under <CACHE_DIR>/smoke (default smoke_<mode>_<n>d)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3072, help="team steps of the tiny training run")
    ap.add_argument("--n-steps", type=int, default=256, help="team steps per worker per iteration")
    ap.add_argument("--ckpt-every", type=int, default=768)
    ap.add_argument("--eval-per-source", type=int, default=1)
    ap.add_argument("--with-infotaxis", action="store_true", help="also evaluate GMM-Infotaxis (slow)")
    ap.add_argument("--no-compare-run", action="store_true", help="skip the tiny 1-drone run used for the 1-versus-n-drone comparison figure")
    ap.add_argument("--video-width", type=int, default=960)
    ap.add_argument("--video-height", type=int, default=540)
    ap.add_argument("--video-stride", type=int, default=3)
    ap.add_argument("--video-frames", type=int, default=40)
    for s in ("train", "eval", "figs", "video"):
        ap.add_argument(f"--skip-{s}", action="store_true", help=f"reuse the existing {s} outputs of the output directory")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    tag = a.tag or f"smoke_{a.truth_mode.lower()}_{a.n_drones}d"
    out = a.out if a.out is not None else config.CACHE_DIR / "smoke" / tag
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    t_all = time.perf_counter()
    runs = {f"smoke_{a.n_drones}d": out / "train" / f"smoke_{a.n_drones}d"}
    if (out / "train" / "smoke_1d").exists():
        runs["smoke_1d"] = out / "train" / "smoke_1d"
    if not a.skip_train:
        runs = step_train(a, out, rep)
    main_run = runs[f"smoke_{a.n_drones}d"]
    eval_dir = out / "eval"
    if not a.skip_eval and (main_run / "final.pt").exists():
        eval_dir = step_eval(a, out, rep, main_run)
    if not a.skip_figs and (eval_dir / "summary.json").exists():
        step_figures(a, out, rep, runs, eval_dir)
    if not a.skip_video and (eval_dir / "steps").exists():
        step_video(a, out, rep, eval_dir)
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    meta["total_minutes"] = round((time.perf_counter() - t_all) / 60, 1)
    rep.write(out, meta)
    print(f"[smoke] {'PASS' if not rep.failed else 'FAIL'}: {len(rep.rows)} results checked, {len(rep.failed)} failed; report {out / 'SMOKE_REPORT.md'}")
    for f in rep.failed:
        print("   FAILED:", f)
    return 0 if not rep.failed else 1


if __name__ == "__main__":
    sys.exit(main())
