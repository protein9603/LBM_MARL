"""Figures and start-position statistics for docs/baseline_policies_and_initial_positions.md (D9-4).

Usage: python -m srcloc_env.scripts.fig_baseline_explainer starts   --tag t2_4_v2          # replay the resets of an evaluation batch
       python -m srcloc_env.scripts.fig_baseline_explainer figures --tag t2_4_v2 --episode 81 [--out-dir docs/figures]

``starts`` replays env.reset(seed, options) of every episode of <CACHE_DIR>/eval/<tag>/episodes.csv with the 2-drone environment
(the reset is deterministic: same seed -> same source / frame / scale / start positions) and writes starts.csv + starts.json
(distance to the source, drone separation, which drone starts inside the detectable plume region).
``figures`` draws (1) example start positions of a 2-drone episode, (2, 3) the four baselines on one episode with 1 and with 2
drones (from the batch step logs), (4) the lawnmower plan versus the flown track, (5) the expected-entropy scores of
GMM-Infotaxis at one decision (the belief at that step is rebuilt by replaying the logged actions).
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
from matplotlib.patches import Circle, Ellipse

from srcloc_env import config
from srcloc_env.baselines.policies import GmmInfotaxisPolicy, make_policy
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.env.multi_agent import MultiDroneEnv
from srcloc_env.env.source_env import Scene, SourceLocEnv
from srcloc_env.eval.episodes import load_episode_list
from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.baselines.coverage import coverage_waypoints
from srcloc_env.scripts.validate_pf_adjoint import lawnmower_path

METHODS = ("random", "lawnmower", "greedy_map", "gmm_infotaxis")
TITLES = {"random": "random", "lawnmower": "lawnmower", "greedy_map": "greedy-MAP", "gmm_infotaxis": "GMM-Infotaxis"}
COLS = ("tab:blue", "tab:green")


def _plume_mask(be, source: int, frame: int, scale: float, det) -> tuple[np.ndarray, object]:
    sf = be.slab(frame)
    dens = sf.density[list(sf.sources).index(source), sf.z_index(config.DRONE_Z)].astype(float)
    thr = config.ENV_START_PLUME_MIN_COUNTS_FACTOR * det.detection_threshold_cps() * det.T
    return det.expected_counts(dens, scale) >= thr, sf.grid


def _base(ax, om: ObstacleMap, window, dens_grid=None, grid=None, source_xy=None, plume=None):
    if dens_grid is not None:
        ax.imshow(np.log10(np.maximum(dens_grid, dens_grid.max() * 1e-4)), origin="lower",
                  extent=(grid.x0, grid.x0 + grid.nx * grid.res, grid.y0, grid.y0 + grid.ny * grid.res), cmap="Greys",
                  vmin=np.log10(dens_grid.max()) - 4, vmax=np.log10(dens_grid.max()), alpha=0.6, interpolation="nearest")
    if plume is not None:
        ax.contourf(grid.x_centres, grid.y_centres, plume.astype(float), levels=[0.5, 1.5], colors=["orange"], alpha=0.35)
    ox, oy = np.meshgrid(om.x, om.y, indexing="ij")
    ax.contour(ox, oy, om.occ.astype(float), levels=[0.5], colors="0.35", linewidths=0.5)
    if source_xy is not None:
        ax.plot(*source_xy, marker="*", color="gold", markeredgecolor="k", markersize=15, zorder=6)
    ax.set_xlim(window[0], window[2]); ax.set_ylim(window[1], window[3]); ax.set_aspect("equal")


def _window(points: np.ndarray, margin: float = 80.0, min_aspect: float = 0.6):
    lo, hi = points.min(axis=0) - margin, points.max(axis=0) + margin
    for a, b in ((1, 0), (0, 1)):
        span, other = hi[a] - lo[a], hi[b] - lo[b]
        if span < min_aspect * other:
            mid = 0.5 * (hi[a] + lo[a]); lo[a], hi[a] = mid - 0.5 * min_aspect * other, mid + 0.5 * min_aspect * other
    return lo[0], lo[1], hi[0], hi[1]


# ----------------------------------------------------------------------------------------------- start statistics
def replay_starts(tag: str, n_drones: int = 2) -> list[dict]:
    d = config.CACHE_DIR / "eval" / tag
    specs = load_episode_list(d / "episodes.csv")
    scene = Scene.load()
    env = MultiDroneEnv(scene, n_drones=n_drones, sources=config.ALL_SOURCES, truth_mode="F", reflect_prob=0.0, n_particles=50,
                        terminate_on_success=False)
    det, be = env.det, scene.backend
    rows = []
    for sp in specs:
        _, info = env.reset(seed=sp.seed, options=sp.reset_options())
        xy = np.asarray(info["drone_xy"]).reshape(n_drones, 2)
        mask, grid = _plume_mask(be, sp.source, sp.frame, sp.scale, det)
        ix = np.clip(np.floor((xy[:, 0] - grid.x0) / grid.res).astype(int), 0, grid.nx - 1)
        iy = np.clip(np.floor((xy[:, 1] - grid.y0) / grid.res).astype(int), 0, grid.ny - 1)
        inp = mask[iy, ix]
        tx = np.asarray(info["truth_xy"])
        row = {"episode_id": sp.episode_id, "source": sp.source, "frame": sp.frame, "scale": sp.scale, "start_type": info["start_type"],
               "plume_cells": int(mask.sum()), "separation_m": float(np.hypot(*(xy[0] - xy[1]))) if n_drones > 1 else float("nan")}
        for k in range(n_drones):
            row.update({f"x{k}": float(xy[k, 0]), f"y{k}": float(xy[k, 1]), f"dist_src{k}": float(np.hypot(*(xy[k] - tx))),
                        f"in_plume{k}": bool(inp[k])})
        rows.append(row)
    return rows


def start_summary(rows: list[dict]) -> dict:
    a = np.array([[r["in_plume0"], r["in_plume1"]] for r in rows], dtype=bool)
    d0, d1 = np.array([r["dist_src0"] for r in rows]), np.array([r["dist_src1"] for r in rows])
    sep = np.array([r["separation_m"] for r in rows])
    return {"n_episodes": len(rows),
            "drone0_in_plume": float(a[:, 0].mean()), "drone1_in_plume": float(a[:, 1].mean()),
            "both_in_plume": float((a[:, 0] & a[:, 1]).mean()), "neither_in_plume": float((~a[:, 0] & ~a[:, 1]).mean()),
            "at_least_one_in_plume": float((a[:, 0] | a[:, 1]).mean()),
            "dist_to_source_m": {"min": float(min(d0.min(), d1.min())), "p10": float(np.percentile(np.r_[d0, d1], 10)),
                                 "median": float(np.median(np.r_[d0, d1])), "p90": float(np.percentile(np.r_[d0, d1], 90))},
            "separation_m": {"min": float(sep.min()), "p10": float(np.percentile(sep, 10)), "median": float(np.median(sep)),
                             "p90": float(np.percentile(sep, 90))},
            "no_plume_cells_episodes": int(sum(1 for r in rows if r["plume_cells"] == 0))}


# ----------------------------------------------------------------------------------------------- figures
def fig_starts(out: Path, sources=(108, 111), n_draws: int = 40, frame: int = 500, scale: float = 1.0) -> dict:
    scene = Scene.load()
    env = MultiDroneEnv(scene, n_drones=2, sources=config.ALL_SOURCES, n_particles=50, truth_mode="F", reflect_prob=0.0)
    om, det, be = scene.obstacles, env.det, scene.backend
    fig, axes = plt.subplots(1, len(sources), figsize=(7.2 * len(sources), 6.4), constrained_layout=True, squeeze=False)
    stats = {}
    for ax, s in zip(axes[0], sources):
        mask, grid = _plume_mask(be, s, frame, scale, det)
        sf = be.slab(frame)
        dens = sf.density[list(sf.sources).index(s), sf.z_index(config.DRONE_Z)].astype(float)
        tx = np.array(config.SOURCES_XY[s])
        pts = []
        for seed in range(n_draws):
            _, info = env.reset(seed=seed, options={"source": s, "frame": frame, "scale": scale})
            pts.append(np.asarray(info["drone_xy"]))
        pts = np.array(pts)                                           # (n_draws, 2, 2)
        win = (grid.x0, -560.0, grid.x0 + grid.nx * grid.res, 560.0)
        _base(ax, om, win, dens, grid, tx, mask)
        ax.add_patch(Circle(tx, config.ENV_START_MIN_DIST_M, fill=False, edgecolor="k", linestyle="--", linewidth=1.0))
        ix = np.clip(np.floor((pts[..., 0] - grid.x0) / grid.res).astype(int), 0, grid.nx - 1)
        iy = np.clip(np.floor((pts[..., 1] - grid.y0) / grid.res).astype(int), 0, grid.ny - 1)
        inp = mask[iy, ix]
        for k, marker in ((0, "o"), (1, "^")):
            ax.scatter(pts[:, k, 0][inp[:, k]], pts[:, k, 1][inp[:, k]], marker=marker, s=46, c="tab:red", edgecolors="k", linewidths=0.6, zorder=5,
                       label=f"drone {k} - starts inside the plume region" if s == sources[0] else None)
            ax.scatter(pts[:, k, 0][~inp[:, k]], pts[:, k, 1][~inp[:, k]], marker=marker, s=46, c="tab:blue", edgecolors="k", linewidths=0.6, zorder=5,
                       label=f"drone {k} - starts elsewhere (random)" if s == sources[0] else None)
        ax.set_title(f"source {s} (gold star): {n_draws} resets of the 2-drone environment, frame {frame}, scale {scale:g}\n"
                     f"orange = detectable plume region (expected count >= {det.detection_threshold_cps():.1f} cps), dashed = 200 m exclusion circle", fontsize=9)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        stats[s] = {"drone0_in_plume": float(inp[:, 0].mean()), "drone1_in_plume": float(inp[:, 1].mean()), "plume_cells": int(mask.sum())}
    axes[0][0].legend(loc="upper left", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    return stats


def _load_log(tag: str, method: str, n: int, ep: int):
    z = np.load(config.CACHE_DIR / "eval" / tag / "steps" / f"{method}_{n}drones_ep{ep:04d}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))


def fig_policies(tag: str, ep: int, n: int, out: Path, lawn_tag: str | None = None) -> dict:
    be, om = LdmSlabBackend(), ObstacleMap.load()
    logs = {m: _load_log(lawn_tag if (m == "lawnmower" and lawn_tag) else tag, m, n, ep) for m in METHODS}
    meta0 = logs["random"][1]
    src, frame = int(meta0["source"]), int(meta0["frame"])
    sf = be.slab(frame)
    dens = sf.density[list(sf.sources).index(src), sf.z_index(config.DRONE_Z)].astype(float)
    truth = logs["random"][0]["truth_xy"]
    allp = np.vstack([l[0]["drone_xy"].reshape(-1, 2) for l in logs.values()] + [truth[None, :]])
    win = _window(allp)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6), constrained_layout=True)
    summary = {}
    for ax, m in zip(axes.ravel(), METHODS):
        log, meta = logs[m]
        _base(ax, om, win, dens, sf.grid, truth)
        T = log["drone_xy"].shape[0]
        for d in range(n):
            xy = log["drone_xy"][:, d]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(T), cmap="viridis", s=7, zorder=3)
            ax.plot(xy[:, 0], xy[:, 1], "-", color=COLS[d], linewidth=0.7, alpha=0.5, zorder=2)
            ax.plot(*xy[0], marker="s", markersize=8, color=COLS[d], markeredgecolor="k", zorder=5)
            ax.plot(*xy[-1], marker="o", markersize=8, color=COLS[d], markeredgecolor="k", zorder=5)
        mu, cov, w, mk = log["gmm_mu"][-1], log["gmm_cov"][-1], log["gmm_w"][-1], log["gmm_mask"][-1]
        for j in range(len(w)):
            if mk[j]:
                vals, vecs = np.linalg.eigh(cov[j]); vals = np.maximum(vals, 1e-6)
                ax.add_patch(Ellipse(mu[j], 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])),
                                     fill=False, edgecolor="tab:red", linewidth=1 + 2 * float(w[j])))
        ax.plot(*log["map_xy"][-1], marker="x", color="tab:red", markersize=9, markeredgewidth=2, zorder=6)
        dmin = float(np.hypot(*(log["drone_xy"] - truth).transpose(2, 0, 1)).min())
        path = float(np.sum(np.hypot(*np.diff(log["drone_xy"], axis=0).transpose(2, 0, 1))))
        ok = "SUCCESS" if meta["success"] else "no success"
        ax.set_title(f"{TITLES[m]}: {ok}\nfinal MAP error {log['map_error'][-1]:.0f} m | closest approach {dmin:.0f} m | path {path:.0f} m", fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        summary[m] = {"success": bool(meta["success"]), "final_error_m": float(log["map_error"][-1]), "closest_approach_m": dmin, "path_m": path}
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02, location="bottom", aspect=40).set_label("step")
    fig.suptitle(f"Episode {ep}: source {src} (gold star), frozen LDM frame {frame}, scale {float(meta0['scale']):.2f}, {n} drone{'s' if n > 1 else ''}\n"
                 f"squares = start, circles = end, grey = 15 m plume density (log), red = final belief 2-sigma ellipses, x = MAP", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    return summary


def fig_lawnmower_band_plan(tag: str, ep: int, n: int, out: Path) -> dict:
    be, om = LdmSlabBackend(), ObstacleMap.load()
    log, meta = _load_log(tag, "lawnmower", n, ep)
    src, frame = int(meta["source"]), int(meta["frame"])
    sf = be.slab(frame)
    dens = sf.density[list(sf.sources).index(src), sf.z_index(config.DRONE_Z)].astype(float)
    T = log["drone_xy"].shape[0]
    fig, axes = plt.subplots(n, 2, figsize=(15.0, 5.0 * n), constrained_layout=True, squeeze=False)
    info = {}
    for d in range(n):
        xy = log["drone_xy"][:, d]
        path, pinfo = lawnmower_path(om, float(xy[0, 0]), float(xy[0, 1]), config.MAX_EPISODE_STEPS + 1, step_m=config.DRONE_STEP_M,
                                     sweep_width=config.PF_ADJ_SWEEP_WIDTH_M, direction=-1.0, z=config.DRONE_Z, prior_x=config.PF_PRIOR_X,
                                     prior_y=config.PF_PRIOR_Y, phase_up=(d % 2 == 0))
        info[d] = {"plan_y_range_m": float(path[:, 1].max() - path[:, 1].min()), "flown_y_range_m": float(xy[40:, 1].max() - xy[40:, 1].min()),
                   "mean_speed_m_per_step": float(np.hypot(*np.diff(xy, axis=0).T).mean()),
                   "plan_waypoint_spacing_m": float(np.hypot(*np.diff(path, axis=0).T).mean())}
        win = _window(np.vstack([path[:120], xy, np.array(config.SOURCES_XY[src])[None, :]]), margin=60.0)
        for ax, which in zip(axes[d], ("planned waypoints (waypoint number = step number)", "flown track (colour = step)")):
            _base(ax, om, win, dens, sf.grid, config.SOURCES_XY[src])
            if which.startswith("planned"):
                ax.plot(path[:, 0], path[:, 1], ".-", color=COLS[d], linewidth=0.8, markersize=3, alpha=0.8)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(T), cmap="viridis", s=7, zorder=3)
                ax.plot(xy[:, 0], xy[:, 1], "-", color=COLS[d], linewidth=0.7, alpha=0.5)
            ax.plot(*xy[0], marker="s", markersize=8, color=COLS[d], markeredgecolor="k", zorder=5)
            ax.set_title(f"drone {d}: {which}", fontsize=10); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.suptitle(f"Lawnmower, episode {ep} ({n} drone{'s' if n > 1 else ''}): planned sawtooth (100 m wide, runs back and forth along x) versus the track the drone really flies",
                 fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    return info


def fig_lawnmower_plan(tag: str, ep: int, n: int, out: Path) -> dict:
    """Full-coverage lawnmower: planned waypoints per drone (left) versus the flown track (right) over the whole prior box."""
    be, om = LdmSlabBackend(), ObstacleMap.load()
    log, meta = _load_log(tag, "lawnmower", n, ep)
    src, frame = int(meta["source"]), int(meta["frame"])
    sf = be.slab(frame)
    dens = sf.density[list(sf.sources).index(src), sf.z_index(config.DRONE_Z)].astype(float)
    T = log["drone_xy"].shape[0]
    starts = log["drone_xy"][0]                                             # position after the first move (within 5 m of the start)
    plans, pinfo = coverage_waypoints(starts, (config.PF_PRIOR_X, config.PF_PRIOR_Y), "cross", config.LAWN_ROW_SPACING_M,
                                      config.LAWN_WP_SPACING_M, om, config.DRONE_Z)
    win = (300.0, -580.0, 1340.0, 580.0)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.4), constrained_layout=True)
    for d in range(n):
        xy = log["drone_xy"][:, d]
        axes[0].plot(plans[d][:, 0], plans[d][:, 1], ".-", color=COLS[d], linewidth=0.8, markersize=3, label=f"drone {d}: rows {pinfo['orders'][d]} in this order")
        for ax in axes:
            ax.plot(*xy[0], marker="s", markersize=9, color=COLS[d], markeredgecolor="k", zorder=6)
        axes[1].scatter(xy[:, 0], xy[:, 1], c=np.arange(T), cmap="viridis" if d == 0 else "autumn", s=6, zorder=3)
    for ax, title in zip(axes, (f"planned waypoints ({pinfo['n_rows']} cross-wind rows, {config.LAWN_ROW_SPACING_M:.0f} m apart; "
                                f"{'sub-areas: ' + str(pinfo['groups']) if n > 1 else 'one drone sweeps all rows'})", "flown track (colour = step)")):
        _base(ax, om, win, dens, sf.grid, config.SOURCES_XY[src])
        ax.set_title(title, fontsize=10); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Full-coverage lawnmower, episode {ep}, {n} drone{'s' if n > 1 else ''}: rows are transects across the wind (along y); squares = start", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    return {"groups": pinfo["groups"], "orders": pinfo["orders"], "n_rows": pinfo["n_rows"], "success": bool(meta["success"]),
            "final_error_m": float(log["map_error"][-1]), "path_m": [float(np.hypot(*np.diff(log["drone_xy"][:, d], axis=0).T).sum()) for d in range(n)]}


def fig_infotaxis_scores(tag: str, ep: int, step: int, out: Path) -> dict:
    """Rebuild the belief at ``step`` of a 1-drone GMM-Infotaxis episode by replaying its logged actions, then score the 9 actions."""
    log, meta = _load_log(tag, "gmm_infotaxis", 1, ep)
    scene = Scene.load()
    env = SourceLocEnv(scene, sources=config.ALL_SOURCES, truth_mode="F", reflect_prob=0.0, terminate_on_success=False)
    spec = {sp.episode_id: sp for sp in load_episode_list(config.CACHE_DIR / "eval" / tag / "episodes.csv")}[ep]
    obs, info = env.reset(seed=spec.seed, options=spec.reset_options())
    for t in range(step):
        obs, r, term, trunc, info = env.step(int(log["action"][t, 0]))
    pol = GmmInfotaxisPolicy(); pol.reset(env, info)
    a = int(pol.act(env, obs, info)[0])
    scores = pol.last_scores[0]
    mask = env.action_mask()
    names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE", "stay"]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1.0]})
    vals = np.where(mask, scores, np.nan)
    best = np.nanmin(vals)
    colours = ["tab:orange" if (mask[i] and scores[i] <= best + pol.tie_tol) else ("0.75" if not mask[i] else "tab:blue") for i in range(9)]
    ax[0].bar(range(9), np.where(mask, scores, 0.0), color=colours, edgecolor="k", linewidth=0.8)
    ax[0].bar([a], [scores[a]], color="none", edgecolor="tab:green", linewidth=3.5)
    ax[0].set_ylim(max(0.0, np.nanmin(vals) - 0.25), np.nanmax(vals) + 0.1)
    ax[0].axhline(best + pol.tie_tol, color="tab:red", linestyle="--", linewidth=1.0)
    ax[0].text(8.4, best + pol.tie_tol, f" best + {pol.tie_tol} nats (tie band)", va="bottom", ha="right", fontsize=8, color="tab:red")
    ax[0].set_xticks(range(9)); ax[0].set_xticklabels(names)
    ax[0].set_ylabel("expected entropy after 1 measurement [nats]")
    for i in range(9):
        if not mask[i]:
            ax[0].text(i, ax[0].get_ylim()[0] + 0.01, "blocked", rotation=90, ha="center", va="bottom", fontsize=9, color="0.4")
    ax[0].set_title(f"GMM-Infotaxis scores of the 9 actions, episode {ep}, step {step}\n"
                    f"lower = more informative; orange = within the tie band of the best; green outline = chosen action ({names[a]})", fontsize=9)
    be = LdmSlabBackend(); om = ObstacleMap.load()
    sf = be.slab(spec.frame)
    dens = sf.density[list(sf.sources).index(spec.source), sf.z_index(config.DRONE_Z)].astype(float)
    xy = env.xy
    win = (xy[0] - 220, xy[1] - 160, xy[0] + 220, xy[1] + 160)
    _base(ax[1], om, win, dens, sf.grid, env.truth_xy)
    ax[1].scatter(env.pf.xy[:, 0], env.pf.xy[:, 1], s=2, c="tab:red", alpha=0.25, label="PF particles (belief)")
    ax[1].plot(*xy, marker="o", markersize=9, color="tab:blue", markeredgecolor="k", zorder=6, label="drone")
    ax[1].plot(*env.truth_xy, marker="*", color="gold", markeredgecolor="k", markersize=15, zorder=6)
    ax[1].set_title("situation at that step: belief particles (red), drone (blue), true source (star)", fontsize=9); ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("y [m]")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    return {"episode": ep, "step": step, "scores": [None if not mask[i] else float(scores[i]) for i in range(9)], "chosen": names[a],
            "spread_nats": float(np.nanmax(vals) - np.nanmin(vals)), "n_within_tie_band": int(np.sum(mask & (scores <= best + pol.tie_tol)))}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["starts", "figures"])
    ap.add_argument("--tag", default="t2_4_v2")
    ap.add_argument("--episode", type=int, default=81)
    ap.add_argument("--infotaxis-episode", type=int, default=48)
    ap.add_argument("--lawn-tag", default="t2_4_lawn_v2", help="evaluation tag holding the full-coverage lawnmower logs")
    ap.add_argument("--infotaxis-step", type=int, default=25)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "docs" / "figures")
    args = ap.parse_args(argv)
    d = config.CACHE_DIR / "eval" / args.tag
    if args.cmd == "starts":
        rows = replay_starts(args.tag)
        with (d / "starts.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        summ = start_summary(rows)
        (d / "starts.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
        print(json.dumps(summ, indent=2))
        return summ
    out = {}
    out["starts"] = fig_starts(args.out_dir / "start_positions_two_drones.png")
    out["policies_1drone"] = fig_policies(args.tag, args.episode, 1, args.out_dir / "baselines_one_drone_example.png", args.lawn_tag)
    out["policies_2drones"] = fig_policies(args.tag, args.episode, 2, args.out_dir / "baselines_two_drones_example.png", args.lawn_tag)
    out["lawnmower"] = fig_lawnmower_plan(args.lawn_tag, args.episode, 2, args.out_dir / "lawnmower_plan_vs_track.png")
    out["lawnmower_1drone"] = fig_lawnmower_plan(args.lawn_tag, args.episode, 1, args.out_dir / "lawnmower_plan_vs_track_one_drone.png")
    out["lawnmower_band_v1"] = fig_lawnmower_band_plan(args.tag, args.episode, 2, args.out_dir / "lawnmower_v1_band_plan_vs_track.png")
    out["infotaxis"] = fig_infotaxis_scores(args.tag, args.infotaxis_episode, args.infotaxis_step, args.out_dir / "infotaxis_scores_example.png")
    (d / "explainer_numbers.json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return out


if __name__ == "__main__":
    main()
