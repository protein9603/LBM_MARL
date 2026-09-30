"""D5-2: connect the LBM adjoint forward model (pf/lbm_adjoint.LbmAdjointModel, plan 4.2b option B-3) to the
RB-PF (pf/particle_filter.RBPF, plan 4.3) and validate speed and synthetic convergence (plan 4.2b 검증
"갱신당 시간 <= 20 ms"; S1 T1-2 lite with the LBM model as truth; docs/lbm_forward_model.md section 5 step 4).

Usage: python -m srcloc_env.scripts.validate_pf_adjoint [--seed 0] [--n-seeds 5] [--n-steps 150]
Writes config.CACHE_DIR / validate_pf_adjoint.json and config.FIG_DIR / fig_pf_adjoint_convergence.png
(config.FIG_DPI_FINAL) + _preview.png (config.FIG_DPI_PREVIEW).

(1) Timing (plan 4.2b): RBPF(forward=LbmAdjointModel(operator from the real data at z = config.DRONE_Z, default
    AdjointParams), N = config.PF_N_PARTICLES, mode = "grid", obstacles = ObstacleMap.load()).  update() is timed
    along the two-drone lawnmower of source 109: pass 1 visits a new receptor position every step (psi LRU miss
    -> adjoint solve + interpolation + likelihood), pass 2 revisits the same positions (LRU hit -> interpolation
    + likelihood).  Reported: median / p99 of the miss and hit updates and the breakdown measured separately
    (operator.solve_adjoint, operator.interpolate over the N particles, RBPF grid likelihood with a stub forward
    model).  PASS if the miss median < config.PF_ADJ_UPDATE_TIME_TARGET_S (20 ms).

(2) Synthetic convergence with the adjoint model as truth (no model mismatch beyond the off-centre footprint
    interpolation, section 5 step 4): for each source in config.PF_ADJ_SOURCES the truth field is
    operator.solve_forward(true_xy) x kappa_true with kappa_true = config.KAPPA_REF = SENSOR_K0 x q_true; counts
    y ~ Poisson((kappa_true g + b) T) are drawn with Detector.measure(density = g q_true) (R3 sensor model, plan
    4.1).  Two drones fly a sawtooth lawnmower (``lawnmower_path``): each RL step advances config.DRONE_STEP_M in
    -x (start x_s + config.PF_ADJ_START_DOWNWIND_M, i.e. >= 200 m downwind, flying upwind into the plume) while
    y zig-zags with the same step over a band of width config.PF_ADJ_SWEEP_WIDTH_M; the two bands are adjacent
    (y_s +- width / 2).  Waypoints that are not free at z (ObstacleMap.is_free, 2 m margin) or outside the PF
    prior box are skipped (dropped from the sequence); when the pattern runs out of free waypoints the drone
    hovers at its last free one.  RBPF with the same LbmAdjointModel, N = config.PF_N_PARTICLES, config.PF_ADJ_N_SEEDS
    seeds; per run: MAP error (map_estimate, weighted mode) after config.PF_ADJ_REPORT_STEPS steps, N_eff after
    every update, entropy_xy trajectory, kappa posterior quantiles vs kappa_true, number of resamples.
    Expected: open sources (109, 101, 113) median MAP error < config.PF_ADJ_MAP_ERROR_PASS_M after 150 steps;
    110 sits in a courtyard that may be enclosed at 15 m (validate_adjoint: 100 % of its response mass stays in
    the courtyard) - the script reports the maximum expected count along the path, whether any drone-free cell
    lies in the source's free component, and states explicitly whether 110 is observable at 15 m (observable
    iff the maximum expected rate along the path exceeds the Currie threshold, Detector.detection_threshold_cps).

Figure fig_pf_adjoint_convergence.png: MAP error vs RL step for the four sources (5 seeds thin, median bold) and
the final belief (particles coloured by weight) of source 109, seed 0, with the drone paths.
References: R3 (Poisson count likelihood), R5 (RB-PF), R16 (systematic resampling), R17 / R18 (adjoint
source-receptor field for all hypotheses at once).
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
from srcloc_env.field.wind import WindField
from srcloc_env.pf.lbm_adjoint import AdjointParams, AdvectionDiffusionOperator, LbmAdjointModel
from srcloc_env.pf.particle_filter import RBPF
from srcloc_env.sensor.detector import Detector

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# figure colours (dataviz palette slots 1 / 2 / 3 / 7; one series per panel, no adjacent-pair constraint)
SOURCE_COLORS = {109: "#2a78d6", 110: "#eb6834", 101: "#1baf7a", 113: "#4a3aa7"}
SEQ_CMAP = "Blues"                # particle weights (magnitude)
OUTLINE_COLOR = "#444444"         # building outline
INK_MUTED = "#8a8a8a"
TRUTH_COLOR = "#c8501e"           # truth star / MAP marker accent
PATH_COLORS = ("#1baf7a", "#4a3aa7")   # drone 1 / drone 2 tracks


# ---------------------------------------------------------------------------------------- lawnmower path
def lawnmower_path(obstacles: ObstacleMap | None, x_start: float, y_centre: float, n_steps: int,
                   step_m: float = config.DRONE_STEP_M, sweep_width: float = config.PF_ADJ_SWEEP_WIDTH_M,
                   direction: float = -1.0, z: float = config.DRONE_Z,
                   prior_x: tuple[float, float] = config.PF_PRIOR_X,
                   prior_y: tuple[float, float] = config.PF_PRIOR_Y,
                   phase_up: bool = True) -> tuple[np.ndarray, dict[str, int]]:
    """(n_steps, 2) waypoints of a sawtooth lawnmower (plan D5-2 / T1-4 "2대 lawnmower").

    Candidate k (k = 0, 1, ...) advances step_m in x per step along ``direction`` (the sign flips - the drone
    turns around - when the next x would leave the prior box, ``info['n_bounces']``) while y follows
    y_centre + a triangle wave of amplitude sweep_width / 2 that also advances step_m per step (first leg upward
    when phase_up, starting at the lower band edge).  Candidates that are not free at z (ObstacleMap.is_free,
    default 2 m margin) or outside the prior box are dropped (``info['n_skipped']``: the drone continues with the
    next free candidate); the first n_steps free candidates are returned.  Should fewer than n_steps free
    candidates exist among 4 n_steps candidates, the drone hovers at the last free one (``info['n_hover']``).
    """
    half = 0.5 * float(sweep_width)
    period = 4.0 * half / float(step_m)                      # steps per full up-and-down cycle
    max_k = 4 * int(n_steps)
    xs = np.empty(max_k)
    x, dirn, n_bounce = float(x_start), float(direction), 0
    for k in range(max_k):
        if k > 0:
            xn = x + dirn * float(step_m)
            if xn < prior_x[0] or xn > prior_x[1]:
                dirn, n_bounce = -dirn, n_bounce + 1
                xn = x + dirn * float(step_m)
            x = xn
        xs[k] = x
    k = np.arange(max_k, dtype=np.float64)
    phase = np.mod(k, period) / period                       # 0 .. 1
    tri = 4.0 * np.abs(phase - 0.5) - 1.0                    # triangle wave -1 .. 1 starting at +1
    if phase_up:
        tri = -tri                                           # start at -half going up
    y = float(y_centre) + half * tri
    cand = np.column_stack([xs, y])
    ok = ((cand[:, 0] >= prior_x[0]) & (cand[:, 0] <= prior_x[1])
          & (cand[:, 1] >= prior_y[0]) & (cand[:, 1] <= prior_y[1]))
    if obstacles is not None:
        ok &= obstacles.is_free(cand, z)
    free_idx = np.flatnonzero(ok)
    n_take = min(int(n_steps), free_idx.size)
    if n_take == 0:
        raise ValueError("lawnmower_path: no free waypoint at all")
    path = np.empty((int(n_steps), 2))
    path[:n_take] = cand[free_idx[:n_take]]
    path[n_take:] = cand[free_idx[n_take - 1]]
    used = int(free_idx[n_take - 1]) + 1
    info = {"n_candidates": used, "n_skipped": int(used - n_take), "n_hover": int(n_steps - n_take),
            "n_bounces": int(np.sum(np.sign(np.diff(xs[:used]))[1:] != np.sign(np.diff(xs[:used]))[:-1]))}
    return path, info

def two_drone_paths(obstacles: ObstacleMap | None, source_xy: tuple[float, float], n_steps: int,
                    n_drones: int = config.PF_ADJ_N_DRONES,
                    prior_x: tuple[float, float] = config.PF_PRIOR_X,
                    prior_y: tuple[float, float] = config.PF_PRIOR_Y) -> tuple[np.ndarray, list[dict[str, int]]]:
    """(n_steps, n_drones, 2) paths: adjacent y bands of width PF_ADJ_SWEEP_WIDTH_M centred on the source y
    (two drones: [y_s - w, y_s] and [y_s, y_s + w]), all starting PF_ADJ_START_DOWNWIND_M downwind (+x) and
    flying -x (plan D5-2).  The phase alternates so that each drone starts at its outer band edge and the two
    meet on the line y = y_s every w / DRONE_STEP_M steps (first at x_s + 150 m for the defaults)."""
    xs, ys = float(source_xy[0]), float(source_xy[1])
    w = config.PF_ADJ_SWEEP_WIDTH_M
    centres = ys + w * (np.arange(n_drones) - 0.5 * (n_drones - 1))
    paths, infos = [], []
    for d, yc in enumerate(centres):
        p, info = lawnmower_path(obstacles, xs + config.PF_ADJ_START_DOWNWIND_M, float(yc), n_steps,
                                 prior_x=prior_x, prior_y=prior_y, phase_up=(d % 2 == 0))
        paths.append(p)
        infos.append(info)
    return np.stack(paths, axis=1), infos


# ---------------------------------------------------------------------------------------- synthetic truth
class SyntheticTruth:
    """Truth = operator.solve_forward(true_xy) x kappa_true, counts via Detector (plan D5-2 (2), R3)."""

    def __init__(self, op: AdvectionDiffusionOperator, true_xy: tuple[float, float],
                 kappa_true: float = config.KAPPA_REF, detector: Detector | None = None) -> None:
        self.op = op
        self.true_xy = np.asarray(true_xy, dtype=np.float64)
        self.kappa_true = float(kappa_true)
        self.detector = detector if detector is not None else Detector()
        self.q_true = self.kappa_true / self.detector.k0            # kappa = k0 q  (plan 4.3)
        self.field = op.solve_forward(self.true_xy)                # unit-source density [particles/m^3 per particle/s]

    def unit_response(self, xy: np.ndarray) -> np.ndarray:
        return self.op.interpolate(self.field, np.asarray(xy, dtype=np.float64).reshape(-1, 2))

    def expected_counts(self, xy: np.ndarray) -> np.ndarray:
        """(kappa_true g + b) T at the positions (n, 2)."""
        return self.detector.expected_counts(self.unit_response(xy) * self.q_true)

    def counts(self, xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.detector.measure(self.unit_response(xy) * self.q_true, 1.0, rng)


def courtyard_report(op: AdvectionDiffusionOperator, om: ObstacleMap, source_xy: tuple[float, float]) -> dict:
    """Free component of the source cell at z: size, boundary outflow (enclosed iff 0) and whether any of its
    cells is drone-free (ObstacleMap.is_free with the flight margin) - i.e. whether a drone can enter it."""
    g = op.grid
    ix = int(np.floor((source_xy[0] - g.x0) / g.res))
    iy = int(np.floor((source_xy[1] - g.y0) / g.res))
    comp = int(op.component[iy, ix]) if (0 <= ix < g.nx and 0 <= iy < g.ny) else 0
    if comp == 0:
        return {"source_cell_blocked": True}
    sel = op.component == comp
    xx, yy = np.meshgrid(g.x_centres, g.y_centres)
    cells = np.column_stack([xx[sel], yy[sel]])
    free_for_drone = om.is_free(cells, config.DRONE_Z)
    outflow = float(np.bincount(op._comp_free, weights=op.outflow_coef, minlength=op.n_components + 1)[comp])
    return {"source_cell_blocked": False, "component_cells": int(sel.sum()),
            "component_area_m2": float(sel.sum() * g.res ** 2), "component_boundary_outflow_coef": outflow,
            "enclosed_no_outflow": bool(outflow == 0.0), "drone_free_cells_in_component": int(free_for_drone.sum()),
            "main_component_cells": int(np.bincount(op._comp_free)[1:].max())}


# ---------------------------------------------------------------------------------------- PF run
def run_pf(model: LbmAdjointModel, om: ObstacleMap, truth: SyntheticTruth, paths: np.ndarray,
           seed: int, n_particles: int = config.PF_N_PARTICLES,
           report_steps: tuple[int, ...] = config.PF_ADJ_REPORT_STEPS) -> dict:
    """One synthetic episode: counts from ``truth`` along ``paths`` (n_steps, n_drones, 2), sequential RBPF
    updates (plan 4.3 multi-drone fusion), per-step MAP error / entropy / N_eff."""
    rng_meas = np.random.default_rng([seed, 1])
    rng_pf = np.random.default_rng([seed, 2])
    n_steps, n_drones = paths.shape[:2]
    counts = truth.counts(paths.reshape(-1, 2), rng_meas).reshape(n_steps, n_drones)
    expected = truth.expected_counts(paths.reshape(-1, 2)).reshape(n_steps, n_drones)
    pf = RBPF(model, n_particles=n_particles, mode="grid", obstacles=om, rng=rng_pf)
    err = np.empty(n_steps)
    ent = np.empty(n_steps)
    neff = np.empty((n_steps, n_drones))
    marg = np.empty((n_steps, n_drones))
    t0 = time.perf_counter()
    for k in range(n_steps):
        for d in range(n_drones):
            marg[k, d] = pf.update(int(counts[k, d]), np.array([paths[k, d, 0], paths[k, d, 1], config.DRONE_Z]))
            neff[k, d] = pf.neff()
        err[k] = float(np.hypot(*(pf.map_estimate() - truth.true_xy)))
        ent[k] = pf.entropy_xy()
    wall = time.perf_counter() - t0
    q = pf.posterior_kappa_quantiles((0.05, 0.5, 0.95))
    cov = pf.covariance()
    return {
        "seed": int(seed), "n_steps": int(n_steps), "n_drones": int(n_drones), "wall_seconds": wall,
        "map_error_m": {str(s): float(err[s - 1]) for s in report_steps if s <= n_steps},
        "map_error_trajectory_m": err.tolist(), "entropy_trajectory_nats": ent.tolist(),
        "neff_min": float(neff.min()), "neff_median": float(np.median(neff)),
        "neff_after_last_update": float(neff[-1, -1]), "n_resamples": int(pf.n_resamples),
        "mean_error_m": float(np.hypot(*(pf.mean() - truth.true_xy))),
        "posterior_spread_m": float(np.sqrt(np.trace(cov))),
        "kappa_q05_q50_q95": q.tolist(), "kappa_median_over_true": float(q[1] / truth.kappa_true),
        "kappa_edge_mass": list(pf.kappa_edge_mass()),
        "counts_max": int(counts.max()), "counts_mean": float(counts.mean()),
        "expected_counts_max": float(expected.max()),
        "n_detections": int(truth.detector.is_detection(counts).sum()),
        "final_xy": pf.xy.tolist(), "final_w": pf.weights().tolist(), "map_xy": pf.map_estimate().tolist(),
    }


# ---------------------------------------------------------------------------------------- timing
class _StubForward:
    """Returns a fixed (N, M) unit response: isolates the RBPF likelihood cost from the forward model."""

    def __init__(self, g: np.ndarray) -> None:
        self.g = np.asarray(g, dtype=np.float64).reshape(-1, 1)

    def unit_response(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> np.ndarray:
        return self.g[: np.asarray(source_xy).shape[0]]


def _stats(t: np.ndarray) -> dict[str, float]:
    t = np.asarray(t, dtype=np.float64)
    return {"n": int(t.size), "median_s": float(np.median(t)), "mean_s": float(t.mean()),
            "p99_s": float(np.percentile(t, 99)), "max_s": float(t.max())}


def timing_section(model: LbmAdjointModel, om: ObstacleMap, truth: SyntheticTruth, paths: np.ndarray,
                   seed: int) -> dict:
    """update() wall time along the lawnmower: pass 1 (LRU misses) and pass 2 (hits) + component breakdown."""
    op = model.operator
    rng = np.random.default_rng([seed, 7])
    n_steps, n_drones = paths.shape[:2]
    flat = paths.reshape(-1, 2)
    counts = truth.counts(flat, rng)
    model.clear_cache()
    model.hits = model.misses = 0
    pf = RBPF(model, n_particles=config.PF_N_PARTICLES, mode="grid", obstacles=om,
              rng=np.random.default_rng([seed, 8]))
    # pass 1: every waypoint (a new receptor position each step -> psi LRU miss); pass 2: the last
    # min(n, max_cached) waypoints again, in the same order (all still cached -> hits).  Note that revisiting ALL
    # n > max_cached positions sequentially would evict each entry before its reuse (LRU thrash), so a 2 x 150
    # step episode (300 positions > config.ADJ_MAX_CACHED = 256) is essentially all misses.
    n_all = flat.shape[0]
    idx_pass = [np.arange(n_all), np.arange(max(0, n_all - model.max_cached), n_all)]
    t_pass = [[], []]
    kinds = [[], []]
    for p in range(2):
        for i in idx_pass[p]:
            m0 = model.misses
            t0 = time.perf_counter()
            pf.update(int(counts[i]), np.array([flat[i, 0], flat[i, 1], config.DRONE_Z]))
            t_pass[p].append(time.perf_counter() - t0)
            kinds[p].append(model.misses > m0)
    t1, k1 = np.asarray(t_pass[0]), np.asarray(kinds[0])
    t2, k2 = np.asarray(t_pass[1]), np.asarray(kinds[1])
    miss = np.concatenate([t1[k1], t2[k2]])
    hit = np.concatenate([t1[~k1], t2[~k2]])
    # breakdown, measured separately at PF_ADJ_TIMING_N_BREAKDOWN positions of the path
    nb = min(config.PF_ADJ_TIMING_N_BREAKDOWN, flat.shape[0])
    sel = rng.choice(flat.shape[0], nb, replace=False)
    t_solve, t_interp, t_lik = [], [], []
    xy = pf.xy.copy()
    for i in sel:
        t0 = time.perf_counter()
        psi = op.solve_adjoint(flat[i])
        t_solve.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        g = op.interpolate(psi, xy)
        t_interp.append(time.perf_counter() - t0)
    stub = RBPF(_StubForward(np.maximum(g, config.FWD_G_FLOOR)), n_particles=config.PF_N_PARTICLES, mode="grid",
                obstacles=None, rng=np.random.default_rng([seed, 9]))
    for i in sel:
        t0 = time.perf_counter()
        stub.update(int(counts[i]), np.array([flat[i, 0], flat[i, 1], config.DRONE_Z]))
        t_lik.append(time.perf_counter() - t0)
    res = {
        "n_particles": config.PF_N_PARTICLES, "n_grid": config.KAPPA_G, "n_updates_pass1": int(n_all),
        "n_updates_pass2": int(idx_pass[1].size), "n_distinct_positions": int(len({tuple(k) for k in np.round(flat / model.round_m).astype(int)})),
        "pass1_n_miss": int(k1.sum()), "pass1_n_hit": int((~k1).sum()),
        "pass2_n_miss": int(k2.sum()), "pass2_n_hit": int((~k2).sum()),
        "update_miss": _stats(miss) if miss.size else None, "update_hit": _stats(hit) if hit.size else None,
        "pass1_all": _stats(t1), "pass2_all": _stats(t2),
        "breakdown": {"adjoint_solve": _stats(t_solve), "interpolation_N": _stats(t_interp),
                      "likelihood_grid_float64": _stats(t_lik)},
        "n_cached_after": int(model.n_cached), "max_cached": int(model.max_cached),
        "target_s": config.PF_ADJ_UPDATE_TIME_TARGET_S,
    }
    res["pass"] = bool(miss.size and res["update_miss"]["median_s"] < config.PF_ADJ_UPDATE_TIME_TARGET_S)
    return res


# ---------------------------------------------------------------------------------------- figure
def make_figure(runs: dict[int, list[dict]], truths: dict[int, SyntheticTruth], paths_by_src: dict[int, np.ndarray],
                om: ObstacleMap, op: AdvectionDiffusionOperator, path: Path, scatter_source: int = 109) -> None:
    """Top row: MAP error vs RL step per source (seeds thin, median bold); bottom left: final belief of
    ``scatter_source`` seed 0 with the drone paths; bottom right: entropy trajectories of all sources."""
    srcs = list(runs)
    fig = plt.figure(figsize=(16.0, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, len(srcs), height_ratios=[1.0, 1.4])
    n_steps = config.PF_ADJ_N_STEPS
    for j, s in enumerate(srcs):
        ax = fig.add_subplot(gs[0, j])
        col = SOURCE_COLORS.get(s, "#2a78d6")
        E = np.array([r["map_error_trajectory_m"] for r in runs[s]])
        steps = np.arange(1, E.shape[1] + 1)
        for row in E:
            ax.plot(steps, row, color=col, alpha=0.3, linewidth=1.0)
        ax.plot(steps, np.median(E, axis=0), color=col, linewidth=2.2)
        ax.axhline(config.PF_ADJ_MAP_ERROR_PASS_M, color=INK_MUTED, linewidth=1.0, linestyle="--")
        ax.text(steps[-1], config.PF_ADJ_MAP_ERROR_PASS_M, f" {config.PF_ADJ_MAP_ERROR_PASS_M:.0f} m",
                color=INK_MUTED, fontsize=8, va="bottom", ha="right")
        ax.set_yscale("log")
        ax.set_ylim(1.0, 1000.0)
        ax.set_xlim(0, n_steps)
        ax.set_xlabel("RL step (2 drones, 1 update each)")
        if j == 0:
            ax.set_ylabel("MAP error [m] (thin: seeds, bold: median)")
        med = np.median(E[:, -1])
        kind = "trapped" if s in config.T1_3_TRAPPED_SOURCES else "open"
        ax.set_title(f"source {s} ({kind}): final median {med:.0f} m", fontsize=9.5)
        ax.grid(True, which="major", color="#e6e6e6", linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # bottom left: belief scatter of scatter_source, seed 0
    half_cols = max(1, len(srcs) // 2)
    ax = fig.add_subplot(gs[1, :half_cols])
    r = runs[scatter_source][0]
    truth = truths[scatter_source]
    xs, ys = truth.true_xy
    half = config.PF_ADJ_FIG_HALF_WIDTH_M
    mask = om.no_fly_mask(config.DRONE_Z, margin=0.0).T.astype(float)
    ax.contour(om.x, om.y, mask, levels=[0.5], colors=OUTLINE_COLOR, linewidths=0.6)
    with np.errstate(divide="ignore"):
        z = np.log10(truth.field)
    vmax = np.log10(truth.field.max())
    ax.contourf(op.grid.x_centres, op.grid.y_centres, z, levels=np.linspace(vmax - 3.0, vmax, 7), cmap="Greys",
                alpha=0.35)
    xy = np.asarray(r["final_xy"])
    w = np.asarray(r["final_w"])
    order = np.argsort(w)
    sc = ax.scatter(xy[order, 0], xy[order, 1], c=w[order], cmap=SEQ_CMAP, s=9, linewidths=0.0,
                    vmin=0.0, vmax=float(w.max()))
    pth = paths_by_src[scatter_source]
    for d in range(pth.shape[1]):
        ax.plot(pth[:, d, 0], pth[:, d, 1], color=PATH_COLORS[d % len(PATH_COLORS)], linewidth=1.0, alpha=0.9,
                label=f"drone {d + 1} path (circle = start)")
        ax.plot(pth[0, d, 0], pth[0, d, 1], marker="o", color=PATH_COLORS[d % len(PATH_COLORS)], markersize=5)
    ax.plot(xs, ys, marker="*", color=TRUTH_COLOR, markersize=14, linestyle="none", label="true source")
    mx, my = r["map_xy"]
    ax.plot(mx, my, marker="x", color="#111111", markersize=9, markeredgewidth=1.8, linestyle="none",
            label=f"MAP after {r['n_steps']} steps (error {r['map_error_trajectory_m'][-1]:.1f} m)")
    ax.set_xlim(xs - half, xs + half + 100.0)
    ax.set_ylim(ys - half, ys + half)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"source {scatter_source}, seed {r['seed']}: final belief (blue = particle weight, grey = truth field, "
                 f"3 decades); kappa median / true = {r['kappa_median_over_true']:.2f}", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="particle weight")
    # bottom right: entropy trajectories (all sources, seeds thin, median bold, direct labels + legend)
    ax = fig.add_subplot(gs[1, half_cols:])
    for s in srcs:
        col = SOURCE_COLORS.get(s, "#2a78d6")
        H = np.array([r["entropy_trajectory_nats"] for r in runs[s]])
        steps = np.arange(1, H.shape[1] + 1)
        for row in H:
            ax.plot(steps, row, color=col, alpha=0.25, linewidth=0.9)
        med = np.median(H, axis=0)
        ax.plot(steps, med, color=col, linewidth=2.2, label=f"source {s}")
        ax.text(steps[-1] + 1.5, med[-1], str(s), color=col, fontsize=8, va="center")
    ax.set_xlim(0, n_steps + 12)
    ax.set_xlabel("RL step")
    ax.set_ylabel(f"belief entropy [nats] ({config.PF_ENTROPY_CELL_M:.0f} m cells)")
    ax.set_title("entropy of the position belief (plan 4.5 reward signal)", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    p = op.params
    fig.suptitle(f"RB-PF with the LBM adjoint forward model as truth (z = {p.z:.0f} m, K = {p.K} m^2/s, "
                 f"lambda = {p.lam} 1/s, N = {config.PF_N_PARTICLES}, kappa_true = {config.KAPPA_REF:.3g})", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIG_DPI_FINAL)
    fig.savefig(path.with_name(path.stem + "_preview" + path.suffix), dpi=config.FIG_DPI_PREVIEW)
    plt.close(fig)

# ---------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=config.PF_ADJ_N_SEEDS)
    ap.add_argument("--n-steps", type=int, default=config.PF_ADJ_N_STEPS)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_pf_adjoint.json")
    ap.add_argument("--fig", type=Path, default=config.FIG_DIR / "fig_pf_adjoint_convergence.png")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    wf = WindField.load()
    om = ObstacleMap.load()
    params = AdjointParams()
    op = AdvectionDiffusionOperator.from_data(params, wf, om).factorize()
    model = LbmAdjointModel(op)
    setup_s = time.perf_counter() - t0
    det = Detector()

    truths = {s: SyntheticTruth(op, config.SOURCES_XY[s], config.KAPPA_REF, det) for s in config.PF_ADJ_SOURCES}
    paths_by_src, path_info = {}, {}
    for s in config.PF_ADJ_SOURCES:
        paths_by_src[s], path_info[s] = two_drone_paths(om, config.SOURCES_XY[s], args.n_steps)

    # (1) timing on the source-109 path
    timing = timing_section(model, om, truths[109], paths_by_src[109], args.seed)
    print(f"[timing] miss median {timing['update_miss']['median_s'] * 1e3:.2f} ms (p99 "
          f"{timing['update_miss']['p99_s'] * 1e3:.2f}), hit median {timing['update_hit']['median_s'] * 1e3:.2f} ms "
          f"(p99 {timing['update_hit']['p99_s'] * 1e3:.2f}); breakdown solve "
          f"{timing['breakdown']['adjoint_solve']['median_s'] * 1e3:.2f} / interp "
          f"{timing['breakdown']['interpolation_N']['median_s'] * 1e3:.2f} / likelihood "
          f"{timing['breakdown']['likelihood_grid_float64']['median_s'] * 1e3:.2f} ms ->",
          "PASS" if timing["pass"] else "FAIL", flush=True)

    # (2) synthetic convergence
    runs: dict[int, list[dict]] = {}
    per_source: dict[str, dict] = {}
    for s in config.PF_ADJ_SOURCES:
        truth = truths[s]
        paths = paths_by_src[s]
        flat = paths.reshape(-1, 2)
        exp_counts = truth.expected_counts(flat)
        start_dist = [float(np.hypot(*(paths[0, d] - truth.true_xy))) for d in range(paths.shape[1])]
        min_dist = float(np.min(np.hypot(flat[:, 0] - truth.true_xy[0], flat[:, 1] - truth.true_xy[1])))
        runs[s] = [run_pf(model, om, truth, paths, args.seed + i) for i in range(args.n_seeds)]
        err = {str(k): [r["map_error_m"][str(k)] for r in runs[s]] for k in config.PF_ADJ_REPORT_STEPS if k <= args.n_steps}
        max_rate = float(exp_counts.max() / det.T)
        observable = bool(max_rate > det.detection_threshold_cps())
        per_source[str(s)] = {
            "source_xy": list(config.SOURCES_XY[s]), "kind": "trapped" if s in config.T1_3_TRAPPED_SOURCES else "open",
            "kappa_true": truth.kappa_true, "q_true_particles_per_s": truth.q_true,
            "unit_response_max": float(truth.field.max()), "mass_balance": op.mass_balance(truth.field),
            "path_info": path_info[s], "start_distance_m": start_dist, "min_path_distance_m": min_dist,
            "expected_counts_along_path": {"max": float(exp_counts.max()), "median": float(np.median(exp_counts)),
                                           "background_counts": det.background * det.T,
                                           "n_above_currie": int((exp_counts / det.T > det.detection_threshold_cps()).sum())},
            "max_expected_rate_cps": max_rate, "currie_threshold_cps": det.detection_threshold_cps(),
            "observable_at_15m": observable,
            "courtyard": courtyard_report(op, om, config.SOURCES_XY[s]),
            "map_error_m": {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                                "values": [float(x) for x in v]} for k, v in err.items()},
            "neff_min": [r["neff_min"] for r in runs[s]], "neff_median": [r["neff_median"] for r in runs[s]],
            "n_resamples": [r["n_resamples"] for r in runs[s]],
            "entropy_start_mid_end_nats": [[r["entropy_trajectory_nats"][0], r["entropy_trajectory_nats"][len(r["entropy_trajectory_nats"]) // 2],
                                            r["entropy_trajectory_nats"][-1]] for r in runs[s]],
            "kappa_median_over_true": [r["kappa_median_over_true"] for r in runs[s]],
            "kappa_q05_q50_q95": [r["kappa_q05_q50_q95"] for r in runs[s]],
            "posterior_spread_m": [r["posterior_spread_m"] for r in runs[s]],
            "n_detections": [r["n_detections"] for r in runs[s]],
            "wall_seconds_per_run": [r["wall_seconds"] for r in runs[s]],
            "runs": [{k: v for k, v in r.items() if k not in ("final_xy", "final_w")} for r in runs[s]],
        }
        last = str(args.n_steps)
        print(f"[source {s}] MAP error after {last} steps: median {per_source[str(s)]['map_error_m'][last]['median']:.1f} m, "
              f"p90 {per_source[str(s)]['map_error_m'][last]['p90']:.1f} m; max expected counts {exp_counts.max():.1f} "
              f"(background {det.background * det.T:.0f}); observable at 15 m: {observable}; kappa med/true "
              f"{np.median(per_source[str(s)]['kappa_median_over_true']):.2f}; skipped/hover "
              f"{[ (i['n_skipped'], i['n_hover']) for i in path_info[s]]}", flush=True)

    open_srcs = [s for s in config.PF_ADJ_SOURCES if s not in config.T1_3_TRAPPED_SOURCES]
    last = str(args.n_steps)
    open_pass = all(per_source[str(s)]["map_error_m"][last]["median"] < config.PF_ADJ_MAP_ERROR_PASS_M for s in open_srcs)
    make_figure(runs, truths, paths_by_src, om, op, args.fig)

    res = {
        "params": {"K": params.K, "lam": params.lam, "h_layer": params.h_layer, "sigma0": params.sigma0, "z": params.z},
        "n_free": op.n_free, "setup_seconds": setup_s, "n_particles": config.PF_N_PARTICLES, "n_grid": config.KAPPA_G,
        "kappa_ref": config.KAPPA_REF, "grid_decades": config.KAPPA_GRID_DECADES, "eps_mix": config.PF_EPS_MIX,
        "n_seeds": args.n_seeds, "n_steps": args.n_steps, "n_drones": config.PF_ADJ_N_DRONES,
        "sweep_width_m": config.PF_ADJ_SWEEP_WIDTH_M, "start_downwind_m": config.PF_ADJ_START_DOWNWIND_M,
        "step_m": config.DRONE_STEP_M, "timing": timing, "sources": per_source,
        "open_sources": open_srcs, "map_error_pass_m": config.PF_ADJ_MAP_ERROR_PASS_M,
        "open_sources_pass": bool(open_pass), "timing_pass": bool(timing["pass"]),
        "figure": str(args.fig), "total_seconds": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"open sources median MAP error < {config.PF_ADJ_MAP_ERROR_PASS_M:.0f} m after {last} steps:",
          "PASS" if open_pass else "FAIL")
    print(f"timing (miss median < {config.PF_ADJ_UPDATE_TIME_TARGET_S * 1e3:.0f} ms):", "PASS" if timing["pass"] else "FAIL")
    print(f"total {res['total_seconds']:.0f} s; JSON {args.out}; figure {args.fig}")
    return res


if __name__ == "__main__":
    main()