"""D4-4 validation of pf/sph_adjoint.py on the real SPH wind field and building raster (plan 4.2b option B-1;
docs/sph_forward_model.md section 7).

Usage: python -m srcloc_env.scripts.validate_adjoint [--seed 0] [--out cache/validate_adjoint.json]
Writes config.CACHE_DIR / validate_adjoint.json and config.FIG_DIR / fig_adjoint_check.png and prints PASS/FAIL:
  * operator: from_data(AdjointParams(), WindField.load(), ObstacleMap.load()) at z = config.DRONE_Z; n_free,
    blocked fraction, assembly + factorisation time, L+U fill;
  * timing: adjoint solve median / p99 over config.ADJ_VALIDATE_N_RECEPTORS random free receptors; PASS if the
    median < config.ADJ_SOLVE_TIME_TARGET_S (20 ms, the environment-step budget);
  * reciprocity: config.ADJ_VALIDATE_N_PAIRS random (source cell, receptor cell) pairs, max relative difference
    of solve_forward(s)[p] and solve_adjoint(p)[s]; PASS if < config.ADJ_RECIPROCITY_TOL;
  * mass balance: |lam sum(C) area + outflow - 1| of the forward fields; PASS if < config.ADJ_MASS_BALANCE_TOL;
  * trapping (reported): forward fields of config.FWD_VALIDATE_SOURCES (109 open, 110 in the courtyard next
    to the 115 m tower; report 2.6): mass fraction within config.ADJ_VALIDATE_RADIUS_M of the source, figure
    fig_adjoint_check.png (log10 density, building outline, +-ADJ_FIG_HALF_WIDTH_M window);
  * informational: log10(n_LDM / (g q_A)) along the +x centreline at config.FWD_VALIDATE_D_M, the same
    diagnostic validate_forward reports for the analytic plume (skipped if the slab cache is missing).
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
from srcloc_env.pf.sph_adjoint import AdjointParams, AdvectionDiffusionOperator

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

SEQ_CMAP = "Blues"                # single-hue sequential scale for the magnitude (log10 density)
OUTLINE_COLOR = "#444444"         # building outline (neutral ink)
ACCENT_COLOR = "#c8501e"          # source marker and radius circle


def reciprocity_check(op: AdvectionDiffusionOperator, n_pairs: int, rng: np.random.Generator) -> dict:
    cc = op.free_cell_centres()
    rel = []
    for _ in range(n_pairs):
        s, p = op.random_free_cells(2, rng)
        a = op.solve_forward(cc[s]).reshape(-1)[op.unk_to_cell[p]]
        b = op.solve_adjoint(cc[p]).reshape(-1)[op.unk_to_cell[s]]
        m = max(abs(a), abs(b))
        rel.append(abs(a - b) / m if m > 0.0 else 0.0)
    return {"n_pairs": int(n_pairs), "max_rel_diff": float(np.max(rel)), "median_rel_diff": float(np.median(rel))}


def mass_fraction_within(op: AdvectionDiffusionOperator, field: np.ndarray, xy: tuple[float, float],
                         radius: float) -> float:
    xx, yy = np.meshgrid(op.grid.x_centres, op.grid.y_centres)
    near = np.hypot(xx - xy[0], yy - xy[1]) <= radius
    tot = field.sum()
    return float(field[near].sum() / tot) if tot > 0.0 else float("nan")


def centreline_vs_slab(op: AdvectionDiffusionOperator, field: np.ndarray, s: int) -> dict | None:
    """log10(n_LDM / (g q_A)) at config.FWD_VALIDATE_D_M downwind of source s (validate_forward diagnostic)."""
    try:
        from srcloc_env.field.concentration_field import LdmSlabBackend
        be = LdmSlabBackend(config.CACHE_DIR, max_cached_frames=1)
        xs, ys = config.SOURCES_XY[s]
        d = np.asarray(config.FWD_VALIDATE_D_M, dtype=np.float64)
        pts = np.column_stack([xs + d, np.full_like(d, ys)])
        n_ldm = be.density([s], pts, config.FWD_VALIDATE_FRAME_INDEX, config.DRONE_Z)
    except Exception as e:                     # slab cache missing: informational block only
        return {"skipped": str(e)}
    g = op.interpolate(field, pts)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho10 = np.log10(n_ldm / (g * config.Q_RELEASE_A))
    return {"d_m": d.tolist(), "g_unit": g.tolist(), "n_ldm_slab": n_ldm.tolist(),
            "log10_ratio_slab_over_g_qA": [float(v) if np.isfinite(v) else None for v in rho10]}


def make_figure(op: AdvectionDiffusionOperator, om: ObstacleMap, fields: dict[int, np.ndarray],
                fractions: dict[int, float], path: Path) -> None:
    mask = om.no_fly_mask(config.DRONE_Z, margin=0.0).T.astype(float)     # (ny, nx) for contour
    fig, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 5.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    half = config.ADJ_FIG_HALF_WIDTH_M
    for ax, (s, field) in zip(axes, fields.items()):
        xs, ys = config.SOURCES_XY[s]
        vmax = np.log10(field.max())
        with np.errstate(divide="ignore"):
            z = np.log10(field)
        pm = ax.pcolormesh(op.grid.x_centres, op.grid.y_centres, z, cmap=SEQ_CMAP,
                           vmin=vmax - config.FIG_LOG_DECADES, vmax=vmax, shading="nearest")
        ax.contour(om.x, om.y, mask, levels=[0.5], colors=OUTLINE_COLOR, linewidths=0.6)
        ax.plot(xs, ys, marker="*", color=ACCENT_COLOR, markersize=11, linestyle="none")
        ax.add_patch(plt.Circle((xs, ys), config.ADJ_VALIDATE_RADIUS_M, fill=False, color=ACCENT_COLOR,
                                linewidth=1.0, linestyle="--"))
        sub = slice(None, None, 4)
        xx, yy = np.meshgrid(op.grid.x_centres[sub], op.grid.y_centres[sub])
        ax.quiver(xx, yy, op.uv[sub, sub, 0], op.uv[sub, sub, 1], color="#777777", scale=60, width=0.0025)
        ax.set_xlim(xs - half, xs + half)
        ax.set_ylim(ys - half, ys + half)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"source {s}: mass within {config.ADJ_VALIDATE_RADIUS_M:.0f} m = {fractions[s]:.2f}")
        fig.colorbar(pm, ax=ax, shrink=0.8, label="log10 density [particles/m^3 per particle/s]")
    p = op.params
    fig.suptitle(f"SPH adjoint forward model, z = {p.z:.0f} m, K = {p.K} m^2/s, lambda = {p.lam} 1/s, "
                 f"h = {p.h_layer} m (unit source)", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIG_DPI_PREVIEW)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "validate_adjoint.json")
    ap.add_argument("--fig", type=Path, default=config.FIG_DIR / "fig_adjoint_check.png")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    wf = WindField.load()
    om = ObstacleMap.load()
    load_s = time.perf_counter() - t0
    params = AdjointParams()
    t0 = time.perf_counter()
    op = AdvectionDiffusionOperator.from_data(params, wf, om)
    build_s = time.perf_counter() - t0
    op.factorize()

    timing = op.time_adjoint_solves(config.ADJ_VALIDATE_N_RECEPTORS, rng)
    timing_pass = timing["median_s"] < config.ADJ_SOLVE_TIME_TARGET_S
    recip = reciprocity_check(op, config.ADJ_VALIDATE_N_PAIRS, rng)
    recip_pass = recip["max_rel_diff"] < config.ADJ_RECIPROCITY_TOL

    fields: dict[int, np.ndarray] = {}
    per_source: dict[str, dict] = {}
    mass_residuals = []
    for s in config.FWD_VALIDATE_SOURCES:
        xy = config.SOURCES_XY[s]
        field = op.solve_forward(np.array(xy))
        fields[s] = field
        mb = op.mass_balance(field)
        mass_residuals.append(abs(mb["residual"]))
        u, v = wf.uv_at(np.array([xy]), config.DRONE_Z)[0]
        per_source[str(s)] = {
            "source_xy": list(xy), "wind_15m_at_source": [float(u), float(v)],
            "mass_balance": mb,
            "mass_fraction_within_radius": mass_fraction_within(op, field, xy, config.ADJ_VALIDATE_RADIUS_M),
            "max_density_unit_source": float(field.max()),
            "density_at_source_cell": float(op.interpolate(field, np.array(xy))[0]),
            "centreline_vs_slab": centreline_vs_slab(op, field, s),
        }
    mass_pass = max(mass_residuals) < config.ADJ_MASS_BALANCE_TOL
    fractions = {s: per_source[str(s)]["mass_fraction_within_radius"] for s in fields}
    make_figure(op, om, fields, fractions, args.fig)

    res = {
        "params": {"K": params.K, "lam": params.lam, "h_layer": params.h_layer, "sigma0": params.sigma0,
                   "z": params.z, "wind_band": params.wind_band},
        "grid": op.grid.to_dict(), "n_cells": op.n_cells, "n_free": op.n_free,
        "blocked_fraction": float(op.blocked.mean()), "nnz_A": int(op.A.nnz), "nnz_F": int(op.F.nnz),
        "nnz_LU": int(op.lu.L.nnz + op.lu.U.nnz), "footprint_radius_cells": op.footprint_radius_cells,
        "data_load_seconds": load_s, "assembly_seconds_total": build_s,
        "assembly_seconds_operator": op.assembly_seconds, "factorize_seconds": op.factorize_seconds,
        "adjoint_solve_timing": timing, "timing_target_s": config.ADJ_SOLVE_TIME_TARGET_S, "timing_pass": bool(timing_pass),
        "reciprocity": recip, "reciprocity_tol": config.ADJ_RECIPROCITY_TOL, "reciprocity_pass": bool(recip_pass),
        "mass_balance_max_residual": float(max(mass_residuals)), "mass_balance_tol": config.ADJ_MASS_BALANCE_TOL,
        "mass_balance_pass": bool(mass_pass),
        "radius_m": config.ADJ_VALIDATE_RADIUS_M, "sources": per_source, "figure": str(args.fig),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"reciprocity max rel diff {recip['max_rel_diff']:.2e} (< {config.ADJ_RECIPROCITY_TOL:.0e}):",
          "PASS" if recip_pass else "FAIL")
    print(f"mass balance max residual {max(mass_residuals):.2e} (< {config.ADJ_MASS_BALANCE_TOL:.0e}):",
          "PASS" if mass_pass else "FAIL")
    print(f"adjoint solve median {timing['median_s'] * 1e3:.2f} ms, p99 {timing['p99_s'] * 1e3:.2f} ms "
          f"(< {config.ADJ_SOLVE_TIME_TARGET_S * 1e3:.0f} ms):", "PASS" if timing_pass else "FAIL")
    for s in fields:
        print(f"source {s}: mass fraction within {config.ADJ_VALIDATE_RADIUS_M:.0f} m = {fractions[s]:.3f}")


if __name__ == "__main__":
    main()