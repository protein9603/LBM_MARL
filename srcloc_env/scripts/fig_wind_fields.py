"""Figure: the three wind fields the estimator may be given (W0 uniform, W1 potential flow around the buildings, W2 CFD) at the drone
height, over the building footprint, with the 13 sources (D13 / D14 talk figure).

Usage: python -m srcloc_env.scripts.fig_wind_fields [--out 분석그림/icrs15/fig_wind_fields.png] [--dpi 200] [--step 6]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.field.wind_models import build_wind_field
from srcloc_env.preprocess.gridder import SlabGrid

TITLES = {"W0": "W0: one reference wind (1.68 m/s, +x)", "W1": "W1: reference wind + building map (potential flow)", "W2": "W2: CFD wind field (reference)"}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=config.FIG_DIR / "fig_wind_fields.png")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--step", type=int, default=6, help="quiver thinning in slab cells (5 m)")
    args = ap.parse_args(argv)
    om = ObstacleMap.load()
    grid = SlabGrid()
    xx, yy = np.meshgrid(grid.x_centres, grid.y_centres)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    blocked = om.no_fly_mask(config.DRONE_Z, margin=0.0)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), constrained_layout=True, sharey=True)
    vmax = 4.0
    for ax, lv in zip(axes, ("W0", "W1", "W2")):
        wf, _ = build_wind_field(lv, om)
        uv = wf.uv_at(pts, config.DRONE_Z).reshape(grid.ny, grid.nx, 2)
        speed = np.hypot(uv[..., 0], uv[..., 1])
        im = ax.imshow(speed, origin="lower", extent=(grid.x0, grid.x0 + grid.nx * grid.res, grid.y0, grid.y0 + grid.ny * grid.res),
                       cmap="Blues", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~blocked.T, blocked.T), origin="lower", extent=(om.x0, om.x0 + om.nx * om.res, om.y0, om.y0 + om.ny * om.res),
                  cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        s = args.step
        ax.quiver(xx[::s, ::s], yy[::s, ::s], uv[::s, ::s, 0], uv[::s, ::s, 1], color="#0b0b0b", scale=60, width=0.0022, alpha=0.8)
        for sid, (sx, sy) in config.SOURCES_XY.items():
            ax.plot(sx, sy, marker="*", markersize=9, color="#eb6834", markeredgecolor="k", markeredgewidth=0.4, zorder=5)
        ax.set_title(TITLES[lv], fontsize=10)
        ax.set_xlim(grid.x0, grid.x0 + grid.nx * grid.res); ax.set_ylim(grid.y0, grid.y0 + grid.ny * grid.res)
        ax.set_xlabel("x [m]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y [m]")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01); cb.set_label("wind speed at 15 m [m/s]")
    fig.suptitle("Wind knowledge given to the estimator (grey = buildings taller than 15 m, stars = the 13 sources)", fontsize=11)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white"); plt.close(fig)
    print(f"[fig_wind_fields] -> {args.out}")
    return {"png": str(args.out)}


if __name__ == "__main__":
    main()
