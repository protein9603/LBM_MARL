"""Figure 7 candidate: top view of one evaluation episode (plan S2 산출물 '그림 7 후보', S4 그림 7; D9-4).

Usage: python -m srcloc_env.scripts.fig_episode --npz <steps/..._epNNNN.npz> [--out <png>] [--fractions 0.1 0.3 0.6 1.0]
Reads a step log written by eval/run_eval.py --log-steps and draws one panel per time fraction: building outlines
(ObstacleMap occupancy), the 15 m LDM slab density of the episode's source at the episode's frame (log10, 4 decades;
the frozen Mode F snapshot the drones sampled), the drone paths up to that step (points coloured by the normalised log
count), the 2-sigma ellipses of the valid GMM components of the belief at that step, the MAP estimate and the true
source.  The same step logs feed the 3-D video (docs/training_evaluation_spec.md section 11).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.sensor.detector import Detector


def load_step_log(path: Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    arrs = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(str(z["meta"]))
    return arrs, meta


def _ellipse(ax, mu, cov, weight, colour):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * 2 * np.sqrt(vals[1]), 2 * 2 * np.sqrt(vals[0]), angle=ang, fill=False,
                         edgecolor=colour, linewidth=1.0 + 2.0 * weight, alpha=0.55 + 0.45 * weight))


def render(npz: Path, out: Path, fractions=(0.1, 0.3, 0.6, 1.0), margin_m: float = 90.0, backend: LdmSlabBackend | None = None,
           om: ObstacleMap | None = None) -> dict:
    log, meta = load_step_log(npz)
    T = int(log["drone_xy"].shape[0])
    D = int(log["drone_xy"].shape[1])
    truth = np.asarray(log["truth_xy"], dtype=float)
    src, frame = int(meta["source"]), int(meta["frame"])
    be = backend if backend is not None else LdmSlabBackend()
    om = om if om is not None else ObstacleMap.load()
    mode = str(meta.get("mode", "F"))
    time_varying = mode == "T2"                                       # Mode T2: the truth field of step t is frame0 + t (1 frame per step)

    def frame_of(t: int) -> int:
        if "frame" in log:
            return int(log["frame"][t - 1])
        return int(min(config.N_FILES - 1, frame + t - 1)) if time_varying else frame

    def dens_of(fr: int) -> np.ndarray:
        sf_ = be.slab(fr)
        return sf_.density[list(sf_.sources).index(src), sf_.z_index(config.DRONE_Z)].astype(float)            # (ny, nx)

    sf = be.slab(frame)
    g = sf.grid
    pts = log["drone_xy"].reshape(-1, 2)
    lo = np.minimum(pts.min(axis=0), truth) - margin_m
    hi = np.maximum(pts.max(axis=0), truth) + margin_m
    for a, b in ((1, 0), (0, 1)):                                    # keep the window at least 0.55 : 1 so thin paths do not give flat panels
        span, other = hi[a] - lo[a], hi[b] - lo[b]
        if span < 0.55 * other:
            mid = 0.5 * (hi[a] + lo[a])
            lo[a], hi[a] = mid - 0.275 * other, mid + 0.275 * other
    det = Detector()
    yn = det.normalise(log["y"].reshape(-1)).reshape(T, D)
    steps = sorted({max(1, min(T, int(round(f * T)))) for f in fractions})
    dens_by_step = {t: dens_of(frame_of(t)) for t in steps}
    vmax = max(max(float(d_.max()) for d_ in dens_by_step.values()), 1e-6)
    fig, axes = plt.subplots(1, len(steps), figsize=(max(11.0, 4.6 * len(steps)), 5.2), constrained_layout=True, squeeze=False)
    cols = ["tab:blue", "tab:green", "tab:purple"]
    for ax, t in zip(axes[0], steps):
        im = ax.imshow(np.log10(np.maximum(dens_by_step[t], vmax * 1e-4)), origin="lower", extent=(g.x0, g.x0 + g.nx * g.res, g.y0, g.y0 + g.ny * g.res),
                       cmap="Greys", vmin=np.log10(vmax) - 4, vmax=np.log10(vmax), alpha=0.75, interpolation="nearest")
        ox, oy = np.meshgrid(om.x, om.y, indexing="ij")
        ax.contour(ox, oy, om.occ.astype(float), levels=[0.5], colors="0.35", linewidths=0.6)
        for d in range(D):
            xy = log["drone_xy"][:t, d]
            ax.plot(xy[:, 0], xy[:, 1], "-", color=cols[d % 3], linewidth=0.8, alpha=0.6)
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=yn[:t, d], s=6, cmap="autumn_r", vmin=0.0, vmax=1.0, zorder=3)
            ax.plot(*xy[-1], marker="o", markersize=6, markeredgecolor="k", color=cols[d % 3], zorder=4)
        w, mu, cov, mask = log["gmm_w"][t - 1], log["gmm_mu"][t - 1], log["gmm_cov"][t - 1], log["gmm_mask"][t - 1]
        for j in range(len(w)):
            if mask[j]:
                _ellipse(ax, mu[j], cov[j], float(w[j]), "tab:red")
        ax.plot(*log["map_xy"][t - 1], marker="x", color="tab:red", markersize=9, markeredgewidth=2, zorder=5)
        ax.plot(*truth, marker="*", color="gold", markeredgecolor="k", markersize=16, zorder=6)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_title(f"step {t}/{T}" + (f", frame {frame_of(t)}" if time_varying else "") + f": MAP error {log['map_error'][t - 1]:.0f} m, top sigma {log['top_sigma'][t - 1]:.0f} m", fontsize=9)
        ax.set_xlabel("x [m]")
    axes[0][0].set_ylabel("y [m]")
    cb = fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, pad=0.01)
    cb.set_label("normalised log count log(1+y)/log(1+y_max)")
    ok = "success" if meta["success"] else "no success"
    fig.suptitle(f"{meta['method']} ({D} drone{'s' if D > 1 else ''}), source {src}, " + (f"time-varying truth (Mode T2), start frame {frame}" if time_varying else f"frozen LDM frame {frame}") + f" "
                 f"(step {config.index_to_step(frame)}) [interpretation A], scale {float(meta['scale']):.2f}: {ok}\n"
                 f"grey = 15 m slab density of the source at the panel's step (log, 4 decades), red = 2-sigma GMM ellipses, x = MAP, star = truth", fontsize=9)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=config.FIG_DPI_FINAL)
    plt.close(fig)
    return {"out": str(out), "steps": steps, "T": T, "D": D, "source": src, "frame": frame, "success": bool(meta["success"])}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.3, 0.6, 1.0])
    args = ap.parse_args(argv)
    out = args.out if args.out is not None else config.FIG_DIR / f"fig7_{args.npz.stem}.png"
    res = render(args.npz, out, tuple(args.fractions))
    print(res)
    return res


if __name__ == "__main__":
    main()
