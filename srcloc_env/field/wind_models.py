"""Wind fields that need NO detailed CFD solution (wind-knowledge levels W0 / W1, plan D13, 2026-10-03).

Motivation (professor's remark 2026-10-03): feeding the resolved urban wind field into the particle filter assumes
the field is known, but one CFD/SPH run takes days, so it cannot be produced in the field. These two models use only
information a responder could have at once: ONE reference wind (speed U, direction) and, for W1, the building map
(GIS / STL raster, already in ObstacleMap).

W0  ``uniform_wind_field``    the same (U cos a, U sin a, 0) everywhere; paired with the global Gaussian plume
                              (pf/forward_model.GaussianPlume, wind_mode='global') in Scene.
W1  ``potential_flow_field``  mass-consistent uniform flow around the buildings: a 2-D potential flow phi on the free
                              cells of the adjoint SlabGrid (Laplace equation, no-penetration at blocked cells,
                              far-field phi = U (x cos a + y sin a) on the outer boundary), velocity = grad phi.
                              It is the simplest member of the diagnostic (Rockle / Kaplan-Dinar / QUIC-URB /
                              QES-Winds / URock) family: divergence free, channels the flow around and between
                              buildings, no wakes or recirculation; one sparse solve on ~41 k cells (< 1 s).
                              The resulting WindField (one level at config.DRONE_Z, lattice = the slab cell centres)
                              feeds the SAME AdvectionDiffusionOperator as the CFD wind, so W1 and W2 differ only in
                              the wind array.
W2  the CFD wind (field/wind.WindField.load) - the reference, not built here.

Both functions return ``field.wind.WindField`` objects so that the environment observation (wind at the drone) and
``AdvectionDiffusionOperator.wind_at_cells`` work unchanged.  ``potential_flow_diagnostics`` reports the discrete
divergence, the wall-normal velocity at blocked faces and the far-field speed (tests, validate_wind_levels).
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import label
from scipy.sparse.linalg import spsolve

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.field.wind import WindField
from srcloc_env.pf.lbm_adjoint import AdvectionDiffusionOperator
from srcloc_env.preprocess.gridder import SlabGrid


def _unit(dir_deg: float) -> np.ndarray:
    a = np.deg2rad(float(dir_deg))
    return np.array([np.cos(a), np.sin(a)], dtype=np.float64)


def uniform_wind_field(U: float = config.WIND_MEAN_U, dir_deg: float = config.WIND_MEAN_DIR_DEG,
                       z: float = config.DRONE_Z, x_range: tuple[float, float] = config.DOMAIN_X,
                       y_range: tuple[float, float] = config.DOMAIN_Y) -> WindField:
    """W0: a 2 x 2 lattice spanning the domain with the constant horizontal wind (U cos a, U sin a) at one level z."""
    if U < 0.0:
        raise ValueError("U must be >= 0")
    e = _unit(dir_deg) * float(U)
    uvw = np.zeros((1, 2, 2, 3), dtype=np.float32)
    uvw[..., 0], uvw[..., 1] = e[0], e[1]
    return WindField.from_arrays(np.array(x_range, float), np.array(y_range, float), np.array([float(z)]), uvw)


def _laplace_system(free: np.ndarray, res: float, e: np.ndarray, phi_inf: np.ndarray):
    """Sparse Laplace system on the free cells: (A, rhs, idx) with Dirichlet far field on the outer boundary and
    no-flux (natural Neumann) faces towards blocked cells."""
    ny, nx = free.shape
    idx = np.full((ny, nx), -1, dtype=np.intp)
    n = int(free.sum())
    idx[free] = np.arange(n)
    iy, ix = np.nonzero(free)
    me = idx[iy, ix]
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    diag = np.zeros(n)
    for diy, dix in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        jy, jx = iy + diy, ix + dix
        inside = (jy >= 0) & (jy < ny) & (jx >= 0) & (jx < nx)
        nb_free = np.zeros_like(inside)
        nb_free[inside] = free[jy[inside], jx[inside]]
        sel = inside & nb_free                                        # interior free neighbour
        rows.append(me[sel]); cols.append(idx[jy[sel], jx[sel]]); vals.append(-np.ones(int(sel.sum())))
        np.add.at(diag, me[sel], 1.0)
        out = ~inside                                                 # outer boundary: ghost cell at the far-field potential
        g = phi_inf[iy[out], ix[out]] + res * (e[0] * dix + e[1] * diy)
        np.add.at(diag, me[out], 1.0)
        np.add.at(rhs, me[out], g)
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return A, rhs, idx


def _gradient(phi: np.ndarray, free: np.ndarray, phi_inf: np.ndarray, res: float, e: np.ndarray, axis: int) -> np.ndarray:
    """d phi / d(axis) at the free cells (axis 1 = x, 0 = y): central where both neighbours are free, far-field ghost
    on the outer boundary, one-sided (mean of the zero wall flux and the free face) next to a blocked cell, 0 when both
    neighbours are blocked; blocked cells get 0."""
    ny, nx = free.shape
    p = np.pad(phi, 1, mode="constant", constant_values=np.nan)
    f = np.pad(free, 1, mode="constant", constant_values=False)
    inf = np.pad(phi_inf, 1, mode="edge")
    if axis == 1:
        plus, minus, fplus, fminus = p[1:-1, 2:], p[1:-1, :-2], f[1:-1, 2:], f[1:-1, :-2]
        gplus, gminus = inf[1:-1, 2:] + e[0] * res, inf[1:-1, :-2] - e[0] * res
        edge_p = np.zeros((ny, nx), bool); edge_p[:, -1] = True
        edge_m = np.zeros((ny, nx), bool); edge_m[:, 0] = True
    else:
        plus, minus, fplus, fminus = p[2:, 1:-1], p[:-2, 1:-1], f[2:, 1:-1], f[:-2, 1:-1]
        gplus, gminus = inf[2:, 1:-1] + e[1] * res, inf[:-2, 1:-1] - e[1] * res
        edge_p = np.zeros((ny, nx), bool); edge_p[-1, :] = True
        edge_m = np.zeros((ny, nx), bool); edge_m[0, :] = True
    vp = np.where(fplus, plus, np.where(edge_p, gplus, np.nan))
    vm = np.where(fminus, minus, np.where(edge_m, gminus, np.nan))
    okp, okm = ~np.isnan(vp), ~np.isnan(vm)
    g = np.zeros((ny, nx))
    both = okp & okm
    g[both] = (vp[both] - vm[both]) / (2.0 * res)
    only_p = okp & ~okm
    g[only_p] = 0.5 * (vp[only_p] - phi[only_p]) / res
    only_m = okm & ~okp
    g[only_m] = 0.5 * (phi[only_m] - vm[only_m]) / res
    g[~free] = 0.0
    return np.nan_to_num(g)


def potential_flow_uv(blocked: np.ndarray, res: float, U: float, dir_deg: float,
                      x_centres: np.ndarray, y_centres: np.ndarray) -> np.ndarray:
    """(ny, nx, 2) velocity of the mass-consistent uniform flow around the blocked cells (W1).

    Cell-centred finite volumes: phi solves sum over free face neighbours of (phi_nb - phi_c) = 0 (Laplace); a face to
    a blocked cell carries no flux (no-penetration, natural Neumann); a face on the outer grid boundary uses the
    far-field potential phi_inf = U (x cos a + y sin a) at the ghost cell centre (Dirichlet far field).  Velocity =
    grad phi (``_gradient``); blocked cells carry (0, 0); enclosed free components (courtyards) are calm (0, 0).
    """
    blocked = np.asarray(blocked, dtype=bool)
    free = ~blocked
    if not free.any():
        raise ValueError("every cell is blocked")
    e = _unit(dir_deg) * float(U)
    xx, yy = np.meshgrid(np.asarray(x_centres, float), np.asarray(y_centres, float))      # (ny, nx)
    phi_inf = e[0] * xx + e[1] * yy
    # free cells enclosed by buildings (courtyards: a 4-connected component that never touches the outer boundary)
    # have no far-field condition and a pure-Neumann (singular) subproblem: they are calm, phi = const, velocity 0.
    comp, n_comp = label(free)
    touching = np.unique(np.concatenate([comp[0, :], comp[-1, :], comp[:, 0], comp[:, -1]]))
    active = free & np.isin(comp, touching[touching > 0])
    if not active.any():
        raise ValueError("no free cell is connected to the outer boundary")
    A, rhs, idx = _laplace_system(active, float(res), e, phi_inf)
    phi = np.full(free.shape, np.nan)
    phi[free] = 0.0
    phi[active] = spsolve(A.tocsc(), rhs)
    uv = np.zeros(free.shape + (2,))
    uv[..., 0] = _gradient(phi, free, phi_inf, float(res), e, 1)
    uv[..., 1] = _gradient(phi, free, phi_inf, float(res), e, 0)
    uv[blocked] = 0.0
    return uv


def potential_flow_field(obstacles: ObstacleMap, grid: SlabGrid = SlabGrid(), U: float = config.WIND_MEAN_U,
                         dir_deg: float = config.WIND_MEAN_DIR_DEG, z: float = config.DRONE_Z) -> WindField:
    """W1: the potential-flow wind on the slab grid as a one-level WindField (lattice = cell centres; blocked cells
    are dead nodes so that ``uv_at`` never interpolates across a building); speeds above
    config.WIND_POTENTIAL_SPEED_CAP * U (corner singularity) are scaled down to that cap."""
    blocked = AdvectionDiffusionOperator.blocked_at_cells(obstacles, grid, z)             # (ny, nx)
    uv = potential_flow_uv(blocked, grid.res, U, dir_deg, grid.x_centres, grid.y_centres)
    cap = config.WIND_POTENTIAL_SPEED_CAP * float(U)                                        # corner singularity of the potential flow
    speed = np.hypot(uv[..., 0], uv[..., 1])
    fast = speed > cap
    if cap > 0 and fast.any():
        uv[fast] *= (cap / speed[fast])[:, None]
    uvw = np.zeros((1, grid.nx, grid.ny, 3), dtype=np.float32)                            # [level, ix, iy, comp]
    uvw[0, :, :, 0] = uv[..., 0].T
    uvw[0, :, :, 1] = uv[..., 1].T
    mask = np.ascontiguousarray(blocked.T)
    height = np.where(mask, np.inf, 0.0).astype(np.float32)
    return WindField.from_arrays(grid.x_centres, grid.y_centres, np.array([float(z)]), uvw, mask, height)


def potential_flow_diagnostics(uv: np.ndarray, blocked: np.ndarray, res: float, U: float, dir_deg: float) -> dict:
    """Discrete checks of a W1 field: cell divergence (free cells whose four neighbours are free), the velocity
    component normal to blocked faces (should be small), the mean speed on the inflow edge and speed statistics."""
    blocked = np.asarray(blocked, bool)
    free = ~blocked
    u, v = uv[..., 0], uv[..., 1]
    inner = np.zeros_like(free)
    inner[1:-1, 1:-1] = (free[1:-1, 1:-1] & free[1:-1, 2:] & free[1:-1, :-2] & free[2:, 1:-1] & free[:-2, 1:-1])
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2] + v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * res)
    speed = np.hypot(u, v)
    wall_x = free & (np.roll(blocked, 1, axis=1) | np.roll(blocked, -1, axis=1))
    wall_y = free & (np.roll(blocked, 1, axis=0) | np.roll(blocked, -1, axis=0))
    e = _unit(dir_deg)
    col = 0 if e[0] >= 0 else -1
    return {"div_rms": float(np.sqrt(np.mean(div[inner] ** 2))) if inner.any() else 0.0,
            "div_max": float(np.abs(div[inner]).max()) if inner.any() else 0.0,
            "div_scale": float(U / res),
            "wall_normal_median_u": float(np.median(np.abs(u[wall_x]))) if wall_x.any() else 0.0,
            "wall_normal_median_v": float(np.median(np.abs(v[wall_y]))) if wall_y.any() else 0.0,
            "inflow_edge_mean_speed": float(np.mean(speed[free[:, col], col])) if free[:, col].any() else float("nan"),
            "speed_mean_free": float(speed[free].mean()), "speed_max_free": float(speed[free].max()),
            "speed_p95_free": float(np.percentile(speed[free], 95)), "n_free": int(free.sum()), "n_blocked": int(blocked.sum())}


def build_wind_field(level: str, obstacles: ObstacleMap | None = None, grid: SlabGrid = SlabGrid(),
                     U: float = config.WIND_MEAN_U, dir_deg: float = config.WIND_MEAN_DIR_DEG) -> tuple[WindField, float]:
    """(WindField, seconds) of a wind-knowledge level: W0 uniform, W1 potential flow (needs obstacles), W2 CFD."""
    t0 = time.perf_counter()
    if level == "W0":
        wf = uniform_wind_field(U, dir_deg)
    elif level == "W1":
        wf = potential_flow_field(obstacles if obstacles is not None else ObstacleMap.load(), grid, U, dir_deg)
    elif level == "W2":
        wf = WindField.load()
    else:
        raise ValueError(f"wind level must be one of {config.WIND_LEVELS}, got {level!r}")
    return wf, time.perf_counter() - t0
