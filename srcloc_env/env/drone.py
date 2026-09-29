"""Obstacle map at flight altitude and discrete drone kinematics with action masking (plan 4.5, D2 task B).

Data
----
``occupancy_2m_flowframe.npz`` (config.OCCUPANCY_2M_NPZ, report 4.3) was rasterised from Leipzig_buildings.stl
by ``stl_tools.rasterize`` after the STL -> fluid/LDM shift (+689.26, -11.48, 0) m (report 4.2). Keys:
``occ`` bool (659, 659), ``hmap`` float32 (659, 659) roof-top height [m] with 0.0 where there is no building
(occ == (hmap > 0) exactly on all 51,471 occupied cells; config.OCC_HMAP_NO_BUILDING), ``x`` (659,) = 0..1316,
``y`` (659,) = -657.5..658.5, ``meta`` JSON string {res 2, x0 0, y0 -657.5, nx 659, ny 659}, ``shift`` (2,).
Report 4.3 quotes 659 x 658; the stored file is 659 x 659 (one extra, empty y row at 658.5 m).

Axis order (resolved from the data, see scripts/validate_drone.py; config.OCC_AXIS_ORDER)
------------------------------------------------------------------------------------------
``occ[ix, iy]`` / ``hmap[ix, iy]`` with cell centre x = x0 + ix*res, y = y0 + iy*res; cell (ix, iy) covers
[centre - res/2, centre + res/2), i.e. index = floor((p - p0)/res + 0.5) (stl_tools convention, report 4.3).
Evidence: read as [ix, iy] the occupied cells span x 200-1116, y -453.5..452.5, matching the p_type 1000
building bbox x 200.0-1116.75, y +-452.5 (report 4.2); read as [iy, ix] they would span y -457.5.. and
x ..1110. The 115 m tower (report 4.1) is found at (552, 300.5), 40 m from source 110 (525.9, 270.5), and the
69 m tower 50 m from source 106 (369.8, -89.7) only under the [ix, iy] reading (the transposed reading gives
21 m / 26 m maxima there). It is the same [ix, iy] orientation as the wind lattice (config.WIND_AXIS_ORDER).
All arrays of this module keep that orientation: masks are (nx, ny), indexed [ix, iy].

No-fly mask (plan 4.5, report 4.3)
----------------------------------
At flight altitude z a cell is blocked when its roof satisfies hmap >= z - margin (the drone must clear the
roof by ``margin``); the blocked set is then dilated by ceil(margin / res) cells with a full 3 x 3 structuring
element (8-connected: one complete ring per iteration, a conservative Chebyshev margin). Report 4.3 recommends
a 1-2 m margin because LDM particles sit within 2 m of the walls and the 2 m raster is itself conservative at
boundary cells (98.3 % agreement with the exact triangle test, 0 false negatives). Points outside
config.DOMAIN_X / DOMAIN_Y (or outside the raster) are never free.

Ray distances
-------------
``ray_distances`` marches along each heading in steps of ``res`` and returns the distance to the first sample
that is not free (blocked cell, raster edge or domain edge), capped at ``max_range`` (plan 4.5 observation
"8-direction building distance / 100 m"). The resolution is therefore one cell (res).

Drone kinematics (plan 4.5)
---------------------------
Nine discrete actions: 0..7 = the eight headings E, NE, N, NW, W, SW, S, SE (counter-clockwise from +x; every
move is exactly ``step_m`` long, diagonals included), 8 = stay. A move is allowed iff its landing point is
free; stay is always allowed. ``step`` returns the new position and an ``applied`` flag; a masked action leaves
the drone in place (plan 4.5: when every direction is blocked the drone stays).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from srcloc_env import config

HEADING_NAMES: tuple[str, ...] = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")   # action index 0..7
ACTION_STAY: int = config.DRONE_N_HEADINGS                                       # action index 8


def heading_unit_vectors(n_dirs: int = config.DRONE_N_HEADINGS) -> np.ndarray:
    """(n_dirs, 2) unit vectors at angles k * 2 pi / n_dirs counter-clockwise from +x (k = 0 -> E, 2 -> N)."""
    ang = 2.0 * np.pi * np.arange(int(n_dirs)) / float(n_dirs)
    vec = np.column_stack([np.cos(ang), np.sin(ang)])
    vec[np.abs(vec) < 1e-12] = 0.0
    return vec


class ObstacleMap:
    """2 m building occupancy + roof-height raster in the fluid/LDM frame, indexed [ix, iy] (see module doc)."""

    def __init__(self, x0: float, y0: float, res: float, occ: np.ndarray, hmap: np.ndarray,
                 domain_x: tuple[float, float] = config.DOMAIN_X,
                 domain_y: tuple[float, float] = config.DOMAIN_Y) -> None:
        self.x0, self.y0, self.res = float(x0), float(y0), float(res)
        if self.res <= 0.0:
            raise ValueError("res must be positive")
        self.occ = np.ascontiguousarray(occ, dtype=bool)
        self.hmap = np.ascontiguousarray(hmap, dtype=np.float32)
        if self.occ.ndim != 2 or self.hmap.shape != self.occ.shape:
            raise ValueError(f"occ {self.occ.shape} and hmap {self.hmap.shape} must be equal 2-D (nx, ny) arrays")
        self.nx, self.ny = (int(v) for v in self.occ.shape)
        self.domain_x = (float(domain_x[0]), float(domain_x[1]))
        self.domain_y = (float(domain_y[0]), float(domain_y[1]))
        self._cache: dict[tuple[float, float], np.ndarray] = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, path: Path = config.OCCUPANCY_2M_NPZ, domain_x: tuple[float, float] = config.DOMAIN_X,
             domain_y: tuple[float, float] = config.DOMAIN_Y) -> "ObstacleMap":
        """Load occupancy_2m_flowframe.npz (keys occ, hmap, x, y, meta JSON, shift); verifies the [ix, iy] layout."""
        with np.load(path) as z:
            occ = np.asarray(z["occ"])
            hmap = np.asarray(z["hmap"])
            meta = json.loads(str(z["meta"]))
            x = np.asarray(z["x"], dtype=np.float64) if "x" in z.files else None
            y = np.asarray(z["y"], dtype=np.float64) if "y" in z.files else None
        res, x0, y0 = float(meta["res"]), float(meta["x0"]), float(meta["y0"])
        if occ.shape != (int(meta["nx"]), int(meta["ny"])):
            raise ValueError(f"{path}: occ shape {occ.shape} != meta (nx, ny) = ({meta['nx']}, {meta['ny']}); "
                             f"expected axis order {config.OCC_AXIS_ORDER}")
        if x is not None and (x.size != occ.shape[0] or not np.allclose(x, x0 + res * np.arange(occ.shape[0]))):
            raise ValueError(f"{path}: key x does not match meta x0/res along axis 0 (expected occ[ix, iy])")
        if y is not None and (y.size != occ.shape[1] or not np.allclose(y, y0 + res * np.arange(occ.shape[1]))):
            raise ValueError(f"{path}: key y does not match meta y0/res along axis 1 (expected occ[ix, iy])")
        return cls(x0, y0, res, occ, hmap, domain_x, domain_y)

    @classmethod
    def from_arrays(cls, x0: float, y0: float, res: float, occ: np.ndarray, hmap: np.ndarray,
                    domain_x: tuple[float, float] = config.DOMAIN_X,
                    domain_y: tuple[float, float] = config.DOMAIN_Y) -> "ObstacleMap":
        """Build from in-memory arrays (tests). occ/hmap are (nx, ny) with x = x0 + ix*res, y = y0 + iy*res."""
        return cls(x0, y0, res, occ, hmap, domain_x, domain_y)

    # ------------------------------------------------------------------ geometry helpers
    @property
    def x(self) -> np.ndarray:
        """Cell-centre x coordinates (nx,)."""
        return self.x0 + self.res * np.arange(self.nx)

    @property
    def y(self) -> np.ndarray:
        """Cell-centre y coordinates (ny,)."""
        return self.y0 + self.res * np.arange(self.ny)

    def cell_index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(ix, iy) of the cells containing the points (..., 2); stl_tools convention floor((p - p0)/res + 0.5)."""
        xy = np.asarray(xy, dtype=np.float64)
        ix = np.floor((xy[..., 0] - self.x0) / self.res + 0.5).astype(np.int64)
        iy = np.floor((xy[..., 1] - self.y0) / self.res + 0.5).astype(np.int64)
        return ix, iy

    def in_domain(self, xy: np.ndarray) -> np.ndarray:
        """True where the point lies inside the closed box domain_x x domain_y (config.DOMAIN_X/Y by default)."""
        xy = np.asarray(xy, dtype=np.float64)
        return ((xy[..., 0] >= self.domain_x[0]) & (xy[..., 0] <= self.domain_x[1])
                & (xy[..., 1] >= self.domain_y[0]) & (xy[..., 1] <= self.domain_y[1]))

    # ------------------------------------------------------------------ queries
    def no_fly_mask(self, z: float, margin: float = config.BUILDING_MARGIN_M) -> np.ndarray:
        """Blocked cells (nx, ny) at altitude z: occ & (hmap >= z - margin), dilated by ceil(margin/res) cells.

        The dilation uses a full 3 x 3 structuring element (8-connected ring per iteration). The result is
        cached per (z, margin) and returned read-only.
        """
        key = (float(z), float(margin))
        m = self._cache.get(key)
        if m is None:
            blocked = self.occ & (self.hmap >= np.float32(key[0] - key[1]))
            n_iter = int(np.ceil(key[1] / self.res))
            if n_iter > 0:                      # scipy treats iterations < 1 as "until convergence": guard it
                blocked = binary_dilation(blocked, structure=np.ones((3, 3), dtype=bool), iterations=n_iter)
            m = np.ascontiguousarray(blocked, dtype=bool)
            m.setflags(write=False)
            self._cache[key] = m
        return m

    def is_free(self, xy: np.ndarray, z: float, margin: float = config.BUILDING_MARGIN_M) -> np.ndarray:
        """bool (n,) for xy (n, 2) (or a bool scalar for xy (2,)): inside the domain, inside the raster, not no-fly."""
        xy = np.asarray(xy, dtype=np.float64)
        squeeze = xy.ndim == 1
        xy = np.atleast_2d(xy)
        ix, iy = self.cell_index(xy)
        ok = self.in_domain(xy) & (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        free = np.zeros(xy.shape[0], dtype=bool)
        if np.any(ok):
            free[ok] = ~self.no_fly_mask(z, margin)[ix[ok], iy[ok]]
        return bool(free[0]) if squeeze else free

    def ray_distances(self, xy: np.ndarray, z: float, n_dirs: int = config.DRONE_N_HEADINGS,
                      max_range: float = config.RAY_MAX_RANGE_M,
                      margin: float = config.BUILDING_MARGIN_M) -> np.ndarray:
        """(n, n_dirs) distance [m] along headings k * 360/n_dirs deg CCW from +x (8: E, NE, N, NW, W, SW, S, SE)
        to the first sample (step res) that is not free: blocked cell, raster edge or domain edge. Capped at
        max_range; a ray that never hits anything within max_range returns max_range. A point that is itself
        not free still reports the distance to the first not-free *sample* (>= res)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        n = xy.shape[0]
        dirs = heading_unit_vectors(n_dirs)                                  # (d, 2)
        n_steps = int(np.ceil(float(max_range) / self.res))
        steps = self.res * np.arange(1, n_steps + 1, dtype=np.float64)       # (K,)
        pts = xy[:, None, None, :] + dirs[None, :, None, :] * steps[None, None, :, None]   # (n, d, K, 2)
        free = self.is_free(pts.reshape(-1, 2), z, margin).reshape(n, int(n_dirs), n_steps)
        blocked = ~free
        hit = blocked.any(axis=2)
        first = blocked.argmax(axis=2)
        dist = np.where(hit, steps[first], float(max_range))
        return np.minimum(dist, float(max_range))


class DroneKinematics:
    """Discrete 9-action drone motion at fixed altitude with building/domain action masking (plan 4.5)."""

    def __init__(self, obstacles: ObstacleMap, step_m: float = config.DRONE_STEP_M, z: float = config.DRONE_Z,
                 margin: float = config.BUILDING_MARGIN_M) -> None:
        self.obstacles = obstacles
        self.step_m = float(step_m)
        self.z = float(z)
        self.margin = float(margin)
        self.n_actions = int(config.DRONE_N_ACTIONS)
        self.displacements = np.vstack([self.step_m * heading_unit_vectors(config.DRONE_N_HEADINGS),
                                        np.zeros((1, 2))])                   # (9, 2); row 8 = stay

    def landing_points(self, xy: np.ndarray) -> np.ndarray:
        """(n, 9, 2) positions reached from xy (n, 2) by each action (stay -> xy itself)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        return xy[:, None, :] + self.displacements[None, :, :]

    def action_mask(self, xy: np.ndarray) -> np.ndarray:
        """bool (9,) for xy (2,) or (n, 9) for xy (n, 2): True where the action is allowed.

        A move is allowed iff its landing point is free (ObstacleMap.is_free at self.z); stay is always allowed.
        """
        xy = np.asarray(xy, dtype=np.float64)
        squeeze = xy.ndim == 1
        land = self.landing_points(xy)
        n = land.shape[0]
        free = self.obstacles.is_free(land[:, :ACTION_STAY].reshape(-1, 2), self.z, self.margin)
        mask = np.concatenate([free.reshape(n, ACTION_STAY), np.ones((n, 1), dtype=bool)], axis=1)
        return mask[0] if squeeze else mask

    def step(self, xy: np.ndarray, action: int | np.ndarray) -> tuple[np.ndarray, np.ndarray | bool]:
        """Apply action(s) to xy. Returns (new_xy, applied): a masked action leaves the drone unchanged with
        applied False; stay is always applied. Shapes follow xy: (2,) -> ((2,), bool), (n, 2) -> ((n, 2), (n,))."""
        xy = np.asarray(xy, dtype=np.float64)
        squeeze = xy.ndim == 1
        xy2 = np.atleast_2d(xy)
        n = xy2.shape[0]
        act = np.broadcast_to(np.asarray(action, dtype=np.int64), (n,))
        if np.any((act < 0) | (act >= self.n_actions)):
            raise ValueError(f"actions must be in 0..{self.n_actions - 1}")
        mask = np.atleast_2d(self.action_mask(xy2))
        applied = mask[np.arange(n), act]
        new = np.where(applied[:, None], xy2 + self.displacements[act], xy2)
        if squeeze:
            return new[0], bool(applied[0])
        return new, applied