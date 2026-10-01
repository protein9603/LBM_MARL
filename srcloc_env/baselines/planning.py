"""Geodesic (around-the-buildings) distance field for target-following baseline policies (D9-4).

Greedy "move towards the target" policies (greedy-MAP, the verification oracle) are trapped by the first building between
the drone and the target (a local minimum of the Euclidean distance).  GeodesicField builds a coarse free-space graph of
the 15 m no-fly mask (a coarse cell is blocked when any of its fine cells is blocked, or it lies outside the domain;
blocked_fraction raises the tolerated fraction), runs Dijkstra from the target cell and returns the shortest-path length field; the policy then takes the allowed
action whose landing point has the smallest geodesic value (a gradient descent without local minima).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap


class GeodesicField:
    """Shortest-path length [m] to a target over the free space of ``obstacles`` at height z (coarse ``block``-cell grid)."""

    def __init__(self, obstacles: ObstacleMap, z: float = config.DRONE_Z, block: int = 2,
                 margin: float = config.BUILDING_MARGIN_M, blocked_fraction: float = 0.0) -> None:
        om = obstacles
        mask = om.no_fly_mask(z, margin)                                  # (nx, ny), True = blocked
        nx, ny = mask.shape
        b = int(block)
        cx, cy = -(-nx // b), -(-ny // b)
        pad = np.ones((cx * b, cy * b), dtype=bool)                        # outside the raster = blocked
        pad[:nx, :ny] = mask
        frac = pad.reshape(cx, b, cy, b).mean(axis=(1, 3))
        self.res = om.res * b
        self.x0 = om.x0 + 0.5 * (b - 1) * om.res                          # centre of coarse cell (0, 0)
        self.y0 = om.y0 + 0.5 * (b - 1) * om.res
        self.cx, self.cy = cx, cy
        xs = self.x0 + self.res * np.arange(cx)
        ys = self.y0 + self.res * np.arange(cy)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        inside = om.in_domain(np.column_stack([gx.ravel(), gy.ravel()])).reshape(cx, cy)
        self.free = (frac <= blocked_fraction) & inside                 # default: a coarse cell is free only if ALL its fine cells are free,
                                                                        # so a path through free coarse cells never lands in a masked cell
        idx = -np.ones((cx, cy), dtype=np.int64)
        idx[self.free] = np.arange(int(self.free.sum()))
        self._idx = idx
        rows, cols, w = [], [], []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            xa, xb = slice(max(0, -dx), cx - max(0, dx)), slice(max(0, dx), cx - max(0, -dx))
            ya, yb = slice(max(0, -dy), cy - max(0, dy)), slice(max(0, dy), cy - max(0, -dy))
            ok = self.free[xa, ya] & self.free[xb, yb]
            if dx != 0 and dy != 0:                                       # no corner cutting through a blocked cell
                ok &= self.free[xb, ya] & self.free[xa, yb]
            ia, ib = idx[xa, ya][ok], idx[xb, yb][ok]
            rows += [ia, ib]; cols += [ib, ia]
            w += [np.full(ia.size, self.res * np.hypot(dx, dy))] * 2
        n = int(self.free.sum())
        self.graph = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        fx, fy = np.nonzero(self.free)
        self._free_xy = np.column_stack([xs[fx], ys[fy]])
        self._tree = cKDTree(self._free_xy)
        self._free_ij = np.column_stack([fx, fy])

    def field(self, target_xy) -> np.ndarray:
        """(cx, cy) geodesic length to the free coarse cell nearest to target_xy; blocked cells get the length of their best
        free neighbour + one cell (so a drone standing in a conservatively blocked coarse cell still sees a gradient)."""
        _, k = self._tree.query(np.asarray(target_xy, dtype=float))
        d = dijkstra(self.graph, directed=False, indices=int(k))
        f = np.full((self.cx, self.cy), np.inf)
        f[self.free] = d
        for _ in range(2):                                                # fill blocked cells from finite neighbours
            pad = np.pad(f, 1, constant_values=np.inf)
            nb = np.min(np.stack([pad[1 + dx:1 + dx + self.cx, 1 + dy:1 + dy + self.cy]
                                  for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]), axis=0)
            f = np.where(np.isfinite(f), f, nb + self.res)
        return f

    def value(self, field: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Field value at points xy (n, 2) (nearest coarse cell; +Euclid tail outside the grid)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        ix = np.clip(np.rint((xy[:, 0] - self.x0) / self.res).astype(int), 0, self.cx - 1)
        iy = np.clip(np.rint((xy[:, 1] - self.y0) / self.res).astype(int), 0, self.cy - 1)
        return field[ix, iy]
