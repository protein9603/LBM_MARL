"""Full-coverage (boustrophedon) sweep plans for the lawnmower baseline (D9-4 revision; user decision 2026-10-01).

The sweep covers the whole PF prior box.  ``axis='cross'`` lays the rows PERPENDICULAR to the wind (the +x wind blows along
x, so a row is a transect along y at a fixed x): the detectable plume regions are long in x (median 230 m) and thin in y
(median 20 m), so cross-wind transects spaced 150 m (below the 25th percentile of the plume length, 160 m) cross almost every
plume, whereas rows along the wind would need a spacing of 20-50 m.  ``axis='along'`` (rows along x) is kept as the
classical alternative.

For ``n_drones`` > 1 the rows are split into contiguous groups (one per drone; sub-areas), the groups are assigned to the
drones so that the initial travel is smallest, and every drone sweeps only its own sub-area.  Within its group a drone starts
at the row nearest to its start, sweeps towards the side with more remaining rows, then returns to the rows it skipped.
Consecutive rows are traversed in opposite directions (boustrophedon).  Waypoints are spaced ``wp_spacing`` m along the rows;
waypoints inside no-fly cells are dropped (the geodesic follower flies around the buildings between the remaining ones).
"""
from __future__ import annotations

from math import ceil

import numpy as np

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap

Box = tuple[tuple[float, float], tuple[float, float]]


def make_rows(axis: str, spacing: float, box: Box) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rows of the sweep as (endpoint A, endpoint B): 'cross' = transects along y at equally spaced x (A south, B north),
    'along' = rows along x at equally spaced y (A west, B east).  n = ceil(width / spacing) rows, centred in equal strips."""
    (x0, x1), (y0, y1) = box
    if axis == "cross":
        n = max(1, ceil((x1 - x0) / spacing))
        pos = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
        return [(np.array([p, y0]), np.array([p, y1])) for p in pos]
    if axis == "along":
        n = max(1, ceil((y1 - y0) / spacing))
        pos = y0 + (np.arange(n) + 0.5) * (y1 - y0) / n
        return [(np.array([x0, p]), np.array([x1, p])) for p in pos]
    raise ValueError("axis must be 'cross' or 'along'")


def split_rows(n_rows: int, n_drones: int) -> list[np.ndarray]:
    """Contiguous groups of row indices, one per drone (sizes differ by at most one, the first groups are the larger)."""
    return [g for g in np.array_split(np.arange(n_rows), n_drones)]


def _d_end(xy: np.ndarray, row) -> float:
    return float(min(np.hypot(*(xy - row[0])), np.hypot(*(xy - row[1]))))


def assign_groups(groups: list[np.ndarray], rows, starts: np.ndarray) -> list[np.ndarray]:
    """Assign the groups to the drones so that the sum over drones of (distance to the nearest endpoint of the nearest row of
    its group) is smallest (all permutations; n_drones <= 3 in practice)."""
    from itertools import permutations
    best, best_cost = None, np.inf
    for perm in permutations(range(len(groups))):
        cost = sum(min(_d_end(starts[d], rows[r]) for r in groups[perm[d]]) for d in range(len(groups)))
        if cost < best_cost:
            best, best_cost = perm, cost
    return [groups[best[d]] for d in range(len(groups))]


def row_order(group: np.ndarray, rows, start_xy: np.ndarray) -> list[int]:
    """Visiting order of the rows of one group: start at the row nearest to the drone, continue towards the side with more
    rows, then come back to the rows skipped on the other side (each row at most once)."""
    d = [_d_end(start_xy, rows[r]) for r in group]
    g0 = int(np.argmin(d))
    after = list(group[g0 + 1:])
    before = list(group[:g0][::-1])
    return [int(group[g0])] + [int(r) for r in (after + before if len(after) >= len(before) else before + after)]


def row_points(a: np.ndarray, b: np.ndarray, from_a: bool, wp_spacing: float) -> np.ndarray:
    n = max(2, ceil(float(np.hypot(*(b - a))) / wp_spacing) + 1)
    pts = np.linspace(a, b, n)
    return pts if from_a else pts[::-1]


def coverage_waypoints(starts: np.ndarray, box: Box = (config.PF_PRIOR_X, config.PF_PRIOR_Y), axis: str = "cross",
                       spacing: float = config.LAWN_ROW_SPACING_M, wp_spacing: float = config.LAWN_WP_SPACING_M,
                       obstacles: ObstacleMap | None = None, z: float = config.DRONE_Z) -> tuple[list[np.ndarray], dict]:
    """Waypoint list per drone (starts (n, 2)) and an info dict (rows, groups, visiting orders)."""
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    n = starts.shape[0]
    rows = make_rows(axis, spacing, box)
    groups = split_rows(len(rows), n)
    groups = assign_groups(groups, rows, starts) if n > 1 else groups
    plans, orders = [], []
    for d in range(n):
        order = row_order(groups[d], rows, starts[d])
        orders.append(order)
        pts, cur = [], starts[d]
        for r in order:
            a, b = rows[r]
            from_a = np.hypot(*(cur - a)) <= np.hypot(*(cur - b))              # enter the row at its nearer end -> boustrophedon
            seg = row_points(a, b, from_a, wp_spacing)
            pts.append(seg)
            cur = seg[-1]
        wp = np.vstack(pts)
        if obstacles is not None:
            wp = wp[obstacles.is_free(wp, z)]
        plans.append(wp)
    return plans, {"n_rows": len(rows), "groups": [g.tolist() for g in groups], "orders": orders, "axis": axis, "spacing": spacing}
