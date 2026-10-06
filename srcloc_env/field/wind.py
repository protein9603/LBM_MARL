"""SPH wind lookup for the drones and the PF forward model (plan D2, report 3.1-3.6).

Data
----
``levels_uvw.npz`` (config.LEVELS_UVW_NPZ) holds (u, v, w) of the SPH flow field on 19 z-levels
(1.25 ... 371.25 m) of the native 2.5 m lattice, ``fluid_slices_2p5m.npz`` (config.FLUID_SLICES_NPZ)
holds 7 low levels (1.25 ... 21.25 m) plus ``building_mask`` and ``building_height_max`` rasterised from
the p_type == 1000 building-surface points onto the same lattice (report 3.6, 4.2).

Axis order (resolved from the data, see scripts/validate_wind.py; config.WIND_AXIS_ORDER)
-----------------------------------------------------------------------------------------
``uvw[i, ix, iy, c]`` with x = x[ix] (0 .. 1315, dx = 2.5), y = y[iy] (-657.5 .. 657.5), c = (u, v, w).
Evidence: the column ``uvw[0, 0, :, :]`` (x = 0) has mean u = 3.390 m/s and v identically 0.00
(report 3.3 inflow column: u 3.39, v 0.00), whereas the transposed reading ``uvw[0, :, 0, :]`` gives
u = 1.85 and |v| up to 0.47; the building mask read as (ix, iy) spans x 200.0-1117.5, y -452.5..452.5,
matching the p_type 1000 bbox x 200.0-1116.75, y +-452.5 (report 3.2/4.2). Whole-level means
(1.146 / 2.652 / 3.825 / 6.528 m/s at z = 1.25 / 11.25 / 21.25 / 48.75) are orientation independent
and reproduced to < 1e-3 m/s. The frame is the fluid/LDM frame (report 5), lattice cell centres.

Interpolation (report 3.6 recipe 3)
-----------------------------------
Horizontal: bilinear between the four bracketing lattice nodes. Vertical: linear between the two
bracketing z-levels (z = 15 -> 13.75 and 16.25). Positions outside the lattice are clipped to the edge
(the LDM domain sticks 7.6 m past x = 1315, report 3.5). Nodes inside buildings, i.e.
``building_mask & (building_height >= z)``, are "dead": their bilinear weights are zeroed and the
remaining weights renormalised; if all four are dead the wind is (0, 0) (report 3.4: lattice points
inside buildings carry a zero-mean noise of ~0.3 m/s that must not be interpolated).

Report conventions reproduced here (verified against 분석스크립트/verify_ldm-timeseries/v14_fluid_src_velocity_k.py)
------------------------------------------------------------------------------------------------------------------
* Report 3.3 level means are over ALL nodes (buildings included): ``mean_profile``.
* Report 3.3 plume-region band means (0.82 / 1.68 / 2.96 m/s for 0-5 / 10-15 / 20-25 m) are over ALL
  nodes with 330 < x < 1315 and |y| < 500 (OPEN bounds: the lattice nodes on x = 330, 1315 and
  y = +-500 are excluded) pooled over the stored levels inside [z_lo, z_hi): ``band_mean``.
  ``region_mean`` therefore uses open intervals; with the default (330, 1315) x (-500, 500) window and
  ``exclude_buildings=False`` it reproduces the report to 1e-4 m/s (inclusive bounds give 1.704, not 1.682).
* Report 2.6 "근방 풍속" at a source is the mean of |(u, v, w)| over ALL lattice nodes with
  |x - xs| < 15, |y - ys| < 15 and 2 < z < 15 (levels 3.75 ... 13.75): ``box_mean_speed``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from srcloc_env import config


class WindField:
    """Mean-wind lookup on the SPH lattice; float32 storage (~85 MB for the 19 levels)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, z_levels: np.ndarray, uvw: np.ndarray,
                 building_mask: np.ndarray | None = None, building_height: np.ndarray | None = None) -> None:
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.z_levels = np.ascontiguousarray(z_levels, dtype=np.float64)
        self.uvw = np.ascontiguousarray(uvw, dtype=np.float32)          # (nz, nx, ny, 3), order [level, ix, iy, comp]
        nz, nx, ny, nc = self.uvw.shape
        if (nz, nx, ny, nc) != (self.z_levels.size, self.x.size, self.y.size, 3):
            raise ValueError(f"uvw shape {self.uvw.shape} does not match (nz={self.z_levels.size}, nx={self.x.size}, "
                             f"ny={self.y.size}, 3): expected axis order [level, ix, iy, comp]")
        if nx < 2 or ny < 2:
            raise ValueError("need at least 2 lattice nodes per horizontal axis")
        if np.any(np.diff(self.z_levels) <= 0):
            raise ValueError("z_levels must be strictly increasing")
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        if not (np.allclose(np.diff(self.x), self.dx) and np.allclose(np.diff(self.y), self.dy)):
            raise ValueError("x and y must be uniformly spaced lattices")
        if building_mask is None:
            if building_height is not None:
                raise ValueError("building_height given without building_mask")
            self.building_mask = np.zeros((nx, ny), dtype=bool)
            self.building_height = np.zeros((nx, ny), dtype=np.float32)
        else:
            self.building_mask = np.ascontiguousarray(building_mask, dtype=bool)
            if building_height is None:
                # no height map: treat every masked node as blocked at every height
                self.building_height = np.where(self.building_mask, np.inf, 0.0).astype(np.float32)
            else:
                self.building_height = np.ascontiguousarray(building_height, dtype=np.float32)
            if self.building_mask.shape != (nx, ny) or self.building_height.shape != (nx, ny):
                raise ValueError("building_mask / building_height must have shape (nx, ny)")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, z_levels: np.ndarray, uvw: np.ndarray,
                    building_mask: np.ndarray | None = None, building_height: np.ndarray | None = None) -> "WindField":
        """Build from in-memory arrays (tests). uvw must be (nz, nx, ny, 3) with x = x[ix], y = y[iy]."""
        return cls(x, y, z_levels, uvw, building_mask, building_height)

    @classmethod
    def load(cls, levels_path: Path = config.LEVELS_UVW_NPZ,
             slices_path: Path | None = config.FLUID_SLICES_NPZ) -> "WindField":
        """Load the 19-level uvw array (float32, read fully once) and the building mask/height map.

        Keys (inspected 2026-09-29): levels: uvw (19,527,527,3) f32, z_levels (19,), x (527,), y (527,);
        slices: building_mask (527,527) bool, building_height_max (527,527) f32 on the same lattice.
        """
        with np.load(levels_path) as z:
            x, y, z_levels, uvw = z["x"], z["y"], z["z_levels"], z["uvw"]
        mask = height = None
        if slices_path is not None:
            with np.load(slices_path) as s:
                if not (np.allclose(s["x"], x) and np.allclose(s["y"], y)):
                    raise ValueError("slices lattice differs from levels lattice")
                mask, height = s["building_mask"], s["building_height_max"]
        return cls(x, y, z_levels, uvw, mask, height)

    # ------------------------------------------------------------------ helpers
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.uvw.shape[0], self.uvw.shape[1], self.uvw.shape[2]

    def dead_mask(self, z: float) -> np.ndarray:
        """(nx, ny) bool: lattice nodes inside a building at height z (report 3.6 recipe 3)."""
        return self.building_mask & (self.building_height >= float(z))

    def _z_bracket(self, z: float) -> tuple[int, int, float]:
        """Indices (k1, k2) and fraction t so that value = (1-t)*level[k1] + t*level[k2]; clamps outside."""
        zl = self.z_levels
        z = float(z)
        if z <= zl[0]:
            return 0, 0, 0.0
        if z >= zl[-1]:
            return zl.size - 1, zl.size - 1, 0.0
        k2 = int(np.searchsorted(zl, z, side="right"))
        k1 = k2 - 1
        return k1, k2, float((z - zl[k1]) / (zl[k2] - zl[k1]))

    def _levels_in(self, z_range: tuple[float, float]) -> np.ndarray:
        """Indices of the stored levels with z_lo <= z < z_hi (half-open band, report 3.3 / v14 convention)."""
        return np.nonzero((self.z_levels >= z_range[0]) & (self.z_levels < z_range[1]))[0]

    def _xy_window(self, x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Boolean node selectors for the OPEN window x_lo < x < x_hi, y_lo < y < y_hi (report 3.3 / v14)."""
        return (self.x > x_range[0]) & (self.x < x_range[1]), (self.y > y_range[0]) & (self.y < y_range[1])

    def level_field(self, z: float) -> np.ndarray:
        """(nx, ny, 3) float32 field linearly interpolated in z between the bracketing levels (read-only view
        of the stored level when z is a stored level or outside the range)."""
        k1, k2, t = self._z_bracket(z)
        if k1 == k2 or t == 0.0:
            return self.uvw[k1]
        return ((1.0 - t) * self.uvw[k1] + t * self.uvw[k2]).astype(np.float32)

    def _interp(self, xy: np.ndarray, z: float, comps: tuple[int, ...]) -> np.ndarray:
        """Bilinear (x, y) x linear (z) interpolation of the requested components with dead-node handling."""
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy[None, :]
        n = xy.shape[0]
        nx, ny = self.x.size, self.y.size
        px = np.clip(xy[:, 0], self.x[0], self.x[-1])
        py = np.clip(xy[:, 1], self.y[0], self.y[-1])
        fx = (px - self.x[0]) / self.dx
        fy = (py - self.y[0]) / self.dy
        ix0 = np.clip(np.floor(fx).astype(np.intp), 0, nx - 2)
        iy0 = np.clip(np.floor(fy).astype(np.intp), 0, ny - 2)
        tx = np.clip(fx - ix0, 0.0, 1.0)
        ty = np.clip(fy - iy0, 0.0, 1.0)
        ix = np.stack([ix0, ix0 + 1, ix0, ix0 + 1], axis=1)              # (n, 4) corner order 00, 10, 01, 11
        iy = np.stack([iy0, iy0, iy0 + 1, iy0 + 1], axis=1)
        w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)   # (n, 4)
        alive = ~self.dead_mask(z)[ix, iy]                                # (n, 4)
        w = w * alive
        wsum = w.sum(axis=1, keepdims=True)
        w = np.divide(w, wsum, out=np.zeros_like(w), where=wsum > 0)       # all four dead -> zero weights
        k1, k2, t = self._z_bracket(z)
        comps = list(comps)
        vals1 = self.uvw[k1][ix, iy][..., comps].astype(np.float64)       # (n, 4, nc)
        out = np.einsum("nk,nkc->nc", w, vals1)
        if t > 0.0 and k2 != k1:
            vals2 = self.uvw[k2][ix, iy][..., comps].astype(np.float64)
            out = (1.0 - t) * out + t * np.einsum("nk,nkc->nc", w, vals2)
        return out.reshape(n, len(comps))

    # ------------------------------------------------------------------ public queries
    def uv_at(self, xy: np.ndarray, z: float = config.DRONE_Z) -> np.ndarray:
        """Horizontal wind (u, v) [m/s], shape (n, 2), at positions xy (n, 2) and height z."""
        return self._interp(xy, z, (0, 1))

    def w_at(self, xy: np.ndarray, z: float = config.DRONE_Z) -> np.ndarray:
        """Vertical wind w [m/s], shape (n,), same interpolation as uv_at."""
        return self._interp(xy, z, (2,))[:, 0]

    def uvw_at(self, xy: np.ndarray, z: float = config.DRONE_Z) -> np.ndarray:
        """All three components, shape (n, 3)."""
        return self._interp(xy, z, (0, 1, 2))

    def mean_profile(self, z: float) -> tuple[float, float]:
        """(u_mean, v_mean) over ALL lattice nodes at height z (buildings included, as in report 3.3)."""
        k1, k2, t = self._z_bracket(z)
        m1 = self.uvw[k1, :, :, :2].mean(axis=(0, 1), dtype=np.float64)
        if t > 0.0 and k2 != k1:
            m2 = self.uvw[k2, :, :, :2].mean(axis=(0, 1), dtype=np.float64)
            m1 = (1.0 - t) * m1 + t * m2
        return float(m1[0]), float(m1[1])

    def region_mean(self, z: float, x_range: tuple[float, float] = config.PLUME_REGION_X,
                    y_range: tuple[float, float] = config.PLUME_REGION_Y,
                    exclude_buildings: bool = True) -> tuple[float, float]:
        """(u, v) mean at height z over lattice nodes strictly inside the window (x_lo < x < x_hi, y_lo < y < y_hi).

        Open bounds follow report 3.3 / v14 (x > 330, x < 1315, |y| < 500): with the default window and
        exclude_buildings=False the 10-15 m band reproduces 1.68 m/s (inclusive bounds would give 1.70).
        With exclude_buildings=True the dead nodes (mask & height >= z) are dropped.
        """
        sx, sy = self._xy_window(x_range, y_range)
        sel = sx[:, None] & sy[None, :]
        if exclude_buildings:
            sel = sel & ~self.dead_mask(z)
        if not np.any(sel):
            return 0.0, 0.0
        field = self.level_field(z)
        m = field[sel][:, :2].mean(axis=0, dtype=np.float64)
        return float(m[0]), float(m[1])

    def band_mean(self, z_range: tuple[float, float], x_range: tuple[float, float] = config.PLUME_REGION_X,
                  y_range: tuple[float, float] = config.PLUME_REGION_Y,
                  exclude_buildings: bool = False) -> tuple[float, float]:
        """(u, v) pooled over the stored levels with z_lo <= z < z_hi and the open xy window (report 3.3 band means).

        Report 3.3 band values (0.82 / 1.68 / 2.96 for 0-5 / 10-15 / 20-25 m, config.WIND_REF_PLUME_U_BANDS)
        are over ALL nodes, hence exclude_buildings=False by default. Returns (0, 0) if no level or node qualifies.
        """
        kz = self._levels_in(z_range)
        sx, sy = self._xy_window(x_range, y_range)
        if kz.size == 0 or not (sx.any() and sy.any()):
            return 0.0, 0.0
        blk = self.uvw[np.ix_(kz, np.nonzero(sx)[0], np.nonzero(sy)[0])][..., :2].astype(np.float64)   # (nk, nxw, nyw, 2)
        if exclude_buildings:
            alive = np.stack([~self.dead_mask(self.z_levels[k])[np.ix_(sx, sy)] for k in kz])           # (nk, nxw, nyw)
            if not alive.any():
                return 0.0, 0.0
            m = blk[alive].mean(axis=0)
        else:
            m = blk.reshape(-1, 2).mean(axis=0)
        return float(m[0]), float(m[1])

    def box_mean_speed(self, xy: tuple[float, float] | np.ndarray,
                       half_width: float = config.WIND_REF_SOURCE_BOX_HALF_M,
                       z_range: tuple[float, float] = config.WIND_REF_SOURCE_Z_RANGE) -> float:
        """Report 2.6 "근방 풍속": mean |(u, v, w)| over ALL lattice nodes with |x - xs| < half_width,
        |y - ys| < half_width and z_lo < z < z_hi (open bounds, buildings included; v14_fluid_src_velocity_k.py).

        Reproduces 0.33 / 0.63 / 3.79 / 0.43 m/s for sources 103 / 107 / 109 / 110 (config.WIND_REF_SOURCE_SPEED).
        Returns nan if no node qualifies.
        """
        xs, ys = float(xy[0]), float(xy[1])
        sx = np.abs(self.x - xs) < half_width
        sy = np.abs(self.y - ys) < half_width
        kz = np.nonzero((self.z_levels > z_range[0]) & (self.z_levels < z_range[1]))[0]
        if kz.size == 0 or not (sx.any() and sy.any()):
            return float("nan")
        blk = self.uvw[np.ix_(kz, np.nonzero(sx)[0], np.nonzero(sy)[0])].astype(np.float64)
        return float(np.linalg.norm(blk, axis=-1).mean())

    def direction_deg(self, z: float, **region_kwargs) -> float:
        """Mean wind direction atan2(v, u) in degrees (0 = +x) from mean_profile, or region_mean if kwargs given."""
        u, v = self.region_mean(z, **region_kwargs) if region_kwargs else self.mean_profile(z)
        return float(np.degrees(np.arctan2(v, u)))