"""Per-source concentration slabs at drone altitudes from one LDM frame (plan S0).

For each source s and slab height z_k, the slab value at cell centre (x, y, z_k) is the exact Wendland C6
gather (same kernel and units as the simulator's concn) over the AIRBORNE particles of source s, divided
by the lattice cell volume -> particles/m^3, stored as float16 on a 5 m grid (config.SLAB_*).
Only particles with |z - z_k| <= H can contribute, so each gather uses that vertical band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from srcloc_env import config
from srcloc_env.io.ldm_reader import LdmFrame
from srcloc_env.sensor.kernel import c6_gather


@dataclass(frozen=True)
class SlabGrid:
    x0: float = config.SLAB_X0
    y0: float = config.SLAB_Y0
    nx: int = config.SLAB_NX
    ny: int = config.SLAB_NY
    res: float = config.SLAB_RES

    @property
    def x_centres(self) -> np.ndarray:
        return self.x0 + self.res * (np.arange(self.nx) + 0.5)

    @property
    def y_centres(self) -> np.ndarray:
        return self.y0 + self.res * (np.arange(self.ny) + 0.5)

    def cell_centres(self, z: float) -> np.ndarray:
        """(ny*nx, 3) array of cell-centre coordinates at height z, row-major (iy, ix)."""
        xx, yy = np.meshgrid(self.x_centres, self.y_centres)      # shapes (ny, nx)
        return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, float(z))])

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "nx": self.nx, "ny": self.ny, "res": self.res}


def slab_for_source(xyz: np.ndarray, z: float, grid: SlabGrid) -> np.ndarray:
    """Density [particles/m^3] on the (ny, nx) grid at height z from the given particle positions."""
    band = np.abs(xyz[:, 2] - z) <= config.KERNEL_H
    pts = xyz[band].astype(np.float64)
    centres = grid.cell_centres(z)
    if pts.shape[0] == 0:
        return np.zeros((grid.ny, grid.nx), dtype=np.float64)
    dens = c6_gather(centres, pts, cKDTree(pts)) / config.LATTICE_CELL_VOLUME
    return dens.reshape(grid.ny, grid.nx)


def build_slabs(frame: LdmFrame, z_levels: tuple[float, ...], sources: tuple[int, ...] = config.ALL_SOURCES,
                grid: SlabGrid = SlabGrid(), airborne_only: bool = True) -> np.ndarray:
    """Return an array (n_sources, n_z, ny, nx) of float16 densities [particles/m^3]."""
    keep = frame.airborne if airborne_only else np.ones(frame.n, dtype=bool)
    out = np.zeros((len(sources), len(z_levels), grid.ny, grid.nx), dtype=config.SLAB_DTYPE)
    for si, s in enumerate(sources):
        xyz = frame.xyz[keep & (frame.p_type == s)]
        for zi, z in enumerate(z_levels):
            out[si, zi] = slab_for_source(xyz, float(z), grid).astype(config.SLAB_DTYPE)
    return out


def slab_stats(slabs: np.ndarray) -> dict:
    """Per-source, per-level summary used by the T0-4 growth table and figure 2."""
    s = slabs.astype(np.float32)
    return {
        "max": s.max(axis=(2, 3)).tolist(),
        "mean_nonzero": [[float(v[v > 0].mean()) if np.any(v > 0) else 0.0 for v in row] for row in s],
        "occupied_cells": (s > 0).sum(axis=(2, 3)).tolist(),
    }