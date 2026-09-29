"""Wendland C6 kernel and neighbour-sum ("gather") used by the simulator's concn field and by our sensor.

Data fact (report 2.1, verified to float32 precision): for particle i of source s,
    concn_i = sum_{j in s, r_ij <= H} W_lat(r_ij),  W_lat(r) = C6_W0_LATTICE * (1-q)^8 (1+8q+25q^2+32q^3),  q = r/H,
with H = 7.5 m (3 lattice cells) and C6_W0_LATTICE = 1365/(64*pi*27) = 0.251443 (self term included).
Dividing by the lattice cell volume (2.5 m)^3 gives a number density in particles/m^3.
Reference: Dehnen & Aly (2012) Wendland C6 (docs/references.md R1); SPH number density (R2).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from srcloc_env import config


def wendland_c6_shape(q: np.ndarray) -> np.ndarray:
    """Unnormalised Wendland C6 shape (1-q)^8 (1+8q+25q^2+32q^3) for q in [0,1], 0 outside."""
    q = np.asarray(q, dtype=np.float64)
    inside = q < 1.0
    qc = np.where(inside, q, 0.0)
    w = (1.0 - qc) ** 8 * (1.0 + 8.0 * qc + 25.0 * qc**2 + 32.0 * qc**3)
    return np.where(inside, w, 0.0)


def c6_weight_lattice(r: np.ndarray) -> np.ndarray:
    """Kernel weight in lattice units for separation r [m] (matches the files' concn units)."""
    return config.C6_W0_LATTICE * wendland_c6_shape(np.asarray(r, dtype=np.float64) / config.KERNEL_H)


def c6_gather(query_xyz: np.ndarray, particle_xyz: np.ndarray, tree: cKDTree | None = None) -> np.ndarray:
    """Sum of C6 lattice-unit weights of all particles within H of each query point.

    Returns an array (n_query,) in the same units as the files' concn. Particles coincident with a query
    point (distance 0) contribute the self term C6_W0_LATTICE. Use particle_xyz restricted to the source
    (p_type) and the particle set (all / airborne) that the caller wants to measure.
    """
    query_xyz = np.asarray(query_xyz, dtype=np.float64)
    if query_xyz.ndim == 1:
        query_xyz = query_xyz[None, :]
    out = np.zeros(query_xyz.shape[0], dtype=np.float64)
    if particle_xyz.shape[0] == 0:
        return out
    if tree is None:
        tree = cKDTree(np.asarray(particle_xyz, dtype=np.float64))
    qtree = cKDTree(query_xyz)
    pairs = qtree.sparse_distance_matrix(tree, config.KERNEL_H, output_type="ndarray")  # fields i, j, v
    if pairs.shape[0]:
        np.add.at(out, pairs["i"], c6_weight_lattice(pairs["v"]))
    return out


def c6_density_per_m3(query_xyz: np.ndarray, particle_xyz: np.ndarray, tree: cKDTree | None = None) -> np.ndarray:
    """Number density [particles/m^3] = c6_gather / lattice cell volume."""
    return c6_gather(query_xyz, particle_xyz, tree) / config.LATTICE_CELL_VOLUME