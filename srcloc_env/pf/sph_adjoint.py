"""Steady 2-D advection-diffusion source-receptor (adjoint) forward model on the SPH wind field
(plan 4.2b, option B-1; design note docs/sph_forward_model.md sections 2-5 and 7).

Physics (sph_forward_model.md section 2)
----------------------------------------
On the flight-altitude layer z = config.DRONE_Z the steady concentration C(x, y) [particles/m^2 of the layer]
of a continuous source q [particles/s] at x_s obeys

    div(u C) - div(K grad C) + lam C = q f_s(x)

with u(x, y) the SPH horizontal wind (field/wind.py), K the effective horizontal eddy diffusivity (R15 Taylor:
K = sigma_v^2 T_L, config.ADJ_K_DEFAULT), lam a first-order loss rate standing for vertical mixing out of the
layer plus deposition (config.ADJ_LAMBDA_DEFAULT) and f_s a Gaussian footprint of std sigma0 =
config.RELEASE_SIGMA_XY (report 2.6) normalised to 1 over the free cells.  Cells whose roof reaches the layer
are walls (removed from the unknowns, zero flux through their faces); the domain edges are outflow where the
wind points outwards and zero-concentration inflow otherwise.  Densities per m^3 are C / h_layer
(config.ADJ_H_LAYER); that constant factor is absorbed by the PF's kappa (plan 4.3).

Discretisation (section 3; R21 Patankar 1980 Ch. 5)
--------------------------------------------------
Finite volumes on the cell-centred slab grid (preprocess/gridder.SlabGrid, spacing res, area res^2).  For the
face between cells i and j (length res, normal n_ij from i to j) the face-normal velocity is the mean of the
two cell velocities u_n = 0.5 (u_i + u_j) . n_ij and the flux i -> j is

    F_ij = res [max(u_n, 0) C_i - max(-u_n, 0) C_j]  (upwind donor cell)  +  K (C_i - C_j)  (central diffusion)

Summing the outgoing fluxes and adding the loss lam res^2 C_i gives the row of A in A C = b, b = q x footprint.
Every coefficient is positive (Patankar's rule), A is a Z-matrix whose column sums are lam res^2 plus the
boundary-outflow coefficients, hence an M-matrix: C >= 0 and the discrete budget

    lam sum_i C_i res^2 + sum_(boundary) res max(u_n, 0) C_i = sum_i b_i = q            (section 7, mass balance)

holds to round-off because the internal face fluxes cancel pairwise.  The upwind scheme adds a numerical
diffusion ~ |u| res / 2 along the flow direction which is absorbed by the calibration of K (section 8).

Adjoint (section 4; R17 Keats, Yee & Lien 2007; R18 Pudykiewicz 1998; R19 Hourdin & Talagrand 2006)
----------------------------------------------------------------------------------------------------
The problem is linear: the density seen at receptor p from a unit source at s is e_p^T A^-1 F e_s / h with
F the (n_free x n_free) footprint operator whose column s is the footprint of a source at the centre of cell s.
Fixing the receptor and transposing (discretise-then-adjoint, R19) gives the sensitivity field

    psi_p = F^T A^-T e_p / h_layer,      psi_p[s] = density at p for a unit source footprint centred at cell s,

which is exactly what the PF needs for all N hypotheses at once (R17, R18).  The footprint of a source is
restricted to the free cells of the source cell's own 4-connected component (scipy.ndimage.label on the free
mask, the same face connectivity as the operator), so a source next to a wall thinner than the footprint box
does not inject mass on the far side.  F^T is applied exactly: column s of F is w(t - s) [t in comp(s)] /
colsum_s with a separable Gaussian w, and A^-T e_p is nonzero only in the receptor's component(s), so
(F^T phi)_s is the separable correlation of phi masked to each such component, divided by the per-source
normalisation colsum_s (``apply_footprint_transpose``, ~0.5 ms; the sparse ``FT`` is kept as the reference
path and unit-tested against it).
Reciprocity solve_forward(s)[p] == solve_adjoint(p)[s] therefore holds to round-off also next to walls, where
the per-column normalisation makes F non-symmetric.  The receptor e_p is spread bilinearly over the four cell
centres around the receptor position (blocked cells get zero weight and the rest is renormalised); the same
weights are used by ``interpolate`` so that continuous-receptor reciprocity holds too.  A and A^T are both
LU-factorised once with scipy.sparse.linalg.splu (``factorize``; ordering config.ADJ_SPLU_PERMC_SPEC with
SymmetricMode, measured 35 % less fill and 3.9 ms per solve on the real-size grid); every adjoint solve is
then one pair of triangular solves plus the footprint correlation (target config.ADJ_SOLVE_TIME_TARGET_S).

``SphAdjointModel`` wraps the operator with the ``unit_response(source_xy (N, 2), drone_xyz (M, 3)) -> (N, M)``
interface of pf/forward_model.GaussianPlume (section 5 step 4): psi of each drone is LRU-cached by the
receptor position rounded to config.ADJ_CACHE_ROUND_M and bilinearly interpolated at the N hypotheses.  The
drone z is ignored (single-layer model at params.z).  Responses are floored at config.FWD_G_FLOOR like the
analytic plume so the RBPF likelihood stays finite.

Array conventions: fields are (ny, nx) indexed [iy, ix] like the slabs (cell (iy, ix) centred at
x0 + res (ix + 0.5), y0 + res (iy + 0.5)); the wind input ``uv`` is (ny, nx, 2) at the cell centres and
``blocked`` is (ny, nx) bool.  Everything is numpy / scipy.sparse vectorised; all defaults come from
srcloc_env.config.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import correlate1d, label
from scipy.sparse.linalg import SuperLU, splu

from srcloc_env import config
from srcloc_env.env.drone import ObstacleMap
from srcloc_env.field.wind import WindField
from srcloc_env.preprocess.gridder import SlabGrid


@dataclass(frozen=True)
class AdjointParams:
    """Parameters of the layer model (sph_forward_model.md section 2; defaults from config, provenance there).

    K          horizontal eddy diffusivity [m^2/s]        (config.ADJ_K_DEFAULT = sigma_v^2 T_L, R15)
    lam        first-order loss rate [1/s]                (config.ADJ_LAMBDA_DEFAULT; calibrated in T1-3b)
    h_layer    layer thickness [m] for particles/m^3      (config.ADJ_H_LAYER)
    sigma0     source footprint std [m]                   (config.RELEASE_SIGMA_XY, report 2.6)
    z          layer height / wind sampling height [m]    (config.DRONE_Z)
    wind_band  None -> wind interpolated at z; (z_lo, z_hi) -> per-cell mean over the stored SPH levels with
               z_lo <= z_k < z_hi (section 6: "15 m single vs 10-20 m band")
    """

    K: float = config.ADJ_K_DEFAULT
    lam: float = config.ADJ_LAMBDA_DEFAULT
    h_layer: float = config.ADJ_H_LAYER
    sigma0: float = config.RELEASE_SIGMA_XY
    z: float = config.DRONE_Z
    wind_band: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.K < 0.0 or self.lam < 0.0 or self.h_layer <= 0.0 or self.sigma0 <= 0.0:
            raise ValueError("AdjointParams: need K >= 0, lam >= 0, h_layer > 0, sigma0 > 0")
        if self.wind_band is not None:
            if len(self.wind_band) != 2 or not self.wind_band[1] > self.wind_band[0]:
                raise ValueError("wind_band must be (z_lo, z_hi) with z_hi > z_lo")


def _as_xy(a: np.ndarray) -> np.ndarray:
    """(n, 2) float64 view of a (2,) or (n, >=2) array (extra columns such as z are dropped)."""
    p = np.asarray(a, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError(f"expected positions of shape (n, 2) or (2,), got {p.shape}")
    return p[:, :2]


class AdvectionDiffusionOperator:
    """Sparse steady advection-diffusion operator A on the free cells of a SlabGrid (module docstring).

    Attributes after construction: ``A`` (n_free x n_free csr), ``F`` / ``FT`` (footprint operator and its
    transpose, csr, reference path), ``cell_to_unk`` (ny*nx,) with -1 at blocked cells, ``unk_to_cell``
    (n_free,), ``outflow_coef`` (n_free,) boundary outflow coefficients res max(u_n, 0), ``component`` (ny, nx)
    4-connected free-component labels (0 = blocked; ``n_components``), ``assembly_seconds``; after
    ``factorize``: ``lu`` / ``luT`` (SuperLU of A and A^T) and ``factorize_seconds``.
    """

    def __init__(self, grid: SlabGrid, uv: np.ndarray, blocked: np.ndarray,
                 params: AdjointParams | None = None) -> None:
        self.grid = grid
        self.params = params if params is not None else AdjointParams()
        uv = np.asarray(uv, dtype=np.float64)
        blocked = np.asarray(blocked, dtype=bool)
        if uv.shape != (grid.ny, grid.nx, 2):
            raise ValueError(f"uv must have shape (ny={grid.ny}, nx={grid.nx}, 2), got {uv.shape}")
        if blocked.shape != (grid.ny, grid.nx):
            raise ValueError(f"blocked must have shape (ny={grid.ny}, nx={grid.nx}), got {blocked.shape}")
        if grid.nx < 2 or grid.ny < 2:
            raise ValueError("the grid needs at least 2 x 2 cells")
        self.uv = np.ascontiguousarray(uv)
        self.blocked = np.ascontiguousarray(blocked)
        self.free = ~self.blocked
        self.n_cells = int(grid.nx * grid.ny)
        self.unk_to_cell = np.flatnonzero(self.free.ravel())
        self.n_free = int(self.unk_to_cell.size)
        if self.n_free == 0:
            raise ValueError("every cell is blocked")
        self.cell_to_unk = np.full(self.n_cells, -1, dtype=np.intp)
        self.cell_to_unk[self.unk_to_cell] = np.arange(self.n_free)
        self._unk_grid = self.cell_to_unk.reshape(grid.ny, grid.nx)
        self.component, self.n_components = label(self.free)          # 4-connectivity = face coupling of A
        self._comp_free = self.component.ravel()[self.unk_to_cell]
        t0 = time.perf_counter()
        self.A, self.outflow_coef = self._assemble()
        r = self.footprint_radius_cells
        self._fp_kernel = np.exp(-0.5 * (np.arange(-r, r + 1) * grid.res) ** 2 / self.params.sigma0 ** 2)
        self._fp_colsum = np.zeros(self.n_free)
        for c in range(1, self.n_components + 1):
            sel = self._comp_free == c
            self._fp_colsum[sel] = self._correlate((self.component == c).astype(np.float64)).ravel()[self.unk_to_cell][sel]
        self.F = self._footprint_matrix()
        self.FT = sp.csr_matrix(self.F.T)
        self.assembly_seconds = time.perf_counter() - t0
        self.lu: SuperLU | None = None
        self.luT: SuperLU | None = None
        self.factorize_seconds = float("nan")

    # ------------------------------------------------------------------ construction from data
    @classmethod
    def from_arrays(cls, grid: SlabGrid, uv: np.ndarray, blocked: np.ndarray,
                    params: AdjointParams | None = None) -> "AdvectionDiffusionOperator":
        """Build from in-memory arrays (tests): uv (ny, nx, 2) wind at the cell centres, blocked (ny, nx) bool."""
        return cls(grid, uv, blocked, params)

    @staticmethod
    def wind_at_cells(wind_field: WindField, grid: SlabGrid, params: AdjointParams) -> np.ndarray:
        """(ny, nx, 2) wind at the cell centres: WindField.uv_at at params.z, or the per-cell mean over the
        stored levels with z_lo <= z_k < z_hi when params.wind_band is given (section 6)."""
        centres = grid.cell_centres(params.z)[:, :2]
        if params.wind_band is None:
            uv = wind_field.uv_at(centres, params.z)
        else:
            lo, hi = params.wind_band
            levels = wind_field.z_levels[(wind_field.z_levels >= lo) & (wind_field.z_levels < hi)]
            if levels.size == 0:
                raise ValueError(f"no stored wind level in the band [{lo}, {hi})")
            uv = np.mean([wind_field.uv_at(centres, float(zk)) for zk in levels], axis=0)
        return np.asarray(uv, dtype=np.float64).reshape(grid.ny, grid.nx, 2)

    @staticmethod
    def blocked_at_cells(obstacles: ObstacleMap, grid: SlabGrid, z: float) -> np.ndarray:
        """(ny, nx) bool: ObstacleMap.no_fly_mask(z, margin=0) (roof >= z) sampled at the cell centres; cell
        centres outside the raster are free (the raster covers the whole building footprint, report 4.2)."""
        centres = grid.cell_centres(z)[:, :2]
        ix, iy = obstacles.cell_index(centres)
        inside = (ix >= 0) & (ix < obstacles.nx) & (iy >= 0) & (iy < obstacles.ny)
        mask = obstacles.no_fly_mask(z, margin=0.0)
        blocked = np.zeros(centres.shape[0], dtype=bool)
        blocked[inside] = mask[ix[inside], iy[inside]]
        return blocked.reshape(grid.ny, grid.nx)

    @classmethod
    def from_data(cls, params: AdjointParams | None = None, wind_field: WindField | None = None,
                  obstacles: ObstacleMap | None = None, grid: SlabGrid = SlabGrid()) -> "AdvectionDiffusionOperator":
        """Assemble on the real SPH wind (WindField.load) and building raster (ObstacleMap.load) when not given."""
        params = params if params is not None else AdjointParams()
        if wind_field is None:
            wind_field = WindField.load()
        if obstacles is None:
            obstacles = ObstacleMap.load()
        uv = cls.wind_at_cells(wind_field, grid, params)
        blocked = cls.blocked_at_cells(obstacles, grid, params.z)
        return cls(grid, uv, blocked, params)

    # ------------------------------------------------------------------ assembly
    def _assemble(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """Upwind / central finite-volume assembly (module docstring); returns (A csr, outflow_coef)."""
        g, p = self.grid, self.params
        res, K = g.res, p.K
        U, V = self.uv[..., 0], self.uv[..., 1]
        M = self._unk_grid
        free = self.free
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def add(r: np.ndarray, c: np.ndarray, v: np.ndarray) -> None:
            rows.append(r)
            cols.append(c)
            vals.append(v)

        def internal_faces(un: np.ndarray, mi: np.ndarray, mj: np.ndarray, both: np.ndarray) -> None:
            """Faces i -> j with face-normal velocity un (i to j); only faces between two free cells."""
            i, j, u = mi[both], mj[both], un[both]
            out_i = res * np.maximum(u, 0.0)          # advective coefficient leaving i (entering j)
            out_j = res * np.maximum(-u, 0.0)         # advective coefficient leaving j (entering i)
            add(i, i, out_i + K)
            add(i, j, -(out_j + K))
            add(j, j, out_j + K)
            add(j, i, -(out_i + K))

        # x-faces between (iy, ix) and (iy, ix + 1); y-faces between (iy, ix) and (iy + 1, ix)
        internal_faces(0.5 * (U[:, :-1] + U[:, 1:]), M[:, :-1], M[:, 1:], free[:, :-1] & free[:, 1:])
        internal_faces(0.5 * (V[:-1, :] + V[1:, :]), M[:-1, :], M[1:, :], free[:-1, :] & free[1:, :])
        # domain-boundary faces: outflow with the cell value where the cell velocity points outwards, else
        # zero-concentration inflow (no flux); never a diffusive flux
        outflow = np.zeros(self.n_free)
        for un_out, m, fr in ((-U[:, 0], M[:, 0], free[:, 0]), (U[:, -1], M[:, -1], free[:, -1]),
                              (-V[0, :], M[0, :], free[0, :]), (V[-1, :], M[-1, :], free[-1, :])):
            coef = res * np.maximum(un_out[fr], 0.0)
            np.add.at(outflow, m[fr], coef)
        idx = np.arange(self.n_free)
        add(idx, idx, outflow + p.lam * res * res)
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.n_free, self.n_free)).tocsr()
        A.sum_duplicates()
        return A, outflow

    @property
    def footprint_radius_cells(self) -> int:
        """Half-width of the footprint box in cells: ceil(ADJ_FOOTPRINT_TRUNC_SIGMA sigma0 / res)."""
        return int(np.ceil(config.ADJ_FOOTPRINT_TRUNC_SIGMA * self.params.sigma0 / self.grid.res))

    def _correlate(self, field: np.ndarray) -> np.ndarray:
        """Separable correlation of a (ny, nx) field with the truncated Gaussian footprint kernel, zero outside."""
        k = self._fp_kernel
        return correlate1d(correlate1d(field, k, axis=1, mode="constant", cval=0.0), k, axis=0,
                           mode="constant", cval=0.0)

    def _footprint_matrix(self) -> sp.csr_matrix:
        """F (n_free x n_free): column s = Gaussian(sigma0) footprint of a source at the centre of free cell s,
        on the free cells of the (2r+1)^2 box around it that belong to the same connected component,
        normalised to sum 1 (same formula as ``source_vector``)."""
        g = self.grid
        r = self.footprint_radius_cells
        iy_s, ix_s = np.divmod(self.unk_to_cell, g.nx)
        M = self._unk_grid
        rows, cols, vals = [], [], []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                iy, ix = iy_s + dy, ix_s + dx
                ok = (iy >= 0) & (iy < g.ny) & (ix >= 0) & (ix < g.nx)
                tgt = np.full(self.n_free, -1, dtype=np.intp)
                tgt[ok] = M[iy[ok], ix[ok]]
                ok &= tgt >= 0
                ok &= self._comp_free[np.maximum(tgt, 0)] == self._comp_free
                w = self._fp_kernel[dy + r] * self._fp_kernel[dx + r]
                rows.append(tgt[ok])
                cols.append(np.flatnonzero(ok))
                vals.append(np.full(int(ok.sum()), w))
        F = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.n_free, self.n_free)).tocsc()
        F = F @ sp.diags(1.0 / self._fp_colsum)
        return sp.csr_matrix(F)

    def apply_footprint_transpose(self, phi: np.ndarray) -> np.ndarray:
        """F^T phi (n_free,) by separable correlation per connected component carrying nonzero phi (one for an
        adjoint solution): identical to ``FT @ phi`` to round-off, ~6x faster."""
        out = np.zeros(self.n_free)
        for c in np.unique(self._comp_free[phi != 0.0]):
            sel = self._comp_free == c
            corr = self._correlate(self._to_field(np.where(sel, phi, 0.0))).ravel()[self.unk_to_cell]
            out[sel] = corr[sel] / self._fp_colsum[sel]
        return out

    # ------------------------------------------------------------------ vectors
    def source_vector(self, source_xy: np.ndarray) -> np.ndarray:
        """b (n_free,) for q = 1 particle/s: Gaussian(sigma0) footprint of a source at source_xy (2,) evaluated
        at the free cell centres of the box of +-footprint_radius_cells around the cell containing the source
        that belong to the source cell's connected component (all free cells of the box if the source cell is
        itself blocked or outside the grid), normalised to sum 1 (all zeros if no such cell).  At a free cell
        centre it equals F[:, s]."""
        g = self.grid
        x, y = (float(v) for v in _as_xy(source_xy)[0])
        r = self.footprint_radius_cells
        ixs = int(np.floor((x - g.x0) / g.res))
        iys = int(np.floor((y - g.y0) / g.res))
        ix = np.arange(max(ixs - r, 0), min(ixs + r, g.nx - 1) + 1)
        iy = np.arange(max(iys - r, 0), min(iys + r, g.ny - 1) + 1)
        b = np.zeros(self.n_free)
        if ix.size == 0 or iy.size == 0:
            return b
        IY, IX = np.meshgrid(iy, ix, indexing="ij")
        unk = self._unk_grid[IY, IX].ravel()
        d2 = ((g.x_centres[IX] - x) ** 2 + (g.y_centres[IY] - y) ** 2).ravel()
        w = np.exp(-0.5 * d2 / self.params.sigma0 ** 2)
        ok = unk >= 0
        if 0 <= ixs < g.nx and 0 <= iys < g.ny and self.component[iys, ixs] > 0:
            ok &= self.component[IY, IX].ravel() == self.component[iys, ixs]
        tot = w[ok].sum()
        if tot > 0.0:
            b[unk[ok]] = w[ok] / tot
        return b

    def bilinear_weights(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(unk (n, 4), w (n, 4)): free-cell unknown indices (-1 where blocked / unused) and bilinear weights of
        the four cell centres around each point, blocked weights zeroed and the rest renormalised; points
        outside the grid (beyond the outer cell centres by more than half a cell) get all-zero weights."""
        g = self.grid
        p = _as_xy(xy)
        n = p.shape[0]
        fx = (p[:, 0] - g.x0) / g.res - 0.5
        fy = (p[:, 1] - g.y0) / g.res - 0.5
        inside = (fx >= -0.5) & (fx <= g.nx - 0.5) & (fy >= -0.5) & (fy <= g.ny - 0.5)
        fx = np.clip(fx, 0.0, g.nx - 1.0)
        fy = np.clip(fy, 0.0, g.ny - 1.0)
        ix0 = np.minimum(np.floor(fx).astype(np.intp), g.nx - 2)
        iy0 = np.minimum(np.floor(fy).astype(np.intp), g.ny - 2)
        tx, ty = fx - ix0, fy - iy0
        ix = np.stack([ix0, ix0 + 1, ix0, ix0 + 1], axis=1)
        iy = np.stack([iy0, iy0, iy0 + 1, iy0 + 1], axis=1)
        w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)
        unk = self._unk_grid[iy, ix]
        w = w * (unk >= 0) * inside[:, None]
        tot = w.sum(axis=1, keepdims=True)
        w = np.divide(w, tot, out=np.zeros((n, 4)), where=tot > 0.0)
        return unk, w

    def receptor_vector(self, receptor_xy: np.ndarray) -> np.ndarray:
        """e_p (n_free,): the bilinear weights of ``bilinear_weights`` scattered onto the free unknowns."""
        unk, w = self.bilinear_weights(receptor_xy)
        e = np.zeros(self.n_free)
        ok = unk[0] >= 0
        np.add.at(e, unk[0][ok], w[0][ok])
        return e

    def interpolate(self, field: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """(n,) bilinear samples of a (ny, nx) field at xy (n, 2) with the same blocked-aware weights as
        ``receptor_vector``: for a source s at a cell centre and ANY receptor p,
        interpolate(solve_forward(s), p) == solve_adjoint(p)[s] exactly.  For an off-centre source the
        interpolated psi is an approximation (section 5 step 4): the true footprint of an off-centre source is
        a Gaussian centred there, not the bilinear blend of the four centred footprints (~10 % at res/2)."""
        unk, w = self.bilinear_weights(xy)
        vals = np.asarray(field, dtype=np.float64).reshape(-1)[self.unk_to_cell[np.maximum(unk, 0)]]
        return np.einsum("nk,nk->n", w, vals)

    def _to_field(self, values_free: np.ndarray) -> np.ndarray:
        """Scatter (n_free,) onto the (ny, nx) grid with zeros in blocked cells."""
        out = np.zeros(self.n_cells)
        out[self.unk_to_cell] = values_free
        return out.reshape(self.grid.ny, self.grid.nx)

    # ------------------------------------------------------------------ factorisation and solves
    def factorize(self) -> "AdvectionDiffusionOperator":
        """LU-factorise A (forward solves) and A^T (adjoint solves) with SuperLU; returns self.

        Ordering config.ADJ_SPLU_PERMC_SPEC with SymmetricMode = config.ADJ_SPLU_SYMMETRIC_MODE (the sparsity
        pattern of the 5-point operator is symmetric): on the real-size grid L+U holds 0.99 M instead of 1.5 M
        nonzeros (COLAMD) and one triangular solve pair takes ~4 ms instead of ~6 ms (D4-4 measurement).

        A is singular when a free component has no sink at all: lam = 0 and no boundary outflow (an enclosed
        courtyard such as source 110's at 15 m), or K = 0, lam = 0 and zero wind in a cell.  Both cases raise a
        ValueError here instead of SuperLU's bare "Factor is exactly singular" (review B1).
        """
        t0 = time.perf_counter()
        if self.params.lam == 0.0:
            comp_outflow = np.bincount(self._comp_free, weights=self.outflow_coef, minlength=self.n_components + 1)[1:]
            n_trapped = int(np.count_nonzero(comp_outflow == 0.0))
            if n_trapped > 0:
                raise ValueError(f"lam = 0 with {n_trapped} free component(s) without boundary outflow: A is singular "
                                 "there (no sink for the injected mass); use lam > 0 (config.ADJ_LAMBDA_CANDIDATES)")
        opts = dict(SymmetricMode=bool(config.ADJ_SPLU_SYMMETRIC_MODE))
        try:
            self.lu = splu(sp.csc_matrix(self.A), permc_spec=config.ADJ_SPLU_PERMC_SPEC, options=opts)
            self.luT = splu(sp.csc_matrix(self.A.T), permc_spec=config.ADJ_SPLU_PERMC_SPEC, options=opts)
        except RuntimeError as e:                        # SuperLU: "Factor is exactly singular"
            raise ValueError("A is singular: a free cell has neither advective / diffusive coupling nor a loss term "
                             "(K = 0 with zero wind and lam = 0); use K > 0 or lam > 0") from e
        self.factorize_seconds = time.perf_counter() - t0
        return self

    def _require_lu(self) -> None:
        if self.lu is None or self.luT is None:
            self.factorize()

    def solve_forward(self, source_xy: np.ndarray) -> np.ndarray:
        """Density field (ny, nx) [particles/m^3] of a unit source (1 particle/s) at source_xy; zeros in
        blocked cells.  C = A^-1 b(source) / h_layer."""
        self._require_lu()
        c = self.lu.solve(self.source_vector(source_xy))
        return self._to_field(c / self.params.h_layer)

    def solve_adjoint(self, receptor_xy: np.ndarray, sparse_footprint: bool = False) -> np.ndarray:
        """Sensitivity field psi (ny, nx): psi[s] = density [particles/m^3] at receptor_xy for a unit source
        footprint centred at cell s; psi = F^T A^-T e_p / h_layer, zeros in blocked cells.  F^T is applied by
        the separable correlation (default) or by the sparse matrix (sparse_footprint=True, reference path)."""
        self._require_lu()
        phi = self.luT.solve(self.receptor_vector(receptor_xy))
        ft_phi = (self.FT @ phi) if sparse_footprint else self.apply_footprint_transpose(phi)
        return self._to_field(ft_phi / self.params.h_layer)

    def mass_balance(self, field: np.ndarray) -> dict[str, float]:
        """Discrete budget of a unit-source density field (ny, nx): loss = lam sum(C) area, outflow = sum of
        the boundary outflow fluxes, source = 1; residual = loss + outflow - source (round-off for a forward
        solution, section 7)."""
        c = np.asarray(field, dtype=np.float64).reshape(-1)[self.unk_to_cell] * self.params.h_layer
        loss = float(self.params.lam * c.sum() * self.grid.res ** 2)
        outflow = float(self.outflow_coef @ c)
        return {"loss": loss, "outflow": outflow, "source": 1.0, "residual": loss + outflow - 1.0,
                "total_mass_particles": float(c.sum() * self.grid.res ** 2)}

    # ------------------------------------------------------------------ helpers
    def free_cell_centres(self) -> np.ndarray:
        """(n_free, 2) centres of the free cells in unknown order."""
        iy, ix = np.divmod(self.unk_to_cell, self.grid.nx)
        return np.column_stack([self.grid.x_centres[ix], self.grid.y_centres[iy]])

    def random_free_cells(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n,) unknown indices of random free cells."""
        return rng.integers(0, self.n_free, int(n))

    def time_adjoint_solves(self, n: int, rng: np.random.Generator | None = None) -> dict[str, float]:
        """Wall time of n adjoint solves at random free cell centres: median / mean / p99 / max [s]."""
        rng = rng if rng is not None else np.random.default_rng()
        self._require_lu()
        xy = self.free_cell_centres()[self.random_free_cells(n, rng)]
        t = np.empty(int(n))
        for i in range(int(n)):
            t0 = time.perf_counter()
            self.solve_adjoint(xy[i])
            t[i] = time.perf_counter() - t0
        return {"n": int(n), "median_s": float(np.median(t)), "mean_s": float(t.mean()),
                "p99_s": float(np.percentile(t, 99)), "max_s": float(t.max())}


class SphAdjointModel:
    """PF forward model on the adjoint operator with the GaussianPlume interface (section 5 step 4).

    ``unit_response(source_xy (N, 2), drone_xyz (M, 3)) -> (N, M)``: for each drone m the sensitivity field
    psi_m = operator.solve_adjoint(drone_xy) (LRU cache of ``max_cached`` fields keyed by the receptor position
    rounded to ``round_m``; the solve itself uses the rounded position so a key fully determines its field) is
    bilinearly interpolated at the N hypotheses; the result is floored at ``g_floor`` (config.FWD_G_FLOOR) like
    GaussianPlume so the RBPF likelihood stays finite; ``log_unit_response`` = log of that.
    """

    def __init__(self, operator: AdvectionDiffusionOperator, max_cached: int = config.ADJ_MAX_CACHED,
                 g_floor: float = config.FWD_G_FLOOR, round_m: float = config.ADJ_CACHE_ROUND_M) -> None:
        if max_cached < 1 or g_floor <= 0.0 or round_m <= 0.0:
            raise ValueError("need max_cached >= 1, g_floor > 0, round_m > 0")
        self.operator = operator
        self.max_cached = int(max_cached)
        self.g_floor = float(g_floor)
        self.round_m = float(round_m)
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        operator._require_lu()

    def _key(self, xy: np.ndarray) -> tuple[int, int]:
        return int(np.round(xy[0] / self.round_m)), int(np.round(xy[1] / self.round_m))

    def psi(self, drone_xy: np.ndarray) -> np.ndarray:
        """Cached sensitivity field (ny, nx) of one drone position (rounded to round_m); read-only."""
        key = self._key(_as_xy(drone_xy)[0])
        f = self._cache.get(key)
        if f is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return f
        f = self.operator.solve_adjoint(np.array([key[0] * self.round_m, key[1] * self.round_m]))
        f.setflags(write=False)
        self._cache[key] = f
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        self.misses += 1
        return f

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def n_cached(self) -> int:
        return len(self._cache)

    def unit_response(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> np.ndarray:
        """g (N, M) in (particles/m^3) per (particle/s), floored at g_floor."""
        src = _as_xy(source_xy)
        drn = np.asarray(drone_xyz, dtype=np.float64)
        if drn.ndim == 1:
            drn = drn.reshape(1, -1)
        if drn.ndim != 2 or drn.shape[1] != 3:
            raise ValueError(f"drone_xyz must have shape (M, 3) or (3,), got {drn.shape}")
        out = np.empty((src.shape[0], drn.shape[0]))
        for m in range(drn.shape[0]):
            out[:, m] = self.operator.interpolate(self.psi(drn[m, :2]), src)
        return np.maximum(out, self.g_floor)

    def log_unit_response(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> np.ndarray:
        """log g (N, M) = log(max(g, g_floor))."""
        return np.log(self.unit_response(source_xy, drone_xyz))

LbmAdjointModel = SphAdjointModel      # backward-compatible alias (the flow solver is SPH, not LBM; renamed 2026-10-06)
