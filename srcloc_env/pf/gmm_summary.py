"""GMM summary of the particle-filter belief for the RL observation (plan 4.4; reference R7).

Park, Ladosz & Oh (2022, R7) feed a Gaussian-mixture summary of the PF belief to a DRL policy. Here the
belief is the weighted 2-D particle cloud of RBPF (positions + normalised weights); we fit K components
by weighted EM (initialised with weighted k-means++), sort them by weight (permutation consistency), zero-pad
and mask components below GMM_MIN_WEIGHT, merge near-identical components (Bhattacharyya distance < GMM_MERGE_BHAT, moment-preserving), and emit the fixed-length vector of plan 4.4:
    K x (w, mux/1315, muy/657.5, sxx/1e4, syy/1e4, sxy/1e4)  +  K mask flags  +  K x (mu - p_drone)/1000
= 27 numbers for K = 3. Fidelity of the summary is checked with the total-variation distance between the
weighted particle histogram and the GMM density on 10 m cells (T1-5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from srcloc_env import config


@dataclass(frozen=True)
class GmmSummary:
    weights: np.ndarray   # (K,) sorted descending, zero-padded
    means: np.ndarray     # (K, 2) m
    covs: np.ndarray      # (K, 2, 2) m^2
    mask: np.ndarray      # (K,) bool: component is valid (weight >= GMM_MIN_WEIGHT)

    @property
    def k(self) -> int:
        return int(self.weights.shape[0])

    def top(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance of the highest-weight component."""
        return self.means[0], self.covs[0]

    def top_sigma(self) -> float:
        """sqrt(max(sxx, syy)) of the top component - the success criterion's 'sigma' (plan 4.5)."""
        c = self.covs[0]
        return float(np.sqrt(max(c[0, 0], c[1, 1])))

    def to_vector(self, drone_xy: np.ndarray | None = None) -> np.ndarray:
        """Fixed-length observation vector (plan 4.4): K*6 params + K mask + K*2 relative means."""
        nx, ny = config.GMM_NORM_XY
        params = np.zeros((self.k, 6))
        params[:, 0] = self.weights
        params[:, 1] = self.means[:, 0] / nx
        params[:, 2] = self.means[:, 1] / ny
        params[:, 3] = self.covs[:, 0, 0] / config.GMM_NORM_COV_M2
        params[:, 4] = self.covs[:, 1, 1] / config.GMM_NORM_COV_M2
        params[:, 5] = self.covs[:, 0, 1] / config.GMM_NORM_COV_M2
        params[~self.mask] = 0.0
        rel = np.zeros((self.k, 2))
        if drone_xy is not None:
            rel = (self.means - np.asarray(drone_xy, dtype=float)[None, :]) / config.GMM_NORM_REL_M
            rel[~self.mask] = 0.0
        return np.concatenate([params.ravel(), self.mask.astype(float), rel.ravel()])

    def density(self, xy: np.ndarray) -> np.ndarray:
        """Mixture density [1/m^2] at points xy (n, 2) over the valid components."""
        xy = np.asarray(xy, dtype=float)
        out = np.zeros(xy.shape[0])
        for w, m, c, ok in zip(self.weights, self.means, self.covs, self.mask):
            if not ok:
                continue
            inv = np.linalg.inv(c)
            d = xy - m
            q = np.einsum("ni,ij,nj->n", d, inv, d)
            out += w * np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(np.linalg.det(c)))
        return out


def _weighted_kmeanspp(xy: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centres = [xy[rng.choice(xy.shape[0], p=w)]]
    for _ in range(1, k):
        d2 = np.min(((xy[:, None, :] - np.asarray(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        p = w * d2
        if p.sum() <= 0:                       # all mass on the existing centres
            centres.append(xy[rng.choice(xy.shape[0], p=w)])
            continue
        centres.append(xy[rng.choice(xy.shape[0], p=p / p.sum())])
    return np.asarray(centres, dtype=float)


def _bhattacharyya(m1, c1, m2, c2) -> float:
    cb = 0.5 * (c1 + c2)
    d = m1 - m2
    return float(0.125 * d @ np.linalg.solve(cb, d) + 0.5 * np.log(np.linalg.det(cb) / np.sqrt(np.linalg.det(c1) * np.linalg.det(c2))))


def _merge_overlapping(pis: np.ndarray, means: np.ndarray, covs: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge component pairs whose Bhattacharyya distance is below `threshold` (moment-preserving).

    EM with K components on a belief with fewer modes splits a mode into near-identical components whose
    weight order flips from step to step; merging them keeps the summary permutation-stable (plan 4.4).
    """
    pis, means, covs = list(pis), list(means), list(covs)
    merged = True
    while merged and len(pis) > 1:
        merged = False
        best = None
        for i in range(len(pis)):
            for j in range(i + 1, len(pis)):
                b = _bhattacharyya(means[i], covs[i], means[j], covs[j])
                if b < threshold and (best is None or b < best[0]):
                    best = (b, i, j)
        if best is not None:
            _, i, j = best
            w = pis[i] + pis[j]
            m = (pis[i] * means[i] + pis[j] * means[j]) / w
            c = (pis[i] * (covs[i] + np.outer(means[i], means[i])) + pis[j] * (covs[j] + np.outer(means[j], means[j]))) / w - np.outer(m, m)
            pis[i], means[i], covs[i] = w, m, c
            del pis[j], means[j], covs[j]
            merged = True
    k = len(pis)
    return np.asarray(pis), np.asarray(means).reshape(k, 2), np.asarray(covs).reshape(k, 2, 2)


def fit_weighted_gmm(xy: np.ndarray, w: np.ndarray, k: int = config.GMM_K, n_iter: int = config.GMM_EM_ITERS,
                     rng: np.random.Generator | None = None) -> GmmSummary:
    """Weighted EM for a K-component 2-D Gaussian mixture on weighted particles (weights sum to 1).

    Responsibilities are multiplied by the particle weights, so the fit represents the weighted belief.
    Covariances get a diagonal floor GMM_COV_REG_M2. Components are sorted by weight (descending);
    components lighter than GMM_MIN_WEIGHT are zeroed and masked out.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    xy = np.asarray(xy, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    n = xy.shape[0]
    means = _weighted_kmeanspp(xy, w, k, rng)
    covs = np.stack([np.eye(2) * max(np.var(xy, axis=0).mean(), config.GMM_COV_REG_M2)] * k)
    pis = np.full(k, 1.0 / k)
    reg = np.eye(2) * config.GMM_COV_REG_M2
    for _ in range(n_iter):
        # E-step: log responsibilities
        logr = np.empty((n, k))
        for j in range(k):
            inv = np.linalg.inv(covs[j])
            d = xy - means[j]
            q = np.einsum("ni,ij,nj->n", d, inv, d)
            logr[:, j] = np.log(pis[j] + 1e-300) - 0.5 * q - 0.5 * np.log(np.linalg.det(covs[j]))
        logr -= logr.max(axis=1, keepdims=True)
        r = np.exp(logr)
        r /= r.sum(axis=1, keepdims=True)
        rw = r * w[:, None]                         # weighted responsibilities
        nk = rw.sum(axis=0)                         # component masses (sum = 1)
        # M-step
        for j in range(k):
            if nk[j] <= 1e-12:
                means[j] = xy[rng.choice(n, p=w)]
                covs[j] = np.eye(2) * config.GMM_COV_REG_M2 * 100.0
                continue
            means[j] = (rw[:, j] @ xy) / nk[j]
            d = xy - means[j]
            covs[j] = (rw[:, j, None] * d).T @ d / nk[j] + reg
        pis = np.maximum(nk, 1e-12)
        pis /= pis.sum()
    pis, means, covs = _merge_overlapping(pis, means, covs, config.GMM_MERGE_BHAT)
    if pis.shape[0] < k:                          # zero-pad merged-away slots
        pad = k - pis.shape[0]
        pis = np.concatenate([pis, np.zeros(pad)])
        means = np.vstack([means, np.zeros((pad, 2))])
        covs = np.concatenate([covs, np.stack([np.eye(2) * config.GMM_COV_REG_M2] * pad)])
    order = np.argsort(-pis)
    pis, means, covs = pis[order], means[order], covs[order]
    mask = pis >= config.GMM_MIN_WEIGHT
    pis = np.where(mask, pis, 0.0)
    if mask.any():
        pis = pis / pis.sum()
    return GmmSummary(weights=pis, means=means, covs=covs, mask=mask)


def summarise_pf(pf, k: int = config.GMM_K, rng: np.random.Generator | None = None) -> GmmSummary:
    """Fit the GMM to an RBPF's current particles (uses pf.xy and normalised exp(pf.logw))."""
    w = np.exp(pf.logw - np.max(pf.logw))
    return fit_weighted_gmm(pf.xy, w / w.sum(), k=k, rng=rng)


def total_variation_distance(xy: np.ndarray, w: np.ndarray, gmm: GmmSummary, cell: float = config.T1_5_TV_CELL_M,
                             bounds: tuple[tuple[float, float], tuple[float, float]] | None = None) -> float:
    """TV distance between the weighted particle histogram and the GMM mass on `cell`-sized cells (T1-5)."""
    xy = np.asarray(xy, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    if bounds is None:
        pad = 3.0 * cell
        bounds = ((xy[:, 0].min() - pad, xy[:, 0].max() + pad), (xy[:, 1].min() - pad, xy[:, 1].max() + pad))
    (x0, x1), (y0, y1) = bounds
    nx = int(np.ceil((x1 - x0) / cell)); ny = int(np.ceil((y1 - y0) / cell))
    ix = np.clip(((xy[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    hist = np.zeros((nx, ny)); np.add.at(hist, (ix, iy), w)
    cx = x0 + cell * (np.arange(nx) + 0.5); cy = y0 + cell * (np.arange(ny) + 0.5)
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    dens = gmm.density(np.column_stack([gx.ravel(), gy.ravel()])).reshape(nx, ny) * cell * cell
    dens = dens / max(dens.sum(), 1e-300)
    return float(0.5 * np.abs(hist - dens).sum())


def order_flip_rate(prev: GmmSummary, curr: GmmSummary, match_radius: float = 30.0) -> float:
    """Fraction of valid components whose rank changed between two consecutive summaries.

    Components are matched by nearest mean (within match_radius) and the rank order compared; unmatched
    components count as flips. Used for the T1-5 ordering-stability statistic.
    """
    kp = int(prev.mask.sum()); kc = int(curr.mask.sum())
    if kp == 0 or kc == 0:
        return 0.0
    flips = 0
    for i in range(kc):
        if not curr.mask[i]:
            continue
        d = np.linalg.norm(prev.means[prev.mask] - curr.means[i], axis=1)
        j = int(np.argmin(d))
        if d[j] > match_radius or j != i:
            flips += 1
    return flips / kc