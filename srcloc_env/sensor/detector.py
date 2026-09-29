"""Poisson counting sensor model for the drone detector (plan 4.1, `sensor/detector.py`).

Model (R3, Hutchinson et al. 2018 / Ristic et al. 2016: y ~ Poisson((k*C + b)*T)):
    rate     r(p) = k0 * scale * n(p) + b                       [cps]
    counts   y    ~ Poisson(r(p) * T)                             [counts per integration time T]
where n(p) is the airborne particle number density [particles/m^3] delivered by the field backend
(LdmSlabBackend.density, plan 4.1: scenario sources, airborne, x < X_OUTFLOW, z = 15 m slab), k0 the
placeholder sensitivity config.SENSOR_K0 (~1e3 cps at concn 600 = 38.4 particles/m^3, plan 4.1 [계산]),
scale the per-episode log-uniform factor on k0 (config.SENSOR_SCALE_RANGE), b the background
config.SENSOR_BACKGROUND_CPS and T = config.SENSOR_T the integration time (= one RL step).

Detection flag (R4, Currie 1968 decision threshold): r > b + k*sqrt(b*T)/T with k = config.SENSOR_CURRIE_K
(plan 4.1: 20 + 13.4 ~ 33 cps for b = 20, T = 1).

Policy input (plan 4.1): log1p(y) / log1p(y_max), y_max = config.SENSOR_Y_MAX ~ 3,050 counts, clipped to [0, 1].

Per-source truth vector (plan 4.1 item 5 / "로그"): measure_per_source draws ONE Poisson count from the
summed rate and returns alongside it the per-source expected counts (n, n_src) for logging and for the PF
oracle likelihood; nothing is drawn per source because the physical detector sees the sum.

PF likelihood helper: expected_counts_from_kappa(kappa, g) = (kappa*g + b)*T with kappa = k*q the product of
sensitivity and emission rate marginalised by the RB-PF on a log grid (plan 4.3) and g the q = 1 unit
response of the forward model (plan 4.2).  kappa and g broadcast with numpy rules, e.g. kappa (G,) with
g (N, 1) -> (N, G).

All constants come from srcloc_env.config; every function is vectorised over arbitrary input shapes.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from srcloc_env import config

ArrayLike = float | Sequence[float] | np.ndarray


def expected_counts_from_kappa(kappa: ArrayLike, g: ArrayLike, b: float = config.SENSOR_BACKGROUND_CPS,
                               T: float = config.SENSOR_T) -> np.ndarray:
    """Expected counts (kappa * g + b) * T of the PF likelihood (plan 4.3, R3).

    kappa: effective scale k*q [cps per (particle/m^3 per particle/s)], any shape (e.g. the (G,) log grid).
    g:     unit response [particles/m^3 per particle/s] of the forward model, any shape broadcastable with
           kappa (e.g. (N, 1) particles against a (G,) grid -> (N, G)).
    Returns float64 with the broadcast shape.
    """
    return (np.asarray(kappa, dtype=np.float64) * np.asarray(g, dtype=np.float64) + float(b)) * float(T)


class Detector:
    """Poisson counting detector y ~ Poisson((k0*scale*n + b)*T) (plan 4.1; R3 sensor model, R4 threshold).

    Parameters default to config.SENSOR_K0 [cps per particle/m^3], config.SENSOR_BACKGROUND_CPS [cps] and
    config.SENSOR_T [s].  Random draws require an explicit numpy Generator so that episodes are reproducible.
    """

    def __init__(self, k0: float = config.SENSOR_K0, background: float = config.SENSOR_BACKGROUND_CPS,
                 T: float = config.SENSOR_T, y_max: float = config.SENSOR_Y_MAX,
                 currie_k: float = config.SENSOR_CURRIE_K) -> None:
        if k0 < 0.0 or background < 0.0 or T <= 0.0 or y_max <= 0.0 or currie_k < 0.0:
            raise ValueError("Detector: need k0 >= 0, background >= 0, T > 0, y_max > 0, currie_k >= 0")
        self.k0 = float(k0)
        self.background = float(background)
        self.T = float(T)
        self.y_max = float(y_max)
        self.currie_k = float(currie_k)
        self._log1p_y_max = float(np.log1p(self.y_max))

    # ------------------------------------------------------------------ rates and counts
    def expected_rate(self, density: ArrayLike, scale: ArrayLike = 1.0) -> np.ndarray:
        """Expected count rate k0*scale*density + b [cps]; vectorised over any shape (scale broadcasts)."""
        n = np.asarray(density, dtype=np.float64)
        return self.k0 * np.asarray(scale, dtype=np.float64) * n + self.background

    def expected_counts(self, density: ArrayLike, scale: ArrayLike = 1.0) -> np.ndarray:
        """Poisson mean lambda = expected_rate * T [counts per integration time]."""
        return self.expected_rate(density, scale) * self.T

    def measure(self, density: ArrayLike, scale: ArrayLike, rng: np.random.Generator) -> np.ndarray:
        """Counts y ~ Poisson((k0*scale*density + b)*T) as int64 with the broadcast shape of (density, scale).

        The plan 4.1 pseudo-code `measure(field, ...)`: the environment first queries the field backend for
        the summed density n(p) (airborne, x < X_OUTFLOW) and then calls this with that density.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("measure: rng must be a numpy.random.Generator (reproducible episodes)")
        lam = self.expected_counts(density, scale)
        return np.asarray(rng.poisson(lam), dtype=np.int64)

    def measure_per_source(self, density_per_source: ArrayLike, scale: ArrayLike,
                           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """One Poisson draw per row from the summed rate, plus the per-source expected counts (plan 4.1 item 5).

        density_per_source: (n, n_src) densities [particles/m^3] of each scenario source at n drone positions
        ((n_src,) is treated as one position).  scale: scalar or (n,).
        Returns (counts (n,) int64, expected_per_source (n, n_src) float64) where
        expected_per_source = k0*scale*density_per_source*T (background NOT included, so the row sum plus b*T
        equals the Poisson mean of the draw).  The count is drawn ONCE from the summed rate because the physical
        detector observes the total; the per-source vector is the logged "truth" for the PF oracle likelihood.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("measure_per_source: rng must be a numpy.random.Generator")
        d = np.asarray(density_per_source, dtype=np.float64)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        if d.ndim != 2:
            raise ValueError(f"density_per_source must have shape (n, n_src) or (n_src,), got {d.shape}")
        s = np.asarray(scale, dtype=np.float64)
        if s.ndim > 1 or (s.ndim == 1 and s.shape[0] not in (1, d.shape[0])):
            raise ValueError(f"scale must be scalar or (n,) = ({d.shape[0]},), got {s.shape}")
        s_col = s.reshape(-1, 1)                                     # (1,1) or (n,1)
        expected_per_source = self.k0 * s_col * d * self.T          # (n, n_src), no background
        lam_total = expected_per_source.sum(axis=1) + self.background * self.T
        counts = np.asarray(rng.poisson(lam_total), dtype=np.int64)
        return counts, expected_per_source

    # ------------------------------------------------------------------ policy input and detection
    def normalise(self, counts: ArrayLike) -> np.ndarray:
        """Policy observation log1p(y)/log1p(y_max) clipped to [0, 1] (plan 4.1); monotone in y."""
        y = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
        return np.clip(np.log1p(y) / self._log1p_y_max, 0.0, 1.0)

    def detection_threshold_cps(self) -> float:
        """Currie-type decision threshold b + k*sqrt(b*T)/T [cps] (R4; plan 4.1: ~33.4 cps for b=20, T=1)."""
        return self.background + self.currie_k * np.sqrt(self.background * self.T) / self.T

    def is_detection(self, counts: ArrayLike) -> np.ndarray:
        """Detection flag counts/T > detection_threshold_cps() (bool array with the input shape)."""
        return np.asarray(counts, dtype=np.float64) / self.T > self.detection_threshold_cps()

    # ------------------------------------------------------------------ episode scale
    @staticmethod
    def sample_scale(rng: np.random.Generator, lo: float = config.SENSOR_SCALE_RANGE[0],
                     hi: float = config.SENSOR_SCALE_RANGE[1], size: int | tuple[int, ...] | None = None) -> np.ndarray | float:
        """Per-episode sensitivity factor scale ~ log-uniform[lo, hi] (plan 4.1; median = sqrt(lo*hi))."""
        if not (0.0 < lo <= hi):
            raise ValueError(f"sample_scale: need 0 < lo <= hi, got {lo}, {hi}")
        u = rng.uniform(np.log(lo), np.log(hi), size=size)
        out = np.exp(u)
        return float(out) if size is None else out