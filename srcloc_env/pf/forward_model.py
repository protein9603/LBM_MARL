"""Analytic forward model for the particle filter: Gaussian plume with ground reflection (plan 4.2).

Two hypothesis -> response models share one interface (``unit_response`` / ``log_unit_response`` returning
(N, M) arrays for N source hypotheses and M drone positions):

``GaussianPlume``  (plan 4.2, references R13 Gaussian plume, R15 Taylor lateral spread)
    Wind-aligned coordinates: the vector from the hypothesis theta = (x_s, y_s) to the drone is decomposed
    along the wind unit vector e = (cos, sin)(wind_dir_deg) into the downwind distance d and the crosswind
    distance c (plan 4.2 "바람 정렬 좌표").  The default wind is +x (report 3.3); rotation augmentation changes
    wind_dir_deg, y-reflection augmentation keeps +x (the environment mirrors the source/drone y instead).

    Unit response, in (particles/m^3) per (particle/s) of release, ground reflection included (R13):

        g(theta, p) = 1 / (2 pi U sigma_y sigma_z) * exp(-c^2 / (2 sigma_y^2))
                      * [exp(-(z - z_s)^2 / (2 sigma_z^2)) + exp(-(z + z_s)^2 / (2 sigma_z^2))]

    with g = g_floor where d <= 0 (upwind: the plume model has no mass there).  Integrated over the crosswind
    line and the half space z >= 0 the response is exactly 1/U, i.e. the release rate is conserved (the image
    term folds the below-ground half of the Gaussian back into z >= 0).

    Lateral spread (plan 4.2, design choice, R15):
        sigma_y(d)^2 = sigma0^2 + (sigma_v d / U)^2 / (1 + d / (2 U T_L)),   sigma_z = sigma_z_ratio * sigma_y
    sigma0 is the volume-source std (report 2.6), sigma_v = sqrt(2k/3) the turbulent velocity scale and T_L =
    k/epsilon the Lagrangian time scale (report 2.1).  With travel time t = d/U this interpolates Taylor's (1921)
    two regimes sigma_y^2 ~ sigma_v^2 t^2 (t << T_L) and sigma_y^2 ~ 2 sigma_v^2 T_L t (t >> T_L) with a single
    algebraic form; the exact Taylor result 2 sigma_v^2 T_L^2 (t/T_L - 1 + exp(-t/T_L)) has the same limits.

    g is floored at g_floor everywhere (not only upwind) so that log g is finite for the PF; in the PF
    likelihood lambda = (kappa g + b) T the floor is invisible (kappa * 1e-9 << b = 20 cps, plan 4.3).

    Wind modes (plan S1 보강 / 4.2, option A, 2026-09-30):
      ``wind_mode='global'``  one wind for every hypothesis: params.U and params.wind_dir_deg (D3 behaviour).
      ``wind_mode='local'``   the LBM wind AT EACH HYPOTHESIS: (u_i, v_i) = wind_field.uv_at(theta_i, z = wind_z)
        (bilinear, dead-node aware; field/wind.py, report 3.6).  The direction e_i = (u_i, v_i)/|(u_i, v_i)|
        and the speed U_i = max(|(u_i, v_i)|, config.FWD_U_MIN) replace e and U per source; d_i, c_i,
        sigma_y(d_i; U_i) and g follow exactly the formulas above, vectorised over N.  This is the textbook
        convention that U in the plume formula is the wind at the effective release point (R13, Seinfeld &
        Pandis ch. 18), evaluated here on the LBM field instead of a domain mean, so that a hypothesis in a
        channelled street or a courtyard is advected the way the LBM flow actually goes.  Below FWD_U_MIN the
        plume model is meaningless (travel time -> infinity, the LDM plume of source 110 is trapped, report
        2.6) and the robust likelihood mixture (config.PF_EPS_MIX_TRAPPED, D4-3) must carry the hypothesis.
        Dead node (all four bracketing lattice nodes inside a building at wind_z -> (u, v) = (0, 0)) or any
        zero vector: U_i = FWD_U_MIN and the direction FALLS BACK to the global params.wind_dir_deg, so the
        hypothesis is still evaluated (a floor-response would otherwise silently kill it).  Optional blend:
        params.local_wind_blend in [0, 1] mixes the local and global vectors, vec_i = blend * (u_i, v_i) +
        (1 - blend) * U e; default 1 (pure local).  ``local_wind`` exposes the per-source (U_i, dir_i).

``LibraryModel``  (plan 4.2 "라이브러리 모델")
    Candidates are the 13 real sources (config.ALL_SOURCES / SOURCES_XY); the response of candidate j at the
    drone is the cached LDM slab of source j at the current frame (field/concentration_field.py).  NOTE: the
    slab is a density for the ACTUAL release rate of the simulation (6.665 particles per index step, report
    2.2), not a per-unit-q response, so the PF's kappa for this model absorbs only k * scale (plan 4.2), not
    k * scale * q.  It exists for upper-bound performance and model-mismatch quantification only.

Vectorisation: every method is pure numpy broadcasting over (N, 1) x (1, M); the D3 target is < 1 ms for
N = config.PF_N_PARTICLES = 2000 hypotheses x M = 2 drones (scripts/validate_forward.py).  The local mode adds
one bilinear lookup of N points per call (scripts/validate_forward_local.py reports the cost).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from srcloc_env import config
from srcloc_env.field.concentration_field import FieldBackend
from srcloc_env.field.wind import WindField


@dataclass(frozen=True)
class ForwardParams:
    """Parameters of the analytic plume (plan 4.2 ForwardParams; defaults from config, provenance there).

    U             mean advection speed [m/s]           (config.FWD_DEFAULT_U, candidates FWD_U_CANDIDATES)
    sigma_v       turbulent velocity scale [m/s]       (config.FWD_DEFAULT_SIGMA_V = sqrt(2k/3))
    z_s           release height [m]                   (config.SOURCE_Z, report 2.6)
    sigma0        initial (volume-source) std [m]      (config.RELEASE_SIGMA_XY, report 2.6)
    T_L           Lagrangian time scale [s]            (config.FWD_T_L = k/epsilon, report 2.1)
    sigma_z_ratio sigma_z / sigma_y                    (config.FWD_SIGMA_Z_RATIO, plan 4.2 선택)
    g_floor       response where d <= 0 and lower clip (config.FWD_G_FLOOR, plan 4.2)
    wind_dir_deg  wind direction [deg, CCW from +x]    (0 = +x; rotation augmentation changes it)
    local_wind_blend  weight of the local LBM vector in wind_mode='local' (config.FWD_LOCAL_WIND_BLEND = 1:
                  pure local; 0 reproduces the global wind) [plan S1 보강]; ignored in wind_mode='global'.
    """

    U: float = config.FWD_DEFAULT_U
    sigma_v: float = config.FWD_DEFAULT_SIGMA_V
    z_s: float = config.SOURCE_Z
    sigma0: float = config.RELEASE_SIGMA_XY
    T_L: float = config.FWD_T_L
    sigma_z_ratio: float = config.FWD_SIGMA_Z_RATIO
    g_floor: float = config.FWD_G_FLOOR
    wind_dir_deg: float = 0.0
    local_wind_blend: float = config.FWD_LOCAL_WIND_BLEND

    def __post_init__(self) -> None:
        if self.U <= 0.0 or self.sigma_v < 0.0 or self.sigma0 <= 0.0 or self.T_L <= 0.0:
            raise ValueError("ForwardParams: need U > 0, sigma_v >= 0, sigma0 > 0, T_L > 0")
        if self.sigma_z_ratio <= 0.0 or self.g_floor <= 0.0:
            raise ValueError("ForwardParams: need sigma_z_ratio > 0 and g_floor > 0")
        if not 0.0 <= self.local_wind_blend <= 1.0:
            raise ValueError("ForwardParams: need 0 <= local_wind_blend <= 1")


@dataclass(frozen=True)
class LocalWind:
    """Per-hypothesis wind of GaussianPlume.local_wind (plan S1 보강); all arrays are over the N sources.

    uv        (N, 2) raw LBM (u, v) at the hypothesis and wind_z [m/s] (wind_field.uv_at; (0, 0) on dead nodes);
              in wind_mode='global' the global vector U (cos, sin)(wind_dir_deg) is repeated.
    vec       (N, 2) blended vector blend * uv + (1 - blend) * U_global e_global.
    speed     (N,)   |vec| before the clip [m/s].
    U         (N,)   advection speed used in g: max(speed, config.FWD_U_MIN).
    dir_deg   (N,)   advection direction used in g [deg, CCW from +x]; = params.wind_dir_deg where fallback.
    fallback  (N,)   True where speed == 0 (dead node / calm) and the direction fell back to the global one.
    """

    uv: np.ndarray
    vec: np.ndarray
    speed: np.ndarray
    U: np.ndarray
    dir_deg: np.ndarray
    fallback: np.ndarray

    @property
    def e(self) -> np.ndarray:
        """(N, 2) unit advection vectors (cos, sin)(dir_deg)."""
        a = np.deg2rad(self.dir_deg)
        return np.column_stack([np.cos(a), np.sin(a)])


def _as_points(a: np.ndarray, ncol: int, name: str) -> np.ndarray:
    """(n, ncol) float64 view of a (ncol,) or (n, ncol) array; ValueError otherwise."""
    p = np.asarray(a, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.ndim != 2 or p.shape[1] != ncol:
        raise ValueError(f"{name} must have shape (n, {ncol}) or ({ncol},), got {p.shape}")
    return p


def wind_aligned_coords(source_xy: np.ndarray, drone_xy: np.ndarray,
                        wind_dir_deg: float | np.ndarray = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Downwind distance d and crosswind distance c, both (N, M), of M drones relative to N sources.

    r = drone - source is projected on the wind unit vector e = (cos a, sin a), a = wind_dir_deg (CCW from
    +x), and on its left-hand normal e_perp = (-sin a, cos a): d = r . e, c = r . e_perp (plan 4.2).  With
    the default a = 0 this is simply d = dx, c = dy; y-reflection keeps a = 0 (the environment mirrors the
    positions), rotation augmentation passes the rotated wind angle.  ``wind_dir_deg`` may also be an (N,)
    array of per-source angles (wind_mode='local', plan S1 보강): row i then uses its own e_i.
    """
    s = _as_points(source_xy, 2, "source_xy")
    p = _as_points(drone_xy, 2, "drone_xy")
    a = np.deg2rad(np.asarray(wind_dir_deg, dtype=np.float64))
    if a.ndim == 1:
        if a.shape[0] != s.shape[0]:
            raise ValueError(f"wind_dir_deg must be a scalar or have shape (N={s.shape[0]},), got {a.shape}")
        a = a[:, None]                                # (N, 1)
    elif a.ndim != 0:
        raise ValueError(f"wind_dir_deg must be a scalar or an (N,) array, got shape {a.shape}")
    ca, sa = np.cos(a), np.sin(a)
    dx = p[None, :, 0] - s[:, None, 0]          # (N, M)
    dy = p[None, :, 1] - s[:, None, 1]
    d = dx * ca + dy * sa
    c = -dx * sa + dy * ca
    return d, c


class GaussianPlume:
    """Gaussian plume with ground reflection and Taylor-type lateral spread (plan 4.2, R13, R15).

    ``unit_response(source_xy (N,2), drone_xyz (M,3)) -> g (N, M)`` in (particles/m^3) per (particle/s);
    ``log_unit_response`` returns log g computed in the log domain (no underflow before the floor).

    wind_mode 'global' (default, params.U / params.wind_dir_deg for every hypothesis) or 'local' (LBM wind at
    each hypothesis from ``wind_field``, plan S1 보강; see the module docstring and ``local_wind``).
    ``wind_z`` is the lookup height of the local wind (config.FWD_LOCAL_WIND_Z = the 15 m drone slab).
    """

    def __init__(self, params: ForwardParams | None = None, wind_field: WindField | None = None,
                 wind_mode: str = "global", wind_z: float = config.FWD_LOCAL_WIND_Z):
        self.params = params if params is not None else ForwardParams()
        if wind_mode not in config.FWD_WIND_MODES:
            raise ValueError(f"wind_mode must be one of {config.FWD_WIND_MODES}, got {wind_mode!r}")
        if wind_mode == "local" and wind_field is None:
            raise ValueError("wind_mode='local' needs a WindField (field/wind.py)")
        self.wind_field = wind_field
        self.wind_mode = wind_mode
        self.wind_z = float(wind_z)

    # ------------------------------------------------------------------ wind
    def local_wind(self, source_xy: np.ndarray) -> LocalWind:
        """Per-hypothesis advection speed U_i and direction dir_i (plan S1 보강); see ``LocalWind``.

        wind_mode='local': (u_i, v_i) = wind_field.uv_at(theta_i, wind_z); vec_i = blend (u_i, v_i) +
        (1 - blend) U (cos, sin)(wind_dir_deg); U_i = max(|vec_i|, config.FWD_U_MIN); dir_i = atan2(vec_i)
        where |vec_i| > 0, else the global wind_dir_deg (dead node / calm fallback, documented in the module).
        wind_mode='global': the global (U, wind_dir_deg) repeated N times (uv = U e, no fallback).
        """
        p = self.params
        s = _as_points(source_xy, 2, "source_xy")
        n = s.shape[0]
        a = np.deg2rad(p.wind_dir_deg)
        e_global = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
        if self.wind_mode == "global":
            uv = np.broadcast_to(p.U * e_global, (n, 2)).copy()
            return LocalWind(uv=uv, vec=uv.copy(), speed=np.full(n, float(p.U)), U=np.full(n, float(p.U)),
                             dir_deg=np.full(n, float(p.wind_dir_deg)), fallback=np.zeros(n, dtype=bool))
        uv = self.wind_field.uv_at(s, self.wind_z)                                   # (N, 2), (0, 0) on dead nodes
        b = float(p.local_wind_blend)
        vec = uv if b == 1.0 else b * uv + (1.0 - b) * (p.U * e_global)[None, :]
        speed = np.hypot(vec[:, 0], vec[:, 1])
        fallback = speed <= 0.0
        dir_deg = np.where(fallback, float(p.wind_dir_deg), np.degrees(np.arctan2(vec[:, 1], vec[:, 0])))
        U = np.maximum(speed, config.FWD_U_MIN)
        return LocalWind(uv=uv, vec=np.array(vec, dtype=np.float64), speed=speed, U=U, dir_deg=dir_deg,
                         fallback=fallback)

    # ------------------------------------------------------------------ spread
    def sigma_y(self, d: np.ndarray, U: float | np.ndarray | None = None) -> np.ndarray:
        """Lateral std [m] at downwind distance d (array); for d <= 0 the value at d = 0 (sigma0) is returned.

        ``U`` (scalar or array broadcastable with d, e.g. (N, 1) per-source speeds of wind_mode='local')
        defaults to params.U.
        """
        p = self.params
        u = p.U if U is None else np.asarray(U, dtype=np.float64)
        dd = np.maximum(np.asarray(d, dtype=np.float64), 0.0)
        var = p.sigma0 ** 2 + (p.sigma_v * dd / u) ** 2 / (1.0 + dd / (2.0 * u * p.T_L))
        return np.sqrt(var)

    def sigma_z(self, d: np.ndarray, U: float | np.ndarray | None = None) -> np.ndarray:
        """Vertical std [m] = sigma_z_ratio * sigma_y(d) (plan 4.2 선택)."""
        return self.params.sigma_z_ratio * self.sigma_y(d, U)

    # ------------------------------------------------------------------ responses
    def _log_g_unfloored(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(log g without floor, d) both (N, M); log g is meaningless where d <= 0 (callers mask it)."""
        p = self.params
        src = _as_points(source_xy, 2, "source_xy")
        drone = _as_points(drone_xyz, 3, "drone_xyz")
        if self.wind_mode == "local":
            lw = self.local_wind(src)
            u = lw.U[:, None]                                            # (N, 1) per-source speed
            d, c = wind_aligned_coords(src, drone[:, :2], lw.dir_deg)   # per-source direction
        else:
            u = p.U
            d, c = wind_aligned_coords(src, drone[:, :2], p.wind_dir_deg)
        sy = self.sigma_y(d, u)
        sz = p.sigma_z_ratio * sy
        z = drone[None, :, 2]                                            # (1, M)
        inv2sz2 = 0.5 / (sz * sz)
        vert = np.logaddexp(-(z - p.z_s) ** 2 * inv2sz2, -(z + p.z_s) ** 2 * inv2sz2)
        log_g = -np.log(2.0 * np.pi * u * sy * sz) - 0.5 * (c * c) / (sy * sy) + vert
        return log_g, d

    def log_unit_response(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> np.ndarray:
        """log g (N, M); equals log(g_floor) where d <= 0 or where g would fall below g_floor."""
        log_g, d = self._log_g_unfloored(source_xy, drone_xyz)
        log_floor = np.log(self.params.g_floor)
        out = np.maximum(log_g, log_floor)
        out[d <= 0.0] = log_floor
        return out

    def unit_response(self, source_xy: np.ndarray, drone_xyz: np.ndarray) -> np.ndarray:
        """g (N, M) in (particles/m^3) per (particle/s); exactly g_floor where d <= 0 (and as a lower clip)."""
        log_g, d = self._log_g_unfloored(source_xy, drone_xyz)
        g = np.maximum(np.exp(log_g), self.params.g_floor)
        g[d <= 0.0] = self.params.g_floor
        return g


class LibraryModel:
    """Library forward model: candidate j's response = cached LDM slab of source j (plan 4.2 라이브러리 모델).

    ``unit_response(drone_xyz (M,3)) -> (13, M)`` = backend.density([s], xy, frame_index, z, scale) per source
    s in config.ALL_SOURCES.  These are densities for the ACTUAL release rate of the simulation (config.
    PARTICLES_PER_INDEX_STEP_PER_SOURCE per index step), NOT per unit q: the PF's kappa for this model absorbs
    only k * scale (plan 4.2).  The drone z column is ignored; the slab level ``z`` given at construction is
    used (the backend stores exact levels only, no vertical interpolation).  flip_y is forwarded to the backend
    for the y-reflection augmentation (the candidate positions are then mirrored too, see ``candidates_xy``).
    """

    def __init__(self, backend: FieldBackend, frame_index: int, z: float = config.DRONE_Z, scale: float = 1.0,
                 candidates: tuple[int, ...] = config.ALL_SOURCES, g_floor: float = config.FWD_G_FLOOR):
        self.backend = backend
        self.frame_index = int(frame_index)
        self.z = float(z)
        self.scale = float(scale)
        self.candidates = tuple(int(s) for s in candidates)
        self.g_floor = float(g_floor)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    def candidates_xy(self, flip_y: bool = False) -> np.ndarray:
        """(n_candidates, 2) source positions (config.SOURCES_XY, report 2.6), y mirrored when flip_y."""
        xy = np.array([config.SOURCES_XY[s] for s in self.candidates], dtype=np.float64)
        if flip_y:
            xy = xy * np.array([1.0, -1.0])
        return xy

    def unit_response(self, drone_xyz: np.ndarray, flip_y: bool = False) -> np.ndarray:
        """(n_candidates, M) slab density [particles/m^3] of each candidate source at the M drone positions."""
        drone = _as_points(drone_xyz, 3, "drone_xyz")
        xy = drone[:, :2]
        out = np.empty((len(self.candidates), xy.shape[0]), dtype=np.float64)
        for j, s in enumerate(self.candidates):
            out[j] = self.backend.density([s], xy, self.frame_index, self.z, self.scale, flip_y)
        return out

    def log_unit_response(self, drone_xyz: np.ndarray, flip_y: bool = False) -> np.ndarray:
        """log of unit_response floored at g_floor (same convention as GaussianPlume)."""
        return np.log(np.maximum(self.unit_response(drone_xyz, flip_y), self.g_floor))