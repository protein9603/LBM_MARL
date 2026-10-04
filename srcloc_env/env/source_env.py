"""Gymnasium single-drone source-localisation environment on the LDM slab truth with the LBM adjoint RB-PF belief
(plan 4.5 환경·보상, 4.3 PF, 4.4 GMM 요약; D8-3 / S2 T2-1, T2-2, T2-3).

One episode = one scene orientation (original or y-reflected), one true source, one sensor scale and one truth
mode; the drone flies at config.DRONE_Z with 9 discrete actions (8 headings x config.DRONE_STEP_M, stay).  Every RL
step (config.RL_STEP_SECONDS = 1 s): action -> DroneKinematics.step (building / domain masking; a masked action is a
stay) -> one count measurement y ~ NB-or-Poisson((k0 scale n_s(p) + b) T) from the slab truth at the current
frame -> RBPF.update(y, p) (LBM adjoint forward model, negative-binomial likelihood) -> GMM summary (every
``gmm_every`` steps) -> observation / reward / termination.

Observation (config.ENV_OBS_DIM = 56; plan 4.5):
    [0:27)   GmmSummary.to_vector(drone_xy): K x (w, mux/1315, muy/657.5, sxx/1e4, syy/1e4, sxy/1e4) + K mask
             + K x (mean - drone)/1000                                                 (pf/gmm_summary.py)
    [27:30)  drone x/1315, y/657.5, remaining-time fraction 1 - t/max_steps
    [30:45)  last ENV_N_RECENT = 5 measurements, most recent first, (normalised log count Detector.normalise(y),
             (x_meas - x_drone)/1000, (y_meas - y_drone)/1000), zero-padded
    [45]     min(1, steps since the last Currie detection / ENV_DETECTION_NORM_STEPS)
    [45+1:48) LBM wind (u, v) at the drone / ENV_WIND_NORM (15 m wind, field/wind.py)
    [48:56)  8-heading building distances / RAY_MAX_RANGE_M (ObstacleMap.ray_distances, E, NE, ..., SE)
Observation version v2 (obs_version="v2", config.ENV_OBS_DIM_V2 = 64; recovery plan D12): egocentric and O(1) instead of the absolute
    GMM means/covariances (which acted as episode fingerprints) and the 1000 m-scaled measurement offsets (values of order 0.004):
    [0:3) GMM weights, [3:6) valid mask, [6:9) log10(sigma_k / 10 m) with sigma_k = sqrt(max(sxx, syy)),
    [9:18) per component (unit bearing to its mean cos, sin; log10(1 + distance / 50 m)), [18:26) cos(heading_j, bearing to the TOP component),
    [26] H / H_0 (normalised belief entropy), [27:30) own position and remaining time as in v1,
    [30:45) last 5 measurements (normalised log count, dx / 25 m, dy / 25 m), [45] steps since the last detection / 50 (1.0 while none yet),
    [46:48) wind (u, v) / 10, [48:56) wind projected on the 8 headings, [56:64) 8 building distances / RAY_MAX_RANGE_M.
Reward (plan 4.5): ENV_REWARD_TIME + ENV_REWARD_INFO x (H_{t-1} - H_t) / H_0 + ENV_REWARD_EXIT x [domain exit]
    + terminal: success ENV_REWARD_SUCCESS; timeout -min(error, ENV_FAIL_ERROR_CAP_M) / ENV_FAIL_ERROR_SCALE_M.
    H = RBPF.entropy_xy (20 m cell weighted histogram), H_0 = entropy of the obstacle-aware prior.
Termination: success = GMM top sigma < ENV_SUCCESS_SIGMA_M (30 m) and |MAP - truth| < ENV_SUCCESS_ERROR_M (50 m) (checked when the GMM is
    refreshed); truncation at ``max_steps`` (config.MAX_EPISODE_STEPS = 300); domain exit terminates (cannot happen
    through DroneKinematics, kept for safety).
Reset (plan 4.5): source ~ ``sources`` (default config.TRAIN_SOURCES); Mode F frame ~ U{FRAME_RANGE_MODE_F} held for
    the episode (D8-2 decision), Mode T2 frame(t) = min(N_FILES - 1, start + t), start ~ U{FRAME_START_MODE_T2};
    scale ~ log-uniform SENSOR_SCALE_RANGE; drone start = free cell >= ENV_START_MIN_DIST_M from the source, with
    probability ENV_START_PLUME_FRAC inside the detectable plume region of the episode's source (expected counts >= the Currie
    threshold at the start, distance >= ENV_START_MIN_DIST_M; info['start_type'] = 'plume' or 'random'); y-reflection with probability
    ``reflect_prob`` (needs ``scene_reflected``, Scene.reflected()); the other 12 sources are removed (single-source
    density query).  All randomness comes from gymnasium's ``self.np_random`` (seed -> reproducible episode).
The PF likelihood uses the environment detector's (background, T) and a kappa grid centred on its k0, so a custom
    Detector keeps the belief consistent with the counts.  info['frame'] is the truth frame of info['y'] (at reset: of
    the first measurement); info['t'] = measurements taken so far (the next one uses frame_at(t)).
Determinism (T2-2): PF, detector and GMM generators are children of ``self.np_random``; the GMM initialisation uses
    its own child generator so a refresh never consumes PF randomness.
References: R3 (Poisson counts), R5 (RB-PF), R7 (GMM belief summary / policy input), R17-R18 (adjoint source-receptor),
    R22-R23 (negative-binomial likelihood), plan 4.5 (reward = information gain + time + terminal, 4.6 PPO input).
"""
from __future__ import annotations

import dataclasses

import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from srcloc_env import config
from srcloc_env.env.drone import ACTION_STAY, DroneKinematics, ObstacleMap, heading_unit_vectors
from srcloc_env.field.concentration_field import LdmSlabBackend
from srcloc_env.field.wind import WindField
from srcloc_env.field.wind_models import build_wind_field, uniform_wind_field
from srcloc_env.pf.forward_model import ForwardParams, GaussianPlume
from srcloc_env.pf.gmm_summary import GmmSummary, summarise_pf
from srcloc_env.pf.lbm_adjoint import AdjointParams, AdvectionDiffusionOperator, LbmAdjointModel
from srcloc_env.pf.particle_filter import RBPF
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.sensor.detector import Detector

ENV_MODES = tuple(config.ENV_MODES)


class TruthBackend(Protocol):
    """Slab-truth density query (LdmSlabBackend.density signature incl. flip_y)."""

    def density(self, src_ids, xy: np.ndarray, frame_index: int, z: float = config.DRONE_Z, scale: float = 1.0,
                flip_y: bool = False) -> np.ndarray: ...


# ------------------------------------------------------------------------------------------- scene
@dataclass
class Scene:
    """Heavy read-only objects shared by every episode of one orientation (original or y-reflected).

    wind / obstacles / sources_xy are mirrored in a reflected scene; the truth backend is shared and queried with
    flip_y=reflected (LdmSlabBackend samples (x, -y)); the adjoint operator is assembled on the mirrored wind and
    obstacles (one LU per orientation, lbm_forward_model.md 4)."""
    wind: WindField
    obstacles: ObstacleMap
    backend: TruthBackend
    model: LbmAdjointModel | GaussianPlume
    sources_xy: dict[int, tuple[float, float]]
    reflected: bool = False
    grid: SlabGrid = SlabGrid()
    params: AdjointParams = AdjointParams(K=config.ADJ_K_CHOSEN, lam=config.ADJ_LAMBDA_CHOSEN)
    seconds_build: float = 0.0
    wind_level: str = config.WIND_LEVEL_DEFAULT          # W0 / W1 / W2 (field/wind_models.py, plan D13)
    plume_params: ForwardParams | None = None            # W0 only: the global Gaussian plume (U / direction follow ``wind``)
    wind_u: float = config.WIND_MEAN_U                   # reference wind given to W0 / W1 (informational for W2; sensitivity runs)
    wind_dir_deg: float = config.WIND_MEAN_DIR_DEG
    obs_wind: str = "model"                              # D15: 'model' = observation wind from ``wind``; 'uniform' = from a uniform (wind_u, wind_dir_deg) field
    obs_wind_field: WindField | None = None              # the uniform field when obs_wind == 'uniform'

    def observation_wind(self) -> WindField:
        """Wind field the policy observation reads (the estimator's field, or the uniform field of obs_wind='uniform')."""
        return self.obs_wind_field if self.obs_wind_field is not None else self.wind

    @classmethod
    def build(cls, wind: WindField, obstacles: ObstacleMap, backend: TruthBackend,
              sources_xy: dict[int, tuple[float, float]] | None = None, params: AdjointParams | None = None,
              grid: SlabGrid = SlabGrid(), reflected: bool = False, max_cached: int = config.ADJ_MAX_CACHED,
              wind_level: str = config.WIND_LEVEL_DEFAULT, plume_params: ForwardParams | None = None,
              wind_u: float = config.WIND_MEAN_U, wind_dir_deg: float = config.WIND_MEAN_DIR_DEG, obs_wind: str = "model") -> "Scene":
        """Forward model of the wind-knowledge level (config.WIND_LEVEL_FORWARD): 'adjoint' (W1, W2) assembles and
        factorises the advection-diffusion operator on the given wind / obstacles (LbmAdjointModel); 'plume' (W0) is
        the global Gaussian plume whose speed and direction are read from the (uniform) wind field, so that a
        mirrored wind gives a mirrored plume."""
        t0 = time.perf_counter()
        if wind_level not in config.WIND_LEVELS:
            raise ValueError(f"wind_level must be one of {config.WIND_LEVELS}, got {wind_level!r}")
        params = params if params is not None else AdjointParams(K=config.ADJ_K_CHOSEN, lam=config.ADJ_LAMBDA_CHOSEN)
        if config.WIND_LEVEL_FORWARD[wind_level] == "plume":
            centre = np.array([[0.5 * (wind.x[0] + wind.x[-1]), 0.5 * (wind.y[0] + wind.y[-1])]])
            u, v = wind.uv_at(centre, params.z)[0]
            U, direction = float(np.hypot(u, v)), float(np.degrees(np.arctan2(v, u)))
            base = plume_params if plume_params is not None else ForwardParams(U=max(U, config.FWD_U_MIN), sigma_v=config.WIND_PLUME_SIGMA_V)
            plume_params = dataclasses.replace(base, U=max(U, config.FWD_U_MIN), wind_dir_deg=direction)
            model: LbmAdjointModel | GaussianPlume = GaussianPlume(plume_params, wind_mode="global")
            wind_u, wind_dir_deg = round(float(plume_params.U), 6), round(float(plume_params.wind_dir_deg), 6)   # what the plume uses, without float32 noise (review D14)
        else:
            op = AdvectionDiffusionOperator.from_data(params, wind, obstacles, grid=grid).factorize()
            model = LbmAdjointModel(op, max_cached=max_cached)
        src = dict(sources_xy if sources_xy is not None else config.SOURCES_XY)
        if obs_wind not in config.OBS_WIND_MODES:
            raise ValueError(f"obs_wind must be one of {config.OBS_WIND_MODES}, got {obs_wind!r}")
        obs_field = uniform_wind_field(float(wind_u), float(wind_dir_deg)) if obs_wind == "uniform" else None
        return cls(wind, obstacles, backend, model, {int(k): (float(v[0]), float(v[1])) for k, v in src.items()},
                   reflected, grid, params, time.perf_counter() - t0, wind_level, plume_params, float(wind_u), float(wind_dir_deg) + 0.0,
                   obs_wind, obs_field)

    @classmethod
    def load(cls, params: AdjointParams | None = None, backend: TruthBackend | None = None,
             max_cached_frames: int = config.FIELD_MAX_CACHED_FRAMES, wind_level: str = config.WIND_LEVEL_DEFAULT,
             wind_u: float | None = None, wind_dir_deg: float | None = None, obs_wind: str = "model") -> "Scene":
        """The real scene: the wind of the requested knowledge level (W2 WindField.load(), W1 potential flow on the
        building map, W0 uniform; field/wind_models.py), ObstacleMap.load(), LdmSlabBackend(cache) and the chosen
        adjoint combination (config.ADJ_K_CHOSEN / ADJ_LAMBDA_CHOSEN; D8-1).  The truth never depends on the wind."""
        be = backend if backend is not None else LdmSlabBackend(max_cached_frames=max_cached_frames)
        om = ObstacleMap.load()
        u = config.WIND_MEAN_U if wind_u is None else float(wind_u)                  # sensitivity runs perturb the reference wind (W0 / W1 only)
        d = config.WIND_MEAN_DIR_DEG if wind_dir_deg is None else float(wind_dir_deg)
        wf, _ = build_wind_field(wind_level, om, U=u, dir_deg=d)
        return cls.build(wf, om, be, config.SOURCES_XY, params, wind_level=wind_level, wind_u=u, wind_dir_deg=d, obs_wind=obs_wind)

    def reflected_scene(self) -> "Scene":
        """The y-mirrored scene (plan 4.5 학습 시 y 반사): wind (y -> -y, v -> -v, building arrays mirrored),
        obstacle raster mirrored exactly about y = 0 (cell centres y0 + k res <-> -(y0 + k res)), sources (x, -y),
        the adjoint SlabGrid mirrored (the default slab y range -487.5..547.5 is not symmetric); the truth backend
        is shared and queried with flip_y=True."""
        wf = self.wind
        y_ref = -wf.y[::-1]
        uvw = wf.uvw[:, :, ::-1, :].copy()
        uvw[..., 1] *= -1.0
        wind_r = WindField.from_arrays(wf.x, y_ref, wf.z_levels, uvw, wf.building_mask[:, ::-1], wf.building_height[:, ::-1])
        om = self.obstacles
        y0_r = -(om.y0 + (om.ny - 1) * om.res)
        obs_r = ObstacleMap.from_arrays(om.x0, y0_r, om.res, om.occ[:, ::-1], om.hmap[:, ::-1], om.domain_x,
                                        (-om.domain_y[1], -om.domain_y[0]))
        src_r = {s: (x, -y) for s, (x, y) in self.sources_xy.items()}
        g = self.grid                                                  # adjoint grid mirrored too (review D8-3):
        grid_r = SlabGrid(g.x0, -(g.y0 + g.ny * g.res), g.nx, g.ny, g.res)   # cell centres -> -(original centres), involutive
        return Scene.build(wind_r, obs_r, self.backend, src_r, self.params, grid_r, reflected=not self.reflected,
                           max_cached=self.model.max_cached if hasattr(self.model, "max_cached") else config.ADJ_MAX_CACHED,
                           wind_level=self.wind_level, plume_params=self.plume_params, wind_u=self.wind_u, wind_dir_deg=-self.wind_dir_deg,
                           obs_wind=self.obs_wind)


# ------------------------------------------------------------------------------------------- environment
def default_max_steps(mode: str) -> int:
    return config.T2_MAX_STEPS if mode == "T2" else config.MAX_EPISODE_STEPS


def load_scene(mode: str, wind_level: str = config.WIND_LEVEL_DEFAULT, wind_u: float | None = None,
               wind_dir_deg: float | None = None, obs_wind: str = "model") -> Scene:
    """The real scene; Mode T2 reads its truth from the memory-mapped slab stack (D11), Mode F from the per-frame
    cache; ``wind_level`` selects the estimator's wind knowledge (W0 / W1 / W2, plan D13); ``wind_u`` /
    ``wind_dir_deg`` override the reference wind of W0 / W1 (sensitivity runs)."""
    backend = None
    if mode == "T2":
        from srcloc_env.field.slab_stack import StackedSlabBackend
        backend = StackedSlabBackend()
    return Scene.load(backend=backend, wind_level=wind_level, wind_u=wind_u, wind_dir_deg=wind_dir_deg, obs_wind=obs_wind)


class SourceLocEnv(gym.Env):
    """Single-drone Gymnasium environment (module docstring).  ``scene`` is shared between envs / episodes."""

    metadata = {"render_modes": []}

    def __init__(self, scene: Scene, *, sources: Sequence[int] = config.TRAIN_SOURCES,
                 truth_mode: str = config.ENV_TRUTH_MODE_DEFAULT, scene_reflected: Scene | None = None,
                 reflect_prob: float = 0.0, n_particles: int = config.PF_N_PARTICLES,
                 gmm_every: int = config.ENV_GMM_EVERY, gmm_iters: int = config.ENV_GMM_EM_ITERS,
                 gmm_warm_start: bool = config.ENV_GMM_WARM_START, max_steps: int = config.MAX_EPISODE_STEPS,
                 detector: Detector | None = None, frame_range: tuple[int, int] = config.FRAME_RANGE_MODE_F,
                 t2_start_range: tuple[int, int] = config.FRAME_START_MODE_T2,
                 t2_files_per_step: float = config.T1_4_MODE_T2_FILES_PER_STEP,
                 scale_range: tuple[float, float] = config.SENSOR_SCALE_RANGE,
                 start_min_dist: float = config.ENV_START_MIN_DIST_M, start_max_dist: float = float("inf"),
                 obs_version: str = "v1", fail_error_cap_m: float = config.ENV_FAIL_ERROR_CAP_M, shaping: str = "none", shaping_weight: float = 1.0,
                 shaping_cap_m: float = 400.0,
                 start_plume_frac: float = config.ENV_START_PLUME_FRAC,
                 prior_x: tuple[float, float] = config.PF_PRIOR_X, prior_y: tuple[float, float] = config.PF_PRIOR_Y,
                 likelihood: str | None = None, nb_r: float = config.PF_NB_DISPERSION_R,
                 eps_mix: float = config.PF_EPS_MIX, z: float = config.DRONE_Z,
                 n_files: int = config.N_FILES, success_sigma: float = config.ENV_SUCCESS_SIGMA_M,
                 success_error: float = config.ENV_SUCCESS_ERROR_M, terminate_on_success: bool = True) -> None:
        super().__init__()
        if truth_mode not in ENV_MODES:
            raise ValueError(f"truth_mode must be one of {ENV_MODES}, got {truth_mode!r}")
        if not 0.0 <= reflect_prob <= 1.0:
            raise ValueError("reflect_prob must be in [0, 1]")
        if reflect_prob > 0.0 and scene_reflected is None:
            raise ValueError("reflect_prob > 0 needs scene_reflected (Scene.reflected_scene())")
        if gmm_every < 1 or gmm_iters < 1 or max_steps < 1 or n_particles < 1:
            raise ValueError("gmm_every, gmm_iters, max_steps and n_particles must be >= 1")
        srcs = [int(s) for s in sources]
        missing = [s for s in srcs if s not in scene.sources_xy]
        if not srcs or missing:
            raise ValueError(f"sources {srcs} must be non-empty and known to the scene (missing {missing})")
        self.scene0, self.scene_reflected = scene, scene_reflected
        self.sources = tuple(srcs)
        self.truth_mode = truth_mode
        self.reflect_prob = float(reflect_prob)
        self.n_particles = int(n_particles)
        self.gmm_every = int(gmm_every)
        self.gmm_iters = int(gmm_iters)
        self.gmm_warm_start = bool(gmm_warm_start)
        self.max_steps = int(max_steps)
        self.det = detector if detector is not None else Detector()
        if not (self.det.k0 > 0.0 and self.det.T > 0.0):
            raise ValueError("the environment detector needs k0 > 0 and T > 0 (PF kappa grid / likelihood)")
        self.frame_range = (int(frame_range[0]), int(frame_range[1]))
        self.t2_start_range = (int(t2_start_range[0]), int(t2_start_range[1]))
        self.t2_files_per_step = float(t2_files_per_step)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.start_min_dist = float(start_min_dist)
        self.start_max_dist = float(start_max_dist)          # curriculum stage A of the recovery plan: starts within [min, max] of the source; inf = no limit
        self.start_plume_frac = float(start_plume_frac)
        self.start_type = "random"
        self.prior_x, self.prior_y = (float(prior_x[0]), float(prior_x[1])), (float(prior_y[0]), float(prior_y[1]))
        self.likelihood, self.nb_r, self.eps_mix = likelihood, float(nb_r), float(eps_mix)
        self.z = float(z)
        self.n_files = int(n_files)
        self.success_sigma, self.success_error = float(success_sigma), float(success_error)
        self.terminate_on_success = bool(terminate_on_success)
        self.n_recent = int(config.ENV_N_RECENT)
        self.obs_version = str(obs_version)
        if shaping not in ("none", "potential"):
            raise ValueError("shaping must be none or potential")
        self.fail_error_cap_m, self.shaping = float(fail_error_cap_m), shaping            # TRAINING-ONLY reward options (D12); evaluation environments keep the defaults
        self.shaping_weight, self.shaping_cap_m, self._phi_prev = float(shaping_weight), float(shaping_cap_m), 0.0
        self.obs_dim = int(config.env_obs_dim(self.obs_version))        # v1: 56, v2: 64 (module docstring)
        self.observation_space = spaces.Box(-config.ENV_OBS_BOUND, config.ENV_OBS_BOUND, shape=(self.obs_dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(config.DRONE_N_ACTIONS)
        self._kin = {False: DroneKinematics(scene.obstacles, z=self.z)}
        if scene_reflected is not None:
            self._kin[True] = DroneKinematics(scene_reflected.obstacles, z=self.z)
        # episode state (set by reset)
        self.scene: Scene = scene
        self.reflected = False
        self.source = self.sources[0]
        self.truth_xy = np.zeros(2)
        self.scale = 1.0
        self.frame0 = self.frame_range[0]
        self.frame_meas = self.frame0            # truth frame of the last measurement (reset: of the first one)
        self.t = 0
        self.xy = np.zeros(2)
        self.pf: RBPF | None = None
        self.gmm: GmmSummary | None = None
        self.h0 = self.h_prev = 1.0
        self._recent: list[tuple[float, float, float]] = []
        self.steps_since_detection = 0
        self._seen_det = False                                           # a detection happened in this episode (observation v2)
        self._rng_pf = self._rng_det = self._rng_gmm = np.random.default_rng(0)
        self.last_timing: dict[str, float] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def kin(self) -> DroneKinematics:
        return self._kin[self.reflected]

    def frame_at(self, t: int) -> int:
        """Truth frame index of RL step t (0-based measurement index): Mode F = frame0; Mode T2 = min(N - 1,
        round(frame0 + files_per_step t)) (config.T1_4_MODE_SCHEDULES['T2'], D8-2)."""
        if self.truth_mode == "F":
            return self.frame0
        return int(min(self.n_files - 1, round(self.frame0 + self.t2_files_per_step * t)))

    def action_mask(self) -> np.ndarray:
        """(9,) bool: allowed actions at the current position (building / domain masking, plan 4.5)."""
        return self.kin.action_mask(self.xy)

    def _sample_start(self, rng: np.random.Generator) -> np.ndarray:
        """Free point in the PF prior box at >= start_min_dist from the source.  With probability start_plume_frac the point
        must also lie in the detectable plume region of the episode's source: the expected counts (k0 scale n + b) T of the
        truth density at the episode's frame reach ENV_START_PLUME_MIN_COUNTS_FACTOR x the Currie threshold x T.  When that
        region has no admissible cell (weak sources) the start falls back to 'random'.  self.start_type records the outcome."""
        sx, sy = self.truth_xy
        want_plume = bool(rng.random() < self.start_plume_frac)
        om = self.scene.obstacles
        thr = config.ENV_START_PLUME_MIN_COUNTS_FACTOR * self.det.detection_threshold_cps() * self.det.T
        if want_plume and hasattr(self.scene.backend, "slab"):
            # direct draw from the detectable cells of the episode's slab (rejection from a uniform box rarely hits a thin plume);
            # a reflected scene uses the same slab with the cell centres mirrored (x, y) -> (x, -y), like the density queries (flip_y)
            sf = self.scene.backend.slab(self.frame_at(0))
            g = sf.grid
            dens = sf.density[list(sf.sources).index(self.source), sf.z_index(self.z)].astype(np.float64)
            xx, yy = np.meshgrid(g.x_centres, g.y_centres)
            if self.scene.reflected:
                yy = -yy
            okc = ((self.det.expected_counts(dens, self.scale) >= thr) & (np.hypot(xx - sx, yy - sy) >= self.start_min_dist) & (np.hypot(xx - sx, yy - sy) <= self.start_max_dist)
                   & (xx >= self.prior_x[0]) & (xx <= self.prior_x[1]) & (yy >= self.prior_y[0]) & (yy <= self.prior_y[1]))
            cells = np.flatnonzero(okc.ravel())
            for _ in range(config.ENV_START_MAX_BATCHES):
                if cells.size == 0:
                    break
                j = int(cells[int(rng.integers(cells.size))])
                cand = np.array([xx.flat[j] + rng.uniform(-0.5, 0.5) * g.res, yy.flat[j] + rng.uniform(-0.5, 0.5) * g.res])
                dc = float(np.hypot(cand[0] - sx, cand[1] - sy))        # the +-res/2 jitter must not leave the distance ring or the prior box
                if (self.start_min_dist <= dc <= self.start_max_dist and self.prior_x[0] <= cand[0] <= self.prior_x[1]
                        and self.prior_y[0] <= cand[1] <= self.prior_y[1] and om.is_free(cand, self.z)):
                    self.start_type = "plume"
                    return cand
        for attempt in range(2 if want_plume else 1):
            plume = want_plume and attempt == 0
            for _ in range(config.ENV_START_MAX_BATCHES):
                cand = np.column_stack([rng.uniform(self.prior_x[0], self.prior_x[1], config.ENV_START_BATCH),
                                        rng.uniform(self.prior_y[0], self.prior_y[1], config.ENV_START_BATCH)])
                d = cand - np.array([sx, sy])
                dist = np.hypot(d[:, 0], d[:, 1])
                ok = (dist >= self.start_min_dist) & (dist <= self.start_max_dist) & om.is_free(cand, self.z)
                if plume and ok.any():
                    sel = np.flatnonzero(ok)
                    dens = self.scene.backend.density([self.source], cand[sel], self.frame_at(0), self.z, 1.0, flip_y=self.scene.reflected)
                    ok[sel] = self.det.expected_counts(dens, self.scale) >= thr
                idx = np.flatnonzero(ok)
                if idx.size:
                    self.start_type = "plume" if plume else "random"
                    return cand[idx[0]].copy()
        raise RuntimeError(f"no free start {self.start_min_dist}..{self.start_max_dist} m from source {self.source} found")

    def _measure(self) -> tuple[int, float]:
        """One count at the current position / frame: (y, truth density); records the frame in self.frame_meas."""
        self.frame_meas = self.frame_at(self.t)
        dens = float(self.scene.backend.density([self.source], self.xy, self.frame_meas, self.z, 1.0,
                                                flip_y=self.scene.reflected)[0])
        y = int(self.det.measure(dens, self.scale, self._rng_det))
        return y, dens

    def _refresh_gmm(self) -> None:
        """Cold weighted-k-means++ EM (ENV_GMM_EM_ITERS) or, with gmm_warm_start, EM started from the previous
        summary of this episode (D8-4 T2-3 option)."""
        init = self.gmm if (self.gmm_warm_start and self.gmm is not None) else None
        self.gmm = summarise_pf(self.pf, rng=self._rng_gmm, n_iter=self.gmm_iters, init=init)

    def _observation(self) -> np.ndarray:
        if self.obs_version == "v2":
            return self._observation_v2()
        x, y = self.xy
        obs = np.empty(self.obs_dim, dtype=np.float64)
        obs[:config.GMM_VECTOR_DIM] = self.gmm.to_vector(self.xy)
        k = config.GMM_VECTOR_DIM
        obs[k:k + 3] = (x / config.GMM_NORM_XY[0], y / config.GMM_NORM_XY[1], 1.0 - self.t / self.max_steps)
        k += 3
        rec = np.zeros((self.n_recent, 3))
        for i, (yn, mx, my) in enumerate(reversed(self._recent[-self.n_recent:])):
            rec[i] = (yn, (mx - x) / config.ENV_REL_NORM_M, (my - y) / config.ENV_REL_NORM_M)
        obs[k:k + 3 * self.n_recent] = rec.ravel()
        k += 3 * self.n_recent
        obs[k] = min(1.0, self.steps_since_detection / config.ENV_DETECTION_NORM_STEPS)
        k += 1
        obs[k:k + 2] = self.scene.observation_wind().uv_at(self.xy, self.z)[0] / config.ENV_WIND_NORM
        k += 2
        obs[k:k + config.DRONE_N_HEADINGS] = self.scene.obstacles.ray_distances(self.xy, self.z)[0] / config.RAY_MAX_RANGE_M
        k += config.DRONE_N_HEADINGS
        assert k == self.obs_dim
        return np.clip(obs, -config.ENV_OBS_BOUND, config.ENV_OBS_BOUND).astype(np.float32)

    def _observation_v2(self) -> np.ndarray:
        """Observation version v2 (module docstring): egocentric, bounded features; all component features are 0 for invalid components."""
        x, y = self.xy
        g = self.gmm
        K = config.GMM_K
        valid = np.asarray(g.mask, dtype=bool)
        obs = np.zeros(self.obs_dim, dtype=np.float64)
        k = 0
        obs[k:k + K] = np.where(valid, g.weights, 0.0)
        k += K
        obs[k:k + K] = valid.astype(float)
        k += K
        sig = np.sqrt(np.maximum(g.covs[:, 0, 0], g.covs[:, 1, 1]))
        obs[k:k + K] = np.where(valid, np.log10(np.maximum(sig, 1.0) / config.ENV_V2_LOGSIGMA_REF_M), 0.0)
        k += K
        rel = g.means - self.xy[None, :]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        ok = valid & (dist >= 1.0)
        unit = np.zeros((K, 2))
        unit[ok] = rel[ok] / dist[ok, None]
        comp = np.column_stack([unit, np.where(ok, np.log10(1.0 + dist / config.ENV_V2_DIST_REF_M), 0.0)])
        obs[k:k + 3 * K] = comp.ravel()
        k += 3 * K
        heads = heading_unit_vectors()
        obs[k:k + config.DRONE_N_HEADINGS] = heads @ unit[0]
        k += config.DRONE_N_HEADINGS
        obs[k] = min(2.0, max(0.0, self.h_prev / max(self.h0, 1e-9)))
        k += 1
        obs[k:k + 3] = (x / config.GMM_NORM_XY[0], y / config.GMM_NORM_XY[1], 1.0 - self.t / self.max_steps)
        k += 3
        rec = np.zeros((self.n_recent, 3))
        for i, (yn, mx, my) in enumerate(reversed(self._recent[-self.n_recent:])):
            rec[i] = (yn, (mx - x) / config.ENV_V2_REC_NORM_M, (my - y) / config.ENV_V2_REC_NORM_M)
        obs[k:k + 3 * self.n_recent] = rec.ravel()
        k += 3 * self.n_recent
        obs[k] = 1.0 if not self._seen_det else min(1.0, self.steps_since_detection / config.ENV_V2_DET_NORM_STEPS)
        k += 1
        uv = self.scene.observation_wind().uv_at(self.xy, self.z)[0] / config.ENV_WIND_NORM
        obs[k:k + 2] = uv
        k += 2
        obs[k:k + config.DRONE_N_HEADINGS] = heads @ uv
        k += config.DRONE_N_HEADINGS
        obs[k:k + config.DRONE_N_HEADINGS] = self.scene.obstacles.ray_distances(self.xy, self.z)[0] / config.RAY_MAX_RANGE_M
        k += config.DRONE_N_HEADINGS
        assert k == self.obs_dim
        return np.clip(obs, -config.ENV_OBS_BOUND, config.ENV_OBS_BOUND).astype(np.float32)

    def privileged(self) -> np.ndarray:
        """(1, config.ENV_PRIV_DIM) TRAINING-ONLY critic features of the drone: (truth - drone) / 1000 (2), log10(1 + distance / 50 m), log10(sensor scale),
        (truth frame - 400) / 200.  Never part of the observation (asymmetric actor-critic, D12)."""
        return self._privileged_of(self.xy)[None, :]

    def _privileged_of(self, xy: np.ndarray) -> np.ndarray:
        rel = self.truth_xy - np.asarray(xy, dtype=float)
        d = float(np.hypot(*rel))
        return np.array([rel[0] / 1000.0, rel[1] / 1000.0, np.log10(1.0 + d / 50.0), np.log10(self.scale), (self.frame_at(self.t) - 400.0) / 200.0], dtype=np.float32)

    def _potential(self) -> float:
        """Shaping potential Phi = -weight x E_w[min(|x_i - x*|, cap)] / 100 over the PF particles (belief-weighted distance to the TRUE source;
        truth is used only inside the training reward, never in the observation).  0 when shaping is off."""
        if self.shaping == "none":
            return 0.0
        d = np.minimum(np.hypot(*(self.pf.xy - self.truth_xy).T), self.shaping_cap_m)
        return -self.shaping_weight * float(self.pf.weights() @ d) / config.ENV_FAIL_ERROR_SCALE_M

    def _reward(self, gain: float, exited: bool, success: bool, done: bool, err: float) -> tuple[float, float]:
        """(reward, shaping part).  Shaping F = Phi(s') - Phi(s) with Phi = 0 at the end of the episode (Ng et al. 1999: the optimal policy is unchanged)."""
        reward = config.ENV_REWARD_TIME + config.ENV_REWARD_INFO * gain
        if exited:
            reward += config.ENV_REWARD_EXIT
        if success:
            reward += config.ENV_REWARD_SUCCESS
        elif done and not success:
            reward -= min(err, self.fail_error_cap_m) / config.ENV_FAIL_ERROR_SCALE_M
        shaped = 0.0
        if self.shaping != "none":
            phi = 0.0 if done else self._potential()
            shaped, self._phi_prev = phi - self._phi_prev, phi
            reward += shaped
        return reward, shaped

    def _map_error(self) -> float:
        return float(np.hypot(*(self.pf.map_estimate() - self.truth_xy)))

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = {"source": self.source, "truth_xy": self.truth_xy.copy(), "frame": self.frame_meas,   # frame of info["y"] (reset: of the first measurement)
                "scale": self.scale, "reflected": self.reflected, "t": self.t, "drone_xy": self.xy.copy(), "start_type": self.start_type,
                "action_mask": self.action_mask(), "map_error_m": self._map_error(), "entropy": self.h_prev,
                "top_sigma_m": self.gmm.top_sigma()}
        info.update(extra)
        return info

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        """options: source, frame (Mode F) / t2_start (Mode T2), scale, start_xy, reflect (bool) override the draws."""
        super().reset(seed=seed)
        opt = dict(options or {})
        rng = self.np_random
        self._rng_pf = np.random.default_rng(int(rng.integers(2**31 - 1)))
        self._rng_det = np.random.default_rng(int(rng.integers(2**31 - 1)))
        self._rng_gmm = np.random.default_rng(int(rng.integers(2**31 - 1)))
        reflect = bool(opt.get("reflect", rng.random() < self.reflect_prob))
        if reflect and self.scene_reflected is None:
            raise ValueError("reflect requested but no scene_reflected")
        self.reflected = reflect
        self.scene = self.scene_reflected if reflect else self.scene0
        self.source = int(opt.get("source", self.sources[int(rng.integers(len(self.sources)))]))
        if self.source not in self.scene.sources_xy:
            raise ValueError(f"unknown source {self.source}")
        self.truth_xy = np.array(self.scene.sources_xy[self.source], dtype=np.float64)
        if self.truth_mode == "F":
            self.frame0 = int(opt.get("frame", rng.integers(self.frame_range[0], self.frame_range[1] + 1)))
        else:
            self.frame0 = int(opt.get("t2_start", rng.integers(self.t2_start_range[0], self.t2_start_range[1] + 1)))
        self.scale = float(opt.get("scale", Detector.sample_scale(rng, self.scale_range[0], self.scale_range[1])))
        self.t = 0
        self.frame_meas = self.frame_at(0)
        if "start_xy" in opt:
            self.start_type = "given"
        self.xy = (np.asarray(opt["start_xy"], dtype=np.float64).copy() if "start_xy" in opt
                   else self._sample_start(rng))
        if not self.scene.obstacles.is_free(self.xy, self.z):
            raise ValueError(f"start {self.xy} is not a free position")
        self.pf = RBPF(self.scene.model, n_particles=self.n_particles, obstacles=self.scene.obstacles,
                       rng=self._rng_pf, prior_x=self.prior_x, prior_y=self.prior_y, eps_mix=self.eps_mix,
                       likelihood=self.likelihood, nb_r=self.nb_r, z=self.z,
                       background=self.det.background, T=self.det.T,                  # likelihood mean = (kappa g + b) T of THIS detector
                       kappa_ref=config.KAPPA_REF * self.det.k0 / config.SENSOR_K0)   # grid centred on k0 sqrt(q_A q_B)
        self.h0 = max(self.pf.entropy_xy(), 1e-9)
        self._phi_prev = self._potential()
        self.h_prev = self.h0
        self._recent = []
        self.steps_since_detection = 0
        self._seen_det = False                                           # a detection happened in this episode (observation v2)
        self.gmm = None                                # no warm start across episodes
        self._refresh_gmm()
        self.last_timing = {}
        return self._observation(), self._info(success=False, y=None)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.pf is None:
            raise RuntimeError("call reset() first")
        t_all = time.perf_counter()
        a = int(action)
        if not 0 <= a < self.action_space.n:
            raise ValueError(f"action {a} outside 0..{self.action_space.n - 1}")
        mask = self.action_mask()
        applied = bool(mask[a])
        if not applied:
            a = ACTION_STAY                                             # masked action -> stay (plan 4.5)
        self.xy, _ = self.kin.step(self.xy, a)
        exited = not bool(self.scene.obstacles.in_domain(self.xy))      # cannot happen through the kinematics
        # measurement + belief update
        t0 = time.perf_counter()
        y, dens = self._measure()
        self.pf.update(y, np.array([self.xy[0], self.xy[1], self.z]))
        t_pf = time.perf_counter() - t0
        self._recent.append((float(self.det.normalise(y)), float(self.xy[0]), float(self.xy[1])))
        if len(self._recent) > self.n_recent:
            del self._recent[:-self.n_recent]
        hit = bool(self.det.is_detection(y))
        self.steps_since_detection = 0 if hit else self.steps_since_detection + 1
        self._seen_det = self._seen_det or hit
        self.t += 1
        h = self.pf.entropy_xy()
        gain = (self.h_prev - h) / self.h0
        self.h_prev = h
        t0 = time.perf_counter()
        gmm_refreshed = (self.t % self.gmm_every == 0)
        if gmm_refreshed:
            self._refresh_gmm()
        t_gmm = time.perf_counter() - t0
        err = self._map_error()
        sigma = self.gmm.top_sigma()
        success = bool(gmm_refreshed and sigma < self.success_sigma and err < self.success_error)
        strict = bool(gmm_refreshed and sigma < config.SUCCESS_SIGMA_M and err < config.SUCCESS_ERROR_M)
        terminated = (success and self.terminate_on_success) or exited
        truncated = (not terminated) and self.t >= self.max_steps
        reward, shaped = self._reward(gain, exited, success, truncated or terminated, err)
        t0 = time.perf_counter()
        obs = self._observation()
        t_obs = time.perf_counter() - t0
        self.last_timing = {"pf_s": t_pf, "gmm_s": t_gmm, "obs_s": t_obs, "total_s": time.perf_counter() - t_all}
        info = self._info(success=success, success_strict=strict, y=y, density=dens, applied=applied, exited=exited, info_gain=gain,
                          gmm_refreshed=gmm_refreshed, shaping_reward=shaped)   # timing stays in self.last_timing (info must be seed-deterministic, env_checker)
        return obs, float(reward), terminated, truncated, info
