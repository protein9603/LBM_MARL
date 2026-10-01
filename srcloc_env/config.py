"""Central configuration: every path and physical/algorithmic constant used by the package.

Each constant carries its provenance in a comment:
  [측정]  measured from the sample data (report section in brackets)
  [확인됨] confirmed by the simulation code owner (2026-09-29)
  [추정]  inferred / design choice from the plan (ICRS15_연구수행계획.md section in brackets)
Do not hard-code any of these values elsewhere; import them from here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
DATA_DIR = Path(r"F:\김도현박사님 자료\JH")                      # raw data folder (never written to)
LDM_FILE_TEMPLATE = "LDM_{step}stp.vtk"                          # [측정 1.1]
FLUID_FILE = DATA_DIR / "fluid_30000stp.vtk"                      # LBM flow field [확인됨]
STL_FILE = DATA_DIR / "Leipzig_buildings.stl"

ARTIFACT_DIR = DATA_DIR / "분석스크립트"                         # reusable analysis artifacts (report 부록 B)
LEVELS_UVW_NPZ = ARTIFACT_DIR / "fluid" / "levels_uvw.npz"          # 19 z-levels of (u,v,w) on the 2.5 m lattice
FLUID_SLICES_NPZ = ARTIFACT_DIR / "fluid" / "fluid_slices_2p5m.npz"  # z 1.25-21.25 m slices + building mask/height
OCCUPANCY_2M_NPZ = ARTIFACT_DIR / "stl" / "occupancy_2m_flowframe.npz"  # 2 m building occupancy + height map
LDM_30000_NPZ = ARTIFACT_DIR / "ldm_timeseries" / "LDM_30000_positions_ptype_concn.npz"  # full 30000 frame

CACHE_DIR = DATA_DIR / "cache"                                    # NPZ cache + slabs (plan 2)
FIG_DIR = DATA_DIR / "분석그림" / "icrs15"                        # figures for the talk (plan 2)

# --------------------------------------------------------------------------------------
# LDM time-series structure  [측정 2.2]
# --------------------------------------------------------------------------------------
STEP_MIN, STEP_MAX, STEP_STRIDE = 15025, 30000, 25               # file index 0 -> 15025, 599 -> 30000
N_FILES = (STEP_MAX - STEP_MIN) // STEP_STRIDE + 1                # 600


def index_to_step(index: int) -> int:
    """File index (0..599) -> step number in the file name."""
    if not 0 <= index < N_FILES:
        raise ValueError(f"index {index} outside 0..{N_FILES - 1}")
    return STEP_MIN + STEP_STRIDE * index


def step_to_index(step: int) -> int:
    if (step - STEP_MIN) % STEP_STRIDE or not STEP_MIN <= step <= STEP_MAX:
        raise ValueError(f"step {step} is not a valid LDM output step")
    return (step - STEP_MIN) // STEP_STRIDE


def ldm_path(index: int) -> Path:
    return DATA_DIR / LDM_FILE_TEMPLATE.format(step=index_to_step(index))


# --------------------------------------------------------------------------------------
# Physical time  [확인됨 + 추정 A, report 2.3]
# --------------------------------------------------------------------------------------
DT_LDM_SECONDS = 0.25          # 1 LDM step = 0.25 s [확인됨(코드소유자)]
LDM_SUBSTEP = 10               # LDM acts every 10 index steps (emission + motion) [측정 2.2]
SEC_PER_INDEX_STEP = DT_LDM_SECONDS / LDM_SUBSTEP   # interpretation A = 0.025 s [추정]; B would be 0.25 s
RL_STEP_SECONDS = 1.0          # one RL step = sensor integration time T = 1 s [추정, plan 0]
FILES_PER_RL_STEP = RL_STEP_SECONDS / (STEP_STRIDE * SEC_PER_INDEX_STEP)   # 1.6 under A [계산]
FRAME_RANGE_MODE_F = (400, 599)      # fixed-snapshot episodes: steps 25025-30000 [plan 1]
FRAME_START_MODE_T = (100, 350)      # time-varying mode: start index range [plan 1]
MAX_EPISODE_STEPS = 300

# --------------------------------------------------------------------------------------
# Domain, lattice, geometry  [측정 3.1, 4.2, 5]
# --------------------------------------------------------------------------------------
DX = 2.5                                                          # LBM lattice spacing [m]
DOMAIN_X = (0.0, 1315.0)
DOMAIN_Y = (-657.5, 657.5)
DOMAIN_Z = (-8.75, 371.25)
STL_SHIFT = np.array([689.26, -11.48, 0.0])                       # STL frame -> fluid/LDM frame [측정 4.2]
X_OUTFLOW = 1315.0             # particles with x >= X_OUTFLOW are the outflow pile-up artifact [확인됨 2.4]
DEPOSIT_Z = 1e-4               # deposited particles sit exactly here with zero velocity [측정 2.4]

# --------------------------------------------------------------------------------------
# Concentration kernel  [측정 2.1]  concn_i = LATTICE_CELL_VOLUME * sum_j W_C6(r_ij / KERNEL_H)
# --------------------------------------------------------------------------------------
KERNEL_H = 7.5                                                    # support radius H = 3 * DX [m]
LATTICE_CELL_VOLUME = DX**3                                       # 15.625 m^3
C6_W0_SI = 1365.0 / (64.0 * np.pi * KERNEL_H**3)                  # Wendland C6 3-D normalisation, W(0) in 1/m^3 = 0.016092
C6_W0_LATTICE = C6_W0_SI * LATTICE_CELL_VOLUME                    # = 1365/(64*pi*27) = 0.251443 (lattice units) [측정]
W0_CONCN = 0.25144273          # concn of an isolated particle in the files = C6_W0_LATTICE [측정 2.1]
# concn_i = sum_j C6_W0_LATTICE * (1-q)^8 (1+8q+25q^2+32q^3), q = r_ij/KERNEL_H, same p_type, self included;
# particles per m^3 = concn / LATTICE_CELL_VOLUME.

# --------------------------------------------------------------------------------------
# Sources  [측정 2.6]  mean centre of the first 132 particles per p_type (x, y); release height ~5.5 m
# --------------------------------------------------------------------------------------
SOURCE_Z = 5.5
SOURCES_XY: dict[int, tuple[float, float]] = {
    101: (499.6, -335.1), 102: (420.5, -320.6), 103: (597.3, -202.5), 104: (723.9, -139.6),
    105: (947.9, -135.4), 106: (369.8, -89.7), 107: (792.5, -57.3), 108: (345.2, 267.5),
    109: (917.6, 289.7), 110: (525.9, 270.5), 111: (725.9, 163.0), 112: (922.4, 115.3),
    113: (435.6, 91.4),
}
ALL_SOURCES = tuple(sorted(SOURCES_XY))
HOLDOUT_SOURCES = (103, 105, 109)                                 # [plan 1]
EXCLUDED_SOURCES = (110,)                                         # unobservable at the fixed 15 m altitude (enclosed courtyard) [결정 2026-09-30 D7-4, validation_log]
TRAIN_SOURCES = tuple(s for s in ALL_SOURCES if s not in HOLDOUT_SOURCES and s not in EXCLUDED_SOURCES)
RELEASE_SIGMA_XY = 4.4                                            # volume-source horizontal std [측정 2.6]
PARTICLES_PER_INDEX_STEP_PER_SOURCE = 6.665                       # [측정 2.2]

# --------------------------------------------------------------------------------------
# Slab grids for the drone altitude  [plan 3 S0]
# --------------------------------------------------------------------------------------
DRONE_Z = 15.0
SLAB_Z_MAIN = (15.0,)
SLAB_Z_AUX = (12.5, 17.5)
SLAB_RES = 5.0                                                    # [m]
SLAB_X0, SLAB_NX = 330.0, 200                                     # x 330-1330 m covers the LDM extent [측정 2.5]
SLAB_Y0, SLAB_NY = -487.5, 207                                    # y -487.5 to 547.5 m
SLAB_DTYPE = np.float16

# --------------------------------------------------------------------------------------
# Sensor  [plan 4.1]
# --------------------------------------------------------------------------------------
SENSOR_K0 = 1000.0             # cps per (particle/m^3) placeholder [결정 2026-09-30]: with 26.3 only 1.5% of occupied 15 m slab cells
                               # exceeded the Currie threshold (validate_detector); 1000 gives 20% (110: 21%, 108: 25%) and 90th-pct counts ~53 vs b=20.
                               # k0 = activity per particle x detector efficiency is unknown anyway; scale in SENSOR_SCALE_RANGE randomises it.
SENSOR_BACKGROUND_CPS = 20.0
SENSOR_T = RL_STEP_SECONDS
SENSOR_SCALE_RANGE = (0.3, 3.0)                                   # per-episode log-uniform scale on K0

# --------------------------------------------------------------------------------------
# Particle filter / GMM  [plan 4.3, 4.4]
# --------------------------------------------------------------------------------------
PF_N_PARTICLES = 2000
PF_PRIOR_X = (330.0, 1315.0)
PF_PRIOR_Y = (-550.0, 550.0)
PF_EPS_MIX = 0.05
PF_JITTER_M = 3.0
RELEASE_Q_A = PARTICLES_PER_INDEX_STEP_PER_SOURCE / SEC_PER_INDEX_STEP   # 266.6 particles/s per source under interpretation A [계산]
RELEASE_Q_B = PARTICLES_PER_INDEX_STEP_PER_SOURCE / DT_LDM_SECONDS       # 26.66 particles/s per source under interpretation B [계산]
KAPPA_REF = SENSOR_K0 * float(np.sqrt(RELEASE_Q_A * RELEASE_Q_B))      # geometric centre of the A/B nominal ranges (temporary until T1-3) [T1-2b, 결정 2026-09-30]
KAPPA_GRID_DECADES = 3.0       # +-3 decades: covers A and B nominal ranges (2 decades) + ~1 decade model mismatch with >= 1 decade margin [T1-2b]
KAPPA_G = 31                   # 0.2-decade spacing [plan 4.3]
GMM_K = 3

# --------------------------------------------------------------------------------------
# Drone / environment  [plan 4.5]
# --------------------------------------------------------------------------------------
DRONE_STEP_M = 5.0
BUILDING_MARGIN_M = 2.0
SUCCESS_SIGMA_M = 15.0
SUCCESS_ERROR_M = 20.0

# --------------------------------------------------------------------------------------
# Concentration-field query backend  [plan 3 S0 T0-5, docs/data_cache.md]
# --------------------------------------------------------------------------------------
FRAME_P_TYPE_OFFSET = 100      # frames/frame_XXX.npz stores p_type - 100 as uint8 (convert_ldm.py, data_cache.md)
FIELD_MAX_CACHED_FRAMES = 8    # LRU size of float32 slab frames held by LdmSlabBackend (float32: 2.15 MB per z level, 6.5 MB per 3-level frame) [계산]
T0_5_QUERY_LATENCY_S = 1e-4    # T0-5 pass criterion: one cached slab query < 0.1 ms [plan 3 S0]

# --------------------------------------------------------------------------------------
# LBM wind field lookup  [측정 3.1, 3.3, 3.6; field/wind.py]
# --------------------------------------------------------------------------------------
WIND_AXIS_ORDER = "level,ix,iy,comp"   # levels_uvw.npz / fluid_slices_2p5m.npz uvw[i, ix, iy, c]: x = x[ix], y = y[iy] [측정, validate_wind]
PLUME_REGION_X = (330.0, 1315.0)       # plume-region window of the report 3.3 band means: OPEN interval 330 < x < 1315 (v14_fluid_src_velocity_k.py) [측정 3.3]
PLUME_REGION_Y = (-500.0, 500.0)       # |y| < 500, open interval (nodes at y = +-500 excluded) [측정 3.3]
WIND_REF_MEAN_U = {1.25: 1.146, 11.25: 2.652, 21.25: 3.825, 48.75: 6.528, 363.75: 10.136}   # whole-level horizontal mean u [m/s] [측정 3.3]
WIND_REF_INFLOW_U = {1.25: 3.39, 6.25: 4.60, 16.25: 5.52}   # inflow column x=0 mean u [m/s] [측정 3.3]
WIND_REF_PLUME_U_10_15 = 1.68          # plume-region mean u, 10-15 m band, all lattice nodes [m/s] [측정 3.3]
WIND_REF_PLUME_U_BANDS = {(0.0, 5.0): 0.82, (10.0, 15.0): 1.68, (20.0, 25.0): 2.96}   # plume-region mean u per z band [lo, hi) over ALL lattice nodes (stored levels inside the band: 1.25/3.75, 11.25/13.75, 21.25/23.75) [측정 3.3]
WIND_TOL_PLUME_U = 5e-3                # validate_wind tolerance on the band means (report quotes 2 decimals; v14 exact values 0.8197/1.6817/2.9564) [plan D2]
WIND_REF_SOURCE_SPEED_RANGE = (0.3, 3.8)   # loose expectation for the 15 m wind speed at the 13 sources (informational only) [plan D2]
WIND_REF_SOURCE_SPEED = {103: 0.33, 107: 0.63, 109: 3.79, 110: 0.43}   # "근방 풍속" of report 2.6 = mean |(u,v,w)| over ALL lattice nodes with |x-xs| < 15, |y-ys| < 15, 2 < z < 15 (v14_fluid_src_velocity_k.py) [측정 2.6]
WIND_REF_SOURCE_BOX_HALF_M = 15.0      # half-width of that near-source box [m] (report 2.6 / v14) [측정 2.6]
WIND_REF_SOURCE_Z_RANGE = (2.0, 15.0)  # open z interval of that box -> stored levels 3.75 ... 13.75 [측정 2.6]
WIND_REF_SOURCE_Z = 6.25               # stored level nearest the release height (~5.5 m, report 2.6) used for the informational point-sample speeds [추정]
WIND_TOL_SOURCE_SPEED = 1e-2           # validate_wind tolerance on the near-source box-mean speeds [m/s] (report quotes 2 decimals) [plan D2]
WIND_TOL_MEAN_U = 1e-3                 # validate_wind pass tolerance on the level means [m/s] [plan D2]
WIND_TOL_INFLOW_U = 0.1                # validate_wind pass tolerance on the inflow column mean [m/s] [plan D2]

# --------------------------------------------------------------------------------------
# Obstacle map / drone kinematics  [측정 4.1-4.3; plan 4.5; env/drone.py, validate_drone]
# --------------------------------------------------------------------------------------
OCC_AXIS_ORDER = "ix,iy"               # occupancy_2m_flowframe.npz occ/hmap[ix, iy]: x = x0 + ix*res, y = y0 + iy*res (stl_tools cell convention) [측정, validate_drone]
OCC_HMAP_NO_BUILDING = 0.0             # hmap value where there is no building; occ == (hmap > 0) on all 51,471 cells [측정, validate_drone]
BUILDING_BBOX_X = (200.0, 1117.0)      # building footprint extent in the fluid/LDM frame (p_type 1000 bbox x 200.0-1116.75) [측정 4.2]
BUILDING_BBOX_Y = (-453.0, 453.0)      # y -452.5..452.5 [측정 4.2]
TOWER_115_SEARCH_RADIUS_M = 80.0       # the 115 m tower lies within this radius of source 110 (report 2.6/4.1; found at (552, 300.5)) [측정]
DRONE_N_HEADINGS = 8                   # discrete move headings E, NE, N, NW, W, SW, S, SE (plan 4.5: 8방향 x 5 m + 정지)
DRONE_N_ACTIONS = DRONE_N_HEADINGS + 1 # + stay (action index 8)
RAY_MAX_RANGE_M = 100.0                # 8-direction building-distance observation, normalised by 100 m (plan 4.5)
VALIDATE_DRONE_N_RANDOM = 5000         # random points for the stl_tools.is_inside_building agreement check (validate_drone) [plan D2]
VALIDATE_DRONE_N_TIMING = 1000         # batch size for the is_free / ray_distances timing (validate_drone) [plan D2]

# --------------------------------------------------------------------------------------
# Scene figures  [plan S0 산출물 그림 1-2; scripts/fig_scene.py]
# --------------------------------------------------------------------------------------
FIG_SCENE_FRAME_INDEX = N_FILES - 1    # figures 1-2 use the last frame: index 599 = step 30000 [plan S0]
LDM_EXTENT_X_30000 = (330.6, 1322.6)   # LDM particle envelope at step 30000, x [m] (dashed box in figure 1) [측정 2.5, 5]
LDM_EXTENT_Y_30000 = (-485.8, 548.0)   # y [m] [측정 2.5, 5]
FIG_HIGHLIGHT_SOURCE = 110             # source highlighted in figure 2 (next to the 115 m tower, report 2.6 / 4.1) [plan S0]
FIG_ZOOM_HALF_WIDTH_M = 100.0          # figure 2(b): +-100 m window around FIG_HIGHLIGHT_SOURCE [plan S0]
FIG_DPI_FINAL = 300                    # publication PNG dpi [plan S0]
FIG_DPI_PREVIEW = 120                  # preview PNG dpi [plan S0]
FIG_WIND_ARROW_LENGTH_M = 150.0        # drawn length of the mean-wind arrow in figure 1 (annotation only) [plan S0]
FIG_LOG_DECADES = 4.0                  # log10 colour scale of the density maps spans [vmax/10^4, vmax] [plan S0]

# --------------------------------------------------------------------------------------
# Slab validation T0-3 / T0-4  [plan 3 S0; scripts/validate_slabs.py]
# --------------------------------------------------------------------------------------
T0_3_N_QUERIES_PER_SOURCE = 1000   # random drone positions per source inside its occupied slab cells (plan S0 T0-3: 1,000곳) [plan S0]
T0_3_N_UNIFORM = 2000              # extra positions uniform over the whole slab grid, all 13 sources summed [plan D2]
T0_3_MIN_R = 0.9                   # T0-3 pass criterion: pooled Pearson r (bilinear slab vs exact gather) >= 0.9 [plan S0]
T0_3_REL_ERR_MIN_EXACT = 1e-4      # relative error is reported only where the exact density > this [particles/m^3] [plan D2]
T0_3_REL_ERR_PERCENTILE = 90       # upper percentile of the relative error reported next to the median [plan D2]
T0_3_LOG_OFFSET = 1e-6             # log10(value + offset) correlation offset [particles/m^3] (float16 slab resolution ~6e-8 at 1e-4) [plan D2]
T0_4_FRAME_INDICES = (400, 500, 599)   # snapshot growth table indices (steps 25025 / 27525 / 30000) [plan S0 T0-4]
T0_4_EXPECTED_GROWTH_TOTAL = 1.46  # plan S0 T0-4 / 슬라이드 6 각주: total airborne count 599/400 expected about +46% [계산, plan S0]

# --------------------------------------------------------------------------------------
# Sensor model details  [plan 4.1; sensor/detector.py, validate_detector]  (appended D3)
# --------------------------------------------------------------------------------------
SENSOR_REF_DENSITY = 600.0 / LATTICE_CELL_VOLUME   # 38.4 particles/m^3 = max field concn ~600 / 15.625 (plan 4.1 "38 입자/m³") [계산, plan 4.1]
SENSOR_Y_MAX = (max(SENSOR_SCALE_RANGE) * SENSOR_K0 * SENSOR_REF_DENSITY + SENSOR_BACKGROUND_CPS) * SENSOR_T   # ~3,050 counts: policy input log1p(y)/log1p(y_max) [계산, plan 4.1]
SENSOR_CURRIE_K = 3.0          # Currie-type decision threshold b + k*sqrt(b*T)/T (R4; plan 4.1: 20 + 13.4 ~ 33 cps) [plan 4.1]
VALIDATE_DETECTOR_DOWNWIND_M = (50.0, 200.0)   # validate_detector: +x offsets from each source at which expected counts are reported [plan D3]
VALIDATE_DETECTOR_N_POISSON = 100_000          # validate_detector / test_detector: Poisson sample size for the 1% mean check [plan D3]
VALIDATE_DETECTOR_MEAN_TOL = 0.01              # relative tolerance of the empirical Poisson mean [plan D3]

# --------------------------------------------------------------------------------------
# Analytic forward model (Gaussian plume)  [plan 4.2; pf/forward_model.py; validate_forward]  -- appended D3
# --------------------------------------------------------------------------------------
FWD_DEFAULT_U = WIND_REF_PLUME_U_10_15                             # 1.68 m/s: plume-region mean u, 10-15 m band [측정 3.3]
FWD_U_CANDIDATES = (1.68, 2.96, 3.69)                              # 10-15 m band / 20-25 m band / global 10-30 m layer mean [plan 4.2]
FWD_DEFAULT_SIGMA_V = 0.5                                          # sqrt(2k/3) turbulent velocity scale [m/s] [plan 4.2]
FWD_SIGMA_V_CANDIDATES = (0.3, 0.5, 0.9)                           # T1-3 calibration candidates [plan 4.2]
FWD_T_L = 9.5                                                      # Lagrangian time scale k/epsilon ~ 9-10 s [측정 2.1]
FWD_SIGMA_Z_RATIO = 0.6                                            # sigma_z = 0.6 * sigma_y [plan 4.2 선택]
FWD_G_FLOOR = 1e-9                                                 # unit response where d <= 0 (and lower clip) [plan 4.2]
FWD_TIMING_N_HYPOTHESES = PF_N_PARTICLES                           # 2000 hypotheses in the D3 timing target [plan 4.3]
FWD_TIMING_N_DRONES = (2, 1)                                       # drone counts timed by validate_forward [plan D3]
FWD_TIMING_N_CALLS = 100                                           # median over this many calls [plan D3]
FWD_TIMING_TARGET_S = 1e-3                                         # D3 pass criterion: (2000 x 2) unit_response < 1 ms [plan D3]
FWD_VALIDATE_SOURCES = (109, 110)                                  # open (109) vs trapped (110) source, report 2.6 [plan D3]
FWD_VALIDATE_D_M = (25.0, 50.0, 100.0, 200.0, 400.0)               # +x centreline downwind distances compared to the slab [plan D3]
FWD_VALIDATE_FRAME_INDEX = N_FILES - 1                             # index 599 = step 30000 [plan D3]
Q_RELEASE_A = PARTICLES_PER_INDEX_STEP_PER_SOURCE / SEC_PER_INDEX_STEP   # 266.6 particles/s actual release rate under interpretation A [계산]

# --------------------------------------------------------------------------------------
# Figure 2(a) deposited-particle effect  [plan S0 그림 2(a), 슬라이드 5; scripts/fig_data.py]  -- appended D3
# --------------------------------------------------------------------------------------
FIG_DATA_FRAME_INDEX = N_FILES - 1                                 # index 599 = step 30000 [plan S0]
FIG_DATA_Z_LEVELS = (0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0)          # query heights [m]; deposited particles (z = DEPOSIT_Z, report 2.4) reach at most KERNEL_H = 7.5 m [계산]
FIG_DATA_RATIO_LOG_DECADES = 2.0                                   # heatmap colour scale of the with/without ratio spans [1, 10^2] [plan S0]

# --------------------------------------------------------------------------------------
# RB-PF details  [plan 4.3, 4.5; pf/particle_filter.py; validate_kappa_grid]  -- appended D3 (particle_filter)
# --------------------------------------------------------------------------------------
PF_ENTROPY_CELL_M = 20.0            # entropy / weighted-mode histogram cell of the PF belief [m] (plan 4.5 reward: 20 m cells) [plan 4.5]
PF_NB_ALPHA0 = 1.0                  # Gamma prior shape of the b = 0 NB path, alpha_0 = 1, beta_0 = 1/KAPPA_REF (plan 4.3) [plan 4.3]
PF_RESAMPLE_NEFF_FRACTION = 0.5     # systematic resampling when N_eff < 0.5 N (plan 4.3 "N_eff < N/2") [plan 4.3]
PF_JITTER_MAX_TRIES = 10            # re-draws of a jittered particle that leaves the prior box / hits a building, then keep the parent [plan 4.3]
PF_RESET_MAX_ROUNDS = 100           # rejection-sampling rounds of the uniform prior over free cells before giving up [계산]
PF_MAP_TOP_FRACTION = 0.05          # map_estimate(method="top"): weighted mean of the top 5 % particles by weight [plan 4.3]
PF_TIMING_N_UPDATES = 1000          # validate_kappa_grid: number of RBPF.update calls timed (N = PF_N_PARTICLES, G = KAPPA_G) [plan D3]
PF_UPDATE_TIME_TARGET_S = 1e-3      # D3 pass criterion on the median grid update (expected 1e-4..3e-4 s) [plan D3]
T1_2B_MIN_MARGIN_DECADES = 1.0      # T1-2b: nominal kappa ranges must sit >= 1 decade inside both grid ends [plan T1-2b]
T1_2B_GRID_ROUND_DECADES = 0.5      # recommended grid_decades is rounded up to this step (plan: ±3 decade fallback, G = 31) [plan T1-2b]
T1_2B_MISMATCH_DECADES = 0.9        # informational extra: trapped-source model mismatch up to 8x ~ 0.9 decade (report 2.6) [plan 4.3]
Q_RELEASE_B = PARTICLES_PER_INDEX_STEP_PER_SOURCE / DT_LDM_SECONDS   # 26.66 particles/s actual release rate under interpretation B (index step = 0.25 s) [계산]
# --------------------------------------------------------------------------------------
# LBM adjoint forward model, option B-1  [plan 4.2b; docs/lbm_forward_model.md 2-5, 7; pf/lbm_adjoint.py; validate_adjoint]  -- appended D4-4
# --------------------------------------------------------------------------------------
ADJ_SIGMA_V_REF = 0.65             # sigma_v of the default K: middle of the report 2.1 range 0.3-0.9 m/s (FWD_SIGMA_V_CANDIDATES) [추정, plan 4.2b]
ADJ_K_DEFAULT = 4.0                # effective horizontal diffusivity K = sigma_v^2 T_L = 0.65^2 x 9.5 = 4.01 m^2/s (R15 Taylor, far-field limit) [계산]
ADJ_LAMBDA_DEFAULT = 0.02          # first-order loss rate lam [1/s] = vertical mixing out of the 15 m layer + deposition, calibrated in T1-3b [추정]
ADJ_H_LAYER = 10.0                 # layer thickness [m] of the 2-D -> 3-D conversion (particles/m^2 / h -> particles/m^3); constant absorbed by kappa [추정]
ADJ_K_CANDIDATES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)      # T1-3b calibration grid for K [m^2/s] (sigma_v 0.3-0.9 -> 0.9-7.7); 16 / 32 appended in D8-1 because K 8 sat at the grid edge (validation_log 전방모델 결정, T1-3c) [plan 4.2b, D8-1]
ADJ_LAMBDA_CANDIDATES = (0.005, 0.01, 0.02, 0.05, 0.1)   # T1-3b calibration grid for lam [1/s] [plan 4.2b]
ADJ_K_CHOSEN = 16.0                # [결정 2026-10-01 D8-1] adjoint diffusivity used by the PF / environment (filter B, env/source_env): T1-3b frame-599 train std_dense 1.366 (K 8) -> 1.203 (K 16); T1-3c time-mean 1.276 -> 1.148; T1-4 (NB r 1, lawnmower) open-source median 48.5 -> 47.4 m (neutral), 102 success 40 -> 80 %; K 32 / lam 0.01 (grid argmin 1.189) rejected: 108 collapses (12 -> 163 m), open median 54 m [D8-1 결과]
ADJ_LAMBDA_CHOSEN = 0.005          # [결정 2026-10-01 D8-1] loss rate paired with ADJ_K_CHOSEN (T1-3b / T1-3c; lam 0.002 is marginally better on the frame-599 truth only) [D8-1 결과]
ADJ_SOLVE_TIME_TARGET_S = 0.02     # one adjoint solve (receptor -> psi field) must fit the 20 ms environment-step budget [plan 4.2b]
ADJ_FOOTPRINT_TRUNC_SIGMA = 3.0    # source footprint = Gaussian(RELEASE_SIGMA_XY) on the box of +-ceil(3 sigma0 / res) cells, renormalised to 1 over free cells [추정]
ADJ_CACHE_ROUND_M = 0.5            # receptor position rounding of the psi LRU key [m] (plan 4.2b: cell index + bilinear offsets to 0.5 m) [plan 4.2b]
ADJ_MAX_CACHED = 512               # psi fields held by the LbmAdjointModel LRU (207 x 200 float64 = 0.33 MB each -> 85 MB) [계산]  # raised 256 -> 512 (2026-09-30): a 2-drone 150-step episode touches ~300 receptors; ~170 MB
ADJ_FACTORIZE_TIME_TARGET_S = 10.0 # criterion on assembly + LU factorisation of the real-size grid (200 x 207, ~20 % blocked) [plan D4-4]
ADJ_TEST_SOLVE_TIME_LOOSE_S = 0.1  # unit-test (CI) bound on the adjoint solve median; the 20 ms target itself is checked by validate_adjoint [plan D4-4]
ADJ_VALIDATE_N_RECEPTORS = 50      # validate_adjoint: random free receptors timed (median / p99) [plan D4-4]
ADJ_VALIDATE_N_PAIRS = 20          # validate_adjoint: random (source, receptor) cell pairs of the reciprocity check on the real operator [plan D4-4]
ADJ_VALIDATE_RADIUS_M = 60.0       # validate_adjoint: mass fraction within this radius of the source (109 open vs 110 trapped, report 2.6) [plan D4-4]
ADJ_RECIPROCITY_TOL = 1e-6         # validate_adjoint PASS criterion: max |C_s(p) - psi_p(s)| / max(|C_s(p)|, |psi_p(s)|) [plan D4-4]
ADJ_MASS_BALANCE_TOL = 1e-6        # validate_adjoint PASS criterion: |lam sum(C) area + outflow - 1| for a unit source [plan D4-4]
ADJ_FIG_HALF_WIDTH_M = 200.0       # fig_adjoint_check: +-200 m window around each source [plan D4-4]

# --------------------------------------------------------------------------------------
# Local-wind Gaussian plume, option A  [plan S1 보강 (2026-09-30), 4.2; pf/forward_model.py; validate_forward_local]  -- appended D4-2
# --------------------------------------------------------------------------------------
FWD_WIND_MODES = ("global", "local")   # GaussianPlume.wind_mode: 'global' = params.U / wind_dir_deg for all hypotheses (D3), 'local' = LBM wind at each hypothesis [plan S1 보강]
FWD_U_MIN = 0.3                        # [m/s] lower clip of the per-hypothesis speed U_i = max(|(u,v)|, FWD_U_MIN); below this the plume model is meaningless (travel time -> inf, trapped plume of 110) and the robust mixture must carry the hypothesis [plan S1 보강, 결정 2026-09-30]
FWD_LOCAL_WIND_Z = DRONE_Z             # [m] lookup height of the local wind = the 15 m drone slab (stored levels 13.75/16.25 interpolated, report 3.6) [plan S1 보강]
FWD_LOCAL_WIND_BLEND = 1.0             # default ForwardParams.local_wind_blend: 1 = pure local vector, 0 = global (params.U, wind_dir_deg); intermediate values mix the two vectors [plan S1 보강]
FWD_LOCAL_VALIDATE_D_M = (25.0, 50.0, 100.0, 200.0)   # validate_forward_local: centreline distances at which local vs global rho10 is reported [plan D4-2]
PF_EPS_MIX_TRAPPED = 0.1               # epsilon of the robust likelihood mixture for trapped-source scenes (110, 102; report 2.6) - recorded for D4-3 / D5, not used yet [plan S1 보강]
# --------------------------------------------------------------------------------------
# T1-3 shape-based calibration of the analytic plume  [plan S1 T1-3, 4.2 캘리브레이션 규칙; scripts/calibrate_forward.py]  -- appended D4-3
# --------------------------------------------------------------------------------------
T1_3_FRAME_INDEX = N_FILES - 1                     # truth = slab frame index 599 (step 30000), z = DRONE_Z [plan T1-3]
T1_3_OPEN_SOURCES = (101, 108, 109, 111, 113)      # open sources of the T1-3 pass criterion std(rho') < T1_3_STD_PASS (plan T1-3 / T1-4; 109 is also a holdout source) [plan T1-3]
T1_3_TRAPPED_SOURCES = (110, 102)                  # trapped sources whose analytic-plume failure is reported as is (report 2.6, plan S1 보강 / T1-4) [plan S1]
T1_3_STD_PASS = 1.0                                # PASS: std(rho') < 1.0 for every open source at the chosen combination [plan T1-3]
T1_3_G_FLOOR_FACTOR = 10.0                         # downwind cell selection also requires g > 10 * FWD_G_FLOOR (cells whose response is the floor clip carry no shape information) [plan D4-3]
T1_3_MIN_DISCRIMINABILITY = 0.1                    # plan S1 위험·대안: if the spread of mean_train std(rho') over the 9 (U, sigma_v) combos is < 0.1, the physical defaults (FWD_DEFAULT_U, FWD_DEFAULT_SIGMA_V) are to be used and "판별력 부족" recorded in table 1 [plan S1]
T1_3_FIG_SOURCES = (109, 110)                      # figure 3: rho' maps of the open source 109 and the trapped source 110 [plan T1-3, slide 9]
T1_3_FIG_RHO_CLIP = 3.0                            # figure 3 diverging colour scale of rho' clipped to +-3 [plan D4-3]
T1_3_FIG_WINDOW_M = (150.0, 450.0, 250.0)          # figure 3 window around the source: upwind / downwind (+x) / crosswind half-extent [m] [plan D4-3]
T1_3_FIG_ARROW_M = 60.0                            # drawn length of the model wind arrow at the source in figure 3 (annotation only) [plan D4-3]
ADJ_SPLU_PERMC_SPEC = "MMD_AT_PLUS_A"   # SuperLU column ordering: 35 % less fill than COLAMD on the 5-point operator (L+U 0.99 M vs 1.5 M nnz), adjoint solve 3.9 vs 6 ms [측정 D4-4]
ADJ_SPLU_SYMMETRIC_MODE = True          # SuperLU SymmetricMode (structurally symmetric pattern): factorisation 0.17 s instead of 1.96 s with MMD_AT_PLUS_A [측정 D4-4]
T1_3_INFO_DENSITY_FRACTION = 0.01                  # informational only (not the selection statistic): std(rho') restricted to cells with n_LDM > 1 % of the source's slab maximum, i.e. above the single-particle fringe (an isolated particle gives concn 0.2514 / 15.625 = 0.016 particles/m^3, report 2.1) [plan D4-3 진단]

# --------------------------------------------------------------------------------------
# GMM belief summary  [plan 4.4, R7 Park/Ladosz/Oh 2022; pf/gmm_summary.py]
# --------------------------------------------------------------------------------------
GMM_EM_ITERS = 20              # weighted EM iterations per RL step [plan 4.4]
GMM_MIN_WEIGHT = 0.02          # components below this weight are zero-padded and masked [plan 4.4]
GMM_COV_REG_M2 = 1.0           # diagonal covariance floor (1 m)^2 to avoid singular components [계산]
GMM_NORM_XY = (1315.0, 657.5)  # position normalisation (domain extent x, |y|) [plan 4.4]
GMM_NORM_COV_M2 = 1.0e4        # covariance normalisation (100 m)^2 [plan 4.4]
GMM_NORM_REL_M = 1000.0        # drone-relative mean normalisation [plan 4.4]
GMM_VECTOR_DIM = GMM_K * 6 + GMM_K + GMM_K * 2   # 27 = K x (w, mux, muy, sxx, syy, sxy) + mask K + relative K x 2 [plan 4.4]
T1_5_TV_CELL_M = 10.0          # histogram cell for the GMM-vs-PF total-variation fidelity check [plan S1 T1-5]
T1_5_TV_MAX = 0.2              # pass: TV distance < 0.2 [plan S1 T1-5]
T1_5_FLIP_MAX = 0.1            # pass: component-order flip rate between consecutive steps < 10% [plan S1 T1-5]
GMM_MERGE_BHAT = 0.30          # merge EM components with Bhattacharyya distance below this (~1.55 sigma apart for equal covariances); 0.25 left 102's split mode (B 0.26) flipping [계산, D6-2b]

# --------------------------------------------------------------------------------------
# D5-2 PF <-> LBM adjoint connection (plan 4.2b 검증 "갱신당 시간 <= 20 ms", S1 T1-2 lite with the LBM model;
# scripts/validate_pf_adjoint.py, tests/test_pf_adjoint.py)  -- appended D5-2
# --------------------------------------------------------------------------------------
PF_ADJ_SOURCES = (109, 110, 101, 113)      # open 109 (holdout), trapped courtyard 110, two more open sources 101 / 113 (report 2.6) [plan D5-2]
PF_ADJ_N_SEEDS = 5                         # Poisson + PF seeds per source [plan D5-2]
PF_ADJ_N_STEPS = 150                       # RL steps per synthetic run (plan T1-4: 2대 lawnmower 150 스텝) [plan T1-4]
PF_ADJ_N_DRONES = 2                        # drones per run, fused sequentially on one RBPF (plan 4.3) [plan T1-4]
PF_ADJ_REPORT_STEPS = (50, 100, 150)       # MAP error reported after these RL steps [plan D5-2]
PF_ADJ_SWEEP_WIDTH_M = 100.0               # y sweep width of each drone's sawtooth lawnmower; the two drones cover adjacent bands [plan D5-2]
PF_ADJ_START_DOWNWIND_M = 250.0            # start x = x_s + 250 m (>= 200 m from the source, plan D5-2), then fly -x (upwind) with DRONE_STEP_M in x per step [plan D5-2]
PF_ADJ_MAP_ERROR_PASS_M = 30.0             # expectation: open sources median MAP error < 30 m after PF_ADJ_N_STEPS (plan T1-4 open-source criterion) [plan T1-4]
PF_ADJ_UPDATE_TIME_TARGET_S = 0.02         # plan 4.2b: one drone update (adjoint solve + interpolation + likelihood) median < 20 ms [plan 4.2b]
PF_ADJ_TIMING_N_BREAKDOWN = 50             # calls per component in the timing breakdown (adjoint solve / interpolation / likelihood) [plan D5-2]
PF_ADJ_FIG_HALF_WIDTH_M = 200.0            # belief-scatter panel: +-200 m window around the source [plan D5-2]
PF_ADJ_TEST_N_MEASUREMENTS = 60            # unit test: MAP error < PF_ADJ_TEST_MAP_ERROR_CELLS cells after this many measurements on the 40 x 30 synthetic grid [plan D5-2]
PF_ADJ_TEST_MAP_ERROR_CELLS = 2.0          # unit test pass criterion in grid cells [plan D5-2]
# --------------------------------------------------------------------------------------
# T1-3b adjoint-model calibration  [plan 4.2b 캘리브레이션, S1 T1-3b; docs/lbm_forward_model.md 6; scripts/calibrate_adjoint.py]  -- appended D5-1
# --------------------------------------------------------------------------------------
T1_3B_WIND_LAYERS: dict[str, tuple[float, float] | None] = {"single_15m": None, "band_10_20m": (10.0, 20.0)}   # wind layer candidates of the adjoint operator: single 15 m wind (AdjointParams.wind_band None) vs per-cell mean over the stored LBM levels in [10, 20) m (11.25/13.75/16.25/18.75) [plan 4.2b, lbm_forward_model.md 6]
T1_3B_SELECTION_KEY = "mean_train_std_dense"        # T1-3b selection statistic: mean over TRAIN_SOURCES of std(rho') on the dense cells (n_LDM >= T1_3_INFO_DENSITY_FRACTION x source max); the plain std is fringe-dominated (validation_log "T1-3 FAIL의 해석") and its argmin is reported next to it [결정 D5-1]
T1_3B_G_MIN = T1_3_G_FLOOR_FACTOR * FWD_G_FLOOR       # 1e-8 (particles/m^3)/(particle/s): a cell counts as "model present" only above this, the same floor rule as calibrate_forward (D4-3); g == 0 exactly marks the support mismatch (wall cell / other free component) [plan D5-1]
# --------------------------------------------------------------------------------------
# T1-2 kappa-marginalisation bias check  [plan S1 T1-2, 4.3 (R5 Rao-Blackwellisation); scripts/validate_kappa_bias.py,
# tests/test_kappa_bias.py; figure 4]  -- appended D5-3
# --------------------------------------------------------------------------------------
T1_2_CASES = (("analytic", 109), ("analytic", 101), ("adjoint", 109))   # (forward model, true source): GaussianPlume (global, config defaults) for 109 and 101; LbmAdjointModel (real 15 m wind, default AdjointParams) for 109 [plan D5-3]
T1_2_N_REPEATS = 20                     # seeded repeats per case (kappa_true, Poisson counts, PF prior draw) [plan D5-3]
T1_2_N_STEPS = 100                      # lawnmower steps per drone: PF_ADJ_N_DRONES x 100 = 200 measurements per repeat [plan D5-3]
T1_2_KAPPA_TRUE_DECADES = 1.0           # kappa_true ~ log-uniform KAPPA_REF x 10^[-1, +1] per repeat [plan D5-3]
T1_2_KAPPA_FIXED_FACTORS = (3.0, 0.3)   # filters (iii) / (iv): kappa fixed at these multiples of kappa_true [plan T1-2]
T1_2_FIXED_KAPPA_DECADES = 1e-6         # a "fixed kappa" filter = RBPF with n_grid = 2 and this half-width (node ratio 10^(2e-6) = 1 + 4.6e-6): RBPF requires n_grid >= 2, grid_decades > 0, so the plan's (grid_decades 0, n_grid 1) is realised as this degenerate grid [plan D5-3, 결정]
T1_2_REPORT_MEASUREMENTS = (50, 100, 150, 200)   # MAP error recorded after these measurement counts [plan D5-3]
T1_2_MAP_DIFF_PASS_M = 10.0             # PASS 1: median over repeats of |MAP_xy(ii RB-PF) - MAP_xy(i kappa known)| < 10 m [plan T1-2]
T1_2_TEST_N_PARTICLES = 300             # unit-test sizes (tests/test_kappa_bias.py): N, steps per drone (2 x 20 = 40 measurements), repeats [plan D5-3]
T1_2_TEST_N_STEPS = 20
T1_2_TEST_N_REPEATS = 2
# --------------------------------------------------------------------------------------
# Library candidate filter  [plan 4.2 라이브러리 모델, S1 라이브러리 필터 상한 / T1-4; pf/library_filter.py;
# scripts/validate_library_filter.py, tests/test_library_filter.py]  -- appended D5-4
# --------------------------------------------------------------------------------------
LIB_FRAME_INDEX = N_FILES - 1              # measurements and library responses from slab frame index 599 (step 30000), z = DRONE_Z [plan T1-4]
LIB_N_SEEDS = 5                            # Poisson seeds per true source [plan D5-4]
LIB_N_STEPS = PF_ADJ_N_STEPS               # 150 RL steps x PF_ADJ_N_DRONES drones (plan T1-4 "2대 lawnmower 150 스텝"); path = validate_pf_adjoint.two_drone_paths [plan T1-4]
LIB_START_MIN_DISTANCE_M = 200.0           # every drone must start >= 200 m from the true source (plan T1-4); the lawnmower starts PF_ADJ_START_DOWNWIND_M = 250 m downwind [plan T1-4]
LIB_POSTERIOR_PASS = 0.9                   # "identified": posterior of the true candidate > 0.9 (steps-to-0.9 statistic) [plan D5-4]
LIB_SENSOR_SCALE = 1.0                     # validate_library_filter detector scale: kappa_true = SENSOR_K0 x scale = 1000 for the library model (1.93 decades below KAPPA_REF, inside the +-3 decade grid with 1.07 decade margin) [plan 4.2 caveat, 계산]
LIB_ADJOINT_FIELDS_NPZ = CACHE_DIR / "adjoint_fields_chosen.npz"   # calibrate_adjoint.py output (D5-1): chosen (K, lam, wind layer) unit-source fields (13, ny, nx) of the 13 sources; optional second unit_response_fn (kappa absorbs k x scale x q) [plan 4.2b]
LIB_REPORT_STEPS = PF_ADJ_REPORT_STEPS     # posterior of the true candidate reported after these RL steps [plan D5-4]
LIB_TEST_KAPPA_DECADES = (-1.5, 0.0, 1.5)  # tests/test_library_filter.py: kappa_true = KAPPA_REF x 10^d - the selection must not depend on the truth scale [plan D5-4]
# --------------------------------------------------------------------------------------
# T1-4 decisive real-data test of the RB-PF forward models  [plan S1 T1-4, G1; scripts/validate_t1_4.py,
# tests/test_validate_t1_4.py; figure 5 input (T1-5 snapshots)]  -- appended D6-1
# --------------------------------------------------------------------------------------
T1_4_FRAME_INDEX = LIB_FRAME_INDEX                 # truth = LDM slab frame 599 (step 30000), z = DRONE_Z, the same slab as validate_library_filter [plan T1-4]
T1_4_SENSOR_SCALE = LIB_SENSOR_SCALE               # detector scale 1 -> kappa_true = SENSOR_K0 x q; counts drawn exactly as validate_library_filter.run_filter (same rng stream [seed, source, 1]) so every filter sees the identical sequence [plan T1-4]
T1_4_N_SEEDS = LIB_N_SEEDS                         # 5 Poisson / PF seeds per source [plan D6-1]
T1_4_N_STEPS = PF_ADJ_N_STEPS                      # 150 RL steps x PF_ADJ_N_DRONES drones on the validate_pf_adjoint.two_drone_paths lawnmower [plan T1-4]
T1_4_CHECKPOINT_STEPS = (25, 50, 100, 150)         # MAP error recorded after these RL steps [plan D6-1]
T1_4_ENTROPY_STEPS = (0, 50, 150)                  # belief entropy recorded at these steps (0 = prior) [plan D6-1]
T1_4_GMM_EVERY = 5                                 # success test (GMM top sigma < SUCCESS_SIGMA_M and MAP error < SUCCESS_ERROR_M) evaluated every 5 steps to limit the EM cost [plan D6-1]
T1_4_GMM_SEED = 0                                  # rng seed of the weighted k-means++ initialisation of summarise_pf (deterministic success test) [plan D6-1]
T1_4_ANALYTIC_U = 1.68                             # filter A: GaussianPlume global wind, chosen T1-3 combination (calibrate_forward.json chosen: U 1.68, sigma_v 0.9) [plan T1-3 결과]
T1_4_ANALYTIC_SIGMA_V = 0.9                        # [plan T1-3 결과]
T1_4_ADJOINT_K = ADJ_K_CHOSEN                      # filter B: LbmAdjointModel at the chosen combination (D5-1: K 8 / lam 0.005 at the grid edge -> D8-1: K 16 / lam 0.005, calibrate_adjoint.json chosen via --chosen; validation_log 전방모델 결정) [plan T1-3b 결과, D8-1]
T1_4_ADJOINT_LAM = ADJ_LAMBDA_CHOSEN               # [plan T1-3b 결과, D8-1]
T1_4_FILTER_EPS = {"A": PF_EPS_MIX, "A2": PF_EPS_MIX_TRAPPED, "B": PF_EPS_MIX, "B2": PF_EPS_MIX_TRAPPED}   # robust mixture eps per filter label: A/B = 0.05, A2/B2 = 0.1 (trapped-source variant, plan S1 보강) [plan D6-1]
T1_4_FINAL_ERROR_PASS_M = PF_ADJ_MAP_ERROR_PASS_M  # G1 (i)/(ii): open sources (T1_3_OPEN_SOURCES) median final MAP error < 30 m [plan T1-4 / G1]
T1_4_EPS_REPORT_SOURCES = (102, 106)               # G1 (iv): trapped 102 (observable trapped source) and 106 (roof-leak test case) reported with eps 0.05 vs 0.1 [plan D6-1]
T1_4_UNOBSERVABLE_SOURCE = 110                     # G1 (v): enclosed courtyard at 15 m (validation_log 고정 고도 15 m의 관측 한계) [plan D6-1]
T1_4_UNOBSERVABLE_COUNT_FACTOR = 1.5               # 'unobservable at 15 m' iff max expected count along the path <= 1.5 x background counts (b T = 20 -> 30) [plan D6-1]
T1_4_SNAPSHOT_SOURCES = (109, 102)                 # belief snapshots (seed 0, filter B, every T1_4_GMM_EVERY steps) saved to CACHE_DIR / t1_4_snapshots_{src}.npz for T1-5 / figure 5 [plan T1-5]
T1_4_SNAPSHOT_FILTER = "B"                         # [plan T1-5]
T1_4_FIG_REF_LINES_M = (30.0, 20.0)                # figure reference lines: the 30 m open-source criterion and SUCCESS_ERROR_M [plan G1, 4.5]
T1_4_FIG_YLIM_M = (1.0, 1000.0)                    # log y range of the error panels [plan D6-1]
# --------------------------------------------------------------------------------------
# T1-5 GMM summary fidelity on the T1-4 belief snapshots  [plan S1 T1-5, 4.4 (R7); scripts/validate_t1_5.py,
# tests/test_validate_t1_5.py; figure 5]  -- appended D6-2b
# --------------------------------------------------------------------------------------
T1_5_SNAPSHOT_SOURCES = T1_4_SNAPSHOT_SOURCES      # (109 open, 102 trapped): CACHE_DIR / t1_4_snapshots_{src}.npz written by validate_t1_4 (filter B, seed 0, every T1_4_GMM_EVERY steps) [plan T1-5]
T1_5_GMM_SEED = T1_4_GMM_SEED                      # rng seed of the weighted k-means++ initialisation per snapshot (same seed as the T1-4 success test) [plan D6-2b]
T1_5_MATCH_RADIUS_M = 30.0                         # order_flip_rate: components of consecutive snapshots are matched by nearest mean within this radius (gmm_summary default) [plan 4.4]
T1_5_TV_CELL_INFO_M = PF_ENTROPY_CELL_M            # informational second TV cell (20 m = the entropy / MAP cell): the 10 m criterion cell holds ~0.2 particles per cell under the uniform prior, so the early-step TV is sampling-noise dominated [plan D6-2b 진단]
T1_5_FIG_STEPS = (10, 30, 60, 150)                 # figure 5 columns: the nearest saved snapshots (multiples of T1_4_GMM_EVERY) to these RL steps [plan T1-5 figure 5]
T1_5_FIG_HALF_WIDTH_M = 250.0                      # figure 5 window: +-250 m around the true source [plan T1-5 figure 5]
T1_5_FIG_ELLIPSE_SIGMAS = (1.0, 2.0)               # GMM component ellipses drawn at these Mahalanobis radii [plan T1-5 figure 5]
T1_5_FIG_ELLIPSE_LW_M = (0.8, 3.0)                 # ellipse line width range [pt], linear in the component weight 0 -> 1 [plan D6-2b]
T1_5_FIG_PARTICLE_SIZE = 4.0                       # scatter marker size of the particles (coloured by weight) [plan D6-2b]
T1_5_TV_N_SUB = 5                                  # informational TV with the GMM mass integrated on n_sub x n_sub sub-cell quadrature points: the centre-point rule of total_variation_distance overestimates TV (0.25 vs 0.03) once the belief sd (3-5 m) is below the 10 m cell [측정 D6-2b]
T1_5_TV_NOISE_N_REPEATS = 3                        # finite-N noise floor: TV of N i.i.d. samples from the fitted GMM itself vs the GMM mass (same cells, centre rule), averaged over this many seeded draws [plan D6-2b 진단]
T1_5_FIG_INSET_HALF_WIDTH_M = 30.0                 # figure 5: zoom inset (+-30 m around the MAP) drawn when the top sigma is below T1_5_FIG_INSET_SIGMA_M [plan D6-2b]
T1_5_FIG_INSET_SIGMA_M = SUCCESS_SIGMA_M           # [plan 4.5]

# ---- D6-1 review (2026-09-30): source-110 observability rule aligned with D5-2 / validation_log 결정 ('고정 고도 15 m의 관측 한계')
T1_4_UNOBSERVABLE_USE_CURRIE = True                # G1 (v): also 'unobservable at 15 m' when the max expected count along the path is below the Currie decision threshold Detector.detection_threshold_cps() x T (33.4 counts; the rule of validate_pf_adjoint); the 1.5 x background rule alone (30) misses 110 (32.0 counts, belief entropy 7.1 -> 7.0 nats) [D6-1 review, validation_log 결정 2026-09-30]
T1_5_TV_SUBCELLS = 5           # sub-cell quadrature points per axis for the GMM cell mass in total_variation_distance [D6-2b review]
# --------------------------------------------------------------------------------------
# RB-PF count likelihood  [plan 4.3 강건화, S1 T1-2 / T1-4, D7-1; R22 Yee & Chan 1997, R23 Hilbe 2011; pf/particle_filter.py;
# tests/test_particle_filter_negbin.py; validate_kappa_bias / validate_t1_4 --likelihood/--nb-r]  -- appended D7-1
# --------------------------------------------------------------------------------------
PF_LIKELIHOOD = "negbin"  # [결정 2026-09-30 D7-4] default RBPF count likelihood after D7-2/D7-3 (T1-4 open-source median 142 -> 49 m); R22/R23
PF_NB_DISPERSION_R = 1.0  # [결정 2026-09-30 D7-4] NB dispersion r: D7-2 pick (r 1 best in mode F), temporal CV of the 15 m field gives r 0.78 pooled (calibrate_timeavg); earlier spatial estimate 0.3 was too flat
PF_LIKELIHOODS = ("poisson", "negbin")   # accepted RBPF(likelihood=...) values; 'negbin' is grid mode only, the 'nb' Gamma-kappa conjugate path is Poisson only [plan D7-1]
# --------------------------------------------------------------------------------------
# D7-2 Mode T measurement support + (NB dispersion r) x (mode F / T) comparison  [plan 1 시간 모드 (Mode T: 시작 인덱스
# 100~350, 스텝당 1.6 파일 전진, zero-order hold), S1 T1-4, 4.3 강건화, D7-2; R22 / R23; scripts/validate_t1_4.py --mode,
# scripts/validate_d7_modes.py, tests/test_validate_d7_modes.py]  -- appended D7-2
# --------------------------------------------------------------------------------------
T1_4_MODES = ("F", "T")                        # validate_t1_4.generate_measurements modes: 'F' = fixed snapshot T1_4_FRAME_INDEX (D6 behaviour); 'T' = time-varying frames f(t) = min(N_FILES - 1, round(T1_4_MODE_T_START_INDEX + FILES_PER_RL_STEP t)) for RL step t, both drones of a step share the frame (plan 1 Mode T zero-order hold) [plan D7-2]
T1_4_MODE_T_START_INDEX = FRAME_START_MODE_T[0]   # 100: the comparison uses the lower end of the Mode T start range (deterministic truth; 150 steps -> frames 100..338, step 17525..23475) [plan D7-2]
D7_2_OPEN_SOURCES = T1_3_OPEN_SOURCES          # (101, 108, 109, 111, 113): selection statistic of the D7-2 recommendation = median over these of the final MAP error median [plan D7-2]
D7_2_REGRESSION_SOURCES = (102, 104, 106)      # sources filter B already solved with Poisson / Mode F (T1-4: 19 / 12 / 16 m): the regression constraint of the recommendation rule [plan D7-2]
D7_2_N_SEEDS = 3                               # Poisson / PF seeds per (config, source) [plan D7-2 "시드 3"]
D7_2_N_STEPS = PF_ADJ_N_STEPS                  # 150 RL steps x PF_ADJ_N_DRONES drones on the validate_pf_adjoint.two_drone_paths lawnmower [plan T1-4]
D7_2_NB_R_GRID = (0.3, 1.0, 3.0)               # negbin dispersion candidates next to the Poisson baseline (r -> inf); 0.3 = PF_NB_DISPERSION_R [plan D7-2]
D7_2_CHECKPOINT_STEPS = (50, 100, 150)         # MAP error recorded after these RL steps (final = 150) [plan D7-2]
D7_2_REGRESSION_TOLERANCE = 1.5                # admissible config: median over D7_2_REGRESSION_SOURCES of the final error <= 1.5 x its Poisson-F value [plan D7-2]
D7_2_N_TOP = 3                                 # configs listed in the recommendation ranking [plan D7-2]
D7_2_FIG_REF_LINES_M = T1_4_FIG_REF_LINES_M    # 30 m open-source criterion / 20 m SUCCESS_ERROR_M reference lines of fig_d7_modes [plan G1, 4.5]
D7_2_FIG_YLIM_M = T1_4_FIG_YLIM_M              # log y range of the error panels [plan D6-1]
# --------------------------------------------------------------------------------------
# T1-3c time-averaged calibration  [plan S1 T1-3 / T1-4 (G1 FAIL 대응, validation_log '결정·주의 사항' G1 FAIL 원인과 치료), 4.3;
# R22 Yee & Chan 1997 (concentration-fluctuation pdf), R23 Hilbe 2011 (NB r = 1/CV^2); scripts/calibrate_timeavg.py,
# tests/test_calibrate_timeavg.py; fig_timeavg_calibration.png]  -- appended D7-2b
# --------------------------------------------------------------------------------------
T1_3C_FRAME_RANGE = FRAME_RANGE_MODE_F           # (400, 599) inclusive: the Mode F snapshot range, 200 frames, z_levels[0] = DRONE_Z (docs/data_cache.md) [plan D7-2b]
T1_3C_TIMEAVG_NPZ_TEMPLATE = "slab_timeavg_{lo}_{hi}.npz"   # CACHE_DIR / slab_timeavg_400_599.npz: mean / std / cv / cv_norm (13, ny, nx) float32 [plan D7-2b]
T1_3C_DENSE_FRACTION = T1_3_INFO_DENSITY_FRACTION   # dense cells of the fluctuation statistic: mean >= 1 % of the source's max mean (same rule as T1-3 / T1-3b) [plan D7-2b]
T1_3C_ADJ_K_CANDIDATES = (8.0, 16.0, 32.0)        # neighbourhood of the chosen T1-3b combination (K 8 sat at the grid edge, validation_log 전방모델 결정); 32 appended in D8-1 so that K 16 is not an edge value either [plan D7-2b, D8-1]
T1_3C_ADJ_LAMBDA_CANDIDATES = (0.002, 0.005, 0.01)   # [1/s]; 0.01 appended in D8-1 (0.005 sat at the upper edge of the D7-2b neighbourhood) [plan D7-2b, D8-1]
T1_3C_ADJ_WIND_LAYERS: dict[str, tuple[float, float] | None] = {"single_15m": None}   # single 15 m wind only (the layer axis was uninformative in T1-3b) [plan D7-2b]
T1_3C_OFFSET_SOURCES = (101, 109, 111)            # sources whose mean-field 15 m maximum offset from the source is reported ('109 systematic offset' hypothesis, ~90 m) [plan D7-2b]
T1_3C_FIG_SOURCE = 109                            # figure panels (a)-(c): open holdout source 109 [plan D7-2b]
T1_3C_POOLED_SOURCES = T1_3_OPEN_SOURCES          # pooled NB dispersion over the open sources {101, 108, 109, 111, 113} [plan D7-2b]
# --------------------------------------------------------------------------------------
# D7-3 T1-4 re-run with the D7-2 likelihood recommendation + final Table 1 / G1 re-judgement  [plan S1 T1-4 / G1,
# 4.3 강건화, D7-3; R22 Yee & Chan 1997, R23 Hilbe 2011; scripts/validate_t1_4.py --baseline-json / --timeavg-json,
# table1_final_markdown, final_verdicts; scripts/gate_report.py --gate G1; tests/test_validate_t1_4.py]  -- appended D7-3
# --------------------------------------------------------------------------------------
T1_4_POISSON_BASELINE_JSON = CACHE_DIR / "validate_t1_4_poisson.json"   # the D6 Poisson / Mode F run (validate_t1_4.json copied before the D7-3 re-run; fig_t1_4_errors_poisson.png) [plan D7-3]
T1_4_TIMEAVG_JSON = CACHE_DIR / "calibrate_timeavg.json"               # T1-3c output whose 'offset_report' gives the 15 m slab peak offset of T1_4_OFFSET_SOURCE (calibrate_timeavg.py, D7-2b) [plan D7-3]
T1_4_D7_3_LIKELIHOOD = "negbin"    # D7-3 re-run likelihood = the D7-2 recommendation (negative-binomial, R22 / R23); PF_LIKELIHOOD stays 'poisson' for the unit tests / old scripts [D7-2 결과, 결정 2026-09-30]
T1_4_D7_3_NB_R_MODE_F = 1.0        # D7-2 recommendation for Mode F: negbin r = 1 (open-source median 44.7 m vs 48.9 m at r 0.3 and 139.1 m Poisson; regression median 21.2 m <= 1.5 x 18.7 m) [D7-2 결과]
T1_4_D7_3_NB_R_MODE_T = 3.0        # D7-2 recommendation within Mode T: negbin r = 3 (open median 71.3 m > 44.7 m of Mode F -> Mode T is not re-run in D7-3) [D7-2 결과]
T1_4_OFFSET_SOURCE = 109           # source annotated in the final Table 1 with its systematic 15 m offset (validation_log 'G1 FAIL 원인과 치료': 15 m slab maximum ~30 m (time mean) / ~80 m (frame 599) downwind of the source) [plan D7-3]
T1_4_REGRESSION_SOURCES = D7_2_REGRESSION_SOURCES   # (102, 104, 106): final-table verdict 'not regressed vs Poisson' = B(NB) median final error <= max(D7_2_REGRESSION_TOLERANCE x B(Poisson) median, T1_4_FINAL_ERROR_PASS_M) per source and the D7-2 pooled rule [plan D7-3]
