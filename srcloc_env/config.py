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
TRAIN_SOURCES = tuple(s for s in ALL_SOURCES if s not in HOLDOUT_SOURCES)
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
SENSOR_K0 = 26.3               # cps per (particle/m^3) placeholder: ~1e3 cps at concn 600 (=38 /m^3) [계산]
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
KAPPA_REF = SENSOR_K0 * PARTICLES_PER_INDEX_STEP_PER_SOURCE / SEC_PER_INDEX_STEP   # ~7.0e3 (temporary, A) [계산]
KAPPA_GRID_DECADES = 2.5
KAPPA_G = 26
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