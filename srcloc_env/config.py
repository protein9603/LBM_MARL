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
KERNEL_H = 7.5                                                    # support radius = 3 * DX [m]
LATTICE_CELL_VOLUME = DX**3                                       # 15.625 m^3
C6_NORMALISATION = 1365.0 / (64.0 * np.pi * (KERNEL_H / DX) ** 3)  # = 1365/(64*pi*27) per lattice-unit volume
W0_CONCN = 0.25144273          # concn of an isolated particle (self term) [측정]

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