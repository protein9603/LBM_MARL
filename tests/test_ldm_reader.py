"""Unit tests for the LDM reader masks and frame utilities (synthetic data, no raw files)."""
import numpy as np

from srcloc_env import config
from srcloc_env.io.ldm_reader import LdmFrame, airborne_mask, deposited_mask, frame_from_arrays, outflow_mask


def _synthetic_frame() -> LdmFrame:
    xyz = np.array([
        [500.0, 0.0, 10.0],            # airborne
        [600.0, 5.0, config.DEPOSIT_Z],  # deposited (zero velocity)
        [700.0, 5.0, config.DEPOSIT_Z],  # on the ground but moving -> NOT deposited by definition
        [1320.0, 0.0, 12.0],           # outflow pile-up
        [1314.9, 0.0, 12.0],           # just inside the domain
    ], dtype=np.float32)
    vel = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.1, 0.0]], np.float32)
    arrays = {"xyz": xyz, "p_type": np.array([101, 101, 102, 102, 103], np.int32), "velocity": vel,
              "concentration": np.ones(5, np.float32), "concn": np.full(5, config.W0_CONCN, np.float32)}
    return frame_from_arrays(0, arrays)


def test_masks_follow_the_report_definitions():
    f = _synthetic_frame()
    assert deposited_mask(f.xyz, f.velocity).tolist() == [False, True, False, False, False]
    assert outflow_mask(f.xyz).tolist() == [False, False, False, True, False]
    assert airborne_mask(f.xyz, f.velocity).tolist() == [True, False, True, False, True]
    assert f.airborne.sum() == 3


def test_select_and_counts():
    f = _synthetic_frame()
    air = f.select(f.airborne)
    assert air.n == 3 and air.step == config.STEP_MIN
    assert air.counts_by_source() == {101: 1, 102: 1, 103: 1}
    assert f.counts_by_source(f.deposited) == {101: 1}
    s = f.summary()
    assert s["n_total"] == 5 and s["n_deposited"] == 1 and s["n_outflow"] == 1 and s["n_airborne"] == 3


def test_index_step_mapping():
    assert config.index_to_step(0) == 15025 and config.index_to_step(599) == 30000
    assert config.step_to_index(30000) == 599
    assert config.ldm_path(599).name == "LDM_30000stp.vtk"


def test_kernel_constants_are_consistent():
    assert abs(config.C6_W0_LATTICE - config.W0_CONCN) < 1e-6
    assert abs(config.C6_W0_LATTICE - 1365.0 / (64.0 * np.pi * 27.0)) < 1e-12