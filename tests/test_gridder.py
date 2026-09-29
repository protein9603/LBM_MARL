"""Unit tests for the slab gridder on synthetic particles."""
import numpy as np

from srcloc_env import config
from srcloc_env.io.ldm_reader import frame_from_arrays
from srcloc_env.preprocess.gridder import SlabGrid, build_slabs, slab_for_source
from srcloc_env.sensor.kernel import c6_weight_lattice


def test_single_particle_lands_in_the_right_cell():
    grid = SlabGrid(x0=0.0, y0=0.0, nx=10, ny=8, res=5.0)
    p = np.array([[12.5, 22.5, 15.0]])            # centre of cell ix=2, iy=4 at z=15
    dens = slab_for_source(p, 15.0, grid)
    assert dens.shape == (8, 10)
    assert abs(dens[4, 2] - config.C6_W0_SI) < 1e-9        # self term / cell volume
    # neighbouring cell centre at 5 m: weight W(5)/V
    assert abs(dens[4, 3] - c6_weight_lattice(5.0) / config.LATTICE_CELL_VOLUME) < 1e-12
    assert dens[0, 9] == 0.0
    # particle 8 m above the slab does not contribute
    assert slab_for_source(np.array([[12.5, 22.5, 23.5]]), 15.0, grid).sum() == 0.0


def test_build_slabs_masks_deposited_and_outflow():
    grid = SlabGrid(x0=0.0, y0=0.0, nx=4, ny=4, res=5.0)
    xyz = np.array([[7.5, 7.5, 15.0], [7.5, 7.5, config.DEPOSIT_Z], [1320.0, 7.5, 15.0]], np.float32)
    vel = np.array([[0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]], np.float32)
    arrays = {"xyz": xyz, "p_type": np.array([101, 101, 101], np.int32), "velocity": vel,
              "concentration": np.ones(3, np.float32), "concn": np.ones(3, np.float32)}
    frame = frame_from_arrays(0, arrays)
    slabs = build_slabs(frame, (15.0,), sources=(101, 102), grid=grid)
    assert slabs.shape == (2, 1, 4, 4) and slabs.dtype == np.float16
    assert abs(float(slabs[0, 0, 1, 1]) - config.C6_W0_SI) < 1e-4   # float16 tolerance
    assert float(slabs[1].sum()) == 0.0                              # source 102 has no particles