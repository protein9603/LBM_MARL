"""Import smoke test for scripts/fig_data.py: the two particle sets from a synthetic LdmFrame (no raw data)."""
import numpy as np

from srcloc_env import config
from srcloc_env.io.ldm_reader import frame_from_arrays
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.scripts import fig_data


def _frame():
    # source 101: 5 airborne, 3 deposited (z == DEPOSIT_Z, v == 0), 2 outflow (x >= X_OUTFLOW), 1 at DEPOSIT_Z but moving
    # source 102: 4 airborne, 1 deposited
    xyz = np.array([
        [500.0, 0.0, 5.0], [505.0, 0.0, 6.0], [510.0, 1.0, 4.0], [520.0, 2.0, 15.0], [530.0, -1.0, 2.0],
        [500.0, 0.0, config.DEPOSIT_Z], [501.0, 0.0, config.DEPOSIT_Z], [502.0, 0.0, config.DEPOSIT_Z],
        [config.X_OUTFLOW, 0.0, 3.0], [1322.6, 0.0, config.DEPOSIT_Z],
        [503.0, 0.0, config.DEPOSIT_Z],
        [600.0, 0.0, 5.0], [605.0, 0.0, 6.0], [610.0, 1.0, 4.0], [620.0, 2.0, 15.0],
        [600.0, 0.0, config.DEPOSIT_Z],
    ], dtype=np.float32)
    n = xyz.shape[0]
    p_type = np.array([101] * 11 + [102] * 5, dtype=np.int32)
    vel = np.ones((n, 3), np.float32)
    vel[5:8] = 0.0                      # deposited (101)
    vel[9] = 0.0                        # deposited but in the outflow pile-up
    vel[15] = 0.0                       # deposited (102)
    arrays = {"xyz": xyz, "p_type": p_type, "velocity": vel,
              "concentration": np.ones(n, np.float32), "concn": np.ones(n, np.float32)}
    return frame_from_arrays(599, arrays)


def test_particle_sets_counts():
    frame = _frame()
    s101 = fig_data.particle_sets(frame, 101)
    assert s101[fig_data.SET_ALL].shape == (9, 3)          # 11 - 2 outflow
    assert s101[fig_data.SET_AIRBORNE].shape == (6, 3)     # 9 - 3 deposited (the moving z==1e-4 particle stays)
    s102 = fig_data.particle_sets(frame, 102)
    assert s102[fig_data.SET_ALL].shape == (5, 3) and s102[fig_data.SET_AIRBORNE].shape == (4, 3)
    assert s101[fig_data.SET_ALL].dtype == np.float64


def test_density_and_compare_synthetic():
    frame = _frame()
    grid = SlabGrid(x0=480.0, y0=-20.0, nx=12, ny=8, res=5.0)
    sets = fig_data.particle_sets(frame, 101)
    z_levels = (0.5, 10.0)
    trees = {k: fig_data.band_tree(v, z_levels) for k, v in sets.items()}
    d_all = fig_data.density_on_grid(*trees[fig_data.SET_ALL], grid, 0.5)
    d_air = fig_data.density_on_grid(*trees[fig_data.SET_AIRBORNE], grid, 0.5)
    assert d_all.shape == (8, 12) and np.all(d_all >= d_air)
    cmp_ = fig_data.compare_sets(d_all, d_air)
    assert cmp_["cells_all_pos"] > 0 and cmp_["ratio_mean"] > 1.0 and 0.0 <= cmp_["frac_all_pos_airborne_zero"] <= 1.0
    # above the kernel support the deposited particles cannot contribute
    d_all10 = fig_data.density_on_grid(*trees[fig_data.SET_ALL], grid, 10.0)
    d_air10 = fig_data.density_on_grid(*trees[fig_data.SET_AIRBORNE], grid, 10.0)
    assert np.allclose(d_all10, d_air10)
    numbers = fig_data.compute_numbers(frame, z_levels, (101, 102), grid, verbose=False)
    assert set(numbers["per_source"]) == {"101", "102"}
    assert numbers["per_source"]["101"]["n_deposited"] == 3
    assert abs(numbers["pooled"]["10.0"]["ratio_mean"] - 1.0) < 1e-12
    table = fig_data.markdown_table(numbers, (101, 102), z_levels)
    assert table.count("\n") == 4 and "pooled" in table