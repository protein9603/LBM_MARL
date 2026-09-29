"""Unit tests for scripts/validate_slabs.py helpers on synthetic arrays (no raw data, no cache)."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.scripts.validate_slabs import (band_tree, compare_values, exact_density, growth_table,
                                               pearson_r, sample_in_occupied_cells, sample_uniform)


def test_sample_in_occupied_cells_stays_inside_occupied_cells():
    grid = SlabGrid(x0=100.0, y0=-20.0, nx=6, ny=5, res=5.0)
    dens = np.zeros((grid.ny, grid.nx))
    occupied = {(1, 2), (3, 0), (4, 5)}                          # (iy, ix)
    for iy, ix in occupied:
        dens[iy, ix] = 0.5
    rng = np.random.default_rng(1)
    xy = sample_in_occupied_cells(dens, grid, 500, rng)
    assert xy.shape == (500, 2)
    ix = np.floor((xy[:, 0] - grid.x0) / grid.res).astype(int)
    iy = np.floor((xy[:, 1] - grid.y0) / grid.res).astype(int)
    assert set(zip(iy.tolist(), ix.tolist())) <= occupied
    assert len(set(zip(iy.tolist(), ix.tolist()))) == len(occupied)     # every occupied cell gets hit
    # jitter: points are not at the cell centres
    fx = (xy[:, 0] - grid.x0) / grid.res - ix
    assert np.mean(np.abs(fx - 0.5) > 0.05) > 0.8
    with pytest.raises(ValueError):
        sample_in_occupied_cells(np.zeros((3, 3)), grid, 5, rng)
    u = sample_uniform(grid, 300, rng)
    assert np.all(u[:, 0] >= grid.x0) and np.all(u[:, 0] <= grid.x0 + grid.res * grid.nx)
    assert np.all(u[:, 1] >= grid.y0) and np.all(u[:, 1] <= grid.y0 + grid.res * grid.ny)


def test_exact_density_matches_kernel_self_term_and_band():
    z = config.DRONE_Z
    particles = np.array([[12.5, 22.5, z], [40.0, 40.0, z + config.KERNEL_H + 0.5]], np.float32)
    pts, tree = band_tree(particles, z)
    assert pts.shape == (1, 3)                                    # second particle is outside the vertical band
    d = exact_density(np.array([[12.5, 22.5], [12.5 + config.KERNEL_H + 1.0, 22.5], [40.0, 40.0]]), z, pts, tree)
    assert abs(d[0] - config.C6_W0_SI) < 1e-12
    assert d[1] == 0.0 and d[2] == 0.0
    pts0, tree0 = band_tree(np.zeros((0, 3)), z)
    assert tree0 is None and np.all(exact_density(np.array([[1.0, 2.0]]), z, pts0, tree0) == 0.0)


def test_compare_values_statistics():
    exact = np.array([0.0, 0.0, 1e-5, 0.5, 1.0, 2.0, 4.0])
    slab = np.array([0.0, 0.3, 0.0, 1.0, 2.0, 4.0, 8.0])         # exactly 2x where exact > threshold
    st = compare_values(slab, exact)
    assert st["n"] == 7
    assert st["r_raw"] > 0.99 and abs(pearson_r(2.0 * exact, exact) - 1.0) < 1e-12   # linear -> r = 1
    assert st["n_rel_err"] == 4 and abs(st["rel_err_median"] - 1.0) < 1e-12 and abs(st["rel_err_p90"] - 1.0) < 1e-12
    assert abs(st["frac_slab_zero_exact_pos"] - 1 / 7) < 1e-12    # index 2
    assert abs(st["frac_slab_pos_exact_zero"] - 1 / 7) < 1e-12    # index 1
    assert abs(st["zero_agreement"] - 5 / 7) < 1e-12
    assert np.isnan(pearson_r(np.ones(5), np.arange(5)))


def test_growth_table_numbers_and_markdown():
    metas = {
        400: {"n_airborne": 300, "counts_by_source_airborne": {"101": 100, "102": 200}},
        500: {"n_airborne": 360, "counts_by_source_airborne": {"101": 120, "102": 240}},
        599: {"n_airborne": 450, "counts_by_source_airborne": {"101": 150, "102": 300}},
    }
    numbers, table = growth_table(metas)
    assert numbers["indices"] == [400, 500, 599]
    assert numbers["steps"][599] == config.index_to_step(599) == 30000
    assert abs(numbers["growth_ratio_total"] - 1.5) < 1e-12 and abs(numbers["growth_pct_total"] - 50.0) < 1e-9
    assert abs(numbers["growth_ratio_by_source"][101] - 1.5) < 1e-12
    lines = table.splitlines()
    assert lines[0].startswith("| source |") and "step 30000" in lines[0]
    assert lines[1] == "|---|---|---|---|---|"
    assert "| 101 | 100 | 120 | 150 | 1.500 |" in lines
    assert lines[-1].startswith("| **total** |") and "**1.500**" in lines[-1]