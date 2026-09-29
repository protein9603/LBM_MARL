"""Unit tests for scripts/fig_scene.py on synthetic arrays (no raw data, Agg backend)."""
import json

import numpy as np

from srcloc_env import config
from srcloc_env.field.concentration_field import SlabFrame
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.scripts import fig_scene


def _building_map(tmp_path):
    nx, ny, res = 24, 20, 2.0
    x = 300.0 + res * np.arange(nx)
    y = 200.0 + res * np.arange(ny)
    hmap = np.zeros((nx, ny), np.float32)          # [ix, iy] as in config.OCC_AXIS_ORDER
    hmap[4:8, 3:6] = 12.0
    hmap[15, 14] = 115.0                           # tallest cell at x = 330, y = 228
    path = tmp_path / "occupancy_2m_flowframe.npz"
    np.savez(path, occ=hmap > 0, hmap=hmap, x=x, y=y,
             meta=np.array(json.dumps({"res": res, "x0": 300.0, "y0": 200.0, "nx": nx, "ny": ny})))
    return path


def _slab_frame():
    grid = SlabGrid(x0=300.0, y0=200.0, nx=10, ny=8, res=5.0)
    dens = np.zeros((2, 1, grid.ny, grid.nx), np.float32)
    dens[0, 0, 2, 3] = 4.0
    dens[0, 0, 2, 4] = 1.0
    dens[1, 0, 5, 6] = 0.5
    return SlabFrame(index=599, density=dens, z_levels=(config.DRONE_Z,), sources=(101, 110), grid=grid)


def test_load_building_map_axis_order_and_tallest_cell(tmp_path):
    bmap = fig_scene.load_building_map(_building_map(tmp_path))
    assert bmap.hmap.shape == (24, 20) and bmap.res == 2.0
    assert bmap.tallest_cell_xy() == (330.0, 228.0, 115.0)
    assert bmap.extent == (300.0, 348.0, 200.0, 240.0)
    img = bmap.image()
    assert img.shape == (20, 24) and img.mask.sum() == 20 * 24 - 13     # (ny, nx), free cells masked
    assert img[14, 15] == 115.0


def test_slab_helpers():
    frame = _slab_frame()
    s = fig_scene.sum_sources_at_z(frame, config.DRONE_Z)
    assert s.shape == (8, 10) and s[2, 3] == 4.0 and s[5, 6] == 0.5
    assert fig_scene.slab_extent(frame.grid) == (300.0, 350.0, 200.0, 240.0)
    vmin, vmax = fig_scene.log_limits(s)
    assert vmax == 4.0 and vmin == 0.5                                  # min positive above vmax / 10^decades
    vmin, vmax = fig_scene.log_limits(np.array([1e-9, 10.0]))
    assert vmax == 10.0 and abs(vmin - 10.0 / 10**config.FIG_LOG_DECADES) < 1e-12


def test_render_figures_synthetic(tmp_path):
    bmap = fig_scene.load_building_map(_building_map(tmp_path))
    frame = _slab_frame()
    wind = {"speed": 1.5, "u": 1.5, "v": 0.0}
    sources_xy = {101: (317.5, 212.5), 110: (332.5, 227.5)}
    f1, f1p, f2 = tmp_path / "fig1.png", tmp_path / "fig1_preview.png", tmp_path / "fig2.png"
    n1 = fig_scene.render_figure1(bmap, fig_scene.sum_sources_at_z(frame), frame.grid, 30000, wind, f1, f1p,
                                  sources_xy=sources_xy)
    n2 = fig_scene.render_figure2(frame, bmap, f2, highlight=110, sources_xy=sources_xy)
    assert f1.stat().st_size > 0 and f1p.stat().st_size > 0 and f2.stat().st_size > 0
    assert f1.stat().st_size > f1p.stat().st_size                       # 300 dpi vs 120 dpi
    assert n1["log_vmax"] == 4.0 and n1["arrow_speed_label_m_s"] == 1.5
    assert n2["per_source"]["101"] == {"max": 4.0, "mean_nonzero": 2.5, "occupied_cells": 2}
    assert n2["per_source"]["110"] == {"max": 0.5, "mean_nonzero": 0.5, "occupied_cells": 1}
    assert n2["zoom_building_cells"] == 13                              # whole synthetic map inside +-100 m