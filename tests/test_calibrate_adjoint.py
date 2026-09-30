"""Unit tests of the T1-3b helpers in scripts/calibrate_adjoint.py (plan 4.2b 캘리브레이션, S1 T1-3b, D5-1) on
synthetic arrays and small synthetic operators only (AdvectionDiffusionOperator.from_arrays; no raw data):
the shape statistic of a pure-offset field is 0, the support-mismatch fractions are computed correctly, the
grid search recovers the generating (K, lam, layer), and the comparison / table helpers behave as documented."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.pf.lbm_adjoint import AdjointParams, AdvectionDiffusionOperator
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.scripts import calibrate_adjoint as ca

M = 400


def _positive_pair(seed: int = 0, offset: float = 5.0):
    """g > g_min on M cells, slab = offset * g (pure scale offset, identical supports)."""
    rng = np.random.default_rng(seed)
    g = np.exp(rng.uniform(-12.0, -3.0, M))              # >> T1_3B_G_MIN = 1e-8
    return offset * g, g


def test_pure_offset_field_has_zero_shape_residual_and_no_support_mismatch():
    n, g = _positive_pair(0, offset=5.0)
    st, rp = ca.shape_stats_adjoint(n, g)
    assert st.n_cells == M and st.n_occupied == M and st.n_model_positive == M and st.n_floor_excluded == 0
    assert st.median_rho == pytest.approx(np.log(5.0), rel=1e-12)
    for v in (st.std, st.iqr, st.p90_p10, st.std_dense, st.p90_p10_dense, st.mass_weighted_std):
        assert v == pytest.approx(0.0, abs=1e-12)
    assert 0 < st.n_dense <= M
    assert st.ldm_mass_frac_model_zero == 0.0 and st.ldm_cell_frac_model_zero == 0.0
    assert st.ldm_mass_frac_below_floor == 0.0
    assert st.model_mass_frac_ldm_zero == 0.0 and st.model_cell_frac_ldm_zero == 0.0
    assert rp.shape == (M,) and np.allclose(rp, 0.0, atol=1e-12)


def test_support_mismatch_fractions():
    #            LDM present, model absent (g == 0): cells 0, 1 -> mass 1 + 2 = 3 of the LDM total 1+2+3+4 = 10
    #            model present, LDM absent (n == 0): cells 4, 5 -> model mass 0.5 + 1.5 = 2 of the model total 8
    #            0 < g <= g_min: cell 3 (LDM mass 4) -> excluded from the statistic but not "model absent"
    n = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0])
    g = np.array([0.0, 0.0, 2.0, 0.5 * config.T1_3B_G_MIN, 0.5, 1.5, 4.0])
    st, rp = ca.shape_stats_adjoint(n, g)
    assert st.n_occupied == 4 and st.n_model_positive == 5 and st.n_cells == 1 and st.n_floor_excluded == 1
    assert st.ldm_mass_frac_model_zero == pytest.approx(3.0 / 10.0)
    assert st.ldm_cell_frac_model_zero == pytest.approx(2.0 / 4.0)
    assert st.ldm_mass_frac_below_floor == pytest.approx(4.0 / 10.0)
    assert st.model_mass_frac_ldm_zero == pytest.approx((0.5 + 1.5 + 4.0) / (2.0 + 0.5 * config.T1_3B_G_MIN + 0.5 + 1.5 + 4.0))
    assert st.model_cell_frac_ldm_zero == pytest.approx(3.0 / 5.0)
    assert st.median_rho == pytest.approx(np.log(3.0 / 2.0)) and st.std == 0.0
    assert np.isfinite(rp[2]) and np.isnan(rp[[0, 1, 3, 4, 5, 6]]).all()


def test_no_common_support_gives_nan_stats_and_full_mismatch():
    n = np.array([1.0, 2.0, 0.0, 0.0])
    g = np.array([0.0, 0.0, 3.0, 4.0])
    st, rp = ca.shape_stats_adjoint(n, g)
    assert st.n_cells == 0 and np.isnan(st.std) and np.isnan(st.median_rho) and st.n_dense == 0
    assert st.ldm_mass_frac_model_zero == 1.0 and st.model_mass_frac_ldm_zero == 1.0 and np.isnan(rp).all()
    with pytest.raises(ValueError):
        ca.shape_stats_adjoint(n, g[:-1])
    with pytest.raises(ValueError):
        ca.shape_stats_adjoint(n, -g)


def test_shape_statistic_is_scale_invariant_but_shape_sensitive():
    n, g = _positive_pair(4, offset=1.0)
    st_a, _ = ca.shape_stats_adjoint(7.0 * n, g)
    st_b, _ = ca.shape_stats_adjoint(n, g)
    assert st_a.std == pytest.approx(st_b.std, abs=1e-12) and st_a.median_rho == pytest.approx(st_b.median_rho + np.log(7.0))
    rng = np.random.default_rng(5)
    n_noisy = n * np.exp(rng.normal(0.0, 1.0, M))
    st_c, _ = ca.shape_stats_adjoint(n_noisy, g)
    assert 0.8 < st_c.std < 1.2
    # the dense subset is the cells >= 1 % of the source maximum
    assert st_c.n_dense == int((n_noisy >= config.T1_3_INFO_DENSITY_FRACTION * n_noisy.max()).sum())


# ---------------------------------------------------------------------------------------- synthetic grid search
GRID = SlabGrid(x0=0.0, y0=0.0, nx=24, ny=20, res=5.0)
SRC_IDS = (101, 102, 103)                                 # 103 is a holdout source, 101 / 102 train
SRC_XY = np.array([[32.0, 47.0], [58.0, 52.0], [40.0, 30.0]])


def _uv(u: float, v: float) -> np.ndarray:
    uv = np.zeros((GRID.ny, GRID.nx, 2))
    uv[..., 0], uv[..., 1] = u, v
    return uv


def _truth(uv: np.ndarray, K: float, lam: float, scale: float = 3.0) -> np.ndarray:
    blocked = np.zeros((GRID.ny, GRID.nx), dtype=bool)
    op = AdvectionDiffusionOperator.from_arrays(GRID, uv, blocked, AdjointParams(K=K, lam=lam)).factorize()
    return np.stack([scale * op.solve_forward(xy).ravel() for xy in SRC_XY])


def test_run_grid_recovers_generating_combination_and_select_row():
    layers = {"a": None, "b": (10.0, 20.0)}
    uv_by_layer = {"a": _uv(1.5, 0.2), "b": _uv(0.8, -0.6)}
    K0, lam0 = 2.0, 0.02
    truth = _truth(uv_by_layer["b"], K0, lam0)
    blocked = np.zeros((GRID.ny, GRID.nx), dtype=bool)
    rows = ca.run_grid(GRID, uv_by_layer, blocked, SRC_IDS, SRC_XY, truth, train_ids=(101, 102), layers=layers,
                       k_candidates=(1.0, 2.0, 4.0), lam_candidates=(0.01, 0.02, 0.05))
    assert len(rows) == 18 and rows[0]["wind_layer"] == "a" and rows[-1]["wind_layer"] == "b"
    k, spread = ca.select_row(rows)
    assert (rows[k]["wind_layer"], rows[k]["K"], rows[k]["lam"]) == ("b", K0, lam0)
    assert rows[k]["mean_train_std_dense"] == pytest.approx(0.0, abs=1e-8)
    assert rows[k]["mean_train_std"] == pytest.approx(0.0, abs=1e-8)
    assert rows[k]["mean_train_median_rho"] == pytest.approx(np.log(3.0), abs=1e-8)
    assert rows[k]["per_source"]["103"]["std"] == pytest.approx(0.0, abs=1e-8)     # holdout evaluated too
    assert spread["key"] == config.T1_3B_SELECTION_KEY and spread["spread_all_rows"] > 0.0
    k2, _ = ca.select_row(rows, "mean_train_std")
    assert k2 == k
    with pytest.raises(ValueError):
        ca.run_grid(GRID, {"a": uv_by_layer["a"]}, blocked, SRC_IDS, SRC_XY, truth, train_ids=(101,), layers=layers)
    with pytest.raises(ValueError):
        ca.run_grid(GRID, uv_by_layer, blocked, SRC_IDS, SRC_XY, truth, train_ids=(999,), layers=layers)


def test_wall_cells_show_up_as_support_mismatch():
    uv = _uv(1.0, 0.0)
    blocked = np.zeros((GRID.ny, GRID.nx), dtype=bool)
    blocked[8:12, 14:16] = True                                        # a wall block downwind of the sources
    op = AdvectionDiffusionOperator.from_arrays(GRID, uv, blocked, AdjointParams(K=2.0, lam=0.02)).factorize()
    truth = np.stack([2.0 * op.solve_forward(xy).ravel() for xy in SRC_XY])
    truth[:, blocked.ravel()] = 0.05                                   # LDM mass inside the wall cells
    stats, rp, g = ca.evaluate_operator(op, SRC_XY, truth)
    assert g.shape == (3, GRID.ny * GRID.nx) and np.all(g[:, blocked.ravel()] == 0.0)
    for st in stats:
        assert st.ldm_cell_frac_model_zero > 0.0 and st.ldm_mass_frac_model_zero > 0.0
        assert st.std == pytest.approx(0.0, abs=1e-8) and st.model_mass_frac_ldm_zero == 0.0
    with pytest.raises(ValueError):
        ca.evaluate_operator(op, SRC_XY, truth[:, :-1])


# ---------------------------------------------------------------------------------------- comparison / table
def _ps(values: dict[int, tuple[float, float]]) -> dict[str, dict]:
    out = {}
    for s, (std, dense) in values.items():
        out[str(s)] = {"std": std, "std_dense": dense, "ldm_mass_frac_model_zero": 0.1, "model_mass_frac_ldm_zero": 0.2,
                       "n_cells": 10}
    return out


def test_compare_models_counts_and_table():
    ids = (101, 109, 110)
    ana = _ps({101: (3.0, 2.0), 109: (3.5, 1.5), 110: (5.0, 2.8)})
    adj = _ps({101: (2.0, 1.0), 109: (4.0, 1.4), 110: (6.0, float("nan"))})
    cmp = ca.compare_models(ids, ana, adj)
    per = cmp["per_source"]
    assert per["101"]["adjoint_better_std_dense"] and per["101"]["adjoint_better_std"]
    assert per["109"]["adjoint_better_std_dense"] and not per["109"]["adjoint_better_std"]
    assert not per["110"]["adjoint_better_std_dense"]                  # NaN never counts as better
    c = cmp["counts"]
    assert c["open"]["n"] == 2 and c["open"]["adjoint_better_std_dense"] == 2            # 101, 109 (108/111/113 absent)
    assert c["trapped"]["n"] == 1 and c["trapped"]["adjoint_better_std_dense"] == 0       # 110
    assert c["holdout"]["n"] == 1 and c["holdout"]["sources"] == [109]                    # 109 open AND holdout
    assert c["all"]["n"] == 3 and c["all"]["adjoint_better_std"] == 1
    md = ca.table1_markdown(ids, ana, adj, "cap")
    lines = md.splitlines()
    assert lines[0] == "cap" and lines[2].startswith("| source | type | analytic std |") and len(lines) == 2 + 2 + 3
    assert "| 109 | holdout(open) | 3.50 | 1.50 | 4.00 | 1.40 | 0.100 | 0.200 | 10 |" in lines