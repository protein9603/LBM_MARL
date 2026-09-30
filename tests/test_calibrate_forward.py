"""Unit tests of the T1-3 helpers in scripts/calibrate_forward.py (plan S1 T1-3 / 4.2 캘리브레이션 규칙, D4-3)
on synthetic arrays only (no raw data, no WindField file): the shape statistic of a pure-offset field is 0,
upwind and floor cells are excluded, the grid search recovers the generating (U, sigma_v), and the implied
kappa / markdown helpers behave as documented."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.pf.forward_model import ForwardParams, GaussianPlume, wind_aligned_coords
from srcloc_env.scripts import calibrate_forward as cf

M = 400


def _downwind_field(seed: int = 0, offset: float = 5.0):
    """g > 10 g_floor on M cells, all downwind (d > 0), slab = offset * g (pure scale offset)."""
    rng = np.random.default_rng(seed)
    g = np.exp(rng.uniform(-12.0, -3.0, M))              # >> 10 * 1e-9
    d = rng.uniform(1.0, 500.0, M)
    return offset * g, g, d


def test_pure_offset_field_has_zero_shape_residual():
    n, g, d = _downwind_field(0, offset=5.0)
    st, rp = cf.shape_stats(n, g, d)
    assert st.n_cells == M and st.n_occupied == M and st.n_floor_excluded == 0
    assert st.std == pytest.approx(0.0, abs=1e-12) and st.iqr == pytest.approx(0.0, abs=1e-12)
    assert st.p90_p10 == pytest.approx(0.0, abs=1e-12)
    assert st.median_rho == pytest.approx(np.log(5.0), rel=1e-12)
    assert st.upwind_cell_frac == 0.0 and st.upwind_mass_frac == 0.0
    assert st.std_dense == pytest.approx(0.0, abs=1e-12) and st.mass_weighted_std == pytest.approx(0.0, abs=1e-12)
    assert 0 < st.n_dense <= M and st.p90_p10_dense == pytest.approx(0.0, abs=1e-12)
    assert st.n_plan_cells == M and st.std_plan == pytest.approx(0.0, abs=1e-12)
    assert st.median_rho_plan == pytest.approx(np.log(5.0), rel=1e-12)
    assert rp.shape == (M,) and np.all(np.isfinite(rp)) and np.allclose(rp, 0.0, atol=1e-12)


def test_upwind_cells_are_excluded_and_reported_separately():
    n, g, d = _downwind_field(1, offset=2.0)
    k = 50
    n_up = np.full(k, 1e3)                                # huge slab values upwind
    g_up = np.full(k, config.FWD_G_FLOOR)                 # g = floor there (plan 4.2)
    d_up = -np.linspace(0.0, 100.0, k)                    # d <= 0 including exactly 0
    st, rp = cf.shape_stats(np.r_[n, n_up], np.r_[g, g_up], np.r_[d, d_up])
    assert st.n_cells == M and st.std == pytest.approx(0.0, abs=1e-12)     # statistic unchanged
    assert st.median_rho == pytest.approx(np.log(2.0), rel=1e-12)
    assert st.n_occupied == M + k
    assert st.upwind_cell_frac == pytest.approx(k / (M + k))
    assert st.upwind_mass_frac == pytest.approx(n_up.sum() / (n.sum() + n_up.sum()))
    assert np.all(np.isnan(rp[M:])) and np.all(np.isfinite(rp[:M]))


def test_floor_and_empty_cells_are_excluded():
    n, g, d = _downwind_field(2, offset=3.0)
    g2 = g.copy()
    g2[:10] = 5.0 * config.FWD_G_FLOOR                    # downwind but at (below 10x) the floor -> excluded
    n2 = n.copy()
    n2[10:30] = 0.0                                       # not occupied -> excluded, not counted as occupied
    st, rp = cf.shape_stats(n2, g2, d)
    assert st.n_cells == M - 30 and st.n_floor_excluded == 10 and st.n_occupied == M - 20
    assert np.isnan(rp[:30]).all() and np.isfinite(rp[30:]).all()
    assert st.std == pytest.approx(0.0, abs=1e-12)
    # the plan-literal statistic keeps the 10 floor cells (n > 0, d > 0): rho there is log(3 g / 5e-9) >> log 3
    assert st.n_plan_cells == M - 20 and st.median_rho_plan == pytest.approx(np.log(3.0), rel=1e-12)
    assert st.std_plan > 1.0


def test_no_qualifying_cell_gives_nan_stats():
    n, g, d = _downwind_field(3)
    st, rp = cf.shape_stats(n, g, -d)                     # everything upwind
    assert st.n_cells == 0 and np.isnan(st.std) and np.isnan(st.median_rho) and st.n_dense == 0
    assert st.n_plan_cells == 0 and np.isnan(st.std_plan) and np.isnan(st.median_rho_plan)
    assert st.upwind_cell_frac == 1.0 and st.upwind_mass_frac == 1.0 and np.isnan(rp).all()
    with pytest.raises(ValueError):
        cf.shape_stats(n, g, d[:-1])


def test_shape_statistic_is_scale_invariant_but_shape_sensitive():
    n, g, d = _downwind_field(4, offset=1.0)
    st_a, _ = cf.shape_stats(7.0 * n, g, d)
    st_b, _ = cf.shape_stats(n, g, d)
    assert st_a.std == pytest.approx(st_b.std, abs=1e-12)
    n_shape = n * np.exp(0.01 * d)                        # distance-dependent mismatch
    st_c, _ = cf.shape_stats(n_shape, g, d)
    assert st_c.std > 0.5 and st_c.p90_p10 > st_c.iqr > 0.0


def test_dense_and_mass_weighted_extras_ignore_a_sparse_fringe():
    """A fringe of tiny-density cells with wild rho' inflates std but not std_dense / mass_weighted_std."""
    n, g, d = _downwind_field(5, offset=4.0)
    k = 200
    rng = np.random.default_rng(55)
    g_fr = np.exp(rng.uniform(-20.0, -16.0, k))           # far-crosswind cells: tiny g, still > 10 g_floor? no: force it
    g_fr = np.maximum(g_fr, 20.0 * config.FWD_G_FLOOR)
    n_fr = np.full(k, 1e-9 * n.max())                     # single-particle-like densities, far below 1 % of max
    d_fr = rng.uniform(1.0, 500.0, k)
    st, _ = cf.shape_stats(np.r_[n, n_fr], np.r_[g, g_fr], np.r_[d, d_fr], dense_fraction=0.01)
    n_dense_expected = int((n > 0.01 * n.max()).sum())    # the offset field itself spans 4 decades of g
    assert st.n_cells == M + k and st.n_dense == n_dense_expected and 0 < n_dense_expected < M
    assert st.std > 1.0                                    # fringe dominates the plain std
    assert st.std_dense == pytest.approx(0.0, abs=1e-9)    # dense cells are a pure offset
    assert st.mass_weighted_std < 0.05                     # fringe carries ~0 mass
    assert 0.0 <= st.p90_p10_dense < 1e-9


def _synthetic_scene(U_true: float, sv_true: float, q: float = 100.0):
    """Two 'sources' with slabs = q * g of the global-mode plume (U_true, sv_true) on a small cell grid."""
    xs = np.linspace(0.0, 300.0, 31)
    ys = np.linspace(-100.0, 100.0, 21)
    xx, yy = np.meshgrid(xs, ys)
    cells = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, config.DRONE_Z)])
    src_xy = np.array([[20.0, 0.0], [60.0, 30.0]])
    truth = GaussianPlume(ForwardParams(U=U_true, sigma_v=sv_true))
    g = truth.unit_response(src_xy, cells)
    d, _ = wind_aligned_coords(src_xy, cells[:, :2], 0.0)
    slabs = np.where(d > 0, q * g, 0.0)                   # no mass upwind
    return src_xy, cells, slabs, q


def test_grid_search_recovers_generating_combo_global_mode():
    U_true, sv_true = config.FWD_U_CANDIDATES[1], config.FWD_SIGMA_V_CANDIDATES[2]
    src_xy, cells, slabs, q = _synthetic_scene(U_true, sv_true)
    ids = (101, 102)                                      # both training sources
    rows = cf.run_grid(ids, src_xy, cells, slabs, None, train_ids=ids, wind_modes=("global",))
    assert len(rows) == len(config.FWD_U_CANDIDATES) * len(config.FWD_SIGMA_V_CANDIDATES)
    k, spread = cf.select_combo(rows)
    assert rows[k]["U"] == U_true and rows[k]["sigma_v"] == sv_true and rows[k]["wind_mode"] == "global"
    assert rows[k]["mean_train_std"] == pytest.approx(0.0, abs=1e-9)
    assert rows[k]["mean_train_std_dense"] == pytest.approx(0.0, abs=1e-9)
    assert rows[k]["mean_train_mass_weighted_std"] == pytest.approx(0.0, abs=1e-9)
    assert rows[k]["mean_train_median_rho"] == pytest.approx(np.log(q), rel=1e-9)
    assert spread["spread_chosen_mode"] > 0.0 and spread["n_rows_chosen_mode"] == len(rows)
    assert set(rows[k]["per_source"]) == {"101", "102"}
    for st in rows[k]["per_source"].values():
        assert st["upwind_cell_frac"] == 0.0 and st["n_cells"] > 0
    for key in cf.INFO_KEYS:                              # the informational statistics agree on a noise-free scene
        assert cf.select_combo(rows, key)[0] == k
    assert rows[k]["mean_train_std_plan"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        cf.run_grid(ids, src_xy, cells, slabs, None, train_ids=(999,), wind_modes=("global",))
    with pytest.raises(ValueError):
        cf.make_plume("local", 1.0, 0.5, None)


def test_select_combo_ignores_nan_and_breaks_ties_first():
    rows = [{"wind_mode": "global", "U": 1.0, "sigma_v": 0.3, "mean_train_std": float("nan")},
            {"wind_mode": "global", "U": 1.0, "sigma_v": 0.5, "mean_train_std": 0.8},
            {"wind_mode": "local", "U": 1.0, "sigma_v": 0.3, "mean_train_std": 0.8},
            {"wind_mode": "local", "U": 2.0, "sigma_v": 0.3, "mean_train_std": 0.85}]
    k, spread = cf.select_combo(rows)
    assert k == 1
    assert spread["spread_chosen_mode"] == pytest.approx(0.0) and not spread["discriminable"]
    assert spread["plan_fallback_if_not_discriminable"] == {"U": config.FWD_DEFAULT_U, "sigma_v": config.FWD_DEFAULT_SIGMA_V}
    with pytest.raises(ValueError):
        cf.select_combo([{"wind_mode": "global", "U": 1.0, "sigma_v": 0.3, "mean_train_std": float("nan")}])


def test_implied_kappa_and_table_helpers():
    med = np.log([200.0, 300.0])
    k = cf.implied_kappa(med, k0=10.0)
    assert k["m_bar"] == pytest.approx(med.mean()) and k["exp_m_bar_particles_per_s"] == pytest.approx(np.sqrt(200.0 * 300.0))
    assert k["kappa_ref_implied"] == pytest.approx(10.0 * np.sqrt(6e4)) and len(k["per_source_kappa"]) == 2
    with pytest.raises(ValueError):
        cf.implied_kappa([np.nan, 1.0])
    assert cf.source_type(109) == "holdout(open)" and cf.source_type(103) == "holdout"
    assert cf.source_type(110) == "trapped" and cf.source_type(101) == "open" and cf.source_type(104) == "train"
    per = {"101": {"std": 0.5, "p90_p10": 1.2, "median_rho": 5.0, "n_cells": 10, "upwind_cell_frac": 0.1,
                   "upwind_mass_frac": 0.05},
           "110": {"std": 2.5, "p90_p10": 6.0, "median_rho": 3.0, "n_cells": 4, "upwind_cell_frac": 0.6,
                   "upwind_mass_frac": 0.7}}
    md = cf.markdown_table((101, 110), per, "cap")
    lines = md.splitlines()
    assert lines[0] == "cap" and lines[2].startswith("| source | type |") and len(lines) == 6
    assert lines[2].endswith("| upwind cell frac | upwind mass frac |")
    assert "| 110 | trapped | 2.50 | 6.00 | 3.00 | 4 | 0.60 | 0.70 |" in lines


def test_floor_rule_cell_set_depends_on_candidate_but_plan_literal_set_does_not():
    """Gap test (review D4-3): a narrower plume clips more far-crosswind cells to the floor, so n_cells of the
    floor-rule statistic changes with the candidate while the plan-literal set (n > 0, d > 0) is fixed and
    keeps those cells (with rho = log(n / g_floor))."""
    src_xy, cells, slabs, q = _synthetic_scene(config.FWD_U_CANDIDATES[0], config.FWD_SIGMA_V_CANDIDATES[2])
    # a truth slab that is wider than every candidate: q * g of a much wider plume, no mass upwind
    wide = GaussianPlume(ForwardParams(U=config.FWD_U_CANDIDATES[0], sigma_v=5.0 * config.FWD_SIGMA_V_CANDIDATES[2]))
    d, _ = wind_aligned_coords(src_xy, cells[:, :2], 0.0)
    slabs = np.where(d > 0, q * wide.unit_response(src_xy, cells), 0.0)
    narrow = cf.make_plume("global", config.FWD_U_CANDIDATES[2], config.FWD_SIGMA_V_CANDIDATES[0], None)
    broad = cf.make_plume("global", config.FWD_U_CANDIDATES[0], config.FWD_SIGMA_V_CANDIDATES[2], None)
    st_n = cf.evaluate_sources(narrow, src_xy, cells, slabs)[0][0]
    st_b = cf.evaluate_sources(broad, src_xy, cells, slabs)[0][0]
    assert st_n.n_floor_excluded > st_b.n_floor_excluded >= 0
    assert st_n.n_cells < st_b.n_cells                                      # floor-rule set shrinks with the candidate
    assert st_n.n_plan_cells == st_b.n_plan_cells == st_n.n_cells + st_n.n_floor_excluded   # plan set is fixed
    assert st_n.std_plan > st_n.std and st_b.std_plan >= st_b.std           # dropped floor cells only lower the std
    assert st_b.std < st_n.std and st_b.std_plan < st_n.std_plan            # both rank the wider candidate first