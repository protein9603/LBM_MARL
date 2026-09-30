"""Unit tests of the T1-3c / D7-2b helpers in scripts/calibrate_timeavg.py (plan S1 T1-3 / T1-4, 4.3; R22, R23) on
synthetic frames only (no raw data): a constant field has CV = CV_norm = 0, frames that are pure scalings of one
field have CV > 0 but CV_norm = 0, the streaming accumulator matches direct numpy moments, the dense-cell
fluctuation / dispersion helpers and the peak-offset helper behave as documented."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.scripts import calibrate_timeavg as ct

NS, NY, NX, T = 2, 6, 8, 12


def _base(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = rng.uniform(0.0, 1.0, (NS, NY, NX))
    b[:, :, 0] = 0.0                       # a zero column: CV must be NaN there
    return b


def test_constant_field_has_zero_cv_and_zero_cv_norm():
    b = _base()
    ta = ct.timeavg_stats(np.repeat(b[None], T, axis=0))
    assert ta.n_frames == T
    assert np.allclose(ta.mean, b)
    assert np.allclose(ta.std, 0.0, atol=1e-12)
    pos = b > 0
    assert np.allclose(ta.cv[pos], 0.0, atol=1e-12) and np.isnan(ta.cv[~pos]).all()
    assert np.allclose(ta.cv_norm[pos], 0.0, atol=1e-12) and np.isnan(ta.cv_norm[~pos]).all()
    assert ta.totals.shape == (T, NS) and np.allclose(ta.totals, b.reshape(NS, -1).sum(axis=1))


def test_scaled_frames_have_positive_cv_but_zero_cv_norm():
    b = _base(1)
    scale = np.linspace(1.0, 1.4, T)                          # the T0-4 growth trend
    frames = scale[:, None, None, None] * b[None]
    ta = ct.timeavg_stats(frames)
    pos = b > 0
    expected_cv = float(scale.std() / scale.mean())
    assert np.allclose(ta.cv[pos], expected_cv, rtol=1e-9)
    assert (ta.cv[pos] > 0.05).all()
    assert np.allclose(ta.cv_norm[pos], 0.0, atol=1e-9)
    assert np.allclose(ta.mean, scale.mean() * b)
    assert np.allclose(ta.mean_norm.reshape(NS, -1).sum(axis=1), 1.0)


def test_accumulator_matches_direct_moments_and_validates_input():
    rng = np.random.default_rng(2)
    frames = rng.gamma(0.5, 1.0, (T, NS, NY, NX))
    frames[:, :, 0, :] = 0.0
    acc = ct.TimeAverageAccumulator((NS, NY, NX))
    for f in frames:
        acc.add(f)
    ta = acc.result()
    assert np.allclose(ta.mean, frames.mean(axis=0)) and np.allclose(ta.std, frames.std(axis=0))
    tot = frames.reshape(T, NS, -1).sum(axis=2)
    fn = frames / tot[:, :, None, None]
    assert np.allclose(ta.mean_norm, fn.mean(axis=0)) and np.allclose(ta.std_norm, fn.std(axis=0))
    pos = ta.mean > 0
    assert np.allclose(ta.cv[pos], frames.std(axis=0)[pos] / frames.mean(axis=0)[pos])
    assert np.isnan(ta.cv[~pos]).all()
    with pytest.raises(ValueError):
        acc.add(np.zeros((NS, NY, NX + 1)))
    with pytest.raises(ValueError):
        acc.add(-np.ones((NS, NY, NX)))
    with pytest.raises(ValueError):
        ct.TimeAverageAccumulator((NS, NY, NX)).result()
    with pytest.raises(ValueError):
        ct.timeavg_stats(np.zeros((T, NY)))


def test_dense_fluctuation_stats_and_implied_dispersion():
    mean = np.zeros((NY, NX))
    mean[2, 2:6] = [10.0, 5.0, 1.0, 0.05]          # 0.05 < 1 % of 10 -> not dense; three dense cells
    cv = np.full((NY, NX), np.nan)
    cv[2, 2:6] = [2.0, 2.0, 2.0, 9.0]
    cv_norm = np.full((NY, NX), np.nan)
    cv_norm[2, 2:6] = [0.5, 1.0, 2.0, 9.0]
    d = ct.dense_mask(mean)
    assert d.sum() == 3 and d[2, 2] and d[2, 3] and d[2, 4] and not d[2, 5]
    st = ct.dense_fluctuation_stats(mean, cv, cv_norm)
    assert st["n_dense"] == 3 and st["max_mean"] == 10.0
    assert st["cv"]["median"] == pytest.approx(2.0) and st["cv"]["iqr"] == pytest.approx(0.0)
    assert st["cv_norm"]["median"] == pytest.approx(1.0) and st["cv_norm"]["iqr"] == pytest.approx(0.75)
    assert st["r_implied"]["r_median"] == pytest.approx(1.0)            # r = 1 / median(CV_norm)^2 (R23)
    assert st["r_implied"]["r_mean_cv2"] == pytest.approx(1.0 / np.mean([0.25, 1.0, 4.0]))
    assert st["r_implied_raw_cv"]["r_median"] == pytest.approx(0.25)
    empty = ct.dense_fluctuation_stats(np.zeros((NY, NX)), cv, cv_norm)
    assert empty["n_dense"] == 0 and np.isnan(empty["cv"]["median"]) and np.isnan(empty["r_implied"]["r_median"])
    assert ct.implied_dispersion(np.array([np.nan]))["n"] == 0


def test_pooled_dispersion_pools_dense_cells_of_the_listed_sources():
    b = _base(3)
    rng = np.random.default_rng(4)
    frames = b[None] * rng.gamma(4.0, 0.25, (T, NS, NY, NX))       # gamma-type fluctuations, CV ~ 0.5
    ta = ct.timeavg_stats(frames)
    src_ids = (101, 108)
    one = ct.pooled_dispersion(ta, src_ids, (101,))
    both = ct.pooled_dispersion(ta, src_ids, src_ids)
    d0, d1 = ct.dense_mask(ta.mean[0]), ct.dense_mask(ta.mean[1])
    assert one["n_dense"] == int(d0.sum()) and both["n_dense"] == int(d0.sum() + d1.sum())
    assert both["r_implied"]["r_median"] == pytest.approx(1.0 / np.median(np.r_[ta.cv_norm[0][d0], ta.cv_norm[1][d1]]) ** 2)
    assert 1.0 < both["r_implied"]["r_median"] < 20.0


def test_peak_offset_distance_and_direction():
    grid = SlabGrid(x0=0.0, y0=0.0, nx=NX, ny=NY, res=10.0)
    f = np.zeros((NY, NX))
    f[1, 5] = 3.0                                    # centre (55, 15)
    src = (15.0, 15.0)                               # cell (1, 1) centre
    o = ct.peak_offset(f, grid, src)
    assert o["peak_xy"] == [55.0, 15.0] and o["peak_value"] == 3.0
    assert o["peak_distance_m"] == pytest.approx(40.0) and o["peak_direction_deg"] == pytest.approx(0.0)
    assert o["centroid_distance_m"] == pytest.approx(40.0)
    f[4, 1] = 3.0                                    # second equal peak at (15, 45): centroid moves, argmax stays first
    o2 = ct.peak_offset(f, grid, src)
    assert o2["peak_xy"] == [55.0, 15.0]
    assert o2["centroid_xy"] == pytest.approx([35.0, 30.0])
    assert o2["centroid_direction_deg"] == pytest.approx(np.rad2deg(np.arctan2(15.0, 20.0)))
    z = ct.peak_offset(np.zeros((NY, NX)), grid, src)
    assert np.isnan(z["peak_distance_m"]) and z["peak_value"] == 0.0
    with pytest.raises(ValueError):
        ct.peak_offset(np.zeros((NY, NX + 1)), grid, src)


def test_config_constants_are_consistent():
    assert config.T1_3C_FRAME_RANGE == config.FRAME_RANGE_MODE_F
    assert config.T1_3C_DENSE_FRACTION == config.T1_3_INFO_DENSITY_FRACTION
    assert (config.T1_4_ADJOINT_K, config.T1_4_ADJOINT_LAM) in [(k, l) for k in config.T1_3C_ADJ_K_CANDIDATES for l in config.T1_3C_ADJ_LAMBDA_CANDIDATES]
    assert set(config.T1_3C_OFFSET_SOURCES) <= set(config.T1_3_OPEN_SOURCES)
    assert config.T1_3C_TIMEAVG_NPZ_TEMPLATE.format(lo=400, hi=599) == "slab_timeavg_400_599.npz"