"""Unit tests for the T1-5 fidelity helpers of scripts/validate_t1_5.py (plan S1 T1-5 / 4.4) on synthetic weighted
particle clouds only (no data): per-snapshot TV / integrated TV / noise floor / flip rate / vector, the per-source
analysis and summary, the verdict, the nearest-snapshot selection and the figure on an in-memory snapshot without an
obstacle map."""
import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.pf.gmm_summary import total_variation_distance
from srcloc_env.scripts.validate_t1_5 import (analyse_snapshots, make_figure, nearest_snapshot_indices, normalised_weights,
                                               sample_gmm, snapshot_fidelity, summarise, tv_distance_integrated,
                                               tv_noise_floor, verdict)


def gaussian_cloud(rng: np.random.Generator, centre=(600.0, 50.0), sigma=(12.0, 8.0), n: int = 2000):
    xy = rng.normal(centre, sigma, size=(n, 2)).astype(np.float32)
    logw = np.full(n, -np.log(n), dtype=np.float32)
    return xy, logw


def test_snapshot_fidelity_gaussian_cloud_small_tv_and_no_flip():
    rng = np.random.default_rng(0)
    xy, logw = gaussian_cloud(rng)
    st, gmm = snapshot_fidelity(xy, logw, drone_xy=np.array([500.0, 0.0]))
    assert st["tv"] < config.T1_5_TV_MAX and st["tv_integrated"] < st["tv"] + 1e-9
    assert 0.0 < st["tv_noise_floor"] < config.T1_5_TV_MAX and 0.0 <= st["tv_info"] <= 1.0
    assert st["n_valid"] == 1 and st["flip_rate"] is None
    assert abs(st["top_sigma_m"] - 12.0) < 2.0
    assert len(st["vector"]) == config.GMM_VECTOR_DIM and st["vector"][0] == pytest.approx(1.0)
    assert st["neff"] == pytest.approx(2000.0)
    # identical cloud again: same fit (seeded), zero flips
    st2, _ = snapshot_fidelity(xy, logw, prev=gmm)
    assert st2["flip_rate"] == 0.0 and st2["tv"] == pytest.approx(st["tv"])


def test_integrated_tv_removes_centre_point_bias_for_a_narrow_cloud():
    """sd 3 m cloud on 10 m cells: the centre rule overestimates the TV, the sub-cell integration does not."""
    rng = np.random.default_rng(4)
    xy, logw = gaussian_cloud(rng, sigma=(3.0, 3.0), n=4000)
    st, gmm = snapshot_fidelity(xy, logw, noise_floor=False)
    w = normalised_weights(logw)
    assert st["tv_noise_floor"] is None
    assert st["tv"] == pytest.approx(total_variation_distance(xy.astype(float), w, gmm))
    assert st["tv_integrated"] == pytest.approx(tv_distance_integrated(xy, w, gmm))
    centre_rule = total_variation_distance(xy.astype(float), w, gmm, subcells=1)
    assert st["tv_integrated"] < 0.05 < centre_rule                     # centre-point rule over-estimates the TV
    assert tv_distance_integrated(xy, w, gmm, n_sub=1) == pytest.approx(centre_rule, abs=1e-9)   # n_sub 1 == centre rule
    s = sample_gmm(gmm, 500, rng)
    assert s.shape == (500, 2) and np.linalg.norm(s.mean(0) - [600.0, 50.0]) < 1.5
    assert 0.0 < tv_noise_floor(gmm, 4000) < 0.5


def test_flip_rate_detects_swapped_order():
    rng = np.random.default_rng(1)
    a = rng.normal([500.0, 100.0], [10.0, 10.0], size=(1400, 2))
    b = rng.normal([800.0, -200.0], [10.0, 10.0], size=(600, 2))
    xy = np.vstack([a, b]).astype(np.float32)
    logw = np.full(2000, -np.log(2000.0), dtype=np.float32)
    st, g1 = snapshot_fidelity(xy, logw, noise_floor=False)
    assert st["n_valid"] == 2 and np.linalg.norm(g1.means[0] - [500.0, 100.0]) < 5.0
    # re-weight so that cluster b dominates -> the component order swaps -> flips on both matched components
    logw2 = np.where(xy[:, 0] > 650.0, np.log(0.8 / 600.0), np.log(0.2 / 1400.0)).astype(np.float32)
    st2, g2 = snapshot_fidelity(xy, logw2, prev=g1, noise_floor=False)
    assert np.linalg.norm(g2.means[0] - [800.0, -200.0]) < 5.0
    assert st2["flip_rate"] == pytest.approx(1.0)
    assert normalised_weights(logw2).sum() == pytest.approx(1.0)


def synthetic_snapshot(rng: np.random.Generator, n_snap: int = 7, n: int = 1500) -> dict:
    """Belief that shrinks from a wide cloud to a tight one around the true source; the MAP converges too."""
    true_xy = np.array([620.0, 40.0])
    steps = np.arange(n_snap) * config.T1_4_GMM_EVERY
    xs, lws, maps, drones = [], [], [], []
    for i in range(n_snap):
        sig = 80.0 / (i + 1)
        xy, logw = gaussian_cloud(rng, centre=true_xy + 60.0 / (i + 1), sigma=(sig, sig), n=n)
        xs.append(xy); lws.append(logw)
        maps.append(true_xy + 60.0 / (i + 1))
        drones.append(np.array([[900.0 - 25.0 * i, 0.0], [900.0 - 25.0 * i, 100.0]]))
    return {"steps": steps, "xy": np.stack(xs), "logw": np.stack(lws), "drone_xy": np.stack(drones),
            "true_xy": true_xy, "map_xy": np.stack(maps)}


def test_analyse_summarise_verdict_and_nearest():
    snap = synthetic_snapshot(np.random.default_rng(2))
    rec, gmms = analyse_snapshots(snap)
    assert len(gmms) == 7 and rec["steps"] == list(range(0, 35, 5))
    assert rec["flip_rate"][0] is None and all(f is not None for f in rec["flip_rate"][1:])
    assert rec["map_error_m"][0] == pytest.approx(np.hypot(60.0, 60.0)) and rec["map_error_m"][-1] == pytest.approx(np.hypot(60 / 7, 60 / 7))
    assert rec["top_sigma_m"][0] > rec["top_sigma_m"][-1]
    # success: sigma < 15 (from i >= 5: 80/6 = 13.3) and error < 20 (from i >= 4: 12 sqrt2 = 17) -> step 25
    assert rec["first_success_step"] == 25 and rec["success"][-1]
    sm = summarise(rec)
    assert sm["n_snapshots"] == 7 and 0.0 <= sm["tv_median"] <= 1.0 and sm["tv_max"] >= sm["tv_median"] >= sm["tv_min"]
    assert sm["first_success_step"] == 25 and sm["first_step_sigma_below"] == 25 and sm["first_step_error_below"] == 20
    assert sm["n_converged_snapshots"] == 2 and sm["n_transitions"] == 6 and sm["n_transitions_converged"] == 1
    assert sm["n_valid_min"] >= 1 and sm["flip_rate_mean"] >= 0.0 and len(sm["n_valid_by_step"]) == 7
    assert 0.0 <= sm["tv_integrated_median"] <= 1.0 and sm["tv_noise_floor_median"] > 0.0   # integrated <= centre only for narrow clouds
    v = verdict({"109": sm, "102": {**sm, "tv_median": 0.5, "tv_integrated_median": 0.5}})
    assert v["per_source"]["109"]["pass"] == (sm["tv_median"] < config.T1_5_TV_MAX and sm["flip_rate_mean_converged"] < config.T1_5_FLIP_MAX)
    assert v["per_source"]["102"]["pass"] is False and v["overall_pass"] is False and v["n_sources"] == 2
    assert v["informational"]["per_source"]["102"]["pass"] is False and v["informational"]["tv_key"] == "tv_integrated_median"
    assert nearest_snapshot_indices([0, 5, 10, 15, 20, 25, 30], (10, 12, 13, 60)) == [2, 2, 3, 6]


def test_make_figure_without_obstacle_map(tmp_path):
    snap = synthetic_snapshot(np.random.default_rng(3), n_snap=7)
    rec, gmms = analyse_snapshots(snap)
    out = tmp_path / "fig5_test.png"
    chosen = make_figure({109: snap}, {109: rec}, {109: gmms}, out, om=None, paths=None, fig_steps=(0, 10, 30))
    assert chosen == [[0, 10, 30]]
    assert out.exists() and out.with_name("fig5_test_preview.png").exists()