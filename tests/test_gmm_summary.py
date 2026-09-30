"""Unit tests for the GMM belief summary (synthetic weighted particle clouds)."""
import numpy as np

from srcloc_env import config
from srcloc_env.pf.gmm_summary import GmmSummary, fit_weighted_gmm, order_flip_rate, total_variation_distance


def _two_clusters(rng, n=2000, w1=0.7):
    n1 = int(n * w1)
    a = rng.normal([500.0, 100.0], [20.0, 10.0], size=(n1, 2))
    b = rng.normal([800.0, -200.0], [15.0, 15.0], size=(n - n1, 2))
    xy = np.vstack([a, b])
    w = np.full(n, 1.0 / n)
    return xy, w


def test_recovers_two_clusters_sorted_by_weight():
    rng = np.random.default_rng(0)
    xy, w = _two_clusters(rng)
    g = fit_weighted_gmm(xy, w, k=3, rng=rng)
    assert g.k == 3 and np.all(np.diff(g.weights) <= 1e-12)          # sorted descending
    assert abs(g.weights[0] - 0.7) < 0.05
    assert np.linalg.norm(g.means[0] - [500.0, 100.0]) < 5.0
    assert np.linalg.norm(g.means[1] - [800.0, -200.0]) < 5.0
    assert abs(np.sqrt(g.covs[0][0, 0]) - 20.0) < 3.0 and abs(np.sqrt(g.covs[0][1, 1]) - 10.0) < 2.0
    assert g.top_sigma() > 15.0


def test_padding_and_mask_for_a_single_cluster():
    rng = np.random.default_rng(1)
    xy = rng.normal([600.0, 0.0], [10.0, 10.0], size=(1500, 2))
    w = np.full(1500, 1.0 / 1500)
    g = fit_weighted_gmm(xy, w, k=3, rng=rng)
    assert g.mask[0]
    assert abs(g.weights[g.mask].sum() - 1.0) < 1e-9
    assert np.all(g.weights[~g.mask] == 0.0)


def test_weights_matter():
    rng = np.random.default_rng(2)
    xy, _ = _two_clusters(rng, w1=0.5)
    w = np.where(xy[:, 0] < 650, 0.9, 0.1)              # up-weight the first cluster
    w /= w.sum()
    g = fit_weighted_gmm(xy, w, k=2, rng=rng)
    assert g.weights[0] > 0.8 and np.linalg.norm(g.means[0] - [500.0, 100.0]) < 5.0


def test_vector_layout_and_normalisation():
    rng = np.random.default_rng(3)
    xy, w = _two_clusters(rng)
    g = fit_weighted_gmm(xy, w, k=3, rng=rng)
    v = g.to_vector(drone_xy=np.array([400.0, 0.0]))
    assert v.shape == (config.GMM_VECTOR_DIM,)
    k = config.GMM_K
    params = v[: k * 6].reshape(k, 6)
    assert abs(params[0, 1] - g.means[0, 0] / 1315.0) < 1e-12
    assert np.array_equal(v[k * 6: k * 6 + k], g.mask.astype(float))
    rel = v[k * 6 + k:].reshape(k, 2)
    assert abs(rel[0, 0] - (g.means[0, 0] - 400.0) / 1000.0) < 1e-12
    if not g.mask[-1]:
        assert np.all(params[-1] == 0.0) and np.all(rel[-1] == 0.0)


def test_total_variation_small_for_gaussian_cloud_and_flip_rate():
    rng = np.random.default_rng(4)
    xy = rng.normal([700.0, 50.0], [30.0, 20.0], size=(4000, 2))
    w = np.full(4000, 1.0 / 4000)
    g = fit_weighted_gmm(xy, w, k=3, rng=rng)
    tv = total_variation_distance(xy, w, g, cell=10.0)
    assert tv < config.T1_5_TV_MAX
    assert g.mask.sum() <= 2                 # near-identical EM components are merged


def test_order_is_stable_on_a_bimodal_cloud_across_refits():
    rng = np.random.default_rng(6)
    xy, w = _two_clusters(rng, w1=0.7)
    g = fit_weighted_gmm(xy, w, k=3, rng=np.random.default_rng(7))
    g2 = fit_weighted_gmm(xy + rng.normal(0, 1.0, size=xy.shape), w, k=3, rng=np.random.default_rng(8))
    assert g.mask.sum() == 2 and g2.mask.sum() == 2
    assert order_flip_rate(g, g2) == 0.0