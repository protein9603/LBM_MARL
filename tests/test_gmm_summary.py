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


def _fit_reference(xy, w, k, n_iter, rng):
    """The pre-D8-4 per-component einsum EM (kept as the numerical reference of the vectorised version)."""
    from srcloc_env.pf.gmm_summary import _weighted_kmeanspp, _merge_overlapping
    w = w / w.sum()
    n = xy.shape[0]
    means = _weighted_kmeanspp(xy, w, k, rng)
    covs = np.stack([np.eye(2) * max(np.var(xy, axis=0).mean(), config.GMM_COV_REG_M2)] * k)
    pis = np.full(k, 1.0 / k)
    reg = np.eye(2) * config.GMM_COV_REG_M2
    for _ in range(n_iter):
        logr = np.empty((n, k))
        for j in range(k):
            inv = np.linalg.inv(covs[j]); d = xy - means[j]
            q = np.einsum("ni,ij,nj->n", d, inv, d)
            logr[:, j] = np.log(pis[j] + 1e-300) - 0.5 * q - 0.5 * np.log(np.linalg.det(covs[j]))
        logr -= logr.max(axis=1, keepdims=True)
        r = np.exp(logr); r /= r.sum(axis=1, keepdims=True)
        rw = r * w[:, None]; nk = rw.sum(axis=0)
        for j in range(k):
            means[j] = (rw[:, j] @ xy) / nk[j]
            d = xy - means[j]
            covs[j] = (rw[:, j, None] * d).T @ d / nk[j] + reg
        pis = np.maximum(nk, 1e-12); pis /= pis.sum()
    return _merge_overlapping(pis, means, covs, config.GMM_MERGE_BHAT)


def test_vectorised_em_matches_reference_loop():
    """D8-4: the closed-form 2x2 vectorised E / M steps reproduce the per-component einsum EM to round-off."""
    for seed in range(4):
        rng = np.random.default_rng(seed)
        xy, w = _two_clusters(rng, w1=0.55 + 0.1 * seed)
        w = w * rng.gamma(2.0, 1.0, w.size); w /= w.sum()
        pis_r, means_r, covs_r = _fit_reference(xy, w, 3, 12, np.random.default_rng(100 + seed))
        g = fit_weighted_gmm(xy, w, k=3, n_iter=12, rng=np.random.default_rng(100 + seed))
        order = np.argsort(-pis_r)                                     # the reference may have merged slots
        pis_r, means_r, covs_r = pis_r[order], means_r[order], covs_r[order]
        n_ref = pis_r.shape[0]
        mask_r = pis_r >= config.GMM_MIN_WEIGHT
        assert np.array_equal(g.mask[:n_ref], mask_r) and not g.mask[n_ref:].any()
        assert np.allclose(g.weights[:n_ref][mask_r], (pis_r / pis_r[mask_r].sum())[mask_r], atol=1e-9)
        assert np.allclose(g.means[:n_ref][mask_r], means_r[mask_r], atol=1e-6)
        assert np.allclose(g.covs[:n_ref][mask_r], covs_r[mask_r], atol=1e-5)


def test_warm_start_tracks_previous_summary_and_is_cheaper():
    import time
    rng = np.random.default_rng(3)
    xy, w = _two_clusters(rng)
    cold = fit_weighted_gmm(xy, w, k=3, rng=np.random.default_rng(0))
    xy2 = xy + rng.normal(0.0, 1.0, xy.shape)                           # the belief moved a little
    warm = fit_weighted_gmm(xy2, w, k=3, n_iter=3, rng=np.random.default_rng(1), init=cold)
    cold2 = fit_weighted_gmm(xy2, w, k=3, rng=np.random.default_rng(1))
    assert np.linalg.norm(warm.means[0] - cold2.means[0]) < 2.0 and np.linalg.norm(warm.means[1] - cold2.means[1]) < 2.0
    assert abs(warm.weights[0] - cold2.weights[0]) < 0.02
    # a masked slot in init is re-seeded (no NaN / crash) and the result is still a valid summary
    masked = GmmSummary(np.array([1.0, 0.0, 0.0]), cold.means.copy(), cold.covs.copy(), np.array([True, False, False]))
    g = fit_weighted_gmm(xy2, w, k=3, n_iter=5, rng=np.random.default_rng(2), init=masked)
    assert np.all(np.isfinite(g.to_vector())) and g.mask.sum() >= 1
    t0 = time.perf_counter()
    for _ in range(10):
        fit_weighted_gmm(xy2, w, k=3, n_iter=3, rng=np.random.default_rng(1), init=cold)
    t_warm = (time.perf_counter() - t0) / 10
    t0 = time.perf_counter()
    for _ in range(10):
        fit_weighted_gmm(xy2, w, k=3, rng=np.random.default_rng(1))
    t_cold = (time.perf_counter() - t0) / 10
    assert t_warm < 0.6 * t_cold
