"""Unit tests for the Wendland C6 kernel gather (synthetic points only)."""
import numpy as np

from srcloc_env import config
from srcloc_env.sensor.kernel import c6_density_per_m3, c6_gather, c6_weight_lattice, wendland_c6_shape


def test_shape_boundaries():
    assert wendland_c6_shape(0.0) == 1.0
    assert wendland_c6_shape(1.0) == 0.0 and wendland_c6_shape(2.0) == 0.0
    q = np.linspace(0, 1, 11)
    w = wendland_c6_shape(q)
    assert np.all(np.diff(w) <= 0)  # monotone decreasing


def test_isolated_particle_reproduces_file_self_term():
    p = np.array([[100.0, 0.0, 5.0]])
    assert abs(c6_gather(p, p)[0] - config.W0_CONCN) < 1e-7


def test_pair_and_out_of_support():
    a = np.array([[0.0, 0.0, 0.0]])
    for r in (1.0, 3.0, 7.4):
        pts = np.vstack([a, [[r, 0.0, 0.0]]])
        expect = config.C6_W0_LATTICE + c6_weight_lattice(r)
        assert abs(c6_gather(a, pts)[0] - expect) < 1e-12
    far = np.vstack([a, [[7.6, 0.0, 0.0]]])
    assert abs(c6_gather(a, far)[0] - config.C6_W0_LATTICE) < 1e-12


def test_density_units_and_empty_input():
    a = np.array([[0.0, 0.0, 0.0]])
    assert abs(c6_density_per_m3(a, a)[0] - config.C6_W0_SI) < 1e-9
    assert c6_gather(a, np.zeros((0, 3)))[0] == 0.0


def test_query_off_particle_positions():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-20, 20, size=(500, 3))
    q = np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]])
    got = c6_gather(q, pts)
    d = np.linalg.norm(pts - q[0], axis=1)
    brute = c6_weight_lattice(d[d <= config.KERNEL_H]).sum()
    assert abs(got[0] - brute) < 1e-10 and got[1] == 0.0