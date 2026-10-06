"""Full-coverage lawnmower: row layout, sub-area split, boustrophedon order, and the policy on the synthetic scene."""
from __future__ import annotations

import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.baselines.coverage import assign_groups, coverage_waypoints, make_rows, row_order, split_rows
from srcloc_env.baselines.policies import LawnmowerAlongWindPolicy, LawnmowerPolicy, make_policy
from srcloc_env.env.multi_agent import MultiDroneEnv
from srcloc_env.env.source_env import Scene, SourceLocEnv
from srcloc_env.pf.sph_adjoint import AdjointParams
from srcloc_env.preprocess.gridder import SlabGrid
from tests.test_source_env import DOMAIN_X, DOMAIN_Y, SOURCES, SyntheticBackend, _obstacles, _wind

BOX = ((330.0, 1315.0), (-550.0, 550.0))
KW = dict(sources=(1, 2), n_particles=200, prior_x=DOMAIN_X, prior_y=DOMAIN_Y, start_min_dist=50.0, frame_range=(400, 599),
          terminate_on_success=False)


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.build(_wind(), _obstacles(), SyntheticBackend(), SOURCES, AdjointParams(K=16.0, lam=0.005), SlabGrid(0.0, -75.0, 40, 28, 5.0))


def test_rows_cover_the_box_with_the_requested_spacing():
    rows = make_rows("cross", 150.0, BOX)
    assert len(rows) == 7                                            # ceil(985 / 150)
    xs = np.array([a[0] for a, b in rows])
    assert np.allclose(np.diff(xs), 985.0 / 7) and xs[0] > 330.0 and xs[-1] < 1315.0
    assert all(a[1] == -550.0 and b[1] == 550.0 and a[0] == b[0] for a, b in rows)          # transects along y
    along = make_rows("along", 100.0, BOX)
    assert len(along) == 11 and all(a[0] == 330.0 and b[0] == 1315.0 and a[1] == b[1] for a, b in along)
    with pytest.raises(ValueError):
        make_rows("diagonal", 100.0, BOX)


def test_split_assign_and_order():
    groups = split_rows(7, 2)
    assert [g.tolist() for g in groups] == [[0, 1, 2, 3], [4, 5, 6]]
    rows = make_rows("cross", 150.0, BOX)
    east_first = assign_groups(groups, rows, np.array([[1250.0, 0.0], [400.0, 0.0]]))
    assert east_first[0].tolist() == [4, 5, 6] and east_first[1].tolist() == [0, 1, 2, 3]   # each drone gets the sub-area next to it
    # order: start at the nearest row, continue towards the side with more rows, then the skipped rows
    assert row_order(np.arange(7), rows, np.array([330.0, 0.0])) == [0, 1, 2, 3, 4, 5, 6]
    assert row_order(np.arange(7), rows, np.array([1315.0, 0.0])) == [6, 5, 4, 3, 2, 1, 0]
    mid = row_order(np.arange(7), rows, np.array([rows[2][0][0], 0.0]))
    assert mid[0] == 2 and mid[1:5] == [3, 4, 5, 6] and mid[5:] == [1, 0]             # more rows on the east side first, then come back


def test_waypoints_boustrophedon_partition_and_free(scene):
    starts = np.array([[20.0, -60.0], [180.0, 60.0]])
    box = (DOMAIN_X, DOMAIN_Y)
    plans, info = coverage_waypoints(starts, box, "cross", 50.0, 10.0, scene.obstacles, config.DRONE_Z)
    assert info["n_rows"] == 4 and sorted(sum(info["orders"], [])) == [0, 1, 2, 3]       # every row visited exactly once, by one drone
    assert info["groups"] == [[0, 1], [2, 3]] and info["orders"][0][0] == 0 and info["orders"][1][0] == 3
    for plan in plans:
        assert np.all(scene.obstacles.is_free(plan, config.DRONE_Z))
        assert np.all((plan[:, 0] >= DOMAIN_X[0]) & (plan[:, 0] <= DOMAIN_X[1]) & (plan[:, 1] >= DOMAIN_Y[0]) & (plan[:, 1] <= DOMAIN_Y[1]))
    # boustrophedon: consecutive rows are traversed in opposite y directions (first and last y of a row are swapped in the next)
    one, _ = coverage_waypoints(starts[:1], box, "cross", 50.0, 10.0, None, config.DRONE_Z)
    ys = one[0][:, 1]
    first_row, second_row = ys[:16], ys[16:32]                                              # 150 m rows at 10 m spacing -> 16 points per row
    assert np.sign(first_row[-1] - first_row[0]) == -np.sign(second_row[-1] - second_row[0]) != 0


def _visited_rows(positions: np.ndarray, rows, tol: float = 12.0) -> set[int]:
    return {k for k, (a, b) in enumerate(rows) if np.any(np.abs(positions[:, 0] - a[0]) <= tol)}


def test_policy_sweeps_all_rows_single_and_split_between_two_drones(scene):
    rows = make_rows("cross", 50.0, (DOMAIN_X, DOMAIN_Y))
    env = SourceLocEnv(scene, max_steps=260, **KW)
    pol = LawnmowerPolicy(spacing=50.0, wp_spacing=10.0)
    obs, info = env.reset(seed=3, options={"source": 1, "start_xy": (20.0, -60.0)})
    pol.reset(env, info)
    track = [env.xy.copy()]
    for _ in range(240):
        obs, r, term, trunc, info = env.step(int(pol.act(env, obs, info)[0]))
        track.append(env.xy.copy())
        assert scene.obstacles.is_free(env.xy, config.DRONE_Z)
    assert _visited_rows(np.array(track), rows) == {0, 1, 2, 3}                    # one drone covers the whole box
    env2 = MultiDroneEnv(scene, n_drones=2, max_steps=150, **KW)
    pol2 = LawnmowerPolicy(spacing=50.0, wp_spacing=10.0)
    obs, info = env2.reset(seed=3, options={"source": 1, "start_xy": np.array([[20.0, -60.0], [180.0, 60.0]])})
    pol2.reset(env2, info)
    tracks = [[env2.xys[d].copy()] for d in range(2)]
    for _ in range(100):
        obs, r, term, trunc, info = env2.step(pol2.act(env2, obs, info))
        for d in range(2):
            tracks[d].append(env2.xys[d].copy())
    v0, v1 = _visited_rows(np.array(tracks[0]), rows), _visited_rows(np.array(tracks[1]), rows)
    assert v0 == {0, 1} and v1 == {2, 3} and v0 | v1 == {0, 1, 2, 3}              # each drone sweeps its own sub-area, together all of it


def test_variants_and_registry():
    assert make_policy("lawnmower").name == "lawnmower" and make_policy("lawnmower_alongwind").axis == "along"
    assert make_policy("lawnmower_band").name == "lawnmower_band"
    assert LawnmowerAlongWindPolicy().spacing == config.LAWN_ALONG_SPACING_M and LawnmowerPolicy().spacing == config.LAWN_ROW_SPACING_M
