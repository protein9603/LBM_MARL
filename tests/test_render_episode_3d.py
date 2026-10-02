"""Tests of scripts/render_episode_3d.py on synthetic step logs: everything except the drawing is a pure function (log loading with older / newer
keys, truth-frame derivation, step selection, count colours, trail / ellipse / star geometry, camera path, HUD and banner text, render plan, presets,
command line) and is tested without OpenGL; small off-screen renders (fake slab backend, synthetic building mesh, 160 x 90 .. 640 x 360 px) check the MP4,
the poster, the playback options (--stop-at-localisation, --illustrative, the 5 steps per second repeat) and that HUD, caption, banner, watermark, legend and
colour-bar panels never overlap at small window sizes."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from srcloc_env import config
from srcloc_env.field.concentration_field import SlabFrame
from srcloc_env.preprocess.gridder import SlabGrid
from srcloc_env.scripts import render_episode_3d as r

SRC = 101
TRUTH = (60.0, 0.0)
STARTS = np.array([[150.0, 40.0], [150.0, -40.0], [200.0, 0.0]])


def make_log(path: Path, T: int = 40, D: int = 2, mode: str | None = "F", frame0: int = 410, new_keys: bool = False, success_at: int | None = 25,
             start_far: bool = False, with_frame_array: bool = False) -> Path:
    """Write a synthetic step log in the run_eval.py format.  Drones fly from about (150, +-40) to the source; top sigma / MAP error fall below the
    primary success thresholds from step success_at (0-based) on, never when it is None."""
    rng = np.random.default_rng(3)
    t = np.arange(T)[:, None, None]
    start = STARTS[:D][None] + (np.array([300.0, 0.0]) if start_far else 0.0)
    xy = start + (np.array(TRUTH)[None, None, :] + 5.0 - start) * np.minimum(t / max(T - 1, 1), 1.0)
    y = rng.poisson(15.0 + 200.0 * np.exp(-np.hypot(*(xy - np.array(TRUTH)).transpose(2, 0, 1)) / 30.0)).astype(np.int64)
    top_sigma = np.linspace(200.0, 8.0, T)
    map_error = np.linspace(180.0, 12.0, T)
    if success_at is None:
        top_sigma[:], map_error[:] = 150.0, 150.0
    else:
        top_sigma[:success_at], map_error[:success_at] = 120.0, 100.0
        top_sigma[success_at:], map_error[success_at:] = 10.0, 12.5
    mu = np.zeros((T, 3, 2))
    mu[:] = np.array(TRUTH) + np.array([[3.0, 0.0], [40.0, 30.0], [-50.0, 10.0]])[None]
    cov = np.zeros((T, 3, 2, 2))
    for j, s in enumerate((1.0, 2.0, 3.0)):
        for k in range(T):
            cov[k, j] = np.diag([(top_sigma[k] * s) ** 2, (top_sigma[k] * s * 0.6) ** 2])
    arrays = dict(drone_xy=xy, y=y, action=np.zeros((T, D), dtype=np.int64), applied=np.ones((T, D), dtype=bool),
                  gmm_w=np.tile([0.6, 0.3, 0.1], (T, 1)), gmm_mu=mu, gmm_cov=cov, gmm_mask=np.ones((T, 3), dtype=bool),
                  map_xy=np.array(TRUTH)[None] + np.array([map_error, 0 * map_error]).T, entropy=np.linspace(7.0, 3.0, T), top_sigma=top_sigma,
                  map_error=map_error, truth_xy=np.array(TRUTH))
    meta = {"source": SRC, "frame": frame0, "scale": 1.0, "seed": 1, "success": success_at is not None, "method": "gmm_infotaxis", "n_drones": D,
            "start_type": "random"}
    if mode is not None:
        meta["mode"] = mode
        meta["max_steps"] = T
    if new_keys:
        arrays.update(reward=np.zeros(T), calib_inside=np.ones(T, dtype=bool), start_xy=start[0])
    if with_frame_array:
        arrays["frame"] = np.minimum(config.N_FILES - 1, frame0 + np.arange(T)) if mode == "T2" else np.full(T, frame0)
    np.savez_compressed(path, **arrays, meta=json.dumps(meta))
    return path


def drop_keys(src: Path, dst: Path, drop: tuple[str, ...], meta_update: dict | None = None) -> Path:
    """Copy the step log ``src`` to ``dst`` without the arrays ``drop`` (meta is updated with ``meta_update``)."""
    z = dict(np.load(src))
    meta = json.loads(str(z["meta"]))
    meta.update(meta_update or {})
    np.savez_compressed(dst, **{k: v for k, v in z.items() if k not in drop and k != "meta"}, meta=json.dumps(meta))
    return dst


BELIEF_KEYS = ("gmm_w", "gmm_mu", "gmm_cov", "gmm_mask", "map_xy", "top_sigma", "map_error")


# ------------------------------------------------------------------------------------------------ log loading, truth frames
def test_old_log_mode_from_episodes_csv(tmp_path):
    steps = tmp_path / "tag" / "steps"
    steps.mkdir(parents=True)
    (tmp_path / "tag" / "episodes.csv").write_text("episode_id,seed,source,frame,scale,mode,reflect\n2,1,101,410,1.0,F,False\n3,2,101,420,1.0,T2,False\n")
    log = r.load_episode_log(make_log(steps / "greedy_map_2drones_ep0003.npz", mode=None, frame0=420))
    assert (log.mode, log.mode_source) == ("T2", "episodes.csv")
    assert log.frame_derived and log.start_xy is None and log.reward is None
    assert np.array_equal(log.frame, 420 + np.arange(40))                       # frame0 + t, one frame per step
    log2 = r.load_episode_log(make_log(steps / "greedy_map_2drones_ep0002.npz", mode=None, frame0=410))
    assert (log2.mode, log2.mode_source) == ("F", "episodes.csv") and set(log2.frame.tolist()) == {410}


def test_old_log_without_any_mode_hint_defaults_to_f_and_override_wins(tmp_path):
    p = make_log(tmp_path / "x.npz", mode=None)
    log = r.load_episode_log(p)
    assert (log.mode, log.mode_source) == ("F", "default")
    t2 = r.load_episode_log(p, mode="T2")                                       # the --mode T2 override: the field advances with the step
    assert (t2.mode, t2.mode_source) == ("T2", "arg") and np.array_equal(t2.frame, 410 + np.arange(40))
    plan = r.plan_render(t2, stride=5)
    assert plan.frames.tolist() == (410 + plan.steps).tolist() and len(plan.field_frames) == plan.steps.size
    assert r.mode_caption(t2, 3)[1].startswith("truth field: time-varying")     # and the caption says so, whatever the log meta said
    with pytest.raises(ValueError):
        r.load_episode_log(p, mode="T9")


def test_new_log_keys_and_logged_frame_array(tmp_path):
    log = r.load_episode_log(make_log(tmp_path / "n.npz", mode="T2", new_keys=True, with_frame_array=True, frame0=440, T=30))
    assert log.mode_source == "meta" and not log.frame_derived
    assert log.start_xy.shape == (2, 2) and log.reward.shape == (30,) and log.calib_inside.dtype == bool
    assert log.T == 30 and log.D == 2 and log.source == SRC and log.method == "gmm_infotaxis"
    z = dict(np.load(make_log(tmp_path / "m0.npz", mode="T2", frame0=440, T=30)))             # a logged frame array wins over the derived one
    np.savez_compressed(tmp_path / "m.npz", **{k: v for k, v in z.items() if k != "meta"}, frame=np.full(30, 500), meta=str(z["meta"]))
    assert set(r.load_episode_log(tmp_path / "m.npz").frame.tolist()) == {500}


def test_one_drone_log_and_missing_required_key(tmp_path):
    log = r.load_episode_log(make_log(tmp_path / "one.npz", D=1))
    assert log.D == 1 and log.drone_xy.shape == (40, 1, 2) and log.y.shape == (40, 1)
    bad = drop_keys(tmp_path / "one.npz", tmp_path / "bad.npz", ("truth_xy",))
    with pytest.raises(KeyError, match="truth_xy"):
        r.load_episode_log(bad)


def test_log_without_belief_arrays_still_loads(tmp_path):
    p = drop_keys(make_log(tmp_path / "full.npz"), tmp_path / "nobelief.npz", BELIEF_KEYS)
    log = r.load_episode_log(p)
    assert log.gmm_w is None and log.map_xy is None and log.top_sigma is None
    assert r.localisation(log.top_sigma, log.map_error) == (None, None)
    assert "n/a" in " ".join(r.hud_lines(log, 3, None))


def test_frame_sequence_and_hold():
    f = r.frame_sequence("T2", 450, 200)
    assert f[0] == 450 and f[149] == 599 and f[150:].tolist() == [599] * 50          # zero-order hold after the last cached frame
    assert r.field_hold_step(f, "T2", 450) == 150
    assert r.field_hold_step(r.frame_sequence("T2", 410, 150), "T2", 410) is None   # 410 .. 559: no hold
    assert r.field_hold_step(r.frame_sequence("F", 410, 50), "F", 410) is None
    assert r.frame_sequence("F", 434, 5).tolist() == [434] * 5
    assert r.frame_sequence("T2", 400, 3, logged=np.array([7, 8, 9])).tolist() == [7, 8, 9]


def test_localisation_uses_primary_thresholds():
    sig = np.array([100.0, 40.0, 29.9, 10.0])
    err = np.array([90.0, 10.0, 49.0, 5.0])
    assert r.localisation(sig, err) == (2, 49.0)                                    # sigma < 30 and error < 50 at the same step
    assert r.localisation(sig, err, 15.0, 20.0) == (3, 5.0)                         # strict criterion
    assert r.localisation(np.full(4, 31.0), np.full(4, 1.0)) == (None, None)


def test_r3d_constants_are_read_through_config(monkeypatch):
    assert r._cfg("R3D_DOES_NOT_EXIST", 5) == 5
    monkeypatch.setattr(config, "R3D_TEST_ONLY", 7, raising=False)                    # a constant moved into config.py wins over the default
    assert r._cfg("R3D_TEST_ONLY", 5) == 7
    assert len([n for n in dir(r) if n.startswith("R3D_")]) >= 45                       # the full list is in the module docstring / report


# ------------------------------------------------------------------------------------------------ steps, colours, geometry
def test_select_steps():
    assert r.select_steps(10, 3).tolist() == [0, 3, 6, 9]
    assert r.select_steps(11, 3).tolist() == [0, 3, 6, 9, 10]                       # the last step is always shown
    assert r.select_steps(1, 2).tolist() == [0]
    s = r.select_steps(300, 2, max_frames=20)
    assert s.size <= 20 and s[0] == 0 and s[-1] == 299 and np.all(np.diff(s) > 0)
    for bad in ((0, 2, None), (5, 0, None), (5, 2, 0)):
        with pytest.raises(ValueError):
            r.select_steps(*bad)


def test_count_colour_grey_orange_red():
    thr = r.currie_count_threshold()
    assert thr == pytest.approx(33.4, abs=0.1)                                       # b = 20 cps, k = 3, T = 1 s
    base = (0.5, 0.6, 0.7)
    assert np.allclose(r.count_colour(thr - 1.0, base=base), base)
    assert np.allclose(r.count_colour(0, base=base), base)
    assert np.allclose(r.count_colour(thr + 1e-6, base=base), r.R3D_COLOUR_DETECT, atol=1e-3)      # orange at the detection threshold
    assert np.allclose(r.count_colour(thr * r.R3D_COUNT_HIGH_FACTOR * 5), r.R3D_COLOUR_HIGH)       # saturates to red
    g = r.count_colour(np.array([40.0, 80.0, 160.0, 320.0]))[:, 1]
    assert np.all(np.diff(g) < 0)                                                    # orange -> red: the green channel falls monotonically
    assert r.count_colour(np.zeros((4, 2))).shape == (4, 2, 3)
    with pytest.raises(ValueError):
        r.count_colour(1.0, threshold=10.0, high=5.0)


def test_trail_geometry():
    T = 60
    xy = np.column_stack([np.arange(T, dtype=float), np.zeros(T)])
    y = np.zeros(T)
    y[50:] = 400.0
    assert r.trail_geometry(xy, y, 0) is None
    tr = r.trail_geometry(xy, y, 55, n_trail=30)
    assert tr["points"].shape == (30, 3) and tr["rgb"].dtype == np.uint8 and tr["rgb"].shape == (30, 3)
    assert np.allclose(tr["points"][-1], [55.0, 0.0, config.DRONE_Z]) and np.allclose(tr["points"][0], [26.0, 0.0, config.DRONE_Z])
    assert tr["weight"][-1] == pytest.approx(1.0) and tr["weight"][0] == pytest.approx(0.25) and np.all(np.diff(tr["weight"]) > 0)
    assert tr["rgb"][-1, 0] > tr["rgb"][-1, 2] + 100                                  # newest sample is a red / orange detection
    short = r.trail_geometry(xy, y, 4, n_trail=30)
    assert short["points"].shape == (5, 3)


def test_ellipse_points_are_the_sigma_contour():
    cov = np.array([[400.0, 150.0], [150.0, 100.0]])
    mu = np.array([10.0, -20.0])
    pts = r.ellipse_points(mu, cov, n_sigma=2.0, n=90, z=15.4)
    d = pts[:, :2] - mu
    m2 = np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d)
    assert np.allclose(m2, 4.0) and np.allclose(pts[:, 2], 15.4) and pts.shape == (90, 3)
    flat = r.ellipse_points(mu, np.zeros((2, 2)), n=16)                              # degenerate covariance stays finite
    assert np.all(np.isfinite(flat))
    a, b = r.ellipse_style(0.0), r.ellipse_style(1.0)
    assert a["line_opacity"] < b["line_opacity"] and a["fill_opacity"] < b["fill_opacity"] and a["line_width"] < b["line_width"]
    assert r.ellipse_style(7.0) == r.ellipse_style(1.0)                              # weights are clipped


def test_star_faces_the_camera():
    for az in (-60.0, 0.0, 33.0, 120.0):
        sp = r.star_points((10.0, 20.0, 70.0), 5.0, az)
        view = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
        assert sp.shape == (11, 3) and np.allclose(sp[0], [10.0, 20.0, 70.0])
        assert np.allclose((sp - sp[0]) @ view, 0.0, atol=1e-9)                       # billboard: the star plane is perpendicular to the viewing azimuth
        assert sp[1, 2] == pytest.approx(75.0)                                         # first tip points up


def test_log_density_field_four_decades():
    d = np.array([[1.0, 0.5, 1e-3, 1e-4], [1e-5, 0.0, np.nan, 10.0]])
    ld, clim = r.log_density_field(d, vmax=10.0, decades=4.0)
    assert clim == (pytest.approx(-3.0), pytest.approx(1.0))
    assert np.isnan(ld[0, 3]) and np.isnan(ld[1, 0]) and np.isnan(ld[1, 1]) and np.isnan(ld[1, 2])    # below 4 decades, zero, nan
    assert ld[0, 2] == pytest.approx(-3.0, abs=1e-6) and ld[1, 3] == pytest.approx(1.0) and ld.dtype == np.float32


def test_subsample_points():
    pts = np.random.default_rng(0).normal(size=(5000, 3))
    a, b = r.subsample_points(pts, 100, seed=4), r.subsample_points(pts, 100, seed=4)
    assert a.shape == (100, 3) and np.array_equal(a, b) and not np.array_equal(a, r.subsample_points(pts, 100, seed=5))
    assert r.subsample_points(pts, 10_000) is pts and r.subsample_points(pts, 0).shape == (0, 3)


# ------------------------------------------------------------------------------------------------ camera
def _in_view(cam: r.CameraPath, k: int, pts_xy: np.ndarray, aspect: float, fov: float = r.R3D_FOV_DEG) -> np.ndarray:
    """Pinhole check: are the ground points (z = DRONE_Z) of video frame k inside the picture? (vertical / horizontal tangent of the view rays)"""
    f = cam.focal[k] - cam.position[k]
    f = f / np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    v = np.column_stack([pts_xy, np.full(len(pts_xy), config.DRONE_Z)]) - cam.position[k]
    depth = v @ f
    tv, th = math.tan(math.radians(fov) / 2.0), math.tan(math.radians(fov) / 2.0) * aspect
    return (depth > 0) & (np.abs(v @ up / depth) <= tv) & (np.abs(v @ right / depth) <= th)


def test_camera_distance_fits_near_and_far_ground_points():
    for radius in (60.0, 110.0, 400.0):
        d = float(r.camera_distance(radius, 30.0, 16 / 9, 35.0))
        el = math.radians(35.0)
        near = math.degrees(math.atan2(d * math.sin(el), d * math.cos(el) - radius))
        far = math.degrees(math.atan2(d * math.sin(el), d * math.cos(el) + radius))
        assert 35.0 - 15.0 <= far < near <= 35.0 + 15.0                              # both ends inside the +-15 deg vertical view
    d1, d2 = r.camera_distance(100.0, 30.0, 16 / 9, 35.0), r.camera_distance(200.0, 30.0, 16 / 9, 35.0)
    assert d2 == pytest.approx(2.0 * d1, rel=1e-6)                                   # the fit scales linearly with the half-extent
    assert float(r.marker_scale(500.0)) == pytest.approx(500.0 * r.R3D_UNIT_FRAC)


def test_camera_path_follows_then_zooms_to_the_source(tmp_path):
    log = r.load_episode_log(make_log(tmp_path / "c.npz", T=100, start_far=True))
    steps = r.select_steps(log.T, 2)
    cam = r.camera_path(log.drone_xy[steps], log.truth_xy)
    F = steps.size
    assert cam.n_frames == F and cam.position.shape == (F, 3) and np.all(np.isfinite(cam.position))
    assert np.all(cam.zoom[: int(0.8 * F)] == 0.0) and cam.zoom[-1] == pytest.approx(1.0)
    assert np.allclose(cam.focal[-1, :2], log.truth_xy) and np.all(cam.focal[:, 2] == r.R3D_FOCAL_Z_M)
    assert cam.radius[-1] == pytest.approx(r.R3D_RADIUS_ZOOM_M, rel=0.2) and cam.radius[0] > cam.radius[-1]
    assert cam.distance[-1] < cam.distance[0]                                         # zoomed in at the end
    rel = cam.position - cam.focal
    elev = np.degrees(np.arctan2(rel[:, 2], np.hypot(rel[:, 0], rel[:, 1])))
    assert np.allclose(elev, r.R3D_ELEVATION_DEG)                                     # oblique elevation 35 deg throughout
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    assert az[0] == pytest.approx(r.R3D_AZIMUTH_DEG) and az[-1] == pytest.approx(r.R3D_AZIMUTH_DEG + r.R3D_ZOOM_ORBIT_DEG)
    assert np.allclose(cam.azimuth_deg, az)
    # the camera follows: before the zoom the focal point moves with the drones (they fly from x = 450 to x = 65)
    assert cam.focal[0, 0] > cam.focal[int(0.7 * F), 0]


def test_camera_keeps_drones_and_source_in_view(tmp_path):
    aspect = 16 / 9
    for kw in ({"start_far": True}, {"start_far": False, "success_at": None}):
        log = r.load_episode_log(make_log(tmp_path / "v.npz", T=80, **kw))
        steps = r.select_steps(log.T, 2)
        cam = r.camera_path(log.drone_xy[steps], log.truth_xy, aspect=aspect)
        for k, i in enumerate(steps):
            pts = np.vstack([log.drone_xy[i], log.truth_xy[None]])
            assert _in_view(cam, k, pts, aspect).all(), (kw, k)


def test_camera_zoom_stops_short_of_the_source_when_a_drone_is_still_far():
    F = 30
    drones = np.zeros((F, 1, 2))
    drones[:, 0] = [600.0, 0.0]                                                      # a failed episode: the drone stays 540 m from the source
    cam = r.camera_path(drones, (60.0, 0.0))
    assert cam.zoom[-1] == pytest.approx(1.0) and np.allclose(cam.focal[-1, :2], (60.0, 0.0))
    assert cam.radius[-1] >= 540.0 and _in_view(cam, F - 1, drones[-1], 16 / 9).all()
    with pytest.raises(ValueError):
        r.camera_path(np.zeros((5, 2)), (0.0, 0.0))
    one = r.camera_path(np.zeros((1, 2, 2)), (10.0, 10.0))                           # a single frame is allowed
    assert one.n_frames == 1 and np.all(np.isfinite(one.position))


# ------------------------------------------------------------------------------------------------ text, plan
def test_caption_hud_and_banner_text(tmp_path):
    f = r.load_episode_log(make_log(tmp_path / "f.npz", mode="F", frame0=434))
    cap = r.mode_caption(f, 5)
    assert cap[0] == "gmm_infotaxis, 2 drones, source 101, Mode F" and cap[1] == "truth field: frozen snapshot, frame 434"
    assert f"LDM step {config.index_to_step(434)}" in cap[2] and "interpretation A" in cap[2]
    loc, err = r.localisation(f.top_sigma, f.map_error)
    hud = r.hud_lines(f, 10, loc)
    assert hud[0].startswith("step  11/40") and "D1" in hud[1] and "D2" in hud[1] and "MAP error" in hud[3] and hud[4] == "status: searching"
    assert "LOCALISED (step 26)" in r.hud_lines(f, 30, loc)[4]                      # the localisation step stays in the HUD afterwards
    assert r.banner_text(f, loc, err) == ("LOCALISED in 26 steps (error 12.5 m)", "ok")
    nf = r.load_episode_log(make_log(tmp_path / "nf.npz", success_at=None))
    text, kind = r.banner_text(nf, *r.localisation(nf.top_sigma, nf.map_error))
    assert kind == "fail" and text.startswith("NOT LOCALISED after 40 steps") and "150 m" in text
    t2 = r.load_episode_log(make_log(tmp_path / "t2.npz", mode="T2", frame0=590, T=40))
    assert r.mode_caption(t2, 3)[1] == "truth field: time-varying, frame 593 (start 590 + t)"
    assert r.mode_caption(t2, 20)[1] == "truth field: frozen at frame 599 from step 11"      # zero-order hold after the last cached frame


def test_hud_labels_the_two_clocks(tmp_path):
    """Episode time (RL steps x 1 s) and LDM time (interpretation A) are two lines with their own labels, never one ambiguous 't ='."""
    t2 = r.load_episode_log(make_log(tmp_path / "t2.npz", mode="T2", frame0=410, T=300))
    hud = r.hud_lines(t2, 200, None)
    assert hud[0].startswith("step 201/300") and hud[0].endswith("episode time 201 s")
    expect = config.index_to_step(int(t2.frame[200])) * config.SEC_PER_INDEX_STEP
    assert hud[5] == f"LDM t {expect:.0f} s (interp. A)" and "episode" not in hud[5]
    assert " t = " not in " ".join(hud)
    assert r.hud_lines(t2, 200, None, frame=450)[5] != hud[5]                          # an explicit frame (illustrative clip) changes the LDM time only
    assert r.hud_lines(t2, 200, None, frame=450)[0] == hud[0]


def test_banner_for_a_log_without_belief_arrays_claims_no_step(tmp_path):
    full = make_log(tmp_path / "full.npz", T=40, success_at=25)
    ok = r.load_episode_log(drop_keys(full, tmp_path / "ok.npz", BELIEF_KEYS))                  # meta success = True, no arrays to say when
    assert r.banner_text(ok, *r.localisation(ok.top_sigma, ok.map_error)) == ("LOCALISED (step unknown)", "ok")
    bad = r.load_episode_log(drop_keys(full, tmp_path / "bad.npz", BELIEF_KEYS, {"success": False}))
    text, kind = r.banner_text(bad, None, None)
    assert kind == "fail" and text == "NOT LOCALISED after 40 steps" and "nan" not in text
    plan = r.plan_render(ok, stride=4)                                                         # and the plan survives it (banner at the end)
    assert plan.loc_step is None and plan.banner[0] == "LOCALISED (step unknown)" and plan.banner_from == plan.steps.size - math.ceil(r.R3D_BANNER_FRAC * plan.steps.size)


def test_plan_render_frames_banner_and_hold(tmp_path):
    f = r.load_episode_log(make_log(tmp_path / "f.npz", mode="F", frame0=434, T=100))
    plan = r.plan_render(f, stride=2, fps=10, hold_seconds=1.5)
    assert plan.steps[0] == 0 and plan.steps[-1] == 99 and plan.hold_frames == 15 and plan.repeat == 1 and plan.n_steps_used == 100
    assert plan.field_frames == [434] and set(plan.frames.tolist()) == {434}
    assert plan.loc_step == 25 and plan.banner[1] == "ok"
    assert plan.steps[plan.banner_from] >= plan.loc_step > plan.steps[plan.banner_from - 1]      # the banner shows from the localisation picture on
    nf = r.load_episode_log(make_log(tmp_path / "nf.npz", mode="F", frame0=434, T=100, success_at=None))
    pn = r.plan_render(nf, stride=2)
    assert pn.loc_step is None and pn.banner_from == pn.steps.size - math.ceil(r.R3D_BANNER_FRAC * pn.steps.size)       # a failure banner only at the end
    t2 = r.load_episode_log(make_log(tmp_path / "t.npz", mode="T2", frame0=410, T=100))
    p2 = r.plan_render(t2, stride=5, max_frames=10)
    assert p2.steps.size <= 10 and p2.frames[0] == 410 and p2.frames[-1] == 410 + 99 and len(p2.field_frames) == p2.steps.size    # the field advances
    assert p2.camera.n_frames == p2.steps.size


def test_stop_at_localisation_ends_the_video_a_few_seconds_after_it(tmp_path):
    log = r.load_episode_log(make_log(tmp_path / "s.npz", mode="F", frame0=434, T=300, success_at=40))
    full = r.plan_render(log, stride=2, fps=10)
    stop = r.plan_render(log, stride=2, fps=10, stop_at_localisation=True, stop_after_seconds=3.0)
    assert full.steps[-1] == 299 and full.n_steps_used == 300 and stop.loc_step == full.loc_step == 40
    assert stop.n_steps_used == 40 + 1 + 3 * 10 * 2 and stop.steps[-1] == stop.n_steps_used - 1          # 3 s x 10 pictures/s x stride 2 steps after the localisation
    assert stop.steps.size < full.steps.size and np.array_equal(stop.steps, full.steps[: stop.steps.size])      # the same pictures, just fewer of them
    assert stop.steps[stop.banner_from] >= 40 > stop.steps[stop.banner_from - 1]                          # banner from the localisation picture, hold afterwards
    assert stop.camera.n_frames == stop.steps.size and stop.camera.zoom[-1] == pytest.approx(1.0)        # the final zoom happens at the end of the SHORT video
    assert "LOCALISED (step 41)" in r.hud_lines(log, int(stop.steps[-1]), stop.loc_step)[4]               # the HUD keeps the localisation step
    late = r.load_episode_log(make_log(tmp_path / "l.npz", mode="F", frame0=434, T=60, success_at=55))
    assert r.plan_render(late, stride=2, stop_at_localisation=True).n_steps_used == 60                  # never longer than the episode
    nf = r.load_episode_log(make_log(tmp_path / "n.npz", mode="F", frame0=434, T=100, success_at=None))
    assert r.plan_render(nf, stride=2, stop_at_localisation=True).n_steps_used == 100                   # nothing to stop at: the whole episode


def test_frame_repeat_and_presets():
    assert r.frame_repeat(10, 2, None) == 1
    assert r.frame_repeat(30, 1, 5.0) == 6                                           # spec 11.5: 5 steps per second encoded at 30 fps = 6 copies per step
    assert r.frame_repeat(10, 1, 20.0) == 1 and r.frame_repeat(8, 3, 100.0) == 1     # never fewer than one
    with pytest.raises(ValueError):
        r.frame_repeat(30, 1, 0.0)
    d = r.resolve_options()                                                          # the defaults are the quick preview
    assert (d["width"], d["height"], d["fps"], d["stride"], d["steps_per_second"]) == (r.R3D_WINDOW[0], r.R3D_WINDOW[1], r.R3D_FPS, r.R3D_STRIDE, None)
    f = r.resolve_options("final")                                                   # spec 11.5
    assert (f["width"], f["height"], f["fps"], f["stride"], f["steps_per_second"]) == (1920, 1080, 30, 1, 5.0)
    o = r.resolve_options("final", width=640, height=None, stride=2)                 # explicit options override the preset, None does not
    assert (o["width"], o["height"], o["stride"], o["fps"]) == (640, 1080, 2, 30)
    with pytest.raises(ValueError):
        r.resolve_options("nope")


def test_illustrative_clip_plan(tmp_path):
    f = r.illustrative_frames(100)
    assert f[0] == 400 and f[-1] <= 599 and np.all(np.diff(f) >= 0) and f[-1] >= 597 and set(f.tolist()) <= set(range(400, 600))   # one pass over 400..599
    assert r.illustrative_frames(400)[-1] <= 599 and r.illustrative_frames(1).tolist() == [400]
    with pytest.raises(ValueError):
        r.illustrative_frames(0)
    log = r.load_episode_log(make_log(tmp_path / "i.npz", mode="F", frame0=434, T=300))
    plan = r.plan_render(log, fps=10, hold_seconds=2.0, illustrative=True)
    assert plan.illustrative and plan.steps.size == 80 and plan.steps[0] == 0 and plan.steps[-1] == 299                # 8 s of pictures + 2 s hold = 10 s
    assert plan.steps.size / 10 + plan.hold_frames / 10 <= r.R3D_ILLUSTRATIVE_SECONDS + 1e-9
    assert plan.frames[0] == 400 and plan.frames[-1] >= 595 and len(plan.field_frames) > 70                          # the field cycles although the log is Mode F
    assert r.plan_render(log, illustrative=True, max_frames=12).steps.size == 12                                      # an explicit smaller cap is respected
    assert r.plan_render(log, illustrative=True, stop_at_localisation=True).n_steps_used == 300                      # stop-at-localisation is ignored
    cap = r.mode_caption(log, 5, frame=480, illustrative=True)
    assert "ILLUSTRATIVE" in cap[1] and "frame 480" in cap[1] and f"LDM step {config.index_to_step(480)}" in cap[2]
    assert r.R3D_ILLUSTRATIVE_WATERMARK == "illustrative - not the truth seen by the policy" and r.R3D_ILLUSTRATIVE_FRAMES == (400, 599)


# ------------------------------------------------------------------------------------------------ command line
def test_cli_parser_aliases_and_defaults(tmp_path):
    ap = r.build_parser()
    base = ["--episode", str(tmp_path / "a.npz"), "--out", str(tmp_path / "o.mp4")]
    a = ap.parse_args(base)
    assert a.mode is None and a.preset is None and a.width is None and not a.stop_at_localisation and not a.illustrative
    assert ap.parse_args(base + ["--mode", "T2"]).mode == "T2"
    assert ap.parse_args(base + ["--frame-mode", "F"]).mode == "F"                   # --frame-mode is the spec 11.6 alias of --mode
    assert ap.parse_args(base + ["--preset", "final", "--stop-at-localisation", "--illustrative"]).preset == "final"
    with pytest.raises(SystemExit) as e:
        ap.parse_args(base + ["--frame-mode", "T9"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        ap.parse_args(base + ["--preset", "nope"])


def test_main_passes_the_mode_override_and_the_preset(tmp_path, monkeypatch):
    seen = {}

    def fake(npz, out, **kw):
        seen.update(kw, npz=npz, out=out)
        return {"status": "ok", "frames": 1}

    monkeypatch.setattr(r, "render_episode", fake)
    base = ["--episode", str(tmp_path / "a.npz"), "--out", str(tmp_path / "o.mp4")]
    assert r.main(base) == 0
    assert seen["mode"] is None and (seen["width"], seen["height"], seen["fps"], seen["stride"]) == (r.R3D_WINDOW[0], r.R3D_WINDOW[1], r.R3D_FPS, r.R3D_STRIDE)
    assert seen["stop_at_localisation"] is False and seen["illustrative"] is False and seen["steps_per_second"] is None
    assert r.main(base + ["--mode", "T2"]) == 0 and seen["mode"] == "T2"             # --mode T2 reaches the renderer
    assert r.main(base + ["--frame-mode", "T2"]) == 0 and seen["mode"] == "T2"       # and so does the alias
    assert r.main(base + ["--preset", "final", "--stop-at-localisation"]) == 0
    assert (seen["width"], seen["height"], seen["fps"], seen["stride"], seen["steps_per_second"]) == (1920, 1080, 30, 1, 5.0) and seen["stop_at_localisation"]
    assert r.main(base + ["--preset", "final", "--width", "1280", "--height", "720", "--illustrative"]) == 0
    assert (seen["width"], seen["height"], seen["fps"]) == (1280, 720, 30) and seen["illustrative"]


def test_main_with_a_missing_episode_exits_2_without_a_traceback(tmp_path, capsys):
    rc = r.main(["--episode", str(tmp_path / "nope.npz"), "--out", str(tmp_path / "o.mp4")])
    cap = capsys.readouterr()
    assert rc == 2 and "Traceback" not in cap.err and "not found" in cap.err and not (tmp_path / "o.mp4").exists()
    assert json.loads(cap.out)["status"] == "skipped"
    bad = drop_keys(make_log(tmp_path / "ok.npz"), tmp_path / "bad.npz", ("truth_xy",))                   # an unusable log: the same, naming the missing key
    rc = r.main(["--episode", str(bad), "--out", str(tmp_path / "o.mp4")])
    cap = capsys.readouterr()
    assert rc == 2 and "truth_xy" in cap.err and "Traceback" not in cap.err
    ok = make_log(tmp_path / "ok2.npz")
    rc = r.main(["--episode", str(ok), "--out", str(tmp_path / "o.mp4"), "--width", "639"])                          # an odd H.264 width: refused, no traceback
    cap = capsys.readouterr()
    assert rc == 2 and "even" in cap.err and "Traceback" not in cap.err and not (tmp_path / "o.mp4").exists()


def test_command_line_exit_code_and_stderr_of_a_missing_episode(tmp_path):
    """The real process: exit code 2, the reason on stderr, no Python traceback."""
    cmd = [sys.executable, "-m", "srcloc_env.scripts.render_episode_3d", "--episode", str(tmp_path / "nope.npz"), "--out", str(tmp_path / "o.mp4")]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(Path(__file__).resolve().parents[1]))
    assert p.returncode == 2 and "Traceback" not in p.stderr and "step log not found" in p.stderr
    assert json.loads(p.stdout)["status"] == "skipped"


def test_render_episode_skips_a_missing_log(tmp_path):
    m = r.render_episode(tmp_path / "nope.npz", tmp_path / "out.mp4")
    assert m["status"] == "skipped" and "not found" in m["reason"] and m["frames"] == 0 and not (tmp_path / "out.mp4").exists()


def test_load_buildings_shifts_and_decimates(tmp_path):
    pv = pytest.importorskip("pyvista")
    stl = tmp_path / "b.stl"
    pv.Sphere(radius=10.0, theta_resolution=40, phi_resolution=40).save(str(stl))
    full = r.load_buildings(stl, shift=(100.0, 20.0, 0.0), max_triangles=10_000)
    assert np.allclose(np.array(full.center), [100.0, 20.0, 0.0], atol=1e-3)
    small = r.load_buildings(stl, shift=(100.0, 20.0, 0.0), max_triangles=500)
    assert 0 < small.n_cells < full.n_cells and r.load_buildings(stl, shift=(100.0, 20.0, 0.0), max_triangles=500) is small     # cached in memory
    mirrored = r.load_buildings(stl, shift=(100.0, 20.0, 0.0), max_triangles=10_000, mirror_y=True)
    assert np.allclose(np.array(mirrored.center), [100.0, -20.0, 0.0], atol=1e-3)


# ------------------------------------------------------------------------------------------------ off-screen renders
class FakeBackend:
    """Slab backend look-alike: a Gaussian plume on a 40 x 40 grid that drifts with the frame, plus a particle cloud per frame."""

    def __init__(self) -> None:
        self.grid = SlabGrid(x0=-200.0, y0=-200.0, nx=40, ny=40, res=10.0)

    def slab(self, frame: int) -> SlabFrame:
        yy, xx = np.mgrid[0:40, 0:40]
        shift = (frame - 410) * 0.5
        blob = np.exp(-(((xx - 22 - shift) ** 2 + (yy - 20) ** 2) / 30.0)).astype(np.float32)
        return SlabFrame(index=frame, density=np.stack([blob, 0.1 * blob])[:, None], z_levels=(config.DRONE_Z,), sources=(SRC, 102), grid=self.grid)

    def airborne_particles(self, frame: int, src_id: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(frame)
        return np.column_stack([rng.normal(60.0 + (frame - 410) * 5.0, 30.0, 400), rng.normal(0.0, 15.0, 400), rng.uniform(0.0, 40.0, 400)]).astype(np.float32)


@pytest.fixture(scope="module")
def gl():
    """pyvista + imageio, or a skip when no off-screen OpenGL context exists (the pure tests above still ran)."""
    pv = pytest.importorskip("pyvista")
    pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    try:
        probe = pv.Plotter(off_screen=True, window_size=(64, 64))
        probe.render()
        probe.close()
    except Exception as e:                                                           # noqa: BLE001 - no usable OpenGL context
        pytest.skip(f"off-screen rendering unavailable: {e}")
    return pv


def town(pv):
    return pv.Box(bounds=(100.0, 140.0, -60.0, -20.0, 0.0, 30.0)).triangulate()


def mp4_info(path) -> tuple[int, float]:
    """(frames, duration [s]) of an MP4 read from its boxes (moov/mvhd and moov/trak/mdia/minf/stbl/stsz): no ffmpeg process, so it cannot time out."""
    buf = Path(path).read_bytes()

    def boxes(start: int, end: int):
        i = start
        while i + 8 <= end:
            size, typ, head = int.from_bytes(buf[i:i + 4], "big"), buf[i + 4:i + 8].decode("latin1"), 8
            if size == 1:
                size, head = int.from_bytes(buf[i + 8:i + 16], "big"), 16
            elif size == 0:
                size = end - i
            yield typ, i + head, i + size
            i += size

    def find(chain: list[str], start: int = 0, end: int | None = None):
        for typ, a, b in boxes(start, len(buf) if end is None else end):
            if typ == chain[0]:
                return (a, b) if len(chain) == 1 else find(chain[1:], a, b)
        raise AssertionError(f"box {chain} not found in {path}")

    a, _ = find(["moov", "mvhd"])                                                  # version / flags, then creation, modification (4 or 8 bytes each), timescale, duration
    if buf[a] == 0:
        timescale, duration = int.from_bytes(buf[a + 12:a + 16], "big"), int.from_bytes(buf[a + 16:a + 20], "big")
    else:
        timescale, duration = int.from_bytes(buf[a + 20:a + 24], "big"), int.from_bytes(buf[a + 24:a + 32], "big")
    a, _ = find(["moov", "trak", "mdia", "minf", "stbl", "stsz"])
    return int.from_bytes(buf[a + 8:a + 12], "big"), duration / timescale


def decode(path) -> list[np.ndarray]:
    """All frames of the MP4 ``path`` (imageio / ffmpeg; retried because the header reader gives up after 10 s when the machine is busy)."""
    import imageio.v2 as iio
    for attempt in range(3):
        try:
            rd = iio.get_reader(str(path), format="FFMPEG")
            try:
                return [np.asarray(f) for f in rd]
            finally:
                rd.close()
        except OSError:
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def test_render_small_video_and_poster(tmp_path, gl):
    import imageio.v2 as iio
    npz = make_log(tmp_path / "ep.npz", T=24, D=2, mode="T2", frame0=410, new_keys=True, with_frame_array=True, success_at=14)
    out = tmp_path / "v.mp4"
    m = r.render_episode(npz, out, width=192, height=108, fps=5, stride=8, hold_seconds=0.4, backend=FakeBackend(), buildings=town(gl))
    assert m["status"] == "ok" and m["step_frames"] == 4 and m["hold_frames"] == 2 and m["frames"] == 6 and m["size"] == [192, 108] and m["repeat"] == 1
    assert m["mode"] == "T2" and m["n_drones"] == 2 and m["localised_step"] == 15 and m["skipped"] == {} and m["particles_shown_max"] == 400
    assert m["field_frames"] == {"first": 410, "last": 433, "distinct": 4} and m["banner"].startswith("LOCALISED in 15 steps")
    assert m["steps_covered"] == 24 and m["illustrative"] is False and m["stop_at_localisation"] is False
    n, secs = mp4_info(out)
    assert n == 6 and secs == pytest.approx(1.2, abs=0.1)
    frames = decode(out)
    assert len(frames) == 6 and frames[0].shape == (108, 192, 3)
    assert np.abs(frames[0].astype(int) - frames[3].astype(int)).mean() > 1.0       # the drones, the field and the camera move
    assert np.array_equal(frames[-1], frames[-2]) or np.abs(frames[-1].astype(int) - frames[-2].astype(int)).mean() < 1.0    # the held last frame
    poster = iio.imread(m["poster"])
    assert poster.shape == (108, 192, 3) and poster.std() > 5.0                       # not an empty picture


def test_render_mode_override_on_an_old_log_and_the_five_steps_per_second_repeat(tmp_path, gl):
    npz = make_log(tmp_path / "old.npz", T=24, D=1, mode=None, frame0=410, success_at=None)                # no mode in the meta, no frame array
    m = r.render_episode(npz, tmp_path / "o.mp4", width=160, height=90, fps=10, stride=1, max_frames=3, hold_seconds=0.5, steps_per_second=2.5, mode="T2",
                         backend=FakeBackend(), buildings=town(gl))
    assert m["mode"] == "T2" and m["mode_source"] == "arg" and m["field_frames"]["distinct"] == 3 and m["field_frames"]["first"] == 410
    assert m["repeat"] == 4 and m["step_frames"] == 3 and m["frames"] == 3 * 4 + 5                          # every picture is written 4 times (10 fps / 2.5 steps per s)
    n, secs = mp4_info(tmp_path / "o.mp4")
    assert n == 17 and secs == pytest.approx(1.7, abs=0.1)
    assert m["banner"].startswith("NOT LOCALISED after 24 steps") and m["localised_step"] is None


def test_render_stop_at_localisation_and_illustrative_clip(tmp_path, gl):
    import imageio.v2 as iio
    npz = make_log(tmp_path / "ep.npz", T=60, D=2, mode="F", frame0=434, success_at=10)
    m = r.render_episode(npz, tmp_path / "s.mp4", width=160, height=90, fps=5, stride=2, hold_seconds=0.4, stop_at_localisation=True, stop_after_seconds=1.0,
                         backend=FakeBackend(), buildings=town(gl), particles=False)
    assert m["stop_at_localisation"] and m["localised_step"] == 11 and m["steps_covered"] == 10 + 1 + 5 * 2        # 1 s = 5 pictures x 2 steps after the localisation
    assert m["steps_covered"] < m["n_steps"] and m["banner_from_step"] >= 11 and any("stopped" in n for n in m["notes"])
    assert m["frames"] == m["step_frames"] + m["hold_frames"] and m["frames"] < 60 // 2
    ill = r.render_episode(npz, tmp_path / "i.mp4", width=160, height=90, fps=5, hold_seconds=0.4, illustrative=True, max_frames=6, backend=FakeBackend(),
                           buildings=town(gl))
    assert ill["illustrative"] and ill["step_frames"] == 6 and ill["field_frames"]["first"] == 400 and ill["field_frames"]["distinct"] == 6
    assert any("ILLUSTRATIVE" in n for n in ill["notes"]) and iio.imread(ill["poster"]).std() > 5.0


def _boxes_overlap(a, b, tol: float = 0.0) -> bool:
    return a[0] < b[2] - tol and b[0] < a[2] - tol and a[1] < b[3] - tol and b[1] < a[3] - tol


def test_overlap_helper_detects_collisions():
    """Negative control of the layout test: overlapping boxes are reported, touching or disjoint ones are not."""
    assert _boxes_overlap((0, 0, 10, 10), (5, 5, 15, 15)) and _boxes_overlap((0, 0, 10, 10), (2, 2, 3, 3))
    assert not _boxes_overlap((0, 0, 10, 10), (10, 0, 20, 10)) and not _boxes_overlap((0, 0, 10, 10), (0, 11, 10, 20))


def _check_panels(scene: r._Scene, plan: r.RenderPlan, ks: list[int]) -> None:
    """Draw the video frames ``ks`` and check that no two text panels / the legend / the colour bars overlap and that all lie inside the window."""
    W, H = scene.W, scene.H
    for k in ks:
        scene.draw(k, banner=k >= plan.banner_from)
        boxes = scene.text_boxes()
        assert {"hud", "caption", "legend"} <= set(boxes)
        names = sorted(boxes)
        for i, a in enumerate(names):
            x0, y0, x1, y1 = boxes[a]
            assert -1.0 <= x0 < x1 <= W + 1.0 and -1.0 <= y0 < y1 <= H + 1.0, (a, boxes[a], (W, H))
            for b in names[i + 1:]:
                assert not _boxes_overlap(boxes[a], boxes[b]), (k, a, boxes[a], b, boxes[b])
    bx = scene.layout["banner"]                                                       # planned banner box (text of this episode): sized by its length, centred
    assert bx[2] - bx[0] <= r.R3D_BANNER_MAX_WIDTH_FRAC * W + 1.0 and abs(0.5 * (bx[0] + bx[2]) - 0.5 * W) < 2.0 and bx[1] >= 0.0
    assert not _boxes_overlap(bx, scene.layout["legend"]) and not _boxes_overlap(bx, scene.layout["hud"]) and not _boxes_overlap(bx, scene.layout["caption"])


@pytest.mark.parametrize("variant,size", [("localised_2", (320, 180)), ("failed_3_long_banner", (320, 180)), ("failed_3_long_banner", (640, 360)),
                                          ("illustrative", (320, 180))])
def test_text_panels_do_not_overlap_at_small_sizes(tmp_path, gl, variant, size):
    """Finding MEDIUM: at small video sizes the HUD, caption, banner, legend (and watermark, colour bars) used to collide because the font had an 8 pt floor.
    Text now scales linearly with the height and every panel is placed from the measured size of its worst-case text."""
    W, H = size
    if variant == "localised_2":
        log = r.load_episode_log(make_log(tmp_path / "a.npz", T=40, D=2, mode="T2", frame0=410, success_at=14))
        plan = r.plan_render(log, stride=4, aspect=W / H)
    elif variant == "failed_3_long_banner":                                           # the longest banner ('NOT LOCALISED after 300 steps (final error 297 m)') and 3 drones
        log = r.load_episode_log(make_log(tmp_path / "b.npz", T=300, D=3, mode="T2", frame0=410, success_at=None))
        log.map_error[-1] = 297.0
        plan = r.plan_render(log, stride=60, aspect=W / H)
        assert plan.banner[0] == "NOT LOCALISED after 300 steps (final error 297 m)"
    else:
        log = r.load_episode_log(make_log(tmp_path / "c.npz", T=40, D=2, mode="F", frame0=434, success_at=None))
        plan = r.plan_render(log, stride=4, aspect=W / H, illustrative=True)
    scene = r._Scene(log, plan, W, H, FakeBackend(), town(gl), True, 400, {})
    try:
        assert scene.fs == pytest.approx(r.R3D_FONT_UNITS_720 * H / 720.0)             # linear in the height: no floor (the old max(8, ...) made H = 360 text 1.45x too large)
        assert scene.u_hud <= scene.fs + 1e-9 and scene.u_cap <= 0.9 * scene.fs + 1e-9
        _check_panels(scene, plan, [0, plan.steps.size // 2, plan.steps.size - 1])
        if variant == "illustrative":
            assert "watermark" in scene.text_boxes() and "banner" in scene.text_boxes()
        legend = [e[0] for e in scene._legend_entries()]
        assert any("ring" in e and f"{config.SOURCE_Z:g} m" in e for e in legend) and any("star" in e and "true source" in e for e in legend)
        assert len([e for e in legend if e.startswith("drone")]) == log.D
    finally:
        scene.close()


def test_scene_marks_the_release_height_and_colours_particles_by_altitude(tmp_path, gl):
    log = r.load_episode_log(make_log(tmp_path / "m.npz", T=24, D=2, mode="T2", frame0=410, success_at=14))
    plan = r.plan_render(log, stride=8, aspect=16 / 9)
    scene = r._Scene(log, plan, 320, 180, FakeBackend(), town(gl), True, 400, {})
    try:
        scene.draw(1, banner=False)
        z = np.array(scene.ring_prim.mesh.points)[:, 2]                               # a ring at the release height config.SOURCE_Z (5.5 m), not at the column top
        assert z.mean() == pytest.approx(config.SOURCE_Z, abs=1e-3) and z.max() - z.min() < 20.0
        assert np.allclose(np.array(scene.ring_prim.mesh.points)[:, :2].mean(axis=0), log.truth_xy, atol=1e-3)
        star_z = np.array(scene.star.points)[:, 2]
        assert star_z.min() > config.SOURCE_Z + 40.0                                  # the star sits on the 60 m column, far above the ring
        cloud = scene.part_mesh
        assert "height" in cloud.point_data
        assert np.allclose(np.asarray(cloud.point_data["height"]), np.asarray(cloud.points)[:, 2])      # colour = altitude, not one flat colour
        assert float(np.ptp(np.asarray(cloud.point_data["height"]))) > 10.0 and set(scene.bars) == {"density", "particles"}
        assert tuple(scene.part_actor.mapper.scalar_range) == tuple(float(v) for v in r.R3D_PARTICLE_Z_RANGE)         # one fixed height scale for every frame
    finally:
        scene.close()
