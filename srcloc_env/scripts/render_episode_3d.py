"""3-D animation of one evaluation / quick-evaluation episode (docs/training_evaluation_spec.md section 11; final video, slide 12).

Usage:
    python -m srcloc_env.scripts.render_episode_3d --episode <steps/..._epNNNN.npz> --out <video.mp4>
        [--width 1280 --height 720 --fps 10 --stride 2 --max-frames N --no-particles --poster <png> --mode F|T2 (alias --frame-mode)]
        [--preset final] [--stop-at-localisation] [--illustrative]

Input: ONE step log written by eval/run_eval.py --log-steps (keys of spec 11.7; newer logs also carry frame, reward, calib_inside,
start_xy, meta mode / max_steps, older logs do not: the truth frame of every step and the truth mode are then derived, nothing ever
crashes on a missing newer key).  Output: an H.264 MP4 and a poster PNG (the last frame, with the closing banner).  Exit code 0 = video written,
2 = skipped / unusable input (the reason is printed, no traceback).

Presets (documented, the defaults stay the quick preview 1280 x 720, 10 fps, stride 2 = 20 log steps per second):
  --preset final   spec 11.5: 1920 x 1080, ONE log step per picture shown for 1/5 s (5 steps per second) and encoded at 30 fps (each picture is
                   written 6 times); any explicit --width / --height / --fps / --stride / --steps-per-second overrides the preset value.

Scene (PyVista, off-screen; ONE Plotter whose actors are updated per frame):
  * the buildings of config.STL_FILE shifted by config.STL_SHIFT (decimated above R3D_BUILDING_MAX_TRIANGLES, cached in memory) on a ground plane;
  * the 15 m (config.DRONE_Z) truth density of the episode's source as a semi-transparent plane, log10 colour map over R3D_DENSITY_DECADES
    decades below the episode maximum; the slab of the correct truth frame is shown at every video frame.  Mode T2 logs: frame = start frame + t
    (log key frame or derived), so the field visibly evolves; Mode F logs: ONE frozen frame and the caption says 'frozen snapshot, frame k';
  * optionally a subsample (<= R3D_MAX_PARTICLES) of the AIRBORNE particles of that source at that frame, coloured by altitude (colour bar;
    deposited particles are not in the cache and therefore absent: the caption says so);
  * the true source: a billboard star on a vertical column of R3D_COLUMN_HEIGHT_M (column highlighted once the episode is localised) and, at the real
    release height config.SOURCE_Z (5.5 m), a ring around a small sphere; the legend says what each marks;
  * the drones as spheres at 15 m with a fading trail of the last R3D_TRAIL_STEPS steps coloured by the measured count (grey -> orange at the Currie
    detection threshold -> red), drones in different hues, the top three GMM components as 2-sigma ellipses on the 15 m plane (weight = opacity) and
    the MAP marker;
  * an oblique camera (elevation 35 deg, azimuth -60 deg) that frames the drones and the source, follows them and zooms towards the source
    in the last 20 % of the frames, a HUD (step and 'episode time', 'LDM t (interp. A)', counts, top sigma, MAP error, status), a caption (method,
    drones, source, mode, truth-field statement) and a banner ('LOCALISED in N steps (error e m)' shown from the localisation frame on, or
    'NOT LOCALISED ...' at the end).  All text sizes scale linearly with the window height; the banner is sized by its text length, anchored at the
    bottom centre, and the legend sits above it, so that HUD, caption, banner, legend and colour bars never overlap (checked by the tests at
    320 x 180 and by looking at 640 x 360, 960 x 540 and 1280 x 720 renders).
  * --stop-at-localisation ends the video R3D_STOP_AFTER_SECONDS after the first localisation (the HUD keeps 'LOCALISED (step N)');
    --illustrative makes the <= 10 s clip of spec 11.3 option 3: the field cycles over the cached frames 400..599 while the drones fly their
    episode, with the watermark 'illustrative - not the truth seen by the policy' (never for the policy-faithful video).

Everything that is not drawing is a pure function (log loading and truth-frame derivation, step selection, count colours, trail and
ellipse geometry, camera path, HUD / banner text, render plan) and is tested without OpenGL (tests/test_render_episode_3d.py).
render_episode() returns the manifest dict (status, frames, seconds, path, poster, notes ...); a missing optional input (buildings, slab,
particle file) skips that component and records the reason in manifest['skipped'] instead of raising.

The R3D_* constants below are read through getattr(config, name, default) so that moving them into srcloc_env/config.py needs no code change.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from srcloc_env import config
from srcloc_env.scripts.fig_episode import load_step_log


def _cfg(name: str, default: Any) -> Any:
    """Constant ``name`` from srcloc_env.config when it exists there, else ``default`` (see the module docstring)."""
    return getattr(config, name, default)


# ------------------------------------------------------------------------------------------------------ constants (R3D_*)
R3D_WINDOW = _cfg("R3D_WINDOW", (1280, 720))                    # video size (width, height) [px]
R3D_FPS = _cfg("R3D_FPS", 10)                                   # playback frames per second
R3D_STRIDE = _cfg("R3D_STRIDE", 2)                              # log steps advanced per video frame (10 fps x 2 = 20 steps/s)
R3D_HOLD_SECONDS = _cfg("R3D_HOLD_SECONDS", 2.0)                # the last frame (with the banner) is held this long
R3D_BANNER_FRAC = _cfg("R3D_BANNER_FRAC", 0.10)                 # the closing banner covers the last fraction of the step frames (and the hold)
R3D_TRAIL_STEPS = _cfg("R3D_TRAIL_STEPS", 30)                   # fading trail length [steps] (spec 11.5)
R3D_MAX_PARTICLES = _cfg("R3D_MAX_PARTICLES", 20_000)           # airborne-particle subsample size (spec 11.5 allows 100k)
R3D_DENSITY_DECADES = _cfg("R3D_DENSITY_DECADES", 4.0)          # colour range of the 15 m density plane below its episode maximum (spec 11.5)
R3D_BUILDING_MAX_TRIANGLES = _cfg("R3D_BUILDING_MAX_TRIANGLES", 40_000)   # the STL (15 169 triangles) is decimated only above this
R3D_ELEVATION_DEG = _cfg("R3D_ELEVATION_DEG", 35.0)             # camera elevation (spec 11.5)
R3D_AZIMUTH_DEG = _cfg("R3D_AZIMUTH_DEG", -60.0)                # camera azimuth: direction of the camera from the focal point is (cos az, sin az); +x points to the right of the picture
R3D_FOV_DEG = _cfg("R3D_FOV_DEG", 30.0)                         # vertical view angle
R3D_ZOOM_START_FRAC = _cfg("R3D_ZOOM_START_FRAC", 0.8)          # zoom towards the source in the last 20 % of the frames (spec 11.5)
R3D_RADIUS_MIN_M = _cfg("R3D_RADIUS_MIN_M", 200.0)              # smallest horizontal half-extent that stays in view before the zoom [m]
R3D_RADIUS_ZOOM_M = _cfg("R3D_RADIUS_ZOOM_M", 90.0)             # horizontal half-extent at the end of the zoom [m]
R3D_RADIUS_MARGIN = _cfg("R3D_RADIUS_MARGIN", 1.3)              # margin factor on the half-extent of drones + source
R3D_CAMERA_SMOOTH = _cfg("R3D_CAMERA_SMOOTH", 9)                # centred moving-average window of the camera framing [video frames]
R3D_ZOOM_ORBIT_DEG = _cfg("R3D_ZOOM_ORBIT_DEG", 25.0)           # camera azimuth turns by this angle during the final zoom (parallax for depth perception)
R3D_FOCAL_Z_M = _cfg("R3D_FOCAL_Z_M", 15.0)                     # camera focal height [m]
R3D_UNIT_FRAC = _cfg("R3D_UNIT_FRAC", 0.01)                     # marker scale unit = this fraction of the camera distance (constant on-screen size)
R3D_COLUMN_HEIGHT_M = _cfg("R3D_COLUMN_HEIGHT_M", 60.0)         # height of the source column [m]
R3D_ELLIPSE_SIGMA = _cfg("R3D_ELLIPSE_SIGMA", 2.0)              # drawn Mahalanobis radius of the GMM ellipses
R3D_ELLIPSE_POINTS = _cfg("R3D_ELLIPSE_POINTS", 72)
R3D_TOP_COMPONENTS = _cfg("R3D_TOP_COMPONENTS", 3)              # GMM components drawn (the three of config.GMM_K)
R3D_COUNT_HIGH_FACTOR = _cfg("R3D_COUNT_HIGH_FACTOR", 10.0)     # 'high count' (full red) = this factor times the Currie decision count
R3D_DRONE_COLOURS = _cfg("R3D_DRONE_COLOURS", ((0.10, 0.30, 0.85), (0.05, 0.62, 0.30), (0.55, 0.20, 0.75)))        # sphere hue per drone
R3D_TRAIL_BASE = _cfg("R3D_TRAIL_BASE", ((0.52, 0.60, 0.78), (0.52, 0.74, 0.58), (0.68, 0.56, 0.78)))             # background trail grey with a drone tint
R3D_COLOUR_DETECT = _cfg("R3D_COLOUR_DETECT", (1.00, 0.60, 0.05))                                                    # orange: at the detection threshold
R3D_COLOUR_HIGH = _cfg("R3D_COLOUR_HIGH", (0.85, 0.05, 0.05))                                                       # red: high count
R3D_COLOUR_BELIEF = _cfg("R3D_COLOUR_BELIEF", (0.78, 0.05, 0.62))                                                   # magenta: GMM ellipses and MAP marker
R3D_COLOUR_TRUTH = _cfg("R3D_COLOUR_TRUTH", (1.00, 0.78, 0.05))                                                     # gold: true source
R3D_DENSITY_CMAP = _cfg("R3D_DENSITY_CMAP", "viridis")
R3D_GROUND_MARGIN_M = _cfg("R3D_GROUND_MARGIN_M", 1500.0)             # the ground plane extends this far beyond the LDM domain (no visible platform edge)
R3D_CRF = _cfg("R3D_CRF", 20)                                   # x264 constant rate factor
R3D_BACKGROUND = _cfg("R3D_BACKGROUND", ("white", "lightsteelblue"))   # (bottom, top) of the background gradient
R3D_FONT_UNITS_720 = _cfg("R3D_FONT_UNITS_720", 11.0)           # HUD font size (pyvista units, 1 unit = 2 px) in a 720 px high window; EVERY text size scales linearly with the height
R3D_FONT_UNITS_MIN = _cfg("R3D_FONT_UNITS_MIN", 1.5)            # only a guard against a zero-size font in absurdly small windows (no practical floor)
R3D_MARGIN_PX_720 = _cfg("R3D_MARGIN_PX_720", 8.0)              # outer margin of the text panels in a 720 px high window (scales with the height)
R3D_BANNER_FONT_SCALE = _cfg("R3D_BANNER_FONT_SCALE", 1.7)      # banner font = this factor times the HUD font, shrunk until the text fits R3D_BANNER_MAX_WIDTH_FRAC
R3D_BANNER_MAX_WIDTH_FRAC = _cfg("R3D_BANNER_MAX_WIDTH_FRAC", 0.60)   # the banner (anchored at the bottom centre) is at most this fraction of the window width
R3D_PARTICLE_CMAP = _cfg("R3D_PARTICLE_CMAP", "plasma")         # airborne particles are coloured by altitude with this colour map ...
R3D_PARTICLE_Z_RANGE = _cfg("R3D_PARTICLE_Z_RANGE", (0.0, 100.0))   # ... over this height range [m] (1 % / 50 % / 99 % of the airborne particles: ~1 / 15-30 / 70-110 m)
R3D_COLOUR_RELEASE = _cfg("R3D_COLOUR_RELEASE", (0.55, 0.25, 0.00))   # dark orange-brown: the release-height ring at config.SOURCE_Z
R3D_RELEASE_RING_UNITS = _cfg("R3D_RELEASE_RING_UNITS", 2.6)    # release ring radius in marker units (R3D_UNIT_FRAC x camera distance)
R3D_STOP_AFTER_SECONDS = _cfg("R3D_STOP_AFTER_SECONDS", 3.0)    # --stop-at-localisation: the video ends this many seconds (of video) after the localisation picture, plus the R3D_HOLD_SECONDS hold of the last picture
R3D_ILLUSTRATIVE_FRAMES = _cfg("R3D_ILLUSTRATIVE_FRAMES", (400, 599))   # --illustrative: the field cycles over these cached frames (inclusive; spec 11.3 option 3)
R3D_ILLUSTRATIVE_SECONDS = _cfg("R3D_ILLUSTRATIVE_SECONDS", 10.0)       # --illustrative: clip length incl. the hold [s] (spec 11.3: <= 10 s)
R3D_ILLUSTRATIVE_WATERMARK = _cfg("R3D_ILLUSTRATIVE_WATERMARK", "illustrative - not the truth seen by the policy")
R3D_PRESETS = _cfg("R3D_PRESETS", {"final": {"width": 1920, "height": 1080, "stride": 1, "fps": 30, "steps_per_second": 5.0}})   # spec 11.5: 1920 x 1080, 5 steps/s, 30 fps encoding
_BAR_TITLES = {"particles": ["particle height", "above ground [m]"], "density": ["15 m truth density", "log10 [particles/m^3]"]}
_GROUND_RGB = (0.93, 0.93, 0.90)
_BUILDING_RGB = (0.62, 0.64, 0.68)
_FADE_RGB = (0.90, 0.92, 0.95)                                  # trail colours fade towards the background


# ------------------------------------------------------------------------------------------------------ episode log
@dataclass
class EpisodeLog:
    """One step log with every optional key resolved (None when the log does not carry it)."""

    path: Path
    meta: dict
    drone_xy: np.ndarray                 # (T, D, 2) positions after the move of each step
    y: np.ndarray                        # (T, D) counts
    truth_xy: np.ndarray                 # (2,)
    gmm_w: np.ndarray | None             # (T, K)
    gmm_mu: np.ndarray | None            # (T, K, 2)
    gmm_cov: np.ndarray | None           # (T, K, 2, 2)
    gmm_mask: np.ndarray | None          # (T, K) bool
    map_xy: np.ndarray | None            # (T, 2)
    top_sigma: np.ndarray | None         # (T,)
    map_error: np.ndarray | None         # (T,)
    frame: np.ndarray                    # (T,) truth frame of every step (logged or derived)
    frame_derived: bool
    mode: str                            # F or T2
    mode_source: str                     # where the mode came from: arg / meta / episodes.csv / frame array / default
    start_xy: np.ndarray | None          # (D, 2)
    reward: np.ndarray | None
    calib_inside: np.ndarray | None

    @property
    def T(self) -> int:
        return int(self.drone_xy.shape[0])

    @property
    def D(self) -> int:
        return int(self.drone_xy.shape[1])

    @property
    def source(self) -> int:
        return int(self.meta["source"])

    @property
    def method(self) -> str:
        return str(self.meta.get("method", "?"))

    @property
    def reflected(self) -> bool:
        return bool(self.meta.get("reflected", False))


def _episode_csv_mode(npz: Path) -> str | None:
    """Truth mode of the episode from the common list episodes.csv of the evaluation directory (<tag>/steps/<name>_epNNNN.npz), else None."""
    m = re.search(r"ep(\d+)\.npz$", Path(npz).name)
    if m is None:
        return None
    for d in (Path(npz).parent.parent, Path(npz).parent):
        f = d / "episodes.csv"
        if not f.is_file():
            continue
        try:
            with f.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if "mode" in row and "frame" in row and str(row.get("episode_id", "")).strip() == str(int(m.group(1))):
                        return str(row["mode"]).strip().upper()
        except (OSError, ValueError, csv.Error):
            return None
    return None


def frame_sequence(mode: str, frame0: int, n_steps: int, logged: np.ndarray | None = None, n_files: int = config.N_FILES,
                   files_per_step: float = config.T1_4_MODE_T2_FILES_PER_STEP) -> np.ndarray:
    """Truth frame index of every log step.  The logged array wins; otherwise Mode F = frame0 for all steps and Mode T2 =
    min(n_files - 1, round(frame0 + files_per_step * i)) for step i (SourceLocEnv.frame_at: the measurement of log step i is taken at t = i)."""
    if logged is not None and np.asarray(logged).shape == (n_steps,):
        return np.asarray(logged, dtype=np.int64)
    i = np.arange(n_steps)
    if str(mode).upper() == "T2":
        return np.minimum(n_files - 1, np.rint(frame0 + files_per_step * i)).astype(np.int64)
    return np.full(n_steps, int(frame0), dtype=np.int64)


def field_hold_step(frame: np.ndarray, mode: str, frame0: int, n_files: int = config.N_FILES) -> int | None:
    """First log step (0-based) from which a Mode T2 field is held at the last cached frame (zero-order hold), else None."""
    if str(mode).upper() != "T2":
        return None
    f = np.asarray(frame)
    held = np.flatnonzero((f >= n_files - 1) & (np.arange(f.size) + int(frame0) > n_files - 1))
    return int(held[0]) if held.size else None


def load_episode_log(npz: Path | str, mode: str | None = None) -> EpisodeLog:
    """Read a step log (fig_episode.load_step_log) and resolve every optional key; KeyError names a missing REQUIRED key
    (drone_xy, y, truth_xy, meta with source and frame).  ``mode`` overrides the truth-mode inference
    (order: argument, meta mode, the evaluation directory's episodes.csv, a varying frame array, default F)."""
    npz = Path(npz)
    arrs, meta = load_step_log(npz)
    for k in ("drone_xy", "y", "truth_xy"):
        if k not in arrs:
            raise KeyError(f"{npz.name}: required step-log key '{k}' is missing")
    for k in ("source", "frame"):
        if k not in meta:
            raise KeyError(f"{npz.name}: required meta key '{k}' is missing")
    xy = np.asarray(arrs["drone_xy"], dtype=float)
    if xy.ndim == 2:
        xy = xy[:, None, :]
    T, D = xy.shape[:2]
    y = np.asarray(arrs["y"], dtype=float).reshape(T, -1)
    if y.shape[1] != D:
        raise ValueError(f"{npz.name}: y has {y.shape[1]} columns for {D} drones")

    def opt(key: str, shape: tuple, dtype=float) -> np.ndarray | None:
        a = arrs.get(key)
        if a is None:
            return None
        a = np.asarray(a, dtype=dtype)
        return a if a.shape[:len(shape)] == shape else None

    frame_logged = arrs.get("frame")
    if mode is not None:
        use_mode, src = str(mode).upper(), "arg"
    elif "mode" in meta:
        use_mode, src = str(meta["mode"]).upper(), "meta"
    else:
        csv_mode = _episode_csv_mode(npz)
        if csv_mode in config.ENV_MODES:
            use_mode, src = csv_mode, "episodes.csv"
        elif frame_logged is not None and np.asarray(frame_logged).size > 1:
            use_mode, src = ("T2" if np.ptp(np.asarray(frame_logged)) > 0 else "F"), "frame array"
        else:
            use_mode, src = "F", "default"
    if use_mode not in config.ENV_MODES:
        raise ValueError(f"unknown truth mode {use_mode!r}; expected one of {config.ENV_MODES}")
    frame0 = int(meta["frame"])
    frame = frame_sequence(use_mode, frame0, T, frame_logged)
    start = arrs.get("start_xy")
    return EpisodeLog(
        path=npz, meta=meta, drone_xy=xy, y=y, truth_xy=np.asarray(arrs["truth_xy"], dtype=float).reshape(2),
        gmm_w=opt("gmm_w", (T,)), gmm_mu=opt("gmm_mu", (T,)), gmm_cov=opt("gmm_cov", (T,)), gmm_mask=opt("gmm_mask", (T,), bool),
        map_xy=opt("map_xy", (T,)), top_sigma=opt("top_sigma", (T,)), map_error=opt("map_error", (T,)),
        frame=frame, frame_derived=not (frame_logged is not None and np.asarray(frame_logged).shape == (T,)),
        mode=use_mode, mode_source=src, start_xy=None if start is None else np.asarray(start, dtype=float).reshape(-1, 2),
        reward=opt("reward", (T,)), calib_inside=opt("calib_inside", (T,), bool))


def localisation(top_sigma: np.ndarray | None, map_error: np.ndarray | None, sigma_m: float = config.ENV_SUCCESS_SIGMA_M,
                 error_m: float = config.ENV_SUCCESS_ERROR_M) -> tuple[int | None, float | None]:
    """(first step index (0-based) at which the top GMM component has sigma < sigma_m and the MAP error < error_m, MAP error there);
    (None, None) when the log has no belief or the criterion is never met (primary success criterion of the evaluation)."""
    if top_sigma is None or map_error is None:
        return None, None
    ok = np.flatnonzero((np.asarray(top_sigma) < sigma_m) & (np.asarray(map_error) < error_m))
    return (int(ok[0]), float(map_error[ok[0]])) if ok.size else (None, None)


# ------------------------------------------------------------------------------------------------------ pure geometry / colour helpers
def select_steps(n_steps: int, stride: int = R3D_STRIDE, max_frames: int | None = None) -> np.ndarray:
    """Log step indices shown as video frames: every ``stride``-th step plus the last one; at most ``max_frames`` (evenly thinned, first and last kept)."""
    if n_steps < 1:
        raise ValueError("an episode needs at least one step")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    steps = np.arange(0, n_steps, int(stride))
    if steps[-1] != n_steps - 1:
        steps = np.append(steps, n_steps - 1)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if steps.size > max_frames:
            pick = np.unique(np.rint(np.linspace(0, steps.size - 1, max_frames)).astype(int))
            steps = steps[pick]
    return steps


def currie_count_threshold() -> float:
    """Detection decision threshold in COUNTS per integration time (Detector.detection_threshold_cps() * T, 33.4 counts for b = 20, T = 1)."""
    b, T = float(config.SENSOR_BACKGROUND_CPS), float(config.SENSOR_T)
    return (b + float(config.SENSOR_CURRIE_K) * math.sqrt(b * T) / T) * T


def count_colour(y: np.ndarray | float, threshold: float | None = None, high: float | None = None,
                 base: Sequence[float] = (0.55, 0.55, 0.55)) -> np.ndarray:
    """Colour of a measured count (float RGB, shape y.shape + (3,)): ``base`` (grey, drone-tinted) up to the detection threshold, then orange at the
    threshold shading to red at ``high`` counts on a log scale (spec 11.5: grey background -> orange detection -> red high count)."""
    thr = currie_count_threshold() if threshold is None else float(threshold)
    hi = thr * R3D_COUNT_HIGH_FACTOR if high is None else float(high)
    if hi <= thr:
        raise ValueError("high must exceed the detection threshold")
    yy = np.asarray(y, dtype=float)
    t = np.clip(np.log(np.maximum(yy, thr) / thr) / math.log(hi / thr), 0.0, 1.0)[..., None]
    orange, red = np.asarray(R3D_COLOUR_DETECT, dtype=float), np.asarray(R3D_COLOUR_HIGH, dtype=float)
    detected = orange * (1.0 - t) + red * t
    return np.where((yy > thr)[..., None], detected, np.asarray(base, dtype=float))


def trail_geometry(xy: np.ndarray, y: np.ndarray, t: int, *, n_trail: int = R3D_TRAIL_STEPS, z: float = config.DRONE_Z,
                   threshold: float | None = None, high: float | None = None, base: Sequence[float] = (0.55, 0.55, 0.55),
                   fade_rgb: Sequence[float] = _FADE_RGB, min_weight: float = 0.25) -> dict | None:
    """Fading trail of one drone ending at log step ``t``: dict(points (m, 3), rgb uint8 (m, 3), weight (m,)) over the last ``n_trail`` steps, or None
    when fewer than two points exist.  weight = 1 at the newest point falling linearly to ``min_weight`` at age n_trail - 1; colours are the count colours
    faded towards ``fade_rgb`` with the age (so the trail dissolves into the background)."""
    t = int(t)
    lo = max(0, t - int(n_trail) + 1)
    idx = np.arange(lo, t + 1)
    if idx.size < 2:
        return None
    age = (t - idx).astype(float)
    w = 1.0 - (1.0 - float(min_weight)) * age / max(int(n_trail) - 1, 1)
    rgb = count_colour(np.asarray(y)[idx], threshold, high, base)
    rgb = rgb * w[:, None] + np.asarray(fade_rgb, dtype=float) * (1.0 - w[:, None])
    pts = np.column_stack([np.asarray(xy)[idx, 0], np.asarray(xy)[idx, 1], np.full(idx.size, float(z))])
    return {"points": pts, "rgb": np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8), "weight": w}


def ellipse_points(mu: Sequence[float], cov: np.ndarray, n_sigma: float = R3D_ELLIPSE_SIGMA, n: int = R3D_ELLIPSE_POINTS,
                   z: float = config.DRONE_Z) -> np.ndarray:
    """(n, 3) points of the n_sigma ellipse of the 2-D Gaussian (mu, cov) in the plane z (eigenvalues floored so a degenerate covariance stays drawable)."""
    c = np.asarray(cov, dtype=float).reshape(2, 2)
    vals, vecs = np.linalg.eigh(0.5 * (c + c.T))
    axes = float(n_sigma) * np.sqrt(np.maximum(vals, 1e-6))
    th = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    pts = np.asarray(mu, dtype=float)[None, :] + (np.cos(th)[:, None] * axes[0]) * vecs[:, 0][None, :] + (np.sin(th)[:, None] * axes[1]) * vecs[:, 1][None, :]
    return np.column_stack([pts, np.full(int(n), float(z))])


def ellipse_style(weight: float) -> dict:
    """Opacity / line width of a GMM ellipse from its mixture weight (spec 11.5: weights as opacity)."""
    w = float(np.clip(weight, 0.0, 1.0))
    return {"fill_opacity": 0.02 + 0.08 * w, "line_opacity": 0.30 + 0.70 * w, "line_width": 1.5 + 3.0 * w}


def star_points(centre: Sequence[float], radius: float, azimuth_deg: float, n_tips: int = 5, inner_ratio: float = 0.42) -> np.ndarray:
    """(1 + 2 n_tips, 3) vertices of a vertical star (point 0 = centre, then alternating tip / notch) in the plane facing a camera at azimuth
    ``azimuth_deg`` (billboard), tip up."""
    az = math.radians(azimuth_deg)
    right = np.array([-math.sin(az), math.cos(az), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    ang = np.pi / 2.0 + np.arange(2 * n_tips) * np.pi / n_tips
    rad = np.where(np.arange(2 * n_tips) % 2 == 0, 1.0, float(inner_ratio)) * float(radius)
    c = np.asarray(centre, dtype=float)
    ring = c[None, :] + (np.cos(ang) * rad)[:, None] * right[None, :] + (np.sin(ang) * rad)[:, None] * up[None, :]
    return np.vstack([c[None, :], ring])


def log_density_field(dens: np.ndarray, vmax: float, decades: float = R3D_DENSITY_DECADES) -> tuple[np.ndarray, tuple[float, float]]:
    """log10 of the density with every cell below vmax * 10^-decades set to NaN (drawn transparent), and the colour limits (log10 vmax - decades, log10 vmax)."""
    top = math.log10(max(float(vmax), 1e-30))
    d = np.asarray(dens, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ld = np.log10(d)
    ld[~np.isfinite(ld) | (ld < top - float(decades))] = np.nan
    return ld.astype(np.float32), (top - float(decades), top)


def subsample_points(pts: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    """At most ``max_n`` rows of ``pts`` (random without replacement, deterministic in ``seed``; the input when it is small enough)."""
    pts = np.asarray(pts)
    if max_n <= 0:
        return pts[:0]
    if pts.shape[0] <= max_n:
        return pts
    return pts[np.sort(np.random.default_rng(seed).choice(pts.shape[0], size=int(max_n), replace=False))]


# ------------------------------------------------------------------------------------------------------ camera
@dataclass
class CameraPath:
    """Per video frame: camera position / focal point, the azimuth, the horizontal half-extent in view, the distance and the zoom blend (0..1)."""

    position: np.ndarray      # (F, 3)
    focal: np.ndarray         # (F, 3)
    azimuth_deg: np.ndarray   # (F,)
    radius: np.ndarray        # (F,)
    distance: np.ndarray      # (F,)
    zoom: np.ndarray          # (F,)

    @property
    def n_frames(self) -> int:
        return int(self.position.shape[0])


def camera_distance(radius: float | np.ndarray, fov_deg: float, aspect: float, elevation_deg: float, fill: float = 0.9) -> np.ndarray:
    """Smallest camera distance from the focal point (camera at elevation ``elevation_deg``, vertical view angle ``fov_deg``, picture ``aspect`` = w / h)
    at which the ground disc of horizontal radius ``radius`` around the focal point stays inside ``fill`` of the view: the near and far ground points
    on the viewing azimuth fit the vertical half angle (perspective foreshortening included, solved by bisection) and the side points fit the horizontal
    half angle."""
    r = np.asarray(radius, dtype=float)
    el = math.radians(elevation_deg)
    half_v = math.radians(fov_deg) / 2.0
    half_h = math.atan(math.tan(half_v) * float(aspect))
    lim_v, lim_h = float(fill) * half_v, float(fill) * half_h
    d_h = r / math.tan(lim_h)                                                       # lateral points at +-r: the angle is ~ atan(r / d)
    hi = np.maximum(d_h, 1.0) * 40.0
    lo = np.zeros_like(hi)
    for _ in range(60):                                                             # near point (depression above the axis) and far point must fit
        d = 0.5 * (lo + hi)
        near = np.arctan2(d * math.sin(el), d * math.cos(el) - r)                   # depression of the near ground point (negative denominator: behind the camera)
        near = np.where(d * math.cos(el) - r <= 0.0, np.pi, near)
        far = np.arctan2(d * math.sin(el), d * math.cos(el) + r)
        ok = (near - el <= lim_v) & (el - far <= lim_v)
        hi = np.where(ok, d, hi)
        lo = np.where(ok, lo, d)
    return np.maximum(hi, d_h)


def marker_scale(distance: float | np.ndarray) -> np.ndarray:
    """Marker unit [m] for a camera distance: R3D_UNIT_FRAC of it, so spheres, tubes and stars keep their on-screen size while the camera zooms."""
    return R3D_UNIT_FRAC * np.asarray(distance, dtype=float)


def _smooth(a: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along axis 0 with edge replication (odd window <= len(a))."""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    w = max(1, min(int(window), n if n % 2 == 1 else n - 1))
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return a.copy()
    pad = w // 2
    p = np.pad(a, [(pad, pad)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    k = np.ones(w) / w
    if a.ndim == 1:
        return np.convolve(p, k, mode="valid")
    return np.stack([np.convolve(p[:, j], k, mode="valid") for j in range(p.shape[1])], axis=1)


def _rolling_max(a: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling maximum along axis 0 with edge replication (odd window <= len(a))."""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    w = max(1, min(int(window), n if n % 2 == 1 else n - 1))
    if w % 2 == 0:
        w -= 1
    pad = w // 2
    p = np.pad(a, (pad, pad), mode="edge")
    return np.max(np.stack([p[i:i + n] for i in range(w)], axis=0), axis=0)


def camera_path(drone_xy: np.ndarray, truth_xy: Sequence[float], *, aspect: float = R3D_WINDOW[0] / R3D_WINDOW[1],
                elevation_deg: float = R3D_ELEVATION_DEG, azimuth_deg: float = R3D_AZIMUTH_DEG, fov_deg: float = R3D_FOV_DEG,
                zoom_start_frac: float = R3D_ZOOM_START_FRAC, radius_min: float = R3D_RADIUS_MIN_M, radius_zoom: float = R3D_RADIUS_ZOOM_M,
                margin: float = R3D_RADIUS_MARGIN, smooth: int = R3D_CAMERA_SMOOTH, orbit_deg: float = R3D_ZOOM_ORBIT_DEG,
                focal_z: float = R3D_FOCAL_Z_M) -> CameraPath:
    """Camera path over F video frames from the drone positions at those frames ``drone_xy`` (F, D, 2) and the true source.

    Before the zoom the focal point is the centre of the bounding box of the drones and the source and the half-extent in view is the largest
    distance from that centre to one of them (times ``margin``, at least ``radius_min``), both smoothed with a moving average, so the camera follows
    the drones and always keeps the source in view.  In the last (1 - zoom_start_frac) of the frames the focal point glides to the source and the
    half-extent to ``radius_zoom`` (smoothstep), while the azimuth turns by ``orbit_deg``; the zoom stops short of the source when a drone is
    still far away (failed episode), so the drones never leave the picture."""
    d = np.asarray(drone_xy, dtype=float)
    if d.ndim != 3 or d.shape[2] != 2:
        raise ValueError(f"drone_xy must have shape (F, D, 2), got {d.shape}")
    F = d.shape[0]
    truth = np.asarray(truth_xy, dtype=float).reshape(2)
    pts = np.concatenate([d, np.broadcast_to(truth, (F, 1, 2))], axis=1)                  # (F, D + 1, 2)
    centre = 0.5 * (pts.min(axis=1) + pts.max(axis=1))
    rad = np.hypot(*(pts - centre[:, None, :]).transpose(2, 0, 1)).max(axis=1) * float(margin)
    rad = np.maximum(rad, float(radius_min))
    centre_s = _smooth(centre, smooth)
    need = rad + np.hypot(*(centre_s - centre).T)                                           # keep the subjects in view although the focal point is smoothed
    rad = _smooth(_rolling_max(need, smooth), smooth)                                     # max filter then average: never below the need of a frame
    centre = centre_s
    frac = np.linspace(0.0, 1.0, F) if F > 1 else np.ones(1)
    u = np.clip((frac - float(zoom_start_frac)) / max(1.0 - float(zoom_start_frac), 1e-9), 0.0, 1.0)
    s = u * u * (3.0 - 2.0 * u)
    focal_xy = centre * (1.0 - s[:, None]) + truth[None, :] * s[:, None]
    radius = rad * (1.0 - s) + float(radius_zoom) * s
    need = np.hypot(*(pts - focal_xy[:, None, :]).transpose(2, 0, 1)).max(axis=1) * float(margin)      # a drone far from the source stays in view during the zoom
    radius = np.maximum(radius, _smooth(_rolling_max(need, smooth), smooth))
    az = float(azimuth_deg) + float(orbit_deg) * s
    dist = camera_distance(radius, fov_deg, aspect, elevation_deg)
    el = math.radians(elevation_deg)
    azr = np.radians(az)
    direction = np.stack([math.cos(el) * np.cos(azr), math.cos(el) * np.sin(azr), np.full(F, math.sin(el))], axis=1)
    focal = np.column_stack([focal_xy, np.full(F, float(focal_z))])
    return CameraPath(position=focal + dist[:, None] * direction, focal=focal, azimuth_deg=az, radius=radius, distance=dist, zoom=s)


# ------------------------------------------------------------------------------------------------------ text
def ldm_time_s(frame: int) -> tuple[int, float]:
    """(LDM step, LDM time [s] under interpretation A) of the cached frame index ``frame`` (step -1 outside the cache)."""
    step = config.index_to_step(int(frame)) if 0 <= int(frame) < config.N_FILES else -1
    return step, step * config.SEC_PER_INDEX_STEP


def illustrative_frames(n: int, first: int | None = None, last: int | None = None) -> np.ndarray:
    """Truth frame of each of ``n`` video frames of the illustrative clip (spec 11.3 option 3): the field cycles ONCE over the cached frames
    first .. last (default R3D_ILLUSTRATIVE_FRAMES), so the clip loops seamlessly (frame n would be ``first`` again)."""
    lo, hi = (R3D_ILLUSTRATIVE_FRAMES if first is None or last is None else (first, last))
    count = int(hi) - int(lo) + 1
    if n < 1 or count < 1:
        raise ValueError("an illustrative clip needs n >= 1 video frames and first <= last")
    return (int(lo) + (np.arange(int(n)) * count) // int(n)).astype(np.int64)


def mode_caption(log: EpisodeLog, k_step: int, frame: int | None = None, illustrative: bool = False) -> list[str]:
    """Caption lines: method / drones / source / mode, then the truth-field statement ('frozen snapshot, frame k' for Mode F; the frame advances with the
    step in Mode T2, 'field frozen' once the cached frames end; 'ILLUSTRATIVE' for the clip of spec 11.3 option 3) and the LDM step / time of that frame
    (interpretation A).  ``frame`` overrides the logged truth frame of the step (illustrative clip)."""
    frame = int(log.frame[k_step]) if frame is None else int(frame)
    ldm_step, t_phys = ldm_time_s(frame)
    head = f"{log.method}, {log.D} drone{'s' if log.D > 1 else ''}, source {log.source}, Mode {log.mode}"
    if illustrative:
        lo, hi = R3D_ILLUSTRATIVE_FRAMES
        field_line = f"truth field: ILLUSTRATIVE cycle, frame {frame} of {lo}..{hi}"
    elif log.mode == "F":
        field_line = f"truth field: frozen snapshot, frame {frame}"
    else:
        hold = field_hold_step(log.frame, log.mode, int(log.meta["frame"]))
        field_line = f"truth field: time-varying, frame {frame} (start {int(log.meta['frame'])} + t)"
        if hold is not None and k_step >= hold:
            field_line = f"truth field: frozen at frame {frame} from step {hold + 1}"
    return [head, field_line, f"LDM step {ldm_step}, t = {t_phys:.0f} s (interpretation A)"]


def hud_lines(log: EpisodeLog, k_step: int, loc_step: int | None, frame: int | None = None) -> list[str]:
    """HUD lines for log step ``k_step`` (0-based): [0] step and EPISODE time (the RL clock: step x config.RL_STEP_SECONDS), [1] counts, [2] top sigma,
    [3] MAP error, [4] status (the localisation step stays in it), [5] LDM time of the truth field (interpretation A; a different clock: the time of
    the simulated plume, not of the episode)."""
    T = log.T
    counts = "  ".join(f"D{d + 1} {int(round(log.y[k_step, d])):4d}" for d in range(log.D))
    sig = "n/a" if log.top_sigma is None else f"{log.top_sigma[k_step]:6.1f} m"
    err = "n/a" if log.map_error is None else f"{log.map_error[k_step]:6.1f} m"
    status = "searching" if (loc_step is None or k_step < loc_step) else f"LOCALISED (step {loc_step + 1})"
    ldm_t = ldm_time_s(int(log.frame[k_step]) if frame is None else int(frame))[1]
    return [f"step {k_step + 1:3d}/{T}   episode time {(k_step + 1) * config.RL_STEP_SECONDS:.0f} s",
            f"counts {counts}",
            f"top sigma  {sig}",
            f"MAP error  {err}",
            f"status: {status}",
            f"LDM t {ldm_t:.0f} s (interp. A)"]


def banner_text(log: EpisodeLog, loc_step: int | None, loc_error: float | None) -> tuple[str, str]:
    """(closing banner text, 'ok' | 'fail').  Localised: 'LOCALISED in N steps (error e m)'; a log without belief arrays whose meta says success:
    'LOCALISED (step unknown)' (no step is claimed that the log cannot confirm); otherwise 'NOT LOCALISED after T steps (final error e m)'."""
    if loc_step is not None and loc_error is not None:
        return f"LOCALISED in {loc_step + 1} steps (error {loc_error:.1f} m)", "ok"
    if bool(log.meta.get("success", False)):                                              # logged as a success but the belief arrays cannot say when
        return "LOCALISED (step unknown)", "ok"
    if log.map_error is None:
        return f"NOT LOCALISED after {log.T} steps", "fail"
    return f"NOT LOCALISED after {log.T} steps (final error {float(log.map_error[-1]):.0f} m)", "fail"


# ------------------------------------------------------------------------------------------------------ render plan
def frame_repeat(fps: int, stride: int, steps_per_second: float | None) -> int:
    """Encoded frames per rendered picture so that the playback speed is ``steps_per_second`` log steps per second (None: 1, i.e. fps x stride steps
    per second).  Spec 11.5: 5 steps/s encoded at 30 fps with one step per picture -> 6 copies of each picture."""
    if steps_per_second is None:
        return 1
    if steps_per_second <= 0:
        raise ValueError("steps_per_second must be positive")
    return max(1, int(round(float(fps) * int(stride) / float(steps_per_second))))


@dataclass
class RenderPlan:
    """Everything the draw loop needs, computed without OpenGL."""

    steps: np.ndarray             # (F,) log step index of every video frame
    frames: np.ndarray            # (F,) truth frame of every video frame
    camera: CameraPath
    hold_frames: int              # extra copies of the last frame at the end
    banner_from: int              # first video frame (index into steps) that carries the banner (the localisation frame when localised)
    loc_step: int | None          # 0-based log step of the first localisation
    loc_error: float | None
    banner: tuple[str, str]
    field_frames: list[int]       # distinct truth frames shown, ascending
    repeat: int = 1               # encoded copies of every rendered picture (playback speed, see frame_repeat)
    n_steps_used: int = 0         # log steps covered by the video (< log.T when --stop-at-localisation ended it early)
    illustrative: bool = False    # the field cycles over the cached frames independent of the log (spec 11.3 option 3)


def plan_render(log: EpisodeLog, *, stride: int = R3D_STRIDE, max_frames: int | None = None, fps: int = R3D_FPS,
                hold_seconds: float = R3D_HOLD_SECONDS, aspect: float = R3D_WINDOW[0] / R3D_WINDOW[1], steps_per_second: float | None = None,
                stop_at_localisation: bool = False, stop_after_seconds: float | None = None, illustrative: bool = False,
                **cam_kw: Any) -> RenderPlan:
    """Select the video frames, their truth frames, the camera path and the banner timing for ``log``.

    stop_at_localisation: the video ends ``stop_after_seconds`` (default R3D_STOP_AFTER_SECONDS) of video after the first localisation (when the log
    has one); the banner is shown from the first picture at or after the localisation step (without a localisation: during the last R3D_BANNER_FRAC
    of the pictures).  illustrative: ignore the logged truth frames, cycle the field over R3D_ILLUSTRATIVE_FRAMES and fit the whole episode into
    R3D_ILLUSTRATIVE_SECONDS, hold included (an explicit smaller ``max_frames`` is respected; stop_at_localisation is ignored)."""
    rep = frame_repeat(fps, stride, steps_per_second)
    loc_step, loc_err = localisation(log.top_sigma, log.map_error)
    n_use = log.T
    if illustrative:
        stride = 1
        n_clip = max(1, int(round((R3D_ILLUSTRATIVE_SECONDS - hold_seconds) * fps / rep)))
        max_frames = n_clip if max_frames is None else min(int(max_frames), n_clip)                # an explicit cap on the pictures is respected
    elif stop_at_localisation and loc_step is not None:
        after = R3D_STOP_AFTER_SECONDS if stop_after_seconds is None else float(stop_after_seconds)
        extra_pictures = int(math.ceil(after * fps / rep))
        n_use = min(log.T, loc_step + 1 + extra_pictures * int(stride))
    steps = select_steps(n_use, stride, max_frames)
    cam = camera_path(log.drone_xy[steps], log.truth_xy, aspect=aspect, **cam_kw)
    banner = banner_text(log, loc_step, loc_err)
    n = steps.size
    if loc_step is not None:
        banner_from = int(np.searchsorted(steps, loc_step, side="left"))                  # the first picture at or after the localisation step
    else:
        banner_from = max(0, n - max(1, int(math.ceil(R3D_BANNER_FRAC * n))))
    frames = illustrative_frames(n) if illustrative else log.frame[steps]
    return RenderPlan(steps=steps, frames=frames, camera=cam, hold_frames=max(0, int(round(hold_seconds * fps))), banner_from=min(banner_from, n - 1),
                      loc_step=loc_step, loc_error=loc_err, banner=banner, field_frames=sorted({int(f) for f in frames}), repeat=rep,
                      n_steps_used=int(n_use), illustrative=bool(illustrative))


# ------------------------------------------------------------------------------------------------------ scene resources (cached)
_BUILDING_CACHE: dict[tuple, Any] = {}
_BACKEND_CACHE: dict[str, Any] = {}


def load_buildings(path: Path = config.STL_FILE, shift: Sequence[float] = tuple(config.STL_SHIFT), max_triangles: int = R3D_BUILDING_MAX_TRIANGLES,
                   mirror_y: bool = False):
    """Building mesh (pyvista PolyData) of the STL shifted into the fluid / LDM frame, decimated above ``max_triangles``, cached in memory.
    ``mirror_y`` reflects the scene (y -> -y) for y-reflected episodes."""
    import pyvista as pv
    key = (str(path), tuple(float(s) for s in shift), int(max_triangles), bool(mirror_y))
    if key in _BUILDING_CACHE:
        return _BUILDING_CACHE[key]
    mesh = pv.read(str(path)).translate(np.asarray(shift, dtype=float), inplace=False)
    if mesh.n_cells > max_triangles:
        mesh = mesh.decimate_pro(1.0 - max_triangles / mesh.n_cells, preserve_topology=True)
    if mirror_y:
        pts = np.array(mesh.points)
        pts[:, 1] *= -1.0
        mesh = pv.PolyData(pts, mesh.faces).flip_faces()
    _BUILDING_CACHE[key] = mesh
    return mesh


def default_backend():
    """The 15 m truth slab backend: the memory-mapped StackedSlabBackend when the stack file exists, else the per-frame LdmSlabBackend (cached per process)."""
    if "b" not in _BACKEND_CACHE:
        from srcloc_env.field.concentration_field import LdmSlabBackend
        from srcloc_env.field.slab_stack import StackedSlabBackend
        if Path(config.SLAB_STACK_PATH).is_file() and Path(config.SLAB_STACK_META_PATH).is_file():
            _BACKEND_CACHE["b"] = StackedSlabBackend()
        else:
            _BACKEND_CACHE["b"] = LdmSlabBackend()
    return _BACKEND_CACHE["b"]


def source_density(backend, src: int, frame: int, reflected: bool = False, z: float = config.DRONE_Z) -> tuple[np.ndarray, Any]:
    """((ny, nx) float density of source ``src`` at the drone slab of ``frame``, SlabGrid).  y-reflected scenes sample the original slab at (x, -y)."""
    sf = backend.slab(int(frame))
    g = sf.grid
    if not reflected:
        return np.asarray(sf.density[sf.source_indices(int(src))[0], sf.z_index(z)], dtype=np.float32), g
    xs = g.x0 + g.res * (np.arange(g.nx) + 0.5)
    ys = g.y0 + g.res * (np.arange(g.ny) + 0.5)
    gx, gy = np.meshgrid(xs, ys)
    d = backend.density([int(src)], np.column_stack([gx.ravel(), gy.ravel()]), int(frame), z, 1.0, flip_y=True)
    return np.asarray(d, dtype=np.float32).reshape(g.ny, g.nx), g


# ------------------------------------------------------------------------------------------------------ drawing (OpenGL)

class _Prim:
    """A fixed-topology mesh moved / scaled by rewriting its points (no actor rebuild)."""

    def __init__(self, mesh) -> None:
        self.mesh = mesh
        self.base = np.array(mesh.points, dtype=float)

    def place(self, centre: Sequence[float], scale: float | Sequence[float]) -> None:
        self.mesh.points = self.base * np.asarray(scale, dtype=float) + np.asarray(centre, dtype=float)[None, :]


class _Scene:
    """The reusable PyVista scene: built once, then updated per video frame.

    Screen layout (all sizes scale linearly with the window height; checked by tests/test_render_episode_3d.py::test_text_panels_do_not_overlap):
      top left: caption;  top right: HUD;  right column below the HUD: colour bars with their titles (particle height, 15 m density);
      bottom centre: banner (sized by its text) and, in the illustrative clip, the watermark above it;  bottom left, above that stack: legend."""

    def __init__(self, log: EpisodeLog, plan: RenderPlan, width: int, height: int, backend, buildings, particles: bool, max_particles: int,
                 skipped: dict[str, str]) -> None:
        import pyvista as pv
        self.pv = pv
        self.log, self.plan, self.W, self.H = log, plan, int(width), int(height)
        self.backend, self.skipped = backend, skipped
        self.max_particles = int(max_particles)
        self.use_particles = bool(particles)
        self.det_thr = currie_count_threshold()
        self.pl = pv.Plotter(off_screen=True, window_size=(self.W, self.H))
        try:
            self.pl.enable_anti_aliasing("msaa", multi_samples=4)
        except Exception:                                                                # noqa: BLE001 - no multisampling on this GL context: draw aliased
            pass
        self.pl.set_background(R3D_BACKGROUND[0], top=R3D_BACKGROUND[1])
        self.scale = self.H / 720.0
        self.fs = max(float(R3D_FONT_UNITS_MIN), float(R3D_FONT_UNITS_720) * self.scale)     # HUD font [pyvista units; 1 unit = 2 px]: linear in H, no floor
        self.m = float(R3D_MARGIN_PX_720) * self.scale                                       # margin [px]
        self._dens_frame: int | None = None
        self._part_frame: int | None = None
        self.particles_shown = 0
        self.dens_mesh = None
        self.part_mesh = None
        self.vmax = 1.0
        self.bars: dict[str, Any] = {}                                                       # name -> vtkScalarBarActor (created by the builders below)
        self._build_ground_and_buildings(buildings)
        self._build_density()
        self._build_particles()
        self._build_markers()
        self._build_text()

    # ---------------------------------------------------------------- scene construction
    def _extent(self) -> tuple[float, float, float, float]:
        lo = np.array([config.DOMAIN_X[0], config.DOMAIN_Y[0]], dtype=float) - R3D_GROUND_MARGIN_M
        hi = np.array([config.DOMAIN_X[1], config.DOMAIN_Y[1]], dtype=float) + R3D_GROUND_MARGIN_M
        pts = self.log.drone_xy.reshape(-1, 2)
        lo = np.minimum(lo, pts.min(axis=0) - 40.0)
        hi = np.maximum(hi, pts.max(axis=0) + 40.0)
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    def _build_ground_and_buildings(self, buildings) -> None:
        pv = self.pv
        x0, y0, x1, y1 = self._extent()
        ground = pv.Plane(center=(0.5 * (x0 + x1), 0.5 * (y0 + y1), -0.4), direction=(0, 0, 1), i_size=x1 - x0, j_size=y1 - y0)
        self.pl.add_mesh(ground, color=_GROUND_RGB, lighting=False, reset_camera=False, name="ground")
        if buildings is None:
            self.skipped.setdefault("buildings", "no building mesh available")
        else:
            self.pl.add_mesh(buildings, color=_BUILDING_RGB, smooth_shading=False, ambient=0.35, diffuse=0.75, show_edges=False, reset_camera=False,
                             name="buildings")

    def _bar_args(self, title: str) -> dict:
        """Scalar-bar arguments (positions are set by the layout); the bar's own title stays blank (a unique blank string per bar) because the titles are
        text actors placed by the layout."""
        fs = max(3, int(round(self.fs)))
        return dict(title=title, vertical=True, position_x=0.90, position_y=0.10, height=0.17, width=0.035, n_labels=5, fmt="%.1f",
                    title_font_size=fs, label_font_size=fs, color="black")

    def _density_frames(self) -> list[int]:
        """The truth frames over which the colour scale (episode maximum) is taken: all of them, at most 40 evenly spread (first and last included)."""
        ff = self.plan.field_frames
        if len(ff) <= 40:
            return list(ff)
        return [ff[int(i)] for i in np.unique(np.rint(np.linspace(0, len(ff) - 1, 40)).astype(int))]

    def _build_density(self) -> None:
        pv = self.pv
        if self.backend is None:
            self.skipped.setdefault("density", "no slab backend available")
            return
        try:
            dens0, g = source_density(self.backend, self.log.source, int(self.plan.frames[0]), self.log.reflected)
            vmax = float(dens0.max())
            for f in self._density_frames():                                                  # fixed colour scale over the whole episode
                vmax = max(vmax, float(source_density(self.backend, self.log.source, f, self.log.reflected)[0].max()))
        except (FileNotFoundError, OSError) as e:
            self.skipped["density"] = f"slab not readable: {e}"
            return
        self.vmax = max(vmax, 1e-12)
        ld, clim = log_density_field(dens0, self.vmax)
        img = pv.ImageData(dimensions=(g.nx + 1, g.ny + 1, 1), spacing=(g.res, g.res, 1.0), origin=(g.x0, g.y0, config.DRONE_Z))
        img.cell_data["log10 density"] = ld.ravel()
        self.dens_mesh = img
        self._dens_frame = int(self.plan.frames[0])
        self.pl.add_mesh(img, scalars="log10 density", cmap=R3D_DENSITY_CMAP, clim=clim, nan_opacity=0.0, opacity=[float(v) for v in np.linspace(0.12, 0.85, 16)],
                         lighting=False, reset_camera=False, name="density", scalar_bar_args=self._bar_args(" "))
        self.bars["density"] = self.pl.scalar_bars[" "]

    def _build_particles(self) -> None:
        pv = self.pv
        if not self.use_particles:
            return
        if self.backend is None or not hasattr(self.backend, "airborne_particles"):
            self.skipped["particles"] = "backend cannot supply airborne particles"
            self.use_particles = False
            return
        self.part_mesh = pv.PolyData(np.zeros((1, 3), dtype=np.float32))
        self.part_mesh.point_data["height"] = np.zeros(1, dtype=np.float32)
        self.part_actor = self.pl.add_mesh(self.part_mesh, scalars="height", cmap=R3D_PARTICLE_CMAP, clim=tuple(float(v) for v in R3D_PARTICLE_Z_RANGE),
                                           point_size=max(1.0, 2.0 * self.scale), opacity=0.5, render_points_as_spheres=False, lighting=False,
                                           reset_camera=False, name="particles", scalar_bar_args=self._bar_args("  "))
        self.bars["particles"] = self.pl.scalar_bars["  "]

    def _tube(self, pts: np.ndarray, rgb: np.ndarray, weight: np.ndarray, unit: float):
        line = self.pv.lines_from_points(pts)
        line.point_data["rgb"] = rgb
        line.point_data["w"] = np.asarray(weight, dtype=float)
        return line.tube(radius=0.2 * unit, scalars="w", radius_factor=2.5, n_sides=8)

    def _build_markers(self) -> None:
        pv, pl = self.pv, self.pl
        self.drone_prims: list[_Prim] = []
        self.trail_meshes: list = []
        self.trail_actors: list = []
        for d in range(self.log.D):
            sph = _Prim(pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=24))
            a = pl.add_mesh(sph.mesh, color=R3D_DRONE_COLOURS[d % len(R3D_DRONE_COLOURS)], reset_camera=False, name=f"drone{d}")
            a.prop.interpolation = "Phong"                                                    # smooth shading without add_mesh copying the mesh
            self.drone_prims.append(sph)
            tube = self._tube(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.zeros((2, 3), dtype=np.uint8), np.array([0.5, 1.0]), 1.0)
            self.trail_meshes.append(tube)
            self.trail_actors.append(pl.add_mesh(tube, scalars="rgb", rgb=True, lighting=False, reset_camera=False, name=f"trail{d}"))
        # belief: filled ellipses + outlines on the 15 m plane (top components), MAP marker
        n = R3D_ELLIPSE_POINTS
        self.ell_fill, self.ell_line, self.ell_fill_a, self.ell_line_a = [], [], [], []
        fan = np.column_stack([np.full(n, 3), np.zeros(n, dtype=int), np.arange(1, n + 1), np.r_[np.arange(2, n + 1), 1]]).ravel()
        loop = np.r_[n + 1, np.arange(n), 0]
        for j in range(R3D_TOP_COMPONENTS):
            fill = pv.PolyData(np.zeros((n + 1, 3)), faces=fan)
            line = pv.PolyData(np.zeros((n, 3)), lines=loop)
            self.ell_fill.append(fill)
            self.ell_line.append(line)
            self.ell_fill_a.append(pl.add_mesh(fill, color=R3D_COLOUR_BELIEF, opacity=0.1, lighting=False, reset_camera=False, name=f"efill{j}"))
            self.ell_line_a.append(pl.add_mesh(line, color=R3D_COLOUR_BELIEF, line_width=3.0, render_lines_as_tubes=True, lighting=False,
                                               reset_camera=False, name=f"eline{j}"))
        self.map_prim = _Prim(pv.Octahedron(radius=1.0))
        self.map_actor = pl.add_mesh(self.map_prim.mesh, color=R3D_COLOUR_BELIEF, reset_camera=False, name="map")
        # truth: release point (small sphere + ring at the real release height SOURCE_Z), column (+ halo once localised), billboard star with outline
        tx, ty = self.log.truth_xy
        self.base_prim = _Prim(pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=16))
        pl.add_mesh(self.base_prim.mesh, color=R3D_COLOUR_TRUTH, lighting=False, reset_camera=False, name="truth_base")
        self.ring_prim = _Prim(pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.16, u_res=48, v_res=12, w_res=1))
        pl.add_mesh(self.ring_prim.mesh, color=R3D_COLOUR_RELEASE, lighting=False, reset_camera=False, name="release_ring")
        self.col_prim = _Prim(pv.Cylinder(center=(0, 0, 0.5), direction=(0, 0, 1), radius=1.0, height=1.0, resolution=24, capping=True))
        self.col_actor = pl.add_mesh(self.col_prim.mesh, color=R3D_COLOUR_TRUTH, opacity=0.55, lighting=False, reset_camera=False, name="column")
        self.halo_prim = _Prim(pv.Cylinder(center=(0, 0, 0.5), direction=(0, 0, 1), radius=1.0, height=1.0, resolution=24, capping=True))
        self.halo_actor = pl.add_mesh(self.halo_prim.mesh, color=R3D_COLOUR_TRUTH, opacity=0.22, lighting=False, reset_camera=False, name="halo")
        self.halo_actor.SetVisibility(False)
        sp = star_points((tx, ty, config.SOURCE_Z), 1.0, 0.0)
        fan10 = np.column_stack([np.full(10, 3), np.zeros(10, dtype=int), np.arange(1, 11), np.r_[np.arange(2, 11), 1]]).ravel()
        self.star = pv.PolyData(sp, faces=fan10)
        self.star_edge = pv.PolyData(sp.copy(), lines=np.r_[11, np.arange(1, 11), 1])
        pl.add_mesh(self.star, color=R3D_COLOUR_TRUTH, lighting=False, reset_camera=False, name="star")
        pl.add_mesh(self.star_edge, color="black", line_width=2.0, render_lines_as_tubes=True, lighting=False, reset_camera=False, name="star_edge")

    # ---------------------------------------------------------------- text panels and their layout
    def _font_px(self, units: float) -> int:
        """Font size [px] of a pyvista font size ``units`` (pyvista doubles it): rounded to the pixel, so the rounding error is half that of whole units."""
        return max(3, int(round(2.0 * float(units))))

    def _text(self, xy: tuple[float, float], units: float, name: str, *, h: str = "left", v: str = "bottom", font: str | None = None, bold: bool = False):
        """A pixel-positioned text actor on a translucent white panel (legible over the busy scene); update it through ``.input``."""
        a = self.pl.add_text(" ", position=xy, font_size=10, font=font, color="black", name=name)
        a.prop.font_size = self._font_px(units)
        a.prop.justification_horizontal = h
        a.prop.justification_vertical = v
        a.prop.background_color = "white"
        a.prop.background_opacity = 0.72
        a.prop.bold = bold
        return a

    def _box(self, a) -> tuple[float, float, float, float]:
        """Pixel box (x0, y0, x1, y1; origin bottom left) of the text actor ``a`` with its current text, font and anchor."""
        bb = [0.0] * 4
        a.GetBoundingBox(self.pl.renderer, bb)
        x, y = a.GetPosition()
        return float(x + bb[0]), float(y + bb[2]), float(x + bb[1]), float(y + bb[3])

    def _set_text(self, a, text: str, units: float) -> tuple[float, float, float, float]:
        a.input = text
        a.prop.font_size = self._font_px(units)
        return self._box(a)

    def _lift_to(self, a, y_min: float) -> tuple[float, float, float, float]:
        """Move the text actor ``a`` vertically so that the BOTTOM of its box (the text anchor plus the descent of the font) is at ``y_min``."""
        x, y = a.GetPosition()
        a.SetPosition(x, y + (y_min - self._box(a)[1]))
        return self._box(a)

    def _drop_to(self, a, y_max: float) -> tuple[float, float, float, float]:
        """Move the text actor ``a`` vertically so that the TOP of its box (the text anchor plus the ascent of the font) is at ``y_max``."""
        x, y = a.GetPosition()
        a.SetPosition(x, y + (y_max - self._box(a)[3]))
        return self._box(a)

    def _fit_width(self, a, text: str, units: float, max_w: float) -> float:
        """Largest font (<= ``units``) at which ``text`` is at most ``max_w`` px wide in ``a`` (the width is linear in the pixel font size)."""
        for _ in range(6):
            x0, _, x1, _ = self._set_text(a, text, units)
            w = x1 - x0
            if w <= max_w:
                break
            units *= max(0.4, 0.985 * max_w / max(w, 1.0))
        return units

    @staticmethod
    def _longest(texts: Sequence[str]) -> str:
        """The text with the longest line (a monospace font: that line decides the width)."""
        return max(texts, key=lambda t: (max(len(line) for line in t.split("\n")), t.count("\n")))

    def _note_lines(self, shown: int | None = None) -> list[str]:
        if not self.use_particles:
            return ["particles not shown"]
        n = self.particles_shown if shown is None else shown
        return [f"particles: airborne only{f', {n} shown' if n else ''}", "(deposited ones are not in the cache)"]

    def _caption_text(self, k: int, i: int, shown: int | None = None) -> str:
        return "\n".join(mode_caption(self.log, i, frame=int(self.plan.frames[k]), illustrative=self.plan.illustrative) + self._note_lines(shown))

    def _hud_text(self, k: int, i: int) -> str:
        return "\n".join(hud_lines(self.log, i, self.plan.loc_step, int(self.plan.frames[k])))

    def _legend_entries(self) -> list[list]:
        pv = self.pv
        star2d = star_points((0.0, 0.0, 0.0), 1.0, 0.0)[:, 1:]                                   # (right, up) coordinates of the billboard star
        star_face = pv.PolyData(np.column_stack([star2d, np.zeros(len(star2d))]), faces=np.column_stack([np.full(10, 3), np.zeros(10, dtype=int),
                                np.arange(1, 11), np.r_[np.arange(2, 11), 1]]).ravel())
        ring_face = pv.Disc(center=(0.0, 0.0, 0.0), inner=0.30, outer=0.5, normal=(0, 0, 1), r_res=1, c_res=24)
        entries: list[list] = [["trail: background count", (0.55, 0.60, 0.70)], ["trail: detection (Currie)", R3D_COLOUR_DETECT],
                               ["trail: high count", R3D_COLOUR_HIGH]]
        for d in range(self.log.D):
            entries.append([f"drone {d + 1}", R3D_DRONE_COLOURS[d % len(R3D_DRONE_COLOURS)]])
        entries += [["GMM 2-sigma ellipse (opacity = weight)", R3D_COLOUR_BELIEF], ["MAP estimate", R3D_COLOUR_BELIEF],
                    [f"star: true source (column top {R3D_COLUMN_HEIGHT_M:g} m)", R3D_COLOUR_TRUTH, star_face],
                    [f"ring: release height {config.SOURCE_Z:g} m", R3D_COLOUR_RELEASE, ring_face]]
        return entries

    def _build_text(self) -> None:
        """Create the text actors and lay the panels out from MEASURED sizes of their worst-case texts (nothing is placed by guessing a width)."""
        pl, W, H, m, fs, plan = self.pl, self.W, self.H, self.m, self.fs, self.plan
        self.t_hud = self._text((W - m, H - m), fs, "hud", h="right", v="top", font="courier")
        self.t_cap = self._text((m, H - m), 0.9 * fs, "caption", h="left", v="top", font="courier")
        self.t_banner = self._text((0.5 * W, m), fs * R3D_BANNER_FONT_SCALE, "banner", h="center", v="bottom", bold=True)
        self.t_wm = self._text((0.5 * W, m), fs * 1.1, "watermark", h="center", v="bottom", bold=True) if plan.illustrative else None
        self.t_bars: dict[str, Any] = {}
        for name in self.bars:
            self.t_bars[name] = self._text((W - m, 0.5 * H), 0.8 * fs, f"bar_title_{name}", h="right", v="bottom")
        self._text_actors = {"hud": self.t_hud, "caption": self.t_cap, "banner": self.t_banner, **({"watermark": self.t_wm} if self.t_wm is not None else {}),
                             **{f"bar_title_{n}": a for n, a in self.t_bars.items()}}
        # the particle count shown is only known while drawing: the caption is measured with the largest possible one
        hud_worst = self._longest([self._hud_text(k, int(i)) for k, i in enumerate(plan.steps)])
        cap_worst = self._longest([self._caption_text(k, int(i), self.max_particles) for k, i in enumerate(plan.steps)])
        pl.render()                                                                              # text sizes need a live render window
        # 1. HUD (top right) and caption (top left) side by side: shrink both by the same factor until the two panels and three margins fit the width
        u_hud, u_cap = fs, 0.9 * fs
        for _ in range(6):
            w_hud = self._set_text(self.t_hud, hud_worst, u_hud)
            w_cap = self._set_text(self.t_cap, cap_worst, u_cap)
            widths = (w_hud[2] - w_hud[0]) + (w_cap[2] - w_cap[0])
            if widths + 3.0 * m <= W:
                break
            f = 0.98 * (W - 3.0 * m) / max(widths, 1.0)
            u_hud, u_cap = u_hud * f, u_cap * f
        self.u_hud, self.u_cap = u_hud, u_cap
        hud_box, cap_box = self._drop_to(self.t_hud, H - m), self._drop_to(self.t_cap, H - m)
        # 2. banner (bottom centre): sized by its text length, never wider than R3D_BANNER_MAX_WIDTH_FRAC of the window
        banner = plan.banner[0]
        self.u_banner = self._fit_width(self.t_banner, banner, fs * R3D_BANNER_FONT_SCALE, R3D_BANNER_MAX_WIDTH_FRAC * W)
        self._set_text(self.t_banner, banner, self.u_banner)
        banner_box = self._lift_to(self.t_banner, m)
        stack_top = banner_box[3]
        wm_box = None
        if self.t_wm is not None:
            self.u_wm = self._fit_width(self.t_wm, R3D_ILLUSTRATIVE_WATERMARK, fs * 1.1, R3D_BANNER_MAX_WIDTH_FRAC * W)
            self._set_text(self.t_wm, R3D_ILLUSTRATIVE_WATERMARK, self.u_wm)
            wm_box = self._lift_to(self.t_wm, banner_box[3] + 0.5 * m)
            stack_top = wm_box[3]
        # 3. colour bars with their titles in the right column below the HUD, stacked upwards from just above the bottom margin
        names = list(self.bars)                                                               # bottom to top: particle height, density
        names.sort(key=lambda n: 0 if n == "particles" else 1)
        title_h: dict[str, float] = {}
        for n in names:
            b = self._set_text(self.t_bars[n], "\n".join(_BAR_TITLES[n]), 0.8 * fs)
            title_h[n] = b[3] - b[1]
        y_bot = 0.07 * H
        gap, title_gap = 0.045 * H, 0.012 * H
        avail = (hud_box[1] - m) - y_bot - gap * max(len(names) - 1, 0) - sum(title_gap + title_h[n] for n in names)
        bar_h = max(0.05 * H, min(0.17 * H, avail / max(len(names), 1)))
        bar_x, bar_w = 0.90 * W, 0.035 * W
        self.bar_boxes: dict[str, tuple[float, float, float, float]] = {}
        y = y_bot
        for n in names:
            bar = self.bars[n]
            bar.SetPosition(bar_x / W, y / H)
            bar.SetWidth(bar_w / W)
            bar.SetHeight(bar_h / H)
            self.bar_boxes[n] = (bar_x, y, bar_x + bar_w, y + bar_h)
            self._lift_to(self.t_bars[n], y + bar_h + title_gap)
            y += bar_h + title_gap + title_h[n] + gap
        # 4. legend (bottom left) above the banner / watermark stack, below the caption
        entries = self._legend_entries()
        n_e = len(entries)
        y0 = stack_top + m
        entry_h = min(0.034, max(0.018, ((cap_box[1] - m - y0) / H - 0.02) / n_e))
        self.legend_size = (0.25, entry_h * n_e + 0.02)
        pl.add_legend(entries, bcolor=(1.0, 1.0, 1.0), border=True, size=self.legend_size, loc=None, face="rectangle")
        self.legend = pl.legend
        self.legend.SetPosition(m / W, y0 / H)
        self.legend.SetPosition2(*self.legend_size)
        self.layout = {"hud": hud_box, "caption": cap_box, "banner": banner_box, "legend": self.legend_box(),
                       **({"watermark": wm_box} if wm_box is not None else {}), **{f"bar_{n}": b for n, b in self.bar_boxes.items()}}
        for n in names:
            self.layout[f"bar_title_{n}"] = self._box(self.t_bars[n])
        for a in self._text_actors.values():
            a.input = " "

    def legend_box(self) -> tuple[float, float, float, float]:
        """Pixel box (x0, y0, x1, y1; origin bottom left) of the legend."""
        x, y = self.legend.GetPosition()
        w, h = self.legend.GetPosition2()
        return x * self.W, y * self.H, (x + w) * self.W, (y + h) * self.H

    def text_boxes(self) -> dict[str, tuple[float, float, float, float]]:
        """Pixel boxes (x0, y0, x1, y1; origin bottom left) of everything that carries text, as currently drawn: the visible text panels (blank ones are
        left out), the legend and the colour bars with their titles."""
        out = {n: self._box(a) for n, a in self._text_actors.items() if a.GetVisibility() and a.input.strip()}
        out["legend"] = self.legend_box()
        out.update({f"bar_{n}": b for n, b in self.bar_boxes.items()})
        return out

    # ---------------------------------------------------------------- per-frame updates
    def _update_density(self, frame: int) -> None:
        if self.dens_mesh is None or frame == self._dens_frame:
            return
        dens, _ = source_density(self.backend, self.log.source, frame, self.log.reflected)
        ld, _ = log_density_field(dens, self.vmax)
        self.dens_mesh.cell_data["log10 density"][:] = ld.ravel()
        self.dens_mesh.Modified()
        self._dens_frame = frame

    def _update_particles(self, frame: int) -> None:
        if not self.use_particles or frame == self._part_frame:
            return
        try:
            pts = self.backend.airborne_particles(frame, self.log.source)
        except (FileNotFoundError, OSError) as e:
            self.skipped["particles"] = f"frame {frame}: {e}"
            self.use_particles = False
            self.part_actor.SetVisibility(False)
            return
        pts = np.asarray(subsample_points(pts, self.max_particles, seed=frame * 1000 + self.log.source), dtype=np.float32)
        self._part_frame = frame
        self.particles_shown = max(self.particles_shown, int(pts.shape[0]))
        if pts.shape[0] == 0:
            self.part_actor.SetVisibility(False)
            return
        self.part_actor.SetVisibility(True)
        cloud = self.pv.PolyData(pts)
        cloud.point_data["height"] = pts[:, 2]                                                    # colour = altitude above ground
        self.part_mesh.copy_from(cloud)
        self.part_mesh.set_active_scalars("height")

    def draw(self, k: int, banner: bool) -> np.ndarray:
        """Update every actor for video frame ``k`` and return the (H, W, 3) uint8 image."""
        log, plan, cam, pl = self.log, self.plan, self.plan.camera, self.pl
        i = int(plan.steps[k])
        unit = float(marker_scale(cam.distance[k]))
        pl.camera.position = tuple(cam.position[k])
        pl.camera.focal_point = tuple(cam.focal[k])
        pl.camera.up = (0.0, 0.0, 1.0)
        pl.camera.view_angle = R3D_FOV_DEG
        pl.renderer.ResetCameraClippingRange()
        self._update_density(int(plan.frames[k]))
        self._update_particles(int(plan.frames[k]))
        for d in range(log.D):
            x, y = log.drone_xy[i, d]
            self.drone_prims[d].place((x, y, config.DRONE_Z), 1.1 * unit)
            tr = trail_geometry(log.drone_xy[:, d], log.y[:, d], i, z=config.DRONE_Z, threshold=self.det_thr,
                                base=R3D_TRAIL_BASE[d % len(R3D_TRAIL_BASE)])
            if tr is None:
                self.trail_actors[d].SetVisibility(False)
            else:
                self.trail_actors[d].SetVisibility(True)
                self.trail_meshes[d].copy_from(self._tube(tr["points"], tr["rgb"], tr["weight"], unit))
        self._update_belief(i, unit)
        self._update_truth(k, i, unit)
        self._update_text(k, i, banner)
        pl.render()
        return np.ascontiguousarray(pl.screenshot(return_img=True))

    def _update_belief(self, i: int, unit: float) -> None:
        log = self.log
        have = log.gmm_w is not None and log.gmm_mu is not None and log.gmm_cov is not None
        order = np.argsort(-log.gmm_w[i]) if have else []
        for j in range(R3D_TOP_COMPONENTS):
            on = bool(have and j < len(order) and (log.gmm_mask is None or log.gmm_mask[i, order[j]]))
            for a in (self.ell_fill_a[j], self.ell_line_a[j]):
                a.SetVisibility(on)
            if not on:
                continue
            c = int(order[j])
            ring = ellipse_points(log.gmm_mu[i, c], log.gmm_cov[i, c], z=config.DRONE_Z + 0.8)
            st = ellipse_style(float(log.gmm_w[i, c]))
            self.ell_fill[j].points = np.vstack([ring.mean(axis=0, keepdims=True), ring])
            self.ell_line[j].points = ring
            self.ell_fill_a[j].prop.opacity = st["fill_opacity"]
            self.ell_line_a[j].prop.opacity = st["line_opacity"]
            self.ell_line_a[j].prop.line_width = st["line_width"] * self.scale
        if log.map_xy is None:
            self.map_actor.SetVisibility(False)
        else:
            self.map_prim.place((log.map_xy[i, 0], log.map_xy[i, 1], config.DRONE_Z + 1.0), 1.6 * unit)

    def _update_truth(self, k: int, i: int, unit: float) -> None:
        tx, ty = self.log.truth_xy
        done = self.plan.loc_step is not None and i >= self.plan.loc_step
        height = float(R3D_COLUMN_HEIGHT_M)
        self.base_prim.place((tx, ty, config.SOURCE_Z), 0.9 * unit)
        self.ring_prim.place((tx, ty, config.SOURCE_Z), float(R3D_RELEASE_RING_UNITS) * unit)    # the ring marks the real release height
        r = (0.45 if done else 0.22) * unit
        self.col_prim.place((tx, ty, 0.0), (r, r, height))                                    # unit cylinder (radius 1, z 0..1) scaled anisotropically
        self.col_actor.prop.opacity = 0.9 if done else 0.5
        self.halo_actor.SetVisibility(bool(done))
        if done:
            self.halo_prim.place((tx, ty, 0.0), (2.2 * unit, 2.2 * unit, height))
        sp = star_points((tx, ty, height + 2.2 * unit), 1.9 * unit * (1.3 if done else 1.0), float(self.plan.camera.azimuth_deg[k]))
        self.star.points = sp
        self.star_edge.points = sp.copy()

    def _update_text(self, k: int, i: int, banner: bool) -> None:
        plan = self.plan
        self.t_hud.input = self._hud_text(k, i)
        self.t_cap.input = self._caption_text(k, i)
        for n, a in self.t_bars.items():
            a.input = "\n".join(_BAR_TITLES[n])
        if self.t_wm is not None:
            self.t_wm.input = R3D_ILLUSTRATIVE_WATERMARK
            self.t_wm.prop.color = (0.70, 0.08, 0.08)
        if banner:
            text, kind = plan.banner
            self.t_banner.input = text
            self.t_banner.prop.color = (0.05, 0.40, 0.12) if kind == "ok" else (0.70, 0.08, 0.08)
            self.t_banner.prop.background_opacity = 0.80
        else:
            self.t_banner.input = " "
            self.t_banner.prop.background_opacity = 0.0

    def close(self) -> None:
        try:
            self.pl.close()
        except Exception:                                                                    # noqa: BLE001 - closing a broken GL context must not mask the real error
            pass


# ------------------------------------------------------------------------------------------------------ public API
def resolve_options(preset: str | None = None, **explicit: Any) -> dict:
    """Render options: the quick-preview defaults, overridden by the named preset (R3D_PRESETS, e.g. 'final' = spec 11.5), overridden by every explicit
    option that is not None.  Keys: width, height, fps, stride, steps_per_second."""
    opts: dict[str, Any] = {"width": R3D_WINDOW[0], "height": R3D_WINDOW[1], "fps": R3D_FPS, "stride": R3D_STRIDE, "steps_per_second": None}
    if preset is not None:
        if preset not in R3D_PRESETS:
            raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(R3D_PRESETS)}")
        opts.update(R3D_PRESETS[preset])
    opts.update({k: v for k, v in explicit.items() if v is not None})
    return opts


def render_episode(npz: Path | str, out_mp4: Path | str, *, width: int = R3D_WINDOW[0], height: int = R3D_WINDOW[1], fps: int = R3D_FPS,
                   stride: int = R3D_STRIDE, max_frames: int | None = None, particles: bool = True, poster: Path | str | None = None,
                   mode: str | None = None, backend=None, buildings: Any = "auto", max_particles: int = R3D_MAX_PARTICLES,
                   hold_seconds: float = R3D_HOLD_SECONDS, crf: int = R3D_CRF, steps_per_second: float | None = None,
                   stop_at_localisation: bool = False, stop_after_seconds: float | None = None, illustrative: bool = False,
                   progress: Callable[[str], None] | None = None) -> dict:
    """Render one step log to an H.264 MP4 (+ poster PNG of the last frame) and return the manifest dict.

    npz: step log; out_mp4: output video; stride / max_frames: video frames = every stride-th log step (at most max_frames); particles: draw the particle
    subsample; poster: PNG path (default <out stem>_poster.png); mode: F | T2 override of the truth-mode inference; backend: slab backend
    (default: the memory-mapped stack); buildings: 'auto' (the STL), a pyvista mesh, or None (skip); steps_per_second: playback speed in log steps per second
    (None: fps x stride; each picture is then written round(fps x stride / steps_per_second) times, spec 11.5 = 5 steps/s at 30 fps); stop_at_localisation:
    end the video stop_after_seconds (default R3D_STOP_AFTER_SECONDS) after the first localisation; illustrative: the <= 10 s clip of spec 11.3 option 3
    (field cycles over R3D_ILLUSTRATIVE_FRAMES, watermark, not the truth the policy saw).
    Manifest: status ('ok' | 'skipped'), reason (when skipped), path, poster, frames (encoded frames written, copies and hold included), seconds (wall
    time), step_frames (distinct pictures), repeat, hold_frames, size, fps, steps_per_second (effective), video_seconds, mode, mode_source, n_steps,
    steps_covered (< n_steps when the video was stopped at the localisation), n_drones, source, method, localised_step, localised_error_m, field_frames
    (first, last, distinct), particles_shown_max, illustrative, stop_at_localisation, skipped (component -> reason), notes, banner, banner_from_step."""
    t_start = time.perf_counter()
    npz, out_mp4 = Path(npz), Path(out_mp4)
    poster = Path(poster) if poster is not None else out_mp4.with_name(out_mp4.stem + "_poster.png")
    manifest: dict[str, Any] = {"status": "ok", "path": str(out_mp4), "poster": str(poster), "frames": 0, "seconds": 0.0, "episode": str(npz)}
    if not npz.is_file():
        manifest.update(status="skipped", reason=f"step log not found: {npz}", path=None, poster=None)
        return manifest
    if width % 2 or height % 2:
        raise ValueError("width and height must be even (H.264 yuv420p)")
    try:
        import imageio.v2 as iio
        import imageio_ffmpeg  # noqa: F401 - the FFMPEG writer needs its binary
        import pyvista  # noqa: F401
    except ImportError as e:                                                              # missing optional dependency: skip the video, say why
        manifest.update(status="skipped", reason=f"rendering dependency not installed: {e}", path=None, poster=None)
        return manifest
    log = load_episode_log(npz, mode)
    plan = plan_render(log, stride=stride, max_frames=max_frames, fps=fps, hold_seconds=hold_seconds, aspect=width / height, steps_per_second=steps_per_second,
                       stop_at_localisation=stop_at_localisation, stop_after_seconds=stop_after_seconds, illustrative=illustrative)
    skipped: dict[str, str] = {}
    if backend is None:
        try:
            backend = default_backend()
        except (FileNotFoundError, OSError, ValueError, KeyError) as e:
            skipped["density"] = f"slab backend unavailable: {e}"
            skipped["particles"] = "slab backend unavailable"
            backend = None
    if isinstance(buildings, str) and buildings == "auto":
        try:
            buildings = load_buildings(mirror_y=log.reflected)
        except (FileNotFoundError, OSError) as e:
            skipped["buildings"] = f"STL not readable: {e}"
            buildings = None
    notes = []
    if log.frame_derived:
        notes.append(f"truth frame per step derived (mode {log.mode} from {log.mode_source}); the log carries no 'frame' array")
    if log.mode_source == "default":
        notes.append("truth mode defaulted to F (no mode in the log meta, no episodes.csv, no frame array); pass --mode to override")
    if illustrative:
        notes.append(f"ILLUSTRATIVE clip: field cycles over frames {R3D_ILLUSTRATIVE_FRAMES[0]}..{R3D_ILLUSTRATIVE_FRAMES[1]}, not the truth seen by the policy")
    if plan.n_steps_used < log.T:
        notes.append(f"video stopped {plan.n_steps_used - 1 - (plan.loc_step or 0)} steps after the localisation at step {(plan.loc_step or 0) + 1} "
                     f"(episode has {log.T} steps)")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)
    scene = _Scene(log, plan, width, height, backend, buildings, particles, max_particles, skipped)
    writer = iio.get_writer(str(out_mp4), format="FFMPEG", fps=int(fps), codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=2,
                            ffmpeg_params=["-crf", str(int(crf)), "-preset", "medium", "-movflags", "+faststart"])
    n_written = 0
    last = None
    try:
        for k in range(plan.steps.size):
            last = scene.draw(k, banner=k >= plan.banner_from)
            for _ in range(plan.repeat):
                writer.append_data(last)
                n_written += 1
            if progress is not None and (k % 20 == 0 or k == plan.steps.size - 1):
                progress(f"[render_episode_3d] frame {k + 1}/{plan.steps.size} (step {int(plan.steps[k]) + 1}/{log.T})")
        for _ in range(plan.hold_frames):
            writer.append_data(last)
            n_written += 1
        iio.imwrite(str(poster), last)                                                    # before closing the GL scene: the outputs exist even if the close misbehaves
    finally:
        writer.close()
        scene.close()
    advance = float(plan.steps[-1] - plan.steps[0]) / max(plan.steps.size - 1, 1) if plan.steps.size > 1 else float(stride)
    manifest.update(frames=n_written, step_frames=int(plan.steps.size), repeat=int(plan.repeat), hold_frames=int(plan.hold_frames),
                    size=[int(width), int(height)], fps=int(fps), steps_per_second=advance * int(fps) / plan.repeat,
                    mode=log.mode, mode_source=log.mode_source, n_steps=log.T, steps_covered=int(plan.n_steps_used), n_drones=log.D, source=log.source,
                    method=log.method, localised_step=None if plan.loc_step is None else plan.loc_step + 1, localised_error_m=plan.loc_error,
                    field_frames={"first": int(plan.frames[0]), "last": int(plan.frames[-1]), "distinct": len(plan.field_frames)},
                    particles_shown_max=int(scene.particles_shown), illustrative=bool(illustrative), stop_at_localisation=bool(stop_at_localisation),
                    skipped=skipped, notes=notes, banner=plan.banner[0], banner_from_step=int(plan.steps[plan.banner_from]) + 1,
                    video_seconds=n_written / float(fps), seconds=time.perf_counter() - t_start)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog="Exit code 0: video written; 2: skipped or unusable input (reason on stderr).")
    ap.add_argument("--episode", type=Path, required=True, help="step-log NPZ written by eval/run_eval.py --log-steps")
    ap.add_argument("--out", type=Path, required=True, help="output MP4")
    ap.add_argument("--preset", choices=sorted(R3D_PRESETS), default=None,
                    help="named option set; 'final' = spec 11.5 (1920 x 1080, 5 steps per second, encoded at 30 fps); explicit options override it "
                         f"(default without preset: {R3D_WINDOW[0]} x {R3D_WINDOW[1]}, {R3D_FPS} fps, stride {R3D_STRIDE})")
    ap.add_argument("--width", type=int, default=None, help=f"video width (default {R3D_WINDOW[0]})")
    ap.add_argument("--height", type=int, default=None, help=f"video height (default {R3D_WINDOW[1]})")
    ap.add_argument("--fps", type=int, default=None, help=f"encoding frames per second (default {R3D_FPS})")
    ap.add_argument("--stride", type=int, default=None, help=f"log steps per picture (default {R3D_STRIDE})")
    ap.add_argument("--steps-per-second", type=float, default=None,
                    help="playback speed in log steps per second (each picture is repeated to reach it; default: fps x stride)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap on the number of pictures (evenly thinned)")
    ap.add_argument("--no-particles", action="store_true", help="do not draw the airborne particle cloud")
    ap.add_argument("--poster", type=Path, default=None, help="poster PNG (default: <out>_poster.png)")
    ap.add_argument("--mode", "--frame-mode", dest="mode", choices=config.ENV_MODES, default=None,
                    help="override the truth-mode inference (F frozen / T2 time-varying); --frame-mode is the spec 11.6 name of the same option")
    ap.add_argument("--max-particles", type=int, default=R3D_MAX_PARTICLES)
    ap.add_argument("--stop-at-localisation", action="store_true",
                    help=f"end the video {R3D_STOP_AFTER_SECONDS:g} s (--stop-after-seconds) after the first localisation, plus the {R3D_HOLD_SECONDS:g} s hold of the last picture "
                         "(the banner shows from the localisation picture on)")
    ap.add_argument("--stop-after-seconds", type=float, default=None, help="video seconds of motion kept after the localisation with --stop-at-localisation (the hold comes on top)")
    ap.add_argument("--illustrative", action="store_true",
                    help=f"<= {R3D_ILLUSTRATIVE_SECONDS:g} s clip (spec 11.3 option 3): the field cycles over frames {R3D_ILLUSTRATIVE_FRAMES[0]}..{R3D_ILLUSTRATIVE_FRAMES[1]}; "
                         f"watermark '{R3D_ILLUSTRATIVE_WATERMARK}'")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Command line entry point: prints the manifest as JSON on stdout, progress and problems on stderr and returns the exit code (0 = video written,
    2 = skipped, e.g. a missing episode file, or an unusable step log; never a traceback for those)."""
    args = build_parser().parse_args(argv)
    try:
        opts = resolve_options(args.preset, width=args.width, height=args.height, fps=args.fps, stride=args.stride, steps_per_second=args.steps_per_second)
        res = render_episode(args.episode, args.out, width=opts["width"], height=opts["height"], fps=opts["fps"], stride=opts["stride"],
                             steps_per_second=opts["steps_per_second"], max_frames=args.max_frames, particles=not args.no_particles, poster=args.poster,
                             mode=args.mode, max_particles=args.max_particles, stop_at_localisation=args.stop_at_localisation,
                             stop_after_seconds=args.stop_after_seconds, illustrative=args.illustrative,
                             progress=lambda s: print(s, file=sys.stderr, flush=True))
    except (KeyError, ValueError, OSError) as e:                                          # an unusable step log / option: say so, no traceback
        print(f"render_episode_3d: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2, default=str))
    if res.get("status") != "ok":
        print(f"render_episode_3d: skipped: {res.get('reason')}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
