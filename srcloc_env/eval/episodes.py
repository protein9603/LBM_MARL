"""Common evaluation episode lists (plan 4.7 / 5장: 모든 방법이 같은 에피소드 시드 목록을 사용; D9-2).

An ``EpisodeSpec`` fixes everything the environment would otherwise draw: source, truth frame (Mode F) or T2 start,
sensor scale and the reset seed (which fixes the drone starts: drone 0 is drawn identically by SourceLocEnv and
MultiDroneEnv, so 1- and n-drone runs of the same spec start drone 0 at the same place).  Reflection is always off in
evaluation (plan 4.5).  ``make_episode_list`` draws n_per_source specs per source from one base seed; the list is
saved as CSV next to the results so that every method / drone count / seed is evaluated on exactly the same episodes
(paired comparison, plan 5장 지표).
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from srcloc_env import config
from srcloc_env.sensor.detector import Detector


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    seed: int
    source: int
    frame: int            # Mode F truth frame; Mode T2: start frame
    scale: float
    mode: str = "F"
    reflect: bool = False

    def reset_options(self) -> dict:
        key = "frame" if self.mode == "F" else "t2_start"
        return {"source": self.source, key: self.frame, "scale": self.scale, "reflect": self.reflect}


def make_episode_list(sources: Sequence[int] = config.ALL_SOURCES, n_per_source: int = config.EVAL_EPISODES_PER_SOURCE,
                      base_seed: int = config.EVAL_BASE_SEED, mode: str = "F",
                      frame_range: tuple[int, int] = config.FRAME_RANGE_MODE_F,
                      t2_start_range: tuple[int, int] = config.FRAME_START_MODE_T2,
                      scale_range: tuple[float, float] = config.SENSOR_SCALE_RANGE,
                      scale_fixed: float | None = None) -> list[EpisodeSpec]:
    """n_per_source specs per source, source-major order, seeds = base_seed + 1000 * k (k = running index)."""
    if mode not in config.ENV_MODES:
        raise ValueError(f"mode must be one of {config.ENV_MODES}")
    rng = np.random.default_rng(base_seed)
    lo, hi = (frame_range if mode == "F" else t2_start_range)
    out: list[EpisodeSpec] = []
    k = 0
    for s in sources:
        for _ in range(int(n_per_source)):
            frame = int(rng.integers(lo, hi + 1))
            scale = float(scale_fixed) if scale_fixed is not None else float(Detector.sample_scale(rng, scale_range[0], scale_range[1]))
            out.append(EpisodeSpec(episode_id=k, seed=int(base_seed + 1000 * k), source=int(s), frame=frame, scale=scale,
                                   mode=mode, reflect=False))
            k += 1
    return out


def save_episode_list(specs: Iterable[EpisodeSpec], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(EpisodeSpec)])
        w.writeheader()
        for sp in specs:
            w.writerow(asdict(sp))


def load_episode_list(path: Path) -> list[EpisodeSpec]:
    out = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(EpisodeSpec(episode_id=int(row["episode_id"]), seed=int(row["seed"]), source=int(row["source"]),
                                   frame=int(row["frame"]), scale=float(row["scale"]), mode=row["mode"],
                                   reflect=row["reflect"].strip().lower() == "true"))
    return out
