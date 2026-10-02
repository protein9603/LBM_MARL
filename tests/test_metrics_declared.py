"""Declared-success statistics of eval/metrics.aggregate (spec 9.7): non-declaring episodes are failures; overconfidence and delay."""
from __future__ import annotations

import pytest

from srcloc_env.eval.metrics import aggregate


def _rec(i, declared_step, declared_error, success, steps):
    return {"episode_id": i, "source": 101, "success": success, "steps": steps, "final_error_m": 10.0, "first_detection_step": 5,
            "declared_step": declared_step, "declared_error_m": declared_error, "path_length_m": 100.0, "n_masked": 0}


def test_declared_success_rate_counts_non_declaring_episodes_as_failures_and_reports_overconfidence_and_delay():
    recs = [_rec(0, 40, 5.0, True, 30),        # declared at 40, correct (error 5 m < 50 m), actual success at step 30 -> delay 10
            _rec(1, 60, 35.0, True, 50),       # declared at 60, still 35 m off: correct for the 50 m criterion but overconfident (>= 20 m), delay 10
            _rec(2, 70, 80.0, False, 150),     # declared but wrong (80 m)
            _rec(3, None, None, False, 150)]   # never declared: failure
    a = aggregate(recs)["train_open"]
    assert a["n"] == 4 and a["declared_n"] == 3
    assert a["declared_success_rate"] == pytest.approx(2 / 4)                  # denominator = all episodes
    assert a["declared_conditional_rate"] == pytest.approx(2 / 3)              # among the declaring episodes only
    assert a["declared_overconfident_rate"] == pytest.approx(2 / 3)            # 35 m and 80 m are >= 20 m
    assert a["declaration_delay_median"] == pytest.approx(10.0)
