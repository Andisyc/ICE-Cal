from __future__ import annotations

import numpy as np
import pytest

from unilab.algos.torch.distill.fada.slope_metrics import (
    FADASlopeTrajectory,
    aggregate_improvement_summaries,
    aggregate_policy_summaries,
    compact_slope_summary,
    compare_slope_summaries,
    summarize_slope_trajectory,
)
from unilab.algos.torch.distill.fada.target_domain import FADASlopeGeometry


def _trajectory(lateral: list[float], yaw: list[float], *, terminal: str) -> FADASlopeTrajectory:
    count = len(lateral)
    return FADASlopeTrajectory(
        base_pos_w=np.column_stack(
            (np.linspace(1.5, 5.0, count), np.asarray(lateral), np.linspace(0.0, 0.9, count))
        ),
        base_yaw_rad=np.asarray(yaw),
        feet_pos_w=np.zeros((count, 2, 3)),
        forward_velocity_mps=np.full(count, 0.7),
        command_forward_mps=np.full(count, 0.8),
        physics_states=(),
        terminal_reason=terminal,
        control_dt_s=0.02,
    )


def test_slope_summary_preserves_signed_drift_and_reports_errors() -> None:
    geometry = FADASlopeGeometry(15.0, 0.8, 1.5, 8.0, 0.25, 0.5)
    result = summarize_slope_trajectory(
        _trajectory([0.0, -0.1, -0.2], [0.0, 0.1, 0.2], terminal="foot_exit"),
        geometry,
    )

    assert result["final_lateral_m"] == -0.2
    assert result["final_abs_lateral_m"] == 0.2
    assert result["rms_lateral_m"] > result["mean_abs_lateral_m"]
    assert result["final_yaw_error_rad"] == 0.2
    assert result["forward_velocity_mae_mps"] == pytest.approx(0.1)
    assert result["foot_exit"] is True
    assert result["environment_terminated"] is False
    assert result["time_before_terminal_s"] == pytest.approx(0.06)
    assert result["uphill_progress_m"] > 0.0
    assert result["lateral_m"] == [0.0, -0.1, -0.2]


def test_slope_comparison_uses_consistent_improvement_signs() -> None:
    geometry = FADASlopeGeometry(15.0, 0.8, 1.5, 8.0, 0.25, 0.5)
    zero = summarize_slope_trajectory(
        _trajectory(
            [0.0, 0.3, 0.6],
            [0.0, 0.2, 0.4],
            terminal="environment_termination",
        ),
        geometry,
    )
    adapted = summarize_slope_trajectory(
        _trajectory([0.0, 0.1, 0.2], [0.0, 0.05, 0.1], terminal="finish"),
        geometry,
    )
    comparison = compare_slope_summaries(zero, adapted)

    assert comparison["final_abs_lateral_m"] > 0.0
    assert comparison["rms_yaw_error_rad"] > 0.0
    assert comparison["uphill_progress_m"] == (
        adapted["uphill_progress_m"] - zero["uphill_progress_m"]
    )


def test_multi_trial_aggregation_reports_dispersion_terminal_counts_and_wins() -> None:
    summaries = [
        {"forward_velocity_mae_mps": 1.0, "steps": 5, "terminal_reason": "fall"},
        {"forward_velocity_mae_mps": 3.0, "steps": 7, "terminal_reason": "horizon"},
    ]
    policy = aggregate_policy_summaries(summaries)
    improvement = aggregate_improvement_summaries(
        [{"forward_velocity_mae_mps": 2.0}, {"forward_velocity_mae_mps": -1.0}]
    )

    assert policy["metrics"]["forward_velocity_mae_mps"] == {
        "mean": 2.0,
        "sample_std": pytest.approx(2.0**0.5),
    }
    assert policy["terminal_counts"] == {"fall": 1, "horizon": 1}
    assert improvement["forward_velocity_mae_mps"] == {
        "mean": 0.5,
        "sample_std": pytest.approx(3.0 / 2.0**0.5),
        "paired_win_fraction": 0.5,
    }


def test_compact_slope_summary_removes_per_frame_series() -> None:
    geometry = FADASlopeGeometry(15.0, 0.8, 1.5, 8.0, 0.25, 0.5)
    summary = summarize_slope_trajectory(
        _trajectory([0.0, 0.1], [0.0, 0.2], terminal="horizon"), geometry
    )

    compact = compact_slope_summary(summary)

    assert "lateral_m" not in compact
    assert "yaw_error_rad" not in compact
    assert compact["final_lateral_m"] == summary["final_lateral_m"]
