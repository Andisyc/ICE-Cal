from __future__ import annotations

import numpy as np
import pytest

from unilab.algos.torch.distill.fada.slope_metrics import (
    FADASlopeTrajectory,
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
