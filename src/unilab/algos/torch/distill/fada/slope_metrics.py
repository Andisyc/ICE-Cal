"""Ramp-coordinate trajectory metrics for FADA slope evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.algos.torch.distill.fada.target_domain import FADASlopeGeometry


@dataclass(frozen=True)
class FADASlopeTrajectory:
    base_pos_w: np.ndarray
    base_yaw_rad: np.ndarray
    feet_pos_w: np.ndarray
    forward_velocity_mps: np.ndarray
    command_forward_mps: np.ndarray
    physics_states: tuple[np.ndarray, ...]
    terminal_reason: str
    control_dt_s: float

    def validate(self) -> FADASlopeTrajectory:
        base = np.asarray(self.base_pos_w)
        yaw = np.asarray(self.base_yaw_rad)
        feet = np.asarray(self.feet_pos_w)
        velocity = np.asarray(self.forward_velocity_mps)
        command = np.asarray(self.command_forward_mps)
        if base.ndim != 2 or base.shape[1] != 3:
            raise ValueError(f"FADA slope base_pos_w must have shape (T, 3), got {base.shape}")
        count = base.shape[0]
        if count <= 0:
            raise ValueError("FADA slope trajectory must contain at least one step")
        expected = {
            "base_yaw_rad": (count,),
            "feet_pos_w": (count, 2, 3),
            "forward_velocity_mps": (count,),
            "command_forward_mps": (count,),
        }
        observed = {
            "base_yaw_rad": yaw.shape,
            "feet_pos_w": feet.shape,
            "forward_velocity_mps": velocity.shape,
            "command_forward_mps": command.shape,
        }
        for name, shape in expected.items():
            if observed[name] != shape:
                raise ValueError(
                    f"FADA slope {name} shape mismatch: expected={shape} observed={observed[name]}"
                )
        arrays = (base, yaw, feet, velocity, command)
        if not all(bool(np.all(np.isfinite(value))) for value in arrays):
            raise ValueError("FADA slope trajectory arrays must contain only finite values")
        if not np.isfinite(self.control_dt_s) or self.control_dt_s <= 0.0:
            raise ValueError("FADA slope control_dt_s must be finite and positive")
        if self.terminal_reason not in {
            "horizon",
            "fall",
            "environment_termination",
            "truncated",
            "foot_exit",
            "finish",
        }:
            raise ValueError(f"unsupported FADA slope terminal reason: {self.terminal_reason}")
        return self


def _error_statistics(values: np.ndarray, *, prefix: str) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        f"final_{prefix}": float(values[-1]),
        f"final_abs_{prefix}": float(absolute[-1]),
        f"max_abs_{prefix}": float(np.max(absolute)),
        f"mean_abs_{prefix}": float(np.mean(absolute)),
        f"rms_{prefix}": float(np.sqrt(np.mean(np.square(values)))),
    }


def summarize_slope_trajectory(
    trajectory: FADASlopeTrajectory,
    geometry: FADASlopeGeometry,
) -> dict[str, Any]:
    trajectory.validate()
    base_surface = geometry.surface_coordinates(trajectory.base_pos_w)
    lateral = base_surface[:, 1]
    yaw_error = np.arctan2(
        np.sin(trajectory.base_yaw_rad),
        np.cos(trajectory.base_yaw_rad),
    )
    velocity_error = np.asarray(trajectory.forward_velocity_mps) - np.asarray(
        trajectory.command_forward_mps
    )
    result: dict[str, Any] = {}
    result.update(_error_statistics(lateral, prefix="lateral_m"))
    result.update(_error_statistics(yaw_error, prefix="yaw_error_rad"))
    result.update(
        {
            "uphill_progress_m": float(base_surface[-1, 0] - base_surface[0, 0]),
            "forward_velocity_mae_mps": float(np.mean(np.abs(velocity_error))),
            "forward_velocity_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_error)))),
            "steps": int(base_surface.shape[0]),
            "time_before_terminal_s": float(base_surface.shape[0] * trajectory.control_dt_s),
            "terminal_reason": trajectory.terminal_reason,
            "environment_terminated": trajectory.terminal_reason == "environment_termination",
            "fell": trajectory.terminal_reason == "fall",
            "truncated": trajectory.terminal_reason == "truncated",
            "finished": trajectory.terminal_reason == "finish",
            "foot_exit": trajectory.terminal_reason == "foot_exit",
            "lateral_m": lateral.tolist(),
            "yaw_error_rad": yaw_error.tolist(),
        }
    )
    return result


_ERROR_METRICS = (
    "final_abs_lateral_m",
    "max_abs_lateral_m",
    "mean_abs_lateral_m",
    "rms_lateral_m",
    "final_abs_yaw_error_rad",
    "max_abs_yaw_error_rad",
    "mean_abs_yaw_error_rad",
    "rms_yaw_error_rad",
    "forward_velocity_mae_mps",
    "forward_velocity_rmse_mps",
)


def compare_slope_summaries(
    zero_shot: dict[str, Any],
    adapted: dict[str, Any],
) -> dict[str, float]:
    """Positive values always mean that adaptation improved the scalar metric."""

    comparison = {name: float(zero_shot[name]) - float(adapted[name]) for name in _ERROR_METRICS}
    comparison["uphill_progress_m"] = float(adapted["uphill_progress_m"]) - float(
        zero_shot["uphill_progress_m"]
    )
    comparison["time_before_terminal_s"] = float(adapted["time_before_terminal_s"]) - float(
        zero_shot["time_before_terminal_s"]
    )
    return comparison


def compact_slope_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Drop per-frame series while preserving scalar and terminal evidence."""

    return {
        key: value for key, value in summary.items() if key not in {"lateral_m", "yaw_error_rad"}
    }


def _sample_statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise ValueError("FADA evaluation aggregation requires finite scalar values")
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


def aggregate_policy_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated policy trials without merging terminal semantics into scalars."""

    if not summaries:
        raise ValueError("FADA evaluation requires at least one policy summary")
    metric_names = sorted(
        key
        for key, value in summaries[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    if any(
        not isinstance(summary.get(name), (int, float)) or isinstance(summary.get(name), bool)
        for summary in summaries
        for name in metric_names
    ):
        raise ValueError("FADA evaluation policy summaries have inconsistent numeric fields")
    terminal_counts: dict[str, int] = {}
    for summary in summaries:
        reason = summary.get("terminal_reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("FADA evaluation policy summary requires terminal_reason")
        terminal_counts[reason] = terminal_counts.get(reason, 0) + 1
    return {
        "metrics": aggregate_numeric_summaries(summaries, metric_names=metric_names),
        "terminal_counts": dict(sorted(terminal_counts.items())),
    }


def aggregate_numeric_summaries(
    summaries: list[dict[str, Any]],
    *,
    metric_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate a consistent set of numeric diagnostic records."""

    if not summaries:
        raise ValueError("FADA evaluation requires at least one numeric summary")
    names = (
        sorted(
            key
            for key, value in summaries[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        if metric_names is None
        else metric_names
    )
    if any(
        not isinstance(summary.get(name), (int, float)) or isinstance(summary.get(name), bool)
        for summary in summaries
        for name in names
    ):
        raise ValueError("FADA evaluation numeric summaries have inconsistent fields")
    return {
        name: _sample_statistics([float(summary[name]) for summary in summaries]) for name in names
    }


def aggregate_improvement_summaries(
    comparisons: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate paired deltas; positive values and wins mean adapted is better."""

    if not comparisons:
        raise ValueError("FADA evaluation requires at least one paired comparison")
    names = sorted(comparisons[0])
    if any(set(comparison) != set(names) for comparison in comparisons):
        raise ValueError("FADA evaluation comparisons have inconsistent fields")
    result: dict[str, dict[str, float]] = {}
    for name in names:
        values = [float(comparison[name]) for comparison in comparisons]
        stats = _sample_statistics(values)
        stats["paired_win_fraction"] = float(np.mean(np.asarray(values) > 0.0))
        result[name] = stats
    return result
