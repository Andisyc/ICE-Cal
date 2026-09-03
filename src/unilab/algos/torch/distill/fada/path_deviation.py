from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _position_trace(value: np.ndarray, *, name: str) -> np.ndarray:
    trace = np.asarray(value, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[0] == 0 or trace.shape[1] != 2:
        raise ValueError(f"{name} position trace must have shape (steps, 2), got {trace.shape}")
    if not bool(np.all(np.isfinite(trace))):
        raise ValueError(f"{name} position trace must contain only finite values")
    return trace


def _yaw_trace(value: np.ndarray, *, name: str) -> np.ndarray:
    trace = np.asarray(value, dtype=np.float64)
    if trace.ndim != 1 or trace.shape[0] == 0:
        raise ValueError(f"{name} yaw trace must have shape (steps,), got {trace.shape}")
    if not bool(np.all(np.isfinite(trace))):
        raise ValueError(f"{name} yaw trace must contain only finite values")
    return trace


def _branch_summary(
    position_xy_m: np.ndarray,
    yaw_rad: np.ndarray,
    *,
    origin_xy_m: np.ndarray,
    heading_rad: float,
) -> dict[str, Any]:
    delta = position_xy_m - origin_xy_m[None, :]
    lateral = -np.sin(heading_rad) * delta[:, 0] + np.cos(heading_rad) * delta[:, 1]
    absolute = np.abs(lateral)
    yaw_drift = np.arctan2(np.sin(yaw_rad - heading_rad), np.cos(yaw_rad - heading_rad))
    absolute_yaw_drift = np.abs(yaw_drift)
    return {
        "position_xy_m": position_xy_m.tolist(),
        "lateral_m": lateral.tolist(),
        "rms_lateral_m": float(np.sqrt(np.mean(np.square(lateral)))),
        "mean_abs_lateral_m": float(np.mean(absolute)),
        "max_abs_lateral_m": float(np.max(absolute)),
        "final_lateral_m": float(lateral[-1]),
        "yaw_rad": yaw_rad.tolist(),
        "yaw_drift_rad": yaw_drift.tolist(),
        "rms_yaw_drift_rad": float(np.sqrt(np.mean(np.square(yaw_drift)))),
        "mean_abs_yaw_drift_rad": float(np.mean(absolute_yaw_drift)),
        "max_abs_yaw_drift_rad": float(np.max(absolute_yaw_drift)),
        "final_yaw_drift_rad": float(yaw_drift[-1]),
    }


def build_straight_line_deviation_report(
    *,
    nominal_xy_m: np.ndarray,
    faulty_xy_m: np.ndarray,
    nominal_yaw_rad: np.ndarray,
    faulty_yaw_rad: np.ndarray,
    origin_xy_m: np.ndarray,
    heading_rad: float,
    measurement_start_step: int = 0,
) -> Mapping[str, Any]:
    """Compare paired base paths with one start-pose straight-line reference."""

    nominal = _position_trace(nominal_xy_m, name="nominal")
    faulty = _position_trace(faulty_xy_m, name="faulty")
    nominal_yaw = _yaw_trace(nominal_yaw_rad, name="nominal")
    faulty_yaw = _yaw_trace(faulty_yaw_rad, name="faulty")
    if nominal.shape[0] != nominal_yaw.shape[0]:
        raise ValueError("nominal position/yaw frame count mismatch")
    if faulty.shape[0] != faulty_yaw.shape[0]:
        raise ValueError("faulty position/yaw frame count mismatch")
    origin = np.asarray(origin_xy_m, dtype=np.float64)
    heading = float(heading_rad)
    if origin.shape != (2,) or not bool(np.all(np.isfinite(origin))):
        raise ValueError(f"straight-line origin must be finite shape (2,), got {origin.shape}")
    if not bool(np.isfinite(heading)):
        raise ValueError("straight-line heading must be finite")
    if measurement_start_step < 0:
        raise ValueError("measurement_start_step must be non-negative")

    num_samples = min(
        int(nominal.shape[0]),
        int(faulty.shape[0]),
    )
    nominal_summary = _branch_summary(
        nominal[:num_samples], nominal_yaw[:num_samples], origin_xy_m=origin, heading_rad=heading
    )
    faulty_summary = _branch_summary(
        faulty[:num_samples], faulty_yaw[:num_samples], origin_xy_m=origin, heading_rad=heading
    )
    return {
        "schema_version": "fada-path-deviation/v2",
        "reference_line": {
            "origin_xy_m": origin.tolist(),
            "heading_rad": heading,
            "num_samples": num_samples,
            "measurement_start_step": int(measurement_start_step),
        },
        "nominal": nominal_summary,
        "faulty": faulty_summary,
        "excess": {
            "rms_lateral_m": (
                faulty_summary["rms_lateral_m"] - nominal_summary["rms_lateral_m"]
            ),
            "mean_abs_lateral_m": (
                faulty_summary["mean_abs_lateral_m"]
                - nominal_summary["mean_abs_lateral_m"]
            ),
            "max_abs_lateral_m": (
                faulty_summary["max_abs_lateral_m"] - nominal_summary["max_abs_lateral_m"]
            ),
            "final_abs_lateral_m": (
                abs(faulty_summary["final_lateral_m"])
                - abs(nominal_summary["final_lateral_m"])
            ),
            "rms_yaw_drift_rad": (
                faulty_summary["rms_yaw_drift_rad"]
                - nominal_summary["rms_yaw_drift_rad"]
            ),
            "mean_abs_yaw_drift_rad": (
                faulty_summary["mean_abs_yaw_drift_rad"]
                - nominal_summary["mean_abs_yaw_drift_rad"]
            ),
            "max_abs_yaw_drift_rad": (
                faulty_summary["max_abs_yaw_drift_rad"]
                - nominal_summary["max_abs_yaw_drift_rad"]
            ),
            "final_abs_yaw_drift_rad": (
                abs(faulty_summary["final_yaw_drift_rad"])
                - abs(nominal_summary["final_yaw_drift_rad"])
            ),
        },
    }
