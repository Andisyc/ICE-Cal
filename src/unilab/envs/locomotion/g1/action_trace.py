"""Read-only formatting for opt-in G1 action execution diagnostics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def action_trace_enabled() -> bool:
    value = os.environ.get("UNILAB_G1_ACTION_TRACE")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def action_trace_interval() -> int:
    try:
        return max(1, int(os.environ.get("UNILAB_G1_ACTION_TRACE_INTERVAL", "20")))
    except ValueError:
        return 20


@dataclass(frozen=True)
class G1ActionTraceSnapshot:
    step: int
    task_name: str
    action_scale: float
    stand_action_authority: bool
    mode_observation: bool
    reward: np.ndarray
    terminated: np.ndarray
    commands: np.ndarray
    gait_enabled: np.ndarray
    dynamic_mode: np.ndarray
    current_actions: np.ndarray
    executed_actions: np.ndarray
    ctrl: np.ndarray | None
    default_angles: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    virtual_pd_torque: np.ndarray | None
    torques: np.ndarray | None
    linvel: np.ndarray
    gyro: np.ndarray
    base_height_target: float
    base_height: np.ndarray
    tilt_deg: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    left_contact_count: np.ndarray
    right_contact_count: np.ndarray
    base_minus_feet_center_xy: np.ndarray
    reward_log: Mapping[str, Any]


def _stats(name: str, value: Any) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return f"{name}: empty"
    finite = np.isfinite(arr)
    finite_arr = arr[finite]
    if finite_arr.size == 0:
        return f"{name}: shape={arr.shape} finite=0/{arr.size}"
    return (
        f"{name}: shape={arr.shape} finite={finite_arr.size}/{arr.size} "
        f"mean={float(np.mean(finite_arr)):.6g} "
        f"min={float(np.min(finite_arr)):.6g} "
        f"max={float(np.max(finite_arr)):.6g} "
        f"l1_mean={float(np.mean(np.sum(np.abs(np.atleast_2d(arr)), axis=1))):.6g} "
        f"max_abs={float(np.max(np.abs(finite_arr))):.6g}"
    )


def _head(name: str, value: Any, *, count: int = 8) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return f"{name}: {float(arr):.6g}"
    row = arr[0] if arr.ndim > 1 else arr
    head = np.asarray(row[:count], dtype=np.float64)
    return f"{name}[0,:{count}]: {np.array2string(head, precision=4, suppress_small=False)}"


def emit_g1_action_trace(snapshot: G1ActionTraceSnapshot) -> None:
    """Print one immutable diagnostic snapshot without touching environment state."""

    ctrl_delta = (
        None if snapshot.ctrl is None else snapshot.ctrl - snapshot.default_angles
    )
    ctrl_error = None if snapshot.ctrl is None else snapshot.ctrl - snapshot.dof_pos
    base_height_deficit = np.maximum(
        snapshot.base_height_target - snapshot.base_height,
        0.0,
    )

    print("[G1ActionTrace] begin")
    print(
        "[G1ActionTrace] "
        f"step={snapshot.step} task={snapshot.task_name} "
        f"action_scale={snapshot.action_scale:.6g} "
        f"stand_action_authority={snapshot.stand_action_authority} "
        f"mode_observation={snapshot.mode_observation} "
        f"reward_mean={float(np.mean(snapshot.reward)):.6g} "
        f"terminated_frac={float(np.mean(snapshot.terminated.astype(np.float32))):.6g}"
    )
    print("[G1ActionTrace] " + _stats("commands", snapshot.commands))
    print("[G1ActionTrace] " + _head("commands", snapshot.commands, count=3))
    print("[G1ActionTrace] " + _stats("gait_enabled", snapshot.gait_enabled))
    print("[G1ActionTrace] " + _stats("dynamic_mode", snapshot.dynamic_mode))
    print("[G1ActionTrace] " + _stats("current_actions", snapshot.current_actions))
    print("[G1ActionTrace] " + _head("current_actions", snapshot.current_actions))
    print("[G1ActionTrace] " + _stats("executed_actions", snapshot.executed_actions))
    print("[G1ActionTrace] " + _head("executed_actions", snapshot.executed_actions))
    print(
        "[G1ActionTrace] "
        + _stats(
            "executed_minus_current",
            snapshot.executed_actions - snapshot.current_actions,
        )
    )
    if snapshot.ctrl is not None:
        print("[G1ActionTrace] " + _stats("ctrl", snapshot.ctrl))
        print("[G1ActionTrace] " + _head("ctrl", snapshot.ctrl))
        print("[G1ActionTrace] " + _stats("ctrl_minus_default", ctrl_delta))
        print("[G1ActionTrace] " + _stats("ctrl_minus_dof_pos", ctrl_error))
    print(
        "[G1ActionTrace] "
        + _stats("dof_pos_minus_default", snapshot.dof_pos - snapshot.default_angles)
    )
    print("[G1ActionTrace] " + _stats("dof_vel", snapshot.dof_vel))
    if snapshot.virtual_pd_torque is not None:
        print("[G1ActionTrace] " + _stats("virtual_pd_tau", snapshot.virtual_pd_torque))
        print("[G1ActionTrace] " + _head("virtual_pd_tau", snapshot.virtual_pd_torque))
    if snapshot.torques is not None:
        print("[G1ActionTrace] " + _stats("info_torques", snapshot.torques))
    print("[G1ActionTrace] " + _stats("linvel", snapshot.linvel))
    print("[G1ActionTrace] " + _stats("gyro", snapshot.gyro))
    print(f"[G1ActionTrace] base_height_target={snapshot.base_height_target:.6g}")
    print("[G1ActionTrace] " + _stats("base_height", snapshot.base_height))
    print("[G1ActionTrace] " + _stats("base_height_deficit", base_height_deficit))
    print("[G1ActionTrace] " + _stats("tilt_deg", snapshot.tilt_deg))
    print("[G1ActionTrace] " + _stats("left_contact", snapshot.left_contact.astype(float)))
    print("[G1ActionTrace] " + _stats("right_contact", snapshot.right_contact.astype(float)))
    print("[G1ActionTrace] " + _stats("left_contact_count", snapshot.left_contact_count))
    print("[G1ActionTrace] " + _stats("right_contact_count", snapshot.right_contact_count))
    print(
        "[G1ActionTrace] "
        + _stats("base_minus_feet_center_xy", snapshot.base_minus_feet_center_xy)
    )
    for key in sorted(key for key in snapshot.reward_log if key.startswith("reward/")):
        print(f"[G1ActionTrace] {key}={float(snapshot.reward_log[key]):.6g}")
    print("[G1ActionTrace] end")
