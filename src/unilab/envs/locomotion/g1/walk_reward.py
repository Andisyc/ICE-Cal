"""Pure G1 walk reward kernels."""

from __future__ import annotations

import numpy as np

from unilab.dtype_config import get_global_dtype


def tracking_planar_speed(commands: np.ndarray, velocity: np.ndarray, error_scale: float) -> np.ndarray:
    """Exponential of planar error magnitude, with a scale in m/s."""
    error = np.linalg.norm(commands[:, :2] - velocity[:, :2], axis=1)
    return np.asarray(np.exp(-error / error_scale), dtype=get_global_dtype())


def command_direction_speed_deficit(
    commands: np.ndarray, velocity: np.ndarray, min_command: float
) -> np.ndarray:
    """Relative shortfall along commanded translation, retaining reverse speed."""
    planar = commands[:, :2]
    speed = np.linalg.norm(planar, axis=1)
    direction = np.divide(planar, speed[:, None], out=np.zeros_like(planar), where=speed[:, None] > 0)
    achieved = np.sum(velocity[:, :2] * direction, axis=1)
    cost = np.maximum(speed - achieved, 0.0) / np.maximum(speed, min_command)
    return np.asarray(np.where(speed > 0, cost, 0.0), dtype=get_global_dtype())


def normalized_corridor_violation(error: np.ndarray, tolerance: float) -> np.ndarray:
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError("corridor tolerance must be positive")
    excess = np.maximum(np.abs(error) - tolerance, 0.0)
    return np.square(excess / tolerance)


def phase_stance_targets(
    gait_phase: np.ndarray,
    *,
    duty_factor: float,
    is_null: np.ndarray | None = None,
) -> np.ndarray:
    """Desired support mask, shared by contact and stance-only foot orientation."""
    phase = np.asarray(gait_phase, dtype=get_global_dtype())
    if phase.ndim != 2 or phase.shape[1] != 2 or not np.isfinite(phase).all():
        raise ValueError(f"gait_phase must be finite with shape (N, 2), got {phase.shape}")
    duty_factor = float(duty_factor)
    if not 0.5 < duty_factor < 1.0:
        raise ValueError("duty_factor must be in (0.5, 1.0)")
    phase_cycle = np.mod(phase, 2.0 * np.pi)
    stance_boundary = np.asarray(duty_factor * (2.0 * np.pi), dtype=phase_cycle.dtype)
    stance_expected = phase_cycle < stance_boundary
    if is_null is not None:
        null = np.asarray(is_null, dtype=np.bool_)
        if null.shape != (phase.shape[0],):
            raise ValueError("null command mask must match gait_phase rows")
        # A standing command requests double support without resetting the clock.
        stance_expected[null] = True
    return stance_expected


def phase_contact_mismatch_cost(
    gait_phase: np.ndarray,
    left_force_z: np.ndarray,
    right_force_z: np.ndarray,
    *,
    duty_factor: float,
    contact_force_threshold: float,
    is_null: np.ndarray | None = None,
) -> np.ndarray:
    stance_expected = phase_stance_targets(
        gait_phase, duty_factor=duty_factor, is_null=is_null
    )
    contact_force_threshold = float(contact_force_threshold)
    if not np.isfinite(contact_force_threshold) or contact_force_threshold <= 0.0:
        raise ValueError("contact_force_threshold must be finite and positive")
    left = np.asarray(left_force_z, dtype=get_global_dtype())
    right = np.asarray(right_force_z, dtype=get_global_dtype())
    if left.shape != (stance_expected.shape[0],) or right.shape != left.shape:
        raise ValueError("foot vertical forces must match gait_phase rows")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("foot vertical forces must be finite")
    contact_measured = np.column_stack(
        [left > contact_force_threshold, right > contact_force_threshold]
    )
    return np.asarray(
        np.mean(stance_expected != contact_measured, axis=1), dtype=get_global_dtype()
    )


def null_foot_force_balance_l1(
    left_force_z: np.ndarray,
    right_force_z: np.ndarray,
    is_null: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    left = np.asarray(left_force_z, dtype=get_global_dtype())
    right = np.asarray(right_force_z, dtype=get_global_dtype())
    null = np.asarray(is_null, dtype=np.bool_)
    if left.ndim != 1 or right.shape != left.shape or null.shape != left.shape:
        raise ValueError("foot forces and null mask must be matching rank-one arrays")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("foot vertical forces must be finite")
    left = np.maximum(left, 0.0)
    right = np.maximum(right, 0.0)
    total = left + right
    imbalance = np.where(
        total > epsilon,
        np.abs(left - right) / np.maximum(total, epsilon),
        1.0,
    )
    return np.asarray(imbalance * null, dtype=get_global_dtype())


def null_torque_relaxation_l2(
    torques: np.ndarray,
    tau_max: np.ndarray,
    is_null: np.ndarray,
) -> np.ndarray:
    torque = np.asarray(torques, dtype=get_global_dtype())
    limits = np.asarray(tau_max, dtype=get_global_dtype())
    null = np.asarray(is_null, dtype=np.bool_)
    if torque.ndim != 2:
        raise ValueError("torques must have shape (N, A)")
    if limits.shape != (torque.shape[1],) or null.shape != (torque.shape[0],):
        raise ValueError("torque limits and null mask must match torque dimensions")
    if not np.isfinite(torque).all() or not np.isfinite(limits).all() or np.any(limits <= 0.0):
        raise ValueError("torques and positive torque limits must be finite")
    normalized = np.clip(torque, -limits[None, :], limits[None, :]) / limits[None, :]
    return np.asarray(np.mean(np.square(normalized), axis=1) * null, dtype=get_global_dtype())


def stand_action_l2(actions: np.ndarray, stand_mask: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.sum(np.square(actions), axis=1) * stand_mask,
        dtype=get_global_dtype(),
    )


def stand_still_l1(
    dof_pos: np.ndarray, default_angles: np.ndarray, stand_mask: np.ndarray
) -> np.ndarray:
    return np.asarray(
        np.sum(np.abs(dof_pos - default_angles), axis=1) * stand_mask,
        dtype=get_global_dtype(),
    )


def stand_dof_vel_l2(dof_vel: np.ndarray, stand_mask: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.sum(np.square(dof_vel), axis=1) * stand_mask,
        dtype=get_global_dtype(),
    )


def stand_lin_vel_xy_l2(linvel: np.ndarray, stand_mask: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.sum(np.square(linvel[:, :2]), axis=1) * stand_mask,
        dtype=get_global_dtype(),
    )


def stand_yaw_vel_l2(gyro: np.ndarray, stand_mask: np.ndarray) -> np.ndarray:
    return np.asarray(np.square(gyro[:, 2]) * stand_mask, dtype=get_global_dtype())


def stand_tilt_l2(gravity: np.ndarray | None, stand_mask: np.ndarray) -> np.ndarray:
    if gravity is None:
        return np.zeros(stand_mask.shape, dtype=get_global_dtype())
    return np.asarray(
        np.sum(np.square(gravity[:, :2]), axis=1) * stand_mask,
        dtype=get_global_dtype(),
    )


def stand_tilt_margin_l2(
    gravity: np.ndarray | None,
    stand_mask: np.ndarray,
    *,
    soft_limit_deg: float,
    hard_limit_deg: float,
) -> np.ndarray:
    if gravity is None:
        return np.zeros(stand_mask.shape, dtype=get_global_dtype())
    tilt = np.arccos(np.clip(gravity[:, 2], -1.0, 1.0))
    soft_limit = np.deg2rad(float(soft_limit_deg))
    hard_limit = np.deg2rad(float(hard_limit_deg))
    span = max(float(hard_limit - soft_limit), 1.0e-6)
    margin = np.maximum((tilt - soft_limit) / span, 0.0)
    return np.asarray(np.square(margin) * stand_mask, dtype=get_global_dtype())


def stand_fall_l2(
    gravity: np.ndarray | None,
    base_height: np.ndarray | None,
    stand_mask: np.ndarray,
    *,
    max_tilt_deg: float,
    min_base_height: float,
) -> np.ndarray:
    if gravity is None or base_height is None:
        return np.zeros(stand_mask.shape, dtype=get_global_dtype())
    tilt = np.arccos(np.clip(gravity[:, 2], -1.0, 1.0))
    fallen = (tilt > np.deg2rad(float(max_tilt_deg))) | (base_height < float(min_base_height))
    return np.asarray(fallen.astype(get_global_dtype()) * stand_mask, dtype=get_global_dtype())


def stand_height_margin_l2(
    target: np.ndarray,
    observed: np.ndarray,
    stand_mask: np.ndarray,
    *,
    margin: float,
) -> np.ndarray:
    low_deficit = np.maximum(target - float(margin) - observed, 0.0)
    return np.asarray(np.square(low_deficit) * stand_mask, dtype=get_global_dtype())


def stand_height_deficit_l1(
    target: np.ndarray,
    observed: np.ndarray,
    stand_mask: np.ndarray,
    *,
    margin: float,
) -> np.ndarray:
    low_deficit = np.maximum(target - float(margin) - observed, 0.0)
    return np.asarray(low_deficit * stand_mask, dtype=get_global_dtype())


def stand_contact_balance_l1(
    left_count: np.ndarray,
    right_count: np.ndarray,
    stand_mask: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    total = left_count + right_count
    imbalance = np.where(
        total > float(epsilon),
        np.abs(left_count - right_count) / np.maximum(total, float(epsilon)),
        1.0,
    )
    return np.asarray(imbalance * stand_mask, dtype=get_global_dtype())


def resolve_stand_height_target(target: object, *, num_envs: int) -> np.ndarray:
    resolved = np.asarray(target, dtype=get_global_dtype())
    if resolved.ndim == 0:
        return np.full((num_envs,), float(resolved), dtype=get_global_dtype())
    if resolved.ndim == 2 and resolved.shape == (num_envs, 1):
        resolved = resolved[:, 0]
    if resolved.shape != (num_envs,):
        raise ValueError(
            "standing height target must be scalar or have shape "
            f"({num_envs},), got {resolved.shape}"
        )
    return np.asarray(resolved, dtype=get_global_dtype())
