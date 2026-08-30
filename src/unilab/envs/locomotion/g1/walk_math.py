"""G1 joystick locomotion environments."""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.commands import (
    sample_velocity_commands,
    zero_small_xy_commands,
)


def sample_gait_phase_pairs(rng, num_samples: int, mode: str) -> np.ndarray:
    if mode == "independent":
        return np.asarray(
            np.column_stack(
                [
                    rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,)),
                    rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,)),
                ]
            ),
            dtype=get_global_dtype(),
        )

    phase = rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,))
    return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())


def sample_reset_base_qvel(rng, num_samples: int, limit: float) -> np.ndarray:
    return np.asarray(rng.uniform(-limit, limit, size=(num_samples, 6)), dtype=get_global_dtype())


def sample_g1_walk_commands(env: Any, num_samples: int) -> np.ndarray:
    low = np.asarray(env.cfg.commands.vel_limit[0], dtype=get_global_dtype())
    high = np.asarray(env.cfg.commands.vel_limit[1], dtype=get_global_dtype())
    commands = sample_velocity_commands(np.random.default_rng(), num_samples, low, high)
    zero_small_xy_commands(
        commands,
        threshold=float(getattr(env.cfg.commands, "small_xy_threshold", 0.0)),
    )
    standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
    transition_prob = float(getattr(env.cfg.commands, "rel_transition_envs", 0.0))
    standing_prob = min(max(standing_prob, 0.0), 1.0)
    transition_prob = min(max(transition_prob, 0.0), max(1.0 - standing_prob, 0.0))
    draw = np.random.uniform(size=(num_samples,))
    if transition_prob > 0.0:
        low = np.asarray(env.cfg.commands.transition_vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env.cfg.commands.transition_vel_limit[1], dtype=get_global_dtype())
        transition = (draw >= standing_prob) & (draw < standing_prob + transition_prob)
        if np.any(transition):
            commands[transition] = sample_velocity_commands(
                np.random.default_rng(), int(np.sum(transition)), low, high
            )
    if standing_prob > 0.0:
        commands[draw < standing_prob] = 0.0
    if getattr(env.cfg.commands, "heading_command", False):
        commands[:, 2] = 0.0
    return commands


def build_upper_body_pose_weights(pose_weights: list[float]) -> np.ndarray:
    weights = np.asarray(pose_weights, dtype=get_global_dtype()).copy()
    weights[:12] = 0.0
    return np.asarray(weights, dtype=get_global_dtype())


def compute_feet_phase_height_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    def cubic_bezier_height(phi: np.ndarray, swing_height: float) -> np.ndarray:
        phi_normalized = np.fmod(phi + np.pi, 2 * np.pi) - np.pi
        x = (phi_normalized + np.pi) / (2 * np.pi)

        def cubic_bezier_interpolation(
            y_start: np.ndarray, y_end: np.ndarray, t: np.ndarray
        ) -> np.ndarray:
            y_diff = y_end - y_start
            bezier = t**3 + 3 * (t**2 * (1 - t))
            return np.asarray(y_start + y_diff * bezier, dtype=get_global_dtype())

        stance = cubic_bezier_interpolation(np.zeros_like(x), np.full_like(x, swing_height), 2 * x)
        swing = cubic_bezier_interpolation(
            np.full_like(x, swing_height), np.zeros_like(x), 2 * x - 1
        )
        return np.where(x <= 0.5, stance, swing)

    left_target = cubic_bezier_height(gait_phase[:, 0], swing_height)
    right_target = cubic_bezier_height(gait_phase[:, 1], swing_height)
    return left_target, right_target


def _scalarize_sensor_values(sensor_values: np.ndarray) -> np.ndarray:
    sensor_array = np.asarray(sensor_values, dtype=get_global_dtype())
    if sensor_array.ndim == 1:
        return sensor_array
    if sensor_array.ndim == 2 and sensor_array.shape[1] == 1:
        return sensor_array[:, 0]
    raise ValueError(f"Expected scalar sensor values, got shape {sensor_array.shape}")


def compute_aggregated_foot_contact(backend: Any, sensor_names: list[str]) -> np.ndarray:
    contacts = [_scalarize_sensor_values(backend.get_sensor_data(name)) for name in sensor_names]
    return np.asarray(np.any(np.stack(contacts, axis=1) > 0.5, axis=1), dtype=np.bool_)


def compute_aggregated_foot_contact_count(backend: Any, sensor_names: list[str]) -> np.ndarray:
    contacts = [_scalarize_sensor_values(backend.get_sensor_data(name)) for name in sensor_names]
    return np.sum(np.stack(contacts, axis=1) > 0.5, axis=1).astype(get_global_dtype())


def compute_feet_phase_contact_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    contact_height_threshold = swing_height * 0.5
    return left_target <= contact_height_threshold, right_target <= contact_height_threshold


def compute_forward_speed_gate(linvel: np.ndarray, min_forward_speed: float) -> np.ndarray:
    forward_speed = np.maximum(linvel[:, 0], 0.0)
    return np.asarray(forward_speed >= min_forward_speed, dtype=get_global_dtype())


def compute_forward_command_mask(commands: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(commands[:, 0], 0.0) > 1.0e-6, dtype=get_global_dtype())


def compute_command_active_mask(
    commands: np.ndarray, *, xy_threshold: float, yaw_threshold: float
) -> np.ndarray:
    xy_norm = np.linalg.norm(commands[:, :2], axis=1)
    yaw_abs = np.abs(commands[:, 2])
    return np.asarray(
        (xy_norm > xy_threshold) | (yaw_abs > yaw_threshold), dtype=get_global_dtype()
    )


def compute_external_command_mask(commands: np.ndarray, *, epsilon: float = 1.0e-6) -> np.ndarray:
    return np.asarray(np.any(np.abs(commands) > epsilon, axis=1), dtype=get_global_dtype())


def compute_tracking_gate(
    commands: np.ndarray,
    linvel: np.ndarray,
    gyro: np.ndarray,
    *,
    tracking_sigma: float,
    threshold: float,
) -> np.ndarray:
    lin_error = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
    yaw_error = np.square(commands[:, 2] - gyro[:, 2])
    tracking_score = np.exp(-(lin_error + yaw_error) / tracking_sigma)
    return np.asarray(tracking_score > threshold, dtype=get_global_dtype())


def compute_gait_phase_height_violation(
    left_foot_z: np.ndarray,
    right_foot_z: np.ndarray,
    gait_phase: np.ndarray,
    swing_height: float,
) -> np.ndarray:
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    return np.asarray(
        np.square(left_foot_z - left_target) + np.square(right_foot_z - right_target),
        dtype=get_global_dtype(),
    )


def compute_gait_phase_contrast_violation(
    left_foot_z: np.ndarray,
    right_foot_z: np.ndarray,
    gait_phase: np.ndarray,
    swing_height: float,
) -> np.ndarray:
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    actual_delta = left_foot_z - right_foot_z
    target_delta = left_target - right_target
    return np.asarray(np.square(actual_delta - target_delta), dtype=get_global_dtype())


def compute_gait_phase_contact_violation(
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    gait_phase: np.ndarray,
    swing_height: float,
) -> np.ndarray:
    left_target, right_target = compute_feet_phase_contact_targets(gait_phase, swing_height)
    left_error = np.asarray(left_contact != left_target, dtype=get_global_dtype())
    right_error = np.asarray(right_contact != right_target, dtype=get_global_dtype())
    return np.asarray(0.5 * (left_error + right_error), dtype=get_global_dtype())


def compute_forward_progress_failure(
    current_position: np.ndarray,
    initial_position: np.ndarray,
    initial_yaw: np.ndarray,
    steps_before_increment: np.ndarray,
    commands: np.ndarray,
    *,
    ctrl_dt: float,
    grace_steps: int,
    min_command_forward_speed: float,
    min_average_forward_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return progress-failure mask and reset-frame episode-average forward speed."""
    current = np.asarray(current_position, dtype=get_global_dtype())
    initial = np.asarray(initial_position, dtype=get_global_dtype())
    yaw = np.asarray(initial_yaw, dtype=get_global_dtype())
    steps = np.asarray(steps_before_increment)
    command = np.asarray(commands, dtype=get_global_dtype())
    batch = int(current.shape[0])
    if current.shape != (batch, 3) or initial.shape != (batch, 3):
        raise ValueError("forward-progress positions must both have shape (N, 3)")
    if yaw.shape != (batch,) or steps.shape != (batch,) or command.shape != (batch, 3):
        raise ValueError("forward-progress yaw/steps/commands shapes do not match the batch")
    if float(ctrl_dt) <= 0.0 or int(grace_steps) <= 0:
        raise ValueError("forward-progress ctrl_dt and grace_steps must be positive")

    delta = current[:, :2] - initial[:, :2]
    forward_displacement = np.cos(yaw) * delta[:, 0] + np.sin(yaw) * delta[:, 1]
    completed_steps = steps.astype(np.int64, copy=False) + 1
    elapsed_seconds = completed_steps.astype(get_global_dtype()) * float(ctrl_dt)
    average_forward_speed = forward_displacement / elapsed_seconds
    failure = (
        (completed_steps >= int(grace_steps))
        & (command[:, 0] >= float(min_command_forward_speed))
        & (average_forward_speed < float(min_average_forward_speed) - 1.0e-6)
    )
    return np.asarray(failure, dtype=np.bool_), np.asarray(
        average_forward_speed, dtype=get_global_dtype()
    )
