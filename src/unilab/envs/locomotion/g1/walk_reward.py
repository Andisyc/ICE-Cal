"""Pure G1 walk reward kernels."""

from __future__ import annotations

import numpy as np

from unilab.dtype_config import get_global_dtype


def normalized_corridor_violation(error: np.ndarray, tolerance: float) -> np.ndarray:
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError("corridor tolerance must be positive")
    excess = np.maximum(np.abs(error) - tolerance, 0.0)
    return np.square(excess / tolerance)


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
    fallen = (tilt > np.deg2rad(float(max_tilt_deg))) | (
        base_height < float(min_base_height)
    )
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
