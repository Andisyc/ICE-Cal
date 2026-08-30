"""Pure G1 walk observation layout and tensor assembly."""

from __future__ import annotations

import numpy as np


def build_obs_groups_spec(
    *,
    mode_observation: bool,
    height_observation: bool,
    privileged_strength: bool,
    fada_privileged: bool,
    fada_body_count: int,
) -> dict[str, int]:
    mode_dim = 1 if mode_observation else 0
    height_dim = 1 if height_observation else 0
    privileged_strength_dim = 29 if privileged_strength else 0
    fada_privileged_dim = 174 + int(fada_body_count) if fada_privileged else 0
    critic_base_dim = 98 if fada_privileged else 101
    return {
        "obs": 98 + mode_dim + height_dim,
        "critic": (
            critic_base_dim
            + mode_dim
            + height_dim
            + privileged_strength_dim
            + fada_privileged_dim
        ),
    }


def assemble_walk_observation(
    *,
    noisy_gyro: np.ndarray,
    noisy_gravity: np.ndarray,
    noisy_diff: np.ndarray,
    noisy_dof_vel: np.ndarray,
    gyro: np.ndarray,
    gravity: np.ndarray,
    diff: np.ndarray,
    dof_vel: np.ndarray,
    last_actions: np.ndarray,
    command_obs: np.ndarray,
    gait_phase: np.ndarray,
    mode_obs: np.ndarray,
    linvel: np.ndarray,
    mode_observation: bool,
    walk_profile: bool,
    fada_privileged: bool,
    privileged_strength: np.ndarray | None,
    fada_privileged_obs: np.ndarray | None,
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    actor_gyro_scale = 0.25 if walk_profile else 1.0
    actor_dof_vel_scale = 0.05 if walk_profile else 1.0
    actor_parts = [
        noisy_gyro * actor_gyro_scale,
        -noisy_gravity,
        noisy_diff,
        noisy_dof_vel * actor_dof_vel_scale,
        last_actions,
        command_obs,
        gait_phase,
    ]
    if mode_observation:
        actor_parts.append(mode_obs)
    actor = np.concatenate(actor_parts, axis=1, dtype=dtype)

    critic_gyro_scale = 0.25 if walk_profile else 1.0
    critic_dof_vel_scale = 0.05 if walk_profile else 1.0
    critic_linvel_scale = 2.0 if walk_profile else 1.0
    critic_parts = [
        gyro * critic_gyro_scale,
        -gravity,
        diff,
        dof_vel * critic_dof_vel_scale,
        last_actions,
        command_obs,
        gait_phase,
    ]
    if mode_observation:
        critic_parts.append(mode_obs)
    critic = np.concatenate(critic_parts, axis=1, dtype=dtype)
    if not fada_privileged:
        critic = np.concatenate(
            [critic, np.asarray(linvel * critic_linvel_scale, dtype=dtype)],
            axis=1,
            dtype=dtype,
        )
    if privileged_strength is not None:
        critic = np.concatenate([critic, privileged_strength], axis=1, dtype=dtype)
    if fada_privileged_obs is not None:
        critic = np.concatenate([critic, fada_privileged_obs], axis=1, dtype=dtype)
    return {"obs": actor, "critic": critic}
