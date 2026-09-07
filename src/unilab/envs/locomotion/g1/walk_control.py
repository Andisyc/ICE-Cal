"""Pure G1 walk action-authority and gait-phase decisions."""

from __future__ import annotations

import numpy as np


def select_authority_actions(
    actions: np.ndarray,
    active: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    if not enabled or np.all(active):
        return actions
    executed = np.array(actions, copy=True)
    executed[~np.asarray(active, dtype=bool)] = 0.0
    return executed


def advance_gait_phase(
    gait_phase: np.ndarray,
    *,
    active: np.ndarray,
    delta: float | np.ndarray,
    enabled: bool,
    freeze_inactive: bool,
    stand_phase: np.ndarray,
) -> np.ndarray:
    phase = np.array(gait_phase, copy=True)
    if not enabled:
        return np.zeros_like(phase)
    if freeze_inactive:
        active_mask = np.asarray(active, dtype=bool)
        delta_values = np.asarray(delta, dtype=phase.dtype)
        if delta_values.ndim == 0:
            phase[active_mask] = (phase[active_mask] + float(delta_values)) % (2 * np.pi)
        else:
            if delta_values.shape != (phase.shape[0],):
                raise ValueError(f"delta must be scalar or shape ({phase.shape[0]},)")
            phase[active_mask] = (phase[active_mask] + delta_values[active_mask, None]) % (
                2 * np.pi
            )
        phase[~active_mask] = np.asarray(stand_phase, dtype=phase.dtype)
        return phase
    delta_values = np.asarray(delta, dtype=phase.dtype)
    if delta_values.ndim == 0:
        phase[:] = (phase + float(delta_values)) % (2 * np.pi)
    else:
        if delta_values.shape != (phase.shape[0],):
            raise ValueError(f"delta must be scalar or shape ({phase.shape[0]},)")
        phase[:] = (phase + delta_values[:, None]) % (2 * np.pi)
    return phase


def _phase_pair(value: np.ndarray, *, name: str, dtype: np.dtype) -> np.ndarray:
    phase = np.asarray(value, dtype=dtype)
    if phase.shape != (2,) or not np.isfinite(phase).all():
        raise ValueError(f"{name} must contain two finite phases")
    return phase


def reset_command_gait_phase(
    is_null: np.ndarray,
    *,
    startup_phase: np.ndarray,
    stand_phase: np.ndarray,
) -> np.ndarray:
    null = np.asarray(is_null, dtype=np.bool_)
    if null.ndim != 1:
        raise ValueError("is_null must be rank one")
    dtype = np.dtype(np.result_type(startup_phase, stand_phase, np.float32))
    startup = _phase_pair(startup_phase, name="startup_phase", dtype=dtype)
    stand = _phase_pair(stand_phase, name="stand_phase", dtype=dtype)
    return np.asarray(np.where(null[:, None], stand[None, :], startup[None, :]), dtype=dtype)


def step_command_gait_phase(
    gait_phase: np.ndarray,
    *,
    was_null: np.ndarray,
    is_null: np.ndarray,
    frequency: np.ndarray,
    ctrl_dt: float,
    startup_phase: np.ndarray,
    stand_phase: np.ndarray,
) -> np.ndarray:
    phase = np.asarray(gait_phase)
    if phase.ndim != 2 or phase.shape[1] != 2 or not np.isfinite(phase).all():
        raise ValueError(f"gait_phase must be finite with shape (N, 2), got {phase.shape}")
    rows = phase.shape[0]
    previous_null = np.asarray(was_null, dtype=np.bool_)
    current_null = np.asarray(is_null, dtype=np.bool_)
    gait_frequency = np.asarray(frequency, dtype=phase.dtype)
    if previous_null.shape != (rows,) or current_null.shape != (rows,):
        raise ValueError("command regime masks must match gait_phase rows")
    if gait_frequency.shape != (rows,) or not np.isfinite(gait_frequency).all():
        raise ValueError("frequency must be finite and match gait_phase rows")
    if np.any(gait_frequency < 0.0) or not np.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
        raise ValueError("frequency must be non-negative and ctrl_dt must be positive")
    startup = _phase_pair(startup_phase, name="startup_phase", dtype=phase.dtype)
    stand = _phase_pair(stand_phase, name="stand_phase", dtype=phase.dtype)

    result = transition_command_gait_phase(
        phase,
        was_null=previous_null,
        is_null=current_null,
        startup_phase=startup,
        stand_phase=stand,
    )
    continuing = ~previous_null & ~current_null
    delta = 2.0 * np.pi * gait_frequency[continuing] * float(ctrl_dt)
    result[continuing] = (result[continuing] + delta[:, None]) % (2.0 * np.pi)
    return result


def transition_command_gait_phase(
    gait_phase: np.ndarray,
    *,
    was_null: np.ndarray,
    is_null: np.ndarray,
    startup_phase: np.ndarray,
    stand_phase: np.ndarray,
) -> np.ndarray:
    """Commit a new command regime without advancing the phase clock."""

    phase = np.asarray(gait_phase)
    if phase.ndim != 2 or phase.shape[1] != 2 or not np.isfinite(phase).all():
        raise ValueError(f"gait_phase must be finite with shape (N, 2), got {phase.shape}")
    rows = phase.shape[0]
    previous_null = np.asarray(was_null, dtype=np.bool_)
    current_null = np.asarray(is_null, dtype=np.bool_)
    if previous_null.shape != (rows,) or current_null.shape != (rows,):
        raise ValueError("command regime masks must match gait_phase rows")
    startup = _phase_pair(startup_phase, name="startup_phase", dtype=phase.dtype)
    stand = _phase_pair(stand_phase, name="stand_phase", dtype=phase.dtype)
    result = np.array(phase, copy=True)
    result[current_null] = stand
    result[previous_null & ~current_null] = startup
    return result


def estimate_clipped_pd_torque(
    target: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau_max: np.ndarray,
) -> np.ndarray:
    if target is None:
        raise ValueError("target must be provided after an applied action")
    target_values = np.asarray(target)
    position = np.asarray(dof_pos, dtype=target_values.dtype)
    velocity = np.asarray(dof_vel, dtype=target_values.dtype)
    kp_values = np.asarray(kp, dtype=target_values.dtype)
    kd_values = np.asarray(kd, dtype=target_values.dtype)
    limits = np.asarray(tau_max, dtype=target_values.dtype)
    if target_values.ndim != 2:
        raise ValueError("target must have shape (N, A)")
    if any(
        value.shape != target_values.shape for value in (position, velocity, kp_values, kd_values)
    ):
        raise ValueError("target, joint state, and per-row gains must have matching shapes")
    if limits.shape != (target_values.shape[1],):
        raise ValueError("tau_max must have shape (A,)")
    if not all(
        np.isfinite(value).all()
        for value in (target_values, position, velocity, kp_values, kd_values, limits)
    ):
        raise ValueError("PD torque inputs must be finite")
    if np.any(kp_values <= 0.0) or np.any(kd_values < 0.0) or np.any(limits <= 0.0):
        raise ValueError("PD gains and torque limits must be valid")
    torque = kp_values * (target_values - position) - kd_values * velocity
    return np.asarray(np.clip(torque, -limits[None, :], limits[None, :]))
