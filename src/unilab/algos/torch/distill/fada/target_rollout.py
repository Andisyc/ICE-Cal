from __future__ import annotations

from typing import Any

import numpy as np


def target_tracking_camera_kwargs() -> dict[str, Any]:
    """Return the shared single-environment tracking-camera contract."""

    return {"cam_tracking": True, "cam_tracking_env_idx": 0}


def scheduled_target_command(
    command_start: tuple[float, float, float],
    command_target: np.ndarray,
    *,
    ramp_steps: int,
    step: int,
) -> np.ndarray:
    """Return the externally owned command for one rollout step."""

    start = np.asarray(command_start, dtype=np.float32)
    target = np.asarray(command_target, dtype=np.float32)
    if start.shape != (3,) or target.shape != (3,):
        raise ValueError("FADA target command schedule requires 3-D commands")
    if ramp_steps < 0 or step < 0:
        raise ValueError("FADA target ramp_steps and step must be non-negative")
    if ramp_steps == 0 or step >= ramp_steps:
        return target.copy()
    return start + (target - start) * (float(step + 1) / float(ramp_steps))


def apply_external_command(env: Any, command: np.ndarray) -> None:
    """Update the env-owned command row and refresh observations."""

    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    commands = info.get("commands") if isinstance(info, dict) else None
    if not isinstance(commands, np.ndarray) or commands.ndim != 2 or commands.shape[1] < 3:
        raise RuntimeError("FADA target command schedule requires env-owned command rows")
    commands[:, :3] = command[None, :]
    refresh = getattr(env, "refresh_state", None)
    if not callable(refresh):
        raise RuntimeError("FADA target command schedule requires env.refresh_state()")
    refresh()


def rollout_done_flags(state: Any, *, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """Read termination and truncation independently from an environment state."""

    shape = (int(num_envs),)

    def flag(name: str) -> np.ndarray:
        value = getattr(state, name, None)
        if value is None:
            return np.zeros(shape, dtype=np.bool_)
        result = np.asarray(value, dtype=np.bool_).reshape(-1)
        if result.shape != shape:
            raise ValueError(f"{name} mask shape mismatch: expected {shape}, got {result.shape}")
        return result

    return flag("terminated"), flag("truncated")


def rollout_terminal_reasons(state: Any, *, num_envs: int) -> tuple[str | None, ...]:
    """Resolve task-owned fall identity without conflating other termination causes."""

    terminated, truncated = rollout_done_flags(state, num_envs=num_envs)
    info = getattr(state, "info", None)
    fall_raw = info.get("fall_terminated") if isinstance(info, dict) else None
    if fall_raw is None:
        fall = np.zeros((int(num_envs),), dtype=np.bool_)
    else:
        fall = np.asarray(fall_raw, dtype=np.bool_).reshape(-1)
        if fall.shape != terminated.shape:
            raise ValueError(
                f"fall_terminated mask shape mismatch: expected {terminated.shape}, got {fall.shape}"
            )
    reasons: list[str | None] = []
    for index in range(int(num_envs)):
        if bool(fall[index]):
            reasons.append("fall")
        elif bool(terminated[index]):
            reasons.append("environment_termination")
        elif bool(truncated[index]):
            reasons.append("truncated")
        else:
            reasons.append(None)
    return tuple(reasons)
