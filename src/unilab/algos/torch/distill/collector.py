from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .data import DistillationTensorDataset, build_distillation_dataset


def _obs_array(obs: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Observation key {key!r} not found; available keys={sorted(obs.keys())}")
    arr = np.asarray(obs[key], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Observation key {key!r} must be rank-2, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Observation key {key!r} must contain only finite values")
    return arr


def command_active_mask(
    commands: np.ndarray,
    *,
    xy_threshold: float,
    yaw_threshold: float,
) -> np.ndarray:
    """Return rows whose velocity command should use walking teacher samples."""

    commands_np = np.asarray(commands, dtype=np.float32)
    if commands_np.ndim != 2 or commands_np.shape[1] != 3:
        raise ValueError(f"commands must have shape (N, 3), got {commands_np.shape}")
    if not np.all(np.isfinite(commands_np)):
        raise ValueError("commands must contain only finite values")

    xy_threshold = float(xy_threshold)
    yaw_threshold = float(yaw_threshold)
    if not np.isfinite(xy_threshold) or xy_threshold < 0.0:
        raise ValueError(f"xy_threshold must be finite and non-negative, got {xy_threshold}")
    if not np.isfinite(yaw_threshold) or yaw_threshold < 0.0:
        raise ValueError(f"yaw_threshold must be finite and non-negative, got {yaw_threshold}")

    xy_norm = np.linalg.norm(commands_np[:, :2], axis=1)
    yaw_abs = np.abs(commands_np[:, 2])
    return np.asarray((xy_norm > xy_threshold) | (yaw_abs > yaw_threshold), dtype=np.bool_)


def _info_array(
    info: Mapping[str, Any],
    key: str,
    *,
    expected_rows: int,
) -> np.ndarray:
    if key not in info:
        raise KeyError(f"Info key {key!r} not found; available keys={sorted(info.keys())}")
    arr = np.asarray(info[key], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Info key {key!r} must be rank-2, got shape {arr.shape}")
    if arr.shape[0] != int(expected_rows):
        raise ValueError(
            f"Info key {key!r} row mismatch: expected {int(expected_rows)}, got {arr.shape[0]}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Info key {key!r} must contain only finite values")
    return arr


def _command_sample_mask(
    info: Mapping[str, Any],
    *,
    sample_filter: str,
    command_info_key: str,
    expected_rows: int,
    xy_threshold: float,
    yaw_threshold: float,
) -> np.ndarray:
    if sample_filter == "none":
        return np.ones((int(expected_rows),), dtype=np.bool_)

    commands = _info_array(info, command_info_key, expected_rows=int(expected_rows))
    active = command_active_mask(
        commands,
        xy_threshold=xy_threshold,
        yaw_threshold=yaw_threshold,
    )
    if sample_filter == "active":
        return active
    if sample_filter == "inactive":
        return np.asarray(~active, dtype=np.bool_)
    raise ValueError(f"Unsupported command_sample_filter: {sample_filter!r}")


def project_student_obs(
    source_obs: np.ndarray,
    *,
    projection: str,
    expected_student_obs_dim: int,
    student_drop_index: int | None = None,
) -> np.ndarray:
    """Project teacher/env observations into deployable student observations."""

    source_obs = np.asarray(source_obs, dtype=np.float32)
    if projection == "identity":
        student_obs = source_obs
    elif projection == "drop_index":
        if student_drop_index is None:
            raise ValueError("student_drop_index is required when student_projection='drop_index'")
        drop_index = int(student_drop_index)
        if drop_index < 0 or drop_index >= source_obs.shape[1]:
            raise ValueError(
                f"student_drop_index must be in [0, {source_obs.shape[1]}), got {drop_index}"
            )
        student_obs = np.concatenate(
            [source_obs[:, :drop_index], source_obs[:, drop_index + 1 :]],
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported student_projection: {projection!r}")

    if student_obs.shape[1] != int(expected_student_obs_dim):
        raise ValueError(
            "student projection dim mismatch: "
            f"expected {int(expected_student_obs_dim)}, got {student_obs.shape[1]}"
        )
    return np.asarray(student_obs, dtype=np.float32)


def project_teacher_obs(
    source_obs: np.ndarray,
    *,
    projection: str,
    expected_teacher_obs_dim: int,
) -> tuple[np.ndarray, bool]:
    """Project live env observations into the frozen teacher checkpoint input."""

    source_obs = np.asarray(source_obs, dtype=np.float32)
    if projection == "identity":
        teacher_obs = source_obs
        synthetic_tail = False
    elif projection == "pad_zeros":
        pad_dim = int(expected_teacher_obs_dim) - int(source_obs.shape[1])
        if pad_dim < 0:
            raise ValueError(
                "teacher pad_zeros projection cannot shrink observations: "
                f"source={source_obs.shape[1]} expected={int(expected_teacher_obs_dim)}"
            )
        teacher_obs = np.concatenate(
            [source_obs, np.zeros((source_obs.shape[0], pad_dim), dtype=np.float32)],
            axis=1,
        )
        synthetic_tail = pad_dim > 0
    else:
        raise ValueError(f"Unsupported teacher_projection: {projection!r}")

    if teacher_obs.shape[1] != int(expected_teacher_obs_dim):
        raise ValueError(
            "teacher projection dim mismatch: "
            f"expected {int(expected_teacher_obs_dim)}, got {teacher_obs.shape[1]}"
        )
    return np.asarray(teacher_obs, dtype=np.float32), synthetic_tail


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _policy_actions(
    policy: torch.nn.Module,
    obs: np.ndarray,
    *,
    action_dim: int,
    policy_name: str,
) -> np.ndarray:
    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=_module_device(policy),
    )
    with torch.inference_mode():
        action_tensor = policy(obs_tensor)
    if isinstance(action_tensor, tuple):
        action_tensor = action_tensor[0]
    action_tensor = torch.as_tensor(action_tensor).detach()
    if action_tensor.ndim != 2:
        raise ValueError(
            f"{policy_name} action must be rank-2, got shape {tuple(action_tensor.shape)}"
        )
    if action_tensor.shape[0] != obs.shape[0] or action_tensor.shape[1] != int(action_dim):
        raise ValueError(
            f"{policy_name} action shape mismatch: "
            f"expected ({obs.shape[0]}, {int(action_dim)}), "
            f"got {tuple(action_tensor.shape)}"
        )
    return action_tensor.cpu().numpy().astype(np.float32)


def _state_done_mask(state: Any, *, expected_rows: int) -> np.ndarray:
    terminated = getattr(state, "terminated", None)
    truncated = getattr(state, "truncated", None)
    if terminated is None and truncated is None:
        return np.zeros((int(expected_rows),), dtype=np.bool_)

    done = np.zeros((int(expected_rows),), dtype=np.bool_)
    for name, value in (("terminated", terminated), ("truncated", truncated)):
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.bool_).reshape(-1)
        if arr.shape != done.shape:
            raise ValueError(
                f"collector state.{name} shape mismatch: expected {done.shape}, got {arr.shape}"
            )
        np.logical_or(done, arr, out=done)
    return done


def _state_has_autoreset_final_observation(state: Any, done: np.ndarray) -> bool:
    if not np.any(done):
        return False
    final_observation = getattr(state, "final_observation", None)
    if isinstance(final_observation, Mapping):
        return True
    info = getattr(state, "info", None)
    if isinstance(info, Mapping) and isinstance(info.get("final_observation"), Mapping):
        terminal_mask = info.get("_final_observation")
        if terminal_mask is None:
            return True
        mask = np.asarray(terminal_mask, dtype=np.bool_).reshape(-1)
        return mask.shape == done.shape and bool(np.any(mask[done]))
    return False


def _reset_done_rows_after_step(
    env: Any,
    state: Any,
    *,
    num_envs: int,
) -> tuple[dict[str, Any], dict[str, Any], int, int, int]:
    """Return next obs/info after guarding student-policy rollouts from terminal drift.

    UniLab NpEnv autoreset keeps the terminal flag visible while replacing done
    rows with reset observations and recording final_observation. Fake or custom
    envs may return terminal rows without autoreset; those rows are reset here
    before the next collection iteration samples them.
    """

    obs = getattr(state, "obs")
    info = getattr(state, "info", {})
    done = _state_done_mask(state, expected_rows=int(num_envs))
    done_count = int(np.count_nonzero(done))
    if done_count == 0:
        return obs, info, 0, 0, 0
    if _state_has_autoreset_final_observation(state, done):
        return obs, info, done_count, done_count, 0
    if not callable(getattr(env, "reset", None)):
        raise ValueError("collector saw done rows but env.reset is not callable")

    done_indices = np.flatnonzero(done).astype(np.int32)
    reset_obs, reset_info = env.reset(done_indices)
    next_obs = {key: np.asarray(value).copy() for key, value in obs.items()}
    for key, value in reset_obs.items():
        if key not in next_obs:
            raise KeyError(f"Reset observation key {key!r} not found in step observation")
        next_obs[key][done_indices] = value

    next_info = dict(info)
    if reset_info:
        for key, value in reset_info.items():
            if isinstance(value, np.ndarray):
                if key not in next_info:
                    full_shape = (int(num_envs),) + value.shape[1:]
                    next_info[key] = np.zeros(full_shape, dtype=value.dtype)
                next_info[key][done_indices] = value
            else:
                next_info[key] = value
    return next_obs, next_info, done_count, 0, done_count


def _set_transition_command_rows(
    env: Any,
    *,
    command_info_key: str,
    command_rows: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Atomically update vectorized commands and refresh policy observations."""

    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    commands = info.get(command_info_key) if isinstance(info, Mapping) else None
    commands_np = np.asarray(commands, dtype=np.float32) if commands is not None else None
    command_rows = np.asarray(command_rows, dtype=np.float32)
    if commands_np is None or commands_np.ndim != 2 or commands_np.shape[1] < 3:
        raise RuntimeError(
            "transition collection requires env.state.info[command_info_key] "
            "with shape (num_envs, >=3)"
        )
    if command_rows.shape != (commands_np.shape[0], 3):
        raise ValueError(
            "transition command rows shape mismatch: "
            f"expected {(commands_np.shape[0], 3)}, got {command_rows.shape}"
        )
    if not np.all(np.isfinite(command_rows)):
        raise ValueError("transition command rows must contain only finite values")
    commands[:, :3] = command_rows.astype(commands.dtype, copy=False)
    refresh_state = getattr(env, "refresh_state", None)
    if not callable(refresh_state):
        raise RuntimeError("transition collection requires env.refresh_state()")
    refreshed = refresh_state()
    refreshed_state = refreshed if refreshed is not None else getattr(env, "state", None)
    if refreshed_state is None:
        raise RuntimeError("env.refresh_state() did not return or retain an env state")
    refreshed_obs = getattr(refreshed_state, "obs", None)
    refreshed_info = getattr(refreshed_state, "info", None)
    if not isinstance(refreshed_obs, dict) or not isinstance(refreshed_info, Mapping):
        raise RuntimeError("transition command refresh must return dict obs and info")
    return refreshed_obs, dict(refreshed_info)


def collect_transition_distillation_dataset_from_env(
    env: Any,
    *,
    num_samples: int,
    expected_student_obs_dim: int,
    expected_teacher_obs_dim: int,
    walking_teacher_policy: torch.nn.Module,
    standing_teacher_policy: torch.nn.Module,
    rollout_policy: torch.nn.Module | None = None,
    rollout_policies_by_intent: Mapping[str, torch.nn.Module] | None = None,
    pre_switch_steps: int = 8,
    min_post_switch_steps: int = 0,
    walk_command: np.ndarray | tuple[float, float, float] = (0.4, 0.0, 0.0),
    teacher_obs_key: str = "obs",
    teacher_projection: str = "identity",
    student_projection: str = "identity",
    student_drop_index: int | None = None,
    command_info_key: str = "commands",
    max_env_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DistillationTensorDataset:
    """Collect one opt-in walk-to-stop student-state DAgger scenario.

    Each vectorized row starts with an active walking command, switches once to
    the zero command after ``pre_switch_steps``, and resets its own scenario to
    walking if the environment terminates that row. When
    ``min_post_switch_steps`` is positive, the collector requires enough rows
    to expose that many post-switch ages and fails closed otherwise. The
    rollout is driven by either ``rollout_policy`` or the explicit
    ``rollout_policies_by_intent`` map. The latter uses the walking expert
    before the switch and the standing expert after it; teachers only label
    the observed state.
    """

    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if int(pre_switch_steps) <= 0:
        raise ValueError(f"pre_switch_steps must be positive, got {pre_switch_steps}")
    if int(min_post_switch_steps) < 0:
        raise ValueError(
            f"min_post_switch_steps must be non-negative, got {min_post_switch_steps}"
        )
    action_shape = getattr(getattr(env, "action_space", None), "shape", None)
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined for transition collection")
    num_envs = int(getattr(env, "num_envs"))
    minimum_samples = int(num_envs) * (
        int(pre_switch_steps) + int(min_post_switch_steps)
    )
    if int(min_post_switch_steps) > 0 and int(num_samples) < minimum_samples:
        raise ValueError(
            "transition collection requires enough samples to cover the configured "
            f"post-switch horizon: num_samples={int(num_samples)} "
            f"minimum={minimum_samples} num_envs={int(num_envs)} "
            f"pre_switch_steps={int(pre_switch_steps)} "
            f"min_post_switch_steps={int(min_post_switch_steps)}"
        )
    if rollout_policy is None and rollout_policies_by_intent is None:
        raise ValueError(
            "transition collection requires rollout_policy or "
            "rollout_policies_by_intent"
        )
    if rollout_policy is not None and rollout_policies_by_intent is not None:
        raise ValueError(
            "transition collection accepts only one rollout policy contract"
        )
    if rollout_policies_by_intent is not None:
        missing_intents = {"active", "inactive"} - set(rollout_policies_by_intent)
        if missing_intents:
            raise ValueError(
                "rollout_policies_by_intent is missing intents: "
                f"{sorted(missing_intents)}"
            )
    action_dim = int(action_shape[0])
    walk_command_np = np.asarray(walk_command, dtype=np.float32)
    if walk_command_np.shape != (3,):
        raise ValueError(f"walk_command must have shape (3,), got {walk_command_np.shape}")
    if not np.all(np.isfinite(walk_command_np)):
        raise ValueError("walk_command must contain only finite values")
    if not np.any(command_active_mask(
        walk_command_np.reshape(1, 3),
        xy_threshold=0.05,
        yaw_threshold=0.05,
    )):
        raise ValueError("walk_command must be active under the command thresholds")
    effective_max_env_steps = (
        int(max_env_steps)
        if max_env_steps is not None
        else max(
            int(np.ceil(int(num_samples) / max(num_envs, 1))) * (int(pre_switch_steps) + 16),
            1,
        )
    )
    if effective_max_env_steps < 1:
        raise ValueError(f"max_env_steps must be positive, got {effective_max_env_steps}")

    if getattr(env, "state", None) is None and callable(getattr(env, "init_state", None)):
        env.init_state()
    env_indices = np.arange(num_envs, dtype=np.int32)
    obs, current_info = env.reset(env_indices)
    active_command_rows = np.broadcast_to(walk_command_np, (num_envs, 3)).copy()
    zero_command_rows = np.zeros((num_envs, 3), dtype=np.float32)
    obs, current_info = _set_transition_command_rows(
        env,
        command_info_key=str(command_info_key),
        command_rows=active_command_rows,
    )

    student_chunks: list[torch.Tensor] = []
    teacher_chunks: list[torch.Tensor] = []
    teacher_action_chunks: list[torch.Tensor] = []
    command_chunks: list[torch.Tensor] = []
    command_before_chunks: list[torch.Tensor] = []
    command_after_chunks: list[torch.Tensor] = []
    role_labels: list[str] = []
    command_intents: list[str] = []
    scenario_labels: list[str] = []
    transition_age_chunks: list[torch.Tensor] = []
    post_switch = np.zeros((num_envs,), dtype=np.bool_)
    pre_age = np.zeros((num_envs,), dtype=np.int64)
    transition_ages = np.full((num_envs,), -1, dtype=np.int64)
    collected_count = 0
    env_steps = 0
    switch_count = 0
    post_switch_rows = 0
    done_seen_samples = 0
    action_abs_max = 0.0
    synthetic_teacher_tail = False

    while collected_count < int(num_samples):
        source_np = _obs_array(obs, teacher_obs_key)
        teacher_np, synthetic_tail = project_teacher_obs(
            source_np,
            projection=str(teacher_projection),
            expected_teacher_obs_dim=int(expected_teacher_obs_dim),
        )
        synthetic_teacher_tail = synthetic_teacher_tail or synthetic_tail
        student_np = project_student_obs(
            source_np,
            projection=str(student_projection),
            expected_student_obs_dim=int(expected_student_obs_dim),
            student_drop_index=student_drop_index,
        )
        current_commands = _info_array(
            current_info,
            str(command_info_key),
            expected_rows=num_envs,
        )[:, :3]
        walking_actions = _policy_actions(
            walking_teacher_policy,
            teacher_np,
            action_dim=action_dim,
            policy_name="walking_teacher_policy",
        )
        standing_actions = _policy_actions(
            standing_teacher_policy,
            teacher_np,
            action_dim=action_dim,
            policy_name="standing_teacher_policy",
        )
        teacher_actions = np.where(post_switch[:, None], standing_actions, walking_actions)
        if rollout_policies_by_intent is None:
            rollout_actions = _policy_actions(
                rollout_policy,
                student_np,
                action_dim=action_dim,
                policy_name="rollout_policy",
            )
        else:
            active_actions = _policy_actions(
                rollout_policies_by_intent["active"],
                student_np,
                action_dim=action_dim,
                policy_name="active_rollout_policy",
            )
            inactive_actions = _policy_actions(
                rollout_policies_by_intent["inactive"],
                student_np,
                action_dim=action_dim,
                policy_name="inactive_rollout_policy",
            )
            rollout_actions = np.where(
                post_switch[:, None], inactive_actions, active_actions
            )
        if not np.all(np.isfinite(teacher_actions)) or not np.all(np.isfinite(rollout_actions)):
            raise ValueError("transition collection produced non-finite actions")

        remaining = int(num_samples) - collected_count
        take = min(remaining, num_envs)
        student_chunks.append(
            torch.as_tensor(student_np[:take], dtype=torch.float32).clone()
        )
        teacher_chunks.append(
            torch.as_tensor(teacher_np[:take], dtype=torch.float32).clone()
        )
        teacher_action_chunks.append(
            torch.as_tensor(teacher_actions[:take], dtype=torch.float32).clone()
        )
        command_chunks.append(
            torch.as_tensor(current_commands[:take], dtype=torch.float32).clone()
        )
        command_before_chunks.append(
            torch.as_tensor(active_command_rows[:take], dtype=torch.float32).clone()
        )
        command_after_chunks.append(
            torch.as_tensor(current_commands[:take], dtype=torch.float32).clone()
        )
        transition_age_chunks.append(
            torch.as_tensor(transition_ages[:take], dtype=torch.int64).clone()
        )
        role_labels.extend("stand" if value else "walk_flat" for value in post_switch[:take])
        command_intents.extend("inactive" if value else "active" for value in post_switch[:take])
        scenario_labels.extend("walk_to_stop" for _ in range(take))
        post_switch_rows += int(np.count_nonzero(post_switch[:take]))
        collected_count += take
        action_abs_max = max(action_abs_max, float(np.max(np.abs(rollout_actions))))
        if collected_count >= int(num_samples):
            break
        if env_steps >= effective_max_env_steps:
            raise RuntimeError(
                "transition collection exceeded max_env_steps before reaching "
                f"{num_samples} samples; collected={collected_count}"
            )

        state = env.step(rollout_actions)
        done_mask = _state_done_mask(state, expected_rows=num_envs)
        done_seen_samples += int(np.count_nonzero(done_mask))
        obs, current_info, done_count, _autoreset_count, _manual_reset_count = (
            _reset_done_rows_after_step(env, state, num_envs=num_envs)
        )
        env_steps += 1

        previous_post_switch = post_switch.copy()
        post_switch[done_mask] = False
        pre_age[done_mask] = 0
        transition_ages[done_mask] = -1
        pre_age[(~previous_post_switch) & ~done_mask] += 1
        transition_ages[previous_post_switch & ~done_mask] += 1
        switch_mask = (~post_switch) & ~done_mask & (pre_age >= int(pre_switch_steps))
        post_switch[switch_mask] = True
        transition_ages[switch_mask] = 0
        switch_count += int(np.count_nonzero(switch_mask))

        command_rows = np.broadcast_to(walk_command_np, (num_envs, 3)).copy()
        command_rows[post_switch] = zero_command_rows[post_switch]
        if done_count > 0 or bool(np.any(switch_mask)):
            obs, current_info = _set_transition_command_rows(
                env,
                command_info_key=str(command_info_key),
                command_rows=command_rows,
            )

    if switch_count == 0 or post_switch_rows == 0:
        raise RuntimeError(
            "transition collection did not produce both pre-switch and post-switch rows"
        )
    transition_ages_tensor = torch.cat(transition_age_chunks, dim=0)[: int(num_samples)]
    post_switch_ages = transition_ages_tensor[transition_ages_tensor >= 0]
    max_post_switch_age = (
        int(post_switch_ages.max().item()) if post_switch_ages.numel() else -1
    )
    if (
        int(min_post_switch_steps) > 0
        and max_post_switch_age < int(min_post_switch_steps) - 1
    ):
        raise RuntimeError(
            "transition collection did not reach the configured post-switch horizon: "
            f"max_post_switch_age={max_post_switch_age} "
            f"required={int(min_post_switch_steps) - 1}"
        )
    payload = dict(metadata or {})
    payload.update(
        {
            "source": "live_env_transition_rollout",
            "scenario": "walk_to_stop",
            "teacher_obs_key": str(teacher_obs_key),
            "teacher_projection": str(teacher_projection),
            "student_projection": str(student_projection),
            "student_drop_index": student_drop_index,
            "rollout_policy": (
                "command_intent_experts"
                if rollout_policies_by_intent is not None
                else "distillation_student"
            ),
            "pre_switch_steps": int(pre_switch_steps),
            "min_post_switch_steps": int(min_post_switch_steps),
            "max_post_switch_age": int(max_post_switch_age),
            "walk_command": walk_command_np.tolist(),
            "zero_command": zero_command_rows[0].tolist(),
            "command_info_key": str(command_info_key),
            "env_steps": int(env_steps),
            "switch_count": int(switch_count),
            "post_switch_rows": int(post_switch_rows),
            "done_seen_samples": int(done_seen_samples),
            "action_abs_max": float(action_abs_max),
            "synthetic_teacher_tail": bool(synthetic_teacher_tail),
        }
    )
    return build_distillation_dataset(
        torch.cat(student_chunks, dim=0)[: int(num_samples)],
        torch.cat(teacher_chunks, dim=0)[: int(num_samples)],
        expected_student_obs_dim=int(expected_student_obs_dim),
        expected_teacher_obs_dim=int(expected_teacher_obs_dim),
        expected_teacher_action_dim=action_dim,
        metadata=payload,
        role_labels=tuple(role_labels[: int(num_samples)]),
        teacher_actions=torch.cat(teacher_action_chunks, dim=0)[: int(num_samples)],
        commands=torch.cat(command_chunks, dim=0)[: int(num_samples)],
        command_intents=tuple(command_intents[: int(num_samples)]),
        scenario_labels=tuple(scenario_labels[: int(num_samples)]),
        transition_ages=transition_ages_tensor,
        command_before=torch.cat(command_before_chunks, dim=0)[: int(num_samples)],
        command_after=torch.cat(command_after_chunks, dim=0)[: int(num_samples)],
    )


def collect_distillation_dataset_from_env(
    env: Any,
    *,
    num_samples: int,
    expected_student_obs_dim: int,
    expected_teacher_obs_dim: int,
    teacher_obs_key: str = "obs",
    teacher_projection: str = "identity",
    student_projection: str = "identity",
    student_drop_index: int | None = None,
    action_mode: str = "zero",
    action_seed: int | None = None,
    teacher_policy: torch.nn.Module | None = None,
    rollout_policy: torch.nn.Module | None = None,
    command_sample_filter: str = "none",
    command_info_key: str = "commands",
    command_xy_threshold: float = 0.05,
    command_yaw_threshold: float = 0.05,
    max_env_steps: int | None = None,
    role_label: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DistillationTensorDataset:
    """Collect a small observation-only distillation dataset from a UniLab env."""

    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    command_sample_filter = str(command_sample_filter)
    if command_sample_filter not in {"none", "active", "inactive"}:
        raise ValueError(f"Unsupported command_sample_filter: {command_sample_filter!r}")
    if action_mode not in {"zero", "random", "teacher_policy", "student_policy"}:
        raise ValueError(f"Unsupported collect action_mode: {action_mode!r}")
    if action_mode in {"teacher_policy", "student_policy"} and teacher_policy is None:
        raise ValueError(
            f"teacher_policy is required when action_mode={action_mode!r} "
            "to cache teacher target actions"
        )
    if action_mode not in {"teacher_policy", "student_policy"} and teacher_policy is not None:
        raise ValueError(
            "teacher_policy can only be set when action_mode='teacher_policy' "
            "or action_mode='student_policy'"
        )
    if action_mode == "student_policy" and rollout_policy is None:
        raise ValueError("rollout_policy is required when action_mode='student_policy'")
    if action_mode != "student_policy" and rollout_policy is not None:
        raise ValueError("rollout_policy can only be set when action_mode='student_policy'")
    if action_mode in {"teacher_policy", "student_policy"} and action_seed is not None:
        raise ValueError("action_seed is only supported when action_mode='random'")
    action_shape = getattr(getattr(env, "action_space", None), "shape", None)
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined for distillation collection")
    if max_env_steps is not None and int(max_env_steps) < 0:
        raise ValueError(f"max_env_steps must be non-negative when set, got {max_env_steps}")

    num_envs = int(getattr(env, "num_envs"))
    action_dim = int(action_shape[0])
    effective_max_env_steps = (
        int(max_env_steps)
        if max_env_steps is not None
        else (
            max(int(np.ceil(int(num_samples) / max(num_envs, 1))) * 100, 1)
            if command_sample_filter != "none"
            else None
        )
    )
    rng = np.random.default_rng(None if action_seed is None else int(action_seed))
    action_abs_max = 0.0
    if getattr(env, "state", None) is None and callable(getattr(env, "init_state", None)):
        env.init_state()
    obs, current_info = env.reset(np.arange(num_envs, dtype=np.int32))

    student_chunks: list[torch.Tensor] = []
    teacher_chunks: list[torch.Tensor] = []
    teacher_action_chunks: list[torch.Tensor] = []
    command_chunks: list[torch.Tensor] = []
    command_intent_chunks: list[str] = []
    env_steps = 0
    collected_count = 0
    command_seen_samples = 0
    command_selected_samples = 0
    synthetic_teacher_tail = False
    done_seen_samples = 0
    autoreset_done_count = 0
    manual_done_reset_count = 0

    while collected_count < int(num_samples):
        source_np = _obs_array(obs, teacher_obs_key)
        teacher_np, synthetic_teacher_tail = project_teacher_obs(
            source_np,
            projection=str(teacher_projection),
            expected_teacher_obs_dim=int(expected_teacher_obs_dim),
        )
        student_np = project_student_obs(
            source_np,
            projection=str(student_projection),
            expected_student_obs_dim=int(expected_student_obs_dim),
            student_drop_index=student_drop_index,
        )
        commands_np = (
            _info_array(
                current_info,
                str(command_info_key),
                expected_rows=teacher_np.shape[0],
            )
            if command_sample_filter != "none"
            else None
        )
        command_active = (
            command_active_mask(
                commands_np,
                xy_threshold=float(command_xy_threshold),
                yaw_threshold=float(command_yaw_threshold),
            )
            if commands_np is not None
            else None
        )
        row_mask = _command_sample_mask(
            current_info,
            sample_filter=command_sample_filter,
            command_info_key=str(command_info_key),
            expected_rows=teacher_np.shape[0],
            xy_threshold=float(command_xy_threshold),
            yaw_threshold=float(command_yaw_threshold),
        )
        if row_mask.shape[0] != teacher_np.shape[0]:
            raise ValueError(
                "command sample mask row mismatch: "
                f"expected {teacher_np.shape[0]}, got {row_mask.shape[0]}"
            )
        if command_sample_filter != "none":
            command_seen_samples += int(row_mask.shape[0])
            command_selected_samples += int(np.count_nonzero(row_mask))
        label_actions = None
        if teacher_policy is not None:
            label_actions = _policy_actions(
                teacher_policy,
                teacher_np,
                action_dim=action_dim,
                policy_name="teacher_policy",
            )
            if not np.all(np.isfinite(label_actions)):
                raise ValueError("teacher_policy produced non-finite target actions")
        if action_mode == "teacher_policy":
            actions = label_actions
        elif action_mode == "student_policy":
            actions = _policy_actions(
                rollout_policy,
                student_np,
                action_dim=action_dim,
                policy_name="rollout_policy",
            )
            if not np.all(np.isfinite(actions)):
                raise ValueError("rollout_policy produced non-finite rollout actions")
        else:
            actions = None
        selected_teacher_np = teacher_np[row_mask]
        selected_student_np = student_np[row_mask]
        selected_actions = label_actions[row_mask] if label_actions is not None else None
        selected_commands = commands_np[row_mask] if commands_np is not None else None
        selected_command_active = command_active[row_mask] if command_active is not None else None
        remaining = int(num_samples) - collected_count
        take = min(remaining, selected_teacher_np.shape[0])
        if take > 0:
            teacher_chunks.append(torch.as_tensor(selected_teacher_np[:take], dtype=torch.float32))
            student_chunks.append(torch.as_tensor(selected_student_np[:take], dtype=torch.float32))
            if selected_actions is not None:
                teacher_action_chunks.append(
                    torch.as_tensor(selected_actions[:take], dtype=torch.float32)
                )
            if selected_commands is not None and selected_command_active is not None:
                command_chunks.append(torch.as_tensor(selected_commands[:take], dtype=torch.float32))
                command_intent_chunks.extend(
                    "active" if bool(value) else "inactive"
                    for value in selected_command_active[:take]
                )
            collected_count += int(take)
        if actions is not None:
            action_abs_max = max(action_abs_max, float(np.max(np.abs(actions))))
        if collected_count >= int(num_samples):
            break
        if effective_max_env_steps is not None and env_steps >= effective_max_env_steps:
            raise RuntimeError(
                f"command_sample_filter={command_sample_filter!r} selected "
                f"{command_selected_samples}/{command_seen_samples} samples after "
                f"{env_steps} env steps; increase max_env_steps or relax command thresholds"
            )

        if action_mode == "zero":
            actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        elif action_mode == "random":
            actions = rng.uniform(-1.0, 1.0, size=(num_envs, action_dim)).astype(np.float32)
        if not np.all(np.isfinite(actions)):
            raise ValueError(f"collect action_mode={action_mode!r} produced non-finite actions")
        action_abs_max = max(action_abs_max, float(np.max(np.abs(actions))))
        state = env.step(actions)
        env_steps += 1
        obs, current_info, done_count, autoreset_count, manual_reset_count = (
            _reset_done_rows_after_step(env, state, num_envs=num_envs)
        )
        done_seen_samples += done_count
        autoreset_done_count += autoreset_count
        manual_done_reset_count += manual_reset_count

    payload = dict(metadata or {})
    normalized_role_label = None if role_label in (None, "") else str(role_label)
    if normalized_role_label is not None:
        payload["role_label"] = normalized_role_label
    payload.update(
        {
            "source": "live_env_rollout",
            "teacher_obs_key": str(teacher_obs_key),
            "teacher_projection": str(teacher_projection),
            "student_projection": str(student_projection),
            "student_drop_index": student_drop_index,
            "action_mode": str(action_mode),
            "action_seed": None if action_seed is None else int(action_seed),
            "action_abs_max": float(action_abs_max),
            "num_envs": num_envs,
            "env_steps": env_steps,
            "done_seen_samples": int(done_seen_samples),
            "autoreset_done_count": int(autoreset_done_count),
            "manual_done_reset_count": int(manual_done_reset_count),
            "synthetic_teacher_tail": bool(synthetic_teacher_tail),
        }
    )
    if command_sample_filter != "none":
        payload.update(
            {
                "command_sample_filter": command_sample_filter,
                "command_info_key": str(command_info_key),
                "command_xy_threshold": float(command_xy_threshold),
                "command_yaw_threshold": float(command_yaw_threshold),
                "command_seen_samples": int(command_seen_samples),
                "command_selected_samples": int(command_selected_samples),
                "max_env_steps": effective_max_env_steps,
            }
        )
    if action_mode == "student_policy":
        payload["rollout_policy"] = "distillation_student"
    return build_distillation_dataset(
        torch.cat(student_chunks, dim=0),
        torch.cat(teacher_chunks, dim=0),
        expected_student_obs_dim=int(expected_student_obs_dim),
        expected_teacher_obs_dim=int(expected_teacher_obs_dim),
        expected_teacher_action_dim=action_dim,
        metadata=payload,
        teacher_actions=(
            torch.cat(teacher_action_chunks, dim=0) if teacher_action_chunks else None
        ),
        commands=torch.cat(command_chunks, dim=0) if command_chunks else None,
        command_intents=tuple(command_intent_chunks) if command_intent_chunks else None,
        role_labels=(normalized_role_label,) * int(num_samples)
        if normalized_role_label is not None
        else None,
    )
