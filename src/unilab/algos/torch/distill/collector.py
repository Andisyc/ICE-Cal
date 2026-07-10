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
        obs = state.obs
        current_info = state.info

    payload = dict(metadata or {})
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
    )
