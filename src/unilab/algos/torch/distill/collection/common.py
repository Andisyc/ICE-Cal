"""Shared stateless projection, inference, reset, and performance collection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.datasets.dataset import DistillationTensorDataset
from unilab.algos.torch.distill.fada.observation import (
    FADA_G1_STATE_OBSERVATION_CONTRACT,
    project_fada_g1_state,
)
from unilab.algos.torch.distill.observability.performance import (
    DISTILLATION_METRICS_SCHEMA_VERSION,
    DistillationStageObservationAccumulator,
)


def _performance_span(
    accumulator: DistillationStageObservationAccumulator | None,
    stage: str,
):
    return nullcontext() if accumulator is None else accumulator.measure(stage)


def _attach_collector_performance(
    dataset: DistillationTensorDataset,
    *,
    accumulator: DistillationStageObservationAccumulator | None,
    teacher_inference_rows: int,
    student_inference_rows: int,
    env_steps: int,
) -> DistillationTensorDataset:
    if accumulator is None:
        return dataset
    observations = (
        accumulator.observation(
            stage="teacher_inference",
            row_count=teacher_inference_rows,
            env_step_count=0,
        ),
        accumulator.observation(
            stage="student_inference",
            row_count=student_inference_rows,
            env_step_count=0,
        ),
        accumulator.observation(
            stage="env_step",
            row_count=0,
            env_step_count=env_steps,
        ),
        accumulator.observation(
            stage="tensor_pack",
            row_count=dataset.num_samples,
            env_step_count=0,
        ),
    )
    return replace(
        dataset,
        metadata={
            **dataset.metadata,
            "performance_metrics_schema_version": DISTILLATION_METRICS_SCHEMA_VERSION,
            "performance_stage_observations": [
                observation.as_dict() for observation in observations
            ],
        },
    )


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


def _target_height_array(
    info: Mapping[str, Any],
    key: str,
    *,
    expected_rows: int,
) -> np.ndarray:
    target_height = _info_array(info, key, expected_rows=expected_rows)
    if target_height.shape[1] != 1:
        raise ValueError(f"Info key {key!r} must have shape (N, 1), got {target_height.shape}")
    return target_height


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
    elif projection == FADA_G1_STATE_OBSERVATION_CONTRACT:
        if student_drop_index is not None:
            raise ValueError("g1_fada_state_v2 does not accept student_drop_index")
        student_obs = project_fada_g1_state(source_obs)
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


def _resolve_collection_reset(
    env: Any,
    *,
    num_envs: int,
    initial_reset: tuple[Any, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if initial_reset is None:
        reset_output = env.reset(np.arange(num_envs, dtype=np.int32))
    else:
        reset_output = initial_reset
    if not isinstance(reset_output, tuple) or len(reset_output) != 2:
        raise ValueError("collection reset must be a two-item (obs_dict, info_dict) tuple")
    obs, info = reset_output
    if not isinstance(obs, dict) or not isinstance(info, Mapping):
        raise ValueError("collection reset must contain dict obs and mapping info")
    return obs, dict(info)
