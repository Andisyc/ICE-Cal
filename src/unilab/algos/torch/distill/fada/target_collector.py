from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from unilab.algos.torch.distill.collection.common import project_student_obs
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.observation import assert_fada_projection_matches_contract
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.algos.torch.distill.fada.target_data import FADATargetBatch
from unilab.algos.torch.distill.fada.windows import FADACausalTransition, build_fada_causal_window


@dataclass(frozen=True)
class FADATargetCollectionSpec:
    """Runtime-only inputs for Oracle-free Stage-C target collection."""

    observation_key: str = "obs"
    student_projection: str = "identity"
    student_drop_index: int | None = None
    command_info_keys: tuple[str, ...] = ("commands",)
    max_env_steps: int | None = None
    command_start: tuple[float, float, float] | None = None
    command_target: tuple[float, float, float] | None = None
    ramp_steps: int = 0
    settle_steps: int = 0
    single_trajectory: bool = False
    capture_initial_frame: Callable[[], None] | None = None
    capture_frame: Callable[[], None] | None = None


@dataclass(frozen=True)
class FADATargetCollectionResult:
    batch: FADATargetBatch
    env_steps: int
    rejected_done_transitions: int
    rejected_command_windows: int


def _scheduled_command(spec: FADATargetCollectionSpec, step: int) -> np.ndarray:
    if spec.command_start is None or spec.command_target is None:
        raise ValueError("FADA target command schedule is not configured")
    start = np.asarray(spec.command_start, dtype=np.float32)
    target = np.asarray(spec.command_target, dtype=np.float32)
    if start.shape != (3,) or target.shape != (3,):
        raise ValueError("FADA target command schedule requires two 3-D commands")
    if spec.ramp_steps < 0 or spec.settle_steps < 0:
        raise ValueError("FADA target ramp_steps and settle_steps must be non-negative")
    if spec.ramp_steps == 0 or step >= spec.ramp_steps:
        return target
    return start + (target - start) * (float(step + 1) / float(spec.ramp_steps))


def _apply_external_command(env: Any, command: np.ndarray) -> None:
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


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _observation_array(obs: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"observation key {key!r} not found; available={sorted(obs)}")
    value = np.asarray(obs[key], dtype=np.float32)
    if value.ndim != 2 or not bool(np.all(np.isfinite(value))):
        raise ValueError(f"observation {key!r} must be finite rank-2, got {value.shape}")
    return value


def _command_array(
    info: Mapping[str, Any],
    keys: Sequence[str],
    *,
    expected_rows: int,
    expected_dim: int,
) -> np.ndarray:
    if not keys:
        raise ValueError("command_info_keys must not be empty")
    columns: list[np.ndarray] = []
    for key in keys:
        if key not in info:
            raise KeyError(f"command info key {key!r} not found; available={sorted(info)}")
        value = np.asarray(info[key], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[0] != int(expected_rows):
            raise ValueError(
                f"command info key {key!r} must have {expected_rows} rows, got {value.shape}"
            )
        columns.append(value)
    command = np.concatenate(columns, axis=1)
    expected_shape = (int(expected_rows), int(expected_dim))
    if command.shape != expected_shape:
        raise ValueError(
            f"complete command shape mismatch: expected {expected_shape}, got {command.shape} "
            f"from keys={list(keys)}"
        )
    if not bool(np.all(np.isfinite(command))):
        raise ValueError("complete command must contain only finite values")
    return command


def _done_mask(state: Any, *, num_envs: int) -> np.ndarray:
    done = np.zeros((int(num_envs),), dtype=np.bool_)
    for value in (getattr(state, "terminated", None), getattr(state, "truncated", None)):
        if value is None:
            continue
        current = np.asarray(value, dtype=np.bool_).reshape(-1)
        if current.shape != done.shape:
            raise ValueError(
                f"done mask shape mismatch: expected {done.shape}, got {current.shape}"
            )
        done |= current
    return done


def _concat_target_batches(
    batches: Sequence[FADATargetBatch], config: FADAArchitectureConfig
) -> FADATargetBatch:
    return FADATargetBatch(
        **{
            field: torch.cat([getattr(batch, field) for batch in batches], dim=0)
            for field in FADATargetBatch.__dataclass_fields__
        }
    ).validate(config)


def _target_batch_from_window(
    records: Sequence[FADACausalTransition], config: FADAArchitectureConfig
) -> FADATargetBatch | None:
    window = build_fada_causal_window(
        records,
        history_length=config.history_length,
        prediction_horizon=config.prediction_horizon,
        command_scenario="walk",
    )
    if window is None:
        return None
    return FADATargetBatch(
        observation_history=torch.from_numpy(window.observation_history[None]),
        action_history=torch.from_numpy(window.action_history[None]),
        command=torch.from_numpy(window.command[None]),
        realized_future=torch.from_numpy(window.realized_future[None]),
        executed_action_chunk=torch.from_numpy(window.executed_action_chunk[None]),
        episode_id=torch.tensor([window.episode_id], dtype=torch.int64),
        start_timestep=torch.tensor([window.start_timestep], dtype=torch.int64),
    ).validate(config)


def collect_fada_target_windows(
    env: Any,
    rollout_policy: FADAPlannerIDMPolicy,
    config: FADAArchitectureConfig,
    num_windows: int,
    spec: FADATargetCollectionSpec | None = None,
) -> FADATargetCollectionResult:
    """Collect executed target-domain windows without Oracle or training authority."""

    spec = FADATargetCollectionSpec() if spec is None else spec
    if int(num_windows) <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")
    if rollout_policy.config != config:
        raise ValueError("rollout policy architecture must match target collection config")
    assert_fada_projection_matches_contract(
        observation_contract=config.observation_contract,
        projection=spec.student_projection,
    )

    num_envs = int(env.num_envs)
    if num_envs <= 0:
        raise ValueError(f"env.num_envs must be positive, got {num_envs}")
    if spec.single_trajectory and num_envs != 1:
        raise ValueError("FADA target single-trajectory collection requires exactly one env")
    initial_state = env.reset_all()
    if spec.capture_initial_frame is not None:
        spec.capture_initial_frame()
    obs = {
        key: np.asarray(value).copy()
        for key, value in cast(Mapping[str, Any], initial_state.obs).items()
    }
    info = dict(cast(Mapping[str, Any], initial_state.info))
    student_obs = project_student_obs(
        _observation_array(obs, spec.observation_key),
        projection=spec.student_projection,
        expected_student_obs_dim=config.obs_dim,
        student_drop_index=spec.student_drop_index,
    )

    controller = FADAPlaybackController(rollout_policy, device=_module_device(rollout_policy))
    previous_actions = np.zeros((num_envs, config.action_dim), dtype=np.float32)
    record_count = config.history_length + config.prediction_horizon - 1
    records: list[deque[FADACausalTransition]] = [
        deque(maxlen=record_count) for _ in range(num_envs)
    ]
    episode_ids = np.zeros((num_envs,), dtype=np.int64)
    episode_timesteps = np.zeros((num_envs,), dtype=np.int64)
    batches: list[FADATargetBatch] = []
    rejected_done = 0
    rejected_command = 0
    env_steps = 0
    collection_start = int(spec.ramp_steps) + int(spec.settle_steps)
    step_limit = (
        int(spec.max_env_steps)
        if spec.max_env_steps is not None
        else max(record_count + int(np.ceil(num_windows / num_envs)) * 10, 1)
    )
    if step_limit <= 0:
        raise ValueError(f"max_env_steps must be positive, got {step_limit}")

    while len(batches) < int(num_windows):
        if env_steps >= step_limit:
            raise RuntimeError(
                f"FADA target collector produced {len(batches)}/{num_windows} windows after "
                f"{env_steps} env steps; increase max_env_steps or inspect episode/command resets"
            )
        if spec.command_start is not None or spec.command_target is not None:
            scheduled_command = _scheduled_command(spec, env_steps)
            _apply_external_command(env, scheduled_command)
            obs = {key: np.asarray(value).copy() for key, value in env.state.obs.items()}
            info = dict(env.state.info)
            student_obs = project_student_obs(
                _observation_array(obs, spec.observation_key),
                projection=spec.student_projection,
                expected_student_obs_dim=config.obs_dim,
                student_drop_index=spec.student_drop_index,
            )
        command = _command_array(
            info,
            spec.command_info_keys,
            expected_rows=num_envs,
            expected_dim=config.command_dim,
        )
        action_tensor = controller.act_projected(student_obs, command)
        actions = action_tensor.detach().cpu().numpy().astype(np.float32)
        state = env.step(actions)
        env_steps += 1
        if spec.capture_frame is not None:
            spec.capture_frame()
        done = _done_mask(state, num_envs=num_envs)
        next_obs = {
            key: np.asarray(value).copy()
            for key, value in cast(Mapping[str, Any], state.obs).items()
        }
        next_info = dict(cast(Mapping[str, Any], state.info))
        next_student_obs = project_student_obs(
            _observation_array(next_obs, spec.observation_key),
            projection=spec.student_projection,
            expected_student_obs_dim=config.obs_dim,
            student_drop_index=spec.student_drop_index,
        )

        trajectory_ended = False
        for index in range(num_envs):
            if bool(done[index]):
                records[index].clear()
                episode_timesteps[index] = 0
                rejected_done += 1
                if spec.single_trajectory:
                    trajectory_ended = True
                else:
                    episode_ids[index] += 1
                continue
            records[index].append(
                FADACausalTransition(
                    observation=student_obs[index].copy(),
                    previous_action=previous_actions[index].copy(),
                    command=command[index].copy(),
                    executed_action=actions[index].copy(),
                    next_observation=next_student_obs[index].copy(),
                    episode_id=int(episode_ids[index]),
                    timestep=int(episode_timesteps[index]),
                )
            )
            episode_timesteps[index] += 1
            if len(records[index]) == record_count:
                window = _target_batch_from_window(tuple(records[index]), config)
                if window is None:
                    rejected_command += 1
                elif window.start_timestep >= collection_start:
                    batches.append(window)
                    if len(batches) >= int(num_windows):
                        break

        previous_actions = actions.copy()
        if bool(np.any(done)):
            previous_actions[done] = 0.0
            controller.reset(done)
        obs, info, student_obs = next_obs, next_info, next_student_obs
        if trajectory_ended:
            if not batches:
                raise RuntimeError(
                    "FADA target single trajectory ended before producing a usable window "
                    f"after {env_steps} env steps"
                )
            break

    return FADATargetCollectionResult(
        batch=_concat_target_batches(batches[: int(num_windows)], config),
        env_steps=env_steps,
        rejected_done_transitions=rejected_done,
        rejected_command_windows=rejected_command,
    )
