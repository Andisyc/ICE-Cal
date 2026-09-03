from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import torch

from unilab.algos.torch.distill.collection.common import project_student_obs
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.observation import assert_fada_projection_matches_contract
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.algos.torch.distill.fada.target_data import FADATargetBatch
from unilab.algos.torch.distill.fada.target_domain import FADASlopeGeometry
from unilab.algos.torch.distill.fada.target_rollout import (
    apply_external_command,
    rollout_done_flags,
    rollout_terminal_reasons,
    scheduled_target_command,
)
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
    accepted_steps: int = 0
    episode_count: int = 1
    rejected_pre_entry_steps: int = 0
    termination_counts: Mapping[str, int] | None = None
    representative_physics_states: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class FADATargetStepDecision:
    accept: bool
    terminal_reason: str | None


class FADATargetEpisodePolicy(Protocol):
    def command_for_episode(self, episode_id: int) -> np.ndarray: ...

    def classify(
        self,
        *,
        base_pos_w: np.ndarray,
        feet_pos_w: np.ndarray,
        done: bool,
    ) -> FADATargetStepDecision: ...


class FADASlopeEpisodePolicy:
    """Own command cycling and ramp-local episode acceptance for Stage C."""

    def __init__(
        self,
        geometry: FADASlopeGeometry,
        command_sequence: Sequence[Sequence[float]],
    ) -> None:
        commands = tuple(np.asarray(command, dtype=np.float32) for command in command_sequence)
        if not commands or any(command.shape != (3,) for command in commands):
            raise ValueError("FADA slope command sequence must contain 3-D commands")
        self.geometry = geometry
        self.commands = commands

    def command_for_episode(self, episode_id: int) -> np.ndarray:
        if episode_id < 0:
            raise ValueError("FADA slope episode_id must be non-negative")
        return self.commands[episode_id % len(self.commands)].copy()

    def classify(
        self,
        *,
        base_pos_w: np.ndarray,
        feet_pos_w: np.ndarray,
        done: bool,
    ) -> FADATargetStepDecision:
        if done:
            return FADATargetStepDecision(False, "fall")
        if self.geometry.foot_exited(feet_pos_w):
            return FADATargetStepDecision(False, "foot_exit")
        if self.geometry.has_finished(base_pos_w):
            return FADATargetStepDecision(False, "finish")
        return FADATargetStepDecision(
            self.geometry.has_entered(base_pos_w, feet_pos_w),
            None,
        )


def fada_target_window_budget(
    config: FADAArchitectureConfig,
    spec: FADATargetCollectionSpec,
    *,
    control_steps: int,
) -> int:
    """Derive usable steady-state windows from one executed control-step budget."""

    if isinstance(control_steps, bool) or not isinstance(control_steps, int) or control_steps <= 0:
        raise ValueError(f"control_steps must be a positive integer, got {control_steps!r}")
    if spec.ramp_steps < 0 or spec.settle_steps < 0:
        raise ValueError("FADA target ramp_steps and settle_steps must be non-negative")
    record_count = config.history_length + config.prediction_horizon - 1
    collection_start = int(spec.ramp_steps) + int(spec.settle_steps)
    first_usable_step = max(
        record_count,
        collection_start + config.prediction_horizon,
    )
    windows = control_steps - first_usable_step + 1
    if windows <= 0:
        raise ValueError(
            "control_steps cannot produce a usable window: "
            f"control_steps={control_steps} first_usable_step={first_usable_step}"
        )
    return windows


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
    apply_external_command(env, command)


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
    terminated, truncated = rollout_done_flags(state, num_envs=num_envs)
    return terminated | truncated


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


def _single_body_state(env: Any) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(env.get_base_pos(), dtype=np.float64)
    feet = np.asarray(env.get_foot_pos(), dtype=np.float64)
    if base.shape != (1, 3) or feet.shape != (1, 2, 3):
        raise ValueError(
            "FADA slope collection requires base (1, 3) and feet (1, 2, 3), "
            f"got base={base.shape} feet={feet.shape}"
        )
    return base[0].copy(), feet[0].copy()


def _slope_command(
    spec: FADATargetCollectionSpec,
    target: np.ndarray,
    episode_step: int,
) -> np.ndarray:
    if spec.command_start is None:
        raise ValueError("FADA slope command_start must be configured")
    return scheduled_target_command(
        spec.command_start,
        target,
        ramp_steps=spec.ramp_steps,
        step=episode_step,
    )


def collect_fada_slope_windows(
    env: Any,
    rollout_policy: FADAPlannerIDMPolicy,
    config: FADAArchitectureConfig,
    accepted_control_steps: int,
    episode_policy: FADATargetEpisodePolicy,
    spec: FADATargetCollectionSpec,
) -> FADATargetCollectionResult:
    """Collect target-only slope windows across isolated, explicit episodes."""

    if int(accepted_control_steps) <= 0:
        raise ValueError(f"accepted_control_steps must be positive, got {accepted_control_steps}")
    if rollout_policy.config != config:
        raise ValueError("rollout policy architecture must match target collection config")
    if int(env.num_envs) != 1:
        raise ValueError("FADA slope collection requires num_envs=1")
    if spec.max_env_steps is None or int(spec.max_env_steps) <= 0:
        raise ValueError("FADA slope collection requires a positive max_env_steps")
    if spec.command_start is None:
        raise ValueError("FADA slope collection requires command_start")
    assert_fada_projection_matches_contract(
        observation_contract=config.observation_contract,
        projection=spec.student_projection,
    )
    set_autoreset = getattr(env, "set_autoreset", None)
    if not callable(set_autoreset):
        raise RuntimeError("FADA slope collection requires env.set_autoreset()")
    set_autoreset(False)

    controller = FADAPlaybackController(rollout_policy, device=_module_device(rollout_policy))
    record_count = config.history_length + config.prediction_horizon - 1
    records: deque[FADACausalTransition] = deque(maxlen=record_count)
    previous_action = np.zeros((config.action_dim,), dtype=np.float32)
    batches: list[FADATargetBatch] = []
    termination_counts: dict[str, int] = {
        "fall": 0,
        "environment_termination": 0,
        "truncated": 0,
        "foot_exit": 0,
        "finish": 0,
    }
    representative: tuple[np.ndarray, ...] = ()
    episode_states: list[np.ndarray] = []
    episode_accepted = 0
    episode_id = 0
    episode_step = 0
    env_steps = 0
    accepted_steps = 0
    rejected_pre_entry = 0
    rejected_command = 0

    def reset_episode() -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
        nonlocal previous_action, episode_step, episode_states, episode_accepted
        state = env.reset_all()
        controller.reset()
        records.clear()
        previous_action = np.zeros((config.action_dim,), dtype=np.float32)
        episode_step = 0
        episode_accepted = 0
        episode_states = [np.asarray(env.get_physics_state_snapshot()).copy()]
        obs = {key: np.asarray(value).copy() for key, value in state.obs.items()}
        info = dict(state.info)
        projected = project_student_obs(
            _observation_array(obs, spec.observation_key),
            projection=spec.student_projection,
            expected_student_obs_dim=config.obs_dim,
            student_drop_index=spec.student_drop_index,
        )
        return obs, info, projected

    obs, info, student_obs = reset_episode()
    while accepted_steps < int(accepted_control_steps) and env_steps < int(spec.max_env_steps):
        target = episode_policy.command_for_episode(episode_id)
        command_now = _slope_command(spec, target, episode_step)
        _apply_external_command(env, command_now)
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
            expected_rows=1,
            expected_dim=config.command_dim,
        )
        base_before, feet_before = _single_body_state(env)
        before = episode_policy.classify(
            base_pos_w=base_before,
            feet_pos_w=feet_before,
            done=False,
        )
        if before.terminal_reason is not None:
            termination_counts[before.terminal_reason] = (
                termination_counts.get(before.terminal_reason, 0) + 1
            )
            if episode_accepted > 0 and len(episode_states) > len(representative):
                representative = tuple(episode_states)
            episode_id += 1
            obs, info, student_obs = reset_episode()
            continue

        actions = (
            controller.act_projected(student_obs, command).detach().cpu().numpy().astype(np.float32)
        )
        next_state = env.step(actions)
        env_steps += 1
        episode_step += 1
        episode_states.append(np.asarray(env.get_physics_state_snapshot()).copy())
        base_after, feet_after = _single_body_state(env)
        after = episode_policy.classify(
            base_pos_w=base_after,
            feet_pos_w=feet_after,
            done=False,
        )
        lifecycle_reason = rollout_terminal_reasons(next_state, num_envs=1)[0]
        if lifecycle_reason is not None:
            after = FADATargetStepDecision(False, lifecycle_reason)
        next_obs = {key: np.asarray(value).copy() for key, value in next_state.obs.items()}
        next_info = dict(next_state.info)
        next_student_obs = project_student_obs(
            _observation_array(next_obs, spec.observation_key),
            projection=spec.student_projection,
            expected_student_obs_dim=config.obs_dim,
            student_drop_index=spec.student_drop_index,
        )

        collection_ready = episode_step - 1 >= int(spec.ramp_steps) + int(spec.settle_steps)
        if before.accept and collection_ready and after.terminal_reason is None:
            records.append(
                FADACausalTransition(
                    observation=student_obs[0].copy(),
                    previous_action=previous_action.copy(),
                    command=command[0].copy(),
                    executed_action=actions[0].copy(),
                    next_observation=next_student_obs[0].copy(),
                    episode_id=episode_id,
                    timestep=episode_step - 1,
                )
            )
            accepted_steps += 1
            episode_accepted += 1
            if len(records) == record_count:
                window = _target_batch_from_window(tuple(records), config)
                if window is None:
                    rejected_command += 1
                else:
                    batches.append(window)
        elif not before.accept or not collection_ready:
            rejected_pre_entry += 1
            records.clear()

        previous_action = actions[0].copy()
        obs, info, student_obs = next_obs, next_info, next_student_obs
        if after.terminal_reason is not None:
            termination_counts[after.terminal_reason] = (
                termination_counts.get(after.terminal_reason, 0) + 1
            )
            if episode_accepted > 0 and len(episode_states) > len(representative):
                representative = tuple(episode_states)
            episode_id += 1
            obs, info, student_obs = reset_episode()

    if accepted_steps < int(accepted_control_steps):
        raise RuntimeError(
            f"FADA slope collector accepted {accepted_steps}/{accepted_control_steps} steps after "
            f"{env_steps} env steps; accepted_steps={accepted_steps} episodes={episode_id + 1} "
            f"terminations={termination_counts}"
        )
    if not batches:
        raise RuntimeError("FADA slope collector produced no complete causal windows")
    if episode_accepted > 0 and len(episode_states) > len(representative):
        representative = tuple(episode_states)
    return FADATargetCollectionResult(
        batch=_concat_target_batches(batches, config),
        env_steps=env_steps,
        rejected_done_transitions=(
            termination_counts["fall"]
            + termination_counts["environment_termination"]
            + termination_counts["truncated"]
        ),
        rejected_command_windows=rejected_command,
        accepted_steps=accepted_steps,
        episode_count=episode_id + 1,
        rejected_pre_entry_steps=rejected_pre_entry,
        termination_counts=termination_counts,
        representative_physics_states=representative,
    )
