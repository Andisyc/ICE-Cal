"""Identity-bound closed-loop diagnosis for FADA source coverage.

This module observes the existing v007 collection semantics. It does not create
training rows, mutate replay, or authorize a collector repair.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import torch

from unilab.algos.torch.distill.collection.common import (
    project_student_obs,
    project_teacher_obs,
)
from unilab.algos.torch.distill.collection.transition import set_transition_input_rows
from unilab.algos.torch.distill.fada.collection_io import _oracle_actions
from unilab.algos.torch.distill.fada.collection_io import _policy_actions as _policy_actions
from unilab.algos.torch.distill.fada.collector import (
    FADACollectionSpec,
    _command_array,
    _done_mask,
    _fada_actions,
    _obs_array,
    _oracle_shadow_pair,
)
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.observation import assert_fada_projection_matches_contract

FADACoverageVerdict = Literal[
    "COVERAGE_GAP",
    "COVERAGE_CAUSE_REJECTED",
    "IDENTITY_OR_MEASUREMENT_CONFLICT",
]

_RIGHT_LEG_ACTION_INDICES = (6, 7, 8, 9, 10, 11)


@dataclass(frozen=True)
class FADACoverageStep:
    episode_id: int
    timestep: int
    observation_history: tuple[tuple[float, ...], ...]
    action_history: tuple[tuple[float, ...], ...]
    command: tuple[float, ...]
    student_first_action: tuple[float, ...]
    student_predicted_future: tuple[tuple[float, ...], ...]
    student_action_chunk: tuple[tuple[float, ...], ...]
    oracle_first_action: tuple[float, ...]
    oracle_future: tuple[tuple[float, ...], ...]
    oracle_action_chunk: tuple[tuple[float, ...], ...]
    oracle_shadow_valid: bool
    snapshot_owner_transaction_completed: bool
    snapshot_observable_restoration_valid: bool
    command_identity_valid: bool
    terminated: bool
    truncated: bool
    termination_reason: str | None
    v007_planner_row_persisted: bool = False
    v007_rejection_reason: str | None = None
    base_position: tuple[float, ...] | None = None
    base_quaternion: tuple[float, ...] | None = None
    base_height: float | None = None
    planar_displacement: tuple[float, float] | None = None
    gait_phase: tuple[float, ...] | None = None
    left_foot_contact: bool | None = None
    right_foot_contact: bool | None = None
    per_joint_action_error: tuple[float, ...] = ()
    per_joint_action_rate: tuple[float, ...] = ()
    right_leg_action_error: tuple[float, ...] = ()
    right_leg_action_rate: tuple[float, ...] = ()


@dataclass(frozen=True)
class FADACoverageReport:
    verdict: FADACoverageVerdict
    failure_reproduced: bool
    identity_valid: bool
    history_length: int
    prediction_horizon: int
    command: tuple[float, ...]
    stop_reason: str
    coverage_gap_step_indices: tuple[int, ...]
    steps: tuple[FADACoverageStep, ...]


def _tuple_rows(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(value, dtype=np.float32)
    return tuple(tuple(float(item) for item in row) for row in array)


def _optional_row(value: Any, *, index: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or index >= array.shape[0] or not bool(np.all(np.isfinite(array[index]))):
        return None
    return tuple(float(item) for item in array[index])


def _base_row(env: Any, name: str, *, index: int) -> tuple[float, ...] | None:
    getter = getattr(env, name, None)
    return _optional_row(getter() if callable(getter) else None, index=index)


def _info_row(info: Mapping[str, Any], key: str, *, index: int) -> tuple[float, ...] | None:
    return _optional_row(info.get(key), index=index)


def _termination_reason(state: Any, *, index: int) -> str | None:
    info = cast(Mapping[str, Any], getattr(state, "info", {}))
    for key in ("termination_reason", "done_reason", "reset_reason", "truncation_reason"):
        value = info.get(key)
        if value is None:
            continue
        if isinstance(value, np.ndarray) and value.shape and index < value.shape[0]:
            return str(value[index])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if index < len(value):
                return str(value[index])
        return str(value)
    return None


def _visible_snapshot_matches(
    env: Any,
    *,
    source: np.ndarray,
    command: np.ndarray,
    spec: FADACollectionSpec,
) -> bool:
    state = getattr(env, "state", None)
    if state is None:
        return False
    try:
        restored_source = _obs_array(cast(Mapping[str, Any], state.obs), spec.observation_key)
        restored_command = _command_array(
            cast(Mapping[str, Any], state.info),
            spec.command_info_keys,
            expected_rows=int(env.num_envs),
            expected_dim=command.shape[1],
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return bool(
        np.array_equal(restored_source, source) and np.array_equal(restored_command, command)
    )


def _mark_v007_persistence(
    steps: Sequence[FADACoverageStep],
    *,
    history_length: int,
    prediction_horizon: int,
) -> tuple[FADACoverageStep, ...]:
    marked: list[FADACoverageStep] = []
    for index, step in enumerate(steps):
        future = steps[index : index + int(prediction_horizon)]
        same_episode = len(future) == int(prediction_horizon) and all(
            item.episode_id == step.episode_id for item in future
        )
        no_done = same_episode and all(not (item.terminated or item.truncated) for item in future)
        recovery_prefix = step.timestep < int(history_length)
        durable = bool(
            (recovery_prefix and step.oracle_shadow_valid)
            or (step.timestep >= int(history_length) - 1 and no_done)
        )
        rejection_reason: str | None = None
        if not durable:
            if recovery_prefix:
                rejection_reason = "walking_recovery_oracle_shadow_invalid"
            elif not same_episode or any(item.terminated or item.truncated for item in future):
                rejection_reason = "episode_terminated_before_window_completion"
            else:
                rejection_reason = "bounded_trace_ended_before_window_completion"
        marked.append(
            replace(
                step,
                v007_planner_row_persisted=durable,
                v007_rejection_reason=rejection_reason,
            )
        )
    return tuple(marked)


def classify_fada_coverage(
    steps: Sequence[FADACoverageStep],
    *,
    history_length: int,
    prediction_horizon: int,
    command: Sequence[float],
    stop_reason: str,
) -> FADACoverageReport:
    marked = _mark_v007_persistence(
        steps,
        history_length=history_length,
        prediction_horizon=prediction_horizon,
    )
    reproduced = any(step.terminated for step in marked)
    identity_valid = bool(marked) and all(
        step.snapshot_owner_transaction_completed
        and step.snapshot_observable_restoration_valid
        and step.command_identity_valid
        for step in marked
    )
    gap_indices = tuple(
        index
        for index, step in enumerate(marked)
        if step.timestep >= int(history_length)
        and step.oracle_shadow_valid
        and step.snapshot_owner_transaction_completed
        and step.snapshot_observable_restoration_valid
        and step.command_identity_valid
        and not step.v007_planner_row_persisted
        and step.v007_rejection_reason == "episode_terminated_before_window_completion"
    )
    if not reproduced or not identity_valid:
        verdict: FADACoverageVerdict = "IDENTITY_OR_MEASUREMENT_CONFLICT"
    elif gap_indices:
        verdict = "COVERAGE_GAP"
    else:
        verdict = "COVERAGE_CAUSE_REJECTED"
    return FADACoverageReport(
        verdict=verdict,
        failure_reproduced=reproduced,
        identity_valid=identity_valid,
        history_length=int(history_length),
        prediction_horizon=int(prediction_horizon),
        command=tuple(float(item) for item in command),
        stop_reason=str(stop_reason),
        coverage_gap_step_indices=gap_indices,
        steps=marked,
    )


def run_fada_coverage_diagnostic(
    env: Any,
    *,
    student_policy: FADAPlannerIDMPolicy,
    teacher_policy: torch.nn.Module,
    config: FADAArchitectureConfig,
    command: Sequence[float] = (0.4, 0.0, 0.0),
    max_steps: int = 500,
    spec: FADACollectionSpec | None = None,
) -> FADACoverageReport:
    """Run one bounded student episode with same-snapshot final-Oracle queries."""

    spec = FADACollectionSpec(collect_oracle_shadow=True) if spec is None else spec
    if not spec.collect_oracle_shadow:
        raise ValueError("coverage diagnostic requires collect_oracle_shadow=true")
    if int(env.num_envs) != 1:
        raise ValueError("coverage diagnostic requires exactly one environment")
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    if int(max_steps) > 500:
        raise ValueError("coverage diagnostic max_steps must not exceed 500")
    if student_policy.config != config:
        raise ValueError("student checkpoint architecture does not match diagnostic config")
    assert_fada_projection_matches_contract(
        observation_contract=config.observation_contract,
        projection=spec.student_projection,
    )
    command_array = np.asarray(command, dtype=np.float32)
    if command_array.shape != (config.command_dim,) or not bool(np.all(np.isfinite(command_array))):
        raise ValueError(f"command must be finite shape ({config.command_dim},)")

    initial = env.reset_all()
    obs = {key: np.asarray(value).copy() for key, value in initial.obs.items()}
    info = dict(initial.info)
    obs, info = set_transition_input_rows(
        env,
        command_info_key="commands",
        command_rows=command_array[None],
    )
    source = _obs_array(obs, spec.observation_key)
    student_obs = project_student_obs(
        source,
        projection=spec.student_projection,
        expected_student_obs_dim=config.obs_dim,
        student_drop_index=spec.student_drop_index,
    )
    observation_history = np.repeat(student_obs[:, None, :], config.history_length, axis=1)
    action_history = np.zeros((1, config.history_length, config.action_dim), dtype=np.float32)
    initial_base = _base_row(env, "get_base_pos", index=0)
    trace: list[FADACoverageStep] = []
    stop_reason = "max_steps"

    for timestep in range(int(max_steps)):
        source = _obs_array(obs, spec.observation_key)
        student_obs = project_student_obs(
            source,
            projection=spec.student_projection,
            expected_student_obs_dim=config.obs_dim,
            student_drop_index=spec.student_drop_index,
        )
        complete_command = _command_array(
            info,
            spec.command_info_keys,
            expected_rows=1,
            expected_dim=config.command_dim,
        )
        command_identity_valid = bool(
            np.array_equal(complete_command, np.broadcast_to(command_array, complete_command.shape))
        )
        teacher_obs, _ = project_teacher_obs(
            source,
            projection=spec.teacher_projection,
            expected_teacher_obs_dim=int(getattr(teacher_policy, "obs_dim", source.shape[1])),
        )
        oracle_actions = _oracle_actions(
            teacher_policy,
            obs,
            info,
            teacher_obs,
            action_dim=config.action_dim,
        )
        oracle_future, oracle_action_chunk, oracle_valid = _oracle_shadow_pair(
            env,
            teacher_policy=teacher_policy,
            initial_oracle_actions=oracle_actions,
            initial_command=complete_command,
            config=config,
            observation_key=spec.observation_key,
            teacher_projection=spec.teacher_projection,
            student_projection=spec.student_projection,
            student_drop_index=spec.student_drop_index,
            command_info_keys=spec.command_info_keys,
        )
        snapshot_observable_restoration_valid = _visible_snapshot_matches(
            env,
            source=source,
            command=complete_command,
            spec=spec,
        )
        device = next(student_policy.parameters()).device
        with torch.inference_mode():
            student_output = student_policy(
                torch.as_tensor(observation_history, dtype=torch.float32, device=device),
                torch.as_tensor(action_history, dtype=torch.float32, device=device),
                torch.as_tensor(complete_command, dtype=torch.float32, device=device),
            )
        student_future = student_output.predicted_future.detach().cpu().numpy().astype(np.float32)
        student_action_chunk = student_output.action_chunk.detach().cpu().numpy().astype(np.float32)
        student_actions = student_action_chunk[:, 0]
        previous_action = action_history[0, -1].copy()
        base_position = _base_row(env, "get_base_pos", index=0)
        base_quaternion = _base_row(env, "get_base_quat", index=0)
        gait_phase = _info_row(info, "gait_phase", index=0)
        state = env.step(student_actions)
        terminated = bool(np.asarray(state.terminated, dtype=np.bool_).reshape(-1)[0])
        truncated = bool(np.asarray(state.truncated, dtype=np.bool_).reshape(-1)[0])
        action_error = oracle_actions[0] - student_actions[0]
        action_rate = student_actions[0] - previous_action
        right_leg_indices = tuple(
            index for index in _RIGHT_LEG_ACTION_INDICES if index < config.action_dim
        )
        displacement = None
        if base_position is not None and initial_base is not None:
            displacement = (
                float(base_position[0] - initial_base[0]),
                float(base_position[1] - initial_base[1]),
            )
        trace.append(
            FADACoverageStep(
                episode_id=0,
                timestep=timestep,
                observation_history=_tuple_rows(observation_history[0]),
                action_history=_tuple_rows(action_history[0]),
                command=tuple(float(item) for item in complete_command[0]),
                student_first_action=tuple(float(item) for item in student_actions[0]),
                student_predicted_future=_tuple_rows(student_future[0]),
                student_action_chunk=_tuple_rows(student_action_chunk[0]),
                oracle_first_action=tuple(float(item) for item in oracle_actions[0]),
                oracle_future=_tuple_rows(oracle_future[0]),
                oracle_action_chunk=_tuple_rows(oracle_action_chunk[0]),
                oracle_shadow_valid=bool(oracle_valid[0]),
                # The existing public owner transaction completed without an exception;
                # observable equality is recorded separately and does not claim a new
                # backend-private physics fingerprint.
                snapshot_owner_transaction_completed=True,
                snapshot_observable_restoration_valid=snapshot_observable_restoration_valid,
                command_identity_valid=command_identity_valid,
                terminated=terminated,
                truncated=truncated,
                termination_reason=_termination_reason(state, index=0),
                base_position=base_position,
                base_quaternion=base_quaternion,
                base_height=None if base_position is None else float(base_position[2]),
                planar_displacement=displacement,
                gait_phase=gait_phase,
                # No public backend-neutral contact projection exists on this env.
                left_foot_contact=None,
                right_foot_contact=None,
                per_joint_action_error=tuple(float(item) for item in action_error),
                per_joint_action_rate=tuple(float(item) for item in action_rate),
                right_leg_action_error=tuple(
                    float(action_error[index]) for index in right_leg_indices
                ),
                right_leg_action_rate=tuple(
                    float(action_rate[index]) for index in right_leg_indices
                ),
            )
        )
        if terminated or truncated:
            stop_reason = "environment_done"
            break

        next_obs = {key: np.asarray(value).copy() for key, value in state.obs.items()}
        next_info = dict(state.info)
        next_obs, next_info = set_transition_input_rows(
            env,
            command_info_key="commands",
            command_rows=command_array[None],
        )
        next_student_obs = project_student_obs(
            _obs_array(next_obs, spec.observation_key),
            projection=spec.student_projection,
            expected_student_obs_dim=config.obs_dim,
            student_drop_index=spec.student_drop_index,
        )
        observation_history = np.roll(observation_history, shift=-1, axis=1)
        observation_history[:, -1] = next_student_obs
        action_history = np.roll(action_history, shift=-1, axis=1)
        action_history[:, -1] = student_actions
        obs, info = next_obs, next_info

    return classify_fada_coverage(
        trace,
        history_length=config.history_length,
        prediction_horizon=config.prediction_horizon,
        command=command_array.tolist(),
        stop_reason=stop_reason,
    )


__all__ = [
    "FADACoverageReport",
    "FADACoverageStep",
    "FADACoverageVerdict",
    "classify_fada_coverage",
    "run_fada_coverage_diagnostic",
]
