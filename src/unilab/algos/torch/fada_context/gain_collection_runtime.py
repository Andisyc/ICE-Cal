from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration_types import (
    CalibrationAxisSpec,
    FaultAxisCatalog,
)
from unilab.algos.torch.fada_context.gain_collection_artifact import (
    build_gain_calibration_raw_artifact,
)
from unilab.algos.torch.fada_context.gain_collection_types import (
    GainCalibrationCollectionProtocol,
    GainCalibrationPoint,
    GainCalibrationRawIdentity,
    GainCalibrationScenarioResult,
    GainCalibrationScenarioSpec,
    GainCalibrationSplit,
)


def _state_matrix(state: Any, carrier_name: str, key: str) -> np.ndarray:
    carrier = getattr(state, carrier_name, None)
    if not isinstance(carrier, Mapping) or key not in carrier:
        raise ValueError(f"state.{carrier_name}[{key!r}] is missing")
    value = np.asarray(carrier[key], dtype=np.float32)
    if value.ndim != 2 or not bool(np.isfinite(value).all()):
        raise ValueError(f"state.{carrier_name}[{key!r}] must be finite rank-2")
    return value


def _single_done(state: Any) -> bool:
    terminated = np.asarray(getattr(state, "terminated"), dtype=np.bool_)
    truncated = np.asarray(getattr(state, "truncated"), dtype=np.bool_)
    if terminated.shape != (1,) or truncated.shape != (1,):
        raise ValueError("calibration collection done flags must have one environment row")
    return bool(terminated[0] or truncated[0])


def _left_padded_history(
    observations: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    config: FADAArchitectureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    current = observations[-1]
    observation_history = list(observations[-config.history_length :])
    if len(observation_history) < config.history_length:
        observation_history = [current.copy()] * (
            config.history_length - len(observation_history)
        ) + observation_history
    action_history = list(actions[-config.history_length :])
    if len(action_history) < config.history_length:
        action_history = [np.zeros((config.action_dim,), dtype=np.float32)] * (
            config.history_length - len(action_history)
        ) + action_history
    return (
        np.asarray(observation_history, dtype=np.float32)[None],
        np.asarray(action_history, dtype=np.float32)[None],
    )


def _policy_query(
    policy: FADAPlannerIDMPolicy,
    observation_history: np.ndarray,
    action_history: np.ndarray,
    command: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        device = next(policy.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.inference_mode():
        output = policy(
            torch.as_tensor(observation_history, device=device),
            torch.as_tensor(action_history, device=device),
            torch.as_tensor(command, device=device),
        )
    intent = output.predicted_future.detach().cpu().numpy().astype(np.float32)
    chunk = output.action_chunk.detach().cpu().numpy().astype(np.float32)
    action = output.action.detach().cpu().numpy().astype(np.float32)
    expected = (
        (1, policy.config.prediction_horizon, policy.config.obs_dim),
        (1, policy.config.prediction_horizon, policy.config.action_dim),
        (1, policy.config.action_dim),
    )
    if (intent.shape, chunk.shape, action.shape) != expected:
        raise ValueError(
            "frozen Planner-IDM output shape mismatch: "
            f"observed={(intent.shape, chunk.shape, action.shape)} expected={expected}"
        )
    if not bool(
        np.isfinite(intent).all() and np.isfinite(chunk).all() and np.isfinite(action).all()
    ):
        raise ValueError("frozen Planner-IDM produced non-finite output")
    if not np.array_equal(action, chunk[:, 0]):
        raise ValueError("frozen Planner-IDM first action does not match action chunk index zero")
    return intent, chunk, action


def _stack_pending(pending: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tensor_names = tuple(name for name in pending[0] if name != "axis_name")
    return {
        **{name: torch.cat([row[name] for row in pending], dim=0) for name in tensor_names},
        "axis_name": [str(row["axis_name"]) for row in pending],
    }


def collect_gain_calibration_scenario(
    env: Any,
    policy: FADAPlannerIDMPolicy,
    spec: GainCalibrationScenarioSpec,
    *,
    rollout_id_start: int,
    axis_spec: CalibrationAxisSpec,
) -> GainCalibrationScenarioResult:
    """Collect one all-or-nothing episode transaction for a gain/split scenario."""

    spec.validate()
    if policy.training:
        raise ValueError("calibration collection requires an eval-mode frozen policy")
    if policy.config.history_length != 30 or policy.config.prediction_horizon != 6:
        raise ValueError("gain calibration collection requires H=30 and K=6")
    if "gain" not in axis_spec.names:
        raise ValueError("gain calibration collection requires an active gain axis")
    gain_axis_index = axis_spec.names.index("gain")
    if int(getattr(env, "num_envs", -1)) != 1:
        raise ValueError("gain calibration collection requires exactly one environment")
    set_autoreset = getattr(env, "set_autoreset", None)
    reset_all = getattr(env, "reset_all", None)
    if not callable(set_autoreset) or not callable(reset_all):
        raise TypeError("calibration environment must expose set_autoreset and reset_all")
    set_autoreset(False)
    fixed_command = np.asarray(spec.fixed_command, dtype=np.float32)[None]
    rollout_id = int(rollout_id_start)
    rejected = 0
    environment_steps = 0
    query_attempts = 0

    state = reset_all()
    observation = _state_matrix(state, "obs", spec.observation_key)
    command = _state_matrix(state, "info", spec.command_key)
    if observation.shape != (1, policy.config.obs_dim):
        raise ValueError("environment observation dimension does not match the checkpoint")
    if command.shape != fixed_command.shape or not np.array_equal(command, fixed_command):
        raise ValueError("environment reset command does not match the fixed smoke command")
    observations = [observation[0].copy()]
    actions: list[np.ndarray] = []
    pending: list[dict[str, Any]] = []

    while environment_steps < spec.max_environment_steps:
        query_attempts += 1
        if query_attempts > spec.max_environment_steps * 2:
            break
        observation_history, action_history = _left_padded_history(
            observations, actions, policy.config
        )
        try:
            intent, chunk, nominal = _policy_query(
                policy, observation_history, action_history, command
            )
        except ValueError:
            rejected += 1
            pending.clear()
            observations.clear()
            actions.clear()
            rollout_id += 1
            state = reset_all()
            observation = _state_matrix(state, "obs", spec.observation_key)
            command = _state_matrix(state, "info", spec.command_key)
            if command.shape != fixed_command.shape or not np.array_equal(command, fixed_command):
                raise ValueError("environment reset command does not match the fixed smoke command")
            observations.append(observation[0].copy())
            continue

        next_state = env.step(nominal)
        environment_steps += 1
        next_observation = _state_matrix(next_state, "obs", spec.observation_key)
        next_command = _state_matrix(next_state, "info", spec.command_key)
        current = _state_matrix(next_state, "info", "current_actions")
        authority = _state_matrix(next_state, "info", "authority_actions")
        executed = _state_matrix(next_state, "info", "executed_actions")
        if not np.array_equal(current, nominal):
            raise ValueError("environment current_actions no longer exposes the nominal action")
        expected_executed = authority * float(spec.point.gain)
        if not np.allclose(executed, expected_executed, rtol=1.0e-6, atol=1.0e-7):
            raise ValueError("environment executed_actions does not match authority action gain")
        valid_transaction = (
            not _single_done(next_state)
            and next_command.shape == fixed_command.shape
            and np.array_equal(next_command, fixed_command)
        )
        if valid_transaction and len(actions) >= policy.config.history_length:
            coefficients = torch.zeros((1, axis_spec.axis_count), dtype=torch.float32)
            coefficients[0, gain_axis_index] = spec.point.c_true
            pending.append(
                {
                    "observation_history": torch.from_numpy(observation_history.copy()),
                    "action_history": torch.from_numpy(action_history.copy()),
                    "command": torch.from_numpy(command.copy()),
                    "nominal_action_chunk": torch.from_numpy(chunk.copy()),
                    "c_true": coefficients,
                    "is_held_out_combination": torch.zeros((1,), dtype=torch.bool),
                    "injected_strength": torch.tensor([spec.point.gain], dtype=torch.float32),
                    "planner_intent": torch.from_numpy(intent.copy()),
                    "rollout_id": torch.tensor([rollout_id], dtype=torch.int64),
                    "seed": torch.tensor([spec.split.seed], dtype=torch.int64),
                    "split_id": torch.tensor([spec.split.split_id], dtype=torch.int64),
                    "executed_action": torch.from_numpy(executed.copy()),
                    "axis_name": "gain",
                }
            )
        if not valid_transaction:
            rejected += 1
            pending.clear()
            observations.clear()
            actions.clear()
            rollout_id += 1
            state = reset_all()
            observation = _state_matrix(state, "obs", spec.observation_key)
            command = _state_matrix(state, "info", spec.command_key)
            if command.shape != fixed_command.shape or not np.array_equal(command, fixed_command):
                raise ValueError("environment reset command does not match the fixed smoke command")
            observations.append(observation[0].copy())
            continue
        actions.append(nominal[0].copy())
        observations.append(next_observation[0].copy())
        command = next_command
        if len(pending) == spec.accepted_rows:
            return GainCalibrationScenarioResult(
                rows=_stack_pending(pending),
                environment_steps=environment_steps,
                rejected_transactions=rejected,
                next_rollout_id=rollout_id + 1,
            )
    raise RuntimeError(
        "gain calibration scenario exhausted its environment-step budget: "
        f"accepted={len(pending)} requested={spec.accepted_rows} "
        f"steps={environment_steps}"
    )


def _concat_row_trees(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tensor_names = tuple(name for name in rows[0] if name != "axis_name")
    return {
        **{name: torch.cat([row[name] for row in rows], dim=0) for name in tensor_names},
        "axis_name": [name for row in rows for name in row["axis_name"]],
    }

def collect_gain_calibration_rollouts(
    policy: FADAPlannerIDMPolicy,
    protocol: GainCalibrationCollectionProtocol,
    environment_factory: Callable[[GainCalibrationPoint, GainCalibrationSplit], Any],
    *,
    catalog: FaultAxisCatalog,
    identity: GainCalibrationRawIdentity,
    protocol_bytes: bytes,
    resolved_task_backend_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect the complete approved grid and close every factory-owned environment."""

    protocol.validate_approved()
    axis_spec = CalibrationAxisSpec.from_catalog(catalog)
    identity.validate(axis_spec)
    if policy.training:
        raise ValueError("calibration collection requires an eval-mode frozen policy")
    snapshot = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    scenario_rows: list[Mapping[str, Any]] = []
    next_rollout_id = 0
    for split in protocol.splits:
        for point in protocol.points:
            env = environment_factory(point, split)
            try:
                result = collect_gain_calibration_scenario(
                    env,
                    policy,
                    GainCalibrationScenarioSpec(
                        point=point,
                        split=split,
                        fixed_command=protocol.fixed_command,
                        accepted_rows=protocol.accepted_rows_per_scenario,
                        max_environment_steps=protocol.max_environment_steps_per_scenario,
                        observation_key=protocol.observation_key,
                        command_key=protocol.command_key,
                    ),
                    rollout_id_start=next_rollout_id,
                    axis_spec=axis_spec,
                )
                scenario_rows.append(result.rows)
                next_rollout_id = result.next_rollout_id
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
    current = policy.state_dict()
    if current.keys() != snapshot.keys() or any(
        not torch.equal(current[name].detach().cpu(), value) for name, value in snapshot.items()
    ):
        raise RuntimeError("frozen Planner-IDM parameters or buffers changed during collection")
    return build_gain_calibration_raw_artifact(
        _concat_row_trees(scenario_rows),
        policy.config,
        protocol,
        identity,
        axis_spec,
        protocol_bytes=protocol_bytes,
        resolved_task_backend_payload=resolved_task_backend_payload,
    )
