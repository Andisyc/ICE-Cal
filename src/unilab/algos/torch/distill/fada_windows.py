from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

FADACommandScenario = Literal["walk", "static_stand", "walk_to_stand"]


@dataclass(frozen=True)
class FADACausalTransition:
    """One ordinary executed transition with explicit episode/time identity."""

    observation: np.ndarray
    previous_action: np.ndarray
    command: np.ndarray
    executed_action: np.ndarray
    next_observation: np.ndarray
    episode_id: int
    timestep: int


@dataclass(frozen=True)
class FADACausalWindow:
    """Oracle-free history/future window shared by source and target adapters."""

    observation_history: np.ndarray
    action_history: np.ndarray
    command: np.ndarray
    realized_future: np.ndarray
    executed_action_chunk: np.ndarray
    episode_id: int
    start_timestep: int


def _validate_records(records: Sequence[FADACausalTransition]) -> None:
    if not records:
        raise ValueError("causal window records must be non-empty")
    first = records[0]
    expected_shapes = {
        "observation": np.asarray(first.observation).shape,
        "previous_action": np.asarray(first.previous_action).shape,
        "command": np.asarray(first.command).shape,
        "executed_action": np.asarray(first.executed_action).shape,
        "next_observation": np.asarray(first.next_observation).shape,
    }
    if any(len(shape) != 1 for shape in expected_shapes.values()):
        raise ValueError(f"causal transition fields must be rank-1, got {expected_shapes}")
    if expected_shapes["observation"] != expected_shapes["next_observation"]:
        raise ValueError("causal transition observation and next_observation shapes must match")
    for offset, record in enumerate(records):
        for name, expected_shape in expected_shapes.items():
            value = np.asarray(getattr(record, name))
            if value.shape != expected_shape:
                first_value = np.asarray(getattr(first, name))
                raise ValueError(
                    f"causal transition {name} shape mismatch: "
                    f"expected={expected_shape!r} observed={value.shape!r} "
                    f"(offset={offset} "
                    f"expected_dtype={first_value.dtype} observed_dtype={value.dtype} "
                    f"expected_type={type(getattr(first, name)).__name__} "
                    f"observed_type={type(getattr(record, name)).__name__} "
                    f"first_episode={int(first.episode_id)} first_timestep={int(first.timestep)} "
                    f"record_episode={int(record.episode_id)} "
                    f"record_timestep={int(record.timestep)})"
                )
            if not bool(np.all(np.isfinite(value))):
                raise ValueError(f"causal transition {name} must contain only finite values")
        if int(record.timestep) != int(first.timestep) + offset:
            raise ValueError("causal transition timesteps must be consecutive")
        if int(record.episode_id) < 0 or int(record.timestep) < 0:
            raise ValueError("causal transition episode_id and timestep must be non-negative")


def _same_episode(records: Sequence[FADACausalTransition]) -> bool:
    episode_id = int(records[0].episode_id)
    return all(int(record.episode_id) == episode_id for record in records[1:])


def build_fada_causal_window(
    records: Sequence[FADACausalTransition],
    *,
    history_length: int,
    prediction_horizon: int,
    command_scenario: FADACommandScenario = "walk",
) -> FADACausalWindow | None:
    """Build one steady causal window or reject a semantic boundary crossing."""

    expected = int(history_length) + int(prediction_horizon) - 1
    if len(records) != expected:
        raise ValueError(
            f"causal window record count mismatch: expected {expected}, got {len(records)}"
        )
    if int(history_length) <= 0 or int(prediction_horizon) <= 0:
        raise ValueError("history_length and prediction_horizon must be positive")
    if command_scenario not in {"walk", "static_stand", "walk_to_stand"}:
        raise ValueError(f"unsupported FADA command scenario: {command_scenario!r}")
    _validate_records(records)
    if not _same_episode(records):
        return None

    anchor = int(history_length) - 1
    history_records = records[: int(history_length)]
    future_records = records[anchor : anchor + int(prediction_horizon)]
    command = np.asarray(future_records[0].command)
    if any(not np.array_equal(record.command, command) for record in future_records[1:]):
        return None
    if command_scenario == "static_stand":
        if bool(np.any(np.abs(command) > 1.0e-6)):
            return None
    elif command_scenario == "walk_to_stand":
        future_is_standing = not bool(np.any(np.abs(command) > 1.0e-6))
        history_has_walking = any(
            bool(np.any(np.abs(record.command) > 1.0e-6)) for record in history_records[:-1]
        )
        if not future_is_standing or not history_has_walking:
            return None

    return FADACausalWindow(
        observation_history=np.stack([record.observation for record in history_records]).copy(),
        action_history=np.stack([record.previous_action for record in history_records]).copy(),
        command=command.copy(),
        realized_future=np.stack([record.next_observation for record in future_records]).copy(),
        executed_action_chunk=np.stack(
            [record.executed_action for record in future_records]
        ).copy(),
        episode_id=int(future_records[0].episode_id),
        start_timestep=int(future_records[0].timestep),
    )


def build_fada_cold_start_window(
    records: Sequence[FADACausalTransition],
    *,
    history_length: int,
    prediction_horizon: int,
) -> FADACausalWindow | None:
    """Build the exact repeated-reset/zero-action v005 cold-start window."""

    if len(records) != int(prediction_horizon):
        raise ValueError(
            "cold-start causal window record count mismatch: "
            f"expected {prediction_horizon}, got {len(records)}"
        )
    if int(history_length) <= 0 or int(prediction_horizon) <= 0:
        raise ValueError("history_length and prediction_horizon must be positive")
    _validate_records(records)
    if not _same_episode(records):
        return None
    command = np.asarray(records[0].command)
    if any(not np.array_equal(record.command, command) for record in records[1:]):
        return None
    if bool(np.any(np.abs(command) > 1.0e-6)):
        return None

    reset_observation = np.asarray(records[0].observation)
    action_dim = int(np.asarray(records[0].previous_action).shape[0])
    return FADACausalWindow(
        observation_history=np.repeat(reset_observation[None], int(history_length), axis=0).copy(),
        action_history=np.zeros((int(history_length), action_dim), dtype=np.float32),
        command=command.copy(),
        realized_future=np.stack([record.next_observation for record in records]).copy(),
        executed_action_chunk=np.stack([record.executed_action for record in records]).copy(),
        episode_id=int(records[0].episode_id),
        start_timestep=int(records[0].timestep),
    )
