from __future__ import annotations

import importlib
from dataclasses import fields

import numpy as np
import pytest


def _window_module():
    return importlib.import_module("unilab.algos.torch.distill.fada_windows")


def _transition(
    index: int,
    *,
    command: tuple[float, float] = (0.4, -0.1),
    episode_id: int = 7,
):
    module = _window_module()
    return module.FADACausalTransition(
        observation=np.asarray([10.0 + index, 20.0 + index, 30.0 + index], dtype=np.float32),
        previous_action=np.asarray([40.0 + index, 50.0 + index], dtype=np.float32),
        command=np.asarray(command, dtype=np.float32),
        executed_action=np.asarray([60.0 + index, 70.0 + index], dtype=np.float32),
        next_observation=np.asarray([80.0 + index, 90.0 + index, 100.0 + index], dtype=np.float32),
        episode_id=episode_id,
        timestep=index,
    )


def test_build_causal_window_maps_every_history_and_future_index() -> None:
    module = _window_module()
    records = tuple(_transition(index) for index in range(3))

    window = module.build_fada_causal_window(
        records,
        history_length=2,
        prediction_horizon=2,
        command_scenario="walk",
    )

    assert window is not None
    np.testing.assert_array_equal(
        window.observation_history,
        np.stack([records[0].observation, records[1].observation]),
    )
    np.testing.assert_array_equal(
        window.action_history,
        np.stack([records[0].previous_action, records[1].previous_action]),
    )
    np.testing.assert_array_equal(window.command, records[1].command)
    np.testing.assert_array_equal(
        window.realized_future,
        np.stack([records[1].next_observation, records[2].next_observation]),
    )
    np.testing.assert_array_equal(
        window.executed_action_chunk,
        np.stack([records[1].executed_action, records[2].executed_action]),
    )
    assert window.episode_id == 7
    assert window.start_timestep == 1


def test_build_causal_window_rejects_command_or_episode_crossing() -> None:
    module = _window_module()
    command_drift = (_transition(0), _transition(1), _transition(2, command=(0.0, 0.0)))
    episode_drift = (_transition(0), _transition(1), _transition(2, episode_id=8))

    assert (
        module.build_fada_causal_window(
            command_drift,
            history_length=2,
            prediction_horizon=2,
            command_scenario="walk",
        )
        is None
    )
    assert (
        module.build_fada_causal_window(
            episode_drift,
            history_length=2,
            prediction_horizon=2,
            command_scenario="walk",
        )
        is None
    )


def test_build_causal_window_preserves_source_scenario_admission() -> None:
    module = _window_module()
    static = tuple(_transition(index, command=(0.0, 0.0)) for index in range(3))
    transition = (
        _transition(0, command=(0.4, 0.0)),
        _transition(1, command=(0.0, 0.0)),
        _transition(2, command=(0.0, 0.0)),
    )

    assert (
        module.build_fada_causal_window(
            static,
            history_length=2,
            prediction_horizon=2,
            command_scenario="static_stand",
        )
        is not None
    )
    assert (
        module.build_fada_causal_window(
            transition,
            history_length=2,
            prediction_horizon=2,
            command_scenario="walk_to_stand",
        )
        is not None
    )
    assert (
        module.build_fada_causal_window(
            static,
            history_length=2,
            prediction_horizon=2,
            command_scenario="walk_to_stand",
        )
        is None
    )


def test_build_cold_start_window_repeats_nonzero_reset_and_zeros_actions() -> None:
    module = _window_module()
    records = tuple(_transition(index, command=(0.0, 0.0)) for index in range(2))

    window = module.build_fada_cold_start_window(
        records,
        history_length=3,
        prediction_horizon=2,
    )

    assert window is not None
    np.testing.assert_array_equal(
        window.observation_history,
        np.repeat(records[0].observation[None], 3, axis=0),
    )
    np.testing.assert_array_equal(window.action_history, np.zeros((3, 2), dtype=np.float32))
    np.testing.assert_array_equal(window.command, np.zeros((2,), dtype=np.float32))
    assert window.start_timestep == 0


def test_causal_window_fails_closed_for_invalid_record_shape_or_timestep() -> None:
    module = _window_module()
    with pytest.raises(ValueError, match="record count mismatch"):
        module.build_fada_causal_window(
            (_transition(0), _transition(1)),
            history_length=2,
            prediction_horizon=2,
            command_scenario="walk",
        )

    bad_order = (_transition(0), _transition(2), _transition(3))
    with pytest.raises(ValueError, match="consecutive"):
        module.build_fada_causal_window(
            bad_order,
            history_length=2,
            prediction_horizon=2,
            command_scenario="walk",
        )


def test_causal_window_public_schema_has_no_oracle_fields() -> None:
    module = _window_module()
    names = {field.name for field in fields(module.FADACausalWindow)}

    assert names == {
        "observation_history",
        "action_history",
        "command",
        "realized_future",
        "executed_action_chunk",
        "episode_id",
        "start_timestep",
    }
    assert all("oracle" not in name for name in names)
