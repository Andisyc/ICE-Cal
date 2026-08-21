from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base.np_env import NpEnvState
from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.g1.joystick import G1WalkEnv, G1WalkEnvCfg


def _fault_module():
    try:
        return importlib.import_module("unilab.envs.locomotion.g1.calibration_fault")
    except ModuleNotFoundError:
        pytest.fail("public G1 action-execution fault owner is missing")


@pytest.mark.parametrize("gain", [0.8, 1.0, 1.2])
def test_gain_fault_scales_only_the_authority_action(gain: float) -> None:
    fault = _fault_module()
    authority_action = np.asarray([[0.0, 0.0], [1.0, -2.0]], dtype=np.float32)
    config = fault.G1ActionExecutionFaultConfig(mode="gain", gain=gain)

    executed = fault.apply_action_execution_fault(
        authority_action,
        config,
        num_envs=2,
    )

    np.testing.assert_allclose(executed, authority_action * gain)
    np.testing.assert_array_equal(authority_action[0], np.zeros((2,), dtype=np.float32))
    if gain == 1.0:
        np.testing.assert_array_equal(executed, authority_action)


def test_disabled_fault_preserves_the_existing_action_path() -> None:
    fault = _fault_module()
    actions = np.asarray([[1.0, -2.0]], dtype=np.float32)
    executed = fault.apply_action_execution_fault(
        actions,
        None,
        num_envs=1,
    )
    np.testing.assert_array_equal(executed, actions)


def test_registry_can_materialize_the_immutable_fault_config() -> None:
    config = G1WalkEnvCfg()
    apply_cfg_overrides(
        config,
        {"action_execution_fault": {"mode": "gain", "gain": 1.2}},
    )
    assert config.action_execution_fault is not None
    assert config.action_execution_fault.mode == "gain"
    assert config.action_execution_fault.gain == 1.2


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (dict(mode="gain", gain=0.0), "positive"),
        (dict(mode="gain", gain=float("nan")), "finite"),
        (dict(mode="offset", gain=1.0), "unsupported"),
    ],
)
def test_fault_config_rejects_invalid_values(config: dict[str, object], message: str) -> None:
    fault = _fault_module()
    with pytest.raises(ValueError, match=message):
        fault.G1ActionExecutionFaultConfig(**config).validate()


@pytest.mark.parametrize(
    ("actions", "num_envs", "message"),
    [
        (np.zeros((2,), dtype=np.float32), 1, "rank-2"),
        (np.zeros((2, 3), dtype=np.float32), 1, "environment batch"),
    ],
)
def test_fault_transform_rejects_invalid_action_shape(
    actions: np.ndarray,
    num_envs: int,
    message: str,
) -> None:
    fault = _fault_module()
    with pytest.raises(ValueError, match=message):
        fault.apply_action_execution_fault(
            actions,
            fault.G1ActionExecutionFaultConfig(mode="gain", gain=0.8),
            num_envs=num_envs,
        )


def test_g1_apply_action_records_nominal_then_faults_the_authority_action() -> None:
    fault = _fault_module()
    env = object.__new__(G1WalkEnv)
    env._num_envs = 2
    env._cfg = SimpleNamespace(
        control_config=SimpleNamespace(action_scale=2.0),
        action_execution_fault=fault.G1ActionExecutionFaultConfig(mode="gain", gain=0.8),
    )
    env.default_angles = np.asarray([0.5, -0.5], dtype=np.float32)
    env._gait_phase_delta = 0.1
    env._gait_constraint_cfg = lambda: SimpleNamespace(  # type: ignore[method-assign]
        enabled=False,
        freeze_phase_in_stand_mode=False,
    )
    env._actions_for_execution = lambda actions, info: np.asarray(  # type: ignore[method-assign]
        [[0.0, 0.0], actions[1]],
        dtype=np.float32,
    )
    env._debug_action_trace_enabled = lambda: False  # type: ignore[method-assign]
    state = NpEnvState(
        obs={},
        reward=np.zeros((2,), dtype=np.float32),
        terminated=np.zeros((2,), dtype=np.bool_),
        truncated=np.zeros((2,), dtype=np.bool_),
        info={"gait_phase": np.zeros((2, 2), dtype=np.float32)},
    )
    nominal = np.asarray([[3.0, 4.0], [1.0, -2.0]], dtype=np.float32)

    ctrl = env.apply_action(nominal, state)

    np.testing.assert_array_equal(state.info["current_actions"], nominal)
    np.testing.assert_allclose(
        state.info["executed_actions"],
        np.asarray([[0.0, 0.0], [0.8, -1.6]], dtype=np.float32),
    )
    np.testing.assert_allclose(ctrl, state.info["executed_actions"] * 2.0 + env.default_angles)
