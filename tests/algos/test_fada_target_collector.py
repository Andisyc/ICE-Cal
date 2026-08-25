from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill import FADAArchitectureConfig


def _target_module():
    return importlib.import_module("unilab.algos.torch.distill.fada_target_collector")


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=3,
        action_dim=2,
        command_dim=2,
        history_length=2,
        prediction_horizon=2,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _config()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(
        self,
        observation_history: torch.Tensor,
        _action_history: torch.Tensor,
        _command: torch.Tensor,
    ) -> SimpleNamespace:
        latest = observation_history[:, -1]
        return SimpleNamespace(action=latest[:, :2] + torch.tensor([0.5, -0.25]))


@dataclass
class _State:
    obs: dict[str, np.ndarray]
    info: dict[str, np.ndarray]
    terminated: np.ndarray
    truncated: np.ndarray


class _Env:
    def __init__(
        self,
        *,
        done_steps: tuple[int, ...] = (),
        command_schedule: tuple[tuple[float, float], ...] = ((0.4, -0.1),),
    ) -> None:
        self.num_envs = 1
        self.done_steps = set(done_steps)
        self.command_schedule = command_schedule
        self.step_count = 0
        self.current_obs = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)

    def _command(self) -> np.ndarray:
        index = min(self.step_count, len(self.command_schedule) - 1)
        return np.asarray([self.command_schedule[index]], dtype=np.float32)

    def reset_all(self) -> _State:
        self.current_obs = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        self.step_count = 0
        return _State(
            obs={"obs": self.current_obs.copy()},
            info={"commands": self._command()},
            terminated=np.zeros((1,), dtype=np.bool_),
            truncated=np.zeros((1,), dtype=np.bool_),
        )

    def step(self, actions: np.ndarray) -> _State:
        self.step_count += 1
        done = np.asarray([self.step_count in self.done_steps], dtype=np.bool_)
        if bool(done[0]):
            self.current_obs = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        else:
            self.current_obs = self.current_obs + np.concatenate(
                [actions, np.ones((1, 1), dtype=np.float32)], axis=1
            )
        return _State(
            obs={"obs": self.current_obs.copy()},
            info={"commands": self._command()},
            terminated=done,
            truncated=np.zeros_like(done),
        )


def test_target_collector_builds_exact_oracle_free_executed_window() -> None:
    module = _target_module()
    result = module.collect_fada_target_windows(
        _Env(),
        rollout_policy=_Policy(),
        config=_config(),
        num_windows=1,
        spec=module.FADATargetCollectionSpec(max_env_steps=10),
    )

    assert result.env_steps == 3
    assert result.rejected_done_transitions == 0
    assert result.rejected_command_windows == 0
    torch.testing.assert_close(
        result.batch.observation_history[0],
        torch.tensor([[1.0, 2.0, 3.0], [2.5, 3.75, 4.0]]),
    )
    torch.testing.assert_close(
        result.batch.action_history[0],
        torch.tensor([[0.0, 0.0], [1.5, 1.75]]),
    )
    torch.testing.assert_close(
        result.batch.executed_action_chunk[0],
        torch.tensor([[3.0, 3.5], [6.0, 7.0]]),
    )
    torch.testing.assert_close(
        result.batch.realized_future[0],
        torch.tensor([[5.5, 7.25, 5.0], [11.5, 14.25, 6.0]]),
    )
    assert result.batch.episode_id.tolist() == [0]
    assert result.batch.start_timestep.tolist() == [1]


def test_target_collector_clears_history_and_advances_episode_on_done() -> None:
    module = _target_module()
    result = module.collect_fada_target_windows(
        _Env(done_steps=(2,)),
        rollout_policy=_Policy(),
        config=_config(),
        num_windows=1,
        spec=module.FADATargetCollectionSpec(max_env_steps=10),
    )

    assert result.rejected_done_transitions == 1
    assert result.batch.episode_id.tolist() == [1]
    assert result.batch.start_timestep.tolist() == [1]


def test_target_collector_rejects_future_command_change() -> None:
    module = _target_module()
    result = module.collect_fada_target_windows(
        _Env(command_schedule=((0.4, -0.1), (0.4, -0.1), (0.0, 0.0), (0.0, 0.0))),
        rollout_policy=_Policy(),
        config=_config(),
        num_windows=1,
        spec=module.FADATargetCollectionSpec(max_env_steps=10),
    )

    assert result.rejected_command_windows >= 1
    torch.testing.assert_close(result.batch.command, torch.zeros_like(result.batch.command))


def test_target_collector_fails_closed_when_no_complete_window_is_possible() -> None:
    module = _target_module()
    with pytest.raises(RuntimeError, match="produced 0/1 windows"):
        module.collect_fada_target_windows(
            _Env(done_steps=(1, 2, 3, 4, 5)),
            rollout_policy=_Policy(),
            config=_config(),
            num_windows=1,
            spec=module.FADATargetCollectionSpec(max_env_steps=5),
        )


def test_target_collector_rejects_projection_contract_mismatch_before_reset() -> None:
    module = _target_module()

    class CountingEnv(_Env):
        def __init__(self) -> None:
            super().__init__()
            self.reset_calls = 0

        def reset_all(self) -> _State:
            self.reset_calls += 1
            return super().reset_all()

    env = CountingEnv()
    with pytest.raises(ValueError, match="projection.*architecture contract"):
        module.collect_fada_target_windows(
            env,
            rollout_policy=_Policy(),
            config=_config(),
            num_windows=1,
            spec=module.FADATargetCollectionSpec(
                student_projection="g1_fada_state_v2",
                max_env_steps=10,
            ),
        )

    assert env.reset_calls == 0


def test_target_collector_public_boundary_has_no_oracle_or_training_inputs() -> None:
    module = _target_module()
    names = set(inspect.signature(module.collect_fada_target_windows).parameters)

    assert names == {"env", "rollout_policy", "config", "num_windows", "spec"}
    assert not any(
        forbidden in name
        for name in names
        for forbidden in ("oracle", "teacher", "trainer", "optimizer", "replay")
    )
