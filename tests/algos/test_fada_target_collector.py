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


class _V2Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FADAArchitectureConfig(
            obs_dim=66,
            action_dim=29,
            command_dim=3,
            observation_contract="g1_fada_state_v2",
            history_length=2,
            prediction_horizon=2,
            hidden_dim=8,
            num_heads=2,
            planner_layers=1,
            idm_encoder_layers=1,
            idm_decoder_layers=1,
            feedforward_dim=16,
        )
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, observation_history, _action_history, _command):
        return SimpleNamespace(action=observation_history[:, -1, :29])


class _V2Env:
    num_envs = 1

    def __init__(self) -> None:
        self.step_count = 0
        self.current_obs = np.arange(98, dtype=np.float32)[None, :]

    def _state(self) -> _State:
        return _State(
            obs={"obs": self.current_obs.copy()},
            info={"commands": np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32)},
            terminated=np.zeros((1,), dtype=np.bool_),
            truncated=np.zeros((1,), dtype=np.bool_),
        )

    def reset_all(self) -> _State:
        self.step_count = 0
        self.current_obs = np.arange(98, dtype=np.float32)[None, :]
        return self._state()

    def step(self, actions: np.ndarray) -> _State:
        self.step_count += 1
        self.current_obs[:, :29] = actions + float(self.step_count)
        return self._state()


class _SlopeEnv(_V2Env):
    def __init__(
        self,
        *,
        terminate_once: bool = False,
        truncate_once: bool = False,
        fall_once: bool = False,
    ) -> None:
        super().__init__()
        self.terminate_once = terminate_once
        self.truncate_once = truncate_once
        self.fall_once = fall_once
        self.did_terminate = False
        self.did_truncate = False
        self.did_fall = False
        self.state = self._state()

    def set_autoreset(self, enabled: bool) -> None:
        assert enabled is False

    def reset_all(self) -> _State:
        self.state = super().reset_all()
        return self.state

    def refresh_state(self) -> None:
        pass

    def step(self, actions: np.ndarray) -> _State:
        state = super().step(actions)
        if self.terminate_once and not self.did_terminate and self.step_count == 2:
            state.terminated[:] = True
            self.did_terminate = True
        if self.truncate_once and not self.did_truncate and self.step_count == 2:
            state.truncated[:] = True
            self.did_truncate = True
        if self.fall_once and not self.did_fall and self.step_count == 2:
            state.terminated[:] = True
            state.info["fall_terminated"] = np.ones((1,), dtype=np.bool_)
            self.did_fall = True
        self.state = state
        return state

    def get_base_pos(self) -> np.ndarray:
        return np.asarray([[2.0, 0.0, 0.3]])

    def get_foot_pos(self) -> np.ndarray:
        return np.asarray([[[2.0, 0.1, 0.2], [2.0, -0.1, 0.2]]])

    def get_physics_state_snapshot(self) -> np.ndarray:
        return np.asarray([self.step_count], dtype=np.float32)


def test_target_collector_consumes_preprojected_v2_observation_once() -> None:
    module = _target_module()

    result = module.collect_fada_target_windows(
        _V2Env(),
        rollout_policy=_V2Policy(),
        config=_V2Policy().config,
        num_windows=1,
        spec=module.FADATargetCollectionSpec(
            student_projection="g1_fada_state_v2",
            max_env_steps=10,
        ),
    )

    assert result.batch.observation_history.shape == (1, 2, 66)
    assert result.batch.executed_action_chunk.shape == (1, 2, 29)


def test_target_control_step_budget_derives_usable_windows_from_architecture() -> None:
    module = _target_module()
    config = FADAArchitectureConfig(
        obs_dim=66,
        action_dim=29,
        command_dim=3,
        observation_contract="g1_fada_state_v2",
        history_length=30,
        prediction_horizon=6,
        hidden_dim=128,
        num_heads=4,
        planner_layers=3,
        idm_encoder_layers=3,
        idm_decoder_layers=2,
        feedforward_dim=256,
    )
    spec = module.FADATargetCollectionSpec(ramp_steps=25, settle_steps=50)

    assert module.fada_target_window_budget(config, spec, control_steps=6000) == 5920
    with pytest.raises(ValueError, match="control_steps.*usable window"):
        module.fada_target_window_budget(config, spec, control_steps=35)


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


def test_target_command_schedule_ramps_then_holds_without_clipping() -> None:
    owner = importlib.import_module("unilab.algos.torch.distill.fada.target_collector")
    spec = owner.FADATargetCollectionSpec(
        command_start=(0.0, 0.0, 0.0),
        command_target=(0.4, 0.0, 0.0),
        ramp_steps=4,
        settle_steps=2,
    )

    np.testing.assert_allclose(owner._scheduled_command(spec, 0), [0.1, 0.0, 0.0])
    np.testing.assert_allclose(owner._scheduled_command(spec, 3), [0.4, 0.0, 0.0])
    np.testing.assert_allclose(owner._scheduled_command(spec, 9), [0.4, 0.0, 0.0])


def test_target_collector_uses_warmup_as_history_before_collection_boundary() -> None:
    module = _target_module()

    result = module.collect_fada_target_windows(
        _Env(done_steps=(7,)),
        rollout_policy=_Policy(),
        config=_config(),
        num_windows=8,
        spec=module.FADATargetCollectionSpec(
            max_env_steps=10,
            ramp_steps=2,
            settle_steps=1,
            single_trajectory=True,
        ),
    )

    assert result.env_steps == 7
    assert result.batch.observation_history.shape[0] == 2
    assert result.batch.start_timestep.tolist() == [3, 4]


def test_single_trajectory_returns_usable_prefix_when_episode_ends() -> None:
    module = _target_module()
    captured: list[int] = []

    result = module.collect_fada_target_windows(
        _Env(done_steps=(4,)),
        rollout_policy=_Policy(),
        config=_config(),
        num_windows=8,
        spec=module.FADATargetCollectionSpec(
            max_env_steps=10,
            single_trajectory=True,
            capture_frame=lambda: captured.append(1),
        ),
    )

    assert len(captured) == 4
    assert result.env_steps == 4
    assert result.rejected_done_transitions == 1
    assert result.batch.observation_history.shape[0] == 1
    assert result.batch.episode_id.tolist() == [0]
    assert result.batch.start_timestep.tolist() == [1]


def test_single_trajectory_rejects_episode_with_no_usable_window() -> None:
    module = _target_module()

    with pytest.raises(RuntimeError, match="ended before producing a usable window"):
        module.collect_fada_target_windows(
            _Env(done_steps=(2,)),
            rollout_policy=_Policy(),
            config=_config(),
            num_windows=8,
            spec=module.FADATargetCollectionSpec(max_env_steps=10, single_trajectory=True),
        )


def test_target_collector_captures_reset_frame_before_first_action() -> None:
    module = _target_module()
    assert "capture_initial_frame" in module.FADATargetCollectionSpec.__dataclass_fields__
    events: list[str] = []

    module.collect_fada_target_windows(
        _Env(),
        rollout_policy=_Policy(),
        config=_config(),
        num_windows=1,
        spec=module.FADATargetCollectionSpec(
            max_env_steps=10,
            capture_initial_frame=lambda: events.append("initial"),
            capture_frame=lambda: events.append("step"),
        ),
    )

    assert events == ["initial", "step", "step", "step"]


def test_slope_episode_policy_cycles_commands_and_classifies_boundaries() -> None:
    from unilab.algos.torch.distill.fada.target_collector import FADASlopeEpisodePolicy
    from unilab.algos.torch.distill.fada.target_domain import FADASlopeGeometry

    geometry = FADASlopeGeometry(15.0, 0.8, 1.5, 8.0, 0.25, 0.5)
    policy = FADASlopeEpisodePolicy(
        geometry,
        ((0.75, 0.0, 0.0), (0.8, 0.0, 0.0), (0.85, 0.0, 0.0)),
    )
    np.testing.assert_allclose(policy.command_for_episode(4), [0.8, 0.0, 0.0])

    feet = np.array([[2.0, 0.1, 0.2], [2.0, -0.1, 0.2]])
    assert (
        policy.classify(base_pos_w=np.array([1.6, 0.0, 0.2]), feet_pos_w=feet, done=False).accept
        is False
    )
    assert (
        policy.classify(base_pos_w=np.array([2.0, 0.0, 0.2]), feet_pos_w=feet, done=False).accept
        is True
    )
    exited = feet.copy()
    exited[0, 1] = 0.41
    assert (
        policy.classify(
            base_pos_w=np.array([2.0, 0.0, 0.2]), feet_pos_w=exited, done=False
        ).terminal_reason
        == "foot_exit"
    )
    assert (
        policy.classify(
            base_pos_w=np.array([9.0, 0.0, 2.0]), feet_pos_w=feet, done=False
        ).terminal_reason
        == "finish"
    )
    assert (
        policy.classify(
            base_pos_w=np.array([2.0, 0.0, 0.2]), feet_pos_w=feet, done=True
        ).terminal_reason
        == "fall"
    )


@pytest.mark.parametrize(
    ("flag", "reason"),
    [
        ("terminate_once", "environment_termination"),
        ("truncate_once", "truncated"),
        ("fall_once", "fall"),
    ],
)
def test_slope_collection_owns_exact_accepted_steps_and_terminal_provenance(
    flag: str, reason: str
) -> None:
    module = _target_module()
    env = _SlopeEnv(**{flag: True})
    from unilab.algos.torch.distill.fada.target_domain import FADASlopeGeometry

    geometry = FADASlopeGeometry(15.0, 0.8, 1.5, 8.0, 0.25, 0.5)
    result = module.collect_fada_slope_windows(
        env,
        _V2Policy(),
        _V2Policy().config,
        4,
        module.FADASlopeEpisodePolicy(geometry, ((0.8, 0.0, 0.0),)),
        module.FADATargetCollectionSpec(
            max_env_steps=20,
            command_start=(0.8, 0.0, 0.0),
            student_projection="g1_fada_state_v2",
        ),
    )

    assert result.accepted_steps == 4
    assert result.termination_counts is not None
    assert result.termination_counts[reason] == 1
    assert result.batch.observation_history.shape[0] == 1
