from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from unilab.algos.torch.distill.fada import FADAArchitectureConfig
from unilab.algos.torch.fada_context.support_query import (
    SupportContextBatch,
    SupportQueryContextConfig,
)
from unilab.algos.torch.fada_context.support_query_evaluation import (
    evaluate_online_support_closed_loop,
    evaluate_support_query_closed_loop,
)
from unilab.algos.torch.fada_context.support_query_runtime import (
    create_fixed_fault_paired_environments,
)


class _FixedContext(nn.Module):
    def __init__(self, config: FADAArchitectureConfig) -> None:
        super().__init__()
        self.context_config = SupportQueryContextConfig(
            support_length=1,
            context_hidden_dim=1,
            context_layers=1,
        )
        self.config = config
        self.calls: list[tuple[SupportContextBatch, torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        support: SupportContextBatch,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        support.validate(self.config, support_length=1)
        self.calls.append(
            (
                support,
                observation_history.detach().clone(),
                action_history.detach().clone(),
            )
        )
        # strength 0.7 needs action 1 / 0.7, hence residual 3 / 7.
        return torch.full(
            (support.batch_size, self.config.hidden_dim),
            3.0 / 7.0,
            dtype=support.target_future.dtype,
            device=support.target_future.device,
        )


class _FixedPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FADAArchitectureConfig(
            obs_dim=1,
            action_dim=1,
            command_dim=3,
            history_length=1,
            prediction_horizon=1,
            hidden_dim=1,
            num_heads=1,
            planner_layers=1,
            idm_encoder_layers=1,
            idm_decoder_layers=1,
            feedforward_dim=4,
        )
        self.frozen_probe = nn.Parameter(torch.tensor([2.5]), requires_grad=False)
        self.context_encoder = _FixedContext(self.config)
        self.planner = _UnitPlanner()
        self.idm = _UnitIDM()

    def act_with_context(
        self,
        support: SupportContextBatch,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> SimpleNamespace:
        del command
        delta_z = self.context_encoder(support, observation_history, action_history)
        action = torch.ones_like(delta_z[:, :1]) + delta_z[:, :1]
        return SimpleNamespace(
            action=action,
            action_chunk=action[:, None, :],
            delta_z=delta_z,
        )


class _FirstActionOnlyPolicy(_FixedPolicy):
    def act_with_context(
        self,
        support: SupportContextBatch,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> SimpleNamespace:
        output = super().act_with_context(
            support,
            observation_history,
            action_history,
            command,
        )
        return SimpleNamespace(
            action=output.action,
            action_chunk=torch.stack(
                (output.action, torch.full_like(output.action, 99.0)),
                dim=1,
            ),
            delta_z=output.delta_z,
        )


class _MalformedActionPolicy(_FixedPolicy):
    def act_with_context(
        self,
        support: SupportContextBatch,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> SimpleNamespace:
        output = super().act_with_context(
            support,
            observation_history,
            action_history,
            command,
        )
        return SimpleNamespace(
            action=torch.cat((output.action, output.action), dim=-1),
            action_chunk=output.action_chunk,
            delta_z=output.delta_z,
        )


class _SupportIdentityContext(_FixedContext):
    def forward(
        self,
        support: SupportContextBatch,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        support.validate(self.config, support_length=1)
        self.calls.append(
            (
                support,
                observation_history.detach().clone(),
                action_history.detach().clone(),
            )
        )
        return support.realized_state[:, 0, :1]


class _SupportIdentityPolicy(_FixedPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.context_encoder = _SupportIdentityContext(self.config)


class _UnitPlanner(nn.Module):
    def forward(self, observation_history: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        del command
        return torch.zeros(observation_history.shape[0], 1, 1, device=observation_history.device)


class _UnitIDM(nn.Module):
    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        del action_history, future
        return torch.ones(observation_history.shape[0], 1, 1, device=observation_history.device)


class _LinearStrengthEnv:
    def __init__(self, strength: float, *, rows: int = 2) -> None:
        self.num_envs = rows
        self._strength = float(strength)
        self._position = np.zeros((rows, 1), dtype=np.float32)
        self._velocity = np.zeros((rows, 1), dtype=np.float32)
        self._autoreset = True
        self.actions: list[np.ndarray] = []
        self.state = self._make_state()

    def _make_state(self) -> SimpleNamespace:
        strength = np.ones((self.num_envs, 29), dtype=np.float32)
        strength[:, 3] = self._strength
        return SimpleNamespace(
            obs={"obs": self._position.copy()},
            info={
                "commands": np.broadcast_to(
                    np.array([0.4, 0.0, 0.0], dtype=np.float32),
                    (self.num_envs, 3),
                ).copy(),
                "privileged_actuator_strength": strength,
            },
            terminated=np.zeros((self.num_envs,), dtype=np.bool_),
            truncated=np.zeros((self.num_envs,), dtype=np.bool_),
        )

    def set_autoreset(self, enabled: bool) -> None:
        self._autoreset = enabled

    def capture_rollout_snapshot(self) -> tuple[np.ndarray, np.ndarray, SimpleNamespace]:
        return self._position.copy(), self._velocity.copy(), copy.deepcopy(self.state)

    def restore_rollout_snapshot(
        self, snapshot: tuple[np.ndarray, np.ndarray, SimpleNamespace]
    ) -> None:
        self._position, self._velocity, self.state = copy.deepcopy(snapshot)

    def step(self, action: np.ndarray) -> SimpleNamespace:
        self.actions.append(action.copy())
        self._velocity = action * self._strength
        self._position = self._position + self._velocity
        self.state = self._make_state()
        return self.state

    def get_base_pos(self) -> np.ndarray:
        return np.concatenate(
            (self._position, np.zeros((self.num_envs, 2), dtype=np.float32)), axis=1
        )

    def get_base_quat(self) -> np.ndarray:
        quat = np.zeros((self.num_envs, 4), dtype=np.float32)
        quat[:, 0] = 1.0
        return quat

    def get_local_linvel(self) -> np.ndarray:
        return np.concatenate(
            (self._velocity, np.zeros((self.num_envs, 2), dtype=np.float32)), axis=1
        )

    def get_dof_pos(self) -> np.ndarray:
        return self._position.copy()

    def get_dof_vel(self) -> np.ndarray:
        return self._velocity.copy()


class _EndAfterFirstStepEnv(_LinearStrengthEnv):
    def __init__(self, strength: float, *, outcome: str, rows: int = 2) -> None:
        super().__init__(strength, rows=rows)
        self._outcome = outcome
        self._steps_since_restore = 0

    def capture_rollout_snapshot(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, SimpleNamespace], int]:
        return super().capture_rollout_snapshot(), self._steps_since_restore

    def restore_rollout_snapshot(
        self,
        snapshot: tuple[tuple[np.ndarray, np.ndarray, SimpleNamespace], int],
    ) -> None:
        base_snapshot, self._steps_since_restore = copy.deepcopy(snapshot)
        super().restore_rollout_snapshot(base_snapshot)

    def step(self, action: np.ndarray) -> SimpleNamespace:
        state = super().step(action)
        self._steps_since_restore += 1
        if self._steps_since_restore == 1:
            getattr(state, self._outcome)[:] = True
        return state


def _support(rows: int, *, identities: tuple[float, ...] | None = None) -> SupportContextBatch:
    if identities is None:
        values = torch.zeros(rows, 1, 1)
    else:
        values = torch.tensor(identities, dtype=torch.float32).reshape(rows, 1, 1)
    return SupportContextBatch(
        target_future=values.unsqueeze(2),
        realized_state=values,
        executed_action=values,
    )


def _support_command(rows: int) -> torch.Tensor:
    return torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float32).expand(rows, -1)


def test_closed_loop_one_step_executes_exact_first_action_and_records_one_residual() -> None:
    policy = _FirstActionOnlyPolicy()
    healthy = _LinearStrengthEnv(1.0)
    fault = _LinearStrengthEnv(0.7)

    report = evaluate_support_query_closed_loop(
        healthy,
        fault,
        policy,  # type: ignore[arg-type]
        _support(2),
        support_command=_support_command(2),
        steps=1,
        device="cpu",
    )

    # Hand oracle: a=1/0.7=10/7 matches the healthy one-metre step; tail=99 is forbidden.
    assert len(policy.context_encoder.calls) == 1
    assert len(healthy.actions) == 1
    assert len(fault.actions) == 2
    np.testing.assert_array_equal(healthy.actions[0], np.ones((2, 1), dtype=np.float32))
    np.testing.assert_allclose(
        fault.actions[0],
        np.ones((2, 1), dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        fault.actions[1],
        np.full((2, 1), 10.0 / 7.0, dtype=np.float32),
        rtol=1.0e-6,
        atol=0.0,
    )
    assert not np.any(fault.actions[1] == 99.0)
    assert report["context"]["delta_z_trace_shape"] == [1, 2, 1]
    np.testing.assert_allclose(
        np.asarray(report["context"]["delta_z_trace"]),
        np.full((1, 2, 1), 3.0 / 7.0),
        rtol=1.0e-6,
        atol=0.0,
    )


def test_closed_loop_rejects_command_mismatch_before_any_context_action() -> None:
    policy = _FixedPolicy()
    wrong_command = _support_command(2).clone()
    wrong_command[:, 0] = 0.5

    with pytest.raises(ValueError, match="does not match Support command provenance"):
        evaluate_support_query_closed_loop(
            _LinearStrengthEnv(1.0),
            _LinearStrengthEnv(0.7),
            policy,  # type: ignore[arg-type]
            _support(2),
            support_command=wrong_command,
            steps=2,
            device="cpu",
        )

    assert policy.context_encoder.calls == []


def test_closed_loop_rejects_malformed_action_before_fault_context_step() -> None:
    policy = _MalformedActionPolicy()
    fault = _LinearStrengthEnv(0.7)

    with pytest.raises(ValueError, match="emitted an invalid action"):
        evaluate_support_query_closed_loop(
            _LinearStrengthEnv(1.0),
            fault,
            policy,  # type: ignore[arg-type]
            _support(2),
            support_command=_support_command(2),
            steps=2,
            device="cpu",
        )

    assert len(policy.context_encoder.calls) == 1
    # Only the two fault-zero baseline steps occurred; malformed Context Action never reached env.step.
    assert len(fault.actions) == 2


@pytest.mark.parametrize(
    ("outcome", "health_key"),
    [("terminated", "fall_rate"), ("truncated", "truncation_rate")],
)
def test_closed_loop_stops_context_after_all_rows_end(
    outcome: str,
    health_key: str,
) -> None:
    policy = _FixedPolicy()
    fault = _EndAfterFirstStepEnv(0.7, outcome=outcome)
    report = evaluate_support_query_closed_loop(
        _LinearStrengthEnv(1.0),
        fault,
        policy,  # type: ignore[arg-type]
        _support(2),
        support_command=_support_command(2),
        steps=3,
        device="cpu",
    )

    assert report["fault_context"][health_key] == 1.0
    assert report["fault_context"]["survival_steps_mean"] == 1.0
    assert len(policy.context_encoder.calls) == 1
    assert len(fault.actions) == 2
    assert report["context"]["delta_z_trace_shape"] == [1, 2, 1]
    for branch in ("fault_zero_distance_to_healthy", "fault_context_distance_to_healthy"):
        assert report[branch]["aligned_state_row_steps"] == 4.0
        assert report[branch]["aligned_action_row_steps"] == 2.0


def test_closed_loop_context_is_covariant_to_support_row_permutation() -> None:
    identities = (0.25, 0.75)
    permutation = torch.tensor([1, 0], dtype=torch.int64)
    support = _support(2, identities=identities)
    permuted_support = support.index_select(permutation)
    base_policy = _SupportIdentityPolicy()
    permuted_policy = _SupportIdentityPolicy()

    base = evaluate_support_query_closed_loop(
        _LinearStrengthEnv(1.0),
        _LinearStrengthEnv(0.7),
        base_policy,  # type: ignore[arg-type]
        support,
        support_command=_support_command(2),
        steps=2,
        device="cpu",
    )
    permuted = evaluate_support_query_closed_loop(
        _LinearStrengthEnv(1.0),
        _LinearStrengthEnv(0.7),
        permuted_policy,  # type: ignore[arg-type]
        permuted_support,
        support_command=_support_command(2).index_select(0, permutation),
        steps=2,
        device="cpu",
    )

    base_trace = np.asarray(base["context"]["delta_z_trace"], dtype=np.float32)
    permuted_trace = np.asarray(permuted["context"]["delta_z_trace"], dtype=np.float32)
    expected_base = np.array([[[0.25], [0.75]], [[0.25], [0.75]]], dtype=np.float32)
    np.testing.assert_array_equal(base_trace, expected_base)
    np.testing.assert_array_equal(permuted_trace, expected_base[:, [1, 0]])
    assert not np.array_equal(permuted_trace, base_trace)
    assert len({id(call[0]) for call in base_policy.context_encoder.calls}) == 1
    assert len({id(call[0]) for call in permuted_policy.context_encoder.calls}) == 1
    torch.testing.assert_close(
        base_policy.context_encoder.calls[0][0].realized_state[:, 0, 0],
        torch.tensor(identities),
    )
    torch.testing.assert_close(
        permuted_policy.context_encoder.calls[0][0].realized_state[:, 0, 0],
        torch.tensor(identities)[permutation],
    )
    for metric in (
        "actor_observation_mse",
        "base_position_mse_m2",
        "local_velocity_mse_m2ps2",
        "action_mse",
    ):
        assert permuted["fault_context_distance_to_healthy"][metric] == pytest.approx(
            base["fault_context_distance_to_healthy"][metric]
        )


def test_closed_loop_context_is_evaluated_against_healthy_trajectory() -> None:
    policy = _FixedPolicy()
    support = SupportContextBatch(
        target_future=torch.zeros(2, 1, 1, 1),
        realized_state=torch.zeros(2, 1, 1),
        executed_action=torch.zeros(2, 1, 1),
    )

    report = evaluate_support_query_closed_loop(
        _LinearStrengthEnv(1.0),
        _LinearStrengthEnv(0.7),
        policy,  # type: ignore[arg-type]
        support,
        support_command=torch.tensor([[0.4, 0.0, 0.0]]).expand(2, -1),
        steps=4,
        device="cpu",
    )

    assert len(policy.context_encoder.calls) == 4
    assert len({id(call[0]) for call in policy.context_encoder.calls}) == 1
    observed_history = torch.stack([call[1] for call in policy.context_encoder.calls])
    action_history = torch.stack([call[2] for call in policy.context_encoder.calls])
    torch.testing.assert_close(
        observed_history[:, :, -1, 0],
        torch.arange(4, dtype=torch.float32)[:, None].expand(-1, 2),
    )
    torch.testing.assert_close(
        action_history[0],
        torch.zeros_like(action_history[0]),
    )
    torch.testing.assert_close(
        action_history[1:, :, -1, 0],
        torch.full((3, 2), 10.0 / 7.0),
    )
    assert report["context"]["delta_z_trace_shape"] == [4, 2, 1]
    assert report["fault_zero_distance_to_healthy"]["actor_observation_mse"] > 0.0
    assert report["fault_context_distance_to_healthy"]["actor_observation_mse"] < 1.0e-12
    assert report["verdict"]["context_closer_to_healthy"] is True
    assert report["verdict"]["improved_distance_metric_count"] >= 5


def test_closed_loop_evaluator_preserves_frozen_policy_parameter_lifecycle() -> None:
    policy = _FixedPolicy().eval()
    parameters_before = tuple(policy.parameters())
    assert len(parameters_before) == 1
    identities_before = tuple(id(parameter) for parameter in parameters_before)
    values_before = tuple(parameter.detach().clone() for parameter in parameters_before)
    requires_grad_before = tuple(parameter.requires_grad for parameter in parameters_before)
    training_before = tuple((id(module), module.training) for module in policy.modules())

    evaluate_support_query_closed_loop(
        _LinearStrengthEnv(1.0),
        _LinearStrengthEnv(0.7),
        policy,  # type: ignore[arg-type]
        _support(2),
        support_command=_support_command(2),
        steps=4,
        device="cpu",
    )

    parameters_after = tuple(policy.parameters())
    assert len(policy.context_encoder.calls) == 4
    assert tuple(id(parameter) for parameter in parameters_after) == identities_before
    assert tuple(parameter.requires_grad for parameter in parameters_after) == requires_grad_before
    assert tuple((id(module), module.training) for module in policy.modules()) == training_before
    for observed, expected in zip(parameters_after, values_before, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    # Controlled counterexample: the exact oracle distinguishes a one-parameter mutation.
    with torch.no_grad():
        parameters_after[0].add_(1.0)
    assert not torch.equal(parameters_after[0], values_before[0])
    with torch.no_grad():
        parameters_after[0].copy_(values_before[0])


def test_online_support_is_collected_without_context_then_reset_for_repair() -> None:
    policy = _FixedPolicy()
    healthy = _LinearStrengthEnv(1.0)
    fault = _LinearStrengthEnv(0.7)

    report = evaluate_online_support_closed_loop(
        healthy,
        fault,
        policy,  # type: ignore[arg-type]
        steps=4,
        device="cpu",
    )

    assert report["support"] == {
        "source": "same_fault_environment_no_context_rollout",
        "length": 1,
        "reset_before_repaired_rollout": True,
    }
    assert report["fault_context_distance_to_healthy"]["actor_observation_mse"] < 1.0e-12
    assert report["verdict"]["context_closer_to_healthy"] is True


def test_paired_environment_factory_closes_both_envs_when_fault_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _InitEnv:
        def __init__(self, *, fails: bool) -> None:
            self.fails = fails
            self.closed = False

        def init_state(self) -> None:
            if self.fails:
                raise RuntimeError("fault init failed")

        def close(self) -> None:
            self.closed = True

    healthy = _InitEnv(fails=False)
    fault = _InitEnv(fails=True)
    environments = iter((healthy, fault))
    runtime_module = __import__(
        "unilab.algos.torch.fada_context.support_query_runtime", fromlist=["unused"]
    )
    monkeypatch.setattr(runtime_module, "_compose_task", lambda root, task: object())
    monkeypatch.setattr(
        runtime_module,
        "_fixed_fault_override",
        lambda root, task: {"domain_rand": {"actuator_strength": {"multipliers": [1.0] * 29}}},
    )
    monkeypatch.setattr(runtime_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(runtime_module, "apply_training_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_module, "create_env", lambda *args, **kwargs: next(environments))

    with pytest.raises(RuntimeError, match="fault init failed"):
        create_fixed_fault_paired_environments(
            Path("/tmp/repo"),
            SimpleNamespace(task_config="fault"),
            num_envs=2,
            seed=7,
        )

    assert healthy.closed is True
    assert fault.closed is True
