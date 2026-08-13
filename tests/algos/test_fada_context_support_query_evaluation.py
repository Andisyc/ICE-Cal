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

    def forward(self, support: SupportContextBatch) -> torch.Tensor:
        support.validate(self.config, support_length=1)
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
        self.context_encoder = _FixedContext(self.config)
        self.planner = _UnitPlanner()
        self.idm = _UnitIDM()

    def act_with_context(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
        delta_z: torch.Tensor,
    ) -> SimpleNamespace:
        del observation_history, action_history, command
        return SimpleNamespace(action=torch.ones_like(delta_z[:, :1]) + delta_z[:, :1])


class _UnitPlanner(nn.Module):
    def forward(
        self, observation_history: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        del command
        return torch.zeros(
            observation_history.shape[0], 1, 1, device=observation_history.device
        )


class _UnitIDM(nn.Module):
    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        del action_history, future
        return torch.ones(
            observation_history.shape[0], 1, 1, device=observation_history.device
        )


class _LinearStrengthEnv:
    def __init__(self, strength: float, *, rows: int = 2) -> None:
        self.num_envs = rows
        self._strength = float(strength)
        self._position = np.zeros((rows, 1), dtype=np.float32)
        self._velocity = np.zeros((rows, 1), dtype=np.float32)
        self._autoreset = True
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
        steps=4,
        device="cpu",
    )

    assert report["fault_zero_distance_to_healthy"]["actor_observation_mse"] > 0.0
    assert report["fault_context_distance_to_healthy"]["actor_observation_mse"] < 1.0e-12
    assert report["verdict"]["context_closer_to_healthy"] is True
    assert report["verdict"]["improved_distance_metric_count"] >= 5


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
