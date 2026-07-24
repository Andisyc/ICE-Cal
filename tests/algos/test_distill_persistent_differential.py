from __future__ import annotations

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill.collector import (
    collect_distillation_dataset_from_env,
    collect_transition_distillation_dataset_from_env,
)
from unilab.algos.torch.distill.performance import (
    DISTILLATION_METRICS_SCHEMA_VERSION,
    DistillationStageObservation,
)
from unilab.algos.torch.distill.persistent_resources import (
    PersistentResourceCache,
    PersistentResourceIdentity,
)


class _ConstantPolicy(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0], 3), self.value, dtype=obs.dtype)


class _IncrementingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        value = self.value
        self.value += 1.0
        return value


def _performance_observations(dataset) -> tuple[DistillationStageObservation, ...]:
    assert (
        dataset.metadata["performance_metrics_schema_version"]
        == DISTILLATION_METRICS_SCHEMA_VERSION
    )
    return tuple(
        DistillationStageObservation.from_dict(payload)
        for payload in dataset.metadata["performance_stage_observations"]
    )


class _RoleEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.action_space = type("ActionSpace", (), {"shape": (3,)})()
        self.reset_calls = 0
        self.step_calls = 0
        self.state = None
        self.commands = np.asarray([[0.4, 0.0, 0.0]] * 2, dtype=np.float32)

    def _obs(self, offset: int):
        return {"obs": np.arange(16, dtype=np.float32).reshape(2, 8) + offset}

    def init_state(self) -> None:
        self.state = object()

    def reset(self, _env_indices):
        self.reset_calls += 1
        self.step_calls = 0
        return self._obs(0), {"commands": self.commands.copy()}

    def step(self, actions):
        assert actions.shape == (2, 3)
        self.step_calls += 1
        return type(
            "State",
            (),
            {"obs": self._obs(self.step_calls), "info": {"commands": self.commands.copy()}},
        )()

    def close(self) -> None:
        pass


class _TransitionEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.action_space = type("ActionSpace", (), {"shape": (3,)})()
        self.reset_calls = 0
        self.step_calls = 0
        self.commands = np.zeros((2, 3), dtype=np.float32)
        self.state = None

    def _obs(self, offset: int):
        return {"obs": np.arange(16, dtype=np.float32).reshape(2, 8) + offset}

    def _state(self):
        return type(
            "State",
            (),
            {
                "obs": self._obs(self.step_calls),
                "info": {"commands": self.commands},
                "terminated": np.zeros((2,), dtype=bool),
                "truncated": np.zeros((2,), dtype=bool),
                "final_observation": None,
            },
        )()

    def init_state(self) -> None:
        self.state = self._state()

    def reset(self, env_indices):
        self.reset_calls += 1
        self.step_calls = 0
        self.commands[np.asarray(env_indices, dtype=np.int32)] = 0.0
        self.state = self._state()
        return self._obs(0), {"commands": self.commands.copy()}

    def refresh_state(self):
        self.state = self._state()
        return self.state

    def step(self, actions):
        assert actions.shape == (2, 3)
        self.step_calls += 1
        self.state = self._state()
        return self.state

    def close(self) -> None:
        pass


def _identity(role: str) -> PersistentResourceIdentity:
    return PersistentResourceIdentity(
        task_owner=f"g1_{role}/mujoco",
        task_name=f"G1{role.title().replace('_', '')}",
        sim_backend="mujoco",
        env_cfg_fingerprint=f"env-{role}",
        num_envs=2,
        teacher_checkpoint_path=f"/teacher/{role}.pt",
        teacher_checkpoint_sha256=f"teacher-hash-{role}",
        teacher_spec_fingerprint=f"teacher-spec-{role}",
    )


def _semantic_snapshot(dataset) -> dict:
    return {
        "num_samples": dataset.num_samples,
        "student_obs_dim": dataset.student_obs_dim,
        "teacher_obs_dim": dataset.teacher_obs_dim,
        "role_labels": dataset.role_labels,
        "command_intents": dataset.command_intents,
        "scenario_labels": dataset.scenario_labels,
        "transition_ages": (
            None if dataset.transition_ages is None else dataset.transition_ages.tolist()
        ),
        "teacher_identity": dataset.metadata["teacher_checkpoint_sha256"],
        "teacher_actions": (
            None if dataset.teacher_actions is None else dataset.teacher_actions.tolist()
        ),
    }


def test_persistent_role_collection_matches_legacy_semantics_without_double_reset() -> None:
    teacher = _ConstantPolicy(0.2)
    student = _ConstantPolicy(-0.3)
    metadata = {"teacher_checkpoint_sha256": "teacher-hash-walk_flat"}
    legacy_env = _RoleEnv()
    legacy = collect_distillation_dataset_from_env(
        legacy_env,
        num_samples=4,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        action_mode="student_policy",
        teacher_policy=teacher,
        rollout_policy=student,
        command_sample_filter="active",
        role_label="walk_flat",
        metadata=metadata,
    )

    persistent_envs: list[_RoleEnv] = []
    cache = PersistentResourceCache(
        teacher_factory=lambda _identity: teacher,
        env_factory=lambda _identity: persistent_envs.append(_RoleEnv()) or persistent_envs[-1],
    )
    try:
        persistent = cache.run_request(
            _identity("walk_flat"),
            lambda env, cached_teacher, reset_output: collect_distillation_dataset_from_env(
                env,
                num_samples=4,
                expected_student_obs_dim=8,
                expected_teacher_obs_dim=8,
                action_mode="student_policy",
                teacher_policy=cached_teacher,
                rollout_policy=student,
                command_sample_filter="active",
                role_label="walk_flat",
                metadata=metadata,
                initial_reset=reset_output,
            ),
        )
    finally:
        cache.close()

    assert _semantic_snapshot(persistent) == _semantic_snapshot(legacy)
    assert "performance_stage_observations" not in legacy.metadata
    assert "performance_stage_observations" not in persistent.metadata
    assert legacy_env.reset_calls == 1
    assert persistent_envs[0].reset_calls == 1


def test_role_collector_emits_exact_opt_in_performance_observations() -> None:
    dataset = collect_distillation_dataset_from_env(
        _RoleEnv(),
        num_samples=4,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        action_mode="student_policy",
        teacher_policy=_ConstantPolicy(0.2),
        rollout_policy=_ConstantPolicy(-0.3),
        command_sample_filter="active",
        role_label="walk_flat",
        performance_clock=_IncrementingClock(),
    )

    observations = _performance_observations(dataset)
    assert tuple(item.stage for item in observations) == (
        "teacher_inference",
        "student_inference",
        "env_step",
        "tensor_pack",
    )
    assert tuple(item.duration_seconds for item in observations) == (2.0, 2.0, 1.0, 3.0)
    assert tuple(item.row_count for item in observations) == (4, 4, 0, 4)
    assert tuple(item.env_step_count for item in observations) == (0, 0, 1, 0)


def test_transition_collector_emits_exact_opt_in_performance_observations() -> None:
    dataset = collect_transition_distillation_dataset_from_env(
        _TransitionEnv(),
        num_samples=8,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        walking_teacher_policy=_ConstantPolicy(0.1),
        standing_teacher_policy=_ConstantPolicy(0.2),
        rollout_policy=_ConstantPolicy(-0.3),
        pre_switch_steps=2,
        walk_command=(0.4, 0.0, 0.0),
        performance_clock=_IncrementingClock(),
    )

    observations = _performance_observations(dataset)
    assert tuple(item.stage for item in observations) == (
        "teacher_inference",
        "student_inference",
        "env_step",
        "tensor_pack",
    )
    assert tuple(item.duration_seconds for item in observations) == (4.0, 4.0, 3.0, 5.0)
    assert tuple(item.row_count for item in observations) == (16, 8, 0, 8)
    assert tuple(item.env_step_count for item in observations) == (0, 0, 3, 0)


def test_persistent_transition_collection_matches_legacy_semantics_without_double_reset() -> None:
    walking_teacher = _ConstantPolicy(0.1)
    standing_teacher = _ConstantPolicy(0.2)
    student = _ConstantPolicy(-0.3)
    metadata = {"teacher_checkpoint_sha256": "teacher-hash-walk_flat"}
    legacy_env = _TransitionEnv()
    legacy = collect_transition_distillation_dataset_from_env(
        legacy_env,
        num_samples=8,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        walking_teacher_policy=walking_teacher,
        standing_teacher_policy=standing_teacher,
        rollout_policy=student,
        pre_switch_steps=2,
        walk_command=(0.4, 0.0, 0.0),
        metadata=metadata,
    )

    persistent_envs: list[_TransitionEnv] = []
    cache = PersistentResourceCache(
        teacher_factory=lambda _identity: walking_teacher,
        env_factory=lambda _identity: (
            persistent_envs.append(_TransitionEnv()) or persistent_envs[-1]
        ),
    )
    try:
        persistent = cache.run_request(
            _identity("walk_flat"),
            lambda env, cached_teacher, reset_output: (
                collect_transition_distillation_dataset_from_env(
                    env,
                    num_samples=8,
                    expected_student_obs_dim=8,
                    expected_teacher_obs_dim=8,
                    walking_teacher_policy=cached_teacher,
                    standing_teacher_policy=standing_teacher,
                    rollout_policy=student,
                    pre_switch_steps=2,
                    walk_command=(0.4, 0.0, 0.0),
                    metadata=metadata,
                    initial_reset=reset_output,
                )
            ),
        )
    finally:
        cache.close()

    assert _semantic_snapshot(persistent) == _semantic_snapshot(legacy)
    assert "performance_stage_observations" not in legacy.metadata
    assert "performance_stage_observations" not in persistent.metadata
    assert legacy_env.reset_calls == 1
    assert persistent_envs[0].reset_calls == 1


def test_persistent_collection_rejects_malformed_initial_reset() -> None:
    with pytest.raises(ValueError, match="two-item"):
        collect_distillation_dataset_from_env(
            _RoleEnv(),
            num_samples=1,
            expected_student_obs_dim=8,
            expected_teacher_obs_dim=8,
            initial_reset=(None,),  # type: ignore[arg-type]
        )
