from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill import (
    BehaviorDistillationTrainer,
    DistillationBatch,
    MoEStudentPolicy,
    build_distillation_dataset,
    build_multitask_distillation_dataset,
    collect_distillation_dataset_from_env,
    load_distillation_student_policy,
    save_distillation_checkpoint,
    save_distillation_dataset,
)
from unilab.algos.torch.distill.dagger import _aggregate_dagger_datasets


def _height_dataset(
    *,
    role: str,
    heights: torch.Tensor,
    active: bool,
) -> object:
    num_samples = int(heights.shape[0])
    obs = torch.randn(num_samples, 99)
    obs[:, 96:97] = heights
    commands = torch.zeros(num_samples, 3)
    if active:
        commands[:, 0] = 0.4
    intent = "active" if active else "inactive"
    return build_distillation_dataset(
        obs,
        obs.clone(),
        expected_student_obs_dim=99,
        expected_teacher_obs_dim=99,
        expected_teacher_action_dim=29,
        role_labels=(role,) * num_samples,
        teacher_actions=torch.randn(num_samples, 29),
        commands=commands,
        target_height=heights,
        command_intents=(intent,) * num_samples,
    )


def test_target_height_survives_save_batch_multitask_and_dagger(tmp_path) -> None:
    walk = _height_dataset(
        role="walk",
        heights=torch.full((2, 1), 0.754),
        active=True,
    )
    stand_height = _height_dataset(
        role="stand_height",
        heights=torch.tensor([[0.65], [0.70], [0.754]]),
        active=False,
    )
    walk_path = tmp_path / "walk.pt"
    stand_path = tmp_path / "stand_height.pt"
    save_distillation_dataset(walk_path, walk)
    save_distillation_dataset(stand_path, stand_height)

    merged = build_multitask_distillation_dataset(
        (
            {"path": walk_path, "role": "walk"},
            {"path": stand_path, "role": "stand_height"},
        ),
        expected_student_obs_dim=99,
        expected_teacher_obs_dim=99,
        expected_teacher_action_dim=29,
    )

    assert merged.target_height is not None
    torch.testing.assert_close(
        merged.target_height,
        torch.tensor([[0.754], [0.754], [0.65], [0.70], [0.754]]),
    )
    batch = merged.as_batch(start=1, batch_size=3)
    assert batch.target_height is not None
    torch.testing.assert_close(batch.target_height, merged.target_height[1:4])

    dagger = _aggregate_dagger_datasets((walk, stand_height))
    assert dagger.target_height is not None
    torch.testing.assert_close(dagger.target_height, merged.target_height)


def test_target_height_schema_and_98d_99d_mixing_fail_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="target_height must have shape"):
        build_distillation_dataset(
            torch.zeros(2, 99),
            torch.zeros(2, 99),
            target_height=torch.zeros(2),
        )

    legacy_path = tmp_path / "legacy_98.pt"
    height_path = tmp_path / "height_99.pt"
    save_distillation_dataset(
        legacy_path,
        build_distillation_dataset(
            torch.zeros(2, 98),
            torch.zeros(2, 98),
            teacher_actions=torch.zeros(2, 29),
        ),
    )
    save_distillation_dataset(
        height_path,
        _height_dataset(
            role="stand_height",
            heights=torch.full((2, 1), 0.7),
            active=False,
        ),
    )

    with pytest.raises(ValueError, match="student_obs dim mismatch"):
        build_multitask_distillation_dataset(
            (
                {"path": legacy_path, "role": "stand"},
                {"path": height_path, "role": "stand_height"},
            ),
            expected_student_obs_dim=99,
            expected_teacher_obs_dim=99,
            expected_teacher_action_dim=29,
        )

    no_height_path = tmp_path / "no_height_99.pt"
    save_distillation_dataset(
        no_height_path,
        build_distillation_dataset(
            torch.zeros(2, 99),
            torch.zeros(2, 99),
            teacher_actions=torch.zeros(2, 29),
            commands=torch.zeros(2, 3),
            command_intents=("inactive",) * 2,
        ),
    )
    with pytest.raises(ValueError, match="all include target_height"):
        build_multitask_distillation_dataset(
            (
                {"path": no_height_path, "role": "walk"},
                {"path": height_path, "role": "stand_height"},
            ),
            expected_student_obs_dim=99,
            expected_teacher_obs_dim=99,
            expected_teacher_action_dim=29,
        )


class _HeightInfoEnv:
    num_envs = 2
    action_space = SimpleNamespace(shape=(29,))

    def reset(self, _indices):
        height = np.asarray([[0.65], [0.754]], dtype=np.float32)
        obs = np.zeros((2, 99), dtype=np.float32)
        obs[:, 96:97] = height
        return {"obs": obs}, {
            "commands": np.zeros((2, 3), dtype=np.float32),
            "height_commands": height,
        }


def test_collector_persists_commanded_height_from_info_without_stepping_env() -> None:
    dataset = collect_distillation_dataset_from_env(
        _HeightInfoEnv(),
        num_samples=2,
        expected_student_obs_dim=99,
        expected_teacher_obs_dim=99,
        target_height_info_key="height_commands",
    )

    assert dataset.target_height is not None
    torch.testing.assert_close(
        dataset.target_height,
        torch.tensor([[0.65], [0.754]]),
    )
    torch.testing.assert_close(dataset.student_obs[:, 96:97], dataset.target_height)


def _clone_optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
) -> dict[torch.nn.Parameter, dict[str, object]]:
    result: dict[torch.nn.Parameter, dict[str, object]] = {}
    for parameter in parameters:
        state = optimizer.state[parameter]
        result[parameter] = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
    return result


def test_two_expert_selected_update_keeps_inactive_parameters_and_optimizer_state() -> None:
    student = MoEStudentPolicy(
        obs_dim=99,
        action_dim=29,
        num_experts=2,
        expert_hidden_dims=(8,),
        routing_mode="hard",
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(99, 29),
        optimizer=optimizer,
        role_expert_targets={"walk": 0, "stand_height": 1},
        command_intent_expert_targets={"active": 0, "inactive": 1},
        expert_behavior_loss_source="command_intent",
    )
    obs = torch.randn(4, 99)
    trainer.update(
        DistillationBatch(
            student_obs=obs,
            teacher_obs=obs,
            teacher_actions=torch.ones(4, 29),
            role_labels=("stand_height",) * 4,
            command_intents=("inactive",) * 4,
        )
    )
    inactive_parameters = tuple(student.experts[1].parameters())
    parameter_before = tuple(parameter.detach().clone() for parameter in inactive_parameters)
    optimizer_before = _clone_optimizer_state(optimizer, inactive_parameters)

    for _ in range(3):
        trainer.update(
            DistillationBatch(
                student_obs=obs,
                teacher_obs=obs,
                teacher_actions=-torch.ones(4, 29),
                role_labels=("walk",) * 4,
                command_intents=("active",) * 4,
            )
        )

    for parameter, expected in zip(inactive_parameters, parameter_before, strict=True):
        assert torch.equal(parameter, expected)
    optimizer_after = _clone_optimizer_state(optimizer, inactive_parameters)
    for parameter in inactive_parameters:
        assert optimizer_after[parameter].keys() == optimizer_before[parameter].keys()
        for key, expected in optimizer_before[parameter].items():
            actual = optimizer_after[parameter][key]
            if isinstance(expected, torch.Tensor):
                assert isinstance(actual, torch.Tensor)
                assert torch.equal(actual, expected)
            else:
                assert actual == expected


def test_two_expert_checkpoint_strict_roundtrip_outputs_finite_29d_action(tmp_path) -> None:
    torch.manual_seed(17)
    student = MoEStudentPolicy(
        obs_dim=99,
        action_dim=29,
        num_experts=2,
        expert_hidden_dims=(8,),
        router_hidden_dims=(),
        routing_mode="hard",
    )
    checkpoint_path = tmp_path / "synthetic_two_expert.pt"
    runtime_cfg = {
        "student_model_type": "moe",
        "student_obs_dim": 99,
        "student_action_dim": 29,
        "student_num_experts": 2,
        "student_expert_hidden_dims": [8],
        "student_router_hidden_dims": [],
        "student_routing_mode": "hard",
        "student_router_temperature": 1.0,
        "student_activation": "elu",
        "student_squash_action": True,
        "role_expert_targets": {"walk": 0, "stand_height": 1},
        "command_intent_expert_targets": {"active": 0, "inactive": 1},
    }
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=1,
        distill_runtime_cfg=runtime_cfg,
    )

    loaded = load_distillation_student_policy(checkpoint_path, device="cpu")
    obs = torch.randn(3, 99)
    with torch.no_grad():
        expected = student(obs)
        actual = loaded.policy(obs)

    assert loaded.policy.num_experts == 2
    assert actual.shape == (3, 29)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
