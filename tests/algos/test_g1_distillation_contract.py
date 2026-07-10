from __future__ import annotations

import numpy as np
import pytest
import torch


def test_behavior_distillation_update_detaches_teacher_and_updates_student() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MLPStudentPolicy,
    )

    torch.manual_seed(7)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    teacher = torch.nn.Linear(7, 3)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
    )
    batch = DistillationBatch(
        student_obs=torch.randn(4, 5),
        teacher_obs=torch.randn(4, 7),
    )
    before = {name: value.detach().clone() for name, value in student.state_dict().items()}

    stats = trainer.update(batch)

    assert stats.update_count == 1
    assert stats.loss > 0.0
    assert stats.student_grad_norm > 0.0
    assert stats.teacher_action_requires_grad is False
    assert stats.student_action_shape == (4, 3)
    assert stats.teacher_action_shape == (4, 3)
    assert all(param.grad is None for param in teacher.parameters())
    assert any(
        not torch.allclose(before[name], value)
        for name, value in student.state_dict().items()
    )


def test_behavior_distillation_update_uses_cached_teacher_actions() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MLPStudentPolicy,
    )

    class RaisingTeacher(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            del obs
            raise AssertionError("cached teacher_action path must not call teacher")

    torch.manual_seed(13)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=RaisingTeacher(),
        optimizer=optimizer,
    )
    batch = DistillationBatch(
        student_obs=torch.randn(4, 5),
        teacher_obs=torch.empty(4, 0),
        teacher_actions=torch.randn(4, 3, requires_grad=True),
    )

    stats = trainer.update(batch)

    assert stats.update_count == 1
    assert stats.loss > 0.0
    assert stats.student_grad_norm > 0.0
    assert stats.teacher_action_shape == (4, 3)
    assert stats.teacher_action_requires_grad is False
    assert stats.teacher_action_source == "cached"


def test_behavior_distillation_checkpoint_roundtrip(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        load_distillation_checkpoint,
        save_distillation_checkpoint,
    )

    torch.manual_seed(11)
    source = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    target = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    checkpoint_path = tmp_path / "nested" / "distill_model.pt"

    save_distillation_checkpoint(
        checkpoint_path,
        student=source,
        agent_steps=16,
        teacher_metadata={"algo": "sac", "task": "G1WalkHeight"},
        distill_runtime_cfg={"loss_type": "mse"},
    )
    checkpoint = load_distillation_checkpoint(target, checkpoint_path)

    assert checkpoint_path.is_file()
    assert checkpoint["agent_steps"] == 16
    assert checkpoint["teacher_metadata"] == {"algo": "sac", "task": "G1WalkHeight"}
    assert checkpoint["distill_runtime_cfg"] == {"loss_type": "mse"}
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        assert torch.allclose(source_param, target_param)


def test_behavior_distillation_rejects_batch_shape_mismatch() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MLPStudentPolicy,
    )

    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    teacher = torch.nn.Linear(7, 3)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
    )

    try:
        trainer.update(
            DistillationBatch(
                student_obs=torch.randn(4, 5),
                teacher_obs=torch.randn(5, 7),
            )
        )
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:
        raise AssertionError("expected shape mismatch to raise ValueError")


def test_moe_student_policy_routes_and_mixes_expert_actions() -> None:
    from unilab.algos.torch.distill import MoEStudentPolicy

    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=3,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        routing_mode="soft",
        squash_action=False,
    )
    with torch.no_grad():
        for expert, bias in zip(
            student.experts,
            (
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.0, 2.0]),
                torch.tensor([-1.0, 1.0]),
            ),
            strict=True,
        ):
            expert.net[-1].weight.zero_()
            expert.net[-1].bias.copy_(bias)
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()

    obs = torch.zeros(2, 4)
    output = student(obs, return_diagnostics=True)

    assert output.action.shape == (2, 2)
    assert output.router_logits.shape == (2, 3)
    assert output.route_probs.shape == (2, 3)
    assert output.expert_actions.shape == (2, 3, 2)
    assert output.selected_expert is None
    assert torch.allclose(output.route_probs, torch.full((2, 3), 1.0 / 3.0))
    assert torch.allclose(output.expert_usage, torch.full((3,), 2.0 / 3.0))
    assert torch.allclose(output.action, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

    with torch.no_grad():
        student.router[-1].bias.copy_(torch.tensor([-2.0, 3.0, -1.0]))
    hard_output = student(obs, hard_routing=True, return_diagnostics=True)

    assert torch.equal(hard_output.selected_expert, torch.ones(2, dtype=torch.long))
    assert torch.allclose(hard_output.expert_usage, torch.tensor([0.0, 2.0, 0.0]))
    assert torch.allclose(hard_output.action, torch.tensor([[0.0, 2.0], [0.0, 2.0]]))


def test_moe_student_policy_soft_route_backpropagates_router_and_experts() -> None:
    from unilab.algos.torch.distill import MoEStudentPolicy

    torch.manual_seed(41)
    student = MoEStudentPolicy(
        obs_dim=5,
        action_dim=3,
        num_experts=2,
        expert_hidden_dims=(8,),
        router_hidden_dims=(4,),
    )
    output = student(torch.randn(4, 5), return_diagnostics=True)
    loss = output.action.pow(2).mean() + output.route_probs[:, 0].mean()

    loss.backward()

    router_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for param in student.router.parameters()
        if param.grad is not None
    )
    expert_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for expert in student.experts
        for param in expert.parameters()
        if param.grad is not None
    )
    assert output.action.shape == (4, 3)
    assert output.action.requires_grad is True
    assert router_grad_norm > 0.0
    assert expert_grad_norm > 0.0


def test_moe_student_policy_rejects_bad_contract() -> None:
    from unilab.algos.torch.distill import MoEStudentPolicy

    with pytest.raises(ValueError, match="num_experts"):
        MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=1)

    with pytest.raises(ValueError, match="router_temperature"):
        MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=2, router_temperature=0.0)

    student = MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=2)
    with pytest.raises(ValueError, match="Student obs dim mismatch"):
        student(torch.zeros(3, 5))


def test_moe_distillation_trainer_records_aux_loss_and_usage() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MoEStudentPolicy,
    )

    torch.manual_seed(43)
    student = MoEStudentPolicy(
        obs_dim=5,
        action_dim=3,
        num_experts=2,
        expert_hidden_dims=(8,),
        router_hidden_dims=(4,),
    )
    teacher = torch.nn.Linear(7, 3)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        aux_loss_coef=0.25,
    )
    stats = trainer.update(
        DistillationBatch(
            student_obs=torch.randn(4, 5),
            teacher_obs=torch.randn(4, 7),
        )
    )

    router_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for param in student.router.parameters()
        if param.grad is not None
    )
    expert_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for expert in student.experts
        for param in expert.parameters()
        if param.grad is not None
    )
    assert stats.behavior_loss > 0.0
    assert stats.aux_loss >= 0.0
    assert stats.loss == pytest.approx(stats.behavior_loss + 0.25 * stats.aux_loss)
    assert stats.student_action_shape == (4, 3)
    assert stats.teacher_action_shape == (4, 3)
    assert stats.expert_usage is not None
    assert len(stats.expert_usage) == 2
    assert sum(stats.expert_usage) == pytest.approx(4.0)
    assert stats.route_entropy is not None
    assert stats.route_entropy >= 0.0
    assert router_grad_norm > 0.0
    assert expert_grad_norm > 0.0


def test_moe_distillation_trainer_applies_role_conditioned_router_loss() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MoEStudentPolicy,
    )

    torch.manual_seed(47)
    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=3,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    with torch.no_grad():
        for expert in student.experts:
            expert.net[-1].weight.zero_()
            expert.net[-1].bias.zero_()
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        role_loss_coef=0.5,
        role_expert_targets={"stand": 0, "walk": 1, "height": 2},
    )

    stats = trainer.update(
        DistillationBatch(
            student_obs=torch.eye(4),
            teacher_obs=torch.empty(4, 0),
            role_labels=("stand", "stand", "walk", "height"),
            teacher_actions=torch.zeros(4, 2),
        )
    )

    router_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for param in student.router.parameters()
        if param.grad is not None
    )
    assert stats.behavior_loss == pytest.approx(0.0)
    assert stats.aux_loss == pytest.approx(0.0)
    assert stats.role_loss > 0.0
    assert stats.role_target_count == 4
    assert stats.loss == pytest.approx(0.5 * stats.role_loss)
    assert router_grad_norm > 0.0


def test_moe_distillation_trainer_applies_command_intent_router_loss() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MoEStudentPolicy,
    )

    torch.manual_seed(47)
    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    with torch.no_grad():
        for expert in student.experts:
            expert.net[-1].weight.zero_()
            expert.net[-1].bias.zero_()
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        command_intent_loss_coef=0.75,
        command_intent_expert_targets={"inactive": 0, "active": 1},
    )

    stats = trainer.update(
        DistillationBatch(
            student_obs=torch.eye(4),
            teacher_obs=torch.empty(4, 0),
            command_intents=("inactive", "inactive", "active", "active"),
            teacher_actions=torch.zeros(4, 2),
        )
    )

    router_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for param in student.router.parameters()
        if param.grad is not None
    )
    assert stats.behavior_loss == pytest.approx(0.0)
    assert stats.aux_loss == pytest.approx(0.0)
    assert stats.role_loss == pytest.approx(0.0)
    assert stats.command_intent_loss > 0.0
    assert stats.command_intent_target_count == 4
    assert stats.loss == pytest.approx(0.75 * stats.command_intent_loss)
    assert router_grad_norm > 0.0


def test_moe_role_conditioned_router_loss_fails_closed() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MLPStudentPolicy,
        MoEStudentPolicy,
    )

    student = MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=2)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="role_expert_targets"):
        BehaviorDistillationTrainer(
            student=student,
            teacher=torch.nn.Identity(),
            optimizer=optimizer,
            role_loss_coef=0.1,
        )

    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        role_loss_coef=0.1,
        role_expert_targets={"stand": 0},
    )
    with pytest.raises(ValueError, match="role_labels"):
        trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                teacher_actions=torch.zeros(2, 2),
            )
        )
    with pytest.raises(ValueError, match="unmapped role label"):
        trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                role_labels=("stand", "walk"),
                teacher_actions=torch.zeros(2, 2),
            )
        )

    mlp = MLPStudentPolicy(obs_dim=4, action_dim=2, hidden_dims=(8,))
    mlp_trainer = BehaviorDistillationTrainer(
        student=mlp,
        teacher=torch.nn.Identity(),
        optimizer=torch.optim.Adam(mlp.parameters(), lr=1e-2),
        role_loss_coef=0.1,
        role_expert_targets={"stand": 0},
    )
    with pytest.raises(TypeError, match="router logits"):
        mlp_trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                role_labels=("stand", "stand"),
                teacher_actions=torch.zeros(2, 2),
            )
        )


def test_moe_command_intent_router_loss_fails_closed() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MLPStudentPolicy,
        MoEStudentPolicy,
    )

    student = MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=2)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="command_intent_expert_targets"):
        BehaviorDistillationTrainer(
            student=student,
            teacher=torch.nn.Identity(),
            optimizer=optimizer,
            command_intent_loss_coef=0.1,
        )

    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        command_intent_loss_coef=0.1,
        command_intent_expert_targets={"inactive": 0},
    )
    with pytest.raises(ValueError, match="command_intents"):
        trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                teacher_actions=torch.zeros(2, 2),
            )
        )
    with pytest.raises(ValueError, match="unmapped command intent"):
        trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                command_intents=("inactive", "active"),
                teacher_actions=torch.zeros(2, 2),
            )
        )

    mlp = MLPStudentPolicy(obs_dim=4, action_dim=2, hidden_dims=(8,))
    mlp_trainer = BehaviorDistillationTrainer(
        student=mlp,
        teacher=torch.nn.Identity(),
        optimizer=torch.optim.Adam(mlp.parameters(), lr=1e-2),
        command_intent_loss_coef=0.1,
        command_intent_expert_targets={"inactive": 0},
    )
    with pytest.raises(TypeError, match="router logits"):
        mlp_trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                command_intents=("inactive", "inactive"),
                teacher_actions=torch.zeros(2, 2),
            )
        )


def test_moe_expert_diagnostics_explain_toy_roles() -> None:
    from unilab.algos.torch.distill import (
        MoEStudentPolicy,
        diagnose_moe_expert_routes,
        moe_diagnostics_to_dict,
    )

    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=3,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        routing_mode="hard",
        squash_action=False,
    )
    with torch.no_grad():
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()
        student.router[-1].weight[0, 0] = 4.0
        student.router[-1].weight[1, 1] = 4.0
        student.router[-1].weight[2, 2] = 4.0

    obs = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 1.5, 0.0],
        ],
        dtype=torch.float32,
    )
    diagnostics = diagnose_moe_expert_routes(
        student,
        obs,
        role_labels=["stand", "stand", "walk", "walk", "recovery", "recovery"],
        hard_routing=True,
        collapse_fraction=0.95,
    )
    by_role = {summary.role: summary for summary in diagnostics.by_role}
    payload = moe_diagnostics_to_dict(diagnostics)

    assert diagnostics.role_labels_present is True
    assert diagnostics.num_samples == 6
    assert diagnostics.num_experts == 3
    assert diagnostics.overall.expert_usage == pytest.approx((2.0, 2.0, 2.0))
    assert diagnostics.overall.collapse_detected is False
    assert by_role["stand"].dominant_expert == 0
    assert by_role["walk"].dominant_expert == 1
    assert by_role["recovery"].dominant_expert == 2
    assert by_role["stand"].expert_fraction == pytest.approx((1.0, 0.0, 0.0))
    assert by_role["walk"].expert_fraction == pytest.approx((0.0, 1.0, 0.0))
    assert by_role["recovery"].expert_fraction == pytest.approx((0.0, 0.0, 1.0))
    assert payload["by_role"][0]["role"] == "recovery"
    assert payload["overall"]["collapse_detected"] is False


def test_moe_expert_diagnostics_flags_router_collapse_and_label_errors() -> None:
    from unilab.algos.torch.distill import MoEStudentPolicy, diagnose_moe_expert_routes

    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=3,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        routing_mode="hard",
        squash_action=False,
    )
    with torch.no_grad():
        student.router[-1].weight.zero_()
        student.router[-1].bias.copy_(torch.tensor([5.0, 0.0, 0.0]))

    obs = torch.zeros(4, 4)
    diagnostics = diagnose_moe_expert_routes(
        student,
        obs,
        role_labels=["stand", "stand", "stand", "stand"],
        hard_routing=True,
        collapse_fraction=0.75,
    )

    assert diagnostics.overall.dominant_expert == 0
    assert diagnostics.overall.expert_fraction == pytest.approx((1.0, 0.0, 0.0))
    assert diagnostics.overall.collapse_detected is True
    assert diagnostics.by_role[0].collapse_detected is True

    with pytest.raises(ValueError, match="role_labels length"):
        diagnose_moe_expert_routes(student, obs, role_labels=["stand"])


def test_moe_expert_semantics_probe_reports_cached_action_error(tmp_path) -> None:
    from scripts.deploy.check_unilab_g1_distill_moe_expert_semantics import run_check

    from unilab.algos.torch.distill import (
        MoEStudentPolicy,
        build_distillation_dataset,
        save_distillation_checkpoint,
        save_distillation_dataset,
    )

    student = MoEStudentPolicy(
        obs_dim=2,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        routing_mode="hard",
        squash_action=False,
    )
    with torch.no_grad():
        for expert, bias in zip(
            student.experts,
            (torch.tensor([0.1, -0.1]), torch.tensor([0.5, 0.2])),
            strict=True,
        ):
            expert.net[-1].weight.zero_()
            expert.net[-1].bias.copy_(bias)
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()
        student.router[-1].weight[0, 0] = 4.0
        student.router[-1].weight[1, 1] = 4.0

    student_obs = torch.tensor(
        [[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    teacher_actions = torch.tensor(
        [[0.1, -0.1], [0.1, -0.1], [0.5, 0.2], [0.5, 0.2]],
        dtype=torch.float32,
    )
    dataset_path = tmp_path / "role_dataset.pt"
    checkpoint_path = tmp_path / "moe_student.pt"
    dataset = build_distillation_dataset(
        student_obs,
        torch.empty(4, 0),
        expected_student_obs_dim=2,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=2,
        teacher_actions=teacher_actions,
        role_labels=("stand", "stand", "walk_flat", "walk_flat"),
    )
    save_distillation_dataset(dataset_path, dataset)
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=4,
        distill_runtime_cfg={
            "student_model_type": "moe",
            "student_obs_dim": 2,
            "teacher_obs_dim": 0,
            "student_action_dim": 2,
            "student_num_experts": 2,
            "student_expert_hidden_dims": [],
            "student_router_hidden_dims": [],
            "student_routing_mode": "hard",
            "student_squash_action": False,
        },
    )

    checks, details = run_check(
        task="g1_walk_flat/mujoco",
        dataset_path=dataset_path,
        student_checkpoint=checkpoint_path,
        hard_routing=True,
    )

    action_imitation = details["moe_expert/action_imitation"]
    dataset_metadata = details["moe_expert/dataset_metadata"]
    assert all(check.level != "FAIL" for check in checks)
    assert "role_labels" not in dataset_metadata
    assert dataset_metadata["role_label_counts"] == {"stand": 2, "walk_flat": 2}
    assert dataset_metadata["role_label_count_total"] == 4
    assert action_imitation["overall"]["mse"] == pytest.approx(0.0)
    assert action_imitation["by_role"]["stand"]["mse"] == pytest.approx(0.0)
    assert action_imitation["by_role"]["walk_flat"]["mse"] == pytest.approx(0.0)
    assert action_imitation["by_role"]["stand"]["student_action_abs_max"] == pytest.approx(0.1)
    assert action_imitation["by_role"]["walk_flat"]["student_action_abs_max"] == pytest.approx(0.5)


def test_distillation_dataset_roundtrip_preserves_obs_batch_contract(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    student_obs = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    teacher_obs = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    dataset = build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        metadata={"source": "offline-fixture"},
    )

    batch = dataset.as_batch(start=1, batch_size=2)
    assert batch.student_obs.shape == (2, 5)
    assert batch.teacher_obs.shape == (2, 7)
    assert torch.equal(batch.student_obs[0], student_obs[1])
    assert torch.equal(batch.teacher_obs[1], teacher_obs[2])

    checkpoint_path = tmp_path / "distill_dataset.pt"
    save_distillation_dataset(checkpoint_path, dataset)
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
    )

    assert restored.num_samples == 4
    assert restored.student_obs_dim == 5
    assert restored.teacher_obs_dim == 7
    assert restored.metadata["source"] == "offline-fixture"
    assert torch.equal(restored.student_obs, student_obs)
    assert torch.equal(restored.teacher_obs, teacher_obs)
    assert restored.role_labels is None


def test_distillation_dataset_roundtrip_preserves_role_labels_contract(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    student_obs = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    teacher_obs = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    role_labels = ("stand", "walk_height", "stand_height", "walk_height")
    dataset = build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        metadata={"source": "role-fixture"},
        role_labels=role_labels,
    )

    batch = dataset.as_batch(start=1, batch_size=2)
    assert batch.role_labels == ("walk_height", "stand_height")

    checkpoint_path = tmp_path / "role_distill_dataset.pt"
    save_distillation_dataset(checkpoint_path, dataset)
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
    )

    assert restored.role_labels == role_labels
    assert restored.metadata["role_labels"] == list(role_labels)
    assert restored.command_intents == ("inactive", "active", "inactive", "active")
    assert restored.metadata["command_intent_inference_source"] == "role_labels"
    assert restored.metadata["command_intent_counts"] == {"active": 2, "inactive": 2}
    assert restored.as_batch(start=2, batch_size=8).role_labels == (
        "stand_height",
        "walk_height",
    )


def test_distillation_dataset_infers_command_intents_from_legacy_roles(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        load_distillation_dataset,
        run_offline_distillation_updates,
    )

    checkpoint_path = tmp_path / "legacy_role_dataset.pt"
    torch.save(
        {
            "student_obs": torch.randn(4, 5),
            "teacher_obs": torch.empty(4, 0),
            "teacher_actions": torch.randn(4, 3),
            "metadata": {"source": "legacy-role-only"},
            "role_labels": ["walk_flat", "stand", "g1_walk_flat", "g1_stand_still"],
            "num_samples": 4,
        },
        checkpoint_path,
    )
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
    )

    assert restored.command_intents == ("active", "inactive", "active", "inactive")
    assert restored.metadata["command_intent_inference_source"] == "role_labels"
    assert restored.metadata["command_intent_counts"] == {"active": 2, "inactive": 2}

    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(0, 3),
        optimizer=torch.optim.Adam(student.parameters(), lr=1e-2),
    )
    result = run_offline_distillation_updates(
        trainer,
        restored,
        batch_size=4,
        max_updates=1,
        balance_key="command_intent",
        balanced_labels=("inactive", "active"),
    )

    assert result.last_balance_label_counts == {"inactive": 2, "active": 2}


def test_distillation_dataset_keeps_unknown_roles_without_intent_guess() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    dataset = build_distillation_dataset(
        torch.randn(2, 5),
        torch.empty(2, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(2, 3),
        role_labels=("height_low", "height_high"),
    )

    assert dataset.command_intents is None
    assert "command_intent_inference_source" not in dataset.metadata


def test_distillation_dataset_roundtrip_preserves_command_intent_contract(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    student_obs = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    teacher_obs = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    commands = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.0, 0.0, 0.20],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    command_intents = ("inactive", "active", "active", "inactive")
    dataset = build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        metadata={"source": "command-intent-fixture"},
        commands=commands,
        command_intents=command_intents,
        role_labels=("stand", "walk_flat", "walk_flat", "stand"),
    )

    batch = dataset.as_batch(start=1, batch_size=2)
    assert batch.commands is not None
    assert torch.equal(batch.commands, commands[1:3])
    assert batch.command_intents == ("active", "active")
    assert batch.role_labels == ("walk_flat", "walk_flat")

    checkpoint_path = tmp_path / "command_intent_distill_dataset.pt"
    save_distillation_dataset(checkpoint_path, dataset)
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
    )

    assert restored.commands is not None
    assert torch.equal(restored.commands, commands)
    assert restored.command_intents == command_intents
    assert restored.metadata["command_intents"] == list(command_intents)
    assert restored.metadata["command_intent_counts"] == {"active": 2, "inactive": 2}


def test_distillation_dataset_roundtrip_preserves_cached_teacher_actions(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    student_obs = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    teacher_obs = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    teacher_actions = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    dataset = build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        expected_teacher_action_dim=3,
        metadata={"source": "cached-action-fixture"},
        role_labels=("stand", "walk", "height_low", "height_high"),
        teacher_actions=teacher_actions,
    )

    batch = dataset.as_batch(start=1, batch_size=2)
    assert batch.teacher_actions is not None
    assert batch.teacher_actions.shape == (2, 3)
    assert torch.equal(batch.teacher_actions[0], teacher_actions[1])
    assert batch.role_labels == ("walk", "height_low")

    checkpoint_path = tmp_path / "cached_action_distill_dataset.pt"
    save_distillation_dataset(checkpoint_path, dataset)
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        expected_teacher_action_dim=3,
    )

    assert restored.teacher_action_dim == 3
    assert restored.teacher_actions is not None
    assert torch.equal(restored.teacher_actions, teacher_actions)


def test_distillation_dataset_rejects_bad_obs_contract() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    with pytest.raises(ValueError, match="batch size"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(3, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
        )

    with pytest.raises(ValueError, match="student_obs dim"):
        build_distillation_dataset(
            torch.zeros(4, 6),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
        )

    teacher_obs = torch.zeros(4, 7)
    teacher_obs[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            teacher_obs,
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
        )


def test_distillation_dataset_rejects_bad_role_labels_contract() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    with pytest.raises(ValueError, match="role_labels length"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            role_labels=("stand",),
        )

    with pytest.raises(ValueError, match="empty labels"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            role_labels=("stand", "walk", "", "height"),
        )


def test_distillation_dataset_rejects_bad_command_intent_contract() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    with pytest.raises(ValueError, match="commands.*shape"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            commands=torch.zeros(4, 2),
        )

    with pytest.raises(ValueError, match="commands batch size"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            commands=torch.zeros(3, 3),
        )

    commands = torch.zeros(4, 3)
    commands[0, 0] = float("nan")
    with pytest.raises(ValueError, match="commands.*finite"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            commands=commands,
        )

    with pytest.raises(ValueError, match="command_intents length"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            command_intents=("active",),
        )

    with pytest.raises(ValueError, match="command_intents.*active/inactive"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            command_intents=("active", "inactive", "walk", "stand"),
        )


def test_distillation_dataset_rejects_bad_cached_teacher_actions_contract() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    with pytest.raises(ValueError, match="teacher_actions dim"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(4, 4),
        )

    with pytest.raises(ValueError, match="teacher action dataset batch size"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(3, 3),
        )

    teacher_actions = torch.zeros(4, 3)
    teacher_actions[0, 0] = float("nan")
    with pytest.raises(ValueError, match="teacher_actions.*finite"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            expected_teacher_action_dim=3,
            teacher_actions=teacher_actions,
        )


def test_multitask_distillation_dataset_adapter_merges_roles_and_cached_targets(
    tmp_path,
) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    stand_path = tmp_path / "stand.pt"
    walk_path = tmp_path / "walk.pt"
    stand_dataset = build_distillation_dataset(
        torch.full((2, 5), 1.0),
        torch.full((2, 5), 2.0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
        teacher_actions=torch.full((2, 3), 0.25),
        commands=torch.zeros(2, 3),
        command_intents=("inactive", "inactive"),
        metadata={"task_name": "G1StandStill"},
    )
    walk_dataset = build_distillation_dataset(
        torch.full((3, 5), 3.0),
        torch.full((3, 5), 4.0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
        teacher_actions=torch.full((3, 3), -0.5),
        commands=torch.tensor(
            [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]],
            dtype=torch.float32,
        ),
        command_intents=("active", "active", "active"),
        metadata={"task_name": "G1WalkHeight"},
    )
    save_distillation_dataset(stand_path, stand_dataset)
    save_distillation_dataset(walk_path, walk_dataset)

    merged = build_multitask_distillation_dataset(
        [
            {"path": stand_path, "role": "stand"},
            {"path": walk_path, "role": "walk_height"},
        ],
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
    )

    assert merged.num_samples == 5
    assert merged.role_labels == (
        "stand",
        "stand",
        "walk_height",
        "walk_height",
        "walk_height",
    )
    assert merged.teacher_action_dim == 3
    assert merged.teacher_actions is not None
    assert torch.allclose(merged.teacher_actions[:2], torch.full((2, 3), 0.25))
    assert torch.allclose(merged.teacher_actions[2:], torch.full((3, 3), -0.5))
    assert merged.metadata["source"] == "multitask_adapter"
    assert merged.metadata["source_count"] == 2
    assert merged.metadata["source_roles"] == ["stand", "walk_height"]
    assert merged.metadata["source_sample_counts"] == [2, 3]
    assert merged.commands is not None
    assert merged.command_intents == (
        "inactive",
        "inactive",
        "active",
        "active",
        "active",
    )
    assert merged.metadata["command_intent_counts"] == {"active": 3, "inactive": 2}
    assert merged.as_batch(start=1, batch_size=3).role_labels == (
        "stand",
        "walk_height",
        "walk_height",
    )

    roundtrip_path = tmp_path / "merged.pt"
    save_distillation_dataset(roundtrip_path, merged)
    reloaded = load_distillation_dataset(
        roundtrip_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
    )
    assert reloaded.role_labels == merged.role_labels
    assert reloaded.teacher_actions is not None
    assert torch.allclose(reloaded.teacher_actions, merged.teacher_actions)
    assert reloaded.commands is not None
    assert torch.equal(reloaded.commands, merged.commands)
    assert reloaded.command_intents == merged.command_intents


def test_multitask_distillation_dataset_adapter_fails_closed(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        save_distillation_dataset,
    )

    no_action_path = tmp_path / "no_action.pt"
    bad_dim_path = tmp_path / "bad_dim.pt"
    save_distillation_dataset(
        no_action_path,
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 5),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
        ),
    )
    save_distillation_dataset(
        bad_dim_path,
        build_distillation_dataset(
            torch.zeros(2, 6),
            torch.zeros(2, 5),
            expected_student_obs_dim=6,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(2, 3),
        ),
    )
    matching_path = tmp_path / "matching.pt"
    save_distillation_dataset(
        matching_path,
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 5),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(2, 3),
        ),
    )
    command_schema_path = tmp_path / "command_schema.pt"
    save_distillation_dataset(
        command_schema_path,
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 5),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(2, 3),
            commands=torch.zeros(2, 3),
            command_intents=("inactive", "inactive"),
        ),
    )

    with pytest.raises(ValueError, match="at least one source"):
        build_multitask_distillation_dataset([])
    with pytest.raises(ValueError, match="role"):
        build_multitask_distillation_dataset([{"path": no_action_path}])
    with pytest.raises(ValueError, match="cached teacher_actions"):
        build_multitask_distillation_dataset(
            [{"path": no_action_path, "role": "stand"}],
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
        )
    with pytest.raises(ValueError, match="student_obs dim mismatch"):
        build_multitask_distillation_dataset(
            [{"path": bad_dim_path, "role": "walk_height"}],
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
        )
    with pytest.raises(ValueError, match="multitask source .* student_obs dim mismatch"):
        build_multitask_distillation_dataset(
            [
                {"path": matching_path, "role": "stand"},
                {"path": bad_dim_path, "role": "walk_height"},
            ],
        )
    with pytest.raises(ValueError, match="all include commands or none"):
        build_multitask_distillation_dataset(
            [
                {"path": matching_path, "role": "stand"},
                {"path": command_schema_path, "role": "walk_height"},
            ],
        )


def test_command_active_mask_marks_any_velocity_command_active() -> None:
    from unilab.algos.torch.distill import command_active_mask

    commands = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.0, -0.06, 0.0],
            [0.03, 0.04, 0.0],
            [0.04, 0.04, 0.0],
            [0.0, 0.0, 0.06],
            [0.0, 0.0, -0.06],
        ],
        dtype=np.float32,
    )

    mask = command_active_mask(commands, xy_threshold=0.05, yaw_threshold=0.05)

    np.testing.assert_array_equal(
        mask,
        np.asarray([False, True, True, False, True, True, True], dtype=np.bool_),
    )


@pytest.mark.parametrize(
    "commands",
    [
        np.zeros((3,), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
        np.asarray([[0.0, np.nan, 0.0]], dtype=np.float32),
    ],
)
def test_command_active_mask_fails_closed_for_bad_commands(commands: np.ndarray) -> None:
    from unilab.algos.torch.distill import command_active_mask

    with pytest.raises(ValueError, match="commands"):
        command_active_mask(commands, xy_threshold=0.05, yaw_threshold=0.05)


@pytest.mark.parametrize(
    ("xy_threshold", "yaw_threshold"),
    [
        (-0.01, 0.05),
        (0.05, -0.01),
        (np.inf, 0.05),
        (0.05, np.nan),
    ],
)
def test_command_active_mask_fails_closed_for_bad_thresholds(
    xy_threshold: float,
    yaw_threshold: float,
) -> None:
    from unilab.algos.torch.distill import command_active_mask

    with pytest.raises(ValueError, match="threshold"):
        command_active_mask(
            np.zeros((1, 3), dtype=np.float32),
            xy_threshold=xy_threshold,
            yaw_threshold=yaw_threshold,
        )


class _FakeDistillEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.action_space = type("ActionSpace", (), {"shape": (3,)})()
        self.reset_calls = 0
        self.step_calls = 0
        self.state = None
        self.last_actions = None

    def init_state(self) -> None:
        self.state = object()

    def reset(self, env_indices):
        self.reset_calls += 1
        return self._obs(0), {"reset_indices": np.asarray(env_indices)}

    def step(self, actions):
        self.step_calls += 1
        assert actions.shape == (2, 3)
        self.last_actions = np.asarray(actions, dtype=np.float32)
        return type("State", (), {"obs": self._obs(self.step_calls), "info": {}})()

    def _obs(self, offset: int) -> dict[str, np.ndarray]:
        base = np.arange(16, dtype=np.float32).reshape(2, 8) + float(offset)
        return {"obs": base, "critic": base + 100.0}


class _CommandInfoDistillEnv(_FakeDistillEnv):
    def __init__(self, command_batches: list[np.ndarray]) -> None:
        super().__init__()
        self.command_batches = command_batches

    def reset(self, env_indices):
        obs, _info = super().reset(env_indices)
        return obs, {"commands": self.command_batches[0]}

    def step(self, actions):
        state = super().step(actions)
        batch_index = min(self.step_calls, len(self.command_batches) - 1)
        return type(
            "State",
            (),
            {"obs": state.obs, "info": {"commands": self.command_batches[batch_index]}},
        )()


def test_collect_distillation_dataset_from_env_projects_student_obs() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    dataset = collect_distillation_dataset_from_env(
        _FakeDistillEnv(),
        num_samples=3,
        expected_student_obs_dim=7,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="drop_index",
        student_drop_index=3,
        action_mode="zero",
    )

    assert dataset.num_samples == 3
    assert dataset.student_obs_dim == 7
    assert dataset.teacher_obs_dim == 8
    assert dataset.metadata["source"] == "live_env_rollout"
    assert dataset.metadata["student_projection"] == "drop_index"
    assert dataset.metadata["teacher_projection"] == "identity"
    assert dataset.metadata["student_drop_index"] == 3
    assert dataset.metadata["teacher_obs_key"] == "obs"
    assert dataset.metadata["action_mode"] == "zero"
    assert dataset.metadata["synthetic_teacher_tail"] is False
    assert "command_sample_filter" not in dataset.metadata
    assert torch.equal(dataset.teacher_obs[0], torch.arange(8, dtype=torch.float32))
    assert torch.equal(
        dataset.student_obs[0],
        torch.tensor([0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0]),
    )


def test_collect_distillation_dataset_from_env_pads_teacher_obs_tail() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    dataset = collect_distillation_dataset_from_env(
        _FakeDistillEnv(),
        num_samples=1,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=10,
        teacher_obs_key="obs",
        teacher_projection="pad_zeros",
        student_projection="identity",
        action_mode="zero",
    )

    assert dataset.student_obs.shape == (1, 8)
    assert dataset.teacher_obs.shape == (1, 10)
    assert torch.equal(dataset.teacher_obs[0, :8], torch.arange(8, dtype=torch.float32))
    assert torch.equal(dataset.teacher_obs[0, 8:], torch.zeros(2))
    assert dataset.metadata["teacher_projection"] == "pad_zeros"
    assert dataset.metadata["synthetic_teacher_tail"] is True


def test_collect_distillation_dataset_from_env_random_action_mode_is_nonzero() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    env = _FakeDistillEnv()
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=3,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="identity",
        action_mode="random",
        action_seed=7,
    )

    assert dataset.metadata["action_mode"] == "random"
    assert dataset.metadata["action_seed"] == 7
    assert dataset.metadata["action_abs_max"] > 0.0
    assert env.last_actions is not None
    assert np.isfinite(env.last_actions).all()
    assert np.max(np.abs(env.last_actions)) > 0.0
    assert dataset.teacher_actions is None


def test_collect_distillation_dataset_from_env_teacher_policy_action_mode() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    class FakeTeacherPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            assert obs.shape == (2, 8)
            return torch.tanh(obs[:, :3] * 0.01 + 0.1)

    env = _FakeDistillEnv()
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=3,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="identity",
        action_mode="teacher_policy",
        teacher_policy=FakeTeacherPolicy(),
    )

    assert dataset.metadata["action_mode"] == "teacher_policy"
    assert dataset.metadata["action_seed"] is None
    assert dataset.metadata["action_abs_max"] > 0.0
    assert env.last_actions is not None
    assert np.isfinite(env.last_actions).all()
    assert np.max(np.abs(env.last_actions)) > 0.0
    assert dataset.teacher_actions is not None
    assert dataset.teacher_actions.shape == (3, 3)
    assert torch.allclose(dataset.teacher_actions[:2], torch.as_tensor(env.last_actions))
    assert torch.isfinite(dataset.teacher_actions).all()


def test_collect_distillation_dataset_from_env_student_policy_rollout_mode() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    class FakeTeacherPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            assert obs.shape == (2, 8)
            return torch.full((obs.shape[0], 3), 0.25, dtype=obs.dtype, device=obs.device)

    class FakeRolloutPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            assert obs.shape == (2, 8)
            return torch.full((obs.shape[0], 3), -0.5, dtype=obs.dtype, device=obs.device)

    env = _FakeDistillEnv()
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=3,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="identity",
        action_mode="student_policy",
        teacher_policy=FakeTeacherPolicy(),
        rollout_policy=FakeRolloutPolicy(),
    )

    assert dataset.metadata["action_mode"] == "student_policy"
    assert dataset.metadata["action_seed"] is None
    assert dataset.metadata["rollout_policy"] == "distillation_student"
    assert dataset.metadata["action_abs_max"] == pytest.approx(0.5)
    assert env.last_actions is not None
    assert np.allclose(env.last_actions, -0.5)
    assert dataset.teacher_actions is not None
    assert dataset.teacher_actions.shape == (3, 3)
    assert torch.allclose(dataset.teacher_actions, torch.full((3, 3), 0.25))
    assert not torch.allclose(dataset.teacher_actions[:2], torch.as_tensor(env.last_actions))


def test_collect_distillation_dataset_from_env_student_policy_resets_done_rows() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    class FakeTeacherPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.full((obs.shape[0], 3), 0.25, dtype=obs.dtype, device=obs.device)

    class FakeRolloutPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.full((obs.shape[0], 3), -0.5, dtype=obs.dtype, device=obs.device)

    class DoneAfterStepEnv(_FakeDistillEnv):
        def reset(self, env_indices):
            self.reset_calls += 1
            env_indices = np.asarray(env_indices, dtype=np.int32)
            base = np.arange(16, dtype=np.float32).reshape(2, 8)
            if env_indices.shape[0] == self.num_envs:
                rows = base
            else:
                rows = base[env_indices] + 100.0
            return {"obs": rows, "critic": rows + 100.0}, {
                "reset_indices": env_indices,
            }

        def step(self, actions):
            self.step_calls += 1
            self.last_actions = np.asarray(actions, dtype=np.float32)
            return type(
                "State",
                (),
                {
                    "obs": self._obs(self.step_calls),
                    "info": {},
                    "terminated": np.asarray([True, False], dtype=np.bool_),
                    "truncated": np.asarray([False, False], dtype=np.bool_),
                },
            )()

    env = DoneAfterStepEnv()
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=4,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="identity",
        action_mode="student_policy",
        teacher_policy=FakeTeacherPolicy(),
        rollout_policy=FakeRolloutPolicy(),
    )

    assert env.step_calls == 1
    assert env.reset_calls == 2
    assert dataset.metadata["action_mode"] == "student_policy"
    assert dataset.metadata["done_seen_samples"] == 1
    assert dataset.metadata["autoreset_done_count"] == 0
    assert dataset.metadata["manual_done_reset_count"] == 1
    assert torch.equal(dataset.student_obs[0], torch.arange(8, dtype=torch.float32))
    assert torch.equal(dataset.student_obs[1], torch.arange(8, 16, dtype=torch.float32))
    assert torch.equal(dataset.student_obs[2], torch.arange(8, dtype=torch.float32) + 100.0)
    assert torch.equal(dataset.student_obs[3], torch.arange(8, 16, dtype=torch.float32) + 1.0)


def test_collect_distillation_dataset_from_env_filters_active_command_samples() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    env = _CommandInfoDistillEnv(
        [
            np.asarray([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.10], [0.0, 0.0, 0.0]], dtype=np.float32),
        ]
    )
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=2,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        command_sample_filter="active",
        command_xy_threshold=0.05,
        command_yaw_threshold=0.05,
        max_env_steps=1,
    )

    assert dataset.num_samples == 2
    assert dataset.metadata["command_sample_filter"] == "active"
    assert dataset.metadata["command_seen_samples"] == 4
    assert dataset.metadata["command_selected_samples"] == 2
    assert dataset.metadata["env_steps"] == 1
    assert dataset.commands is not None
    assert torch.equal(
        dataset.commands,
        torch.tensor([[0.10, 0.0, 0.0], [0.0, 0.0, 0.10]], dtype=torch.float32),
    )
    assert dataset.command_intents == ("active", "active")
    assert dataset.metadata["command_intent_counts"] == {"active": 2}
    assert torch.equal(dataset.teacher_obs[0], torch.arange(8, 16, dtype=torch.float32))
    assert torch.equal(dataset.teacher_obs[1], torch.arange(8, dtype=torch.float32) + 1.0)


def test_collect_distillation_dataset_from_env_filters_inactive_command_samples() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    env = _CommandInfoDistillEnv(
        [
            np.asarray([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.10], [0.0, 0.0, 0.0]], dtype=np.float32),
        ]
    )
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=2,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        command_sample_filter="inactive",
        command_xy_threshold=0.05,
        command_yaw_threshold=0.05,
        max_env_steps=1,
    )

    assert dataset.num_samples == 2
    assert dataset.metadata["command_sample_filter"] == "inactive"
    assert dataset.metadata["command_seen_samples"] == 4
    assert dataset.metadata["command_selected_samples"] == 2
    assert dataset.metadata["env_steps"] == 1
    assert dataset.commands is not None
    assert torch.equal(
        dataset.commands,
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    assert dataset.command_intents == ("inactive", "inactive")
    assert dataset.metadata["command_intent_counts"] == {"inactive": 2}
    assert torch.equal(dataset.teacher_obs[0], torch.arange(8, dtype=torch.float32))
    assert torch.equal(dataset.teacher_obs[1], torch.arange(8, 16, dtype=torch.float32) + 1.0)


def test_collect_distillation_dataset_from_env_filter_requires_command_info() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    with pytest.raises(KeyError, match="commands"):
        collect_distillation_dataset_from_env(
            _FakeDistillEnv(),
            num_samples=1,
            expected_student_obs_dim=8,
            expected_teacher_obs_dim=8,
            command_sample_filter="active",
        )


def test_collect_distillation_dataset_from_env_filter_fails_when_budget_exhausts() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    env = _CommandInfoDistillEnv(
        [
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        ]
    )
    with pytest.raises(RuntimeError, match="command_sample_filter='active'"):
        collect_distillation_dataset_from_env(
            env,
            num_samples=1,
            expected_student_obs_dim=8,
            expected_teacher_obs_dim=8,
            command_sample_filter="active",
            max_env_steps=1,
        )


def test_collect_distillation_dataset_from_env_rejects_half_open_projection() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    with pytest.raises(ValueError, match="student_drop_index"):
        collect_distillation_dataset_from_env(
            _FakeDistillEnv(),
            num_samples=1,
            expected_student_obs_dim=7,
            expected_teacher_obs_dim=8,
            student_projection="drop_index",
            student_drop_index=None,
        )


def test_offline_distillation_run_updates_and_saves_checkpoint(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        load_distillation_checkpoint,
        run_offline_distillation_updates,
    )

    torch.manual_seed(23)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    teacher = torch.nn.Linear(7, 3)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        loss_type="mse",
    )
    dataset = build_distillation_dataset(
        torch.randn(4, 5),
        torch.randn(4, 7),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
    )
    checkpoint_path = tmp_path / "offline_student.pt"

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=2,
        max_updates=2,
        checkpoint_path=checkpoint_path,
        teacher_metadata={"algo": "linear-test"},
        distill_runtime_cfg={"loss_type": "mse"},
    )

    assert result.update_count == 2
    assert result.samples_seen == 4
    assert result.checkpoint_path == checkpoint_path
    assert result.last_loss >= 0.0
    assert result.last_behavior_loss == pytest.approx(result.last_loss)
    assert result.last_aux_loss == pytest.approx(0.0)
    assert result.last_expert_usage is None
    assert result.last_route_entropy is None
    assert result.last_teacher_action_source == "teacher"
    assert result.last_student_grad_norm > 0.0
    assert result.student_action_shape == (2, 3)
    assert result.teacher_action_shape == (2, 3)
    assert checkpoint_path.exists()

    restored = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    checkpoint = load_distillation_checkpoint(restored, checkpoint_path)
    assert checkpoint["agent_steps"] == 4
    assert checkpoint["teacher_metadata"] == {"algo": "linear-test"}
    assert checkpoint["distill_runtime_cfg"] == {"loss_type": "mse"}
    assert "optimizer_state_dict" in checkpoint
    for trained_param, restored_param in zip(student.parameters(), restored.parameters()):
        assert torch.allclose(trained_param, restored_param)


def test_offline_distillation_run_accepts_cached_teacher_actions(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    class RaisingTeacher(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            del obs
            raise AssertionError("offline cached teacher_action path must not call teacher")

    torch.manual_seed(29)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=RaisingTeacher(),
        optimizer=optimizer,
    )
    dataset = build_distillation_dataset(
        torch.randn(4, 5),
        torch.empty(4, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(4, 3),
        role_labels=("stand", "stand", "walk_height", "walk_height"),
    )

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=2,
        max_updates=2,
        checkpoint_path=tmp_path / "cached_model.pt",
    )

    assert result.update_count == 2
    assert result.samples_seen == 4
    assert result.teacher_action_requires_grad is False
    assert result.teacher_action_shape == (2, 3)
    assert result.last_teacher_action_source == "cached"
    assert result.last_student_grad_norm > 0.0


def test_offline_distillation_run_can_repeat_dataset_for_multiple_updates() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    torch.manual_seed(31)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(7, 3),
        optimizer=optimizer,
    )
    dataset = build_distillation_dataset(
        torch.randn(3, 5),
        torch.randn(3, 7),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        role_labels=("walk", "stand", "walk"),
    )

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=2,
        max_updates=4,
        repeat_dataset=True,
        shuffle=True,
        seed=5,
    )

    assert result.update_count == 4
    assert result.samples_seen == 6
    assert len(result.losses) == 4
    assert result.last_student_grad_norm > 0.0


def test_offline_distillation_run_balances_role_batches() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    class RaisingTeacher(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            del obs
            raise AssertionError("balanced cached-target path must not call teacher")

    torch.manual_seed(37)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=RaisingTeacher(),
        optimizer=optimizer,
    )
    dataset = build_distillation_dataset(
        torch.randn(6, 5),
        torch.empty(6, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(6, 3),
        role_labels=("stand", "walk", "walk", "walk", "walk", "walk"),
    )

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=4,
        max_updates=3,
        balance_key="role",
        balanced_labels=("stand", "walk"),
        seed=11,
    )

    assert result.update_count == 3
    assert result.samples_seen == 12
    assert result.batch_label_counts == (
        {"stand": 2, "walk": 2},
        {"stand": 2, "walk": 2},
        {"stand": 2, "walk": 2},
    )
    assert result.last_balance_label_counts == {"stand": 2, "walk": 2}
    assert result.last_teacher_action_source == "cached"
    assert result.last_student_grad_norm > 0.0


def test_offline_distillation_run_balances_command_intent_batches() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    torch.manual_seed(41)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(0, 3),
        optimizer=torch.optim.Adam(student.parameters(), lr=1e-2),
    )
    dataset = build_distillation_dataset(
        torch.randn(6, 5),
        torch.empty(6, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(6, 3),
        command_intents=("inactive", "active", "active", "active", "active", "active"),
    )

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=4,
        max_updates=2,
        balance_key="command_intent",
        balanced_labels=("inactive", "active"),
        seed=13,
    )

    assert result.batch_label_counts == (
        {"inactive": 2, "active": 2},
        {"inactive": 2, "active": 2},
    )
    assert result.last_balance_label_counts == {"inactive": 2, "active": 2}
    assert result.samples_seen == 8


def test_offline_distillation_run_balanced_sampler_fails_closed() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(0, 3),
        optimizer=torch.optim.Adam(student.parameters(), lr=1e-2),
    )
    dataset = build_distillation_dataset(
        torch.randn(2, 5),
        torch.empty(2, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(2, 3),
    )

    with pytest.raises(ValueError, match="role_labels"):
        run_offline_distillation_updates(
            trainer,
            dataset,
            batch_size=2,
            max_updates=1,
            balance_key="role",
        )


def test_distillation_student_checkpoint_loads_for_student_only_playback(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        load_distillation_student_policy,
        save_distillation_checkpoint,
    )

    torch.manual_seed(29)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    checkpoint_path = tmp_path / "student_play.pt"
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=4,
        teacher_metadata={"task_name": "G1WalkHeight"},
        distill_runtime_cfg={
            "student_obs_dim": 5,
            "student_action_dim": 3,
            "student_hidden_dims": [8],
            "student_activation": "elu",
            "student_squash_action": True,
        },
    )

    loaded = load_distillation_student_policy(checkpoint_path, device="cpu")
    obs = torch.randn(2, 5)
    action = loaded.policy(obs)

    assert loaded.obs_dim == 5
    assert loaded.action_dim == 3
    assert loaded.agent_steps == 4
    assert loaded.teacher_metadata == {"task_name": "G1WalkHeight"}
    assert action.shape == (2, 3)
    assert action.requires_grad is False
    assert torch.isfinite(action).all()

    with pytest.raises(ValueError, match="Student obs dim mismatch"):
        loaded.policy(torch.randn(2, 6))


def test_distillation_moe_student_checkpoint_loads_for_student_only_playback(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        MoEStudentPolicy,
        load_distillation_student_policy,
        save_distillation_checkpoint,
    )

    torch.manual_seed(31)
    student = MoEStudentPolicy(
        obs_dim=5,
        action_dim=3,
        num_experts=3,
        expert_hidden_dims=(8,),
        router_hidden_dims=(4,),
        routing_mode="soft",
        router_temperature=0.75,
        squash_action=False,
    )
    checkpoint_path = tmp_path / "moe_student_play.pt"
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=7,
        teacher_metadata={"task_name": "G1WalkHeight", "student": "moe"},
        distill_runtime_cfg={
            "student_model_type": "moe",
            "student_obs_dim": 5,
            "student_action_dim": 3,
            "student_num_experts": 3,
            "student_expert_hidden_dims": [8],
            "student_router_hidden_dims": [4],
            "student_routing_mode": "soft",
            "student_router_temperature": 0.75,
            "student_activation": "elu",
            "student_squash_action": False,
        },
    )

    loaded = load_distillation_student_policy(checkpoint_path, device="cpu")
    obs = torch.randn(2, 5)
    action = loaded.policy(obs)

    assert isinstance(loaded.policy, MoEStudentPolicy)
    assert loaded.obs_dim == 5
    assert loaded.action_dim == 3
    assert loaded.agent_steps == 7
    assert loaded.teacher_metadata == {"task_name": "G1WalkHeight", "student": "moe"}
    assert loaded.distill_runtime_cfg["student_model_type"] == "moe"
    assert loaded.policy.num_experts == 3
    assert loaded.policy.router_temperature == pytest.approx(0.75)
    assert action.shape == (2, 3)
    assert action.requires_grad is False
    assert torch.isfinite(action).all()
    assert all(param.requires_grad is False for param in loaded.policy.parameters())

    with pytest.raises(ValueError, match="Student obs dim mismatch"):
        loaded.policy(torch.randn(2, 6))


def test_distillation_student_playback_rejects_unknown_model_type(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        load_distillation_student_policy,
        save_distillation_checkpoint,
    )

    checkpoint_path = tmp_path / "student_unknown_model.pt"
    save_distillation_checkpoint(
        checkpoint_path,
        student=MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,)),
        agent_steps=4,
        distill_runtime_cfg={
            "student_model_type": "unknown",
            "student_obs_dim": 5,
            "student_action_dim": 3,
        },
    )

    with pytest.raises(ValueError, match="student_model_type"):
        load_distillation_student_policy(checkpoint_path, device="cpu")


def test_distillation_student_playback_rejects_missing_runtime_dims(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        load_distillation_student_policy,
        save_distillation_checkpoint,
    )

    checkpoint_path = tmp_path / "student_missing_dims.pt"
    save_distillation_checkpoint(
        checkpoint_path,
        student=MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,)),
        agent_steps=4,
        distill_runtime_cfg={"loss_type": "mse"},
    )

    with pytest.raises(ValueError, match="student_obs_dim"):
        load_distillation_student_policy(checkpoint_path, device="cpu")


def test_sac_teacher_checkpoint_loads_with_dim_guard(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        DistillationTeacherSpec,
        load_sac_teacher_policy,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    torch.manual_seed(17)
    actor = SACActor(
        obs_dim=5,
        action_dim=3,
        hidden_dim=8,
        use_layer_norm=False,
        device="cpu",
    )
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"actor": actor.state_dict(), "update_count": 3}, checkpoint_path)

    teacher = load_sac_teacher_policy(
        checkpoint_path,
        DistillationTeacherSpec(
            algo_type="sac",
            obs_dim=5,
            action_dim=3,
            actor_hidden_dim=8,
            use_layer_norm=False,
        ),
    )
    action = teacher(torch.randn(4, 5))

    assert action.shape == (4, 3)
    assert action.requires_grad is False
    assert torch.isfinite(action).all()
    assert all(param.requires_grad is False for param in teacher.parameters())


def test_sac_teacher_checkpoint_inspector_reports_actor_input_dim(tmp_path) -> None:
    from unilab.algos.torch.distill import inspect_sac_teacher_checkpoint
    from unilab.algos.torch.fast_sac.learner import SACActor

    actor = SACActor(
        obs_dim=5,
        action_dim=3,
        hidden_dim=8,
        use_layer_norm=False,
        device="cpu",
    )
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"actor": actor.state_dict()}, checkpoint_path)

    info = inspect_sac_teacher_checkpoint(checkpoint_path)

    assert info.checkpoint_path == str(checkpoint_path)
    assert info.actor_input_dim == 5
    assert info.first_weight_key == "net.0.weight"


def test_sac_teacher_checkpoint_rejects_dim_mismatch(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        DistillationTeacherSpec,
        load_sac_teacher_policy,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    actor = SACActor(
        obs_dim=5,
        action_dim=3,
        hidden_dim=8,
        use_layer_norm=False,
        device="cpu",
    )
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"actor": actor.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="checkpoint actor input dim=5"):
        load_sac_teacher_policy(
            checkpoint_path,
            DistillationTeacherSpec(
                algo_type="sac",
                obs_dim=6,
                action_dim=3,
                actor_hidden_dim=8,
                use_layer_norm=False,
            ),
        )
