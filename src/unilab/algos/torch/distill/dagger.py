from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .checkpoint import save_distillation_checkpoint
from .collector import collect_distillation_dataset_from_env
from .data import DistillationTensorDataset, build_distillation_dataset
from .moe_student import MoEStudentPolicy
from .offline import OfflineDistillationRunResult, run_offline_distillation_updates
from .trainer import BehaviorDistillationTrainer


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@dataclass(frozen=True)
class IterativeDaggerRunResult:
    """Summary of a student-rollout, teacher-label, immediate-update DAgger run."""

    iteration_count: int
    update_count: int
    samples_collected: int
    samples_seen: int
    checkpoint_path: Path | None
    iteration_results: tuple[OfflineDistillationRunResult, ...]
    collection_metadata: tuple[dict[str, Any], ...]


class _FixedExpertRolloutPolicy(torch.nn.Module):
    def __init__(self, student: MoEStudentPolicy, expert_index: int) -> None:
        super().__init__()
        if not 0 <= int(expert_index) < student.num_experts:
            raise ValueError(
                f"DAgger rollout expert index out of range: "
                f"index={int(expert_index)} num_experts={student.num_experts}"
            )
        self.student = student
        self.expert_index = int(expert_index)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.student.experts[self.expert_index](obs)


def resolve_command_intent_rollout_policies(
    student: MoEStudentPolicy,
    runtime_cfg: Mapping[str, Any],
) -> tuple[dict[str, torch.nn.Module], dict[str, int]]:
    """Resolve deployment-aligned active/inactive expert modules for rollout."""

    raw_targets = runtime_cfg.get("command_intent_expert_targets")
    if not isinstance(raw_targets, Mapping):
        raise ValueError(
            "command-intent rollout requires command_intent_expert_targets"
        )
    missing_intents = {"active", "inactive"} - set(raw_targets)
    if missing_intents:
        raise ValueError(
            "command-intent rollout targets are missing intents: "
            f"{sorted(missing_intents)}"
        )
    expert_targets = {
        intent: int(raw_targets[intent]) for intent in ("active", "inactive")
    }
    if any(
        target < 0 or target >= int(student.num_experts)
        for target in expert_targets.values()
    ):
        raise ValueError(
            "command-intent rollout expert target out of range: "
            f"targets={expert_targets} num_experts={int(student.num_experts)}"
        )
    return (
        {
            intent: student.experts[target]
            for intent, target in expert_targets.items()
        },
        expert_targets,
    )


def _resolve_dagger_rollout_policy(
    trainer: BehaviorDistillationTrainer,
    *,
    command_sample_filter: str,
    role_label: str | None,
) -> tuple[torch.nn.Module, int | None, str]:
    candidates: list[tuple[int, str]] = []
    if command_sample_filter in {"active", "inactive"}:
        target = trainer.command_intent_expert_targets.get(command_sample_filter)
        if target is not None:
            candidates.append((int(target), "command_intent"))
    if role_label is not None:
        target = trainer.role_expert_targets.get(str(role_label))
        if target is not None:
            candidates.append((int(target), "role"))
    if not candidates:
        return trainer.student, None, "student"

    expert_indices = {index for index, _source in candidates}
    if len(expert_indices) != 1:
        raise ValueError(f"DAgger rollout expert targets conflict: {candidates}")
    if not isinstance(trainer.student, MoEStudentPolicy):
        raise TypeError("expert-conditioned DAgger rollout requires MoEStudentPolicy")
    expert_index = candidates[0][0]
    sources = {source for _index, source in candidates}
    source = "command_intent+role" if len(sources) > 1 else candidates[0][1]
    return _FixedExpertRolloutPolicy(trainer.student, expert_index), expert_index, source


def _attach_role_label(
    dataset: DistillationTensorDataset,
    role_label: str | None,
) -> DistillationTensorDataset:
    if role_label is None:
        return dataset
    label = str(role_label)
    if not label:
        raise ValueError("role_label must not be empty")
    metadata = dict(dataset.metadata)
    metadata["role_label"] = label
    return build_distillation_dataset(
        dataset.student_obs,
        dataset.teacher_obs,
        expected_student_obs_dim=dataset.student_obs_dim,
        expected_teacher_obs_dim=dataset.teacher_obs_dim,
        expected_teacher_action_dim=dataset.teacher_action_dim,
        metadata=metadata,
        role_labels=(label,) * dataset.num_samples,
        teacher_actions=dataset.teacher_actions,
        commands=dataset.commands,
        command_intents=dataset.command_intents,
    )


def _aggregate_dagger_datasets(
    datasets: Sequence[DistillationTensorDataset],
) -> DistillationTensorDataset:
    if not datasets:
        raise ValueError("DAgger aggregation requires at least one dataset")

    def cat_tensor(name: str) -> torch.Tensor | None:
        values = [getattr(dataset, name) for dataset in datasets]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"DAgger datasets disagree on optional tensor {name!r}")
        return torch.cat(values, dim=0)

    def cat_labels(name: str) -> tuple[str, ...] | None:
        values = [getattr(dataset, name) for dataset in datasets]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"DAgger datasets disagree on optional labels {name!r}")
        return tuple(label for labels in values for label in labels)

    first = datasets[0]
    metadata = dict(datasets[-1].metadata)
    metadata.update(
        {
            "source": "iterative_dagger_aggregate",
            "dagger_aggregate_iterations": len(datasets),
            "dagger_aggregate_num_samples": sum(dataset.num_samples for dataset in datasets),
        }
    )
    if len(datasets) == 1:
        return replace(first, metadata=metadata)
    return build_distillation_dataset(
        torch.cat([dataset.student_obs for dataset in datasets], dim=0),
        torch.cat([dataset.teacher_obs for dataset in datasets], dim=0),
        expected_student_obs_dim=first.student_obs_dim,
        expected_teacher_obs_dim=first.teacher_obs_dim,
        expected_teacher_action_dim=first.teacher_action_dim,
        metadata=metadata,
        role_labels=cat_labels("role_labels"),
        teacher_actions=cat_tensor("teacher_actions"),
        commands=cat_tensor("commands"),
        command_intents=cat_labels("command_intents"),
    )


def run_iterative_dagger_updates(
    env: Any,
    *,
    trainer: BehaviorDistillationTrainer,
    num_iterations: int,
    samples_per_iteration: int,
    batch_size: int,
    updates_per_iteration: int,
    expected_student_obs_dim: int,
    expected_teacher_obs_dim: int,
    teacher_obs_key: str = "obs",
    teacher_projection: str = "identity",
    student_projection: str = "identity",
    student_drop_index: int | None = None,
    command_sample_filter: str = "none",
    command_info_key: str = "commands",
    command_xy_threshold: float = 0.05,
    command_yaw_threshold: float = 0.05,
    max_env_steps: int | None = None,
    role_label: str | None = None,
    shuffle: bool = True,
    seed: int = 1,
    balance_key: str = "none",
    balanced_labels: Sequence[str] | None = None,
    checkpoint_path: str | Path | None = None,
    teacher_metadata: Mapping[str, Any] | None = None,
    distill_runtime_cfg: Mapping[str, Any] | None = None,
) -> IterativeDaggerRunResult:
    """Run DAgger by recollecting student-state labels after every update phase."""

    for name, value in (
        ("num_iterations", num_iterations),
        ("samples_per_iteration", samples_per_iteration),
        ("batch_size", batch_size),
        ("updates_per_iteration", updates_per_iteration),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    iteration_results: list[OfflineDistillationRunResult] = []
    collection_metadata: list[dict[str, Any]] = []
    collected_datasets: list[DistillationTensorDataset] = []
    samples_collected = 0
    samples_seen = 0
    rollout_policy, rollout_expert_index, rollout_policy_source = (
        _resolve_dagger_rollout_policy(
            trainer,
            command_sample_filter=str(command_sample_filter),
            role_label=role_label,
        )
    )
    for iteration in range(int(num_iterations)):
        dataset = collect_distillation_dataset_from_env(
            env,
            num_samples=int(samples_per_iteration),
            expected_student_obs_dim=int(expected_student_obs_dim),
            expected_teacher_obs_dim=int(expected_teacher_obs_dim),
            teacher_obs_key=str(teacher_obs_key),
            teacher_projection=str(teacher_projection),
            student_projection=str(student_projection),
            student_drop_index=student_drop_index,
            action_mode="student_policy",
            teacher_policy=trainer.teacher,
            rollout_policy=rollout_policy,
            command_sample_filter=str(command_sample_filter),
            command_info_key=str(command_info_key),
            command_xy_threshold=float(command_xy_threshold),
            command_yaw_threshold=float(command_yaw_threshold),
            max_env_steps=max_env_steps,
            metadata={
                "dagger_iteration": iteration + 1,
                "dagger_rollout_policy_source": rollout_policy_source,
                "dagger_rollout_expert_index": rollout_expert_index,
            },
        )
        dataset = _attach_role_label(dataset, role_label).to(_module_device(trainer.student))
        collected_datasets.append(dataset)
        training_dataset = _aggregate_dagger_datasets(collected_datasets)
        collection_metadata.append(dict(training_dataset.metadata))
        samples_collected += dataset.num_samples

        result = run_offline_distillation_updates(
            trainer,
            training_dataset,
            batch_size=int(batch_size),
            max_updates=int(updates_per_iteration),
            repeat_dataset=True,
            shuffle=bool(shuffle),
            seed=int(seed) + iteration,
            balance_key=str(balance_key),
            balanced_labels=balanced_labels,
        )
        iteration_results.append(result)
        samples_seen += result.samples_seen

    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if resolved_checkpoint is not None:
        runtime_cfg = dict(distill_runtime_cfg or {})
        runtime_cfg.update(
            {
                "distill_source": "iterative_dagger",
                "dagger_iterations": int(num_iterations),
                "dagger_samples_per_iteration": int(samples_per_iteration),
                "dagger_updates_per_iteration": int(updates_per_iteration),
                "dagger_role_label": role_label,
                "dagger_rollout_policy_source": rollout_policy_source,
                "dagger_rollout_expert_index": rollout_expert_index,
            }
        )
        initial_steps = int(
            trainer.student_init_metadata.get(
                "agent_steps",
                trainer.student_init_metadata.get("student_init_agent_steps", 0),
            )
        )
        save_distillation_checkpoint(
            resolved_checkpoint,
            student=trainer.student,
            optimizer=trainer.optimizer,
            agent_steps=initial_steps + samples_seen,
            teacher_metadata=teacher_metadata,
            distill_runtime_cfg=runtime_cfg,
        )

    return IterativeDaggerRunResult(
        iteration_count=int(num_iterations),
        update_count=trainer.update_count,
        samples_collected=samples_collected,
        samples_seen=samples_seen,
        checkpoint_path=resolved_checkpoint,
        iteration_results=tuple(iteration_results),
        collection_metadata=tuple(collection_metadata),
    )
