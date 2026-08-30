from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from unilab.algos.torch.distill.contracts.dataset import (
    _command_intents_from_commands,
    _command_intents_from_role_labels,
    _validate_action_tensor,
    _validate_command_intents,
    _validate_commands,
    _validate_obs_tensor,
    _validate_role_labels,
    _validate_scenario_labels,
    _validate_target_height,
    _validate_transition_fields,
)
from unilab.algos.torch.distill.datasets.diagnostics import _TRANSITION_SCENARIOS, _label_counts
from unilab.algos.torch.distill.learning.trainer import DistillationBatch


@dataclass(frozen=True)
class DistillationTensorDataset:
    """In-memory offline distillation observations with explicit shape contracts."""

    student_obs: torch.Tensor
    teacher_obs: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    role_labels: tuple[str, ...] | None = None
    teacher_actions: torch.Tensor | None = None
    commands: torch.Tensor | None = None
    target_height: torch.Tensor | None = None
    command_intents: tuple[str, ...] | None = None
    scenario_labels: tuple[str, ...] | None = None
    transition_ages: torch.Tensor | None = None
    command_before: torch.Tensor | None = None
    command_after: torch.Tensor | None = None

    @property
    def num_samples(self) -> int:
        return int(self.student_obs.shape[0])

    @property
    def student_obs_dim(self) -> int:
        return int(self.student_obs.shape[-1])

    @property
    def teacher_obs_dim(self) -> int:
        return int(self.teacher_obs.shape[-1])

    @property
    def teacher_action_dim(self) -> int | None:
        if self.teacher_actions is None:
            return None
        return int(self.teacher_actions.shape[-1])

    def to(self, device: str | torch.device) -> DistillationTensorDataset:
        """Move every tensor field to one learner device while preserving labels."""

        return replace(
            self,
            student_obs=self.student_obs.to(device),
            teacher_obs=self.teacher_obs.to(device),
            teacher_actions=(
                None if self.teacher_actions is None else self.teacher_actions.to(device)
            ),
            commands=None if self.commands is None else self.commands.to(device),
            target_height=(None if self.target_height is None else self.target_height.to(device)),
            transition_ages=(
                None if self.transition_ages is None else self.transition_ages.to(device)
            ),
            command_before=(
                None if self.command_before is None else self.command_before.to(device)
            ),
            command_after=(None if self.command_after is None else self.command_after.to(device)),
        )

    def as_batch(self, *, start: int = 0, batch_size: int | None = None) -> DistillationBatch:
        if start < 0 or start >= self.num_samples:
            raise ValueError(f"start must be in [0, {self.num_samples}), got {start}")
        end = self.num_samples if batch_size is None else min(self.num_samples, start + batch_size)
        if end <= start:
            raise ValueError(f"batch_size must select at least one sample, got {batch_size}")
        return DistillationBatch(
            student_obs=self.student_obs[start:end],
            teacher_obs=self.teacher_obs[start:end],
            role_labels=None if self.role_labels is None else self.role_labels[start:end],
            teacher_actions=(
                None if self.teacher_actions is None else self.teacher_actions[start:end]
            ),
            commands=None if self.commands is None else self.commands[start:end],
            target_height=(None if self.target_height is None else self.target_height[start:end]),
            command_intents=(
                None if self.command_intents is None else self.command_intents[start:end]
            ),
            scenario_labels=(
                None if self.scenario_labels is None else self.scenario_labels[start:end]
            ),
            transition_ages=(
                None if self.transition_ages is None else self.transition_ages[start:end]
            ),
            command_before=(
                None if self.command_before is None else self.command_before[start:end]
            ),
            command_after=(None if self.command_after is None else self.command_after[start:end]),
        )


def build_distillation_dataset(
    student_obs: torch.Tensor,
    teacher_obs: torch.Tensor,
    *,
    expected_student_obs_dim: int | None = None,
    expected_teacher_obs_dim: int | None = None,
    expected_teacher_action_dim: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    role_labels: list[str] | tuple[str, ...] | None = None,
    teacher_actions: torch.Tensor | None = None,
    commands: torch.Tensor | None = None,
    target_height: torch.Tensor | None = None,
    command_intents: list[str] | tuple[str, ...] | None = None,
    scenario_labels: list[str] | tuple[str, ...] | None = None,
    transition_ages: torch.Tensor | None = None,
    command_before: torch.Tensor | None = None,
    command_after: torch.Tensor | None = None,
) -> DistillationTensorDataset:
    """Validate and package offline student/teacher observations for distillation."""

    _validate_obs_tensor(
        "student_obs",
        student_obs,
        expected_dim=expected_student_obs_dim,
    )
    _validate_obs_tensor(
        "teacher_obs",
        teacher_obs,
        expected_dim=expected_teacher_obs_dim,
    )
    if student_obs.shape[0] != teacher_obs.shape[0]:
        raise ValueError(
            "student/teacher dataset batch size mismatch: "
            f"student={student_obs.shape[0]} teacher={teacher_obs.shape[0]}"
        )
    if teacher_actions is not None:
        _validate_action_tensor(
            "teacher_actions",
            teacher_actions,
            expected_dim=expected_teacher_action_dim,
        )
        if student_obs.shape[0] != teacher_actions.shape[0]:
            raise ValueError(
                "student/teacher action dataset batch size mismatch: "
                f"student={student_obs.shape[0]} teacher_actions={teacher_actions.shape[0]}"
            )
    validated_commands = _validate_commands(
        commands,
        num_samples=int(student_obs.shape[0]),
    )
    validated_target_height = _validate_target_height(
        target_height,
        num_samples=int(student_obs.shape[0]),
    )
    metadata_dict = dict(metadata or {})
    metadata_role_labels = metadata_dict.get("role_labels")
    if role_labels is None and metadata_role_labels is not None:
        if not isinstance(metadata_role_labels, list | tuple):
            raise ValueError("metadata role_labels must be a list or tuple")
        role_labels = [str(label) for label in metadata_role_labels]
    metadata_command_intents = metadata_dict.get("command_intents")
    if command_intents is None and metadata_command_intents is not None:
        if not isinstance(metadata_command_intents, list | tuple):
            raise ValueError("metadata command_intents must be a list or tuple")
        command_intents = [str(intent) for intent in metadata_command_intents]
    validated_role_labels = _validate_role_labels(
        role_labels,
        num_samples=int(student_obs.shape[0]),
    )
    if command_intents is None and validated_commands is not None:
        command_intents = _command_intents_from_commands(
            validated_commands,
            xy_threshold=float(metadata_dict.get("command_xy_threshold", 0.05)),
            yaw_threshold=float(metadata_dict.get("command_yaw_threshold", 0.05)),
        )
        metadata_dict["command_intent_inference_source"] = "commands"
    if command_intents is None and validated_role_labels is not None:
        command_intents = _command_intents_from_role_labels(validated_role_labels)
        if command_intents is not None:
            metadata_dict["command_intent_inference_source"] = "role_labels"
    validated_command_intents = _validate_command_intents(
        command_intents,
        num_samples=int(student_obs.shape[0]),
    )
    (
        validated_scenario_labels,
        validated_transition_ages,
        validated_command_before,
        validated_command_after,
    ) = _validate_transition_fields(
        scenario_labels=scenario_labels,
        transition_ages=transition_ages,
        command_before=command_before,
        command_after=command_after,
        num_samples=int(student_obs.shape[0]),
    )
    if validated_role_labels is not None:
        metadata_dict["role_labels"] = list(validated_role_labels)
    if validated_command_intents is not None:
        metadata_dict["command_intents"] = list(validated_command_intents)
        metadata_dict["command_intent_counts"] = _label_counts(validated_command_intents)
    if validated_scenario_labels is not None:
        metadata_dict["scenario_labels"] = list(validated_scenario_labels)
        metadata_dict["scenario_counts"] = _label_counts(validated_scenario_labels)
        metadata_dict["transition_schema"] = "DISTILL-TRAIN-v002"
    return DistillationTensorDataset(
        student_obs=student_obs,
        teacher_obs=teacher_obs,
        metadata=metadata_dict,
        role_labels=validated_role_labels,
        teacher_actions=teacher_actions,
        commands=validated_commands,
        target_height=validated_target_height,
        command_intents=validated_command_intents,
        scenario_labels=validated_scenario_labels,
        transition_ages=validated_transition_ages,
        command_before=validated_command_before,
        command_after=validated_command_after,
    )


def make_fake_distillation_dataset(
    *,
    num_samples: int,
    student_obs_dim: int,
    teacher_obs_dim: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> DistillationTensorDataset:
    """Create a deterministic shape-valid dataset for offline connectivity probes."""

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    student_obs = torch.randn(
        int(num_samples),
        int(student_obs_dim),
        generator=generator,
    ).to(device)
    teacher_obs = torch.randn(
        int(num_samples),
        int(teacher_obs_dim),
        generator=generator,
    ).to(device)
    return build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=int(student_obs_dim),
        expected_teacher_obs_dim=int(teacher_obs_dim),
        metadata={"source": "fake_probe", "seed": int(seed)},
    )


def annotate_distillation_dataset_scenario(
    dataset: DistillationTensorDataset,
    scenario_label: str,
) -> DistillationTensorDataset:
    """Explicitly bind a legacy role dataset to a workflow scenario.

    This is a workflow annotation, not a teacher or action rewrite. It makes
    the scenario fields required by transition-aware aggregation explicit while
    preserving row-level role labels and cached targets.
    """

    scenario = str(scenario_label)
    if scenario not in _TRANSITION_SCENARIOS:
        raise ValueError(
            f"workflow scenario must be static_stand/walk_flat/walk_to_stop, got {scenario!r}"
        )
    if dataset.scenario_labels is not None:
        if any(label != scenario for label in dataset.scenario_labels):
            raise ValueError(
                f"dataset scenario labels do not match requested scenario {scenario!r}"
            )
        return dataset
    if scenario == "walk_to_stop":
        raise ValueError("walk_to_stop source must already contain transition fields")

    commands = dataset.commands
    if commands is None:
        if scenario == "walk_flat":
            raise ValueError("walk_flat scenario annotation requires dataset.commands")
        commands = torch.zeros(
            (dataset.num_samples, 3),
            dtype=dataset.student_obs.dtype,
            device=dataset.student_obs.device,
        )
    expected_intent = "active" if scenario == "walk_flat" else "inactive"
    if dataset.command_intents is not None and any(
        intent != expected_intent for intent in dataset.command_intents
    ):
        raise ValueError(f"{scenario} scenario annotation conflicts with command_intents")
    metadata = dict(dataset.metadata)
    metadata["scenario_annotation"] = "workflow_explicit"
    return build_distillation_dataset(
        dataset.student_obs,
        dataset.teacher_obs,
        expected_student_obs_dim=dataset.student_obs_dim,
        expected_teacher_obs_dim=dataset.teacher_obs_dim,
        expected_teacher_action_dim=dataset.teacher_action_dim,
        metadata=metadata,
        role_labels=dataset.role_labels,
        teacher_actions=dataset.teacher_actions,
        commands=commands,
        target_height=dataset.target_height,
        command_intents=dataset.command_intents,
        scenario_labels=(scenario,) * dataset.num_samples,
        transition_ages=torch.full(
            (dataset.num_samples,),
            -1,
            dtype=torch.int64,
            device=dataset.student_obs.device,
        ),
        command_before=commands.clone(),
        command_after=commands.clone(),
    )
