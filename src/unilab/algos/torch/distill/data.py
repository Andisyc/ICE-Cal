from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .trainer import DistillationBatch


def _validate_obs_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_dim: int | None,
) -> int:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {tuple(tensor.shape)}")
    obs_dim = int(tensor.shape[-1])
    if expected_dim is not None and obs_dim != int(expected_dim):
        raise ValueError(f"{name} dim mismatch: expected {int(expected_dim)}, got {obs_dim}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return obs_dim


def _validate_action_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_dim: int | None,
) -> int:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {tuple(tensor.shape)}")
    action_dim = int(tensor.shape[-1])
    if expected_dim is not None and action_dim != int(expected_dim):
        raise ValueError(f"{name} dim mismatch: expected {int(expected_dim)}, got {action_dim}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return action_dim


def _validate_role_labels(
    role_labels: list[str] | tuple[str, ...] | None,
    *,
    num_samples: int,
) -> tuple[str, ...] | None:
    if role_labels is None:
        return None
    if len(role_labels) != int(num_samples):
        raise ValueError(
            "role_labels length mismatch: "
            f"labels={len(role_labels)} samples={int(num_samples)}"
        )
    labels = tuple(str(label) for label in role_labels)
    if any(label == "" for label in labels):
        raise ValueError("role_labels must not contain empty labels")
    return labels


def _validate_commands(
    commands: torch.Tensor | None,
    *,
    num_samples: int,
) -> torch.Tensor | None:
    if commands is None:
        return None
    if commands.ndim != 2 or int(commands.shape[-1]) != 3:
        raise ValueError(f"commands must have shape (N, 3), got {tuple(commands.shape)}")
    if int(commands.shape[0]) != int(num_samples):
        raise ValueError(
            "commands batch size mismatch: "
            f"commands={int(commands.shape[0])} samples={int(num_samples)}"
        )
    if not torch.isfinite(commands).all():
        raise ValueError("commands must contain only finite values")
    return commands


def _validate_command_intents(
    command_intents: list[str] | tuple[str, ...] | None,
    *,
    num_samples: int,
) -> tuple[str, ...] | None:
    if command_intents is None:
        return None
    if len(command_intents) != int(num_samples):
        raise ValueError(
            "command_intents length mismatch: "
            f"intents={len(command_intents)} samples={int(num_samples)}"
        )
    intents = tuple(str(intent) for intent in command_intents)
    allowed = {"active", "inactive"}
    if any(intent not in allowed for intent in intents):
        raise ValueError("command_intents must contain only active/inactive labels")
    return intents


def _label_counts(labels: tuple[str, ...]) -> dict[str, int]:
    return {label: labels.count(label) for label in sorted(set(labels))}


@dataclass(frozen=True)
class DistillationTensorDataset:
    """In-memory offline distillation observations with explicit shape contracts."""

    student_obs: torch.Tensor
    teacher_obs: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    role_labels: tuple[str, ...] | None = None
    teacher_actions: torch.Tensor | None = None
    commands: torch.Tensor | None = None
    command_intents: tuple[str, ...] | None = None

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
            command_intents=(
                None if self.command_intents is None else self.command_intents[start:end]
            ),
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
    command_intents: list[str] | tuple[str, ...] | None = None,
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
    validated_command_intents = _validate_command_intents(
        command_intents,
        num_samples=int(student_obs.shape[0]),
    )
    if validated_role_labels is not None:
        metadata_dict["role_labels"] = list(validated_role_labels)
    if validated_command_intents is not None:
        metadata_dict["command_intents"] = list(validated_command_intents)
        metadata_dict["command_intent_counts"] = _label_counts(validated_command_intents)
    return DistillationTensorDataset(
        student_obs=student_obs,
        teacher_obs=teacher_obs,
        metadata=metadata_dict,
        role_labels=validated_role_labels,
        teacher_actions=teacher_actions,
        commands=validated_commands,
        command_intents=validated_command_intents,
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


def _source_value(source: Mapping[str, Any], key: str) -> Any:
    value = source.get(key)
    if value in (None, ""):
        raise ValueError(f"multitask source must define non-empty {key!r}")
    return value


def build_multitask_distillation_dataset(
    sources: Sequence[Mapping[str, Any]],
    *,
    expected_student_obs_dim: int | None = None,
    expected_teacher_obs_dim: int | None = None,
    expected_teacher_action_dim: int | None = None,
    device: str | torch.device = "cpu",
) -> DistillationTensorDataset:
    """Merge saved role-specific datasets into one cached-target dataset."""

    if not sources:
        raise ValueError("multitask distillation dataset requires at least one source")

    datasets: list[DistillationTensorDataset] = []
    source_roles: list[str] = []
    source_paths: list[str] = []
    source_sample_counts: list[int] = []
    source_metadata: list[dict[str, Any]] = []
    source_student_obs_dim: int | None = None
    source_teacher_obs_dim: int | None = None
    source_teacher_action_dim: int | None = None
    source_has_commands: bool | None = None
    source_has_command_intents: bool | None = None
    for source in sources:
        path = Path(_source_value(source, "path"))
        role = str(_source_value(source, "role"))
        dataset = load_distillation_dataset(
            path,
            expected_student_obs_dim=expected_student_obs_dim,
            expected_teacher_obs_dim=expected_teacher_obs_dim,
            expected_teacher_action_dim=expected_teacher_action_dim,
            device=device,
        )
        if dataset.teacher_actions is None:
            raise ValueError(
                f"multitask source {path} must contain cached teacher_actions"
            )
        has_commands = dataset.commands is not None
        if source_has_commands is None:
            source_has_commands = has_commands
        elif has_commands != source_has_commands:
            raise ValueError("multitask sources must either all include commands or none")
        has_command_intents = dataset.command_intents is not None
        if source_has_command_intents is None:
            source_has_command_intents = has_command_intents
        elif has_command_intents != source_has_command_intents:
            raise ValueError(
                "multitask sources must either all include command_intents or none"
            )
        if source_student_obs_dim is None:
            source_student_obs_dim = dataset.student_obs_dim
        elif dataset.student_obs_dim != source_student_obs_dim:
            raise ValueError(
                f"multitask source {path} role={role!r} student_obs dim mismatch: "
                f"expected {source_student_obs_dim}, got {dataset.student_obs_dim}"
            )
        if source_teacher_obs_dim is None:
            source_teacher_obs_dim = dataset.teacher_obs_dim
        elif dataset.teacher_obs_dim != source_teacher_obs_dim:
            raise ValueError(
                f"multitask source {path} role={role!r} teacher_obs dim mismatch: "
                f"expected {source_teacher_obs_dim}, got {dataset.teacher_obs_dim}"
            )
        if source_teacher_action_dim is None:
            source_teacher_action_dim = dataset.teacher_action_dim
        elif dataset.teacher_action_dim != source_teacher_action_dim:
            raise ValueError(
                f"multitask source {path} role={role!r} teacher_actions dim mismatch: "
                f"expected {source_teacher_action_dim}, got {dataset.teacher_action_dim}"
            )
        datasets.append(dataset)
        source_roles.append(role)
        source_paths.append(str(path))
        source_sample_counts.append(dataset.num_samples)
        source_metadata.append(dict(dataset.metadata))

    student_obs = torch.cat([dataset.student_obs for dataset in datasets], dim=0)
    teacher_obs = torch.cat([dataset.teacher_obs for dataset in datasets], dim=0)
    teacher_actions = torch.cat(
        [
            dataset.teacher_actions
            for dataset in datasets
            if dataset.teacher_actions is not None
        ],
        dim=0,
    )
    commands = (
        torch.cat([dataset.commands for dataset in datasets if dataset.commands is not None], dim=0)
        if source_has_commands
        else None
    )
    command_intents = (
        tuple(
            intent
            for dataset in datasets
            if dataset.command_intents is not None
            for intent in dataset.command_intents
        )
        if source_has_command_intents
        else None
    )
    role_labels = tuple(
        role
        for role, dataset in zip(source_roles, datasets, strict=True)
        for _ in range(dataset.num_samples)
    )
    metadata = {
        "source": "multitask_adapter",
        "source_count": len(datasets),
        "source_paths": source_paths,
        "source_roles": source_roles,
        "source_sample_counts": source_sample_counts,
        "source_metadata": source_metadata,
    }
    if command_intents is not None:
        metadata["command_intent_counts"] = _label_counts(command_intents)
    return build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=expected_student_obs_dim,
        expected_teacher_obs_dim=expected_teacher_obs_dim,
        expected_teacher_action_dim=expected_teacher_action_dim,
        metadata=metadata,
        role_labels=role_labels,
        teacher_actions=teacher_actions,
        commands=commands,
        command_intents=command_intents,
    )


def save_distillation_dataset(path: str | Path, dataset: DistillationTensorDataset) -> None:
    """Persist an offline distillation observation dataset."""

    payload = {
        "student_obs": dataset.student_obs.detach().cpu(),
        "teacher_obs": dataset.teacher_obs.detach().cpu(),
        "metadata": dict(dataset.metadata),
        "role_labels": None if dataset.role_labels is None else list(dataset.role_labels),
        "teacher_actions": (
            None if dataset.teacher_actions is None else dataset.teacher_actions.detach().cpu()
        ),
        "commands": None if dataset.commands is None else dataset.commands.detach().cpu(),
        "command_intents": (
            None if dataset.command_intents is None else list(dataset.command_intents)
        ),
        "student_obs_dim": dataset.student_obs_dim,
        "teacher_obs_dim": dataset.teacher_obs_dim,
        "teacher_action_dim": dataset.teacher_action_dim,
        "num_samples": dataset.num_samples,
    }
    torch.save(payload, Path(path))


def load_distillation_dataset(
    path: str | Path,
    *,
    expected_student_obs_dim: int | None = None,
    expected_teacher_obs_dim: int | None = None,
    expected_teacher_action_dim: int | None = None,
    device: str | torch.device = "cpu",
) -> DistillationTensorDataset:
    """Load and validate an offline distillation observation dataset."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    teacher_actions = payload.get("teacher_actions")
    commands = payload.get("commands")
    dataset = build_distillation_dataset(
        payload["student_obs"].to(device),
        payload["teacher_obs"].to(device),
        expected_student_obs_dim=expected_student_obs_dim,
        expected_teacher_obs_dim=expected_teacher_obs_dim,
        expected_teacher_action_dim=expected_teacher_action_dim,
        metadata=payload.get("metadata", {}),
        role_labels=payload.get("role_labels"),
        teacher_actions=None if teacher_actions is None else teacher_actions.to(device),
        commands=None if commands is None else commands.to(device),
        command_intents=payload.get("command_intents"),
    )
    expected_count = payload.get("num_samples")
    if expected_count is not None and int(expected_count) != dataset.num_samples:
        raise ValueError(
            "distillation dataset num_samples mismatch: "
            f"payload={expected_count} tensors={dataset.num_samples}"
        )
    return dataset
