from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import save_distillation_checkpoint
from .data import DistillationTensorDataset
from .trainer import BehaviorDistillationTrainer, DistillationBatch


@dataclass(frozen=True)
class OfflineDistillationRunResult:
    """Summary of a bounded offline behavior-distillation update loop."""

    update_count: int
    samples_seen: int
    last_loss: float
    last_student_grad_norm: float
    student_obs_shape: tuple[int, ...]
    teacher_obs_shape: tuple[int, ...]
    student_action_shape: tuple[int, ...]
    teacher_action_shape: tuple[int, ...]
    teacher_action_requires_grad: bool
    last_teacher_action_source: str
    checkpoint_path: Path | None
    losses: tuple[float, ...]
    student_grad_norms: tuple[float, ...]
    last_behavior_loss: float
    last_aux_loss: float
    last_role_loss: float
    last_role_target_count: int
    last_command_intent_loss: float
    last_command_intent_target_count: int
    last_expert_usage: tuple[float, ...] | None
    last_route_entropy: float | None
    balance_key: str
    batch_label_counts: tuple[dict[str, int], ...]
    last_balance_label_counts: dict[str, int]


def _indexed_batch(dataset: DistillationTensorDataset, indices: torch.Tensor) -> DistillationBatch:
    indices = indices.to(device=dataset.student_obs.device)
    role_labels = None
    if dataset.role_labels is not None:
        role_labels = tuple(dataset.role_labels[int(index)] for index in indices.detach().cpu())
    return DistillationBatch(
        student_obs=dataset.student_obs.index_select(0, indices),
        teacher_obs=dataset.teacher_obs.index_select(0, indices),
        role_labels=role_labels,
        teacher_actions=(
            None
            if dataset.teacher_actions is None
            else dataset.teacher_actions.index_select(0, indices)
        ),
        commands=None if dataset.commands is None else dataset.commands.index_select(0, indices),
        command_intents=(
            None
            if dataset.command_intents is None
            else tuple(dataset.command_intents[int(index)] for index in indices.detach().cpu())
        ),
    )


def _labels_for_balance_key(
    dataset: DistillationTensorDataset,
    balance_key: str,
) -> tuple[str, ...] | None:
    if balance_key == "none":
        return None
    if balance_key == "role":
        if dataset.role_labels is None:
            raise ValueError("offline balance_key='role' requires dataset.role_labels")
        return dataset.role_labels
    if balance_key == "command_intent":
        if dataset.command_intents is None:
            raise ValueError(
                "offline balance_key='command_intent' requires dataset.command_intents"
            )
        return dataset.command_intents
    raise ValueError(
        "offline balance_key must be one of 'none', 'role', or 'command_intent', "
        f"got {balance_key!r}"
    )


def _balanced_batch_indices(
    labels: tuple[str, ...],
    *,
    batch_size: int,
    balanced_labels: Sequence[str] | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, int]]:
    selected_labels = (
        tuple(str(label) for label in balanced_labels)
        if balanced_labels
        else tuple(sorted(set(labels)))
    )
    if not selected_labels:
        raise ValueError("offline balanced sampler requires at least one label")
    if len(set(selected_labels)) != len(selected_labels):
        raise ValueError(f"offline balanced labels must be unique: {selected_labels}")
    if int(batch_size) < len(selected_labels):
        raise ValueError(
            "offline balanced sampler requires batch_size >= number of labels: "
            f"batch_size={int(batch_size)} labels={len(selected_labels)}"
        )

    label_to_indices = {
        label: torch.as_tensor(
            [idx for idx, value in enumerate(labels) if value == label],
            dtype=torch.long,
        )
        for label in selected_labels
    }
    missing = [label for label, indices in label_to_indices.items() if indices.numel() == 0]
    if missing:
        raise ValueError(f"offline balanced sampler missing labels: {missing}")

    base_quota = int(batch_size) // len(selected_labels)
    remainder = int(batch_size) % len(selected_labels)
    chunks: list[torch.Tensor] = []
    counts: dict[str, int] = {}
    for label_idx, label in enumerate(selected_labels):
        quota = base_quota + (1 if label_idx < remainder else 0)
        source = label_to_indices[label]
        picks = torch.randint(
            int(source.numel()),
            (quota,),
            generator=generator,
            dtype=torch.long,
        )
        chunks.append(source.index_select(0, picks))
        counts[label] = int(quota)
    indices = torch.cat(chunks, dim=0)
    order = torch.randperm(int(indices.numel()), generator=generator)
    return indices.index_select(0, order), counts


def run_offline_distillation_updates(
    trainer: BehaviorDistillationTrainer,
    dataset: DistillationTensorDataset,
    *,
    batch_size: int,
    max_updates: int,
    checkpoint_path: str | Path | None = None,
    teacher_metadata: Mapping[str, Any] | None = None,
    distill_runtime_cfg: Mapping[str, Any] | None = None,
    repeat_dataset: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
    balance_key: str = "none",
    balanced_labels: Sequence[str] | None = None,
) -> OfflineDistillationRunResult:
    """Run a bounded sequential offline distillation loop over a validated dataset."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_updates <= 0:
        raise ValueError(f"max_updates must be positive, got {max_updates}")
    if dataset.num_samples <= 0:
        raise ValueError("offline distillation dataset must contain at least one sample")

    losses: list[float] = []
    grad_norms: list[float] = []
    samples_seen = 0
    last_student_obs_shape: tuple[int, ...] | None = None
    last_teacher_obs_shape: tuple[int, ...] | None = None
    last_student_action_shape: tuple[int, ...] | None = None
    last_teacher_action_shape: tuple[int, ...] | None = None
    last_teacher_action_requires_grad = False
    last_teacher_action_source = "teacher"
    last_behavior_loss = 0.0
    last_aux_loss = 0.0
    last_role_loss = 0.0
    last_role_target_count = 0
    last_command_intent_loss = 0.0
    last_command_intent_target_count = 0
    last_expert_usage: tuple[float, ...] | None = None
    last_route_entropy: float | None = None
    resolved_balance_key = str(balance_key)
    balance_labels = _labels_for_balance_key(dataset, resolved_balance_key)
    batch_label_counts: list[dict[str, int]] = []
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))

    def _order() -> torch.Tensor:
        if shuffle:
            return torch.randperm(dataset.num_samples, generator=generator)
        return torch.arange(dataset.num_samples)

    order = _order()
    cursor = 0

    for update_idx in range(int(max_updates)):
        label_counts: dict[str, int] = {}
        if balance_labels is not None:
            indices, label_counts = _balanced_batch_indices(
                balance_labels,
                batch_size=int(batch_size),
                balanced_labels=balanced_labels,
                generator=generator,
            )
            batch = _indexed_batch(dataset, indices)
        elif repeat_dataset or shuffle:
            if cursor >= dataset.num_samples:
                if not repeat_dataset:
                    break
                order = _order()
                cursor = 0
            end = min(dataset.num_samples, cursor + int(batch_size))
            batch = _indexed_batch(dataset, order[cursor:end])
            cursor = end
        else:
            start = update_idx * int(batch_size)
            if start >= dataset.num_samples:
                break
            batch = dataset.as_batch(start=start, batch_size=int(batch_size))
        batch_label_counts.append(label_counts)
        stats = trainer.update(batch)

        samples_seen += int(batch.student_obs.shape[0])
        losses.append(stats.loss)
        grad_norms.append(stats.student_grad_norm)
        last_student_obs_shape = tuple(batch.student_obs.shape)
        last_teacher_obs_shape = tuple(batch.teacher_obs.shape)
        last_student_action_shape = stats.student_action_shape
        last_teacher_action_shape = stats.teacher_action_shape
        last_teacher_action_requires_grad = stats.teacher_action_requires_grad
        last_teacher_action_source = stats.teacher_action_source
        last_behavior_loss = stats.behavior_loss
        last_aux_loss = stats.aux_loss
        last_role_loss = stats.role_loss
        last_role_target_count = stats.role_target_count
        last_command_intent_loss = stats.command_intent_loss
        last_command_intent_target_count = stats.command_intent_target_count
        last_expert_usage = stats.expert_usage
        last_route_entropy = stats.route_entropy

    if not losses:
        raise ValueError("offline distillation loop did not execute any update")

    resolved_checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if resolved_checkpoint_path is not None:
        save_distillation_checkpoint(
            resolved_checkpoint_path,
            student=trainer.student,
            optimizer=trainer.optimizer,
            agent_steps=samples_seen,
            teacher_metadata=teacher_metadata,
            distill_runtime_cfg=distill_runtime_cfg,
        )

    return OfflineDistillationRunResult(
        update_count=trainer.update_count,
        samples_seen=samples_seen,
        last_loss=losses[-1],
        last_student_grad_norm=grad_norms[-1],
        student_obs_shape=last_student_obs_shape or (),
        teacher_obs_shape=last_teacher_obs_shape or (),
        student_action_shape=last_student_action_shape or (),
        teacher_action_shape=last_teacher_action_shape or (),
        teacher_action_requires_grad=last_teacher_action_requires_grad,
        last_teacher_action_source=last_teacher_action_source,
        checkpoint_path=resolved_checkpoint_path,
        losses=tuple(losses),
        student_grad_norms=tuple(grad_norms),
        last_behavior_loss=last_behavior_loss,
        last_aux_loss=last_aux_loss,
        last_role_loss=last_role_loss,
        last_role_target_count=last_role_target_count,
        last_command_intent_loss=last_command_intent_loss,
        last_command_intent_target_count=last_command_intent_target_count,
        last_expert_usage=last_expert_usage,
        last_route_entropy=last_route_entropy,
        balance_key=resolved_balance_key,
        batch_label_counts=tuple(batch_label_counts),
        last_balance_label_counts=batch_label_counts[-1] if batch_label_counts else {},
    )
