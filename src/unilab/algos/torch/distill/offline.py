from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import save_distillation_checkpoint
from .data import DistillationTensorDataset
from .performance import (
    DistillationStageObservation,
    DistillationStageObservationAccumulator,
)
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
    last_behavior_action_shape: tuple[int, ...]
    last_behavior_action_source: str
    last_behavior_target_count: int
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
    performance_stage_observations: tuple[DistillationStageObservation, ...] = ()


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
        scenario_labels=(
            None
            if dataset.scenario_labels is None
            else tuple(dataset.scenario_labels[int(index)] for index in indices.detach().cpu())
        ),
        transition_ages=(
            None
            if dataset.transition_ages is None
            else dataset.transition_ages.index_select(0, indices)
        ),
        command_before=(
            None
            if dataset.command_before is None
            else dataset.command_before.index_select(0, indices)
        ),
        command_after=(
            None
            if dataset.command_after is None
            else dataset.command_after.index_select(0, indices)
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
    if balance_key == "scenario":
        if dataset.scenario_labels is None:
            raise ValueError("offline balance_key='scenario' requires dataset.scenario_labels")
        return dataset.scenario_labels
    raise ValueError(
        "offline balance_key must be one of 'none', 'role', 'command_intent', or 'scenario', "
        f"got {balance_key!r}"
    )


def _balanced_batch_indices(
    labels: tuple[str, ...],
    *,
    batch_size: int,
    balanced_labels: Sequence[str] | None,
    balance_quotas: Mapping[str, float] | None,
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

    label_to_indices = _build_balanced_label_pools(labels, selected_labels)
    return _sample_balanced_batch_indices_from_pools(
        label_to_indices,
        batch_size=batch_size,
        balance_quotas=balance_quotas,
        generator=generator,
    )


def _build_balanced_label_pools(
    labels: tuple[str, ...],
    selected_labels: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Build immutable CPU row-index pools for the selected balance labels."""
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
    return label_to_indices


def _sample_balanced_batch_indices_from_pools(
    label_to_indices: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    balance_quotas: Mapping[str, float] | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Sample one balanced CPU index batch from prevalidated label pools."""
    selected_labels = tuple(label_to_indices)

    if balance_quotas:
        quota_values = {str(label): float(value) for label, value in balance_quotas.items()}
        unknown = sorted(set(quota_values) - set(selected_labels))
        missing = sorted(set(selected_labels) - set(quota_values))
        if unknown or missing:
            raise ValueError(
                "offline balance_quotas labels must match balanced_labels: "
                f"unknown={unknown} missing={missing}"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in quota_values.values()):
            raise ValueError("offline balance_quotas must contain finite positive weights")
        total_weight = sum(quota_values[label] for label in selected_labels)
        exact_quotas = [
            int(batch_size) * quota_values[label] / total_weight for label in selected_labels
        ]
        counts_list = [int(quota) for quota in exact_quotas]
        if any(count < 1 for count in counts_list):
            raise ValueError(
                "offline balance_quotas must allocate at least one sample per label: "
                f"batch_size={int(batch_size)} quotas={quota_values}"
            )
        remainder = int(batch_size) - sum(counts_list)
        order = sorted(
            range(len(selected_labels)),
            key=lambda index: exact_quotas[index] - counts_list[index],
            reverse=True,
        )
        for index in order[:remainder]:
            counts_list[index] += 1
    else:
        base_quota = int(batch_size) // len(selected_labels)
        remainder = int(batch_size) % len(selected_labels)
        counts_list = [
            base_quota + (1 if index < remainder else 0) for index in range(len(selected_labels))
        ]
    chunks: list[torch.Tensor] = []
    counts: dict[str, int] = {}
    for label_idx, label in enumerate(selected_labels):
        quota = counts_list[label_idx]
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


def _required_balanced_replay_updates(
    labels: tuple[str, ...],
    *,
    batch_size: int,
    balanced_labels: Sequence[str] | None,
    balance_quotas: Mapping[str, float] | None,
    replay_labels: Sequence[str],
    replay_passes: int,
) -> int:
    """Return updates needed for expected replay passes of selected labels."""

    if int(replay_passes) <= 0 or not replay_labels:
        return 0
    selected_labels = (
        tuple(str(label) for label in balanced_labels)
        if balanced_labels
        else tuple(sorted(set(labels)))
    )
    if not selected_labels:
        raise ValueError("offline replay requires at least one balanced label")
    replay_label_set = {str(label) for label in replay_labels}
    unknown = sorted(replay_label_set - set(selected_labels))
    if unknown:
        raise ValueError(
            f"offline replay labels must be present in balanced_labels: unknown={unknown}"
        )
    if not balance_quotas:
        base = int(batch_size) // len(selected_labels)
        remainder = int(batch_size) % len(selected_labels)
        batch_counts = {
            label: base + int(index < remainder) for index, label in enumerate(selected_labels)
        }
    else:
        quota_values = {str(label): float(value) for label, value in balance_quotas.items()}
        unknown = sorted(set(quota_values) - set(selected_labels))
        missing = sorted(set(selected_labels) - set(quota_values))
        if unknown or missing:
            raise ValueError(
                "offline balance_quotas labels must match balanced_labels: "
                f"unknown={unknown} missing={missing}"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in quota_values.values()):
            raise ValueError("offline balance_quotas must contain finite positive weights")
        total_weight = sum(quota_values[label] for label in selected_labels)
        exact = {
            label: int(batch_size * quota_values[label] / total_weight) for label in selected_labels
        }
        remainder = int(batch_size) - sum(exact.values())
        order = sorted(
            selected_labels,
            key=lambda label: batch_size * quota_values[label] / total_weight - exact[label],
            reverse=True,
        )
        for label in order[:remainder]:
            exact[label] += 1
        batch_counts = exact

    required = 0
    for label in sorted(replay_label_set):
        dataset_count = sum(value == label for value in labels)
        samples_per_update = int(batch_counts[label])
        if dataset_count <= 0 or samples_per_update <= 0:
            raise ValueError(
                "offline replay label has no usable samples: "
                f"label={label!r} dataset_count={dataset_count} "
                f"batch_count={samples_per_update}"
            )
        required = max(
            required,
            int(math.ceil(dataset_count * int(replay_passes) / samples_per_update)),
        )
    return required


def required_balanced_replay_updates(
    dataset: DistillationTensorDataset,
    *,
    balance_key: str,
    batch_size: int,
    balanced_labels: Sequence[str] | None,
    balance_quotas: Mapping[str, float] | None,
    replay_labels: Sequence[str],
    replay_passes: int,
) -> int:
    """Compute the minimum update budget for a balanced replay contract."""

    labels = _labels_for_balance_key(dataset, str(balance_key))
    return _required_balanced_replay_updates(
        labels or (),
        batch_size=int(batch_size),
        balanced_labels=balanced_labels,
        balance_quotas=balance_quotas,
        replay_labels=replay_labels,
        replay_passes=int(replay_passes),
    )


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
    balance_quotas: Mapping[str, float] | None = None,
    min_balanced_replay_passes: int = 0,
    min_balanced_replay_labels: Sequence[str] | None = None,
    progress_interval: int = 0,
    progress_callback: Callable[[int, int, Any], None] | None = None,
    performance_clock: Callable[[], float] | None = None,
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
    last_behavior_action_shape: tuple[int, ...] | None = None
    last_behavior_action_source = "student_action"
    last_behavior_target_count = 0
    last_aux_loss = 0.0
    last_role_loss = 0.0
    last_role_target_count = 0
    last_command_intent_loss = 0.0
    last_command_intent_target_count = 0
    last_expert_usage: tuple[float, ...] | None = None
    last_route_entropy: float | None = None
    resolved_balance_key = str(balance_key)
    balance_labels = _labels_for_balance_key(dataset, resolved_balance_key)
    replay_labels = tuple(str(label) for label in (min_balanced_replay_labels or ()))
    required_updates = _required_balanced_replay_updates(
        balance_labels or (),
        batch_size=int(batch_size),
        balanced_labels=balanced_labels,
        balance_quotas=balance_quotas,
        replay_labels=replay_labels,
        replay_passes=int(min_balanced_replay_passes),
    )
    if int(max_updates) < required_updates:
        raise ValueError(
            "offline balanced replay budget is too small: "
            f"max_updates={int(max_updates)} required_updates={required_updates} "
            f"replay_labels={list(replay_labels)} "
            f"replay_passes={int(min_balanced_replay_passes)} "
            f"batch_size={int(batch_size)}"
        )
    batch_label_counts: list[dict[str, int]] = []
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    progress_interval = int(progress_interval)
    if progress_interval < 0:
        raise ValueError(f"progress_interval must be non-negative, got {progress_interval}")

    def _order() -> torch.Tensor:
        if shuffle:
            return torch.randperm(dataset.num_samples, generator=generator)
        return torch.arange(dataset.num_samples)

    order = _order()
    cursor = 0
    performance = (
        None
        if performance_clock is None
        else DistillationStageObservationAccumulator(clock=performance_clock)
    )

    for update_idx in range(int(max_updates)):
        label_counts: dict[str, int] = {}
        staging_span = (
            nullcontext() if performance is None else performance.measure("learner_batch_staging")
        )
        with staging_span:
            if balance_labels is not None:
                indices, label_counts = _balanced_batch_indices(
                    balance_labels,
                    batch_size=int(batch_size),
                    balanced_labels=balanced_labels,
                    balance_quotas=balance_quotas,
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
        stats = trainer.update(batch, performance=performance)

        completed_updates = update_idx + 1
        if progress_interval > 0 and (
            completed_updates % progress_interval == 0 or completed_updates == int(max_updates)
        ):
            if progress_callback is not None:
                progress_callback(completed_updates, int(max_updates), stats)
            else:
                print(
                    "[distill-progress] "
                    f"updates={completed_updates}/{int(max_updates)} "
                    f"loss={stats.loss:.6f}",
                    flush=True,
                )

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
        last_behavior_action_shape = stats.behavior_action_shape
        last_behavior_action_source = stats.behavior_action_source
        last_behavior_target_count = stats.behavior_target_count
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
        checkpoint_span = (
            nullcontext() if performance is None else performance.measure("checkpoint_save")
        )
        with checkpoint_span:
            save_distillation_checkpoint(
                resolved_checkpoint_path,
                student=trainer.student,
                optimizer=trainer.optimizer,
                agent_steps=samples_seen,
                teacher_metadata=teacher_metadata,
                distill_runtime_cfg=distill_runtime_cfg,
            )

    performance_observations = (
        ()
        if performance is None
        else tuple(
            performance.observation(
                stage=stage,
                row_count=samples_seen,
                env_step_count=0,
            )
            for stage in (
                "learner_batch_staging",
                "learner_forward",
                "learner_backward",
                "optimizer_step",
                "checkpoint_save",
            )
        )
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
        last_behavior_action_shape=last_behavior_action_shape or (),
        last_behavior_action_source=last_behavior_action_source,
        last_behavior_target_count=last_behavior_target_count,
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
        performance_stage_observations=performance_observations,
    )
