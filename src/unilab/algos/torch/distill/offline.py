from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import save_distillation_checkpoint
from .data import DistillationTensorDataset
from .trainer import BehaviorDistillationTrainer


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
    last_expert_usage: tuple[float, ...] | None
    last_route_entropy: float | None


def run_offline_distillation_updates(
    trainer: BehaviorDistillationTrainer,
    dataset: DistillationTensorDataset,
    *,
    batch_size: int,
    max_updates: int,
    checkpoint_path: str | Path | None = None,
    teacher_metadata: Mapping[str, Any] | None = None,
    distill_runtime_cfg: Mapping[str, Any] | None = None,
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
    last_expert_usage: tuple[float, ...] | None = None
    last_route_entropy: float | None = None

    for update_idx in range(int(max_updates)):
        start = update_idx * int(batch_size)
        if start >= dataset.num_samples:
            break
        batch = dataset.as_batch(start=start, batch_size=int(batch_size))
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
        last_expert_usage=last_expert_usage,
        last_route_entropy=last_route_entropy,
    )
