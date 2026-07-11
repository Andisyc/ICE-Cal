from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import save_distillation_checkpoint
from .collector import collect_distillation_dataset_from_env
from .data import DistillationTensorDataset, build_distillation_dataset
from .offline import OfflineDistillationRunResult, run_offline_distillation_updates
from .trainer import BehaviorDistillationTrainer


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
    samples_collected = 0
    samples_seen = 0
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
            rollout_policy=trainer.student,
            command_sample_filter=str(command_sample_filter),
            command_info_key=str(command_info_key),
            command_xy_threshold=float(command_xy_threshold),
            command_yaw_threshold=float(command_yaw_threshold),
            max_env_steps=max_env_steps,
            metadata={"dagger_iteration": iteration + 1},
        )
        dataset = _attach_role_label(dataset, role_label)
        collection_metadata.append(dict(dataset.metadata))
        samples_collected += dataset.num_samples

        result = run_offline_distillation_updates(
            trainer,
            dataset,
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
