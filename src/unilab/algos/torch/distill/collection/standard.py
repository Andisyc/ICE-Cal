"""Public adapter for standard role distillation collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from unilab.algos.torch.distill.collection.standard_transaction import (
    StandardCollectionSpec,
    StandardCollectionTransaction,
)
from unilab.algos.torch.distill.datasets.dataset import DistillationTensorDataset


def collect_distillation_dataset_from_env(
    env: Any,
    *,
    num_samples: int,
    expected_student_obs_dim: int,
    expected_teacher_obs_dim: int,
    teacher_obs_key: str = "obs",
    teacher_projection: str = "identity",
    student_projection: str = "identity",
    student_drop_index: int | None = None,
    action_mode: str = "zero",
    action_seed: int | None = None,
    teacher_policy: torch.nn.Module | None = None,
    rollout_policy: torch.nn.Module | None = None,
    command_sample_filter: str = "none",
    command_info_key: str = "commands",
    target_height_info_key: str | None = None,
    command_xy_threshold: float = 0.05,
    command_yaw_threshold: float = 0.05,
    max_env_steps: int | None = None,
    role_label: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    initial_reset: tuple[Any, Any] | None = None,
    performance_clock: Callable[[], float] | None = None,
) -> DistillationTensorDataset:
    """Collect a fixed-size dataset while preserving the legacy public API."""

    spec = StandardCollectionSpec(
        num_samples=int(num_samples),
        expected_student_obs_dim=int(expected_student_obs_dim),
        expected_teacher_obs_dim=int(expected_teacher_obs_dim),
        teacher_obs_key=str(teacher_obs_key),
        teacher_projection=str(teacher_projection),
        student_projection=str(student_projection),
        student_drop_index=student_drop_index,
        action_mode=str(action_mode),
        action_seed=None if action_seed is None else int(action_seed),
        command_sample_filter=str(command_sample_filter),
        command_info_key=str(command_info_key),
        target_height_info_key=target_height_info_key,
        command_xy_threshold=float(command_xy_threshold),
        command_yaw_threshold=float(command_yaw_threshold),
        max_env_steps=None if max_env_steps is None else int(max_env_steps),
        role_label=role_label,
        metadata=metadata,
    )
    return StandardCollectionTransaction(
        env,
        spec,
        teacher_policy=teacher_policy,
        rollout_policy=rollout_policy,
        initial_reset=initial_reset,
        performance_clock=performance_clock,
    ).run()
