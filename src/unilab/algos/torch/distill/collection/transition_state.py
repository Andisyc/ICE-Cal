"""State and coverage helpers for walk/stand transition collection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.collection.common import command_active_mask


@dataclass(frozen=True)
class _TransitionCaseAssignment:
    walk_commands: np.ndarray
    post_switch_target_heights: np.ndarray | None
    case_commands: np.ndarray
    case_target_heights: np.ndarray | None
    env_case_indices: np.ndarray
    active_command_rows: np.ndarray
    nominal_target_rows: np.ndarray | None
    post_switch_target_rows: np.ndarray | None


@dataclass
class TransitionRowBuffer:
    """Own the ordered tensor and label rows captured by transition collection."""

    student_obs: list[torch.Tensor] = field(default_factory=list)
    teacher_obs: list[torch.Tensor] = field(default_factory=list)
    teacher_actions: list[torch.Tensor] = field(default_factory=list)
    commands: list[torch.Tensor] = field(default_factory=list)
    target_height: list[torch.Tensor] = field(default_factory=list)
    command_before: list[torch.Tensor] = field(default_factory=list)
    command_after: list[torch.Tensor] = field(default_factory=list)
    transition_ages: list[torch.Tensor] = field(default_factory=list)
    role_labels: list[str] = field(default_factory=list)
    command_intents: list[str] = field(default_factory=list)
    scenario_labels: list[str] = field(default_factory=list)

    def append(
        self,
        *,
        student_obs: torch.Tensor,
        teacher_obs: torch.Tensor,
        teacher_actions: torch.Tensor,
        role_labels: tuple[str, ...],
        command_intents: tuple[str, ...],
        scenario_labels: tuple[str, ...],
        transition_ages: torch.Tensor,
        command_before: torch.Tensor,
        command_after: torch.Tensor,
        commands: torch.Tensor | None = None,
        target_height: torch.Tensor | None = None,
    ) -> None:
        self.student_obs.append(student_obs)
        self.teacher_obs.append(teacher_obs)
        self.teacher_actions.append(teacher_actions)
        if commands is not None:
            self.commands.append(commands)
        if target_height is not None:
            self.target_height.append(target_height)
        self.command_before.append(command_before)
        self.command_after.append(command_after)
        self.transition_ages.append(transition_ages)
        self.role_labels.extend(role_labels)
        self.command_intents.extend(command_intents)
        self.scenario_labels.extend(scenario_labels)


def _build_transition_case_assignment(
    *,
    num_envs: int,
    walk_command: np.ndarray | tuple[float, float, float],
    walk_commands: Sequence[Sequence[float]] | np.ndarray | None,
    target_height_info_key: str | None,
    nominal_walk_target_height: float | None,
    post_switch_target_heights: Sequence[float] | np.ndarray | None,
) -> _TransitionCaseAssignment:
    if walk_commands is None or np.asarray(walk_commands).size == 0:
        fallback_command = np.asarray(walk_command, dtype=np.float32)
        if fallback_command.shape != (3,):
            raise ValueError(
                "walk_command must have shape (3,) when transition walk commands "
                f"are not configured, got {fallback_command.shape}"
            )
        configured_commands = fallback_command.reshape(1, 3)
    else:
        configured_commands = np.asarray(walk_commands, dtype=np.float32)
    if configured_commands.ndim != 2 or configured_commands.shape[1] != 3:
        raise ValueError(
            "transition walk commands must have shape (num_commands, 3), "
            f"got {configured_commands.shape}"
        )
    if not np.all(np.isfinite(configured_commands)):
        raise ValueError("transition walk commands must contain only finite values")
    active_commands = command_active_mask(
        configured_commands,
        xy_threshold=0.05,
        yaw_threshold=0.05,
    )
    if not bool(np.all(active_commands)):
        invalid = np.flatnonzero(~active_commands).tolist()
        raise ValueError(
            "every transition walk command must be active under the command thresholds; "
            f"inactive_indices={invalid}"
        )

    configured_targets = (
        np.asarray([], dtype=np.float32)
        if post_switch_target_heights is None
        else np.asarray(post_switch_target_heights, dtype=np.float32)
    )
    if configured_targets.ndim != 1:
        raise ValueError(
            "post-switch target heights must have shape (num_heights,), "
            f"got {configured_targets.shape}"
        )
    if not np.all(np.isfinite(configured_targets)):
        raise ValueError("post-switch target heights must contain only finite values")

    if configured_targets.size == 0:
        if nominal_walk_target_height is not None:
            raise ValueError("nominal_walk_target_height requires post_switch_target_heights")
        case_commands = configured_commands.copy()
        case_targets = None
        nominal_target_rows = None
        post_switch_target_rows = None
    else:
        if target_height_info_key in (None, ""):
            raise ValueError("post-switch target heights require target_height_info_key")
        if nominal_walk_target_height is None or not np.isfinite(float(nominal_walk_target_height)):
            raise ValueError(
                "post-switch target heights require a finite nominal_walk_target_height"
            )
        case_commands = np.repeat(
            configured_commands,
            repeats=int(configured_targets.shape[0]),
            axis=0,
        )
        case_targets = np.tile(configured_targets, int(configured_commands.shape[0]))
        nominal_target_rows = np.full(
            (int(num_envs), 1),
            float(nominal_walk_target_height),
            dtype=np.float32,
        )
        post_switch_target_rows = None

    case_count = int(case_commands.shape[0])
    if int(num_envs) < case_count:
        raise ValueError(
            "transition collection requires at least one env row per command-height case: "
            f"num_envs={int(num_envs)} case_count={case_count}"
        )
    env_case_indices = np.arange(int(num_envs), dtype=np.int64) % case_count
    active_command_rows = case_commands[env_case_indices].copy()
    if case_targets is not None:
        post_switch_target_rows = case_targets[env_case_indices, None].copy()

    return _TransitionCaseAssignment(
        walk_commands=configured_commands,
        post_switch_target_heights=(None if configured_targets.size == 0 else configured_targets),
        case_commands=case_commands,
        case_target_heights=case_targets,
        env_case_indices=env_case_indices,
        active_command_rows=active_command_rows,
        nominal_target_rows=nominal_target_rows,
        post_switch_target_rows=post_switch_target_rows,
    )


def _validate_transition_coverage(
    *,
    rows: TransitionRowBuffer,
    num_samples: int,
    switch_count: int,
    post_switch_rows: int,
    nominal_settle_steps: int,
    min_post_switch_steps: int,
    case_height_tracking_counts: np.ndarray,
    case_max_height_tracking_ages: np.ndarray,
) -> tuple[torch.Tensor, int, int]:
    if switch_count == 0 or post_switch_rows == 0:
        raise RuntimeError(
            "transition collection did not produce both pre-switch and post-switch rows"
        )
    missing_case_indices = np.flatnonzero(case_height_tracking_counts == 0).tolist()
    if missing_case_indices:
        raise RuntimeError(
            "transition collection did not produce requested-height rows for every case: "
            f"missing_case_indices={missing_case_indices}"
        )
    transition_ages_tensor = torch.cat(rows.transition_ages, dim=0)[: int(num_samples)]
    post_switch_ages = transition_ages_tensor[transition_ages_tensor >= 0]
    max_post_switch_age = int(post_switch_ages.max().item()) if post_switch_ages.numel() else -1
    max_height_tracking_age = max_post_switch_age - int(nominal_settle_steps)
    if int(min_post_switch_steps) > 0 and max_height_tracking_age < int(
        min_post_switch_steps
    ) - 1:
        raise RuntimeError(
            "transition collection did not reach the configured requested-height horizon: "
            f"max_height_tracking_age={max_height_tracking_age} "
            f"required={int(min_post_switch_steps) - 1}"
        )
    if int(min_post_switch_steps) > 0:
        required_case_age = int(min_post_switch_steps) - 1
        short_case_indices = np.flatnonzero(
            case_max_height_tracking_ages < required_case_age
        ).tolist()
        if short_case_indices:
            raise RuntimeError(
                "transition collection did not reach the configured requested-height horizon "
                "for every case: "
                f"short_case_indices={short_case_indices} required={required_case_age}"
            )
    return transition_ages_tensor, max_post_switch_age, max_height_tracking_age


def _build_transition_case_metadata(
    *,
    transition_cases: _TransitionCaseAssignment,
    case_sample_counts: np.ndarray,
    case_post_switch_counts: np.ndarray,
    case_max_post_switch_ages: np.ndarray,
    case_nominal_settle_counts: np.ndarray,
    case_height_tracking_counts: np.ndarray,
    case_max_height_tracking_ages: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case_index, case_command in enumerate(transition_cases.case_commands):
        case_target = (
            None
            if transition_cases.case_target_heights is None
            else float(transition_cases.case_target_heights[case_index])
        )
        result.append(
            {
                "index": int(case_index),
                "walk_command": case_command.tolist(),
                "post_switch_target_height": case_target,
                "sample_count": int(case_sample_counts[case_index]),
                "post_switch_sample_count": int(case_post_switch_counts[case_index]),
                "max_post_switch_age": int(case_max_post_switch_ages[case_index]),
                "nominal_settle_sample_count": int(case_nominal_settle_counts[case_index]),
                "height_tracking_sample_count": int(case_height_tracking_counts[case_index]),
                "max_height_tracking_age": int(case_max_height_tracking_ages[case_index]),
            }
        )
    return result
