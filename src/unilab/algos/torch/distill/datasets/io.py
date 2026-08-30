from __future__ import annotations

import os
import threading
from pathlib import Path

import torch

from unilab.algos.torch.distill.datasets.dataset import (
    DistillationTensorDataset,
    build_distillation_dataset,
)
from unilab.algos.torch.distill.datasets.diagnostics import (
    _abort_for_native_capture,
    _emit_data_runtime,
    _native_abort_for_impossible_callable_error_requested,
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
        "target_height": (
            None if dataset.target_height is None else dataset.target_height.detach().cpu()
        ),
        "command_intents": (
            None if dataset.command_intents is None else list(dataset.command_intents)
        ),
        "scenario_labels": (
            None if dataset.scenario_labels is None else list(dataset.scenario_labels)
        ),
        "transition_ages": (
            None if dataset.transition_ages is None else dataset.transition_ages.detach().cpu()
        ),
        "command_before": (
            None if dataset.command_before is None else dataset.command_before.detach().cpu()
        ),
        "command_after": (
            None if dataset.command_after is None else dataset.command_after.detach().cpu()
        ),
        "student_obs_dim": dataset.student_obs_dim,
        "teacher_obs_dim": dataset.teacher_obs_dim,
        "teacher_action_dim": dataset.teacher_action_dim,
        "num_samples": dataset.num_samples,
    }
    resolved_path = Path(path)
    _emit_data_runtime(
        "serialization/before_torch_save",
        path=str(resolved_path),
        payload_keys=sorted(payload),
        payload_value_types={key: type(value).__name__ for key, value in payload.items()},
        num_samples=dataset.num_samples,
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved_path.with_name(
        f".{resolved_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(resolved_path)
    except Exception as error:
        native_abort_requested = _native_abort_for_impossible_callable_error_requested(error)
        _emit_data_runtime(
            "serialization/torch_save_failure",
            path=str(resolved_path),
            tmp_path=str(tmp_path),
            error_type=type(error).__name__,
            error_repr=repr(error),
            native_abort_requested=native_abort_requested,
        )
        if native_abort_requested:
            _abort_for_native_capture()
        raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    _emit_data_runtime(
        "serialization/after_torch_save",
        path=str(resolved_path),
        payload_keys=sorted(payload),
        file_exists=resolved_path.is_file(),
        file_size=resolved_path.stat().st_size if resolved_path.is_file() else None,
    )


def load_distillation_dataset(
    path: str | Path,
    *,
    expected_student_obs_dim: int | None = None,
    expected_teacher_obs_dim: int | None = None,
    expected_teacher_action_dim: int | None = None,
    device: str | torch.device = "cpu",
) -> DistillationTensorDataset:
    """Load and validate an offline distillation observation dataset."""

    resolved_path = Path(path)
    _emit_data_runtime(
        "serialization/before_torch_load",
        path=str(resolved_path),
        device=str(device),
        file_exists=resolved_path.is_file(),
        file_size=resolved_path.stat().st_size if resolved_path.is_file() else None,
    )
    try:
        payload = torch.load(resolved_path, map_location=device, weights_only=False)
    except Exception as error:
        native_abort_requested = _native_abort_for_impossible_callable_error_requested(error)
        _emit_data_runtime(
            "serialization/torch_load_failure",
            path=str(resolved_path),
            device=str(device),
            error_type=type(error).__name__,
            error_repr=repr(error),
            native_abort_requested=native_abort_requested,
        )
        if native_abort_requested:
            _abort_for_native_capture()
        raise
    _emit_data_runtime(
        "serialization/after_torch_load",
        path=str(resolved_path),
        device=str(device),
        payload_type=type(payload).__name__,
        payload_keys=sorted(payload),
        payload_value_types={key: type(value).__name__ for key, value in payload.items()},
    )
    teacher_actions = payload.get("teacher_actions")
    commands = payload.get("commands")
    target_height = payload.get("target_height")
    transition_ages = payload.get("transition_ages")
    command_before = payload.get("command_before")
    command_after = payload.get("command_after")
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
        target_height=None if target_height is None else target_height.to(device),
        command_intents=payload.get("command_intents"),
        scenario_labels=payload.get("scenario_labels"),
        transition_ages=(None if transition_ages is None else transition_ages.to(device)),
        command_before=(None if command_before is None else command_before.to(device)),
        command_after=(None if command_after is None else command_after.to(device)),
    )
    expected_count = payload.get("num_samples")
    if expected_count is not None and int(expected_count) != dataset.num_samples:
        raise ValueError(
            "distillation dataset num_samples mismatch: "
            f"payload={expected_count} tensors={dataset.num_samples}"
        )
    return dataset

