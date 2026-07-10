from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

import torch
from torch import nn


def save_distillation_checkpoint(
    path: str | Path,
    *,
    student: nn.Module,
    agent_steps: int,
    teacher_metadata: Mapping[str, Any] | None = None,
    distill_runtime_cfg: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    obs_normalizer: nn.Module | None = None,
) -> None:
    """Save the deployable student and distillation provenance."""

    payload: dict[str, Any] = {
        "student_state_dict": student.state_dict(),
        "agent_steps": int(agent_steps),
        "teacher_metadata": dict(teacher_metadata or {}),
        "distill_runtime_cfg": dict(distill_runtime_cfg or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if obs_normalizer is not None:
        payload["obs_normalizer"] = obs_normalizer.state_dict()
    torch.save(payload, Path(path))


def load_distillation_checkpoint(
    student: nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    obs_normalizer: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a student-only distillation checkpoint."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    student_state = checkpoint.get("student_state_dict")
    if student_state is None:
        raise ValueError(f"Checkpoint does not contain student_state_dict: {path}")
    student.load_state_dict(student_state, strict=True)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    normalizer_state = checkpoint.get("obs_normalizer")
    if obs_normalizer is not None and normalizer_state is not None:
        obs_normalizer.load_state_dict(normalizer_state)

    return cast(dict[str, Any], checkpoint)
