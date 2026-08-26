"""Cold-path checkpoint Gateway for the single frozen FADA Oracle."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .playback import load_distillation_student_policy
from .teacher import DistillationTeacherSpec, load_sac_teacher_policy


def _is_distillation_student_checkpoint(payload: object) -> bool:
    return isinstance(payload, dict) and {
        "student_state_dict",
        "distill_runtime_cfg",
    }.issubset(payload)


def load_fada_oracle_policy(
    checkpoint_path: str | Path,
    spec: DistillationTeacherSpec,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load one SAC or distillation policy behind the FADA tensor-policy contract."""

    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if _is_distillation_student_checkpoint(payload):
        loaded = load_distillation_student_policy(path, device=device)
        if loaded.obs_dim != int(spec.obs_dim):
            raise ValueError(
                "FADA Oracle obs dim mismatch: "
                f"checkpoint={loaded.obs_dim}, configured={int(spec.obs_dim)}"
            )
        if loaded.action_dim != int(spec.action_dim):
            raise ValueError(
                "FADA Oracle action dim mismatch: "
                f"checkpoint={loaded.action_dim}, configured={int(spec.action_dim)}"
            )
        policy = loaded.policy
    else:
        policy = load_sac_teacher_policy(path, spec, device=device)

    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


__all__ = ["load_fada_oracle_policy"]
