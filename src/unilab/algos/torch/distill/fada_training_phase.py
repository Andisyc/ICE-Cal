"""FADA two-stage training phase and sealed IDM identity owner."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Mapping

import torch


class FADATrainingPhase(str, Enum):
    IDM_PRETRAIN = "idm_pretrain"
    PLANNER = "planner"

    @property
    def collect_intermediate_oracles(self) -> bool:
        return self is FADATrainingPhase.IDM_PRETRAIN

    @property
    def optimizer_owner(self) -> str:
        return "idm" if self is FADATrainingPhase.IDM_PRETRAIN else "planner"

    def main_rollout_uses_student(self, *, iteration: int) -> bool:
        if int(iteration) < 0:
            raise ValueError("FADA iteration must be non-negative")
        return self is FADATrainingPhase.PLANNER and int(iteration) > 0


def parse_fada_training_phase(value: object) -> FADATrainingPhase:
    try:
        return FADATrainingPhase(str(value))
    except ValueError as exc:
        allowed = ", ".join(phase.value for phase in FADATrainingPhase)
        raise ValueError(f"training.fada.phase must be one of [{allowed}], got {value!r}") from exc


def canonical_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor identity independent of mapping order, device, and torch serialization."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict[{name!r}] must be a tensor")
        value = tensor.detach().to(device="cpu").contiguous()
        identity = f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode("ascii")
        digest.update(len(identity).to_bytes(8, "big"))
        digest.update(identity)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def canonical_module_sha256(module: torch.nn.Module) -> str:
    return canonical_state_dict_sha256(module.state_dict())


__all__ = [
    "FADATrainingPhase",
    "canonical_module_sha256",
    "canonical_state_dict_sha256",
    "parse_fada_training_phase",
]
