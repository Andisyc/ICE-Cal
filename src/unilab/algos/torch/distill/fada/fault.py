"""Configuration-owned FADA target fault identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class FADAFaultSpec:
    name: str
    task: str
    task_name: str
    backend: str
    fault_profile: str
    command_limit: list[list[float]]
    actuator_index: int
    actuator_strength: float
    actuator_count: int


def resolve_fada_fault(cfg: DictConfig) -> FADAFaultSpec:
    raw = OmegaConf.to_container(OmegaConf.select(cfg, "fault"), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("FADA fault config must be a mapping")
    try:
        return FADAFaultSpec(
            name=str(raw["name"]), task=str(raw["task"]), task_name=str(raw["task_name"]),
            backend=str(raw["backend"]), fault_profile=str(raw["fault_profile"]),
            command_limit=[[float(v) for v in row] for row in raw["command_limit"]],
            actuator_index=int(raw["actuator_index"]),
            actuator_strength=float(raw["actuator_strength"]),
            actuator_count=int(raw["actuator_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("incomplete FADA fault config") from exc
