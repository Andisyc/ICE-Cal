from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from unilab.algos.torch.distill.fada import FADAArchitectureConfig

CONTEXT_TRAJECTORY_DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ContextTrajectoryDataset:
    """Paired healthy-reference and fault-probe trajectories for Context training."""

    observation_history: torch.Tensor
    action_history: torch.Tensor
    command: torch.Tensor
    healthy_reference: torch.Tensor
    fault_state: torch.Tensor
    fault_action: torch.Tensor
    pair_id: torch.Tensor

    def validate(self, config: FADAArchitectureConfig) -> None:
        batch = int(self.observation_history.shape[0])
        expected = {
            "observation_history": (
                batch,
                config.history_length,
                config.obs_dim,
            ),
            "action_history": (
                batch,
                config.history_length,
                config.action_dim,
            ),
            "command": (batch, config.command_dim),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape mismatch: expected={shape} observed={tuple(value.shape)}")
        if self.healthy_reference.ndim != 3 or tuple(self.healthy_reference.shape[::2]) != (
            batch,
            config.obs_dim,
        ):
            raise ValueError("healthy_reference must have shape [batch, horizon, obs_dim]")
        if self.fault_state.ndim != 3 or tuple(self.fault_state.shape[::2]) != (
            batch,
            config.obs_dim,
        ):
            raise ValueError("fault_state must have shape [batch, steps+1, obs_dim]")
        if self.fault_action.shape != (
            batch,
            self.fault_state.shape[1] - 1,
            config.action_dim,
        ):
            raise ValueError("fault_action must align one-to-one with contiguous fault transitions")
        if self.pair_id.shape != (batch,) or self.pair_id.dtype != torch.int64:
            raise ValueError("pair_id must be int64 with shape [batch]")
        for name in (
            "observation_history",
            "action_history",
            "command",
            "healthy_reference",
            "fault_state",
            "fault_action",
        ):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} must contain only finite values")

    def fault_transition_batch(self):
        from unilab.algos.torch.fada_context.fault_dynamics import FaultTransitionBatch

        return FaultTransitionBatch(
            state=self.fault_state[:, :-1],
            action=self.fault_action,
            next_state=self.fault_state[:, 1:],
        )

    def to(self, device: str | torch.device) -> ContextTrajectoryDataset:
        return ContextTrajectoryDataset(
            observation_history=self.observation_history.to(device),
            action_history=self.action_history.to(device),
            command=self.command.to(device),
            healthy_reference=self.healthy_reference.to(device),
            fault_state=self.fault_state.to(device),
            fault_action=self.fault_action.to(device),
            pair_id=self.pair_id.to(device),
        )


def save_context_trajectory_dataset(
    path: str | Path,
    dataset: ContextTrajectoryDataset,
    config: FADAArchitectureConfig,
    *,
    metadata: Mapping[str, Any],
) -> None:
    dataset.validate(config)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CONTEXT_TRAJECTORY_DATASET_SCHEMA_VERSION,
        "architecture": asdict(config),
        "metadata": dict(metadata),
        "observation_history": dataset.observation_history.detach().cpu(),
        "action_history": dataset.action_history.detach().cpu(),
        "command": dataset.command.detach().cpu(),
        "healthy_reference": dataset.healthy_reference.detach().cpu(),
        "fault_state": dataset.fault_state.detach().cpu(),
        "fault_action": dataset.fault_action.detach().cpu(),
        "pair_id": dataset.pair_id.detach().cpu(),
    }
    torch.save(payload, target)


def load_context_trajectory_dataset(
    path: str | Path,
    config: FADAArchitectureConfig,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ContextTrajectoryDataset, Mapping[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Context trajectory dataset must be a mapping")
    if payload.get("schema_version") != CONTEXT_TRAJECTORY_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported Context trajectory dataset schema")
    if payload.get("architecture") != asdict(config):
        raise ValueError("Context trajectory dataset architecture mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Context trajectory dataset metadata must be a mapping")
    names = (
        "observation_history",
        "action_history",
        "command",
        "healthy_reference",
        "fault_state",
        "fault_action",
        "pair_id",
    )
    if any(not isinstance(payload.get(name), torch.Tensor) for name in names):
        raise ValueError("Context trajectory dataset is missing tensor fields")
    dataset = ContextTrajectoryDataset(**{name: payload[name] for name in names})
    dataset.validate(config)
    return dataset, metadata
