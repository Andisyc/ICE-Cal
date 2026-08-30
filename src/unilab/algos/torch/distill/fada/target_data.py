from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig

FADA_TARGET_ARTIFACT_SCHEMA_VERSION = "fada-target-batch/v2"
FADA_LEGACY_TARGET_ARTIFACT_SCHEMA_VERSION = "fada-target-batch/v1"
_REQUIRED_METADATA = {
    "policy_checkpoint_sha256",
    "config_fingerprint",
    "task",
    "fault_profile",
    "num_envs",
    "num_windows",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FADATargetBatch:
    """Oracle-free Stage-C histories and executed target futures."""

    observation_history: torch.Tensor
    action_history: torch.Tensor
    command: torch.Tensor
    realized_future: torch.Tensor
    executed_action_chunk: torch.Tensor
    episode_id: torch.Tensor
    start_timestep: torch.Tensor

    def validate(self, config: FADAArchitectureConfig) -> FADATargetBatch:
        _validate_sequence(
            "observation_history",
            self.observation_history,
            length=config.history_length,
            feature_dim=config.obs_dim,
        )
        _validate_sequence(
            "action_history",
            self.action_history,
            length=config.history_length,
            feature_dim=config.action_dim,
        )
        _validate_matrix("command", self.command, feature_dim=config.command_dim)
        _validate_sequence(
            "realized_future",
            self.realized_future,
            length=config.prediction_horizon,
            feature_dim=config.obs_dim,
        )
        _validate_sequence(
            "executed_action_chunk",
            self.executed_action_chunk,
            length=config.prediction_horizon,
            feature_dim=config.action_dim,
        )
        for name, tensor in {
            "episode_id": self.episode_id,
            "start_timestep": self.start_timestep,
        }.items():
            if tensor.ndim != 1 or tensor.dtype != torch.int64:
                raise ValueError(
                    f"{name} must be rank-1 torch.int64, got shape={tuple(tensor.shape)} "
                    f"dtype={tensor.dtype}"
                )
            if bool((tensor < 0).any()):
                raise ValueError(f"{name} must be non-negative")
        tensors = (
            self.observation_history,
            self.action_history,
            self.command,
            self.realized_future,
            self.executed_action_chunk,
            self.episode_id,
            self.start_timestep,
        )
        batch_sizes = {int(tensor.shape[0]) for tensor in tensors}
        if len(batch_sizes) != 1:
            raise ValueError(f"FADA target batch sizes must match, got {sorted(batch_sizes)}")
        if next(iter(batch_sizes)) <= 0:
            raise ValueError("FADA target batch must contain at least one row")
        return self


@dataclass(frozen=True)
class LoadedFADATargetArtifact:
    batch: FADATargetBatch
    metadata: Mapping[str, Any]


def _validate_sequence(name: str, tensor: torch.Tensor, *, length: int, feature_dim: int) -> None:
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (int(length), int(feature_dim)):
        raise ValueError(
            f"{name} shape mismatch: expected [batch, {length}, {feature_dim}], "
            f"got {tuple(tensor.shape)}"
        )
    _validate_floating(name, tensor)


def _validate_matrix(name: str, tensor: torch.Tensor, *, feature_dim: int) -> None:
    if tensor.ndim != 2 or int(tensor.shape[1]) != int(feature_dim):
        raise ValueError(
            f"{name} shape mismatch: expected [batch, {feature_dim}], got {tuple(tensor.shape)}"
        )
    _validate_floating(name, tensor)


def _validate_floating(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be torch.float32, got dtype={tensor.dtype}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validated_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_num_windows: int,
) -> dict[str, Any]:
    result = dict(metadata)
    missing = sorted(_REQUIRED_METADATA - set(result))
    if missing:
        raise ValueError(f"FADA target metadata missing required fields: {missing}")
    for key in ("policy_checkpoint_sha256", "config_fingerprint"):
        if not isinstance(result[key], str) or _SHA256.fullmatch(result[key]) is None:
            raise ValueError(f"FADA target metadata {key} must be a lowercase SHA-256 hex digest")
    for key in ("task", "fault_profile"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"FADA target metadata {key} must be a non-empty string")
    for key in ("num_envs", "num_windows"):
        if (
            isinstance(result[key], bool)
            or not isinstance(result[key], Integral)
            or result[key] <= 0
        ):
            raise ValueError(f"FADA target metadata {key} must be a positive integer")
        result[key] = int(result[key])
    if result["num_windows"] != expected_num_windows:
        raise ValueError(
            "FADA target metadata num_windows must equal target batch row count: "
            f"metadata={result['num_windows']} rows={expected_num_windows}"
        )
    return result


def _batch_to_cpu(batch: FADATargetBatch) -> FADATargetBatch:
    return FADATargetBatch(
        **{
            field: getattr(batch, field).detach().to("cpu").contiguous()
            for field in FADATargetBatch.__dataclass_fields__
        }
    )


def save_fada_target_artifact(
    path: str | Path,
    batch: FADATargetBatch,
    *,
    config: FADAArchitectureConfig,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically persist one strict, CPU-owned Stage-C artifact."""

    validated = _batch_to_cpu(batch.validate(config)).validate(config)
    validated_metadata = _validated_metadata(
        metadata,
        expected_num_windows=int(validated.observation_history.shape[0]),
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "schema_version": FADA_TARGET_ARTIFACT_SCHEMA_VERSION,
            "architecture": asdict(config),
            "batch": {
                field: getattr(validated, field) for field in FADATargetBatch.__dataclass_fields__
            },
            "metadata": validated_metadata,
        },
        temporary,
    )
    temporary.replace(target)
    return target


def load_fada_target_artifact(
    path: str | Path,
    *,
    config: FADAArchitectureConfig,
) -> LoadedFADATargetArtifact:
    """Strict-load one Stage-C artifact before any downstream consumption."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        FADA_LEGACY_TARGET_ARTIFACT_SCHEMA_VERSION,
        FADA_TARGET_ARTIFACT_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported or malformed FADA target artifact schema")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("FADA target artifact architecture must be a mapping")
    if (
        payload.get("schema_version") == FADA_TARGET_ARTIFACT_SCHEMA_VERSION
        and "observation_contract" not in architecture
    ):
        raise ValueError("FADA target artifact architecture must contain observation_contract")
    try:
        observed = FADAArchitectureConfig(**architecture)
    except (TypeError, ValueError) as exc:
        raise ValueError("FADA target artifact architecture is invalid") from exc
    if observed != config:
        raise ValueError(
            f"FADA target artifact architecture mismatch: expected={config} observed={observed}"
        )
    tensors = payload.get("batch")
    if not isinstance(tensors, dict) or set(tensors) != set(FADATargetBatch.__dataclass_fields__):
        raise ValueError("FADA target artifact tensor fields are incomplete")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("FADA target artifact metadata must be a mapping")
    batch = FADATargetBatch(**tensors).validate(config)
    return LoadedFADATargetArtifact(
        batch=batch,
        metadata=_validated_metadata(
            metadata,
            expected_num_windows=int(batch.observation_history.shape[0]),
        ),
    )
