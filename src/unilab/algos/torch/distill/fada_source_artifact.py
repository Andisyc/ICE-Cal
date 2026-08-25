"""FADA causal-window artifact persistence owner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .fada import FADAArchitectureConfig, FADASourceBatch

FADA_SOURCE_BATCH_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class LoadedFADASourceBatch:
    """One validated collector artifact and its iteration metadata."""

    batch: FADASourceBatch
    metadata: Mapping[str, Any]


def _load_architecture_config(
    architecture: Any,
    *,
    schema_version: Any,
    contract_schema_version: Any,
    context: str,
) -> FADAArchitectureConfig:
    """Normalize a legacy architecture while requiring identity in current schemas."""

    if not isinstance(architecture, dict):
        raise ValueError(f"{context} architecture must be a mapping")
    if schema_version == contract_schema_version and "observation_contract" not in architecture:
        raise ValueError(f"{context} architecture must contain observation_contract")
    try:
        return FADAArchitectureConfig(**architecture)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {context} architecture: {architecture}") from exc


def _batch_to_device(batch: FADASourceBatch, device: torch.device) -> FADASourceBatch:
    return FADASourceBatch(
        observation_history=batch.observation_history.to(device),
        action_history=batch.action_history.to(device),
        command=batch.command.to(device),
        realized_future=batch.realized_future.to(device),
        executed_action_chunk=batch.executed_action_chunk.to(device),
        oracle_future=batch.oracle_future.to(device),
        oracle_action_chunk=batch.oracle_action_chunk.to(device),
        oracle_shadow_valid=batch.oracle_shadow_valid.to(device),
        idm_source_role=batch.idm_source_role.to(device),
        oracle_first_action=batch.oracle_first_action.to(device),
        command_scenario=batch.command_scenario.to(device),
        planner_eligible=batch.planner_eligible.to(device),
        cold_start=batch.cold_start.to(device),
    )


load_architecture_config = _load_architecture_config
batch_to_device = _batch_to_device


def save_fada_source_batch(
    path: str | Path,
    batch: FADASourceBatch,
    *,
    config: FADAArchitectureConfig,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically persist one CPU causal-window artifact from the collector process."""

    validated = _batch_to_device(batch.validate(config), torch.device("cpu"))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "schema_version": FADA_SOURCE_BATCH_SCHEMA_VERSION,
            "architecture": asdict(config),
            "batch": {
                field: getattr(validated, field) for field in FADASourceBatch.__dataclass_fields__
            },
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(target)
    return target


def load_fada_source_batch(
    path: str | Path,
    *,
    config: FADAArchitectureConfig,
) -> LoadedFADASourceBatch:
    """Load and validate one collector artifact before it enters learner replay."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != FADA_SOURCE_BATCH_SCHEMA_VERSION:
        raise ValueError("unsupported or malformed FADA source batch schema")
    observed = _load_architecture_config(
        payload.get("architecture"),
        schema_version=payload.get("schema_version"),
        contract_schema_version=FADA_SOURCE_BATCH_SCHEMA_VERSION,
        context="FADA source batch",
    )
    if observed != config:
        raise ValueError(
            f"FADA source batch architecture mismatch: expected={config} observed={observed}"
        )
    tensors = payload.get("batch")
    if not isinstance(tensors, dict) or set(tensors) != set(FADASourceBatch.__dataclass_fields__):
        raise ValueError("FADA source batch tensor fields are incomplete")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("FADA source batch metadata must be a mapping")
    batch = FADASourceBatch(**tensors).validate(config)
    return LoadedFADASourceBatch(batch=batch, metadata=metadata)
