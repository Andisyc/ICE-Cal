from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from unilab.algos.torch.distill.fada import FADAArchitectureConfig
from unilab.algos.torch.fada_context.support_query import (
    ContextQueryBatch,
    SupportContextBatch,
    SupportQueryBatch,
)

SUPPORT_QUERY_DATASET_SCHEMA_VERSION = 2


def support_query_split_identity_sha256(batch: SupportQueryBatch) -> str:
    """Return an order-independent digest of exact pair and rollout membership."""

    identity = (
        torch.stack((batch.pair_id, batch.support_rollout_id, batch.query_rollout_id), dim=1)
        .detach()
        .cpu()
    )
    identity = identity.index_select(0, torch.argsort(identity[:, 0])).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(identity.shape)).encode("ascii"))
    digest.update(identity.numpy().tobytes())
    return digest.hexdigest()


def split_support_query_by_rollout(
    batch: SupportQueryBatch,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[SupportQueryBatch, SupportQueryBatch]:
    """Split complete reset/rollout groups without train-validation overlap."""

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    rollout_groups = (
        torch.stack((batch.support_rollout_id, batch.query_rollout_id), dim=1).detach().cpu()
    )
    unique_groups, inverse = torch.unique(rollout_groups, dim=0, return_inverse=True)
    group_count = int(unique_groups.shape[0])
    if group_count < 2:
        raise ValueError("Support-Query split requires at least two independent rollout groups")
    validation_groups = max(1, int(round(group_count * validation_fraction)))
    if validation_groups >= group_count:
        raise ValueError("validation split must leave at least one training rollout group")
    order = torch.randperm(group_count, generator=torch.Generator().manual_seed(seed))
    validation_group_ids = order[:validation_groups]
    validation_mask = torch.isin(inverse, validation_group_ids)
    train_indices = (
        torch.nonzero(~validation_mask, as_tuple=False).flatten().to(batch.pair_id.device)
    )
    validation_indices = (
        torch.nonzero(validation_mask, as_tuple=False).flatten().to(batch.pair_id.device)
    )
    return batch.index_select(train_indices), batch.index_select(validation_indices)


def _tensor_payload(batch: SupportQueryBatch) -> dict[str, torch.Tensor]:
    return {
        "support_target_future": batch.support.target_future.detach().cpu(),
        "support_realized_state": batch.support.realized_state.detach().cpu(),
        "support_executed_action": batch.support.executed_action.detach().cpu(),
        "query_observation_history": batch.query.observation_history.detach().cpu(),
        "query_action_history": batch.query.action_history.detach().cpu(),
        "query_command": batch.query.command.detach().cpu(),
        "query_planner_intent": batch.query.planner_intent.detach().cpu(),
        "query_realized_future": batch.query.realized_future.detach().cpu(),
        "query_executed_action": batch.query.executed_action.detach().cpu(),
        "query_window_anchor": batch.query.window_anchor.detach().cpu(),
        "query_valid_window_mask": batch.query.valid_window_mask.detach().cpu(),
        "support_command": batch.support_command.detach().cpu(),
        "pair_id": batch.pair_id.detach().cpu(),
        "support_rollout_id": batch.support_rollout_id.detach().cpu(),
        "query_rollout_id": batch.query_rollout_id.detach().cpu(),
    }


def _batch_from_payload(
    payload: Mapping[str, Any],
    config: FADAArchitectureConfig,
) -> SupportQueryBatch:
    names = (
        "support_target_future",
        "support_realized_state",
        "support_executed_action",
        "query_observation_history",
        "query_action_history",
        "query_command",
        "query_planner_intent",
        "query_realized_future",
        "query_executed_action",
        "query_window_anchor",
        "query_valid_window_mask",
        "support_command",
        "pair_id",
        "support_rollout_id",
        "query_rollout_id",
    )
    missing = [name for name in names if not isinstance(payload.get(name), torch.Tensor)]
    if missing:
        raise ValueError(f"Support-Query dataset is missing tensor fields: {missing}")
    return SupportQueryBatch(
        support=SupportContextBatch(
            target_future=payload["support_target_future"],
            realized_state=payload["support_realized_state"],
            executed_action=payload["support_executed_action"],
        ),
        query=ContextQueryBatch(
            observation_history=payload["query_observation_history"],
            action_history=payload["query_action_history"],
            command=payload["query_command"],
            planner_intent=payload["query_planner_intent"],
            realized_future=payload["query_realized_future"],
            executed_action=payload["query_executed_action"],
            window_anchor=payload["query_window_anchor"],
            valid_window_mask=payload["query_valid_window_mask"],
        ),
        support_command=payload["support_command"],
        pair_id=payload["pair_id"],
        support_rollout_id=payload["support_rollout_id"],
        query_rollout_id=payload["query_rollout_id"],
    )


def save_support_query_dataset(
    path: str | Path,
    batch: SupportQueryBatch,
    config: FADAArchitectureConfig,
    *,
    support_length: int,
    query_length: int,
    metadata: Mapping[str, Any],
) -> Path:
    batch.validate(config, support_length=support_length)
    if torch.unique(batch.pair_id).numel() != batch.pair_id.numel():
        raise ValueError("sealed Support-Query dataset pair_id values must be unique")
    required_metadata = {
        "source_checkpoint_sha256",
        "task_config",
        "fault_joint",
        "fault_strength",
        "command",
        "seed",
    }
    missing = sorted(required_metadata - set(metadata))
    if missing:
        raise ValueError(f"Support-Query dataset metadata is missing: {missing}")
    payload: dict[str, Any] = {
        "schema_version": SUPPORT_QUERY_DATASET_SCHEMA_VERSION,
        "architecture": asdict(config),
        "support_length": int(support_length),
        "query_length": int(query_length),
        "metadata": dict(metadata),
        **_tensor_payload(batch),
    }
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_support_query_dataset(
    path: str | Path,
    config: FADAArchitectureConfig,
    *,
    support_length: int,
    query_length: int,
    map_location: str | torch.device = "cpu",
) -> tuple[SupportQueryBatch, Mapping[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Support-Query dataset must be a mapping")
    schema = int(payload.get("schema_version", -1))
    if schema != SUPPORT_QUERY_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported Support-Query dataset schema")
    if payload.get("architecture") != asdict(config):
        raise ValueError("Support-Query dataset architecture mismatch")
    if payload.get("support_length") != int(support_length):
        raise ValueError("Support-Query dataset support length mismatch")
    if payload.get("query_length") != int(query_length):
        raise ValueError("Support-Query dataset Query length mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Support-Query dataset metadata must be a mapping")
    batch = _batch_from_payload(payload, config).validate(config, support_length=support_length)
    if torch.unique(batch.pair_id).numel() != batch.pair_id.numel():
        raise ValueError("sealed Support-Query dataset pair_id values must be unique")
    return batch, metadata
