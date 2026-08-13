from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.support_query import (
    FADASupportContextEncoder,
    FrozenIDMSupportQueryPolicy,
    SupportQueryBatch,
    SupportQueryContextConfig,
    context_first_action_loss,
)

CONTEXT_SUPPORT_QUERY_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PreparedSupportQueryTraining:
    policy: FrozenIDMSupportQueryPolicy
    optimizer: torch.optim.Optimizer


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def prepare_support_query_training(
    healthy_policy: FADAPlannerIDMPolicy,
    context_config: SupportQueryContextConfig,
    *,
    learning_rate: float,
) -> PreparedSupportQueryTraining:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    device = next(healthy_policy.parameters()).device
    context = FADASupportContextEncoder(healthy_policy.config, context_config).to(device)
    policy = FrozenIDMSupportQueryPolicy(
        healthy_policy.planner,
        healthy_policy.idm,
        context,
    )
    optimizer = torch.optim.Adam(context.parameters(), lr=learning_rate)
    context_ids = {id(parameter) for parameter in context.parameters()}
    if _optimizer_parameter_ids(optimizer) != context_ids:
        raise RuntimeError("Context optimizer must own exactly Context Encoder parameters")
    if any(parameter.requires_grad for parameter in policy.planner.parameters()):
        raise RuntimeError("Planner must be frozen")
    if any(parameter.requires_grad for parameter in policy.idm.parameters()):
        raise RuntimeError("IDM must be frozen")
    return PreparedSupportQueryTraining(policy=policy, optimizer=optimizer)


@torch.no_grad()
def evaluate_context_action_mse(
    policy: FrozenIDMSupportQueryPolicy,
    batch: SupportQueryBatch,
) -> float:
    return float(context_first_action_loss(policy, batch).detach())


def save_context_support_query_checkpoint(
    path: str | Path,
    setup: PreparedSupportQueryTraining,
    *,
    source_checkpoint_sha256: str,
    dataset_sha256: str,
    train_split_sha256: str,
    validation_split_sha256: str,
    step: int,
    split_seed: int,
    metrics: Mapping[str, float],
    resolved_config: Mapping[str, Any],
) -> Path:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    identities = {
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "train_split_sha256": train_split_sha256,
        "validation_split_sha256": validation_split_sha256,
    }
    if any(not value for value in identities.values()):
        raise ValueError("checkpoint identity digests must be non-empty")
    if any(not torch.isfinite(torch.tensor(float(value))) for value in metrics.values()):
        raise ValueError("checkpoint metrics must be finite")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    torch.save(
        {
            "schema_version": CONTEXT_SUPPORT_QUERY_CHECKPOINT_SCHEMA_VERSION,
            "fada_architecture": asdict(setup.policy.config),
            "context_config": asdict(setup.policy.context_encoder.context_config),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "dataset_sha256": dataset_sha256,
            "train_split_sha256": train_split_sha256,
            "validation_split_sha256": validation_split_sha256,
            "step": int(step),
            "split_seed": int(split_seed),
            "metrics": {name: float(value) for name, value in metrics.items()},
            "context_state_dict": setup.policy.context_encoder.state_dict(),
            "optimizer_state_dict": setup.optimizer.state_dict(),
            "resolved_config": dict(resolved_config),
        },
        temporary,
    )
    temporary.replace(target)
    return target


def load_context_support_query_checkpoint(
    path: str | Path,
    setup: PreparedSupportQueryTraining,
    *,
    expected_source_checkpoint_sha256: str,
    expected_dataset_sha256: str,
    expected_train_split_sha256: str,
    expected_validation_split_sha256: str,
    load_optimizer: bool = False,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Context Support-Query checkpoint must be a mapping")
    if payload.get("schema_version") != CONTEXT_SUPPORT_QUERY_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported Context Support-Query checkpoint schema")
    if payload.get("fada_architecture") != asdict(setup.policy.config):
        raise ValueError("Context checkpoint FADA architecture mismatch")
    if payload.get("context_config") != asdict(
        setup.policy.context_encoder.context_config
    ):
        raise ValueError("Context checkpoint architecture mismatch")
    if payload.get("source_checkpoint_sha256") != expected_source_checkpoint_sha256:
        raise ValueError("Context checkpoint healthy source identity mismatch")
    expected_identities = {
        "dataset_sha256": expected_dataset_sha256,
        "train_split_sha256": expected_train_split_sha256,
        "validation_split_sha256": expected_validation_split_sha256,
    }
    for name, expected in expected_identities.items():
        if payload.get(name) != expected:
            raise ValueError(f"Context checkpoint {name} mismatch")
    state = payload.get("context_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Context checkpoint is missing context_state_dict")
    if load_optimizer:
        optimizer_state = payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("Context checkpoint is missing optimizer_state_dict")
    else:
        optimizer_state = None
    context_before = copy.deepcopy(setup.policy.context_encoder.state_dict())
    optimizer_before = copy.deepcopy(setup.optimizer.state_dict())
    try:
        setup.policy.context_encoder.load_state_dict(state, strict=True)
        if optimizer_state is not None:
            setup.optimizer.load_state_dict(optimizer_state)
    except Exception:
        setup.policy.context_encoder.load_state_dict(context_before, strict=True)
        setup.optimizer.load_state_dict(optimizer_before)
        raise
    return payload
