from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from .fada_adaptation import (
    FADALoRAConfig,
    assert_fada_adaptation_parameter_ownership,
    fada_adapter_named_parameters,
    inject_fada_idm_lora,
)
from .fada_checkpoint import (
    FADA_CHECKPOINT_SCHEMA_VERSION,
    LoadedFADAPlannerIDMPolicy,
    load_fada_policy_checkpoint,
)

FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION = "fada-adapted/v2"
FADA_LEGACY_ADAPTED_CHECKPOINT_SCHEMA_VERSION = "fada-adapted/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def assert_fada_adaptation_source_checkpoint(
    loaded: LoadedFADAPlannerIDMPolicy,
) -> LoadedFADAPlannerIDMPolicy:
    """Keep the active Stage-C/D adaptation contract pinned to source schema 3."""

    checkpoint = getattr(loaded, "checkpoint", None)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema_version") != 3:
        raise ValueError("FADA adaptation v002 requires schema-3 source checkpoint")
    return loaded


def _validate_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _lora_payload(config: FADALoRAConfig) -> dict[str, Any]:
    return {
        "rank": int(config.rank),
        "alpha": float(config.alpha),
        "dropout": float(config.dropout),
        "target_modules": list(config.target_modules),
    }


def save_fada_adapted_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    lora_config: FADALoRAConfig,
    source_checkpoint_sha256: str,
    target_artifact_sha256: str,
    completed_steps: int,
    samples_seen: int,
    runtime_config: Mapping[str, Any],
) -> Path:
    """Atomically persist a self-contained frozen-base plus adapter checkpoint."""

    assert_fada_adaptation_parameter_ownership(policy, lora_config)
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
    ):
        raise ValueError("completed_steps must be a non-negative integer")
    if isinstance(samples_seen, bool) or not isinstance(samples_seen, int) or samples_seen < 0:
        raise ValueError("samples_seen must be a non-negative integer")
    adapter_ids = {id(parameter) for _name, parameter in fada_adapter_named_parameters(policy)}
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != adapter_ids:
        raise ValueError(
            "adapted checkpoint optimizer must own only adapter parameters exactly once"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "schema_version": FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
        "architecture": asdict(policy.config),
        "lora_config": _lora_payload(lora_config),
        "policy_state_dict": policy.state_dict(),
        "trainable_parameter_names": list(dict(fada_adapter_named_parameters(policy))),
        "optimizer_state_dict": optimizer.state_dict(),
        "source_checkpoint_sha256": _validate_sha256(
            "source_checkpoint_sha256", source_checkpoint_sha256
        ),
        "target_artifact_sha256": _validate_sha256(
            "target_artifact_sha256", target_artifact_sha256
        ),
        "completed_steps": completed_steps,
        "samples_seen": samples_seen,
        "runtime_config": dict(runtime_config),
    }
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def _parse_lora_config(payload: Any) -> FADALoRAConfig:
    if not isinstance(payload, dict) or set(payload) != {
        "rank",
        "alpha",
        "dropout",
        "target_modules",
    }:
        raise ValueError("adapted checkpoint LoRA config or manifest is malformed")
    targets = payload.get("target_modules")
    if not isinstance(targets, list) or not all(isinstance(name, str) for name in targets):
        raise ValueError("adapted checkpoint LoRA target manifest is malformed")
    return FADALoRAConfig(
        rank=payload["rank"],
        alpha=payload["alpha"],
        dropout=payload["dropout"],
        target_modules=tuple(targets),
    )


def load_fada_adapted_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedFADAPlannerIDMPolicy:
    """Fresh-construct and strict-load one self-contained adapted policy."""

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        FADA_LEGACY_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
        FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported or malformed FADA adapted checkpoint schema")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("adapted checkpoint architecture must be a mapping")
    if (
        payload.get("schema_version") == FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION
        and "observation_contract" not in architecture
    ):
        raise ValueError("adapted checkpoint architecture must contain observation_contract")
    try:
        config = FADAArchitectureConfig(**architecture)
    except (TypeError, ValueError) as exc:
        raise ValueError("adapted checkpoint architecture is invalid") from exc
    lora_config = _parse_lora_config(payload.get("lora_config"))
    adapted = inject_fada_idm_lora(FADAPlannerIDMPolicy(config).to(device), lora_config)
    expected_trainable = list(dict(fada_adapter_named_parameters(adapted.policy)))
    if payload.get("trainable_parameter_names") != expected_trainable:
        raise ValueError("adapted checkpoint trainable parameter manifest is incompatible")
    state = payload.get("policy_state_dict")
    if not isinstance(state, dict):
        raise ValueError("adapted checkpoint policy state must be a mapping")
    try:
        adapted.policy.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("adapted checkpoint architecture or policy state is incompatible") from exc
    _validate_sha256("source_checkpoint_sha256", payload.get("source_checkpoint_sha256"))
    _validate_sha256("target_artifact_sha256", payload.get("target_artifact_sha256"))
    for name in ("completed_steps", "samples_seen"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"adapted checkpoint {name} must be a non-negative integer")
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        raise ValueError("adapted checkpoint optimizer state must be a mapping")
    if not isinstance(payload.get("runtime_config"), dict):
        raise ValueError("adapted checkpoint runtime_config must be a mapping")
    assert_fada_adaptation_parameter_ownership(adapted.policy, adapted.lora_config)
    adapted.policy.eval()
    return LoadedFADAPlannerIDMPolicy(policy=adapted.policy, checkpoint=payload)


def load_fada_deployable_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedFADAPlannerIDMPolicy:
    """Dispatch supported source and adapted schemas for the official playback consumer."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("unsupported or malformed FADA deployable checkpoint schema")
    schema = payload.get("schema_version")
    if schema in {1, 2, 3, FADA_CHECKPOINT_SCHEMA_VERSION}:
        return load_fada_policy_checkpoint(path, device=device)
    if schema in {
        FADA_LEGACY_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
        FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
    }:
        return load_fada_adapted_checkpoint(path, device=device)
    raise ValueError(f"unsupported FADA deployable checkpoint schema: {schema!r}")
