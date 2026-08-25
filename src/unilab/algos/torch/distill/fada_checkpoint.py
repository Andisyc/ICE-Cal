"""Paired FADA policy and optimizer checkpoint owner."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .fada import FADAPlannerIDMPolicy
from .fada_source_artifact import load_architecture_config
from .fada_trainer import FADATrainer
from .fada_training_phase import (
    FADATrainingPhase,
    canonical_module_sha256,
    canonical_state_dict_sha256,
)

FADA_CHECKPOINT_SCHEMA_VERSION = 4


FADA_V005_REQUIRED_QUALITY_METRICS = (
    "scenario/walk/planner_idm_oracle_action_mse",
    "scenario/walk/cold_start_fraction",
    "scenario/walk/cold_start_planner_mse",
    "scenario/walk/steady_state_planner_mse",
    "scenario/static_stand/planner_idm_oracle_action_mse",
    "scenario/walk_to_stand/planner_idm_oracle_action_mse",
    "scenario/static_stand/cold_start_fraction",
    "scenario/static_stand/cold_start_planner_mse",
    "scenario/static_stand/steady_state_planner_mse",
)


@dataclass(frozen=True)
class LoadedFADAPlannerIDMPolicy:
    """Inference-ready FADA policy reconstructed from one strict checkpoint."""

    policy: FADAPlannerIDMPolicy
    checkpoint: Mapping[str, Any]


def _validated_schema4_idm_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("idm_state_dict")
    if not isinstance(state, dict) or payload.get("idm_sha256") != canonical_state_dict_sha256(
        state
    ):
        raise ValueError("schema-4 FADA checkpoint IDM identity mismatch")
    return state


def save_fada_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer,
    *,
    completed_iterations: int,
    samples_seen: int,
    runtime_config: Mapping[str, Any],
    phase_completed: bool,
    quality_metrics: Mapping[str, float] | None = None,
) -> Path:
    """Atomically persist the paired FADA module and optimizer identity."""

    metrics = dict(quality_metrics or {})
    v005_enabled = bool(dict(runtime_config.get("v005_replay") or {}).get("enabled", False))
    if (
        trainer.phase is FADATrainingPhase.PLANNER
        and v005_enabled
        and int(completed_iterations) > 0
    ):
        missing = [name for name in FADA_V005_REQUIRED_QUALITY_METRICS if name not in metrics]
        if missing:
            raise ValueError(f"v005 FADA checkpoint quality metrics are missing: {missing}")
        invalid = [
            name
            for name, value in metrics.items()
            if not isinstance(value, (int, float)) or not math.isfinite(float(value))
        ]
        if invalid:
            raise ValueError(f"v005 FADA checkpoint quality metrics are non-finite: {invalid}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    trainer.assert_phase_integrity()
    idm_sha256 = canonical_module_sha256(policy.idm)
    payload = {
        "schema_version": FADA_CHECKPOINT_SCHEMA_VERSION,
        "architecture": asdict(policy.config),
        "planner_state_dict": policy.planner.state_dict(),
        "idm_state_dict": policy.idm.state_dict(),
        "training_phase": trainer.phase.value,
        "phase_completed": bool(phase_completed),
        "optimizer_owner": trainer.phase.optimizer_owner,
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "idm_sha256": idm_sha256,
        "pretrained_idm_sha256": trainer.pretrained_idm_sha256,
        "completed_iterations": int(completed_iterations),
        "samples_seen": int(samples_seen),
        "runtime_config": dict(runtime_config),
        "quality_metrics": metrics,
    }
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_fada_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore one exact FADA architecture and optionally its paired optimizers."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2, 3, 4}:
        raise ValueError("unsupported or malformed FADA checkpoint schema")
    if trainer is not None:
        raise ValueError(
            "v010 training resume is disabled; load without a trainer for state inspection"
        )
    observed = load_architecture_config(
        payload.get("architecture"),
        schema_version=payload.get("schema_version"),
        contract_schema_version=FADA_CHECKPOINT_SCHEMA_VERSION,
        context="FADA checkpoint",
    )
    if observed != policy.config:
        raise ValueError(
            f"FADA checkpoint architecture mismatch: expected={policy.config} observed={observed}"
        )
    if payload.get("schema_version") == FADA_CHECKPOINT_SCHEMA_VERSION:
        _validated_schema4_idm_state(payload)
    policy.planner.load_state_dict(payload["planner_state_dict"], strict=True)
    policy.idm.load_state_dict(payload["idm_state_dict"], strict=True)
    return payload


def load_fada_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedFADAPlannerIDMPolicy:
    """Construct an inference-only Planner-IDM policy from checkpoint-owned architecture."""

    # B1: 只读取 tensor 与基础类型, 并在模型构造前关闭 schema/architecture 不匹配.
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2, 3, 4}:
        raise ValueError("unsupported or malformed FADA checkpoint schema")
    config = load_architecture_config(
        payload.get("architecture"),
        schema_version=payload.get("schema_version"),
        contract_schema_version=FADA_CHECKPOINT_SCHEMA_VERSION,
        context="FADA checkpoint",
    )
    if payload.get("schema_version") == FADA_CHECKPOINT_SCHEMA_VERSION:
        _validated_schema4_idm_state(payload)

    # B2: 严格恢复 Planner 与 IDM, 产出 eval 模式的唯一 playback policy owner.
    policy = FADAPlannerIDMPolicy(config).to(device)
    policy.planner.load_state_dict(payload["planner_state_dict"], strict=True)
    policy.idm.load_state_dict(payload["idm_state_dict"], strict=True)
    policy.eval()
    return LoadedFADAPlannerIDMPolicy(policy=policy, checkpoint=payload)


def load_pretrained_idm_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    *,
    map_location: str | torch.device = "cpu",
) -> str:
    """Strictly admit a completed schema-4 IDM checkpoint and load IDM weights only."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 4:
        raise ValueError("Planner phase requires a schema-4 completed IDM-pretrain checkpoint")
    if payload.get("training_phase") != FADATrainingPhase.IDM_PRETRAIN.value or not bool(
        payload.get("phase_completed", False)
    ):
        raise ValueError("Planner phase requires a completed IDM-pretrain checkpoint")
    if payload.get("optimizer_owner") != "idm" or "optimizer_state_dict" not in payload:
        raise ValueError("completed IDM-pretrain checkpoint has invalid optimizer ownership")
    if any(name in payload for name in ("planner_optimizer_state_dict", "idm_optimizer_state_dict")):
        raise ValueError("schema-4 FADA checkpoint must contain exactly one optimizer state")
    observed = load_architecture_config(
        payload.get("architecture"),
        schema_version=payload.get("schema_version"),
        contract_schema_version=FADA_CHECKPOINT_SCHEMA_VERSION,
        context="FADA checkpoint",
    )
    if observed != policy.config:
        raise ValueError(
            f"pretrained IDM architecture mismatch: expected={policy.config} observed={observed}"
        )
    state = _validated_schema4_idm_state(payload)
    observed_sha256 = canonical_state_dict_sha256(state)
    policy.idm.load_state_dict(state, strict=True)
    return observed_sha256
