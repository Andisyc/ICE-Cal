from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada import FADAArchitectureConfig
from unilab.algos.torch.fada_context.calibration import (
    CALIBRATION_AXIS_CATALOG_VERSION,
    CALIBRATION_AXIS_NAMES,
    CALIBRATION_METHOD_CONTRACT_ID,
    CALIBRATION_TRAINING_CONTRACT_ID,
    CalibrationRolloutBatch,
    FaultAxis,
    FaultAxisCatalog,
)

CALIBRATION_DATASET_SCHEMA = "unilab_fada_calibration_dataset_v1"


def load_fault_axis_catalog(path: str | Path) -> FaultAxisCatalog:
    payload = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("axes"), list):
        raise ValueError("fault-axis catalog must contain an axes list")
    version = payload.get("catalog_version")
    if not isinstance(version, str) or not version:
        raise ValueError("fault-axis catalog version is missing")
    axes = []
    for raw in payload["axes"]:
        if not isinstance(raw, dict):
            raise ValueError("fault-axis catalog entries must be mappings")
        normalized_range = raw.get("normalized_range")
        if not isinstance(normalized_range, list) or len(normalized_range) != 2:
            raise ValueError("fault-axis normalized_range must contain two values")
        axes.append(
            FaultAxis(
                name=str(raw.get("name", "")),
                normalized_range=(float(normalized_range[0]), float(normalized_range[1])),
                units=str(raw.get("units", "")),
                injection=str(raw.get("injection", "")),
            )
        )
    catalog = FaultAxisCatalog(tuple(axes), version=version)
    if catalog.names != tuple(payload.get("axis_order", catalog.names)):
        raise ValueError("fault-axis catalog order does not match axis_order")
    return catalog


def prepare_calibration_rollout_batch(
    raw: Mapping[str, Any],
    config: FADAArchitectureConfig,
    catalog: FaultAxisCatalog,
) -> CalibrationRolloutBatch:
    tensor_names = (
        "observation_history",
        "action_history",
        "command",
        "nominal_action_chunk",
        "c_true",
        "is_held_out_combination",
        "injected_strength",
        "planner_intent",
        "rollout_id",
        "seed",
        "split_id",
    )
    missing = [name for name in tensor_names if not isinstance(raw.get(name), torch.Tensor)]
    if missing:
        raise ValueError(f"raw calibration rollout is missing tensor fields: {missing}")
    axis_names = raw.get("axis_name")
    batch_size = int(raw["observation_history"].shape[0])
    if not isinstance(axis_names, (list, tuple)) or len(axis_names) != batch_size:
        raise ValueError("raw calibration rollout axis_name must bind every row")
    held_out = raw["is_held_out_combination"]
    if held_out.shape != (batch_size,) or held_out.dtype != torch.bool:
        raise ValueError("raw calibration rollout combination role must be rank-1 bool")
    supplied_targets = raw.get("target_action_chunk")
    if bool(held_out.any()) and not isinstance(supplied_targets, torch.Tensor):
        raise ValueError("held-out combination rows require explicit analytic targets")
    target = torch.empty_like(raw["nominal_action_chunk"])
    axis_id = torch.full((batch_size,), -1, dtype=torch.int64)
    for row in range(batch_size):
        if bool(held_out[row]):
            assert isinstance(supplied_targets, torch.Tensor)
            target[row] = supplied_targets[row]
            continue
        axis_name = str(axis_names[row])
        axis_id[row] = catalog.index(axis_name)
        target[row : row + 1] = catalog.analytic_target(
            axis_name,
            raw["nominal_action_chunk"][row : row + 1],
            float(raw["injected_strength"][row]),
        )
    return CalibrationRolloutBatch(
        observation_history=raw["observation_history"],
        action_history=raw["action_history"],
        command=raw["command"],
        nominal_action_chunk=raw["nominal_action_chunk"],
        target_action_chunk=target,
        c_true=raw["c_true"],
        axis_id=axis_id,
        is_held_out_combination=held_out,
        injected_strength=raw["injected_strength"],
        planner_intent=raw["planner_intent"],
        rollout_id=raw["rollout_id"],
        seed=raw["seed"],
        split_id=raw["split_id"],
    ).validate(config, axis_count=len(catalog.axes))


def calibration_split_identity_sha256(batch: CalibrationRolloutBatch) -> str:
    identity = {
        "rollout_id": batch.rollout_id.tolist(),
        "seed": batch.seed.tolist(),
        "split_id": batch.split_id.tolist(),
        "axis_id": batch.axis_id.tolist(),
        "is_held_out_combination": batch.is_held_out_combination.tolist(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fields(batch: CalibrationRolloutBatch) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in asdict(batch).items()}


def save_calibration_dataset(
    path: str | Path,
    batch: CalibrationRolloutBatch,
    config: FADAArchitectureConfig,
    *,
    metadata: Mapping[str, Any],
) -> Path:
    batch.validate(config, axis_count=batch.c_true.shape[-1])
    if batch.c_true.shape[-1] != len(CALIBRATION_AXIS_NAMES):
        raise ValueError("calibration dataset axis count does not match the active catalog")
    required = {"source_tracker_sha256", "axis_catalog_version", "split_identity_sha256"}
    missing = sorted(
        name for name in required if not isinstance(metadata.get(name), str) or not metadata[name]
    )
    if missing:
        raise ValueError(f"calibration dataset metadata is missing: {missing}")
    if metadata["axis_catalog_version"] != CALIBRATION_AXIS_CATALOG_VERSION:
        raise ValueError("calibration dataset axis catalog mismatch")
    if metadata["split_identity_sha256"] != calibration_split_identity_sha256(batch):
        raise ValueError("calibration dataset split identity does not match its rows")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CALIBRATION_DATASET_SCHEMA,
        "method_contract_id": CALIBRATION_METHOD_CONTRACT_ID,
        "training_contract_id": CALIBRATION_TRAINING_CONTRACT_ID,
        "architecture": asdict(config),
        "axis_count": int(batch.c_true.shape[-1]),
        "axis_names": CALIBRATION_AXIS_NAMES,
        "metadata": dict(metadata),
        **_fields(batch),
    }
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_calibration_dataset(
    path: str | Path,
    config: FADAArchitectureConfig,
) -> tuple[CalibrationRolloutBatch, Mapping[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CALIBRATION_DATASET_SCHEMA:
        raise ValueError("unsupported calibration dataset schema")
    if payload.get("method_contract_id") != CALIBRATION_METHOD_CONTRACT_ID:
        raise ValueError("calibration dataset method Contract mismatch")
    if payload.get("training_contract_id") != CALIBRATION_TRAINING_CONTRACT_ID:
        raise ValueError("calibration dataset training Contract mismatch")
    if payload.get("architecture") != asdict(config):
        raise ValueError("calibration dataset architecture mismatch")
    if (
        payload.get("axis_count") != len(CALIBRATION_AXIS_NAMES)
        or tuple(payload.get("axis_names", ())) != CALIBRATION_AXIS_NAMES
    ):
        raise ValueError("calibration dataset axis catalog identity mismatch")
    names = tuple(CalibrationRolloutBatch.__dataclass_fields__)
    missing = [name for name in names if not isinstance(payload.get(name), torch.Tensor)]
    if missing:
        raise ValueError(f"calibration dataset missing tensor fields: {missing}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("calibration dataset metadata must be a mapping")
    if metadata.get("axis_catalog_version") != CALIBRATION_AXIS_CATALOG_VERSION:
        raise ValueError("calibration dataset axis catalog mismatch")
    if (
        not isinstance(metadata.get("source_tracker_sha256"), str)
        or not metadata["source_tracker_sha256"]
    ):
        raise ValueError("calibration dataset source Tracker identity is missing")
    if (
        not isinstance(metadata.get("split_identity_sha256"), str)
        or not metadata["split_identity_sha256"]
    ):
        raise ValueError("calibration dataset split identity is missing")
    batch = CalibrationRolloutBatch(**{name: payload[name] for name in names})
    batch.validate(config, axis_count=int(payload.get("axis_count", -1)))
    if metadata["split_identity_sha256"] != calibration_split_identity_sha256(batch):
        raise ValueError("calibration dataset split identity does not match its rows")
    return batch, metadata
