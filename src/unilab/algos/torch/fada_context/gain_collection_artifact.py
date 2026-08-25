from __future__ import annotations

import hashlib
import io
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.distill.fada import FADAArchitectureConfig
from unilab.algos.torch.fada_context.calibration_types import (
    CALIBRATION_METHOD_CONTRACT_ID,
    CALIBRATION_TRAINING_CONTRACT_ID,
    CalibrationAxisSpec,
    FaultAxisCatalog,
)
from unilab.algos.torch.fada_context.gain_collection_provenance import (
    _canonical_json_value,
    _protocol_from_bytes,
    _validate_resolved_task_backend_payload,
    sha256_canonical_mapping,
)
from unilab.algos.torch.fada_context.gain_collection_types import (
    _HEX_DIGITS,
    _LEGACY_AXIS_CATALOG_VERSION,
    _LEGACY_AXIS_NAMES,
    _LEGACY_GAIN_CALIBRATION_RAW_SCHEMA,
    _LEGACY_METHOD_CONTRACT_ID,
    _LEGACY_TRAINING_CONTRACT_ID,
    _RESERVED_AXIS_METADATA_KEYS,
    GAIN_CALIBRATION_RAW_SCHEMA,
    GainCalibrationCollectionProtocol,
    GainCalibrationRawIdentity,
)


def build_gain_calibration_raw_artifact(
    rows: Mapping[str, Any],
    config: FADAArchitectureConfig,
    protocol: GainCalibrationCollectionProtocol,
    identity: GainCalibrationRawIdentity,
    axis_spec: CalibrationAxisSpec,
    *,
    protocol_bytes: bytes,
    resolved_task_backend_payload: Mapping[str, Any],
) -> dict[str, Any]:
    protocol.validate_approved()
    identity.validate(axis_spec)
    if _protocol_from_bytes(protocol_bytes) != protocol:
        raise ValueError("embedded protocol bytes do not match the approved protocol object")
    observed_protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    normalized_payload = _canonical_json_value(resolved_task_backend_payload)
    if not isinstance(normalized_payload, dict):
        raise TypeError("resolved task/backend provenance must be a mapping")
    observed_task_backend_sha256 = sha256_canonical_mapping(normalized_payload)
    if (
        observed_protocol_sha256 != identity.protocol_sha256
        or observed_task_backend_sha256 != identity.resolved_task_backend_sha256
    ):
        raise ValueError("gain calibration raw rollout provenance digest mismatch")
    _validate_resolved_task_backend_payload(normalized_payload, protocol)
    metadata = asdict(identity)
    metadata.pop("axis_catalog_version")
    return {
        "schema_version": GAIN_CALIBRATION_RAW_SCHEMA,
        "method_contract_id": CALIBRATION_METHOD_CONTRACT_ID,
        "training_contract_id": CALIBRATION_TRAINING_CONTRACT_ID,
        "architecture": asdict(config),
        "axis_spec": axis_spec.to_payload(),
        "protocol_bytes": protocol_bytes,
        "resolved_task_backend_payload": normalized_payload,
        "metadata": metadata,
        **rows,
    }

def _validate_gain_calibration_raw_artifact(
    artifact: Mapping[str, Any],
    *,
    catalog: FaultAxisCatalog,
    legacy: bool,
    expected_source_sha256: str | None = None,
    expected_architecture: FADAArchitectureConfig | None = None,
) -> Mapping[str, Any]:
    expected_schema = _LEGACY_GAIN_CALIBRATION_RAW_SCHEMA if legacy else GAIN_CALIBRATION_RAW_SCHEMA
    if artifact.get("schema_version") != expected_schema:
        raise ValueError("unsupported gain calibration raw rollout schema")
    expected_method = _LEGACY_METHOD_CONTRACT_ID if legacy else CALIBRATION_METHOD_CONTRACT_ID
    expected_training = _LEGACY_TRAINING_CONTRACT_ID if legacy else CALIBRATION_TRAINING_CONTRACT_ID
    if artifact.get("method_contract_id") != expected_method:
        raise ValueError("gain calibration raw rollout method Contract mismatch")
    if artifact.get("training_contract_id") != expected_training:
        raise ValueError("gain calibration raw rollout training Contract mismatch")
    try:
        config = FADAArchitectureConfig(**artifact["architecture"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("gain calibration raw rollout architecture is malformed") from exc
    if expected_architecture is not None and config != expected_architecture:
        raise ValueError("gain calibration raw rollout architecture identity mismatch")
    active_axis_spec = CalibrationAxisSpec.from_catalog(catalog)
    if legacy:
        if (
            catalog.version != _LEGACY_AXIS_CATALOG_VERSION
            or catalog.names != _LEGACY_AXIS_NAMES
            or artifact.get("axis_count") != len(_LEGACY_AXIS_NAMES)
            or tuple(artifact.get("axis_names", ())) != _LEGACY_AXIS_NAMES
        ):
            raise ValueError("legacy gain calibration raw rollout axis catalog mismatch")
    else:
        if artifact.get("axis_spec") != active_axis_spec.to_payload():
            raise ValueError("gain calibration raw rollout axis spec mismatch")
        duplicated_identity = sorted(
            (_RESERVED_AXIS_METADATA_KEYS - {"axis_spec"}).intersection(artifact)
        )
        if duplicated_identity:
            raise ValueError(
                f"active raw rollout contains duplicate axis identity: {duplicated_identity}"
            )
    raw_identity = artifact.get("metadata")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("gain calibration raw rollout metadata must be a mapping")
    try:
        identity_payload = dict(raw_identity)
        if not legacy:
            reserved = sorted(_RESERVED_AXIS_METADATA_KEYS.intersection(identity_payload))
            if reserved:
                raise ValueError("active raw rollout metadata contains reserved axis identity")
            identity_payload["axis_catalog_version"] = active_axis_spec.catalog_version
        identity = GainCalibrationRawIdentity(**identity_payload)
        if legacy:
            for name in (
                "source_checkpoint_sha256",
                "protocol_sha256",
                "resolved_task_backend_sha256",
            ):
                value = getattr(identity, name)
                if len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
                    raise ValueError(f"legacy raw rollout {name} must be a lowercase SHA256")
            if not identity.source_checkpoint_path:
                raise ValueError("legacy raw rollout source checkpoint path is required")
            if identity.axis_catalog_version != _LEGACY_AXIS_CATALOG_VERSION:
                raise ValueError("legacy raw rollout catalog version mismatch")
        else:
            identity.validate(active_axis_spec)
    except (TypeError, ValueError) as exc:
        raise ValueError("gain calibration raw rollout metadata identity is malformed") from exc
    protocol_bytes = artifact.get("protocol_bytes")
    if not isinstance(protocol_bytes, bytes):
        raise ValueError("raw rollout exact protocol bytes are missing")
    protocol = _protocol_from_bytes(protocol_bytes)
    resolved_payload = artifact.get("resolved_task_backend_payload")
    if not isinstance(resolved_payload, Mapping):
        raise ValueError("resolved task/backend provenance material is missing")
    normalized_payload = _canonical_json_value(resolved_payload)
    if not isinstance(normalized_payload, dict) or normalized_payload != resolved_payload:
        raise ValueError("resolved task/backend provenance material is not canonical")
    observed_protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    observed_task_backend_sha256 = sha256_canonical_mapping(normalized_payload)
    if (
        observed_protocol_sha256 != identity.protocol_sha256
        or observed_task_backend_sha256 != identity.resolved_task_backend_sha256
    ):
        raise ValueError("gain calibration raw rollout provenance digest mismatch")
    _validate_resolved_task_backend_payload(normalized_payload, protocol)
    if (
        expected_source_sha256 is not None
        and identity.source_checkpoint_sha256 != expected_source_sha256
    ):
        raise ValueError("gain calibration raw rollout source checkpoint SHA256 mismatch")
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
        "executed_action",
    )
    missing = [name for name in tensor_names if not isinstance(artifact.get(name), torch.Tensor)]
    if missing:
        raise ValueError(f"gain calibration raw rollout is missing tensor fields: {missing}")
    batch = int(artifact["observation_history"].shape[0])
    expected_shapes = {
        "observation_history": (batch, config.history_length, config.obs_dim),
        "action_history": (batch, config.history_length, config.action_dim),
        "command": (batch, config.command_dim),
        "nominal_action_chunk": (batch, config.prediction_horizon, config.action_dim),
        "c_true": (batch, active_axis_spec.axis_count),
        "is_held_out_combination": (batch,),
        "injected_strength": (batch,),
        "planner_intent": (batch, config.prediction_horizon, config.obs_dim),
        "rollout_id": (batch,),
        "seed": (batch,),
        "split_id": (batch,),
        "executed_action": (batch, config.action_dim),
    }
    for name, shape in expected_shapes.items():
        tensor = artifact[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"gain calibration raw rollout {name} shape mismatch")
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"gain calibration raw rollout {name} must be finite")
    axis_name = artifact.get("axis_name")
    if not isinstance(axis_name, (list, tuple)) or axis_name != ["gain"] * batch:
        raise ValueError("gain calibration raw rollout must bind every row to gain")
    if "gain" not in active_axis_spec.names:
        raise ValueError("gain calibration raw rollout catalog lacks gain")
    gain_axis_index = active_axis_spec.names.index("gain")
    if artifact["is_held_out_combination"].dtype != torch.bool or bool(
        artifact["is_held_out_combination"].any()
    ):
        raise ValueError("gain-only smoke rows cannot be held-out combinations")
    omitted_axes = torch.ones(active_axis_spec.axis_count, dtype=torch.bool)
    omitted_axes[gain_axis_index] = False
    if not bool((artifact["c_true"][:, omitted_axes] == 0.0).all()):
        raise ValueError("gain-only smoke rows cannot label omitted axes")
    fixed = torch.tensor(protocol.fixed_command, dtype=artifact["command"].dtype)
    if not torch.equal(artifact["command"], fixed[None].expand(batch, -1)):
        raise ValueError("gain calibration raw rollout command identity mismatch")
    expected_total = (
        len(protocol.points) * len(protocol.splits) * protocol.accepted_rows_per_scenario
    )
    if batch != expected_total:
        raise ValueError(
            f"gain calibration raw rollout row count mismatch: expected={expected_total} got={batch}"
        )
    for split in protocol.splits:
        for point in protocol.points:
            mask = (
                (artifact["split_id"] == split.split_id)
                & (artifact["seed"] == split.seed)
                & (artifact["c_true"][:, gain_axis_index] == point.c_true)
                & (artifact["injected_strength"] == point.gain)
            )
            if int(mask.sum()) != protocol.accepted_rows_per_scenario:
                raise ValueError("gain calibration raw rollout scenario quota mismatch")
            if torch.unique(artifact["rollout_id"][mask]).numel() != 1:
                raise ValueError("gain calibration scenario crosses rollout identities")
    train_ids = set(artifact["rollout_id"][artifact["split_id"] == 0].tolist())
    validation_ids = set(artifact["rollout_id"][artifact["split_id"] == 1].tolist())
    if not train_ids.isdisjoint(validation_ids):
        raise ValueError("gain calibration train and validation rollout identities overlap")
    return artifact


def validate_gain_calibration_raw_artifact(
    artifact: Mapping[str, Any],
    *,
    catalog: FaultAxisCatalog,
    expected_source_sha256: str | None = None,
    expected_architecture: FADAArchitectureConfig | None = None,
) -> Mapping[str, Any]:
    """Validate only the active raw v2 envelope used by current writers."""

    return _validate_gain_calibration_raw_artifact(
        artifact,
        catalog=catalog,
        legacy=False,
        expected_source_sha256=expected_source_sha256,
        expected_architecture=expected_architecture,
    )


def _load_legacy_gain_calibration_raw_gateway(
    artifact: Mapping[str, Any],
    *,
    catalog: FaultAxisCatalog,
    expected_source_sha256: str | None,
    expected_architecture: FADAArchitectureConfig | None,
) -> Mapping[str, Any]:
    """Read the exact historical v1 donor envelope for one-time dataset resealing."""

    return _validate_gain_calibration_raw_artifact(
        artifact,
        catalog=catalog,
        legacy=True,
        expected_source_sha256=expected_source_sha256,
        expected_architecture=expected_architecture,
    )


def save_gain_calibration_raw_rollouts(
    path: str | Path,
    artifact: Mapping[str, Any],
    *,
    catalog: FaultAxisCatalog,
) -> Path:
    validate_gain_calibration_raw_artifact(artifact, catalog=catalog)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(dict(artifact), temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_gain_calibration_raw_rollouts(
    path: str | Path,
    *,
    catalog: FaultAxisCatalog,
    expected_source_sha256: str | None = None,
    expected_architecture: FADAArchitectureConfig | None = None,
) -> Mapping[str, Any]:
    serialized = Path(path).expanduser().resolve().read_bytes()
    payload = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("gain calibration raw rollout artifact must be a mapping")
    if payload.get("schema_version") == _LEGACY_GAIN_CALIBRATION_RAW_SCHEMA:
        return _load_legacy_gain_calibration_raw_gateway(
            payload,
            catalog=catalog,
            expected_source_sha256=expected_source_sha256,
            expected_architecture=expected_architecture,
        )
    return validate_gain_calibration_raw_artifact(
        payload,
        catalog=catalog,
        expected_source_sha256=expected_source_sha256,
        expected_architecture=expected_architecture,
    )
