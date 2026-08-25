from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from unilab.algos.torch.distill.fada import FADAArchitectureConfig
from unilab.algos.torch.fada_context.calibration_models import (
    CoefficientEncoder,
    DirectionBank,
)
from unilab.algos.torch.fada_context.calibration_readout import MonotoneScaleCurve
from unilab.algos.torch.fada_context.calibration_types import (
    CALIBRATION_ARTIFACT_SCHEMA,
    CALIBRATION_METHOD_CONTRACT_ID,
    CALIBRATION_TRAINING_CONTRACT_ID,
    CalibrationAxisSpec,
    FaultAxisCatalog,
)


def _validate_calibration_artifact_metadata(metadata: object) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("calibration artifact metadata is missing")
    digest_fields = (
        "source_tracker_sha256",
        "dataset_sha256",
        "split_sha256",
        "parent_stage_sha256",
        "scale_evidence_sha256",
    )
    for name in digest_fields:
        value = metadata.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"calibration artifact lineage {name} must be a "
                "64-character lowercase hexadecimal digest"
            )
    if any(
        name in metadata
        for name in ("axis_catalog_version", "axis_count", "axis_names", "axis_spec")
    ):
        raise ValueError("calibration artifact metadata contains reserved axis identity")
    if metadata.get("stage") != "complete":
        raise ValueError("calibration artifact lineage stage must be complete")
    return metadata


def save_calibration_artifact(
    path: str | Path,
    *,
    config: FADAArchitectureConfig,
    direction_bank: DirectionBank,
    scale_curves: tuple[MonotoneScaleCurve, ...],
    coefficient_encoder: CoefficientEncoder,
    axis_spec: CalibrationAxisSpec,
    metadata: Mapping[str, Any],
) -> Path:
    if len(scale_curves) != direction_bank.axis_count:
        raise ValueError("one scale curve is required per direction axis")
    if direction_bank.axis_count != axis_spec.axis_count:
        raise ValueError("calibration artifact axis count does not match its axis spec")
    if tuple(direction_bank.directions.shape[1:]) != (
        config.prediction_horizon,
        config.hidden_dim,
    ):
        raise ValueError("calibration artifact Direction Bank architecture mismatch")
    direction_norms = direction_bank.directions.detach().flatten(1).norm(dim=1)
    if not torch.allclose(direction_norms, torch.ones_like(direction_norms), rtol=1e-5, atol=1e-6):
        raise ValueError("calibration artifact requires normalized Direction Bank fields")
    if bool((direction_bank.normalization_scale <= 0).any()) or not bool(
        torch.isfinite(direction_bank.normalization_scale).all()
    ):
        raise ValueError("calibration artifact normalization scale must be finite and positive")
    if coefficient_encoder.history_length != config.history_length:
        raise ValueError("calibration artifact Encoder history mismatch")
    expected_encoder = (
        config.obs_dim,
        config.action_dim,
        axis_spec.axis_count,
        128,
        2,
    )
    observed_encoder = (
        coefficient_encoder.state_embedding.in_features,
        coefficient_encoder.action_embedding.in_features,
        coefficient_encoder.axis_count,
        coefficient_encoder.state_embedding.out_features,
        len(coefficient_encoder.encoder.layers),
    )
    if observed_encoder != expected_encoder:
        raise ValueError("calibration artifact Coefficient Encoder architecture mismatch")
    _validate_calibration_artifact_metadata(metadata)
    for curve in scale_curves:
        if curve.x.numel() != 21:
            raise ValueError("calibration artifact requires a 21-point scale grid")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CALIBRATION_ARTIFACT_SCHEMA,
        "method_contract_id": CALIBRATION_METHOD_CONTRACT_ID,
        "training_contract_id": CALIBRATION_TRAINING_CONTRACT_ID,
        "architecture": asdict(config),
        "axis_spec": axis_spec.to_payload(),
        "direction_bank": direction_bank.state_dict(),
        "coefficient_encoder": coefficient_encoder.state_dict(),
        "coefficient_encoder_config": {
            "state_dim": coefficient_encoder.state_embedding.in_features,
            "action_dim": coefficient_encoder.action_embedding.in_features,
            "axis_count": coefficient_encoder.axis_count,
            "hidden_dim": coefficient_encoder.state_embedding.out_features,
            "layers": len(coefficient_encoder.encoder.layers),
        },
        "scale_curves": [
            {"x": curve.x, "y": curve.y, "slopes": curve.slopes, "kind": curve.kind}
            for curve in scale_curves
        ],
        "metadata": dict(metadata),
    }
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_calibration_artifact(
    path: str | Path,
    catalog: FaultAxisCatalog,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CALIBRATION_ARTIFACT_SCHEMA
    ):
        raise ValueError("unsupported calibration artifact schema")
    if payload.get("method_contract_id") != CALIBRATION_METHOD_CONTRACT_ID:
        raise ValueError("calibration artifact method Contract mismatch")
    if payload.get("training_contract_id") != CALIBRATION_TRAINING_CONTRACT_ID:
        raise ValueError("calibration artifact training Contract mismatch")
    if not isinstance(payload.get("architecture"), Mapping):
        raise ValueError("calibration artifact architecture is missing")
    axis_spec = CalibrationAxisSpec.from_payload(payload.get("axis_spec"), catalog)
    _validate_calibration_artifact_metadata(payload.get("metadata"))
    if (
        not isinstance(payload.get("direction_bank"), Mapping)
        or not isinstance(payload.get("coefficient_encoder"), Mapping)
        or not isinstance(payload.get("coefficient_encoder_config"), Mapping)
        or not isinstance(payload.get("scale_curves"), list)
    ):
        raise ValueError("calibration artifact is missing typed owners")
    validate_finite_state_tree("calibration artifact", payload)
    direction_state = payload["direction_bank"]
    directions = direction_state.get("directions")
    normalization_scale = direction_state.get("normalization_scale")
    if (
        not isinstance(directions, torch.Tensor)
        or directions.ndim != 3
        or directions.shape[0] != axis_spec.axis_count
        or not isinstance(normalization_scale, torch.Tensor)
        or normalization_scale.shape != (directions.shape[0],)
    ):
        raise ValueError("calibration artifact Direction Bank state is malformed")
    norms = directions.flatten(1).norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-6):
        raise ValueError("calibration artifact Direction Bank is not normalized")
    if bool((normalization_scale <= 0).any()):
        raise ValueError("calibration artifact normalization scale must be positive")
    expected_encoder_config = {
        "state_dim": int(payload["architecture"]["obs_dim"]),
        "action_dim": int(payload["architecture"]["action_dim"]),
        "axis_count": axis_spec.axis_count,
        "hidden_dim": 128,
        "layers": 2,
    }
    if payload["coefficient_encoder_config"] != expected_encoder_config:
        raise ValueError("calibration artifact Coefficient Encoder architecture mismatch")
    if len(payload["scale_curves"]) != axis_spec.axis_count:
        raise ValueError("calibration artifact scale curve count mismatch")
    for curve in payload["scale_curves"]:
        _validate_scale_curve_payload(curve)
    return payload


def validate_finite_state_tree(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta" or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} tensors must be finite materialized values")
        return
    if isinstance(value, Mapping):
        for child in value.values():
            validate_finite_state_tree(name, child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            validate_finite_state_tree(name, child)


def _validate_scale_curve_payload(value: Any) -> MonotoneScaleCurve:
    if (
        not isinstance(value, Mapping)
        or not all(isinstance(value.get(name), torch.Tensor) for name in ("x", "y", "slopes"))
        or value.get("kind") != "pchip"
    ):
        raise ValueError("calibration scale curve is malformed")
    if value["x"].numel() != 21:
        raise ValueError("calibration scale curve requires a 21-point grid")
    validated = MonotoneScaleCurve.fit(value["x"], value["y"])
    if value["slopes"].shape != validated.slopes.shape or not torch.allclose(
        value["slopes"],
        validated.slopes,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("calibration PCHIP slopes are inconsistent")
    return validated
