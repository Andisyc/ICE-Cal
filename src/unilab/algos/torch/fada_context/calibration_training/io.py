from __future__ import annotations

import hashlib
import io
import math
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import (
    CALIBRATION_METHOD_CONTRACT_ID,
    CALIBRATION_TRAINING_CONTRACT_ID,
    CalibrationAxisSpec,
    CoefficientEncoder,
    DirectionBank,
    MonotoneScaleCurve,
    _validate_finite_state_tree,
    _validate_scale_curve_payload,
    save_calibration_artifact,
)
from unilab.algos.torch.fada_context.calibration_training.types import (
    _COEFFICIENT_ERROR_LIMIT,
    _COEFFICIENT_STAGE,
    _COMPENSATION_RATIO_LIMIT,
    _DIRECTION_STAGE,
    _IDENTITY_FIELDS,
    CalibrationStageIdentity,
)

CALIBRATION_STAGE_ARTIFACT_SCHEMA = "unilab_fada_calibration_stage_artifact_v3"
CALIBRATION_SCALE_EVIDENCE_SCHEMA = "unilab_fada_calibration_scale_evidence_v2"


@dataclass(frozen=True)
class CalibrationScaleEvidence:
    coefficient_scan_grid: torch.Tensor
    readings: torch.Tensor
    candidate_scales: torch.Tensor
    action_errors: torch.Tensor
    axis_spec: CalibrationAxisSpec
    metadata: Mapping[str, str]

    def validate(self) -> CalibrationScaleEvidence:
        _validate_scale_evidence_tensors(
            self.coefficient_scan_grid,
            self.readings,
            self.candidate_scales,
            self.action_errors,
            self.axis_spec,
        )
        if any(
            not isinstance(self.metadata.get(name), str) or not self.metadata[name]
            for name in _IDENTITY_FIELDS
        ):
            raise ValueError("calibration scale evidence metadata identity is incomplete")
        if any(
            name in self.metadata
            for name in ("axis_catalog_version", "axis_count", "axis_names", "axis_spec")
        ):
            raise ValueError("calibration scale evidence metadata contains reserved axis identity")
        return self


def _validate_scale_evidence_tensors(
    coefficient_scan_grid: torch.Tensor,
    readings: torch.Tensor,
    candidate_scales: torch.Tensor,
    action_errors: torch.Tensor,
    axis_spec: CalibrationAxisSpec,
) -> None:
    expected_grid = torch.linspace(
        -1.0,
        1.0,
        21,
        dtype=coefficient_scan_grid.dtype,
        device=coefficient_scan_grid.device,
    ).repeat(axis_spec.axis_count, 1)
    if coefficient_scan_grid.shape != expected_grid.shape or not torch.equal(
        coefficient_scan_grid, expected_grid
    ):
        raise ValueError("Stage 3 coefficient scan grid must be m [-1,1] 21-point rows")
    if readings.ndim != 3 or readings.shape[1:] != (21, 32):
        raise ValueError("Stage 3 requires 21 points and 32 repetitions per axis")
    if readings.shape[0] != axis_spec.axis_count:
        raise ValueError("Stage 3 evidence axis count does not match the axis spec")
    if (
        candidate_scales.ndim != 1
        or candidate_scales.numel() < 2
        or bool((candidate_scales[1:] <= candidate_scales[:-1]).any())
    ):
        raise ValueError("Stage 3 candidate scales must be strictly increasing")
    if action_errors.shape != (*readings.shape, candidate_scales.numel()):
        raise ValueError("Stage 3 Action errors must be [axis,21,32,candidate]")
    if not bool(
        torch.isfinite(coefficient_scan_grid).all()
        and torch.isfinite(readings).all()
        and torch.isfinite(candidate_scales).all()
        and torch.isfinite(action_errors).all()
    ):
        raise ValueError("Stage 3 evidence must be finite")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _load_exact_torch_payload(path: str | Path) -> tuple[Any, str]:
    serialized = Path(path).expanduser().resolve().read_bytes()
    digest = _sha256_bytes(serialized)
    payload = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=True)
    return payload, digest


def _atomic_torch_save(target_path: str | Path, payload: object) -> Path:
    target = Path(target_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _atomic_save_deployment_artifact(
    target_path: str | Path,
    *,
    policy: FADAPlannerIDMPolicy,
    direction_bank: DirectionBank,
    coefficient_encoder: CoefficientEncoder,
    scale_curves: tuple[MonotoneScaleCurve, ...],
    axis_spec: CalibrationAxisSpec,
    metadata: Mapping[str, str],
) -> Path:
    target = Path(target_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".staging",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    nested_temporary = temporary.with_suffix(f"{temporary.suffix}.tmp")
    try:
        save_calibration_artifact(
            temporary,
            config=policy.config,
            direction_bank=direction_bank,
            coefficient_encoder=coefficient_encoder,
            scale_curves=scale_curves,
            axis_spec=axis_spec,
            metadata=metadata,
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
        nested_temporary.unlink(missing_ok=True)
    return target


def save_calibration_scale_evidence(
    path: str | Path,
    evidence: CalibrationScaleEvidence,
) -> Path:
    evidence.validate()
    return _atomic_torch_save(
        path,
        {
            "schema_version": CALIBRATION_SCALE_EVIDENCE_SCHEMA,
            "method_contract_id": CALIBRATION_METHOD_CONTRACT_ID,
            "training_contract_id": CALIBRATION_TRAINING_CONTRACT_ID,
            "axis_spec": evidence.axis_spec.to_payload(),
            "coefficient_scan_grid": evidence.coefficient_scan_grid.detach().cpu(),
            "readings": evidence.readings.detach().cpu(),
            "candidate_scales": evidence.candidate_scales.detach().cpu(),
            "action_errors": evidence.action_errors.detach().cpu(),
            "metadata": dict(evidence.metadata),
        },
    )


def load_calibration_scale_evidence(
    path: str | Path,
    *,
    expected_identity: CalibrationStageIdentity,
) -> CalibrationScaleEvidence:
    payload, _ = _load_exact_torch_payload(path)
    return _calibration_scale_evidence_from_payload(payload, expected_identity)


def _calibration_scale_evidence_from_payload(
    payload: object,
    expected_identity: CalibrationStageIdentity,
) -> CalibrationScaleEvidence:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CALIBRATION_SCALE_EVIDENCE_SCHEMA
    ):
        raise ValueError("unsupported calibration scale evidence schema")
    if payload.get("method_contract_id") != CALIBRATION_METHOD_CONTRACT_ID:
        raise ValueError("calibration scale evidence method Contract mismatch")
    if payload.get("training_contract_id") != CALIBRATION_TRAINING_CONTRACT_ID:
        raise ValueError("calibration scale evidence training Contract mismatch")
    if payload.get("axis_spec") != expected_identity.axis_spec.to_payload():
        raise ValueError("calibration scale evidence axis spec mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or any(
        metadata.get(name) != getattr(expected_identity, name) for name in _IDENTITY_FIELDS
    ):
        raise ValueError("calibration scale evidence metadata identity mismatch")
    tensors = (
        payload.get("coefficient_scan_grid"),
        payload.get("readings"),
        payload.get("candidate_scales"),
        payload.get("action_errors"),
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError("calibration scale evidence tensor fields are missing")
    coefficient_scan_grid, readings, candidate_scales, action_errors = tensors
    assert isinstance(coefficient_scan_grid, torch.Tensor)
    assert isinstance(readings, torch.Tensor)
    assert isinstance(candidate_scales, torch.Tensor)
    assert isinstance(action_errors, torch.Tensor)
    return CalibrationScaleEvidence(
        coefficient_scan_grid=coefficient_scan_grid,
        readings=readings,
        candidate_scales=candidate_scales,
        action_errors=action_errors,
        axis_spec=expected_identity.axis_spec,
        metadata=dict(metadata),
    ).validate()


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _identity_payload(
    identity: CalibrationStageIdentity,
    *,
    include_axis_spec: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {name: str(getattr(identity, name)) for name in _IDENTITY_FIELDS}
    if include_axis_spec:
        payload["axis_spec"] = identity.axis_spec.to_payload()
    return payload


def _stage_envelope(
    *,
    policy: FADAPlannerIDMPolicy,
    identity: CalibrationStageIdentity,
    stage: Literal["direction_frozen", "coefficient_frozen"],
    gate: Mapping[str, object],
    owners: Mapping[str, object],
    parent_stage_sha256: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CALIBRATION_STAGE_ARTIFACT_SCHEMA,
        "method_contract_id": CALIBRATION_METHOD_CONTRACT_ID,
        "training_contract_id": CALIBRATION_TRAINING_CONTRACT_ID,
        "stage": stage,
        "architecture": asdict(policy.config),
        "dimensions": {
            "history_length": policy.config.history_length,
            "prediction_horizon": policy.config.prediction_horizon,
            "latent_dim": policy.config.hidden_dim,
        },
        "identity": _identity_payload(identity),
        "gate": dict(gate),
        "owners": dict(owners),
    }
    if parent_stage_sha256 is not None:
        payload["parent_stage_sha256"] = parent_stage_sha256
    return payload


def _validate_common_stage_envelope(
    payload: object,
    *,
    policy: FADAPlannerIDMPolicy,
    identity: CalibrationStageIdentity,
    expected_stage: Literal["direction_frozen", "coefficient_frozen"],
) -> dict[str, object]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CALIBRATION_STAGE_ARTIFACT_SCHEMA
    ):
        raise ValueError("unsupported calibration stage artifact schema")
    if payload.get("method_contract_id") != CALIBRATION_METHOD_CONTRACT_ID:
        raise ValueError("calibration stage artifact method Contract mismatch")
    if payload.get("training_contract_id") != CALIBRATION_TRAINING_CONTRACT_ID:
        raise ValueError("calibration stage artifact training Contract mismatch")
    if payload.get("stage") != expected_stage:
        raise ValueError(
            f"calibration stage artifact stage mismatch: expected={expected_stage} "
            f"observed={payload.get('stage')}"
        )
    if payload.get("architecture") != asdict(policy.config):
        raise ValueError("calibration stage artifact architecture mismatch")
    if payload.get("dimensions") != {
        "history_length": policy.config.history_length,
        "prediction_horizon": policy.config.prediction_horizon,
        "latent_dim": policy.config.hidden_dim,
    }:
        raise ValueError("calibration stage artifact H/K/D mismatch")
    if payload.get("identity") != _identity_payload(identity):
        raise ValueError("calibration stage artifact transaction identity mismatch")
    if not isinstance(payload.get("gate"), Mapping):
        raise ValueError("calibration stage artifact gate is missing")
    if not isinstance(payload.get("owners"), Mapping):
        raise ValueError("calibration stage artifact owners are missing")
    _validate_finite_state_tree("calibration stage artifact", payload)
    return payload


def _load_direction_stage_artifact(
    path: str | Path,
    *,
    policy: FADAPlannerIDMPolicy,
    identity: CalibrationStageIdentity,
) -> tuple[DirectionBank, str, dict[str, object]]:
    payload, digest = _load_exact_torch_payload(path)
    payload = _validate_common_stage_envelope(
        payload,
        policy=policy,
        identity=identity,
        expected_stage=_DIRECTION_STAGE,
    )
    owners = payload["owners"]
    assert isinstance(owners, Mapping)
    if set(owners) != {"direction_bank"}:
        raise ValueError("direction stage artifact has forbidden owners")
    direction_owner = owners.get("direction_bank")
    expected_config = {
        "axis_count": identity.axis_spec.axis_count,
        "prediction_horizon": policy.config.prediction_horizon,
        "latent_dim": policy.config.hidden_dim,
    }
    if not isinstance(direction_owner, Mapping) or direction_owner.get("config") != expected_config:
        raise ValueError("direction stage artifact Direction Bank config mismatch")
    direction_state = direction_owner.get("state_dict")
    if not isinstance(direction_state, Mapping):
        raise ValueError("direction stage artifact Direction Bank state is missing")
    directions = direction_state.get("directions")
    normalization_scale = direction_state.get("normalization_scale")
    if (
        not isinstance(directions, torch.Tensor)
        or directions.shape
        != (
            identity.axis_spec.axis_count,
            policy.config.prediction_horizon,
            policy.config.hidden_dim,
        )
        or not isinstance(normalization_scale, torch.Tensor)
        or normalization_scale.shape != (identity.axis_spec.axis_count,)
    ):
        raise ValueError("direction stage artifact Direction Bank state is malformed")
    if not torch.allclose(
        directions.flatten(1).norm(dim=1),
        torch.ones(directions.shape[0]),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("direction stage artifact directions are not normalized")
    if bool((normalization_scale <= 0).any()):
        raise ValueError("direction stage artifact normalization scale must be positive")
    gate = payload["gate"]
    assert isinstance(gate, Mapping)
    ratios = gate.get("result")
    threshold = gate.get("threshold")
    if (
        gate.get("name") != "compensation_ratio"
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 < float(threshold) <= _COMPENSATION_RATIO_LIMIT
        or not isinstance(ratios, list)
        or len(ratios) != identity.axis_spec.axis_count
        or any(not isinstance(value, (int, float)) for value in ratios)
        or any(not math.isfinite(float(value)) for value in ratios)
        or any(float(value) > float(threshold) for value in ratios)
    ):
        raise ValueError("direction stage artifact gate did not pass")
    bank = DirectionBank(**expected_config)
    bank.load_state_dict(dict(direction_state), strict=True)
    bank.requires_grad_(False)
    return bank, digest, payload


def _coefficient_encoder_config(
    policy: FADAPlannerIDMPolicy,
    axis_count: int,
) -> dict[str, int]:
    return {
        "state_dim": policy.config.obs_dim,
        "action_dim": policy.config.action_dim,
        "axis_count": axis_count,
        "hidden_dim": 128,
        "layers": 2,
    }


def _load_coefficient_stage_artifact(
    path: str | Path,
    *,
    policy: FADAPlannerIDMPolicy,
    identity: CalibrationStageIdentity,
) -> tuple[DirectionBank, CoefficientEncoder, str, dict[str, object]]:
    payload, digest = _load_exact_torch_payload(path)
    payload = _validate_common_stage_envelope(
        payload,
        policy=policy,
        identity=identity,
        expected_stage=_COEFFICIENT_STAGE,
    )
    parent_digest = payload.get("parent_stage_sha256")
    if (
        not isinstance(parent_digest, str)
        or len(parent_digest) != 64
        or any(character not in "0123456789abcdef" for character in parent_digest)
    ):
        raise ValueError("coefficient stage artifact parent digest is missing")
    owners = payload["owners"]
    assert isinstance(owners, Mapping)
    if set(owners) != {"direction_bank", "coefficient_encoder"}:
        raise ValueError("coefficient stage artifact owner set is invalid")
    direction_owner = owners.get("direction_bank")
    encoder_owner = owners.get("coefficient_encoder")
    direction_config = {
        "axis_count": identity.axis_spec.axis_count,
        "prediction_horizon": policy.config.prediction_horizon,
        "latent_dim": policy.config.hidden_dim,
    }
    encoder_config = _coefficient_encoder_config(policy, identity.axis_spec.axis_count)
    if (
        not isinstance(direction_owner, Mapping)
        or direction_owner.get("config") != direction_config
        or not isinstance(direction_owner.get("state_dict"), Mapping)
    ):
        raise ValueError("coefficient stage artifact Direction Bank is malformed")
    if (
        not isinstance(encoder_owner, Mapping)
        or encoder_owner.get("config") != encoder_config
        or not isinstance(encoder_owner.get("state_dict"), Mapping)
    ):
        raise ValueError("coefficient stage artifact Encoder is malformed")
    directions = direction_owner["state_dict"].get("directions")
    normalization_scale = direction_owner["state_dict"].get("normalization_scale")
    if (
        not isinstance(directions, torch.Tensor)
        or directions.shape
        != (
            identity.axis_spec.axis_count,
            policy.config.prediction_horizon,
            policy.config.hidden_dim,
        )
        or not isinstance(normalization_scale, torch.Tensor)
        or normalization_scale.shape != (identity.axis_spec.axis_count,)
        or not torch.allclose(
            directions.flatten(1).norm(dim=1),
            torch.ones(directions.shape[0]),
            rtol=1e-5,
            atol=1e-6,
        )
        or bool((normalization_scale <= 0).any())
    ):
        raise ValueError("coefficient stage artifact directions are invalid")
    gate = payload["gate"]
    assert isinstance(gate, Mapping)
    error = gate.get("result")
    threshold = gate.get("threshold")
    if (
        gate.get("name") != "coefficient_error"
        or not isinstance(error, (int, float))
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(error))
        or not math.isfinite(float(threshold))
        or not 0 < float(threshold) <= _COEFFICIENT_ERROR_LIMIT
        or float(error) > float(threshold)
    ):
        raise ValueError("coefficient stage artifact gate did not pass")
    bank = DirectionBank(**direction_config)
    bank.load_state_dict(dict(direction_owner["state_dict"]), strict=True)
    bank.requires_grad_(False)
    encoder = CoefficientEncoder(**encoder_config)
    encoder.load_state_dict(dict(encoder_owner["state_dict"]), strict=True)
    encoder.requires_grad_(False)
    return bank, encoder, digest, payload
