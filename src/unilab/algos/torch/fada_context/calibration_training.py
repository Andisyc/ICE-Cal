from __future__ import annotations

import hashlib
import io
import math
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import (
    CALIBRATION_AXIS_CATALOG_VERSION,
    CALIBRATION_AXIS_NAMES,
    CALIBRATION_METHOD_CONTRACT_ID,
    CALIBRATION_TRAINING_CONTRACT_ID,
    CalibrationRolloutBatch,
    CoefficientEncoder,
    DirectionBank,
    MonotoneScaleCurve,
    _validate_finite_state_tree,
    _validate_scale_curve_payload,
    fit_scale_curve_bank,
    save_calibration_artifact,
)

CALIBRATION_STAGE_ARTIFACT_SCHEMA = "unilab_fada_calibration_stage_artifact_v2"
CALIBRATION_SCALE_EVIDENCE_SCHEMA = "unilab_fada_calibration_scale_evidence_v1"
_IDENTITY_FIELDS = (
    "source_tracker_sha256",
    "dataset_sha256",
    "split_sha256",
    "axis_catalog_version",
)
_DIRECTION_STAGE: Literal["direction_frozen"] = "direction_frozen"
_COEFFICIENT_STAGE: Literal["coefficient_frozen"] = "coefficient_frozen"
_COMPENSATION_RATIO_LIMIT = 0.1
_COEFFICIENT_ERROR_LIMIT = 0.05


@dataclass(frozen=True)
class CalibrationStageIdentity:
    source_tracker_sha256: str
    dataset_sha256: str
    split_sha256: str
    axis_catalog_version: str

    def validate(self) -> CalibrationStageIdentity:
        for name in _IDENTITY_FIELDS[:-1]:
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"calibration stage identity {name} must be a "
                    "64-character lowercase hexadecimal digest"
                )
        if not isinstance(self.axis_catalog_version, str) or not self.axis_catalog_version:
            raise ValueError("calibration stage identity axis catalog is incomplete")
        if self.axis_catalog_version != CALIBRATION_AXIS_CATALOG_VERSION:
            raise ValueError("calibration stage identity axis catalog mismatch")
        return self


@dataclass(frozen=True)
class DirectionStageConfig:
    steps_per_axis: int = 100
    learning_rate: float = 3.0e-4
    compensation_ratio_threshold: float = 0.1
    training_split_id: int = 0
    validation_split_id: int = 1

    def __post_init__(self) -> None:
        if self.steps_per_axis <= 0:
            raise ValueError("Stage 1 steps_per_axis must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Stage 1 learning_rate must be finite and positive")
        if not 0 < self.compensation_ratio_threshold <= _COMPENSATION_RATIO_LIMIT:
            raise ValueError("Stage 1 compensation_ratio_threshold must be in (0,0.1]")
        if self.training_split_id == self.validation_split_id:
            raise ValueError("Stage 1 training and validation split IDs must differ")


@dataclass(frozen=True)
class CoefficientStageConfig:
    steps: int = 1000
    learning_rate: float = 3.0e-4
    coefficient_error_threshold: float = 0.05
    training_split_id: int = 0
    validation_split_id: int = 1

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("Stage 2 steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Stage 2 learning_rate must be finite and positive")
        if not 0 < self.coefficient_error_threshold <= _COEFFICIENT_ERROR_LIMIT:
            raise ValueError("Stage 2 coefficient_error_threshold must be in (0,0.05]")
        if self.training_split_id == self.validation_split_id:
            raise ValueError("Stage 2 training and validation split IDs must differ")


@dataclass(frozen=True)
class DirectionStageResult:
    stage: Literal["direction_frozen"]
    artifact_path: Path
    artifact_sha256: str
    compensation_ratios: tuple[float, float, float]


@dataclass(frozen=True)
class CoefficientStageResult:
    stage: Literal["coefficient_frozen"]
    artifact_path: Path
    artifact_sha256: str
    parent_stage_sha256: str
    coefficient_error: float


@dataclass(frozen=True)
class ScaleStageResult:
    stage: Literal["complete"]
    artifact_path: Path
    artifact_sha256: str
    parent_stage_sha256: str
    scale_evidence_sha256: str


@dataclass(frozen=True)
class SerialCalibrationConfig:
    stage1_steps_per_axis: int = 100
    stage2_steps: int = 1000
    learning_rate: float = 3.0e-4
    compensation_ratio_threshold: float = 0.1
    coefficient_error_threshold: float = 0.05
    training_split_id: int = 0
    validation_split_id: int = 1

    def __post_init__(self) -> None:
        if self.stage1_steps_per_axis <= 0 or self.stage2_steps <= 0:
            raise ValueError("serial calibration steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("serial calibration learning_rate must be finite and positive")
        if not 0 < self.compensation_ratio_threshold <= _COMPENSATION_RATIO_LIMIT:
            raise ValueError("compensation_ratio_threshold must be in (0,0.1]")
        if not 0 < self.coefficient_error_threshold <= _COEFFICIENT_ERROR_LIMIT:
            raise ValueError("coefficient_error_threshold must be in (0,0.05]")
        if self.training_split_id == self.validation_split_id:
            raise ValueError("training and validation split IDs must differ")

    def direction_stage(self) -> DirectionStageConfig:
        return DirectionStageConfig(
            steps_per_axis=self.stage1_steps_per_axis,
            learning_rate=self.learning_rate,
            compensation_ratio_threshold=self.compensation_ratio_threshold,
            training_split_id=self.training_split_id,
            validation_split_id=self.validation_split_id,
        )

    def coefficient_stage(self) -> CoefficientStageConfig:
        return CoefficientStageConfig(
            steps=self.stage2_steps,
            learning_rate=self.learning_rate,
            coefficient_error_threshold=self.coefficient_error_threshold,
            training_split_id=self.training_split_id,
            validation_split_id=self.validation_split_id,
        )


@dataclass(frozen=True)
class CalibrationScaleEvidence:
    coefficient_scan_grid: torch.Tensor
    readings: torch.Tensor
    candidate_scales: torch.Tensor
    action_errors: torch.Tensor
    metadata: Mapping[str, str]

    def validate(self) -> CalibrationScaleEvidence:
        _validate_scale_evidence_tensors(
            self.coefficient_scan_grid,
            self.readings,
            self.candidate_scales,
            self.action_errors,
        )
        if any(
            not isinstance(self.metadata.get(name), str) or not self.metadata[name]
            for name in _IDENTITY_FIELDS
        ):
            raise ValueError("calibration scale evidence metadata identity is incomplete")
        if self.metadata["axis_catalog_version"] != CALIBRATION_AXIS_CATALOG_VERSION:
            raise ValueError("calibration scale evidence axis catalog mismatch")
        return self


def _validate_scale_evidence_tensors(
    coefficient_scan_grid: torch.Tensor,
    readings: torch.Tensor,
    candidate_scales: torch.Tensor,
    action_errors: torch.Tensor,
) -> None:
    expected_grid = torch.linspace(
        -1.0,
        1.0,
        21,
        dtype=coefficient_scan_grid.dtype,
        device=coefficient_scan_grid.device,
    ).repeat(len(CALIBRATION_AXIS_NAMES), 1)
    if coefficient_scan_grid.shape != expected_grid.shape or not torch.equal(
        coefficient_scan_grid, expected_grid
    ):
        raise ValueError("Stage 3 coefficient scan grid must be three [-1,1] 21-point rows")
    if readings.ndim != 3 or readings.shape[1:] != (21, 32):
        raise ValueError("Stage 3 requires 21 points and 32 repetitions per axis")
    if readings.shape[0] != len(CALIBRATION_AXIS_NAMES):
        raise ValueError("Stage 3 evidence axis count does not match the active catalog")
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
            "axis_names": CALIBRATION_AXIS_NAMES,
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
    expected_metadata: Mapping[str, str],
) -> CalibrationScaleEvidence:
    payload, _ = _load_exact_torch_payload(path)
    return _calibration_scale_evidence_from_payload(payload, expected_metadata)


def _calibration_scale_evidence_from_payload(
    payload: object,
    expected_metadata: Mapping[str, str],
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
    if tuple(payload.get("axis_names", ())) != CALIBRATION_AXIS_NAMES:
        raise ValueError("calibration scale evidence axis catalog mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or any(
        metadata.get(name) != expected_metadata.get(name) for name in _IDENTITY_FIELDS
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
        metadata=dict(metadata),
    ).validate()


def _validate_stage_batch(
    policy: FADAPlannerIDMPolicy, batch: CalibrationRolloutBatch, axis_count: int
) -> None:
    batch.validate(policy.config, axis_count=axis_count)


@torch.no_grad()
def validate_calibration_source_projection(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
) -> None:
    if any(module.training for module in policy.modules()):
        raise ValueError("source Planner/Tracker must be in frozen evaluation mode")
    predicted_future = policy.planner(batch.observation_history, batch.command)
    nominal_latent = policy.idm.encode_latent(
        batch.observation_history,
        batch.action_history,
        predicted_future,
    )
    nominal_actions = policy.idm.decode_latent(nominal_latent)
    if not torch.allclose(
        predicted_future,
        batch.planner_intent.to(predicted_future),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("dataset Planner Intent does not match the source policy")
    if not torch.allclose(
        nominal_actions,
        batch.nominal_action_chunk.to(nominal_actions),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("dataset nominal Action does not match the source Tracker")


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _restore(module: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(snapshot, strict=True)


def _require_unchanged(
    name: str,
    module: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> None:
    if any(not torch.equal(value, snapshot[key]) for key, value in module.state_dict().items()):
        raise ValueError(f"frozen owner mutated during {name}")


@contextmanager
def _freeze(module: torch.nn.Module) -> Iterator[None]:
    original = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original, strict=True):
            parameter.requires_grad_(requires_grad)


def direction_stage_loss(
    policy: FADAPlannerIDMPolicy,
    direction_bank: DirectionBank,
    batch: CalibrationRolloutBatch,
    *,
    axis_index: int,
) -> torch.Tensor:
    if axis_index < 0 or axis_index >= direction_bank.axis_count:
        raise ValueError("axis_index is outside Direction Bank")
    _validate_stage_batch(policy, batch, direction_bank.axis_count)
    selected = torch.nonzero(
        (batch.axis_id == axis_index) & ~batch.is_held_out_combination,
        as_tuple=False,
    ).flatten()
    if selected.numel() == 0:
        raise ValueError(f"Stage 1 has no rows for axis {axis_index}")
    batch = batch.index_select(selected)
    with _freeze(policy):
        predicted_future = policy.planner(batch.observation_history, batch.command)
        latent = policy.idm.encode_latent(
            batch.observation_history, batch.action_history, predicted_future
        )
        coefficients = torch.zeros(
            batch.observation_history.shape[0],
            direction_bank.axis_count,
            device=latent.device,
            dtype=latent.dtype,
        )
        coefficients[:, axis_index] = batch.c_true[:, axis_index].to(latent)
        calibrated = direction_bank.compose(latent, coefficients)
        predicted = policy.idm.decode_latent(calibrated)
    return F.mse_loss(predicted, batch.target_action_chunk.to(predicted))


def calibration_compensation_ratio(
    nominal_action: torch.Tensor,
    compensated_action: torch.Tensor,
    target_action: torch.Tensor,
) -> torch.Tensor:
    if (
        nominal_action.shape != target_action.shape
        or compensated_action.shape != target_action.shape
    ):
        raise ValueError("compensation ratio tensors must have identical shapes")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (nominal_action, compensated_action, target_action)
    ):
        raise ValueError("compensation ratio tensors must be finite")
    uncompensated_error = F.mse_loss(nominal_action, target_action)
    if bool(uncompensated_error <= 0):
        raise ValueError("uncompensated error must be positive")
    return F.mse_loss(compensated_action, target_action) / uncompensated_error


@torch.no_grad()
def direction_stage_compensation_ratio(
    policy: FADAPlannerIDMPolicy,
    direction_bank: DirectionBank,
    batch: CalibrationRolloutBatch,
    *,
    axis_index: int,
) -> torch.Tensor:
    selected = torch.nonzero(
        (batch.axis_id == axis_index) & ~batch.is_held_out_combination,
        as_tuple=False,
    ).flatten()
    if selected.numel() == 0:
        raise ValueError(f"Stage 1 has no validation rows for axis {axis_index}")
    selected_batch = batch.index_select(selected)
    predicted_future = policy.planner(
        selected_batch.observation_history,
        selected_batch.command,
    )
    latent = policy.idm.encode_latent(
        selected_batch.observation_history,
        selected_batch.action_history,
        predicted_future,
    )
    coefficients = torch.zeros(
        selected.numel(),
        direction_bank.axis_count,
        device=latent.device,
        dtype=latent.dtype,
    )
    coefficients[:, axis_index] = selected_batch.c_true[:, axis_index].to(latent)
    compensated = policy.idm.decode_latent(direction_bank.compose(latent, coefficients))
    return calibration_compensation_ratio(
        selected_batch.nominal_action_chunk.to(compensated),
        compensated,
        selected_batch.target_action_chunk.to(compensated),
    )


def coefficient_stage_loss(
    policy: FADAPlannerIDMPolicy,
    direction_bank: DirectionBank,
    encoder: CoefficientEncoder,
    batch: CalibrationRolloutBatch,
    *,
    action_weight: float = 0.1,
) -> torch.Tensor:
    if action_weight != 0.1:
        raise ValueError("Stage 2 action_weight is fixed at 0.1")
    _validate_stage_batch(policy, batch, direction_bank.axis_count)
    selected = torch.nonzero(~batch.is_held_out_combination, as_tuple=False).flatten()
    if selected.numel() == 0:
        raise ValueError("Stage 2 has no single-axis construction rows")
    batch = batch.index_select(selected)
    with _freeze(policy), _freeze(direction_bank):
        coefficients = encoder(batch.observation_history[:, -30:], batch.action_history[:, -30:])
        predicted_future = policy.planner(batch.observation_history, batch.command)
        latent = policy.idm.encode_latent(
            batch.observation_history, batch.action_history, predicted_future
        )
        calibrated = direction_bank.compose(latent, coefficients)
        predicted = policy.idm.decode_latent(calibrated)
    coefficient_loss = F.mse_loss(coefficients, batch.c_true.to(coefficients))
    action_loss = F.mse_loss(predicted, batch.target_action_chunk.to(predicted))
    return coefficient_loss + 0.1 * action_loss


def coefficient_validation_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("coefficient validation tensors must be matching matrices")
    if not bool(torch.isfinite(predicted).all() and torch.isfinite(target).all()):
        raise ValueError("coefficient validation tensors must be finite")
    return (predicted - target).abs().max()


def validate_encoder_gradients(encoder: CoefficientEncoder) -> None:
    gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
    if (
        not gradients
        or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        or not any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    ):
        raise ValueError("Stage 2 produced no finite nonzero Encoder gradient")


def fit_scale_stage(
    readings: torch.Tensor,
    candidate_scales: torch.Tensor,
    action_errors: torch.Tensor,
) -> tuple[MonotoneScaleCurve, ...]:
    coefficient_scan_grid = torch.linspace(
        -1.0,
        1.0,
        21,
        dtype=readings.dtype,
        device=readings.device,
    ).repeat(len(CALIBRATION_AXIS_NAMES), 1)
    _validate_scale_evidence_tensors(
        coefficient_scan_grid,
        readings,
        candidate_scales,
        action_errors,
    )
    optimal_indices = action_errors.argmin(dim=-1)
    optimal_scales = candidate_scales.to(action_errors)[optimal_indices]
    curves = fit_scale_curve_bank(readings.mean(dim=2), optimal_scales.mean(dim=2))
    for axis_index, curve in enumerate(curves):
        predicted, _ = curve.map(readings[axis_index].reshape(-1))
        expected = optimal_scales[axis_index].reshape(-1)
        residual = torch.sum((expected - predicted) ** 2)
        centered = torch.sum((expected - expected.mean()) ** 2)
        if bool(centered <= 0):
            raise ValueError("Stage 3 R^2 requires non-constant scale evidence")
        r_squared = 1.0 - residual / centered
        if not bool(torch.isfinite(r_squared)) or bool(r_squared < 0.95):
            raise ValueError(f"Stage 3 R^2 {float(r_squared):.6f} is below 0.95")
    return curves


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _identity_payload(identity: CalibrationStageIdentity) -> dict[str, str]:
    return {name: str(getattr(identity, name)) for name in _IDENTITY_FIELDS}


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
        "axis_names": CALIBRATION_AXIS_NAMES,
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
    if tuple(payload.get("axis_names", ())) != CALIBRATION_AXIS_NAMES:
        raise ValueError("calibration stage artifact axis order mismatch")
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
        "axis_count": len(CALIBRATION_AXIS_NAMES),
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
            len(CALIBRATION_AXIS_NAMES),
            policy.config.prediction_horizon,
            policy.config.hidden_dim,
        )
        or not isinstance(normalization_scale, torch.Tensor)
        or normalization_scale.shape != (len(CALIBRATION_AXIS_NAMES),)
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
        or len(ratios) != len(CALIBRATION_AXIS_NAMES)
        or any(not isinstance(value, (int, float)) for value in ratios)
        or any(not math.isfinite(float(value)) for value in ratios)
        or any(float(value) > float(threshold) for value in ratios)
    ):
        raise ValueError("direction stage artifact gate did not pass")
    bank = DirectionBank(**expected_config)
    bank.load_state_dict(dict(direction_state), strict=True)
    bank.requires_grad_(False)
    return bank, digest, payload


def _coefficient_encoder_config(policy: FADAPlannerIDMPolicy) -> dict[str, int]:
    return {
        "state_dim": policy.config.obs_dim,
        "action_dim": policy.config.action_dim,
        "axis_count": len(CALIBRATION_AXIS_NAMES),
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
        "axis_count": len(CALIBRATION_AXIS_NAMES),
        "prediction_horizon": policy.config.prediction_horizon,
        "latent_dim": policy.config.hidden_dim,
    }
    encoder_config = _coefficient_encoder_config(policy)
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
            len(CALIBRATION_AXIS_NAMES),
            policy.config.prediction_horizon,
            policy.config.hidden_dim,
        )
        or not isinstance(normalization_scale, torch.Tensor)
        or normalization_scale.shape != (len(CALIBRATION_AXIS_NAMES),)
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


def _split_stage_batch(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    *,
    training_split_id: int,
    validation_split_id: int,
    stage_name: str,
) -> tuple[CalibrationRolloutBatch, CalibrationRolloutBatch]:
    axis_count = len(CALIBRATION_AXIS_NAMES)
    batch.validate(policy.config, axis_count=axis_count)
    training_rows = torch.nonzero(
        (batch.split_id == training_split_id) & ~batch.is_held_out_combination,
        as_tuple=False,
    ).flatten()
    validation_rows = torch.nonzero(
        (batch.split_id == validation_split_id) & ~batch.is_held_out_combination,
        as_tuple=False,
    ).flatten()
    if training_rows.numel() == 0 or validation_rows.numel() == 0:
        raise ValueError(f"{stage_name} requires non-empty train and validation rows")
    validate_calibration_source_projection(policy, batch)
    return batch.index_select(training_rows), batch.index_select(validation_rows)


def run_direction_stage_training(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    output_path: str | Path,
    identity: CalibrationStageIdentity,
    config: DirectionStageConfig,
) -> DirectionStageResult:
    identity.validate()
    training_batch, validation_batch = _split_stage_batch(
        policy,
        batch,
        training_split_id=config.training_split_id,
        validation_split_id=config.validation_split_id,
        stage_name="Stage 1",
    )
    axis_count = len(CALIBRATION_AXIS_NAMES)
    direction_bank = DirectionBank(
        axis_count=axis_count,
        prediction_horizon=policy.config.prediction_horizon,
        latent_dim=policy.config.hidden_dim,
    ).to(batch.observation_history.device)
    policy_snapshot = _snapshot(policy)
    policy.zero_grad(set_to_none=True)
    ratios: list[float] = []
    try:
        for axis_index in range(axis_count):
            if not bool((training_batch.axis_id == axis_index).any()) or not bool(
                (validation_batch.axis_id == axis_index).any()
            ):
                raise ValueError(
                    f"Stage 1 axis {axis_index} is missing train or validation evidence"
                )
            optimizer = torch.optim.Adam(
                [direction_bank.directions],
                lr=config.learning_rate,
            )
            for _ in range(config.steps_per_axis):
                optimizer.zero_grad(set_to_none=True)
                loss = direction_stage_loss(
                    policy,
                    direction_bank,
                    training_batch,
                    axis_index=axis_index,
                )
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Stage 1 produced a non-finite loss")
                loss.backward()
                bank_snapshot = _snapshot(direction_bank)
                optimizer.step()
                _require_unchanged("Stage 1", policy, policy_snapshot)
                other_axes = (
                    torch.arange(axis_count, device=direction_bank.directions.device) != axis_index
                )
                if not torch.equal(
                    direction_bank.directions[other_axes],
                    bank_snapshot["directions"][other_axes],
                ):
                    raise ValueError("frozen Direction Bank axis mutated during Stage 1")
            direction_bank.normalize_axis_(axis_index)
            ratio = float(
                direction_stage_compensation_ratio(
                    policy,
                    direction_bank,
                    validation_batch,
                    axis_index=axis_index,
                )
            )
            if not torch.isfinite(torch.tensor(ratio)):
                raise ValueError("Stage 1 produced a non-finite compensation ratio")
            if ratio > config.compensation_ratio_threshold:
                raise ValueError(
                    f"Stage 1 axis {axis_index} compensation ratio {ratio:.6f} exceeds "
                    f"{config.compensation_ratio_threshold:.6f}"
                )
            ratios.append(ratio)
    except Exception:
        _restore(policy, policy_snapshot)
        raise
    _require_unchanged("Stage 1", policy, policy_snapshot)
    direction_bank.requires_grad_(False)
    direction_bank.zero_grad(set_to_none=True)
    payload = _stage_envelope(
        policy=policy,
        identity=identity,
        stage=_DIRECTION_STAGE,
        gate={
            "name": "compensation_ratio",
            "threshold": config.compensation_ratio_threshold,
            "result": ratios,
        },
        owners={
            "direction_bank": {
                "config": {
                    "axis_count": axis_count,
                    "prediction_horizon": policy.config.prediction_horizon,
                    "latent_dim": policy.config.hidden_dim,
                },
                "state_dict": _cpu_state_dict(direction_bank),
            }
        },
    )
    artifact_path = _atomic_torch_save(output_path, payload)
    return DirectionStageResult(
        stage=_DIRECTION_STAGE,
        artifact_path=artifact_path,
        artifact_sha256=_sha256_file(artifact_path),
        compensation_ratios=(ratios[0], ratios[1], ratios[2]),
    )


def run_coefficient_stage_training(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    direction_artifact_path: str | Path,
    output_path: str | Path,
    identity: CalibrationStageIdentity,
    config: CoefficientStageConfig,
) -> CoefficientStageResult:
    identity.validate()
    training_batch, validation_batch = _split_stage_batch(
        policy,
        batch,
        training_split_id=config.training_split_id,
        validation_split_id=config.validation_split_id,
        stage_name="Stage 2",
    )
    direction_bank, parent_digest, _ = _load_direction_stage_artifact(
        direction_artifact_path,
        policy=policy,
        identity=identity,
    )
    direction_bank = direction_bank.to(batch.observation_history.device)
    encoder_config = _coefficient_encoder_config(policy)
    encoder = CoefficientEncoder(**encoder_config).to(batch.observation_history.device)
    policy_snapshot = _snapshot(policy)
    direction_snapshot = _snapshot(direction_bank)
    policy.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)
    try:
        for _ in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = coefficient_stage_loss(policy, direction_bank, encoder, training_batch)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Stage 2 produced a non-finite loss")
            loss.backward()
            validate_encoder_gradients(encoder)
            optimizer.step()
            _require_unchanged("Stage 2", policy, policy_snapshot)
            _require_unchanged("Stage 2", direction_bank, direction_snapshot)
    except Exception:
        _restore(policy, policy_snapshot)
        _restore(direction_bank, direction_snapshot)
        raise
    with torch.no_grad():
        coefficient_error = float(
            coefficient_validation_error(
                encoder(
                    validation_batch.observation_history[:, -30:],
                    validation_batch.action_history[:, -30:],
                ),
                validation_batch.c_true,
            )
        )
    if not torch.isfinite(torch.tensor(coefficient_error)):
        raise ValueError("Stage 2 produced a non-finite coefficient error")
    if coefficient_error > config.coefficient_error_threshold:
        raise ValueError(
            f"Stage 2 coefficient error {coefficient_error:.6f} exceeds "
            f"{config.coefficient_error_threshold:.6f}"
        )
    _require_unchanged("Stage 2", policy, policy_snapshot)
    _require_unchanged("Stage 2", direction_bank, direction_snapshot)
    encoder.requires_grad_(False)
    payload = _stage_envelope(
        policy=policy,
        identity=identity,
        stage=_COEFFICIENT_STAGE,
        parent_stage_sha256=parent_digest,
        gate={
            "name": "coefficient_error",
            "threshold": config.coefficient_error_threshold,
            "result": coefficient_error,
        },
        owners={
            "direction_bank": {
                "config": {
                    "axis_count": len(CALIBRATION_AXIS_NAMES),
                    "prediction_horizon": policy.config.prediction_horizon,
                    "latent_dim": policy.config.hidden_dim,
                },
                "state_dict": _cpu_state_dict(direction_bank),
            },
            "coefficient_encoder": {
                "config": encoder_config,
                "state_dict": _cpu_state_dict(encoder),
            },
        },
    )
    artifact_path = _atomic_torch_save(output_path, payload)
    return CoefficientStageResult(
        stage=_COEFFICIENT_STAGE,
        artifact_path=artifact_path,
        artifact_sha256=_sha256_file(artifact_path),
        parent_stage_sha256=parent_digest,
        coefficient_error=coefficient_error,
    )


def _atomic_save_deployment_artifact(
    target_path: str | Path,
    *,
    policy: FADAPlannerIDMPolicy,
    direction_bank: DirectionBank,
    coefficient_encoder: CoefficientEncoder,
    scale_curves: tuple[MonotoneScaleCurve, ...],
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
            metadata=metadata,
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
        nested_temporary.unlink(missing_ok=True)
    return target


def run_scale_stage_fitting(
    policy: FADAPlannerIDMPolicy,
    coefficient_artifact_path: str | Path,
    scale_evidence_path: str | Path,
    output_path: str | Path,
    identity: CalibrationStageIdentity,
) -> ScaleStageResult:
    identity.validate()
    direction_bank, encoder, parent_digest, _ = _load_coefficient_stage_artifact(
        coefficient_artifact_path,
        policy=policy,
        identity=identity,
    )
    scale_payload, scale_digest = _load_exact_torch_payload(scale_evidence_path)
    evidence = _calibration_scale_evidence_from_payload(
        scale_payload,
        _identity_payload(identity),
    )
    curves = fit_scale_stage(
        evidence.readings,
        evidence.candidate_scales,
        evidence.action_errors,
    )
    artifact_path = _atomic_save_deployment_artifact(
        output_path,
        policy=policy,
        direction_bank=direction_bank,
        coefficient_encoder=encoder,
        scale_curves=curves,
        metadata={
            **_identity_payload(identity),
            "stage": "complete",
            "parent_stage_sha256": parent_digest,
            "scale_evidence_sha256": scale_digest,
        },
    )
    return ScaleStageResult(
        stage="complete",
        artifact_path=artifact_path,
        artifact_sha256=_sha256_file(artifact_path),
        parent_stage_sha256=parent_digest,
        scale_evidence_sha256=scale_digest,
    )


def run_serial_calibration_training(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    *,
    output_dir: str | Path,
    source_tracker_sha256: str,
    dataset_sha256: str,
    split_sha256: str,
    axis_catalog_version: str,
    scale_evidence: CalibrationScaleEvidence | None = None,
    scale_evidence_path: str | Path | None = None,
    config: SerialCalibrationConfig = SerialCalibrationConfig(),
) -> dict[str, object]:
    """Compose S1/S2/S3 through the same persisted boundaries as independent runs."""

    identity = CalibrationStageIdentity(
        source_tracker_sha256=source_tracker_sha256,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        axis_catalog_version=axis_catalog_version,
    ).validate()
    if (scale_evidence is None) == (scale_evidence_path is None):
        raise ValueError("serial calibration requires exactly one typed scale evidence source")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    direction_result = run_direction_stage_training(
        policy,
        batch,
        output / "stage1_direction_frozen.pt",
        identity,
        config.direction_stage(),
    )
    coefficient_result = run_coefficient_stage_training(
        policy,
        batch,
        direction_result.artifact_path,
        output / "stage2_coefficient_frozen.pt",
        identity,
        config.coefficient_stage(),
    )
    if scale_evidence is not None:
        scale_path = save_calibration_scale_evidence(
            output / "scale_evidence.pt",
            scale_evidence,
        )
    else:
        assert scale_evidence_path is not None
        scale_path = Path(scale_evidence_path).expanduser().resolve()
    scale_result = run_scale_stage_fitting(
        policy,
        coefficient_result.artifact_path,
        scale_path,
        output / "calibration_artifact.pt",
        identity,
    )
    return {
        "stage": "complete",
        "artifact_path": str(scale_result.artifact_path),
        "coefficient_error": coefficient_result.coefficient_error,
        "axis_count": len(CALIBRATION_AXIS_NAMES),
        "direction_stage": direction_result,
        "coefficient_stage": coefficient_result,
        "scale_stage": scale_result,
    }
