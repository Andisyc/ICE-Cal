from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

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

CALIBRATION_CHECKPOINT_SCHEMA = "unilab_fada_calibration_checkpoint_v1"
CALIBRATION_SCALE_EVIDENCE_SCHEMA = "unilab_fada_calibration_scale_evidence_v1"
_STAGES = {"prepared", "direction_frozen", "coefficient_frozen", "complete"}
_IDENTITY_FIELDS = (
    "source_tracker_sha256",
    "dataset_sha256",
    "split_sha256",
    "axis_catalog_version",
)


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
        if self.learning_rate <= 0:
            raise ValueError("serial calibration learning_rate must be positive")
        if not 0 < self.compensation_ratio_threshold < 1:
            raise ValueError("compensation_ratio_threshold must be in (0,1)")
        if not 0 < self.coefficient_error_threshold < 1:
            raise ValueError("coefficient_error_threshold must be in (0,1)")
        if self.training_split_id == self.validation_split_id:
            raise ValueError("training and validation split IDs must differ")


@dataclass(frozen=True)
class CalibrationScaleEvidence:
    readings: torch.Tensor
    candidate_scales: torch.Tensor
    action_errors: torch.Tensor
    metadata: Mapping[str, str]

    def validate(self) -> CalibrationScaleEvidence:
        _validate_scale_evidence_tensors(
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
    readings: torch.Tensor,
    candidate_scales: torch.Tensor,
    action_errors: torch.Tensor,
) -> None:
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
        torch.isfinite(readings).all()
        and torch.isfinite(candidate_scales).all()
        and torch.isfinite(action_errors).all()
    ):
        raise ValueError("Stage 3 evidence must be finite")


def save_calibration_scale_evidence(
    path: str | Path,
    evidence: CalibrationScaleEvidence,
) -> Path:
    evidence.validate()
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    torch.save(
        {
            "schema_version": CALIBRATION_SCALE_EVIDENCE_SCHEMA,
            "method_contract_id": CALIBRATION_METHOD_CONTRACT_ID,
            "training_contract_id": CALIBRATION_TRAINING_CONTRACT_ID,
            "axis_names": CALIBRATION_AXIS_NAMES,
            "readings": evidence.readings.detach().cpu(),
            "candidate_scales": evidence.candidate_scales.detach().cpu(),
            "action_errors": evidence.action_errors.detach().cpu(),
            "metadata": dict(evidence.metadata),
        },
        temporary,
    )
    temporary.replace(target)
    return target


def load_calibration_scale_evidence(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, str],
) -> CalibrationScaleEvidence:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
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
        payload.get("readings"),
        payload.get("candidate_scales"),
        payload.get("action_errors"),
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError("calibration scale evidence tensor fields are missing")
    readings, candidate_scales, action_errors = tensors
    assert isinstance(readings, torch.Tensor)
    assert isinstance(candidate_scales, torch.Tensor)
    assert isinstance(action_errors, torch.Tensor)
    return CalibrationScaleEvidence(
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
    _validate_scale_evidence_tensors(readings, candidate_scales, action_errors)
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


def save_calibration_training_checkpoint(
    path: str | Path,
    *,
    policy: FADAPlannerIDMPolicy,
    direction_bank: DirectionBank,
    coefficient_encoder: CoefficientEncoder,
    stage: str,
    metadata: dict[str, str],
    scale_curves: tuple[MonotoneScaleCurve, ...] | None = None,
) -> str:
    if stage not in _STAGES:
        raise ValueError(f"unknown calibration stage: {stage}")
    required_metadata = {
        "source_tracker_sha256",
        "dataset_sha256",
        "split_sha256",
        "axis_catalog_version",
    }
    if any(not metadata.get(name) for name in required_metadata):
        raise ValueError(
            "calibration checkpoint requires source, dataset, split, and catalog identity"
        )
    if metadata["axis_catalog_version"] != CALIBRATION_AXIS_CATALOG_VERSION:
        raise ValueError("calibration checkpoint axis catalog mismatch")
    if stage != "prepared":
        norms = direction_bank.directions.detach().flatten(1).norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-6):
            raise ValueError("frozen calibration checkpoint requires normalized directions")
    if stage == "complete" and (
        scale_curves is None or len(scale_curves) != direction_bank.axis_count
    ):
        raise ValueError("complete calibration checkpoint requires every scale curve")
    if stage != "complete" and scale_curves is not None:
        raise ValueError("scale curves are only valid in a complete calibration checkpoint")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    torch.save(
        {
            "schema_version": CALIBRATION_CHECKPOINT_SCHEMA,
            "method_contract_id": CALIBRATION_METHOD_CONTRACT_ID,
            "training_contract_id": CALIBRATION_TRAINING_CONTRACT_ID,
            "architecture": policy.config.__dict__,
            "axis_names": CALIBRATION_AXIS_NAMES,
            "stage": stage,
            "planner_state_dict": policy.planner.state_dict(),
            "idm_state_dict": policy.idm.state_dict(),
            "direction_bank_state_dict": direction_bank.state_dict(),
            "coefficient_encoder_state_dict": coefficient_encoder.state_dict(),
            "scale_curves": None
            if scale_curves is None
            else [
                {"x": curve.x, "y": curve.y, "slopes": curve.slopes, "kind": curve.kind}
                for curve in scale_curves
            ],
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(target)
    return str(target)


def load_calibration_training_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    direction_bank: DirectionBank,
    coefficient_encoder: CoefficientEncoder,
    *,
    expected_metadata: dict[str, str],
    expected_stage: str,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CALIBRATION_CHECKPOINT_SCHEMA
    ):
        raise ValueError("unsupported calibration checkpoint schema")
    if payload.get("method_contract_id") != CALIBRATION_METHOD_CONTRACT_ID:
        raise ValueError("calibration checkpoint method Contract mismatch")
    if payload.get("training_contract_id") != CALIBRATION_TRAINING_CONTRACT_ID:
        raise ValueError("calibration checkpoint training Contract mismatch")
    if payload.get("architecture") != policy.config.__dict__:
        raise ValueError("calibration checkpoint architecture mismatch")
    if payload.get("stage") not in _STAGES:
        raise ValueError("calibration checkpoint has invalid stage ordering")
    if payload.get("stage") != expected_stage:
        raise ValueError(
            f"calibration checkpoint stage mismatch: expected={expected_stage} "
            f"observed={payload.get('stage')}"
        )
    if expected_stage == "complete" and not isinstance(payload.get("scale_curves"), list):
        raise ValueError("complete calibration checkpoint is missing scale curves")
    if expected_stage != "complete" and payload.get("scale_curves") is not None:
        raise ValueError("incomplete calibration checkpoint cannot contain scale curves")
    if expected_stage == "complete":
        if len(payload["scale_curves"]) != len(CALIBRATION_AXIS_NAMES):
            raise ValueError("complete calibration checkpoint scale curve count mismatch")
        for curve in payload["scale_curves"]:
            _validate_scale_curve_payload(curve)
    if tuple(payload.get("axis_names", ())) != CALIBRATION_AXIS_NAMES:
        raise ValueError("calibration checkpoint axis catalog mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or any(
        metadata.get(name) != expected_metadata.get(name)
        for name in (
            "source_tracker_sha256",
            "dataset_sha256",
            "split_sha256",
            "axis_catalog_version",
        )
    ):
        raise ValueError("calibration checkpoint metadata identity mismatch")
    _validate_finite_state_tree("calibration checkpoint", payload)
    if expected_stage != "prepared":
        direction_state = payload.get("direction_bank_state_dict")
        directions = (
            direction_state.get("directions") if isinstance(direction_state, dict) else None
        )
        if not isinstance(directions, torch.Tensor) or not torch.allclose(
            directions.flatten(1).norm(dim=1),
            torch.ones(directions.shape[0]),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("frozen calibration checkpoint directions are not normalized")
    owners: tuple[torch.nn.Module, ...] = (
        policy.planner,
        policy.idm,
        direction_bank,
        coefficient_encoder,
    )
    state_names = (
        "planner_state_dict",
        "idm_state_dict",
        "direction_bank_state_dict",
        "coefficient_encoder_state_dict",
    )
    snapshots = tuple(
        {name: value.detach().clone() for name, value in owner.state_dict().items()}
        for owner in owners
    )
    try:
        for owner, state_name in zip(owners, state_names, strict=True):
            owner.load_state_dict(payload[state_name], strict=True)
    except Exception:
        for owner, snapshot in zip(owners, snapshots, strict=True):
            owner.load_state_dict(snapshot, strict=True)
        raise
    return payload


def run_serial_calibration_training(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    *,
    output_dir: str | Path,
    source_tracker_sha256: str,
    dataset_sha256: str,
    split_sha256: str,
    axis_catalog_version: str,
    scale_evidence: CalibrationScaleEvidence,
    config: SerialCalibrationConfig = SerialCalibrationConfig(),
) -> dict[str, object]:
    """Run S1/S2/S3 as one serial transaction over already-collected labeled data."""

    axis_count = int(batch.c_true.shape[-1])
    batch.validate(policy.config, axis_count=axis_count)
    training_rows = torch.nonzero(
        (batch.split_id == config.training_split_id) & ~batch.is_held_out_combination,
        as_tuple=False,
    ).flatten()
    validation_rows = torch.nonzero(
        (batch.split_id == config.validation_split_id) & ~batch.is_held_out_combination,
        as_tuple=False,
    ).flatten()
    if training_rows.numel() == 0 or validation_rows.numel() == 0:
        raise ValueError("serial calibration requires non-empty train and validation rows")
    validate_calibration_source_projection(policy, batch)
    checkpoint_metadata = {
        "source_tracker_sha256": source_tracker_sha256,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "axis_catalog_version": axis_catalog_version,
    }
    scale_evidence.validate()
    if any(
        scale_evidence.metadata.get(name) != value for name, value in checkpoint_metadata.items()
    ):
        raise ValueError("Stage 3 scale evidence metadata identity mismatch")
    training_batch = batch.index_select(training_rows)
    validation_batch = batch.index_select(validation_rows)
    direction_bank = DirectionBank(
        axis_count=axis_count,
        prediction_horizon=policy.config.prediction_horizon,
        latent_dim=policy.config.hidden_dim,
    ).to(batch.observation_history.device)
    encoder = CoefficientEncoder(
        state_dim=policy.config.obs_dim,
        action_dim=policy.config.action_dim,
        axis_count=axis_count,
    ).to(batch.observation_history.device)
    policy.zero_grad(set_to_none=True)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Stage 1: one optimizer transaction per axis; the loss filters its own axis rows.
    for axis_index in range(axis_count):
        if not bool((training_batch.axis_id == axis_index).any()) or not bool(
            (validation_batch.axis_id == axis_index).any()
        ):
            raise ValueError(f"Stage 1 axis {axis_index} is missing train or validation evidence")
        optimizer = torch.optim.Adam([direction_bank.directions], lr=config.learning_rate)
        for _ in range(config.stage1_steps_per_axis):
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
            owner_snapshots = (_snapshot(policy), _snapshot(direction_bank), _snapshot(encoder))
            try:
                optimizer.step()
                _require_unchanged("Stage 1", policy, owner_snapshots[0])
                _require_unchanged("Stage 1", encoder, owner_snapshots[2])
                before_directions = owner_snapshots[1]["directions"]
                other_axes = (
                    torch.arange(axis_count, device=direction_bank.directions.device) != axis_index
                )
                if not torch.equal(
                    direction_bank.directions[other_axes], before_directions[other_axes]
                ):
                    raise ValueError("frozen Direction Bank axis mutated during Stage 1")
            except Exception:
                _restore(policy, owner_snapshots[0])
                _restore(direction_bank, owner_snapshots[1])
                _restore(encoder, owner_snapshots[2])
                raise
        direction_bank.normalize_axis_(axis_index)
        ratio = direction_stage_compensation_ratio(
            policy,
            direction_bank,
            validation_batch,
            axis_index=axis_index,
        )
        if bool(ratio > config.compensation_ratio_threshold):
            raise ValueError(
                f"Stage 1 axis {axis_index} compensation ratio {float(ratio):.6f} exceeds "
                f"{config.compensation_ratio_threshold:.6f}"
            )
    direction_bank.requires_grad_(False)
    direction_bank.zero_grad(set_to_none=True)
    save_calibration_training_checkpoint(
        str(output / "stage1_direction_frozen.pt"),
        policy=policy,
        direction_bank=direction_bank,
        coefficient_encoder=encoder,
        stage="direction_frozen",
        metadata=checkpoint_metadata,
    )

    # Stage 2: the only optimizer parameter is the Coefficient Encoder.
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)
    for _ in range(config.stage2_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = coefficient_stage_loss(policy, direction_bank, encoder, training_batch)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Stage 2 produced a non-finite loss")
        loss.backward()
        validate_encoder_gradients(encoder)
        owner_snapshots = (_snapshot(policy), _snapshot(direction_bank), _snapshot(encoder))
        try:
            optimizer.step()
            _require_unchanged("Stage 2", policy, owner_snapshots[0])
            _require_unchanged("Stage 2", direction_bank, owner_snapshots[1])
        except Exception:
            _restore(policy, owner_snapshots[0])
            _restore(direction_bank, owner_snapshots[1])
            _restore(encoder, owner_snapshots[2])
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
    encoder.requires_grad_(False)
    if coefficient_error > config.coefficient_error_threshold:
        raise ValueError(
            f"Stage 2 coefficient error {coefficient_error:.6f} exceeds "
            f"{config.coefficient_error_threshold:.6f}"
        )
    save_calibration_training_checkpoint(
        str(output / "stage2_coefficient_frozen.pt"),
        policy=policy,
        direction_bank=direction_bank,
        coefficient_encoder=encoder,
        stage="coefficient_frozen",
        metadata=checkpoint_metadata,
    )

    # Stage 3: fit curve artifacts only; no optimizer is constructed here.
    curves = fit_scale_stage(
        scale_evidence.readings,
        scale_evidence.candidate_scales,
        scale_evidence.action_errors,
    )
    artifact_path = save_calibration_artifact(
        output / "calibration_artifact.pt",
        config=policy.config,
        direction_bank=direction_bank,
        coefficient_encoder=encoder,
        scale_curves=curves,
        metadata={
            **checkpoint_metadata,
            "stage": "complete",
        },
    )
    save_calibration_training_checkpoint(
        str(output / "stage3_complete.pt"),
        policy=policy,
        direction_bank=direction_bank,
        coefficient_encoder=encoder,
        stage="complete",
        metadata=checkpoint_metadata,
        scale_curves=curves,
    )
    return {
        "stage": "complete",
        "artifact_path": str(artifact_path),
        "coefficient_error": coefficient_error,
        "axis_count": axis_count,
    }
