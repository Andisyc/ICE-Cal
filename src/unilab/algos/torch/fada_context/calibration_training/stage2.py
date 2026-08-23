from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import (
    CalibrationRolloutBatch,
    CoefficientEncoder,
    DirectionBank,
)
from unilab.algos.torch.fada_context.calibration_training.io import (
    _atomic_torch_save,
    _coefficient_encoder_config,
    _cpu_state_dict,
    _load_direction_stage_artifact,
    _sha256_file,
    _stage_envelope,
)
from unilab.algos.torch.fada_context.calibration_training.lifecycle import (
    _freeze,
    _require_unchanged,
    _restore,
    _snapshot,
    _split_stage_batch,
    _validate_stage_batch,
)
from unilab.algos.torch.fada_context.calibration_training.types import (
    _COEFFICIENT_STAGE,
    CalibrationStageIdentity,
    CoefficientStageConfig,
    CoefficientStageResult,
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
        axis_count=identity.axis_spec.axis_count,
    )
    direction_bank, parent_digest, _ = _load_direction_stage_artifact(
        direction_artifact_path,
        policy=policy,
        identity=identity,
    )
    direction_bank = direction_bank.to(batch.observation_history.device)
    encoder_config = _coefficient_encoder_config(policy, identity.axis_spec.axis_count)
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
                    "axis_count": identity.axis_spec.axis_count,
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
