from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import (
    CalibrationRolloutBatch,
    DirectionBank,
)
from unilab.algos.torch.fada_context.calibration_training.io import (
    _atomic_torch_save,
    _cpu_state_dict,
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
    _DIRECTION_STAGE,
    CalibrationStageIdentity,
    DirectionStageConfig,
    DirectionStageResult,
)


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
        axis_count=identity.axis_spec.axis_count,
    )
    axis_count = identity.axis_spec.axis_count
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
        compensation_ratios=tuple(ratios),
    )
