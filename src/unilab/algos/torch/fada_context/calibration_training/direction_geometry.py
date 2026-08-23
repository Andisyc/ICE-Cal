from __future__ import annotations

from dataclasses import dataclass

import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import CalibrationRolloutBatch
from unilab.algos.torch.fada_context.calibration_training.lifecycle import (
    _freeze,
    _require_unchanged,
    _snapshot,
    _split_stage_batch,
)
from unilab.algos.torch.fada_context.calibration_training.stage1 import (
    calibration_compensation_ratio,
)
from unilab.algos.torch.fada_context.calibration_training.types import (
    _COMPENSATION_RATIO_LIMIT,
    CalibrationStageIdentity,
    DirectionGeometryAxisReport,
    DirectionGeometryConfig,
    DirectionGeometrySplitReport,
)


@dataclass(frozen=True)
class _GeometryRows:
    latent: torch.Tensor
    coefficients: torch.Tensor
    nominal_action: torch.Tensor
    target_action: torch.Tensor
    split_id: int
    excluded_zero_coefficient_count: int
    excluded_zero_target_error_count: int


def _first_action_row_mse(
    action_chunk: torch.Tensor,
    target_action_chunk: torch.Tensor,
) -> torch.Tensor:
    if action_chunk.shape != target_action_chunk.shape or action_chunk.ndim != 3:
        raise ValueError("direction geometry action chunks must be matching rank-3 tensors")
    values = (action_chunk[:, 0] - target_action_chunk[:, 0]).square().mean(dim=1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("direction geometry first-action errors must be finite")
    return values


def _decode_with_first_token_direction(
    policy: FADAPlannerIDMPolicy,
    rows: _GeometryRows,
    directions: torch.Tensor,
) -> torch.Tensor:
    if directions.ndim == 1:
        directions = directions.unsqueeze(0).expand(rows.latent.shape[0], -1)
    if directions.shape != (rows.latent.shape[0], rows.latent.shape[2]):
        raise ValueError("direction geometry directions must be [row, latent_dim]")
    first_token = rows.latent[:, :1] + (
        rows.coefficients[:, None, None] * directions[:, None].to(rows.latent)
    )
    return policy.idm.decode_latent(torch.cat((first_token, rows.latent[:, 1:]), dim=1))


@torch.no_grad()
def _compensation_ratios(
    policy: FADAPlannerIDMPolicy,
    rows: _GeometryRows,
    directions: torch.Tensor,
) -> torch.Tensor:
    predicted = _decode_with_first_token_direction(policy, rows, directions)
    baseline = _first_action_row_mse(rows.nominal_action, rows.target_action)
    if bool((baseline <= 0).any()):
        raise ValueError("direction geometry uncompensated first-action error must be positive")
    return _first_action_row_mse(predicted, rows.target_action) / baseline


@torch.no_grad()
def _shared_compensation_ratio(
    policy: FADAPlannerIDMPolicy,
    rows: _GeometryRows,
    direction: torch.Tensor,
) -> torch.Tensor:
    predicted = _decode_with_first_token_direction(policy, rows, direction)
    return calibration_compensation_ratio(
        rows.nominal_action[:, 0],
        predicted[:, 0],
        rows.target_action[:, 0],
    )


@torch.no_grad()
def _canonical_directions(
    policy: FADAPlannerIDMPolicy,
    rows: _GeometryRows,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_head = policy.idm.action_head
    if not isinstance(action_head, torch.nn.Linear):
        raise TypeError("direction geometry requires the linear Tracker action_head")
    weight = action_head.weight.to(rows.latent)
    if weight.shape != (rows.target_action.shape[2], rows.latent.shape[2]):
        raise ValueError("direction geometry action_head shape does not match the dataset")
    pseudoinverse = torch.linalg.pinv(weight)
    decoded_nominal = policy.idm.decode_latent(rows.latent)[:, 0]
    target_delta = rows.target_action[:, 0].to(decoded_nominal) - decoded_nominal
    canonical_action_delta = target_delta / rows.coefficients[:, None]
    individual = canonical_action_delta @ pseudoinverse.T
    shared_action_delta = (
        (rows.coefficients[:, None] * target_delta).sum(dim=0)
        / rows.coefficients.square().sum()
    )
    shared = shared_action_delta @ pseudoinverse.T
    if not bool(torch.isfinite(individual).all() and torch.isfinite(shared).all()):
        raise ValueError("direction geometry analytic solution must be finite")
    return shared, individual


@torch.no_grad()
def _prepare_geometry_rows(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    *,
    axis_index: int,
    split_id: int,
    minimum_abs_coefficient: float,
) -> _GeometryRows:
    selected = torch.nonzero(
        (batch.axis_id == axis_index) & ~batch.is_held_out_combination,
        as_tuple=False,
    ).flatten()
    if selected.numel() == 0:
        raise ValueError(f"direction geometry axis {axis_index} has no split {split_id} rows")
    selected_batch = batch.index_select(selected)
    coefficients = selected_batch.c_true[:, axis_index]
    coefficient_mask = coefficients.abs() > minimum_abs_coefficient
    excluded_zero_coefficient_count = int((~coefficient_mask).sum())
    selected_batch = selected_batch.index_select(torch.nonzero(coefficient_mask).flatten())
    coefficients = coefficients[coefficient_mask]
    if coefficients.numel() == 0:
        raise ValueError(f"direction geometry axis {axis_index} has no identifiable coefficients")

    target_error = _first_action_row_mse(
        selected_batch.nominal_action_chunk,
        selected_batch.target_action_chunk,
    )
    target_mask = target_error > 0
    excluded_zero_target_error_count = int((~target_mask).sum())
    selected_batch = selected_batch.index_select(torch.nonzero(target_mask).flatten())
    coefficients = coefficients[target_mask]
    if coefficients.numel() < 2:
        raise ValueError(
            f"direction geometry axis {axis_index} split {split_id} requires "
            "at least two identifiable rows"
        )

    predicted_future = policy.planner(
        selected_batch.observation_history,
        selected_batch.command,
    )
    latent = policy.idm.encode_latent(
        selected_batch.observation_history,
        selected_batch.action_history,
        predicted_future,
    )
    return _GeometryRows(
        latent=latent.detach(),
        coefficients=coefficients.to(latent).detach(),
        nominal_action=selected_batch.nominal_action_chunk.to(latent).detach(),
        target_action=selected_batch.target_action_chunk.to(latent).detach(),
        split_id=split_id,
        excluded_zero_coefficient_count=excluded_zero_coefficient_count,
        excluded_zero_target_error_count=excluded_zero_target_error_count,
    )


def summarize_direction_geometry(
    directions: torch.Tensor,
    individual_ratios: torch.Tensor,
    *,
    split_id: int,
    excluded_zero_coefficient_count: int,
    excluded_zero_target_error_count: int,
) -> DirectionGeometrySplitReport:
    if directions.ndim != 2 or individual_ratios.shape != (directions.shape[0],):
        raise ValueError("direction geometry summary expects [row, latent] and [row] tensors")
    if not bool(torch.isfinite(directions).all() and torch.isfinite(individual_ratios).all()):
        raise ValueError("direction geometry summary tensors must be finite")
    if bool((individual_ratios < 0).any()):
        raise ValueError("direction geometry compensation ratios must be non-negative")
    if min(excluded_zero_coefficient_count, excluded_zero_target_error_count) < 0:
        raise ValueError("direction geometry exclusion counts must be non-negative")

    norms = directions.norm(dim=1)
    nonzero = norms > 0
    zero_direction_count = int((~nonzero).sum())
    if int(nonzero.sum()) < 2:
        raise ValueError("direction geometry requires at least two nonzero fitted directions")
    normalized = directions[nonzero] / norms[nonzero, None]
    nonzero_norms = norms[nonzero]
    _, singular_values, right_vectors = torch.linalg.svd(normalized, full_matrices=False)
    energy = singular_values.square()
    top1_energy_fraction = energy[0] / energy.sum()
    consensus = right_vectors[0]
    if bool((normalized.mean(dim=0) * consensus).sum() < 0):
        consensus = -consensus
    cosines = normalized @ consensus
    norm_p10 = torch.quantile(nonzero_norms, 0.1)
    norm_p90 = torch.quantile(nonzero_norms, 0.9)
    norm_p90_p10_ratio = norm_p90 / norm_p10

    values = torch.cat(
        (
            individual_ratios,
            norms,
            top1_energy_fraction.reshape(1),
            cosines,
            norm_p90_p10_ratio.reshape(1),
        )
    )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("direction geometry summary produced a non-finite metric")
    return DirectionGeometrySplitReport(
        split_id=split_id,
        sample_count=int(directions.shape[0]),
        excluded_zero_coefficient_count=excluded_zero_coefficient_count,
        excluded_zero_target_error_count=excluded_zero_target_error_count,
        zero_direction_count=zero_direction_count,
        individual_ratio_median=float(torch.quantile(individual_ratios, 0.5)),
        individual_ratio_p90=float(torch.quantile(individual_ratios, 0.9)),
        individual_ratio_max=float(individual_ratios.max()),
        individual_gate_fraction=float(
            (individual_ratios <= _COMPENSATION_RATIO_LIMIT).float().mean()
        ),
        top1_energy_fraction=float(top1_energy_fraction),
        cosine_to_consensus_mean=float(cosines.mean()),
        cosine_to_consensus_p10=float(torch.quantile(cosines, 0.1)),
        opposing_direction_fraction=float((cosines < 0).float().mean()),
        direction_norm_p10=float(norm_p10),
        direction_norm_median=float(torch.quantile(nonzero_norms, 0.5)),
        direction_norm_p90=float(norm_p90),
        direction_norm_p90_p10_ratio=float(norm_p90_p10_ratio),
    )


def diagnose_direction_geometry(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    identity: CalibrationStageIdentity,
    config: DirectionGeometryConfig,
) -> tuple[DirectionGeometryAxisReport, ...]:
    identity.validate()
    training_batch, validation_batch = _split_stage_batch(
        policy,
        batch,
        training_split_id=config.training_split_id,
        validation_split_id=config.validation_split_id,
        stage_name="Stage 1 direction geometry diagnostic",
        axis_count=identity.axis_spec.axis_count,
    )
    policy_snapshot = _snapshot(policy)
    reports: list[DirectionGeometryAxisReport] = []
    try:
        with _freeze(policy):
            for axis_index in range(identity.axis_spec.axis_count):
                training_rows = _prepare_geometry_rows(
                    policy,
                    training_batch,
                    axis_index=axis_index,
                    split_id=config.training_split_id,
                    minimum_abs_coefficient=config.minimum_abs_coefficient,
                )
                validation_rows = _prepare_geometry_rows(
                    policy,
                    validation_batch,
                    axis_index=axis_index,
                    split_id=config.validation_split_id,
                    minimum_abs_coefficient=config.minimum_abs_coefficient,
                )
                shared_direction, training_directions = _canonical_directions(
                    policy, training_rows
                )
                _, validation_directions = _canonical_directions(policy, validation_rows)
                training_ratios = _compensation_ratios(
                    policy,
                    training_rows,
                    training_directions,
                )
                validation_ratios = _compensation_ratios(
                    policy,
                    validation_rows,
                    validation_directions,
                )
                reports.append(
                    DirectionGeometryAxisReport(
                        axis_index=axis_index,
                        supervision_scope="executed_first_action",
                        solver="linear_decoder_minimum_norm",
                        shared_training_ratio=float(
                            _shared_compensation_ratio(
                                policy, training_rows, shared_direction
                            )
                        ),
                        shared_validation_ratio=float(
                            _shared_compensation_ratio(
                                policy, validation_rows, shared_direction
                            )
                        ),
                        training=summarize_direction_geometry(
                            training_directions,
                            training_ratios,
                            split_id=config.training_split_id,
                            excluded_zero_coefficient_count=(
                                training_rows.excluded_zero_coefficient_count
                            ),
                            excluded_zero_target_error_count=(
                                training_rows.excluded_zero_target_error_count
                            ),
                        ),
                        validation=summarize_direction_geometry(
                            validation_directions,
                            validation_ratios,
                            split_id=config.validation_split_id,
                            excluded_zero_coefficient_count=(
                                validation_rows.excluded_zero_coefficient_count
                            ),
                            excluded_zero_target_error_count=(
                                validation_rows.excluded_zero_target_error_count
                            ),
                        ),
                    )
                )
    finally:
        _require_unchanged("Stage 1 direction geometry diagnostic", policy, policy_snapshot)
    return tuple(reports)
