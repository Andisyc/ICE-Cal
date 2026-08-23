from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from unilab.algos.torch.distill.fada import FADAArchitectureConfig, PlannerIDMOutput
from unilab.algos.torch.fada_context.calibration import CalibrationRolloutBatch

CALIBRATION_UPPER_BOUND_SCHEMA = "unilab_fada_calibration_full_finetune_upper_bound_v1"


@dataclass(frozen=True)
class CalibrationFullFinetuneUpperBound:
    action_chunk: torch.Tensor
    rollout_id: torch.Tensor


def save_calibration_full_finetune_upper_bound(
    path: str | Path,
    upper_bound: CalibrationFullFinetuneUpperBound,
    *,
    metadata: dict[str, str],
) -> Path:
    required = ("source_tracker_sha256", "dataset_sha256", "split_sha256")
    if any(not metadata.get(name) for name in required):
        raise ValueError("full-finetune upper-bound metadata identity is incomplete")
    if (
        upper_bound.rollout_id.ndim != 1
        or upper_bound.rollout_id.dtype != torch.int64
        or upper_bound.action_chunk.shape[0] != upper_bound.rollout_id.shape[0]
        or not bool(torch.isfinite(upper_bound.action_chunk).all())
    ):
        raise ValueError("full-finetune upper-bound payload is malformed")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    torch.save(
        {
            "schema_version": CALIBRATION_UPPER_BOUND_SCHEMA,
            "action_chunk": upper_bound.action_chunk.detach().cpu(),
            "rollout_id": upper_bound.rollout_id.detach().cpu(),
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(target)
    return target


def load_calibration_full_finetune_upper_bound(
    path: str | Path,
    *,
    expected_metadata: dict[str, str],
) -> CalibrationFullFinetuneUpperBound:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CALIBRATION_UPPER_BOUND_SCHEMA
    ):
        raise ValueError("unsupported calibration full-finetune upper-bound schema")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or any(
        metadata.get(name) != value for name, value in expected_metadata.items()
    ):
        raise ValueError("full-finetune upper-bound metadata identity mismatch")
    action_chunk = payload.get("action_chunk")
    rollout_id = payload.get("rollout_id")
    if (
        not isinstance(action_chunk, torch.Tensor)
        or not isinstance(rollout_id, torch.Tensor)
        or rollout_id.ndim != 1
        or rollout_id.dtype != torch.int64
        or action_chunk.shape[0] != rollout_id.shape[0]
        or not bool(torch.isfinite(action_chunk).all())
    ):
        raise ValueError("full-finetune upper-bound payload is malformed")
    return CalibrationFullFinetuneUpperBound(action_chunk=action_chunk, rollout_id=rollout_id)


class _NominalPolicy(Protocol):
    config: FADAArchitectureConfig

    def __call__(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> PlannerIDMOutput: ...


class _CalibratedPolicy(_NominalPolicy, Protocol):
    def reconstruct_with_coefficients(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> PlannerIDMOutput: ...


def evaluate_held_out_calibration(
    nominal_policy: _NominalPolicy,
    calibrated_policy: _CalibratedPolicy,
    batch: CalibrationRolloutBatch,
    *,
    full_finetune: CalibrationFullFinetuneUpperBound,
) -> dict[str, object]:
    """Compare four first-action routes on rows excluded from construction."""

    if nominal_policy.config != calibrated_policy.config:
        raise ValueError("nominal and calibrated policy architectures must match")
    axis_count = int(batch.c_true.shape[-1])
    batch.validate(nominal_policy.config, axis_count=axis_count)
    if axis_count < 2:
        raise ValueError("held-out combination evaluation is not applicable to a one-axis run")
    if full_finetune.action_chunk.shape != batch.target_action_chunk.shape:
        raise ValueError("full-finetune action chunks must bind every dataset row")
    if not torch.equal(full_finetune.rollout_id.to(batch.rollout_id), batch.rollout_id):
        raise ValueError("full-finetune rollout identity does not match the dataset")
    if not bool(torch.isfinite(full_finetune.action_chunk).all()):
        raise ValueError("full-finetune action chunks must be finite")
    held_out = torch.nonzero(batch.is_held_out_combination, as_tuple=False).flatten()
    if held_out.numel() < 2:
        raise ValueError("evaluation requires at least two held-out combination rows")
    selected = batch.index_select(held_out)
    full_finetune_action = full_finetune.action_chunk.index_select(0, held_out)
    with torch.no_grad():
        nominal = nominal_policy(
            selected.observation_history,
            selected.action_history,
            selected.command,
        ).action_chunk
        calibrated = calibrated_policy(
            selected.observation_history,
            selected.action_history,
            selected.command,
        ).action_chunk
        permutation = torch.arange(held_out.numel() - 1, -1, -1, device=held_out.device)
        shuffled = calibrated_policy.reconstruct_with_coefficients(
            selected.observation_history,
            selected.action_history,
            selected.command,
            selected.c_true.index_select(0, permutation),
        ).action_chunk
    expected_shape = selected.target_action_chunk.shape
    for name, value in (
        ("nominal", nominal),
        ("calibrated", calibrated),
        ("shuffled", shuffled),
        ("full-finetune", full_finetune_action),
    ):
        if value.shape != expected_shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} evaluation route returned an invalid Action chunk")
    target_first = selected.target_action_chunk[:, 0]
    first_actions = {
        "nominal": nominal[:, 0],
        "calibrated": calibrated[:, 0],
        "shuffled_coefficient": shuffled[:, 0],
        "full_finetune_upper_bound": full_finetune_action[:, 0],
    }
    mse = {
        name: float((value - target_first).square().mean()) for name, value in first_actions.items()
    }
    predicted_correction = first_actions["calibrated"] - first_actions["nominal"]
    realized_correction = target_first - first_actions["nominal"]
    return {
        "schema": "unilab_fada_calibration_held_out_evaluation_v1",
        "held_out_rows": int(held_out.numel()),
        "executed_action_index": 0,
        "first_action_mse": mse,
        "additive_correction_mse": float(
            (predicted_correction - realized_correction).square().mean()
        ),
        "rollout_ids": selected.rollout_id.tolist(),
    }
