"""Read-only FADA diagnostics computed on each policy's own rollout windows."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.target_data import FADATargetBatch

_LOWER_IS_BETTER = (
    "own_rollout_idm_loss",
    "planner_realized_action_gap_rmse",
    "planner_executed_action_replay_rmse",
)


def _device(policy: torch.nn.Module) -> torch.device:
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_device(batch: FADATargetBatch, device: torch.device) -> FADATargetBatch:
    return FADATargetBatch(
        **{
            field: getattr(batch, field).to(device)
            for field in FADATargetBatch.__dataclass_fields__
        }
    )


@torch.no_grad()
def summarize_fada_own_rollout(
    policy: FADAPlannerIDMPolicy,
    batch: FADATargetBatch,
) -> dict[str, float | int]:
    """Measure IDM fit and Planner-IDM consistency on this policy's own data."""

    validated = batch.validate(policy.config)
    device_batch = _to_device(validated, _device(policy))
    was_training = policy.training
    policy.eval()
    try:
        predicted_future = policy.planner(
            device_batch.observation_history,
            device_batch.command,
        )
        planned_action = policy.idm(
            device_batch.observation_history,
            device_batch.action_history,
            predicted_future,
        )[:, 0]
        realized_action = policy.idm(
            device_batch.observation_history,
            device_batch.action_history,
            device_batch.realized_future,
        )[:, 0]
        executed_action = device_batch.executed_action_chunk[:, 0]
        idm_loss = functional.mse_loss(realized_action, executed_action)
        consistency_gap = functional.mse_loss(planned_action, realized_action).sqrt()
        replay_gap = functional.mse_loss(planned_action, executed_action).sqrt()
    finally:
        policy.train(was_training)
    return {
        "num_windows": int(device_batch.observation_history.shape[0]),
        "own_rollout_idm_loss": float(idm_loss.item()),
        "planner_realized_action_gap_rmse": float(consistency_gap.item()),
        "planner_executed_action_replay_rmse": float(replay_gap.item()),
    }


def compare_fada_rollout_diagnostics(
    zero_shot: dict[str, Any],
    adapted: dict[str, Any],
) -> dict[str, float]:
    """Return positive values when adaptation lowers a diagnostic error."""

    return {name: float(zero_shot[name]) - float(adapted[name]) for name in _LOWER_IS_BETTER}
