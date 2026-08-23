from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import CalibrationRolloutBatch

_SOURCE_PROJECTION_RTOL = 1.0e-4
_PLANNER_PROJECTION_ATOL = 1.0e-3
_ACTION_PROJECTION_ATOL = 5.0e-4


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
        rtol=_SOURCE_PROJECTION_RTOL,
        atol=_PLANNER_PROJECTION_ATOL,
    ):
        raise ValueError("dataset Planner Intent does not match the source policy")
    if not torch.allclose(
        nominal_actions,
        batch.nominal_action_chunk.to(nominal_actions),
        rtol=_SOURCE_PROJECTION_RTOL,
        atol=_ACTION_PROJECTION_ATOL,
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


def _split_stage_batch(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    *,
    training_split_id: int,
    validation_split_id: int,
    stage_name: str,
    axis_count: int,
) -> tuple[CalibrationRolloutBatch, CalibrationRolloutBatch]:
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
