"""FADA sealed source-batch quality evaluation owner."""

from __future__ import annotations

import torch

from unilab.algos.torch.distill.fada.model import (
    FADA_COMMAND_SCENARIOS,
    FADA_SCENARIO_IDS,
    FADAPlannerIDMPolicy,
    FADASourceBatch,
)
from unilab.algos.torch.distill.fada.source_artifact import batch_to_device


@torch.no_grad()
def evaluate_fada_source_batch(
    policy: FADAPlannerIDMPolicy,
    batch: FADASourceBatch,
    *,
    require_scenario_metrics: bool = False,
) -> dict[str, float]:
    """Measure the three adjacent source-quality boundaries on one sealed batch."""

    # B1: 在 policy device 上重放 causal rows, 产出 true-future IDM boundary error.
    device = next(policy.parameters()).device
    current = batch_to_device(batch.validate(policy.config), device)
    trajectory_action = policy.idm(
        current.observation_history,
        current.action_history,
        current.realized_future,
    )[:, 0]
    trajectory_mse = torch.mean(
        torch.square(trajectory_action - current.executed_action_chunk[:, 0])
    )

    # B2: 单独测量 final-Oracle shadow support; 没有 valid row 时 fail-closed.
    valid = current.oracle_shadow_valid
    if not bool(valid.any()):
        raise ValueError("FADA quality evaluation requires at least one valid Oracle-shadow row")
    shadow_action = policy.idm(
        current.observation_history[valid],
        current.action_history[valid],
        current.oracle_future[valid],
    )[:, 0]
    shadow_mse = torch.mean(torch.square(shadow_action - current.oracle_action_chunk[valid, 0]))

    # B3: 测量 Planner 经 IDM 的 Oracle-action error 与 future support drift, 交给 checkpoint consumer.
    output = policy(
        current.observation_history,
        current.action_history,
        current.command,
    )
    planner_action_mse = torch.mean(torch.square(output.action - current.oracle_first_action))
    planner_future_realized_mse = torch.mean(
        torch.square(output.predicted_future - current.realized_future)
    )
    metrics = {
        "trajectory_idm_action_mse": float(trajectory_mse),
        "oracle_shadow_idm_action_mse": float(shadow_mse),
        "planner_idm_oracle_action_mse": float(planner_action_mse),
        "planner_future_realized_mse": float(planner_future_realized_mse),
        "oracle_shadow_valid_fraction": float(valid.float().mean()),
    }
    if require_scenario_metrics:
        eligible = current.planner_eligible
        if not bool(eligible.any()):
            raise ValueError("FADA scenario quality requires Planner-eligible rows")
        for scenario in FADA_COMMAND_SCENARIOS:
            mask = eligible & (current.command_scenario == FADA_SCENARIO_IDS[scenario])
            if not bool(mask.any()):
                raise ValueError(f"FADA scenario quality is missing {scenario!r} rows")
            metrics[f"scenario/{scenario}/row_fraction"] = float(mask.float().mean())
            metrics[f"scenario/{scenario}/planner_idm_oracle_action_mse"] = float(
                torch.mean(torch.square(output.action[mask] - current.oracle_first_action[mask]))
            )
        for scenario in ("walk", "static_stand"):
            scenario_mask = eligible & (current.command_scenario == FADA_SCENARIO_IDS[scenario])
            cold = scenario_mask & current.cold_start
            steady = scenario_mask & ~current.cold_start
            if not bool(cold.any()) or not bool(steady.any()):
                raise ValueError(
                    f"FADA scenario quality requires {scenario} cold-start and steady rows"
                )
            metrics[f"scenario/{scenario}/cold_start_fraction"] = float(
                cold.float().sum() / scenario_mask.float().sum()
            )
            metrics[f"scenario/{scenario}/cold_start_planner_mse"] = float(
                torch.mean(torch.square(output.action[cold] - current.oracle_first_action[cold]))
            )
            metrics[f"scenario/{scenario}/steady_state_planner_mse"] = float(
                torch.mean(
                    torch.square(output.action[steady] - current.oracle_first_action[steady])
                )
            )
    if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
        raise ValueError(f"FADA quality metrics must be finite: {metrics}")
    return metrics
