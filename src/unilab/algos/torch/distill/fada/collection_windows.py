from __future__ import annotations

from collections.abc import Sequence

import torch

from unilab.algos.torch.distill.fada.collection_contract import FADACollectionTransition
from unilab.algos.torch.distill.fada.model import (
    FADA_IDM_SOURCE_ROLE_IDS,
    FADA_SCENARIO_IDS,
    FADAArchitectureConfig,
    FADASourceBatch,
)
from unilab.algos.torch.distill.fada.windows import (
    FADACommandScenario,
    build_fada_causal_window,
    build_fada_cold_start_window,
)


def _window_from_records(
    records: Sequence[FADACollectionTransition],
    config: FADAArchitectureConfig,
    *,
    command_scenario: FADACommandScenario,
    planner_eligible: bool,
    idm_source_role: int,
) -> FADASourceBatch | None:
    causal = build_fada_causal_window(
        records,
        history_length=config.history_length,
        prediction_horizon=config.prediction_horizon,
        command_scenario=command_scenario,
    )
    if causal is None:
        return None
    anchor = config.history_length - 1
    future_records = records[anchor : anchor + config.prediction_horizon]
    return FADASourceBatch(
        observation_history=torch.from_numpy(causal.observation_history[None]),
        action_history=torch.from_numpy(causal.action_history[None]),
        command=torch.from_numpy(causal.command[None]),
        realized_future=torch.from_numpy(causal.realized_future[None]),
        executed_action_chunk=torch.from_numpy(causal.executed_action_chunk[None]),
        oracle_future=torch.from_numpy(future_records[0].oracle_future[None]),
        oracle_action_chunk=torch.from_numpy(future_records[0].oracle_action_chunk[None]),
        oracle_shadow_valid=torch.tensor([future_records[0].oracle_shadow_valid], dtype=torch.bool),
        idm_source_role=torch.tensor([idm_source_role], dtype=torch.int64),
        oracle_first_action=torch.from_numpy(future_records[0].oracle_action[None]),
        command_scenario=torch.tensor([FADA_SCENARIO_IDS[command_scenario]], dtype=torch.int64),
        planner_eligible=torch.tensor([planner_eligible], dtype=torch.bool),
        cold_start=torch.zeros((1,), dtype=torch.bool),
    ).validate(config)


def _cold_start_window_from_records(
    records: Sequence[FADACollectionTransition],
    config: FADAArchitectureConfig,
    *,
    planner_eligible: bool,
    idm_source_role: int,
) -> FADASourceBatch | None:
    causal = build_fada_cold_start_window(
        records,
        history_length=config.history_length,
        prediction_horizon=config.prediction_horizon,
    )
    if causal is None:
        return None
    return FADASourceBatch(
        observation_history=torch.from_numpy(causal.observation_history[None]),
        action_history=torch.from_numpy(causal.action_history[None]),
        command=torch.from_numpy(causal.command[None]),
        realized_future=torch.from_numpy(causal.realized_future[None]),
        executed_action_chunk=torch.from_numpy(causal.executed_action_chunk[None]),
        oracle_future=torch.from_numpy(records[0].oracle_future[None]),
        oracle_action_chunk=torch.from_numpy(records[0].oracle_action_chunk[None]),
        oracle_shadow_valid=torch.tensor([records[0].oracle_shadow_valid], dtype=torch.bool),
        idm_source_role=torch.tensor([idm_source_role], dtype=torch.int64),
        oracle_first_action=torch.from_numpy(records[0].oracle_action[None]),
        command_scenario=torch.tensor([FADA_SCENARIO_IDS["static_stand"]], dtype=torch.int64),
        planner_eligible=torch.tensor([planner_eligible], dtype=torch.bool),
        cold_start=torch.ones((1,), dtype=torch.bool),
    ).validate(config)


def _walking_recovery_window(
    *,
    index: int,
    observation_history,
    action_history,
    command,
    oracle_future,
    oracle_action_chunk,
    oracle_first_action,
    config: FADAArchitectureConfig,
    planner_eligible: bool,
) -> FADASourceBatch:
    row = slice(int(index), int(index) + 1)
    return FADASourceBatch(
        observation_history=torch.from_numpy(observation_history[row].copy()),
        action_history=torch.from_numpy(action_history[row].copy()),
        command=torch.from_numpy(command[row].copy()),
        realized_future=torch.from_numpy(oracle_future[row].copy()),
        executed_action_chunk=torch.from_numpy(oracle_action_chunk[row].copy()),
        oracle_future=torch.from_numpy(oracle_future[row].copy()),
        oracle_action_chunk=torch.from_numpy(oracle_action_chunk[row].copy()),
        oracle_shadow_valid=torch.ones((1,), dtype=torch.bool),
        idm_source_role=torch.tensor([1], dtype=torch.int64),
        oracle_first_action=torch.from_numpy(oracle_first_action[row].copy()),
        command_scenario=torch.tensor([FADA_SCENARIO_IDS["walk"]], dtype=torch.int64),
        planner_eligible=torch.tensor([planner_eligible], dtype=torch.bool),
        cold_start=torch.ones((1,), dtype=torch.bool),
    ).validate(config)


def _terminal_planner_window(
    *,
    observation_history,
    action_history,
    command,
    oracle_future,
    oracle_action_chunk,
    oracle_first_action,
    config: FADAArchitectureConfig,
    command_scenario: FADACommandScenario,
    planner_eligible: bool,
) -> FADASourceBatch:
    return FADASourceBatch(
        observation_history=torch.from_numpy(observation_history.copy()),
        action_history=torch.from_numpy(action_history.copy()),
        command=torch.from_numpy(command.copy()),
        realized_future=torch.from_numpy(oracle_future.copy()),
        executed_action_chunk=torch.from_numpy(oracle_action_chunk.copy()),
        oracle_future=torch.from_numpy(oracle_future.copy()),
        oracle_action_chunk=torch.from_numpy(oracle_action_chunk.copy()),
        oracle_shadow_valid=torch.zeros((1,), dtype=torch.bool),
        idm_source_role=torch.tensor(
            [FADA_IDM_SOURCE_ROLE_IDS["oracle_shadow"]], dtype=torch.int64
        ),
        oracle_first_action=torch.from_numpy(oracle_first_action.copy()),
        command_scenario=torch.tensor([FADA_SCENARIO_IDS[command_scenario]], dtype=torch.int64),
        planner_eligible=torch.tensor([planner_eligible], dtype=torch.bool),
        cold_start=torch.zeros((1,), dtype=torch.bool),
    ).validate(config)


def _concat_batches(
    batches: Sequence[FADASourceBatch], config: FADAArchitectureConfig
) -> FADASourceBatch:
    return FADASourceBatch(
        **{
            field: torch.cat([getattr(batch, field) for batch in batches], dim=0)
            for field in FADASourceBatch.__dataclass_fields__
        }
    ).validate(config)
