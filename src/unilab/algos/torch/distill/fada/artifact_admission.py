"""Parent-side FADA source artifact admission and quality-batch owner."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.async_runtime import allocate_fada_command_scenarios
from unilab.algos.torch.distill.fada.model import (
    FADA_IDM_SOURCE_ROLE_IDS,
    FADA_SCENARIO_IDS,
    FADAArchitectureConfig,
    FADASourceBatch,
)
from unilab.algos.torch.distill.fada.replay import FADAReplayBuffer
from unilab.algos.torch.distill.fada.workflow_setup import (
    build_fada_architecture_config,
)
from unilab.algos.torch.distill.fada.workflow_setup import (
    fada_v005_replay_settings as _fada_v005_replay_settings,
)


def _slice_fada_batch(batch: FADASourceBatch, limit: int) -> FADASourceBatch:
    size = min(int(limit), int(batch.command.shape[0]))
    if size <= 0:
        raise ValueError(f"quality_eval_max_windows must be positive, got {limit}")
    return FADASourceBatch(
        **{field: getattr(batch, field)[:size] for field in FADASourceBatch.__dataclass_fields__}
    )


def _fada_quality_batch(
    batch: FADASourceBatch,
    *,
    config: FADAArchitectureConfig,
    limit: int,
    scenario_ratios: Mapping[str, float],
    walk_cold_start_ratio: float,
    static_cold_start_ratio: float,
) -> FADASourceBatch:
    size = min(int(limit), int(batch.command.shape[0]))
    if size <= 0:
        raise ValueError(f"quality_eval_max_windows must be positive, got {limit}")
    replay = FADAReplayBuffer(config, capacity=int(batch.command.shape[0]))
    replay.add(batch)
    return replay.sample_planner(
        size,
        scenario_ratios=scenario_ratios,
        walk_cold_start_ratio=walk_cold_start_ratio,
        static_cold_start_ratio=static_cold_start_ratio,
        generator=torch.Generator().manual_seed(0),
    )


def _require_fada_curriculum_artifact(
    cfg: DictConfig,
    metadata: Mapping[str, Any],
    batch: FADASourceBatch | None = None,
) -> None:
    """在 replay mutation 前验证 scenario 配额与 Oracle role artifact contract."""

    # B1: 由当前 owner config 重算 expected allocations, 产出独立校验基准.
    curriculum = OmegaConf.select(cfg, "training.fada.stand_transition_curriculum")
    if curriculum is None or not bool(OmegaConf.select(curriculum, "enabled", default=False)):
        return
    expected = dict(
        allocate_fada_command_scenarios(
            int(OmegaConf.select(cfg, "training.fada.windows_per_iteration")),
            {
                "walk": float(OmegaConf.select(curriculum, "walk_ratio")),
                "static_stand": float(OmegaConf.select(curriculum, "static_stand_ratio")),
                "walk_to_stand": float(OmegaConf.select(curriculum, "walk_to_stand_ratio")),
            },
        )
    )
    if metadata.get("stand_transition_curriculum_enabled") is not True:
        raise ValueError("FADA async artifact omitted enabled standing curriculum identity")
    if dict(metadata.get("scenario_allocations") or {}) != expected:
        raise ValueError(
            "FADA async artifact scenario allocation mismatch: "
            f"expected={expected} observed={metadata.get('scenario_allocations')}"
        )
    # B2: 聚合 main-source summaries, 产出 observed scenario counts 与 role mapping.
    summaries = metadata.get("collections")
    if not isinstance(summaries, list):
        raise ValueError("FADA async artifact collections must be a list")
    main = [item for item in summaries if item.get("source") == "optimal_or_current_policy"]
    observed = {
        scenario: sum(
            int(item.get("windows", 0)) for item in main if item.get("command_scenario") == scenario
        )
        for scenario in expected
    }
    if observed != expected:
        raise ValueError(
            f"FADA async artifact scenario summary mismatch: expected={expected} observed={observed}"
        )
    for item in main:
        scenario = str(item.get("command_scenario"))
        if item.get("oracle_role") != "unified":
            raise ValueError(
                "FADA async artifact Oracle role mismatch: "
                f"scenario={scenario!r} expected='unified' "
                f"observed={item.get('oracle_role')!r}"
            )
    # B3: 拒绝 unified final-Oracle 漂移和 intermediate Oracle 越权, 再允许 replay consumer.
    if any(
        item.get("command_scenario") != "walk" or item.get("oracle_role") != "walking"
        for item in summaries
        if item.get("source") == "intermediate_oracle"
    ):
        raise ValueError("intermediate Oracle artifacts must remain walking-source only")
    v005_enabled, _planner_ratios, walk_cold_ratio, static_cold_ratio = _fada_v005_replay_settings(
        cfg.training.fada,
        batch_size=int(OmegaConf.select(cfg, "training.fada.batch_size", default=512)),
    )
    if not v005_enabled:
        return
    if metadata.get("v005_replay_enabled") is not True:
        raise ValueError("FADA async artifact omitted enabled v005 replay identity")
    profile_ratios = {
        "walk": walk_cold_ratio,
        "static_stand": static_cold_ratio,
    }
    for scenario, cold_ratio in profile_ratios.items():
        expected_cold = int(math.floor(expected[scenario] * cold_ratio + 0.5))
        expected_profiles = {
            "cold_start": expected_cold,
            "steady_state": expected[scenario] - expected_cold,
        }
        observed_profiles = {
            profile: sum(
                int(item.get("windows", 0))
                for item in main
                if item.get("command_scenario") == scenario
                and item.get("window_profile") == profile
            )
            for profile in expected_profiles
        }
        if observed_profiles != expected_profiles:
            raise ValueError(
                f"v005 {scenario} profile summary mismatch: "
                f"expected={expected_profiles} observed={observed_profiles}"
            )
    if batch is None:
        raise ValueError("v005 FADA artifact validation requires row-level source identity")
    batch.validate(build_fada_architecture_config(cfg))
    main_windows = int(metadata.get("main_windows", 0))
    if main_windows <= 0 or main_windows > int(batch.command.shape[0]):
        raise ValueError(f"invalid v005 FADA main_windows={main_windows}")
    main_mask = torch.arange(batch.command.shape[0]) < main_windows
    intermediate_mask = ~main_mask
    if bool(
        (batch.idm_source_role[intermediate_mask] != FADA_IDM_SOURCE_ROLE_IDS["trajectory"]).any()
    ):
        raise ValueError("FADA intermediate-Oracle IDM role must be trajectory")
    if not bool(batch.planner_eligible[main_mask].all()):
        raise ValueError("v005 main-source rows must remain Planner eligible")
    if bool(batch.planner_eligible[~main_mask].any()):
        raise ValueError("v005 intermediate-Oracle rows must be excluded from Planner replay")
    observed_rows = {
        scenario: int((batch.command_scenario[main_mask] == scenario_id).sum())
        for scenario, scenario_id in FADA_SCENARIO_IDS.items()
        if scenario in expected
    }
    if observed_rows != expected:
        raise ValueError(
            f"v005 row scenario counts mismatch: expected={expected} observed={observed_rows}"
        )
    for scenario, cold_ratio in profile_ratios.items():
        scenario_mask = main_mask & (batch.command_scenario == FADA_SCENARIO_IDS[scenario])
        expected_cold = int(math.floor(expected[scenario] * cold_ratio + 0.5))
        observed_cold = int((scenario_mask & batch.cold_start).sum())
        if observed_cold != expected_cold:
            raise ValueError(
                f"v005 {scenario.replace('_stand', '')} cold-start count mismatch: "
                f"expected={expected_cold} observed={observed_cold}"
            )
    main_iterations = {int(item.get("iteration", -1)) for item in main}
    if len(main_iterations) != 1:
        raise ValueError("FADA main-source summaries must share one iteration")
    iteration = next(iter(main_iterations))
    walking_recovery_mask = (
        main_mask & batch.cold_start & (batch.command_scenario == FADA_SCENARIO_IDS["walk"])
    )
    if bool(
        (
            batch.idm_source_role[walking_recovery_mask]
            != FADA_IDM_SOURCE_ROLE_IDS["oracle_shadow"]
        ).any()
    ):
        raise ValueError("FADA walking recovery IDM role must be oracle_shadow")
    ordinary_main_mask = main_mask & ~walking_recovery_mask
    if iteration == 0:
        valid_ordinary_role = batch.idm_source_role == FADA_IDM_SOURCE_ROLE_IDS["oracle_shadow"]
    else:
        trajectory_role = batch.idm_source_role == FADA_IDM_SOURCE_ROLE_IDS["trajectory"]
        planner_only_terminal = (
            batch.idm_source_role == FADA_IDM_SOURCE_ROLE_IDS["oracle_shadow"]
        ) & ~batch.oracle_shadow_valid
        valid_ordinary_role = trajectory_role | planner_only_terminal
    if bool((ordinary_main_mask & ~valid_ordinary_role).any()):
        raise ValueError("FADA main-source IDM role does not match alternating iteration")


slice_fada_batch = _slice_fada_batch
fada_quality_batch = _fada_quality_batch
require_fada_curriculum_artifact = _require_fada_curriculum_artifact
