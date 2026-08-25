from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import unilab.algos.torch.distill.fada_artifact_admission as fada_artifact_admission
import unilab.algos.torch.distill.fada_async_runtime as fada_async_runtime
import unilab.algos.torch.distill.fada_training as fada_training
import unilab.algos.torch.distill.fada_workflow as fada_workflow
import unilab.algos.torch.distill.fada_workflow_setup as fada_workflow_setup
from tests.algos._fada_training_test_support import (
    ROOT,
    _CommandControlledEnv,
    _config,
    _curriculum_config,
    _FakeEnv,
    _load_fada_source_diagnostic_script,
    _load_train_distill,
    _Oracle,
    _paper_persistent_training_cfg,
    _paper_role_batch,
    _source_batch,
    _StandingOracle,
    _State,
)
from unilab.algos.torch.distill import (
    FADA_CHECKPOINT_SCHEMA_VERSION,
    FADA_SOURCE_BATCH_SCHEMA_VERSION,
    FADAArchitectureConfig,
    FADACollectionSpec,
    FADAPlannerIDMPolicy,
    FADAReplayBuffer,
    FADATrainer,
    collect_fada_source_windows,
    evaluate_fada_source_batch,
    load_fada_checkpoint,
    load_fada_policy_checkpoint,
    load_fada_source_batch,
    save_fada_checkpoint,
    save_fada_source_batch,
)
from unilab.algos.torch.distill.async_runtime import DaggerCollectRequest
from unilab.algos.torch.distill.fada import (
    FADA_IDM_SOURCE_ROLE_IDS,
    FADA_SCENARIO_IDS,
    FADASourceBatch,
)
from unilab.algos.torch.distill.fada_async_runtime import (
    FADA_ASYNC_SCENARIO,
    PersistentFADACollectorWorker,
    _fada_runtime_device,
    allocate_fada_command_scenarios,
    build_persistent_fada_runtime,
)
from unilab.algos.torch.distill.fada_source_diagnostics import (
    classify_fada_coverage,
    run_fada_coverage_diagnostic,
)
from unilab.algos.torch.distill.fada_training import FADAPaperSourcePlan


def test_fada_replay_default_fifo_reproduces_role_drift_after_overflow() -> None:
    config = _curriculum_config()
    replay = FADAReplayBuffer(config, capacity=24)
    for _ in range(3):
        replay.add(_paper_role_batch(config, main_rows=3, intermediate_rows=6))

    assert replay._batch is not None
    assert int(replay._batch.planner_eligible.sum()) == 6
    assert int((~replay._batch.planner_eligible).sum()) == 18


def test_fada_replay_paper_retention_preserves_one_to_two_roles() -> None:
    config = _curriculum_config()
    replay = FADAReplayBuffer(
        config,
        capacity=24,
        suboptimal_retention_ratio=2,
    )
    for _ in range(3):
        replay.add(_paper_role_batch(config, main_rows=3, intermediate_rows=6))

    counts = replay.source_role_counts()
    assert len(replay) == 24
    assert counts.planner_eligible == 8
    assert counts.planner_ineligible == 16


def test_fada_replay_paper_retention_uses_largest_complete_ratio_capacity() -> None:
    config = _curriculum_config()
    replay = FADAReplayBuffer(
        config,
        capacity=25,
        suboptimal_retention_ratio=2,
    )
    for _ in range(3):
        replay.add(_paper_role_batch(config, main_rows=3, intermediate_rows=6))

    counts = replay.source_role_counts()
    assert replay.effective_capacity == 24
    assert len(replay) == 24
    assert counts.planner_eligible == 8
    assert counts.planner_ineligible == 16


def test_fada_replay_paper_retention_normalizes_integral_float_ratio() -> None:
    config = _curriculum_config()
    replay = FADAReplayBuffer(
        config,
        capacity=24,
        suboptimal_retention_ratio=2.0,  # type: ignore[arg-type]
    )
    for _ in range(3):
        replay.add(_paper_role_batch(config, main_rows=3, intermediate_rows=6))

    assert replay.suboptimal_retention_ratio == 2
    assert isinstance(replay.effective_capacity, int)
    assert replay.source_role_counts().planner_eligible == 8
    assert replay.source_role_counts().planner_ineligible == 16


def test_fada_replay_paper_retention_keeps_newest_rows_in_role_order() -> None:
    config = _curriculum_config()
    replay = FADAReplayBuffer(
        config,
        capacity=24,
        suboptimal_retention_ratio=2,
    )
    for iteration in range(3):
        batch = _paper_role_batch(config, main_rows=3, intermediate_rows=6)
        row_ids = torch.arange(9, dtype=torch.float32) + iteration * 9
        replay.add(replace(batch, command=row_ids[:, None].repeat(1, config.command_dim)))

    assert replay._batch is not None
    torch.testing.assert_close(
        replay._batch.command[:, 0],
        torch.tensor(
            [
                1,
                2,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
            ],
            dtype=torch.float32,
        ),
    )


def test_fada_replay_paper_retention_failure_is_atomic() -> None:
    config = _curriculum_config()
    replay = FADAReplayBuffer(
        config,
        capacity=9,
        suboptimal_retention_ratio=2,
    )
    replay.add(_paper_role_batch(config, main_rows=2, intermediate_rows=4))
    assert replay._batch is not None
    before = {
        field: getattr(replay._batch, field).clone()
        for field in FADASourceBatch.__dataclass_fields__
    }

    with pytest.raises(ValueError, match="lacks Planner-ineligible intermediate rows"):
        replay.add(_paper_role_batch(config, main_rows=4, intermediate_rows=0))

    assert replay._batch is not None
    for field, expected in before.items():
        torch.testing.assert_close(getattr(replay._batch, field), expected)


@pytest.mark.parametrize(
    ("ratio", "capacity", "message"),
    [
        (0, 24, "positive integer"),
        (2.5, 24, "positive integer"),
        (2, 2, "complete replay role-ratio block"),
    ],
)
def test_fada_replay_paper_retention_rejects_invalid_contract(
    ratio: int | float,
    capacity: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FADAReplayBuffer(
            _curriculum_config(),
            capacity=capacity,
            suboptimal_retention_ratio=ratio,  # type: ignore[arg-type]
        )


def test_planner_replay_preserves_scenario_and_cold_start_quotas() -> None:
    config = _curriculum_config()
    batch = _source_batch(config, size=20)
    scenario = torch.tensor(
        [
            *([FADA_SCENARIO_IDS["walk"]] * 4),
            *([FADA_SCENARIO_IDS["static_stand"]] * 4),
            *([FADA_SCENARIO_IDS["walk_to_stand"]] * 2),
            *([FADA_SCENARIO_IDS["walk"]] * 10),
        ],
        dtype=torch.int64,
    )
    replay = FADAReplayBuffer(config, capacity=20)
    replay.add(
        replace(
            batch,
            command_scenario=scenario,
            planner_eligible=torch.tensor([True] * 10 + [False] * 10),
            cold_start=torch.tensor(
                [True, True, False, False] + [True, True, False, False] + [False] * 12
            ),
        )
    )

    sampled = replay.sample_planner(
        40,
        scenario_ratios={"walk": 0.5, "static_stand": 0.25, "walk_to_stand": 0.25},
        walk_cold_start_ratio=0.5,
        static_cold_start_ratio=0.5,
        generator=torch.Generator().manual_seed(5),
    )

    assert bool(sampled.planner_eligible.all())
    walk = sampled.command_scenario == FADA_SCENARIO_IDS["walk"]
    assert int(walk.sum()) == 20
    assert int((walk & sampled.cold_start).sum()) == 10
    static = sampled.command_scenario == FADA_SCENARIO_IDS["static_stand"]
    assert int(static.sum()) == 10
    assert int((static & sampled.cold_start).sum()) == 5
    assert int((sampled.command_scenario == FADA_SCENARIO_IDS["walk_to_stand"]).sum()) == 10


def test_planner_replay_fails_closed_when_required_stratum_is_missing() -> None:
    config = _curriculum_config()
    replay = FADAReplayBuffer(config, capacity=8)
    replay.add(_source_batch(config, size=8))

    with pytest.raises(ValueError, match="walk/cold_start"):
        replay.sample_planner(
            8,
            scenario_ratios={"walk": 0.5, "static_stand": 0.25, "walk_to_stand": 0.25},
            walk_cold_start_ratio=0.5,
            static_cold_start_ratio=0.5,
        )


def test_collector_builds_same_state_oracle_shadow_without_advancing_main_rollout() -> None:
    config = _config()
    env = _FakeEnv()
    result = collect_fada_source_windows(
        env,
        teacher_policy=_Oracle(),
        config=config,
        num_windows=1,
        spec=FADACollectionSpec(collect_oracle_shadow=True),
    )

    assert bool(result.batch.oracle_shadow_valid.all())
    torch.testing.assert_close(
        result.batch.oracle_action_chunk[:, 0], result.batch.oracle_first_action
    )
    expected_first = result.batch.observation_history[:, -1].clone()
    expected_first[:, :2] += result.batch.oracle_action_chunk[:, 0]
    expected_first[:, 2] += 1.0
    torch.testing.assert_close(result.batch.oracle_future[:, 0], expected_first)
    assert env.step_count == result.env_steps


def test_collector_prefers_authoritative_reset_all_state_over_detached_reset_return() -> None:
    config = _config()

    class _CarrierEnv(_FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self.reset_called = False

        def reset_all(self):
            self.current_obs = np.zeros((1, 3), dtype=np.float32)
            self.step_count = 0
            return _State(
                obs={"obs": self.current_obs.copy()},
                info={"commands": np.asarray([[0.4, -0.1]], dtype=np.float32)},
                terminated=np.zeros((1,), dtype=np.bool_),
                truncated=np.zeros((1,), dtype=np.bool_),
            )

        def reset(self, _indices: np.ndarray):
            self.reset_called = True
            return super().reset(_indices)

    env = _CarrierEnv()
    collect_fada_source_windows(env, teacher_policy=_Oracle(), config=config, num_windows=1)
    assert env.reset_called is False


def test_intermediate_oracle_rollout_keeps_final_oracle_planner_label() -> None:
    config = _config()

    class _Intermediate(_Oracle):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros((obs.shape[0], 2), device=obs.device)

    result = collect_fada_source_windows(
        _FakeEnv(),
        teacher_policy=_Oracle(),
        rollout_teacher_policy=_Intermediate(),
        config=config,
        num_windows=1,
        spec=FADACollectionSpec(planner_eligible=False),
    )
    assert result.rollout_mode == "intermediate_oracle"
    torch.testing.assert_close(result.batch.executed_action_chunk[:, 0], torch.zeros(1, 2))
    assert bool(torch.any(result.batch.oracle_first_action != 0.0))
    assert not bool(result.batch.planner_eligible.any())


def test_quality_evaluator_separates_idm_shadow_and_planner_boundaries() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    metrics = evaluate_fada_source_batch(policy, _source_batch(config, size=3))

    assert set(metrics) == {
        "trajectory_idm_action_mse",
        "oracle_shadow_idm_action_mse",
        "planner_idm_oracle_action_mse",
        "planner_future_realized_mse",
        "oracle_shadow_valid_fraction",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["oracle_shadow_valid_fraction"] == 1.0


def test_quality_evaluator_reports_required_v005_strata() -> None:
    config = _curriculum_config()
    batch = _source_batch(config, size=8)
    batch = replace(
        batch,
        command_scenario=torch.tensor(
            [
                FADA_SCENARIO_IDS["walk"],
                FADA_SCENARIO_IDS["walk"],
                FADA_SCENARIO_IDS["static_stand"],
                FADA_SCENARIO_IDS["static_stand"],
                FADA_SCENARIO_IDS["static_stand"],
                FADA_SCENARIO_IDS["static_stand"],
                FADA_SCENARIO_IDS["walk_to_stand"],
                FADA_SCENARIO_IDS["walk_to_stand"],
            ],
            dtype=torch.int64,
        ),
        cold_start=torch.tensor([True, False, True, True, False, False, False, False]),
    )

    metrics = evaluate_fada_source_batch(
        FADAPlannerIDMPolicy(config),
        batch,
        require_scenario_metrics=True,
    )

    assert metrics["scenario/walk/row_fraction"] == pytest.approx(0.25)
    assert metrics["scenario/static_stand/row_fraction"] == pytest.approx(0.5)
    assert metrics["scenario/walk_to_stand/row_fraction"] == pytest.approx(0.25)
    assert metrics["scenario/walk/cold_start_fraction"] == pytest.approx(0.5)
    assert "scenario/walk/cold_start_planner_mse" in metrics
    assert "scenario/walk/steady_state_planner_mse" in metrics
    assert metrics["scenario/static_stand/cold_start_fraction"] == pytest.approx(0.5)
    assert "scenario/static_stand/cold_start_planner_mse" in metrics
    assert "scenario/static_stand/steady_state_planner_mse" in metrics


def test_paper_source_preflight_requires_exact_intermediate_oracle_set(tmp_path: Path) -> None:
    checkpoints = [tmp_path / f"model_{index}.pt" for index in range(20)]
    for checkpoint in checkpoints:
        checkpoint.touch()
    cfg = OmegaConf.create(
        {
            "training": {
                "fada": {
                    "paper_source_enabled": True,
                    "oracle_shadow_enabled": True,
                    "resume_path": None,
                    "suboptimal_data_ratio": 2.0,
                    "windows_per_iteration": 10,
                    "intermediate_oracle_count": 20,
                    "intermediate_oracle_checkpoint_paths": [str(path) for path in checkpoints],
                }
            }
        }
    )

    plan = fada_workflow_setup.paper_source_plan(cfg)
    assert plan.checkpoint_paths == tuple(checkpoints)
    assert sum(windows for _, windows in plan.source_allocations) == 20
    assert all(windows == 1 for _, windows in plan.source_allocations)
    cfg.training.fada.intermediate_oracle_count = 19
    with pytest.raises(ValueError, match="intermediate_oracle_count=20"):
        fada_workflow_setup.paper_source_plan(cfg)
    cfg.training.fada.intermediate_oracle_count = 20
    cfg.training.fada.intermediate_oracle_checkpoint_paths = [
        str(path) for path in checkpoints[:-1]
    ]
    with pytest.raises(ValueError, match="exactly 20 unique"):
        fada_workflow_setup.paper_source_plan(cfg)


def test_standing_curriculum_rejects_legacy_execution_mode() -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "fada": {
                    "execution_mode": "legacy",
                    "stand_transition_curriculum": {"enabled": True},
                }
            }
        }
    )
    with pytest.raises(ValueError, match="requires.*persistent_async"):
        fada_workflow_setup.fada_execution_mode(cfg)
    cfg.training.fada.stand_transition_curriculum.enabled = False
    assert fada_workflow_setup.fada_execution_mode(cfg) == "legacy"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, "cpu"), ("", "cpu"), ("cuda:0", "cuda:0")],
)
def test_fada_async_runtime_normalizes_device(configured: str | None, expected: str) -> None:
    cfg = OmegaConf.create({"training": {"device": configured}})
    assert _fada_runtime_device(cfg) == expected


def test_v005_replay_settings_reject_contract_ratio_drift() -> None:
    fada = OmegaConf.create(
        {
            "v005_replay": {
                "enabled": True,
                "walk_cold_start_ratio": 0.5,
                "static_cold_start_ratio": 0.5,
                "planner_scenario_ratios": {
                    "walk": 0.5,
                    "static_stand": 0.25,
                    "walk_to_stand": 0.25,
                },
            }
        }
    )
    assert fada_workflow_setup.fada_v005_replay_settings(fada, batch_size=8)[0] is True

    fada.v005_replay.planner_scenario_ratios.walk = 0.4
    with pytest.raises(ValueError, match="scenario ratios are fixed"):
        fada_workflow_setup.fada_v005_replay_settings(fada, batch_size=8)
    fada.v005_replay.planner_scenario_ratios.walk = 0.5
    fada.v005_replay.walk_cold_start_ratio = 0.25
    with pytest.raises(ValueError, match="walk_cold_start_ratio is fixed"):
        fada_workflow_setup.fada_v005_replay_settings(fada, batch_size=8)
    fada.v005_replay.walk_cold_start_ratio = 0.5
    fada.v005_replay.static_cold_start_ratio = 0.25
    with pytest.raises(ValueError, match="static_cold_start_ratio is fixed"):
        fada_workflow_setup.fada_v005_replay_settings(fada, batch_size=8)


def test_parent_curriculum_artifact_guard_rejects_oracle_role_drift() -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "fada": {
                    "windows_per_iteration": 4,
                    "stand_transition_curriculum": {
                        "enabled": True,
                        "walk_ratio": 0.5,
                        "static_stand_ratio": 0.25,
                        "walk_to_stand_ratio": 0.25,
                    },
                }
            }
        }
    )
    metadata = {
        "stand_transition_curriculum_enabled": True,
        "scenario_allocations": {"walk": 2, "static_stand": 1, "walk_to_stand": 1},
        "collections": [
            {
                "source": "optimal_or_current_policy",
                "command_scenario": "walk",
                "oracle_role": "walking",
                "windows": 2,
            },
            {
                "source": "optimal_or_current_policy",
                "command_scenario": "static_stand",
                "oracle_role": "standing",
                "windows": 1,
            },
            {
                "source": "optimal_or_current_policy",
                "command_scenario": "walk_to_stand",
                "oracle_role": "standing",
                "windows": 1,
            },
            {
                "source": "intermediate_oracle",
                "command_scenario": "walk",
                "oracle_role": "walking",
                "windows": 8,
            },
        ],
    }
    fada_artifact_admission.require_fada_curriculum_artifact(cfg, metadata)
    metadata["collections"][2]["oracle_role"] = "walking"
    with pytest.raises(ValueError, match="Oracle role mismatch"):
        fada_artifact_admission.require_fada_curriculum_artifact(cfg, metadata)


def test_v005_parent_artifact_guard_requires_row_provenance() -> None:
    config = _curriculum_config()
    cfg = OmegaConf.create(
        {
            "student": {"obs_dim": config.obs_dim, "action_dim": config.action_dim},
            "training": {
                "fada": {
                    "phase": "planner",
                    "windows_per_iteration": 8,
                    "batch_size": 8,
                    "command_dim": config.command_dim,
                    "history_length": config.history_length,
                    "prediction_horizon": config.prediction_horizon,
                    "hidden_dim": config.hidden_dim,
                    "num_heads": config.num_heads,
                    "planner_layers": config.planner_layers,
                    "idm_encoder_layers": config.idm_encoder_layers,
                    "idm_decoder_layers": config.idm_decoder_layers,
                    "feedforward_dim": config.feedforward_dim,
                    "dropout": config.dropout,
                    "stand_transition_curriculum": {
                        "enabled": True,
                        "walk_ratio": 0.5,
                        "static_stand_ratio": 0.25,
                        "walk_to_stand_ratio": 0.25,
                    },
                    "v005_replay": {
                        "enabled": True,
                        "walk_cold_start_ratio": 0.5,
                        "static_cold_start_ratio": 0.5,
                        "planner_scenario_ratios": {
                            "walk": 0.5,
                            "static_stand": 0.25,
                            "walk_to_stand": 0.25,
                        },
                    },
                }
            },
        }
    )
    metadata = {
        "stand_transition_curriculum_enabled": True,
        "v005_replay_enabled": True,
        "main_windows": 8,
        "scenario_allocations": {"walk": 4, "static_stand": 2, "walk_to_stand": 2},
        "collections": [
            {
                "iteration": 1,
                "source": "optimal_or_current_policy",
                "command_scenario": "walk",
                "oracle_role": "walking",
                "window_profile": "cold_start",
                "windows": 2,
            },
            {
                "iteration": 1,
                "source": "optimal_or_current_policy",
                "command_scenario": "walk",
                "oracle_role": "walking",
                "window_profile": "steady_state",
                "windows": 2,
            },
            {
                "iteration": 1,
                "source": "optimal_or_current_policy",
                "command_scenario": "static_stand",
                "oracle_role": "standing",
                "window_profile": "cold_start",
                "windows": 1,
            },
            {
                "iteration": 1,
                "source": "optimal_or_current_policy",
                "command_scenario": "static_stand",
                "oracle_role": "standing",
                "window_profile": "steady_state",
                "windows": 1,
            },
            {
                "iteration": 1,
                "source": "optimal_or_current_policy",
                "command_scenario": "walk_to_stand",
                "oracle_role": "standing",
                "window_profile": "steady_state",
                "windows": 2,
            },
            {
                "iteration": 1,
                "source": "intermediate_oracle",
                "command_scenario": "walk",
                "oracle_role": "walking",
                "windows": 2,
            },
        ],
    }
    batch = replace(
        _source_batch(config, size=10),
        idm_source_role=torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int64),
        command_scenario=torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 0, 0], dtype=torch.int64),
        planner_eligible=torch.tensor([True] * 8 + [False, False], dtype=torch.bool),
        cold_start=torch.tensor(
            [True, True, False, False, True, False, False, False, False, False],
            dtype=torch.bool,
        ),
    )

    fada_artifact_admission.require_fada_curriculum_artifact(cfg, metadata, batch)
    cfg.training.fada.phase = "idm_pretrain"
    idm_roles = batch.idm_source_role.clone()
    idm_roles[:8] = FADA_IDM_SOURCE_ROLE_IDS["oracle_shadow"]
    fada_artifact_admission.require_fada_curriculum_artifact(
        cfg,
        metadata,
        replace(batch, idm_source_role=idm_roles),
    )
    cfg.training.fada.phase = "planner"
    planner_only_roles = batch.idm_source_role.clone()
    planner_only_roles[2] = FADA_IDM_SOURCE_ROLE_IDS["oracle_shadow"]
    planner_only_valid = batch.oracle_shadow_valid.clone()
    planner_only_valid[2] = False
    planner_only_terminal = replace(
        batch,
        idm_source_role=planner_only_roles,
        oracle_shadow_valid=planner_only_valid,
    )
    fada_artifact_admission.require_fada_curriculum_artifact(cfg, metadata, planner_only_terminal)
    with pytest.raises(ValueError, match="main-source IDM role"):
        fada_artifact_admission.require_fada_curriculum_artifact(
            cfg,
            metadata,
            replace(planner_only_terminal, oracle_shadow_valid=torch.ones(10, dtype=torch.bool)),
        )
    invalid_recovery_role = replace(batch, idm_source_role=torch.zeros(10, dtype=torch.int64))
    with pytest.raises(ValueError, match="walking recovery IDM role"):
        fada_artifact_admission.require_fada_curriculum_artifact(
            cfg, metadata, invalid_recovery_role
        )
    mismatched_profiles = {
        **metadata,
        "collections": [dict(item) for item in metadata["collections"]],
    }
    mismatched_profiles["collections"][0]["windows"] = 1
    mismatched_profiles["collections"][1]["windows"] = 3
    with pytest.raises(ValueError, match="walk profile summary mismatch"):
        fada_artifact_admission.require_fada_curriculum_artifact(cfg, mismatched_profiles, batch)
    missing_walk_cold = replace(
        batch,
        cold_start=torch.tensor(
            [False, False, False, False, True, False, False, False, False, False],
            dtype=torch.bool,
        ),
    )
    with pytest.raises(ValueError, match="walk cold-start count mismatch"):
        fada_artifact_admission.require_fada_curriculum_artifact(cfg, metadata, missing_walk_cold)
    with pytest.raises(ValueError, match="requires row-level source identity"):
        fada_artifact_admission.require_fada_curriculum_artifact(cfg, metadata)
    invalid = replace(batch, planner_eligible=torch.ones(10, dtype=torch.bool))
    with pytest.raises(ValueError, match="intermediate-Oracle rows"):
        fada_artifact_admission.require_fada_curriculum_artifact(cfg, metadata, invalid)
