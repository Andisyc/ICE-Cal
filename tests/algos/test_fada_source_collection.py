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

import unilab.algos.torch.distill.fada_async_runtime as fada_async_runtime
import unilab.algos.torch.distill.fada_training as fada_training
import unilab.algos.torch.distill.fada_workflow as fada_workflow
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
from unilab.algos.torch.distill.fada import FADA_SCENARIO_IDS, FADASourceBatch
from unilab.algos.torch.distill.fada_async_runtime import (
    FADA_ASYNC_SCENARIO,
    PersistentFADACollectorWorker,
    _fada_runtime_device,
    allocate_fada_command_scenarios,
    build_persistent_fada_runtime,
)
from unilab.algos.torch.distill.fada_collection_transaction import (
    _default_collection_step_limit,
)
from unilab.algos.torch.distill.fada_source_diagnostics import (
    classify_fada_coverage,
    run_fada_coverage_diagnostic,
)
from unilab.algos.torch.distill.fada_training import FADAPaperSourcePlan


def test_collector_builds_causal_oracle_bootstrap_windows() -> None:
    config = _config()
    result = collect_fada_source_windows(
        _FakeEnv(),
        teacher_policy=_Oracle(),
        config=config,
        num_windows=2,
    )

    assert result.rollout_mode == "oracle"
    assert result.batch.observation_history.shape == (2, 2, 3)
    torch.testing.assert_close(
        result.batch.executed_action_chunk[:, 0], result.batch.oracle_first_action
    )
    expected_next = result.batch.observation_history[:, -1].clone()
    expected_next[:, :2] += result.batch.executed_action_chunk[:, 0]
    expected_next[:, 2] += 1.0
    torch.testing.assert_close(result.batch.realized_future[:, 0], expected_next)


def test_default_collection_budget_does_not_assume_ten_percent_acceptance() -> None:
    # Production regression: 48,219/65,536 windows were still making progress when
    # the old 10x ideal-parallel-step heuristic stopped at 10,275 env steps.
    assert (
        _default_collection_step_limit(
            num_windows=65_536,
            num_envs=64,
            record_count=35,
        )
        == 65_571
    )


def test_collection_limit_error_reports_rejection_diagnostics() -> None:
    with pytest.raises(RuntimeError) as error:
        collect_fada_source_windows(
            _FakeEnv(done_steps=(1,)),
            teacher_policy=_Oracle(),
            config=_config(),
            num_windows=1,
            spec=FADACollectionSpec(max_env_steps=1),
        )

    message = str(error.value)
    assert "acceptance=0.00%" in message
    assert "rejected_done=1" in message
    assert "rejected_command=0" in message
    assert "rejected_scenario=0" in message


def test_source_collector_characterization_maps_every_causal_index() -> None:
    result = collect_fada_source_windows(
        _FakeEnv(),
        teacher_policy=_Oracle(),
        config=_config(),
        num_windows=1,
    )

    torch.testing.assert_close(
        result.batch.observation_history[0],
        torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 1.0]]),
    )
    torch.testing.assert_close(
        result.batch.action_history[0],
        torch.tensor([[0.0, 0.0], [0.25, 0.25]]),
    )
    torch.testing.assert_close(
        result.batch.realized_future[0],
        torch.tensor([[0.75, 0.75, 2.0], [1.75, 1.75, 3.0]]),
    )
    torch.testing.assert_close(
        result.batch.executed_action_chunk[0],
        torch.tensor([[0.5, 0.5], [1.0, 1.0]]),
    )
    torch.testing.assert_close(result.batch.oracle_first_action[0], torch.tensor([0.5, 0.5]))


def test_source_collector_rejects_projection_contract_mismatch_before_reset() -> None:
    class _ResetCountingEnv(_FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self.reset_calls = 0

        def reset(self, indices: np.ndarray):
            self.reset_calls += 1
            return super().reset(indices)

    env = _ResetCountingEnv()
    with pytest.raises(ValueError, match="projection does not match"):
        collect_fada_source_windows(
            env,
            teacher_policy=_Oracle(),
            config=_config(),
            num_windows=1,
            spec=FADACollectionSpec(student_projection="g1_fada_state_v2"),
        )

    assert env.reset_calls == 0


def test_source_cold_start_characterization_repeats_nonzero_reset_observation() -> None:
    class _NonzeroResetEnv(_CommandControlledEnv):
        def reset_all(self) -> _State:
            self.current_obs = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
            self.step_count = 0
            self.state = self._state(np.zeros((1, 3), dtype=np.float32))
            return self.state

    result = collect_fada_source_windows(
        _NonzeroResetEnv(),
        teacher_policy=_Oracle(),
        config=_curriculum_config(),
        num_windows=1,
        spec=FADACollectionSpec(
            collect_oracle_shadow=True,
            command_scenario="static_stand",
            cold_start_windows=True,
        ),
    )

    torch.testing.assert_close(
        result.batch.observation_history,
        torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]),
    )
    torch.testing.assert_close(
        result.batch.action_history,
        torch.zeros_like(result.batch.action_history),
    )


def test_static_collector_builds_exact_reset_aligned_cold_start_window() -> None:
    config = _curriculum_config()
    result = collect_fada_source_windows(
        _CommandControlledEnv(),
        teacher_policy=_Oracle(),
        config=config,
        num_windows=1,
        spec=FADACollectionSpec(
            collect_oracle_shadow=True,
            command_scenario="static_stand",
            cold_start_windows=True,
        ),
    )

    assert result.window_profile == "cold_start"
    assert result.env_steps == config.prediction_horizon
    assert bool(result.batch.cold_start.all())
    assert bool(result.batch.planner_eligible.all())
    assert torch.equal(
        result.batch.command_scenario,
        torch.full((1,), FADA_SCENARIO_IDS["static_stand"], dtype=torch.int64),
    )
    torch.testing.assert_close(
        result.batch.observation_history,
        torch.zeros_like(result.batch.observation_history),
    )
    torch.testing.assert_close(
        result.batch.action_history,
        torch.zeros_like(result.batch.action_history),
    )
    torch.testing.assert_close(result.batch.command, torch.zeros_like(result.batch.command))


def test_walking_recovery_prefix_keeps_oracle_label_when_student_falls_early() -> None:
    config = _config()

    class _ActionTerminatingEnv(_FakeEnv):
        def step(self, actions: np.ndarray) -> _State:
            state = super().step(actions)
            state.terminated[:] = np.any(np.abs(actions) > 1.0, axis=1)
            return state

    rollout_policy = FADAPlannerIDMPolicy(config)
    with torch.no_grad():
        for parameter in rollout_policy.parameters():
            parameter.zero_()
        rollout_policy.idm.action_head.bias.fill_(5.0)

    result = collect_fada_source_windows(
        _ActionTerminatingEnv(),
        teacher_policy=_Oracle(),
        rollout_policy=rollout_policy,
        config=config,
        num_windows=2,
        spec=FADACollectionSpec(
            collect_oracle_shadow=True,
            command_scenario="walk",
            cold_start_windows=True,
            max_env_steps=4,
        ),
    )

    assert result.window_profile == "cold_start"
    assert result.rejected_done_transitions >= 1
    assert bool(result.batch.cold_start.all())
    assert torch.equal(
        result.batch.command_scenario,
        torch.full((2,), FADA_SCENARIO_IDS["walk"], dtype=torch.int64),
    )
    torch.testing.assert_close(
        result.batch.observation_history,
        torch.zeros_like(result.batch.observation_history),
    )
    torch.testing.assert_close(
        result.batch.action_history,
        torch.zeros_like(result.batch.action_history),
    )
    torch.testing.assert_close(result.batch.realized_future, result.batch.oracle_future)
    torch.testing.assert_close(
        result.batch.executed_action_chunk,
        result.batch.oracle_action_chunk,
    )
    torch.testing.assert_close(
        result.batch.oracle_first_action,
        result.batch.oracle_action_chunk[:, 0],
    )
    assert bool(result.batch.oracle_shadow_valid.all())
    assert bool((result.batch.idm_source_role == 1).all())


def test_terminal_prefall_label_does_not_consume_walking_cold_start_quota() -> None:
    class _EpisodeDoneEnv(_FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self.episode_step = 0

        def reset(self, indices: np.ndarray):
            self.episode_step = 0
            return super().reset(indices)

        def step(self, actions: np.ndarray) -> _State:
            state = super().step(actions)
            self.episode_step += 1
            state.terminated[:] = self.episode_step == 3
            return state

        @contextmanager
        def preserve_rollout_state(self):
            episode_step = self.episode_step
            with super().preserve_rollout_state():
                yield
            self.episode_step = episode_step

    result = collect_fada_source_windows(
        _EpisodeDoneEnv(),
        teacher_policy=_Oracle(),
        config=_config(),
        num_windows=4,
        spec=FADACollectionSpec(
            collect_oracle_shadow=True,
            command_scenario="walk",
            cold_start_windows=True,
            max_env_steps=12,
        ),
    )

    assert result.rejected_done_transitions >= 1
    assert result.window_profile == "cold_start"
    assert bool(result.batch.cold_start.all())


def test_ordinary_walking_keeps_terminal_prefall_planner_label() -> None:
    result = collect_fada_source_windows(
        _FakeEnv(done_steps=(3, 6)),
        teacher_policy=_Oracle(),
        config=_config(),
        num_windows=1,
        spec=FADACollectionSpec(
            command_scenario="walk",
            collect_oracle_shadow=True,
            max_env_steps=6,
        ),
    )

    assert result.rejected_done_transitions >= 1
    assert bool(result.batch.planner_eligible.all())
    assert bool(result.batch.cold_start.logical_not().all())
    assert not bool(result.batch.oracle_shadow_valid.any())
    assert bool((result.batch.idm_source_role == 1).all())


def test_v007_bound_coverage_diagnostic_classifies_all_three_verdicts() -> None:
    config = _curriculum_config()

    class _PostHistoryFallEnv(_CommandControlledEnv):
        def __init__(self) -> None:
            super().__init__()
            self.in_shadow = False

        def step(self, actions: np.ndarray) -> _State:
            state = super().step(actions)
            if not self.in_shadow and self.step_count >= config.history_length + 1:
                state.terminated[:] = np.any(np.abs(actions) > 1.0, axis=1)
                self.state = state
            return state

        @contextmanager
        def preserve_rollout_state(self):
            self.in_shadow = True
            try:
                with super().preserve_rollout_state():
                    yield
            finally:
                self.in_shadow = False

    student = FADAPlannerIDMPolicy(config)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.zero_()
        student.idm.action_head.bias.fill_(2.0)

    report = run_fada_coverage_diagnostic(
        _PostHistoryFallEnv(),
        student_policy=student,
        teacher_policy=_Oracle(),
        config=config,
        max_steps=8,
        spec=FADACollectionSpec(collect_oracle_shadow=True),
    )

    assert report.verdict == "COVERAGE_GAP"
    assert report.failure_reproduced
    assert report.identity_valid
    assert report.coverage_gap_step_indices
    assert report.steps[-1].v007_rejection_reason == ("episode_terminated_before_window_completion")

    early_fall = tuple(replace(step, timestep=0) for step in report.steps[-1:])
    rejected = classify_fada_coverage(
        early_fall,
        history_length=config.history_length,
        prediction_horizon=config.prediction_horizon,
        command=(0.4, 0.0, 0.0),
        stop_reason="environment_done",
    )
    assert rejected.verdict == "COVERAGE_CAUSE_REJECTED"

    identity_conflict = classify_fada_coverage(
        tuple(replace(step, snapshot_observable_restoration_valid=False) for step in report.steps),
        history_length=config.history_length,
        prediction_horizon=config.prediction_horizon,
        command=(0.4, 0.0, 0.0),
        stop_reason="environment_done",
    )
    assert identity_conflict.verdict == "IDENTITY_OR_MEASUREMENT_CONFLICT"

    truncation_only = classify_fada_coverage(
        tuple(replace(step, terminated=False, truncated=True) for step in report.steps[-1:]),
        history_length=config.history_length,
        prediction_horizon=config.prediction_horizon,
        command=(0.4, 0.0, 0.0),
        stop_reason="environment_done",
    )
    assert truncation_only.verdict == "IDENTITY_OR_MEASUREMENT_CONFLICT"

    command_conflict = classify_fada_coverage(
        tuple(replace(step, command_identity_valid=False) for step in report.steps),
        history_length=config.history_length,
        prediction_horizon=config.prediction_horizon,
        command=(0.4, 0.0, 0.0),
        stop_reason="environment_done",
    )
    assert command_conflict.verdict == "IDENTITY_OR_MEASUREMENT_CONFLICT"

    with pytest.raises(ValueError, match="must not exceed 500"):
        run_fada_coverage_diagnostic(
            _PostHistoryFallEnv(),
            student_policy=student,
            teacher_policy=_Oracle(),
            config=config,
            max_steps=501,
            spec=FADACollectionSpec(collect_oracle_shadow=True),
        )


def test_fada_source_diagnostic_checkout_identity_is_json_serializable() -> None:
    module = _load_fada_source_diagnostic_script()
    identity = module._checkout_identity()

    assert identity["head"]
    assert len(identity["tracked_diff_sha256"]) == 64
    assert identity["tracked_diff_bytes"] > 0
    assert "scripts/diagnose_fada_source_coverage.py" in identity["bound_source_sha256"]
    assert json.dumps(identity, sort_keys=True)


def test_walking_recovery_prefix_tracks_early_deployment_history() -> None:
    config = _config()
    rollout_policy = FADAPlannerIDMPolicy(config)
    with torch.no_grad():
        for parameter in rollout_policy.parameters():
            parameter.zero_()
        rollout_policy.idm.action_head.bias.fill_(0.5)

    result = collect_fada_source_windows(
        _FakeEnv(),
        teacher_policy=_Oracle(),
        rollout_policy=rollout_policy,
        config=config,
        num_windows=2,
        spec=FADACollectionSpec(
            collect_oracle_shadow=True,
            command_scenario="walk",
            cold_start_windows=True,
            max_env_steps=2,
        ),
    )

    assert result.env_steps == 1
    torch.testing.assert_close(result.batch.action_history[0], torch.zeros((2, 2)))
    torch.testing.assert_close(
        result.batch.action_history[1],
        torch.tensor([[0.0, 0.0], [0.5, 0.5]]),
    )
    torch.testing.assert_close(
        result.batch.observation_history[1],
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 1.0]]),
    )


def test_collector_rejects_command_change() -> None:
    config = _config()
    env = _FakeEnv(
        command_schedule=[
            (0.0, 0.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
        ],
    )
    result = collect_fada_source_windows(
        env,
        teacher_policy=_Oracle(),
        config=config,
        num_windows=1,
        spec=FADACollectionSpec(max_env_steps=12),
    )

    assert result.rejected_command_windows == 1
    torch.testing.assert_close(result.batch.command, torch.tensor([[1.0, 0.0]]))


def test_command_curriculum_allocation_is_exact_and_stable() -> None:
    assert allocate_fada_command_scenarios(
        10,
        {"walk": 0.5, "static_stand": 0.25, "walk_to_stand": 0.25},
    ) == (("walk", 5), ("static_stand", 3), ("walk_to_stand", 2))
    with pytest.raises(ValueError, match="sum to 1"):
        allocate_fada_command_scenarios(
            10,
            {"walk": 0.5, "static_stand": 0.5, "walk_to_stand": 0.5},
        )
    with pytest.raises(ValueError, match="at least one window"):
        allocate_fada_command_scenarios(
            2,
            {"walk": 0.5, "static_stand": 0.25, "walk_to_stand": 0.25},
        )


def test_static_stand_windows_use_zero_command_and_unified_oracle() -> None:
    result = collect_fada_source_windows(
        _CommandControlledEnv(),
        teacher_policy=_Oracle(),
        config=_curriculum_config(),
        num_windows=2,
        spec=FADACollectionSpec(
            command_scenario="static_stand",
            collect_oracle_shadow=True,
        ),
    )

    assert result.command_scenario == "static_stand"
    assert result.oracle_role == "unified"
    torch.testing.assert_close(result.batch.command, torch.zeros((2, 3)))
    torch.testing.assert_close(
        result.batch.executed_action_chunk[:, 0], result.batch.oracle_first_action
    )
    assert bool(result.batch.oracle_shadow_valid.all())
    torch.testing.assert_close(
        result.batch.oracle_action_chunk[:, 0], result.batch.oracle_first_action
    )


def test_walk_to_stand_admits_active_history_with_zero_future_and_unified_oracle() -> None:
    env = _CommandControlledEnv()
    result = collect_fada_source_windows(
        env,
        teacher_policy=_Oracle(),
        config=_curriculum_config(),
        num_windows=2,
        spec=FADACollectionSpec(
            command_scenario="walk_to_stand",
            transition_walk_command=(0.4, 0.0, 0.0),
            transition_pre_switch_steps=2,
            transition_post_switch_steps=3,
            max_env_steps=20,
        ),
    )

    assert result.command_scenario == "walk_to_stand"
    assert result.oracle_role == "unified"
    assert result.rejected_command_windows > 0
    assert any(bool(np.any(np.abs(command) > 0.0)) for command in env.command_history)
    assert any(bool(np.all(command == 0.0)) for command in env.command_history)
    torch.testing.assert_close(result.batch.command, torch.zeros((2, 3)))
    assert torch.isfinite(result.batch.oracle_first_action).all()


def test_collector_clears_history_at_episode_boundary() -> None:
    config = _config()
    result = collect_fada_source_windows(
        _FakeEnv(done_steps=(1,)),
        teacher_policy=_Oracle(),
        config=config,
        num_windows=1,
        spec=FADACollectionSpec(max_env_steps=12),
    )

    assert result.rejected_done_transitions == 1
    assert result.batch.observation_history[0, 0, 2] >= 0.0
