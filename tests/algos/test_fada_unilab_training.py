from __future__ import annotations

import importlib.util
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
from unilab.algos.torch.distill import (
    FADAArchitectureConfig,
    FADACollectionSpec,
    FADAPlannerIDMPolicy,
    FADAReplayBuffer,
    FADATrainer,
    collect_fada_source_windows,
    evaluate_fada_source_batch,
    load_fada_checkpoint,
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
from unilab.algos.torch.distill.fada_training import FADAPaperSourcePlan

ROOT = Path(__file__).resolve().parents[2]


def _load_train_distill() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_fada_train_distill",
        ROOT / "scripts" / "train_distill.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=3,
        action_dim=2,
        command_dim=2,
        history_length=2,
        prediction_horizon=2,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


class _Oracle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.obs_dim = 3

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, :2] + 0.25


@dataclass
class _State:
    obs: dict[str, np.ndarray]
    info: dict[str, np.ndarray]
    terminated: np.ndarray
    truncated: np.ndarray


class _FakeEnv:
    def __init__(
        self,
        *,
        command_schedule: list[tuple[float, float]] | None = None,
        done_steps: tuple[int, ...] = (),
    ) -> None:
        self.num_envs = 1
        self.state = object()
        self.action_space = type("ActionSpace", (), {"shape": (2,)})()
        self.command_schedule = command_schedule or [(0.4, -0.1)]
        self.done_steps = set(done_steps)
        self.step_count = 0
        self.current_obs = np.zeros((1, 3), dtype=np.float32)
        self.closed = False

    def _command(self) -> np.ndarray:
        index = min(self.step_count, len(self.command_schedule) - 1)
        return np.asarray([self.command_schedule[index]], dtype=np.float32)

    def reset(self, _indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        self.current_obs = np.zeros((1, 3), dtype=np.float32)
        return {"obs": self.current_obs.copy()}, {"commands": self._command()}

    def step(self, actions: np.ndarray) -> _State:
        self.current_obs = self.current_obs + np.concatenate(
            [actions, np.ones((1, 1), dtype=np.float32)], axis=1
        )
        self.step_count += 1
        done = np.asarray([self.step_count in self.done_steps], dtype=np.bool_)
        return _State(
            obs={"obs": self.current_obs.copy()},
            info={"commands": self._command()},
            terminated=done,
            truncated=np.zeros_like(done),
        )

    def close(self) -> None:
        self.closed = True

    @contextmanager
    def preserve_rollout_state(self):
        current_obs = self.current_obs.copy()
        step_count = self.step_count
        try:
            yield
        finally:
            self.current_obs = current_obs
            self.step_count = step_count


class _CommandControlledEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.action_space = type("ActionSpace", (), {"shape": (2,)})()
        self.step_count = 0
        self.current_obs = np.zeros((1, 3), dtype=np.float32)
        self.command_history: list[np.ndarray] = []
        self.closed = False
        self.state = self._state(np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32))

    def _state(self, commands: np.ndarray) -> _State:
        return _State(
            obs={"obs": self.current_obs.copy()},
            info={"commands": commands.copy()},
            terminated=np.zeros((1,), dtype=np.bool_),
            truncated=np.zeros((1,), dtype=np.bool_),
        )

    def reset_all(self) -> _State:
        self.current_obs = np.zeros((1, 3), dtype=np.float32)
        self.step_count = 0
        self.state = self._state(np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32))
        return self.state

    def refresh_state(self) -> _State:
        commands = np.asarray(self.state.info["commands"], dtype=np.float32)
        self.state = self._state(commands)
        return self.state

    def step(self, actions: np.ndarray) -> _State:
        commands = np.asarray(self.state.info["commands"], dtype=np.float32).copy()
        self.command_history.append(commands.copy())
        self.current_obs = self.current_obs + np.concatenate(
            [actions, np.ones((1, 1), dtype=np.float32)], axis=1
        )
        self.step_count += 1
        self.state = self._state(commands)
        return self.state

    @contextmanager
    def preserve_rollout_state(self):
        current_obs = self.current_obs.copy()
        step_count = self.step_count
        state = self._state(np.asarray(self.state.info["commands"], dtype=np.float32))
        command_history = list(self.command_history)
        try:
            yield
        finally:
            self.current_obs = current_obs
            self.step_count = step_count
            self.state = state
            self.command_history = command_history

    def close(self) -> None:
        self.closed = True


class _StandingOracle(_Oracle):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0], 2), 0.75, device=obs.device)


def _curriculum_config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=3,
        action_dim=2,
        command_dim=3,
        history_length=2,
        prediction_horizon=2,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


def _source_batch(config: FADAArchitectureConfig, size: int = 4) -> FADASourceBatch:
    return FADASourceBatch(
        observation_history=torch.randn(size, config.history_length, config.obs_dim),
        action_history=torch.randn(size, config.history_length, config.action_dim),
        command=torch.randn(size, config.command_dim),
        realized_future=torch.randn(size, config.prediction_horizon, config.obs_dim),
        executed_action_chunk=torch.randn(size, config.prediction_horizon, config.action_dim),
        oracle_future=torch.randn(size, config.prediction_horizon, config.obs_dim),
        oracle_action_chunk=torch.randn(size, config.prediction_horizon, config.action_dim),
        oracle_shadow_valid=torch.ones(size, dtype=torch.bool),
        oracle_first_action=torch.randn(size, config.action_dim),
        command_scenario=torch.zeros(size, dtype=torch.int64),
        planner_eligible=torch.ones(size, dtype=torch.bool),
        cold_start=torch.zeros(size, dtype=torch.bool),
    )


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


def test_static_collector_builds_exact_reset_aligned_cold_start_window() -> None:
    config = _curriculum_config()
    result = collect_fada_source_windows(
        _CommandControlledEnv(),
        teacher_policy=_Oracle(),
        standing_teacher_policy=_StandingOracle(),
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
            cold_start=torch.tensor([False] * 4 + [True, True, False, False] + [False] * 12),
        )
    )

    sampled = replay.sample_planner(
        40,
        scenario_ratios={"walk": 0.5, "static_stand": 0.25, "walk_to_stand": 0.25},
        static_cold_start_ratio=0.5,
        generator=torch.Generator().manual_seed(5),
    )

    assert bool(sampled.planner_eligible.all())
    assert int((sampled.command_scenario == FADA_SCENARIO_IDS["walk"]).sum()) == 20
    static = sampled.command_scenario == FADA_SCENARIO_IDS["static_stand"]
    assert int(static.sum()) == 10
    assert int((static & sampled.cold_start).sum()) == 5
    assert int((sampled.command_scenario == FADA_SCENARIO_IDS["walk_to_stand"]).sum()) == 10


def test_planner_replay_fails_closed_when_required_stratum_is_missing() -> None:
    config = _curriculum_config()
    replay = FADAReplayBuffer(config, capacity=8)
    replay.add(_source_batch(config, size=8))

    with pytest.raises(ValueError, match="static_stand/cold_start"):
        replay.sample_planner(
            8,
            scenario_ratios={"walk": 0.5, "static_stand": 0.25, "walk_to_stand": 0.25},
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
        cold_start=torch.tensor([False, False, True, True, False, False, False, False]),
    )

    metrics = evaluate_fada_source_batch(
        FADAPlannerIDMPolicy(config),
        batch,
        require_scenario_metrics=True,
    )

    assert metrics["scenario/walk/row_fraction"] == pytest.approx(0.25)
    assert metrics["scenario/static_stand/row_fraction"] == pytest.approx(0.5)
    assert metrics["scenario/walk_to_stand/row_fraction"] == pytest.approx(0.25)
    assert metrics["scenario/static_stand/cold_start_fraction"] == pytest.approx(0.5)
    assert "scenario/static_stand/cold_start_planner_mse" in metrics
    assert "scenario/static_stand/steady_state_planner_mse" in metrics


def test_paper_source_preflight_requires_exact_intermediate_oracle_set(tmp_path: Path) -> None:
    module = _load_train_distill()
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

    plan = module._paper_source_plan(cfg)
    assert plan.checkpoint_paths == tuple(checkpoints)
    assert sum(windows for _, windows in plan.source_allocations) == 20
    assert all(windows == 1 for _, windows in plan.source_allocations)
    cfg.training.fada.intermediate_oracle_count = 19
    with pytest.raises(ValueError, match="intermediate_oracle_count=20"):
        module._paper_source_plan(cfg)
    cfg.training.fada.intermediate_oracle_count = 20
    cfg.training.fada.intermediate_oracle_checkpoint_paths = [
        str(path) for path in checkpoints[:-1]
    ]
    with pytest.raises(ValueError, match="exactly 20 unique"):
        module._paper_source_plan(cfg)


def test_standing_curriculum_rejects_legacy_execution_mode() -> None:
    module = _load_train_distill()
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
        module._fada_execution_mode(cfg)
    cfg.training.fada.stand_transition_curriculum.enabled = False
    assert module._fada_execution_mode(cfg) == "legacy"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, "cpu"), ("", "cpu"), ("cuda:0", "cuda:0")],
)
def test_fada_async_runtime_normalizes_device(configured: str | None, expected: str) -> None:
    cfg = OmegaConf.create({"training": {"device": configured}})
    assert _fada_runtime_device(cfg) == expected


def test_v005_replay_settings_reject_contract_ratio_drift() -> None:
    module = _load_train_distill()
    fada = OmegaConf.create(
        {
            "v005_replay": {
                "enabled": True,
                "static_cold_start_ratio": 0.5,
                "planner_scenario_ratios": {
                    "walk": 0.5,
                    "static_stand": 0.25,
                    "walk_to_stand": 0.25,
                },
            }
        }
    )
    assert module._fada_v005_replay_settings(fada, batch_size=8)[0] is True

    fada.v005_replay.planner_scenario_ratios.walk = 0.4
    with pytest.raises(ValueError, match="scenario ratios are fixed"):
        module._fada_v005_replay_settings(fada, batch_size=8)
    fada.v005_replay.planner_scenario_ratios.walk = 0.5
    fada.v005_replay.static_cold_start_ratio = 0.25
    with pytest.raises(ValueError, match="static_cold_start_ratio is fixed"):
        module._fada_v005_replay_settings(fada, batch_size=8)


def test_parent_curriculum_artifact_guard_rejects_oracle_role_drift() -> None:
    module = _load_train_distill()
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
    module._require_fada_curriculum_artifact(cfg, metadata)
    metadata["collections"][2]["oracle_role"] = "walking"
    with pytest.raises(ValueError, match="Oracle role mismatch"):
        module._require_fada_curriculum_artifact(cfg, metadata)


def test_v005_parent_artifact_guard_requires_row_provenance() -> None:
    module = _load_train_distill()
    config = _curriculum_config()
    cfg = OmegaConf.create(
        {
            "student": {"obs_dim": config.obs_dim, "action_dim": config.action_dim},
            "training": {
                "fada": {
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
                "source": "optimal_or_current_policy",
                "command_scenario": "walk",
                "oracle_role": "walking",
                "windows": 4,
            },
            {
                "source": "optimal_or_current_policy",
                "command_scenario": "static_stand",
                "oracle_role": "standing",
                "windows": 2,
            },
            {
                "source": "optimal_or_current_policy",
                "command_scenario": "walk_to_stand",
                "oracle_role": "standing",
                "windows": 2,
            },
            {
                "source": "intermediate_oracle",
                "command_scenario": "walk",
                "oracle_role": "walking",
                "windows": 2,
            },
        ],
    }
    batch = replace(
        _source_batch(config, size=10),
        command_scenario=torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 0, 0], dtype=torch.int64),
        planner_eligible=torch.tensor([True] * 8 + [False, False], dtype=torch.bool),
        cold_start=torch.tensor(
            [False, False, False, False, True, False, False, False, False, False],
            dtype=torch.bool,
        ),
    )

    module._require_fada_curriculum_artifact(cfg, metadata, batch)
    with pytest.raises(ValueError, match="requires row-level source identity"):
        module._require_fada_curriculum_artifact(cfg, metadata)
    invalid = replace(batch, planner_eligible=torch.ones(10, dtype=torch.bool))
    with pytest.raises(ValueError, match="intermediate-Oracle rows"):
        module._require_fada_curriculum_artifact(cfg, metadata, invalid)


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


def test_static_stand_windows_use_zero_command_and_standing_oracle() -> None:
    result = collect_fada_source_windows(
        _CommandControlledEnv(),
        teacher_policy=_Oracle(),
        standing_teacher_policy=_StandingOracle(),
        config=_curriculum_config(),
        num_windows=2,
        spec=FADACollectionSpec(
            command_scenario="static_stand",
            collect_oracle_shadow=True,
        ),
    )

    assert result.command_scenario == "static_stand"
    assert result.oracle_role == "standing"
    torch.testing.assert_close(result.batch.command, torch.zeros((2, 3)))
    torch.testing.assert_close(result.batch.executed_action_chunk[:, 0], torch.full((2, 2), 0.75))
    torch.testing.assert_close(result.batch.oracle_first_action, torch.full((2, 2), 0.75))
    assert bool(result.batch.oracle_shadow_valid.all())
    torch.testing.assert_close(result.batch.oracle_action_chunk[:, 0], torch.full((2, 2), 0.75))


def test_walk_to_stand_admits_active_history_with_zero_future_and_standing_oracle() -> None:
    env = _CommandControlledEnv()
    result = collect_fada_source_windows(
        env,
        teacher_policy=_Oracle(),
        standing_teacher_policy=_StandingOracle(),
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
    assert result.oracle_role == "standing"
    assert result.rejected_command_windows > 0
    assert any(bool(np.any(np.abs(command) > 0.0)) for command in env.command_history)
    assert any(bool(np.all(command == 0.0)) for command in env.command_history)
    torch.testing.assert_close(result.batch.command, torch.zeros((2, 3)))
    torch.testing.assert_close(result.batch.oracle_first_action, torch.full((2, 2), 0.75))


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


def test_replay_trainer_and_checkpoint_keep_paired_owners(tmp_path: Path) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )
    replay = FADAReplayBuffer(config, capacity=5)
    replay.add(_source_batch(config, size=7))
    assert len(replay) == 5

    stats = trainer.update(replay.sample(3), idm_updates=1, planner_updates=1)
    assert stats.idm_grad_norm > 0.0
    assert stats.planner_grad_norm > 0.0
    assert all(parameter.grad is None for parameter in policy.idm.parameters())

    checkpoint = tmp_path / "fada.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=2,
        samples_seen=10,
        runtime_config={"enabled": True},
        quality_metrics={"planner_idm_oracle_action_mse": 0.25},
    )
    restored_policy = FADAPlannerIDMPolicy(config)
    restored_trainer = FADATrainer(
        restored_policy,
        idm_optimizer=torch.optim.Adam(restored_policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(restored_policy.planner.parameters(), lr=1.0e-3),
    )
    payload = load_fada_checkpoint(checkpoint, restored_policy, restored_trainer)
    assert payload["completed_iterations"] == 2
    assert payload["schema_version"] == 2
    assert payload["quality_metrics"] == {"planner_idm_oracle_action_mse": 0.25}
    for expected, observed in zip(policy.parameters(), restored_policy.parameters(), strict=True):
        torch.testing.assert_close(expected, observed)


def test_fada_resume_checkpoint_uses_weights_only_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )
    checkpoint = tmp_path / "safe-resume.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=0,
        samples_seen=0,
        runtime_config={},
    )
    original_load = torch.load
    weights_only_values: list[object] = []

    def traced_load(*args, **kwargs):
        weights_only_values.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(fada_training.torch, "load", traced_load)
    load_fada_checkpoint(checkpoint, policy, trainer)

    assert weights_only_values == [True]


def test_fada_source_artifact_round_trip_validates_architecture(tmp_path: Path) -> None:
    config = _config()
    batch = _source_batch(config, size=3)
    artifact = tmp_path / "source.pt"
    save_fada_source_batch(
        artifact,
        batch,
        config=config,
        metadata={"iteration": 2, "main_windows": 1},
    )

    loaded = load_fada_source_batch(artifact, config=config)
    assert loaded.metadata == {"iteration": 2, "main_windows": 1}
    torch.testing.assert_close(loaded.batch.command, batch.command)

    incompatible = FADAArchitectureConfig(
        **{**config.__dict__, "command_dim": config.command_dim + 1}
    )
    with pytest.raises(ValueError, match="architecture mismatch"):
        load_fada_source_batch(artifact, config=incompatible)


def test_v005_checkpoint_serializer_requires_finite_scenario_metrics(tmp_path: Path) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )
    runtime_config = {"v005_replay": {"enabled": True}}
    required = {
        "scenario/walk/planner_idm_oracle_action_mse": 0.1,
        "scenario/static_stand/planner_idm_oracle_action_mse": 0.2,
        "scenario/walk_to_stand/planner_idm_oracle_action_mse": 0.3,
        "scenario/static_stand/cold_start_fraction": 0.5,
        "scenario/static_stand/cold_start_planner_mse": 0.4,
        "scenario/static_stand/steady_state_planner_mse": 0.2,
    }

    with pytest.raises(ValueError, match="quality metrics are missing"):
        save_fada_checkpoint(
            tmp_path / "missing.pt",
            policy,
            trainer,
            completed_iterations=1,
            samples_seen=1,
            runtime_config=runtime_config,
            quality_metrics={},
        )
    with pytest.raises(ValueError, match="non-finite"):
        save_fada_checkpoint(
            tmp_path / "nonfinite.pt",
            policy,
            trainer,
            completed_iterations=1,
            samples_seen=1,
            runtime_config=runtime_config,
            quality_metrics={**required, "extra": float("nan")},
        )

    checkpoint = tmp_path / "valid.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=1,
        samples_seen=1,
        runtime_config=runtime_config,
        quality_metrics=required,
    )
    payload = load_fada_checkpoint(checkpoint, FADAPlannerIDMPolicy(config))
    assert payload["quality_metrics"] == required


def test_fada_persistent_worker_collects_one_versioned_iteration_artifact(
    tmp_path: Path,
) -> None:
    config = _config()
    worker = PersistentFADACollectorWorker.__new__(PersistentFADACollectorWorker)
    worker.config = config
    worker.device = "cpu"
    worker.cfg = OmegaConf.create(
        {
            "training": {
                "fada": {
                    "windows_per_iteration": 1,
                    "oracle_shadow_enabled": True,
                    "observation_key": "obs",
                    "teacher_projection": "identity",
                    "student_projection": "identity",
                    "student_drop_index": None,
                    "command_info_keys": ["commands"],
                    "max_env_steps": 12,
                }
            }
        }
    )
    worker.env = _FakeEnv()
    worker.student = FADAPlannerIDMPolicy(config)
    worker.final_teacher = _Oracle()
    worker.teacher_spec = object()
    worker.source_allocations = ((str(tmp_path / "intermediate.pt"), 1),)
    worker._teacher_loader = lambda *_args, **_kwargs: _Oracle()

    class _WeightSync:
        def read_weights_into(self, _state_dict) -> int:
            return 7

        def close(self) -> None:
            return None

    worker.weight_sync = _WeightSync()
    output = tmp_path / "iteration.pt"
    request = DaggerCollectRequest(
        request_id="fada-1-v7",
        scenario=FADA_ASYNC_SCENARIO,
        iteration=1,
        checkpoint_path=str((tmp_path / "fada.pt").resolve()),
        output_path=str(output.resolve()),
        expected_weight_version=7,
    )

    result = worker.collect(request)
    loaded = load_fada_source_batch(output, config=config)

    assert result.observed_weight_version == 7
    assert result.num_samples == 2
    assert loaded.metadata["main_windows"] == 1
    assert [item["rollout_mode"] for item in loaded.metadata["collections"]] == [
        "planner_idm",
        "intermediate_oracle",
    ]


def test_fada_worker_constructor_rolls_back_partial_resident_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()

    class _WeightSync:
        instances: list[_WeightSync] = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.close_count = 0
            self.instances.append(self)

        def read_weights_into(self, _state_dict) -> int:
            return 1

        def close(self) -> None:
            self.close_count += 1

    class _Env:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    class _BackendAdapter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def build_task_env_cfg_override(self) -> dict[str, object]:
            return {}

    walking_env = _Env()
    env_calls = 0

    def env_factory(*_args, **_kwargs):
        nonlocal env_calls
        env_calls += 1
        if env_calls == 1:
            return walking_env
        raise RuntimeError("standing environment construction failed")

    monkeypatch.setattr(
        fada_async_runtime,
        "load_fada_policy_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(policy=FADAPlannerIDMPolicy(config)),
    )
    monkeypatch.setattr(fada_async_runtime, "SharedWeightSync", _WeightSync)
    monkeypatch.setattr(fada_async_runtime, "BackendAdapter", _BackendAdapter)
    monkeypatch.setattr(fada_async_runtime, "ensure_registries", lambda **_kwargs: None)

    cfg_payload = {
        "teacher": {
            "algo_type": "sac",
            "obs_dim": config.obs_dim,
            "action_dim": config.action_dim,
            "actor_hidden_dim": 8,
            "use_layer_norm": False,
            "obs_normalization": False,
        },
        "training": {
            "task_name": "G1WalkFlat",
            "sim_backend": "mujoco",
            "fada": {"num_envs": 1},
        },
    }
    standing_cfg_payload = {"training": {"task_name": "G1StandStill", "sim_backend": "mujoco"}}

    with pytest.raises(RuntimeError, match="standing environment construction failed"):
        PersistentFADACollectorWorker(
            root_dir=str(ROOT),
            cfg_payload=cfg_payload,
            standing_cfg_payload=standing_cfg_payload,
            architecture=asdict(config),
            final_teacher_checkpoint="walking.pt",
            standing_teacher_checkpoint="standing.pt",
            source_allocations=(),
            initial_checkpoint_path="student.pt",
            device="cpu",
            weight_sync_name="test-sync",
            weight_sync_lock=object(),
            weight_param_shapes={},
            env_factory=env_factory,
            teacher_loader=lambda *_args, **_kwargs: _Oracle(),
        )

    assert walking_env.close_count == 1
    assert len(_WeightSync.instances) == 1
    assert _WeightSync.instances[0].close_count == 1


def test_fada_persistent_worker_collects_v005_cold_and_steady_standing_artifact(
    tmp_path: Path,
) -> None:
    config = _curriculum_config()
    worker = PersistentFADACollectorWorker.__new__(PersistentFADACollectorWorker)
    worker.config = config
    worker.device = "cpu"
    worker.cfg = OmegaConf.create(
        {
            "training": {
                "fada": {
                    "windows_per_iteration": 6,
                    "oracle_shadow_enabled": False,
                    "observation_key": "obs",
                    "teacher_projection": "identity",
                    "student_projection": "identity",
                    "student_drop_index": None,
                    "command_info_keys": ["commands"],
                    "max_env_steps": 30,
                    "stand_transition_curriculum": {
                        "enabled": True,
                        "walk_ratio": 0.5,
                        "static_stand_ratio": 0.25,
                        "walk_to_stand_ratio": 0.25,
                        "walk_command": [0.4, 0.0, 0.0],
                        "pre_switch_steps": 2,
                        "post_switch_steps": 3,
                    },
                    "v005_replay": {
                        "enabled": True,
                        "static_cold_start_ratio": 0.5,
                    },
                }
            }
        }
    )
    worker.env = _CommandControlledEnv()
    worker.standing_env = _CommandControlledEnv()
    worker.student = FADAPlannerIDMPolicy(config)
    worker.final_teacher = _Oracle()
    worker.standing_teacher = _StandingOracle()
    worker.teacher_spec = object()
    worker.source_allocations = ()

    class _WeightSync:
        def read_weights_into(self, _state_dict) -> int:
            return 4

        def close(self) -> None:
            return None

    worker.weight_sync = _WeightSync()
    output = tmp_path / "curriculum.pt"
    result = worker.collect(
        DaggerCollectRequest(
            request_id="fada-0-v4",
            scenario=FADA_ASYNC_SCENARIO,
            iteration=0,
            checkpoint_path=str((tmp_path / "fada.pt").resolve()),
            output_path=str(output.resolve()),
            expected_weight_version=4,
        )
    )
    loaded = load_fada_source_batch(output, config=config)

    assert result.num_samples == 6
    assert loaded.metadata["scenario_allocations"] == {
        "walk": 3,
        "static_stand": 2,
        "walk_to_stand": 1,
    }
    assert [item["command_scenario"] for item in loaded.metadata["collections"]] == [
        "walk",
        "static_stand",
        "static_stand",
        "walk_to_stand",
    ]
    assert [item["oracle_role"] for item in loaded.metadata["collections"]] == [
        "walking",
        "standing",
        "standing",
        "standing",
    ]
    assert [item["window_profile"] for item in loaded.metadata["collections"]] == [
        "steady_state",
        "cold_start",
        "steady_state",
        "steady_state",
    ]
    assert loaded.metadata["v005_replay_enabled"] is True
    static = loaded.batch.command_scenario == FADA_SCENARIO_IDS["static_stand"]
    assert int(static.sum()) == 2
    assert int((static & loaded.batch.cold_start).sum()) == 1
    assert worker.standing_env.step_count > 0
    assert worker.env.step_count > 0


def test_enabled_worker_curriculum_rejects_missing_standing_oracle(tmp_path: Path) -> None:
    worker = PersistentFADACollectorWorker.__new__(PersistentFADACollectorWorker)
    worker.config = _curriculum_config()
    worker.device = "cpu"
    worker.cfg = OmegaConf.create(
        {
            "training": {
                "fada": {
                    "windows_per_iteration": 3,
                    "oracle_shadow_enabled": False,
                    "observation_key": "obs",
                    "teacher_projection": "identity",
                    "student_projection": "identity",
                    "student_drop_index": None,
                    "command_info_keys": ["commands"],
                    "max_env_steps": 20,
                    "stand_transition_curriculum": {
                        "enabled": True,
                        "walk_ratio": 1 / 3,
                        "static_stand_ratio": 1 / 3,
                        "walk_to_stand_ratio": 1 / 3,
                    },
                }
            }
        }
    )
    worker.env = _CommandControlledEnv()
    worker.standing_env = _CommandControlledEnv()
    worker.student = FADAPlannerIDMPolicy(worker.config)
    worker.final_teacher = _Oracle()
    worker.standing_teacher = None
    worker.source_allocations = ()

    class _WeightSync:
        def read_weights_into(self, _state_dict) -> int:
            return 1

    worker.weight_sync = _WeightSync()
    with pytest.raises(ValueError, match="loaded standing Oracle"):
        worker.collect(
            DaggerCollectRequest(
                request_id="fada-0-v1",
                scenario=FADA_ASYNC_SCENARIO,
                iteration=0,
                checkpoint_path=str((tmp_path / "fada.pt").resolve()),
                output_path=str((tmp_path / "missing.pt").resolve()),
                expected_weight_version=1,
            )
        )


def test_enabled_worker_curriculum_rejects_missing_standing_environment(
    tmp_path: Path,
) -> None:
    worker = PersistentFADACollectorWorker.__new__(PersistentFADACollectorWorker)
    worker.config = _curriculum_config()
    worker.device = "cpu"
    worker.cfg = OmegaConf.create(
        {
            "training": {
                "fada": {
                    "windows_per_iteration": 3,
                    "oracle_shadow_enabled": False,
                    "observation_key": "obs",
                    "teacher_projection": "identity",
                    "student_projection": "identity",
                    "student_drop_index": None,
                    "command_info_keys": ["commands"],
                    "max_env_steps": 20,
                    "stand_transition_curriculum": {
                        "enabled": True,
                        "walk_ratio": 1 / 3,
                        "static_stand_ratio": 1 / 3,
                        "walk_to_stand_ratio": 1 / 3,
                    },
                }
            }
        }
    )
    worker.env = _CommandControlledEnv()
    worker.standing_env = None
    worker.student = FADAPlannerIDMPolicy(worker.config)
    worker.final_teacher = _Oracle()
    worker.standing_teacher = _StandingOracle()
    worker.source_allocations = ()

    class _WeightSync:
        def read_weights_into(self, _state_dict) -> int:
            return 1

    worker.weight_sync = _WeightSync()
    with pytest.raises(ValueError, match="G1StandStill environment"):
        worker.collect(
            DaggerCollectRequest(
                request_id="fada-0-v1",
                scenario=FADA_ASYNC_SCENARIO,
                iteration=0,
                checkpoint_path=str((tmp_path / "fada.pt").resolve()),
                output_path=str((tmp_path / "missing-env.pt").resolve()),
                expected_weight_version=1,
            )
        )


def test_persistent_runtime_rejects_missing_standing_checkpoint_before_worker_start(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "device": "cpu",
                "fada": {
                    "windows_per_iteration": 4,
                    "command_info_keys": ["commands"],
                    "stand_transition_curriculum": {
                        "enabled": True,
                        "standing_teacher_checkpoint_path": str(tmp_path / "missing.pt"),
                        "walk_ratio": 0.5,
                        "static_stand_ratio": 0.25,
                        "walk_to_stand_ratio": 0.25,
                        "walk_command": [0.4, 0.0, 0.0],
                        "pre_switch_steps": 2,
                        "post_switch_steps": 2,
                    },
                },
            }
        }
    )
    with pytest.raises(FileNotFoundError, match="standing Oracle checkpoint"):
        build_persistent_fada_runtime(
            cfg=cfg,
            architecture=_curriculum_config(),
            paper_source_plan=FADAPaperSourcePlan(enabled=False, source_allocations=()),
            final_teacher_checkpoint=tmp_path / "walking.pt",
            request_timeout_seconds=1.0,
        )


def test_unilab_fada_persistent_async_keeps_collection_behind_version_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_train_distill()
    checkpoint = tmp_path / "fada_async.pt"
    artifact_dir = tmp_path / "artifacts"
    cfg = OmegaConf.create(
        {
            "student": {"obs_dim": 3, "action_dim": 2},
            "teacher": {
                "obs_dim": 3,
                "action_dim": 2,
                "algo_type": "sac",
                "actor_hidden_dim": 8,
                "use_layer_norm": False,
                "obs_normalization": False,
            },
            "training": {
                "task_name": "FakeTask",
                "device": "cpu",
                "sim_backend": "mujoco",
                "fada": {
                    "enabled": True,
                    "execution_mode": "persistent_async",
                    "async_artifact_dir": str(artifact_dir),
                    "async_request_timeout_seconds": 10.0,
                    "paper_source_enabled": False,
                    "oracle_shadow_enabled": True,
                    "history_length": 2,
                    "prediction_horizon": 2,
                    "command_dim": 2,
                    "hidden_dim": 8,
                    "num_heads": 2,
                    "planner_layers": 1,
                    "idm_encoder_layers": 1,
                    "idm_decoder_layers": 1,
                    "feedforward_dim": 16,
                    "dropout": 0.0,
                    "iterations": 2,
                    "windows_per_iteration": 1,
                    "num_envs": 1,
                    "replay_capacity": 4,
                    "batch_size": 1,
                    "idm_updates": 1,
                    "planner_updates": 1,
                    "idm_learning_rate": 0.001,
                    "planner_learning_rate": 0.001,
                    "max_grad_norm": 1.0,
                    "observation_key": "obs",
                    "teacher_projection": "identity",
                    "student_projection": "identity",
                    "student_drop_index": None,
                    "command_info_keys": ["commands"],
                    "max_env_steps": 12,
                    "quality_eval_max_windows": 1,
                    "checkpoint_path": str(checkpoint),
                    "resume_path": None,
                },
            },
        }
    )
    config = _config()

    class _Runtime:
        def __init__(self) -> None:
            self.activations: list[str] = []
            self.closed = False

        def activate_checkpoint(self, path: Path) -> int:
            self.activations.append(str(path))
            return len(self.activations)

        def collect(self, request):
            batch = _source_batch(config, size=1)
            summary = {
                "iteration": request.iteration,
                "source": "optimal_or_current_policy",
                "rollout_mode": "oracle" if request.iteration == 0 else "planner_idm",
                "windows": 1,
                "env_steps": 1,
                "rejected_done_transitions": 0,
                "rejected_command_windows": 0,
            }
            save_fada_source_batch(
                request.output_path,
                batch,
                config=config,
                metadata={
                    "iteration": request.iteration,
                    "main_windows": 1,
                    "collections": [summary],
                },
            )
            return type(
                "Result",
                (),
                {"num_samples": 1},
            )()

        def close(self) -> None:
            self.closed = True

    runtime = _Runtime()
    monkeypatch.setattr(module, "_require_teacher_policy_collection_route", lambda _cfg: None)
    monkeypatch.setattr(module, "_apply_collect_command_distribution_overrides", lambda _cfg: {})
    monkeypatch.setattr(module, "build_persistent_fada_runtime", lambda **_kwargs: runtime)

    result = module.run_fada_training(cfg, teacher_checkpoint=tmp_path / "oracle.pt")

    assert result["execution_mode"] == "persistent_async"
    assert [item["rollout_mode"] for item in result["collections"]] == [
        "oracle",
        "planner_idm",
    ]
    assert result["samples_seen"] == 2
    assert len(runtime.activations) == 2
    assert runtime.closed is True


def test_main_flag_off_preserves_legacy_dispatch_and_on_has_priority(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_train_distill()
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "run_single_entry_workflow",
        lambda _cfg: calls.append("legacy") or {"route": "legacy"},
    )
    monkeypatch.setattr(
        module,
        "run_fada_training",
        lambda _cfg: calls.append("fada") or {"route": "fada"},
    )
    invoke = module.main.__wrapped__

    off_cfg = OmegaConf.create(
        {"training": {"fada": {"enabled": False}, "workflow": {"enabled": True}}}
    )
    invoke(off_cfg)
    assert calls == ["legacy"]
    assert '"route": "legacy"' in capsys.readouterr().out

    on_cfg = OmegaConf.create(
        {"training": {"fada": {"enabled": True}, "workflow": {"enabled": True}}}
    )
    invoke(on_cfg)
    assert calls == ["legacy", "fada"]
    assert '"route": "fada"' in capsys.readouterr().out


def test_unilab_fada_workflow_bootstraps_then_uses_planner_idm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_train_distill()
    checkpoint = tmp_path / "fada.pt"
    cfg = OmegaConf.create(
        {
            "student": {"obs_dim": 3, "action_dim": 2},
            "teacher": {
                "obs_dim": 3,
                "action_dim": 2,
                "algo_type": "sac",
                "actor_hidden_dim": 8,
                "use_layer_norm": False,
                "obs_normalization": False,
            },
            "training": {
                "task_name": "FakeTask",
                "device": "cpu",
                "sim_backend": "mujoco",
                "fada": {
                    "enabled": True,
                    "paper_source_enabled": False,
                    "oracle_shadow_enabled": True,
                    "history_length": 2,
                    "prediction_horizon": 2,
                    "command_dim": 2,
                    "hidden_dim": 8,
                    "num_heads": 2,
                    "planner_layers": 1,
                    "idm_encoder_layers": 1,
                    "idm_decoder_layers": 1,
                    "feedforward_dim": 16,
                    "dropout": 0.0,
                    "iterations": 2,
                    "windows_per_iteration": 1,
                    "num_envs": 1,
                    "replay_capacity": 4,
                    "batch_size": 1,
                    "idm_updates": 1,
                    "planner_updates": 1,
                    "idm_learning_rate": 0.001,
                    "planner_learning_rate": 0.001,
                    "max_grad_norm": 1.0,
                    "observation_key": "obs",
                    "teacher_projection": "identity",
                    "student_projection": "identity",
                    "student_drop_index": None,
                    "command_info_keys": ["commands"],
                    "max_env_steps": 12,
                    "quality_eval_max_windows": 1,
                    "checkpoint_path": str(checkpoint),
                    "resume_path": None,
                },
            },
        }
    )
    monkeypatch.setattr(module, "_require_teacher_policy_collection_route", lambda _cfg: None)
    monkeypatch.setattr(module, "_apply_collect_command_distribution_overrides", lambda _cfg: {})
    monkeypatch.setattr(module, "load_sac_teacher_policy", lambda *_args, **_kwargs: _Oracle())
    env = _FakeEnv()

    result = module.run_fada_training(
        cfg,
        teacher_checkpoint=tmp_path / "oracle.pt",
        create_env_fn=lambda *_args, **_kwargs: env,
        env_cfg_override_fn=lambda _cfg: {},
    )

    assert [item["rollout_mode"] for item in result["collections"]] == [
        "oracle",
        "planner_idm",
    ]
    assert result["completed_iterations"] == 2
    assert result["samples_seen"] == 2
    assert result["quality_metrics"]["rollout_rejected_done_transitions"] == 0.0
    assert result["quality_metrics"]["rollout_rejected_command_windows"] == 0.0
    assert checkpoint.is_file()
    assert env.closed is True
