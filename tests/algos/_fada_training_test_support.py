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

import unilab.algos.torch.distill.fada.async_runtime as fada_async_runtime
import unilab.algos.torch.distill.fada.training as fada_training
import unilab.algos.torch.distill.fada.workflow as fada_workflow
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
from unilab.algos.torch.distill.fada_source_diagnostics import (
    classify_fada_coverage,
    run_fada_coverage_diagnostic,
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


def _load_fada_source_diagnostic_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_fada_source_diagnostic_script",
        ROOT / "scripts" / "diagnose_fada_source_coverage.py",
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

    @contextmanager
    def isolated_rollout_branch(self):
        with self.preserve_rollout_state():
            yield


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

    @contextmanager
    def isolated_rollout_branch(self):
        with self.preserve_rollout_state():
            yield

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
        idm_source_role=torch.zeros(size, dtype=torch.int64),
        oracle_first_action=torch.randn(size, config.action_dim),
        command_scenario=torch.zeros(size, dtype=torch.int64),
        planner_eligible=torch.ones(size, dtype=torch.bool),
        cold_start=torch.zeros(size, dtype=torch.bool),
    )


def _paper_role_batch(
    config: FADAArchitectureConfig,
    *,
    main_rows: int,
    intermediate_rows: int,
) -> FADASourceBatch:
    size = main_rows + intermediate_rows
    return replace(
        _source_batch(config, size=size),
        planner_eligible=torch.tensor(
            [True] * main_rows + [False] * intermediate_rows,
            dtype=torch.bool,
        ),
        idm_source_role=torch.tensor(
            [1] * main_rows + [0] * intermediate_rows,
            dtype=torch.int64,
        ),
    )


def _paper_persistent_training_cfg(tmp_path: Path) -> OmegaConf:
    intermediate_paths = [tmp_path / f"intermediate_{index:02d}.pt" for index in range(20)]
    for path in intermediate_paths:
        path.touch()
    return OmegaConf.create(
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
                    "async_artifact_dir": str(tmp_path / "artifacts"),
                    "async_request_timeout_seconds": 10.0,
                    "paper_source_enabled": True,
                    "oracle_shadow_enabled": True,
                    "intermediate_oracle_checkpoint_paths": [
                        str(path) for path in intermediate_paths
                    ],
                    "intermediate_oracle_count": 20,
                    "suboptimal_data_ratio": 2.0,
                    "history_length": 2,
                    "prediction_horizon": 2,
                    "command_dim": 3,
                    "hidden_dim": 8,
                    "num_heads": 2,
                    "planner_layers": 1,
                    "idm_encoder_layers": 1,
                    "idm_decoder_layers": 1,
                    "feedforward_dim": 16,
                    "dropout": 0.0,
                    "iterations": 3,
                    "windows_per_iteration": 12,
                    "num_envs": 1,
                    "replay_capacity": 96,
                    "batch_size": 512,
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
                    "quality_eval_max_windows": 12,
                    "checkpoint_path": str(tmp_path / "planner_idm_v011.pt"),
                    "resume_path": None,
                    "initial_weights_path": None,
                    "stand_transition_curriculum": {
                        "enabled": True,
                        "standing_task": "FakeStandTask",
                        "walk_ratio": 0.5,
                        "static_stand_ratio": 0.25,
                        "walk_to_stand_ratio": 0.25,
                        "walk_command": [0.4, 0.0, 0.0],
                        "pre_switch_steps": 2,
                        "post_switch_steps": 2,
                    },
                    "v005_replay": {
                        "enabled": True,
                        "walk_cold_start_ratio": 0.2,
                        "static_cold_start_ratio": 0.5,
                        "planner_scenario_ratios": {
                            "walk": 0.5,
                            "static_stand": 0.25,
                            "walk_to_stand": 0.25,
                        },
                    },
                },
            },
        }
    )
