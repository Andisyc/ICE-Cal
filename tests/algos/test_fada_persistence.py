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
from unilab.algos.torch.distill.fada_training_phase import (
    FADATrainingPhase,
    canonical_module_sha256,
)


def _trainer(policy: FADAPlannerIDMPolicy, phase: FADATrainingPhase) -> FADATrainer:
    module = policy.idm if phase is FADATrainingPhase.IDM_PRETRAIN else policy.planner
    return FADATrainer(
        policy,
        phase=phase,
        optimizer=torch.optim.Adam(module.parameters(), lr=1.0e-3),
        pretrained_idm_sha256=(
            canonical_module_sha256(policy.idm)
            if phase is FADATrainingPhase.PLANNER
            else None
        ),
    )


def test_replay_trainer_and_checkpoint_keep_phase_owner(tmp_path: Path) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = _trainer(policy, FADATrainingPhase.IDM_PRETRAIN)
    replay = FADAReplayBuffer(config, capacity=5)
    replay.add(_source_batch(config, size=7))
    assert len(replay) == 5

    stats = trainer.update(replay.sample(3), updates=1)
    assert stats.idm_grad_norm is not None and stats.idm_grad_norm > 0.0
    assert stats.planner_grad_norm is None

    checkpoint = tmp_path / "fada.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=2,
        samples_seen=10,
        runtime_config={"enabled": True},
        quality_metrics={"planner_idm_oracle_action_mse": 0.25},
        phase_completed=True,
    )
    restored_policy = FADAPlannerIDMPolicy(config)
    restored_trainer = _trainer(restored_policy, FADATrainingPhase.IDM_PRETRAIN)
    restored_before = {
        name: tensor.detach().clone() for name, tensor in restored_policy.state_dict().items()
    }
    with pytest.raises(ValueError, match="resume is disabled"):
        load_fada_checkpoint(checkpoint, restored_policy, restored_trainer)
    assert not restored_trainer.optimizer.state
    for name, tensor in restored_policy.state_dict().items():
        torch.testing.assert_close(tensor, restored_before[name])

    payload = load_fada_checkpoint(checkpoint, restored_policy)
    assert payload["completed_iterations"] == 2
    assert payload["schema_version"] == FADA_CHECKPOINT_SCHEMA_VERSION
    assert payload["quality_metrics"] == {"planner_idm_oracle_action_mse": 0.25}
    for expected, observed in zip(policy.parameters(), restored_policy.parameters(), strict=True):
        torch.testing.assert_close(expected, observed)


def test_fada_resume_checkpoint_uses_weights_only_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = _trainer(policy, FADATrainingPhase.IDM_PRETRAIN)
    checkpoint = tmp_path / "safe-resume.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=0,
        samples_seen=0,
        runtime_config={},
        phase_completed=False,
    )
    original_load = torch.load
    weights_only_values: list[object] = []

    def traced_load(*args, **kwargs):
        weights_only_values.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(fada_training.torch, "load", traced_load)
    with pytest.raises(ValueError, match="resume is disabled"):
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


def test_v4_source_artifacts_require_explicit_idm_source_role(
    tmp_path: Path,
) -> None:
    config = _config()
    batch = _source_batch(config, size=1)
    artifact = tmp_path / "source-v3.pt"
    save_fada_source_batch(artifact, batch, config=config, metadata={"iteration": 0})
    payload = torch.load(artifact, map_location="cpu", weights_only=True)

    assert FADA_SOURCE_BATCH_SCHEMA_VERSION == 4
    assert payload["schema_version"] == 4
    assert payload["architecture"]["observation_contract"] == "legacy_actor_obs_v1"

    payload["architecture"].pop("observation_contract")
    torch.save(payload, artifact)
    with pytest.raises(ValueError, match="observation_contract"):
        load_fada_source_batch(artifact, config=config)

    for legacy_schema in (2, 3):
        payload["schema_version"] = legacy_schema
        torch.save(payload, artifact)
        with pytest.raises(ValueError, match="unsupported or malformed FADA source batch schema"):
            load_fada_source_batch(artifact, config=config)


def test_v4_checkpoint_requires_observation_contract_but_legacy_v2_is_loadable(
    tmp_path: Path,
) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = _trainer(policy, FADATrainingPhase.IDM_PRETRAIN)
    checkpoint = tmp_path / "checkpoint.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=0,
        samples_seen=0,
        runtime_config={},
        phase_completed=False,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert FADA_CHECKPOINT_SCHEMA_VERSION == 4
    assert payload["architecture"]["observation_contract"] == "legacy_actor_obs_v1"

    payload["architecture"].pop("observation_contract")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="observation_contract"):
        load_fada_checkpoint(checkpoint, policy)
    with pytest.raises(ValueError, match="resume is disabled"):
        load_fada_checkpoint(checkpoint, policy, trainer)

    payload["schema_version"] = 2
    torch.save(payload, checkpoint)
    load_fada_checkpoint(checkpoint, policy)
    with pytest.raises(ValueError, match="resume is disabled"):
        load_fada_checkpoint(checkpoint, policy, trainer)


def test_official_v2_offline_transaction_collects_persists_and_plays_back(
    tmp_path: Path,
) -> None:
    from unilab.algos.torch.distill.fada_playback import FADAPlaybackController

    cfg = OmegaConf.load(ROOT / "conf" / "distill" / "config.yaml")
    config = fada_workflow.build_fada_architecture_config(cfg)
    command = np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32)

    class _G1Oracle(torch.nn.Module):
        obs_dim = 98

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.tanh(obs[:, 6:35] + 0.1)

    class _G1Env:
        num_envs = 1
        state = object()
        action_space = type("ActionSpace", (), {"shape": (29,)})()

        def __init__(self) -> None:
            self.step_count = 0
            self.current_obs = np.zeros((1, 98), dtype=np.float32)

        def reset(self, _indices: np.ndarray):
            self.step_count = 0
            self.current_obs.fill(0.0)
            self.current_obs[:, 93:96] = command
            return {"obs": self.current_obs.copy()}, {"commands": command.copy()}

        def step(self, actions: np.ndarray) -> _State:
            self.step_count += 1
            self.current_obs[:, 6:35] += actions * 0.01
            self.current_obs[:, 35:64] = actions
            self.current_obs[:, 64:93] = actions
            self.current_obs[:, 93:96] = command
            self.current_obs[:, 96] = np.sin(self.step_count * 0.1)
            self.current_obs[:, 97] = np.cos(self.step_count * 0.1)
            return _State(
                obs={"obs": self.current_obs.copy()},
                info={"commands": command.copy()},
                terminated=np.zeros((1,), dtype=np.bool_),
                truncated=np.zeros((1,), dtype=np.bool_),
            )

    collection = collect_fada_source_windows(
        _G1Env(),
        teacher_policy=_G1Oracle(),
        config=config,
        num_windows=1,
        spec=FADACollectionSpec(student_projection="g1_fada_state_v2"),
    )
    assert collection.batch.observation_history.shape == (1, 30, 66)
    assert collection.batch.realized_future.shape == (1, 6, 66)

    source_artifact = tmp_path / "source-v3.pt"
    save_fada_source_batch(
        source_artifact,
        collection.batch,
        config=config,
        metadata={"iteration": 0},
    )
    persisted = load_fada_source_batch(source_artifact, config=config)
    replay = FADAReplayBuffer(config, capacity=4)
    replay.add(persisted.batch)
    assert len(replay) == 1

    policy = FADAPlannerIDMPolicy(config)
    trainer = _trainer(policy, FADATrainingPhase.IDM_PRETRAIN)
    checkpoint = tmp_path / "idm-v4.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=0,
        samples_seen=1,
        runtime_config={"observation_contract": "g1_fada_state_v2"},
        phase_completed=False,
    )
    loaded = load_fada_policy_checkpoint(checkpoint, device="cpu")
    controller = FADAPlaybackController(loaded.policy, device="cpu")
    actions = controller.act(torch.zeros(1, 98), torch.from_numpy(command.copy()))
    assert actions.shape == (1, 29)
    assert loaded.checkpoint["schema_version"] == 4


def test_v005_checkpoint_serializer_requires_finite_scenario_metrics(tmp_path: Path) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = _trainer(policy, FADATrainingPhase.PLANNER)
    runtime_config = {"v005_replay": {"enabled": True}}
    required = {
        "scenario/walk/planner_idm_oracle_action_mse": 0.1,
        "scenario/walk/cold_start_fraction": 0.5,
        "scenario/walk/cold_start_planner_mse": 0.4,
        "scenario/walk/steady_state_planner_mse": 0.2,
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
            phase_completed=True,
        )
    with pytest.raises(ValueError, match="walk/steady_state_planner_mse"):
        save_fada_checkpoint(
            tmp_path / "missing-walk-profile.pt",
            policy,
            trainer,
            completed_iterations=1,
            samples_seen=1,
            runtime_config=runtime_config,
            quality_metrics={
                name: value
                for name, value in required.items()
                if name != "scenario/walk/steady_state_planner_mse"
            },
            phase_completed=True,
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
            phase_completed=True,
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
        phase_completed=True,
    )
    payload = load_fada_checkpoint(checkpoint, FADAPlannerIDMPolicy(config))
    assert payload["quality_metrics"] == required
