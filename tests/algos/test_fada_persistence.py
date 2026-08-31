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
    validate_fada_async_artifact_identity,
)
from unilab.algos.torch.distill.async_runtime import DaggerCollectRequest
from unilab.algos.torch.distill.fada import FADA_SCENARIO_IDS, FADASourceBatch
from unilab.algos.torch.distill.fada.async_config import fada_training_schedule
from unilab.algos.torch.distill.fada.persistent_workflow import (
    _admit_fada_artifact,
    _load_or_collect_admitted_artifact,
    _reuse_complete_artifact,
)
from unilab.algos.torch.distill.fada.source_artifact import (
    FADA_SHARDED_SOURCE_SCHEMA_VERSION,
    FADAShardedSourceWriter,
    LoadedFADASourceArtifact,
    open_fada_source_artifact,
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


def _trainer(policy: FADAPlannerIDMPolicy) -> FADATrainer:
    return FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )


def test_replay_trainer_and_checkpoint_keep_alternating_owner(tmp_path: Path) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = _trainer(policy)
    replay = FADAReplayBuffer(config, capacity=5)
    replay.add(_source_batch(config, size=7))
    assert len(replay) == 5

    stats = trainer.update(replay.sample(3), idm_updates=1, planner_updates=1)
    assert stats.idm_grad_norm is not None and stats.idm_grad_norm > 0.0
    assert stats.planner_grad_norm > 0.0

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
    restored_trainer = _trainer(restored_policy)
    restored_before = {
        name: tensor.detach().clone() for name, tensor in restored_policy.state_dict().items()
    }
    with pytest.raises(ValueError, match="resume is disabled"):
        load_fada_checkpoint(checkpoint, restored_policy, restored_trainer)
    assert not restored_trainer.idm_optimizer.state
    assert not restored_trainer.planner_optimizer.state
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
    trainer = _trainer(policy)
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


def test_fada_sharded_source_artifact_is_lazy_and_v4_loader_compatible(tmp_path: Path) -> None:
    config = _config()
    first = _source_batch(config, size=2)
    second = _source_batch(config, size=3)
    artifact_path = tmp_path / "source-sharded.pt"

    with FADAShardedSourceWriter(artifact_path, config=config) as writer:
        writer.append(first)
        writer.append(second)
        writer.commit(metadata={"iteration": 2, "main_windows": 2})

    manifest = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert manifest["schema_version"] == FADA_SHARDED_SOURCE_SCHEMA_VERSION
    assert "batch" not in manifest
    assert manifest["num_samples"] == 5
    assert [entry["rows"] for entry in manifest["shards"]] == [2, 3]

    opened = open_fada_source_artifact(artifact_path, config=config)
    assert opened.legacy_batch is None
    assert opened.num_samples == 5
    assert [batch.command.shape[0] for batch in opened.iter_batches()] == [2, 3]

    materialized = load_fada_source_batch(artifact_path, config=config)
    assert materialized.batch.command.shape[0] == 5
    assert materialized.metadata == {"iteration": 2, "main_windows": 2}


def test_fada_sharded_writer_removes_uncommitted_transaction(tmp_path: Path) -> None:
    config = _config()
    artifact_path = tmp_path / "aborted.pt"
    writer = FADAShardedSourceWriter(artifact_path, config=config)

    with writer:
        writer.append(_source_batch(config, size=1))
        shard_dir = writer._shard_dir
        manifest_temporary = writer._manifest_temporary
        assert shard_dir.is_dir()

    assert not artifact_path.exists()
    assert not shard_dir.exists()
    assert not manifest_temporary.exists()


def test_fada_sharded_manifest_rejects_paths_outside_owned_directory(tmp_path: Path) -> None:
    config = _config()
    artifact_path = tmp_path / "unsafe.pt"
    torch.save(
        {
            "schema_version": FADA_SHARDED_SOURCE_SCHEMA_VERSION,
            "architecture": asdict(config),
            "num_samples": 1,
            "shards": [
                {
                    "path": "unowned/shard_0000.pt",
                    "rows": 1,
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                    "planner_eligible": True,
                }
            ],
            "metadata": {},
        },
        artifact_path,
    )

    with pytest.raises(ValueError, match="unsafe FADA source artifact shard path"):
        open_fada_source_artifact(artifact_path, config=config)


def test_fada_sharded_writer_commit_failure_cleans_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    artifact_path = tmp_path / "commit-failure.pt"
    writer = FADAShardedSourceWriter(artifact_path, config=config)
    original_save = torch.save

    def fail_manifest(payload, path, *args, **kwargs):
        if Path(path) == writer._manifest_temporary:
            raise OSError("injected manifest failure")
        return original_save(payload, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure"):
        with writer:
            writer.append(_source_batch(config, size=1))
            writer.commit(metadata={})

    assert not artifact_path.exists()
    assert not writer._shard_dir.exists()
    assert not writer._manifest_temporary.exists()


def test_sharded_writer_forbids_overwriting_an_open_generation(tmp_path: Path) -> None:
    config = _config()
    artifact_path = tmp_path / "replace.pt"
    with FADAShardedSourceWriter(artifact_path, config=config) as writer:
        writer.append(_source_batch(config, size=2))
        writer.commit(metadata={"generation": 1})
    first_reader = open_fada_source_artifact(artifact_path, config=config)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        with FADAShardedSourceWriter(artifact_path, config=config):
            pass

    assert [batch.command.shape[0] for batch in first_reader.iter_batches()] == [2]


def test_v4_mmap_artifact_replay_releases_discarded_storage(tmp_path: Path) -> None:
    config = _config()
    artifact_path = tmp_path / "legacy-v4.pt"
    save_fada_source_batch(
        artifact_path,
        _source_batch(config, size=128),
        config=config,
        metadata={},
    )
    artifact = open_fada_source_artifact(artifact_path, config=config)
    assert artifact.legacy_batch is not None
    artifact.identity_fields()
    source_storage_bytes = artifact.legacy_batch.command.untyped_storage().nbytes()
    replay = FADAReplayBuffer(config, capacity=1)

    replay.add_artifact(artifact)

    retained = replay._chunks[0].command
    assert retained.untyped_storage().nbytes() == retained.numel() * retained.element_size()
    assert retained.untyped_storage().nbytes() < source_storage_bytes


def test_replay_loads_only_shards_selected_by_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    artifact_path = tmp_path / "streamed.pt"
    with FADAShardedSourceWriter(artifact_path, config=config) as writer:
        for _ in range(3):
            writer.append(_source_batch(config, size=2))
        writer.commit(metadata={})
    artifact = open_fada_source_artifact(artifact_path, config=config)
    replay = FADAReplayBuffer(config, capacity=2)
    loaded_indices: list[int] = []
    original_load = LoadedFADASourceArtifact.load_batch

    def traced_load(self, index: int):
        loaded_indices.append(index)
        return original_load(self, index)

    monkeypatch.setattr(LoadedFADASourceArtifact, "load_batch", traced_load)
    artifact.identity_fields()
    loaded_indices.clear()
    replay.add_artifact(artifact)

    assert loaded_indices == [2]
    assert len(replay) == 2


def test_persistent_retry_reuses_exact_completed_artifact(tmp_path: Path) -> None:
    config = _config()
    artifact_path = tmp_path / "iteration_0002.pt"
    request = DaggerCollectRequest(
        request_id="fada-0002-v3",
        scenario=FADA_ASYNC_SCENARIO,
        iteration=2,
        checkpoint_path=str((tmp_path / "student.pt").resolve()),
        output_path=str(artifact_path.resolve()),
        expected_weight_version=3,
    )
    metadata = {
        "request_id": request.request_id,
        "scenario": request.scenario,
        "iteration": request.iteration,
        "checkpoint_path": request.checkpoint_path,
        "expected_weight_version": request.expected_weight_version,
        "producer_pid": 1234,
        "main_windows": 2,
        "scenario_allocations": {"walk": 2},
        "collections": [
            {
                "source": "optimal_or_current_policy",
                "command_scenario": "walk",
                "windows": 2,
            }
        ],
    }
    with FADAShardedSourceWriter(artifact_path, config=config) as writer:
        writer.append(_source_batch(config, size=2))
        writer.commit(metadata=metadata)

    result, artifact = _reuse_complete_artifact(
        artifact_path,
        config=config,
        request=request,
    )

    assert result.worker_pid == 1234
    assert result.num_samples == 2
    assert result.metrics == {"artifact_reused": 1.0}
    assert artifact.metadata == metadata


def test_persistent_retry_rejects_completed_artifact_with_wrong_summary(tmp_path: Path) -> None:
    config = _config()
    cfg = _paper_persistent_training_cfg(tmp_path)
    cfg.training.fada.windows_per_iteration = 65536
    artifact_path = tmp_path / "iteration_0002.pt"
    request = DaggerCollectRequest(
        request_id="fada-0002-v3",
        scenario=FADA_ASYNC_SCENARIO,
        iteration=2,
        checkpoint_path=str((tmp_path / "student.pt").resolve()),
        output_path=str(artifact_path.resolve()),
        expected_weight_version=3,
    )
    metadata = {
        "request_id": request.request_id,
        "scenario": request.scenario,
        "iteration": request.iteration,
        "checkpoint_path": request.checkpoint_path,
        "expected_weight_version": request.expected_weight_version,
        "producer_pid": 1234,
        "training_schedule": fada_training_schedule(cfg.training.fada),
        "main_windows": 12,
        "stand_transition_curriculum_enabled": True,
        "scenario_allocations": {
            "walk": 32768,
            "static_stand": 16384,
            "walk_to_stand": 16384,
        },
        "collections": [
            {
                "source": "optimal_or_current_policy",
                "command_scenario": "walk",
                "windows": 32768,
            },
            {
                "source": "optimal_or_current_policy",
                "command_scenario": "static_stand",
                "windows": 16385,
            },
            {
                "source": "optimal_or_current_policy",
                "command_scenario": "walk_to_stand",
                "windows": 16384,
            },
        ],
    }
    with FADAShardedSourceWriter(artifact_path, config=config) as writer:
        writer.append(_source_batch(config, size=12))
        writer.commit(metadata=metadata)

    result, loaded = _reuse_complete_artifact(
        artifact_path,
        config=config,
        request=request,
    )
    with pytest.raises(ValueError, match="scenario summary mismatch"):
        _admit_fada_artifact(
            cfg,
            loaded=loaded,
            result=result,
            request=request,
            training_schedule=fada_training_schedule(cfg.training.fada),
        )


def test_invalid_reused_summary_is_recollected_once_before_consumer_mutation(
    tmp_path: Path,
) -> None:
    config = _config()
    cfg = _paper_persistent_training_cfg(tmp_path)
    cfg.training.fada.v005_replay.enabled = False
    schedule = fada_training_schedule(cfg.training.fada)
    artifact_path = tmp_path / "iteration_0002.pt"
    request = DaggerCollectRequest(
        request_id="fada-0002-v3",
        scenario=FADA_ASYNC_SCENARIO,
        iteration=2,
        checkpoint_path=str((tmp_path / "student.pt").resolve()),
        output_path=str(artifact_path.resolve()),
        expected_weight_version=3,
    )

    def metadata(*, static_windows: int, producer_pid: int) -> dict[str, object]:
        return {
            "request_id": request.request_id,
            "scenario": request.scenario,
            "iteration": request.iteration,
            "checkpoint_path": request.checkpoint_path,
            "expected_weight_version": request.expected_weight_version,
            "producer_pid": producer_pid,
            "training_schedule": schedule,
            "main_windows": 12,
            "stand_transition_curriculum_enabled": True,
            "scenario_allocations": {"walk": 6, "static_stand": 3, "walk_to_stand": 3},
            "collections": [
                {
                    "source": "optimal_or_current_policy",
                    "command_scenario": "walk",
                    "oracle_role": "unified",
                    "windows": 6,
                },
                {
                    "source": "optimal_or_current_policy",
                    "command_scenario": "static_stand",
                    "oracle_role": "unified",
                    "windows": static_windows,
                },
                {
                    "source": "optimal_or_current_policy",
                    "command_scenario": "walk_to_stand",
                    "oracle_role": "unified",
                    "windows": 3,
                },
            ],
        }

    with FADAShardedSourceWriter(artifact_path, config=config) as writer:
        writer.append(_source_batch(config, size=12))
        writer.commit(metadata=metadata(static_windows=4, producer_pid=111))

    class Runtime:
        collect_calls = 0

        def collect(self, collect_request: DaggerCollectRequest):
            self.collect_calls += 1
            with FADAShardedSourceWriter(
                artifact_path,
                config=config,
                replace_existing=True,
            ) as writer:
                writer.append(_source_batch(config, size=12))
                writer.commit(metadata=metadata(static_windows=3, producer_pid=222))
            return SimpleNamespace(
                request_id=collect_request.request_id,
                scenario=collect_request.scenario,
                iteration=collect_request.iteration,
                checkpoint_path=collect_request.checkpoint_path,
                output_path=collect_request.output_path,
                expected_weight_version=collect_request.expected_weight_version,
                observed_weight_version=collect_request.expected_weight_version,
                num_samples=12,
                worker_pid=222,
            )

    runtime = Runtime()
    _, loaded, main_windows, summaries, _ = _load_or_collect_admitted_artifact(
        cfg,
        runtime=runtime,
        artifact_path=artifact_path,
        config=config,
        request=request,
        training_schedule=schedule,
    )

    assert runtime.collect_calls == 1
    assert main_windows == 12
    assert sum(int(item["windows"]) for item in summaries[:3]) == 12
    assert loaded.metadata["producer_pid"] == 222


def test_collector_writer_atomically_replaces_invalid_manifest_generation(tmp_path: Path) -> None:
    config = _config()
    artifact_path = tmp_path / "iteration.pt"
    with FADAShardedSourceWriter(artifact_path, config=config) as writer:
        writer.append(_source_batch(config, size=1))
        writer.commit(metadata={"generation": 1})
    old_reader = open_fada_source_artifact(artifact_path, config=config)
    old_shard_dirs = {shard.path.parent for shard in old_reader.shards}

    with FADAShardedSourceWriter(
        artifact_path,
        config=config,
        replace_existing=True,
    ) as writer:
        writer.append(_source_batch(config, size=2))
        writer.commit(metadata={"generation": 2})

    assert all(not directory.exists() for directory in old_shard_dirs)
    replacement = open_fada_source_artifact(artifact_path, config=config)
    assert replacement.metadata == {"generation": 2}
    assert replacement.num_samples == 2


def test_collector_writer_failed_replacement_keeps_old_generation_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    artifact_path = tmp_path / "iteration.pt"
    with FADAShardedSourceWriter(artifact_path, config=config) as writer:
        writer.append(_source_batch(config, size=1))
        writer.commit(metadata={"generation": 1})
    old_reader = open_fada_source_artifact(artifact_path, config=config)
    old_shard_dirs = {shard.path.parent for shard in old_reader.shards}
    original_replace = Path.replace

    def fail_manifest_swap(path: Path, target: Path):
        if target == artifact_path:
            raise OSError("injected manifest swap failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_swap)
    with pytest.raises(OSError, match="injected manifest swap failure"):
        with FADAShardedSourceWriter(
            artifact_path,
            config=config,
            replace_existing=True,
        ) as writer:
            writer.append(_source_batch(config, size=2))
            writer.commit(metadata={"generation": 2})

    retained = open_fada_source_artifact(artifact_path, config=config)
    assert retained.metadata == {"generation": 1}
    assert retained.num_samples == 1
    assert all(directory.is_dir() for directory in old_shard_dirs)


def test_fada_async_artifact_identity_fails_closed_on_stale_request() -> None:
    metadata = {
        "request_id": "fada-0001-v2",
        "scenario": "fada_iteration",
        "iteration": 1,
        "checkpoint_path": "/tmp/student.pt",
        "expected_weight_version": 2,
        "producer_pid": 123,
    }

    validate_fada_async_artifact_identity(
        metadata,
        expected={**metadata},
    )
    with pytest.raises(ValueError, match="identity mismatch.*request_id"):
        validate_fada_async_artifact_identity(
            metadata,
            expected={**metadata, "request_id": "fada-0002-v2"},
        )


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


def test_v5_checkpoint_requires_observation_contract_but_legacy_v2_is_loadable(
    tmp_path: Path,
) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = _trainer(policy)
    checkpoint = tmp_path / "checkpoint.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=0,
        samples_seen=0,
        runtime_config={},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert FADA_CHECKPOINT_SCHEMA_VERSION == 5
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
    trainer = _trainer(policy)
    checkpoint = tmp_path / "idm-v4.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=0,
        samples_seen=1,
        runtime_config={"observation_contract": "g1_fada_state_v2"},
    )
    loaded = load_fada_policy_checkpoint(checkpoint, device="cpu")
    controller = FADAPlaybackController(loaded.policy, device="cpu")
    actions = controller.act(torch.zeros(1, 98), torch.from_numpy(command.copy()))
    assert actions.shape == (1, 29)
    assert loaded.checkpoint["schema_version"] == 5


def test_v005_checkpoint_serializer_requires_finite_scenario_metrics(tmp_path: Path) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = _trainer(policy)
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
