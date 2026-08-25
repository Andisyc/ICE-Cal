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
                    "phase": "idm_pretrain",
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
        "oracle",
        "intermediate_oracle",
    ]


def test_fada_persistent_worker_reuses_one_intermediate_teacher_across_rounds(
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
                    "phase": "idm_pretrain",
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
    worker.source_allocations = tuple(
        (str(tmp_path / name), 1) for name in ("intermediate-a.pt", "intermediate-b.pt")
    )
    constructed: list[torch.nn.Module] = []
    reloaded_paths: list[str] = []

    def load_teacher(*_args, **_kwargs):
        teacher = _Oracle()
        constructed.append(teacher)
        return teacher

    def reload_teacher(teacher, checkpoint_path, *_args, **_kwargs):
        assert teacher is constructed[0]
        reloaded_paths.append(str(checkpoint_path))

    worker._teacher_loader = load_teacher
    worker._teacher_reloader = reload_teacher
    worker.intermediate_teacher = None
    worker.intermediate_teacher_checkpoint = None

    class _WeightSync:
        def read_weights_into(self, _state_dict) -> int:
            return 7

        def close(self) -> None:
            return None

    worker.weight_sync = _WeightSync()
    for iteration in (1, 2):
        output = tmp_path / f"iteration-{iteration}.pt"
        worker.collect(
            DaggerCollectRequest(
                request_id=f"fada-{iteration}-v7",
                scenario=FADA_ASYNC_SCENARIO,
                iteration=iteration,
                checkpoint_path=str((tmp_path / "fada.pt").resolve()),
                output_path=str(output.resolve()),
                expected_weight_version=7,
            )
        )

    assert len(constructed) == 1
    assert reloaded_paths == [
        str(tmp_path / "intermediate-b.pt"),
        str(tmp_path / "intermediate-a.pt"),
        str(tmp_path / "intermediate-b.pt"),
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
            self.physics_guard_max_abs: float | None = None

        def set_physics_envelope_guard(self, max_abs_state: float | None) -> None:
            self.physics_guard_max_abs = max_abs_state

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


def test_fada_persistent_worker_collects_v005_walk_and_static_profile_artifact(
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
                    "phase": "idm_pretrain",
                    "windows_per_iteration": 6,
                    "oracle_shadow_enabled": True,
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
                        "walk_cold_start_ratio": 0.5,
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
        "walk",
        "static_stand",
        "static_stand",
        "walk_to_stand",
    ]
    assert [item["oracle_role"] for item in loaded.metadata["collections"]] == [
        "walking",
        "walking",
        "standing",
        "standing",
        "standing",
    ]
    assert [item["window_profile"] for item in loaded.metadata["collections"]] == [
        "cold_start",
        "steady_state",
        "cold_start",
        "steady_state",
        "steady_state",
    ]
    assert loaded.metadata["v005_replay_enabled"] is True
    static = loaded.batch.command_scenario == FADA_SCENARIO_IDS["static_stand"]
    walk = loaded.batch.command_scenario == FADA_SCENARIO_IDS["walk"]
    assert int(walk.sum()) == 3
    assert int((walk & loaded.batch.cold_start).sum()) == 2
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
                    "phase": "idm_pretrain",
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
                    "phase": "idm_pretrain",
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
                    "phase": "idm_pretrain",
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
