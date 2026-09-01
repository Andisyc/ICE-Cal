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
    _State,
)
from unilab.algos.torch.distill import (
    FADA_CHECKPOINT_SCHEMA_VERSION,
    FADA_SOURCE_BATCH_SCHEMA_VERSION,
    FADAArchitectureConfig,
    FADACollectionSpec,
    FADAPaperSourcePlan,
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
from unilab.algos.torch.distill.fada.async_collection import _require_exact_collection_rows
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


def _reuse_test_collection_environment(worker, environment) -> None:
    @contextmanager
    def collection_environment():
        yield environment

    worker.collection_environment = collection_environment


def test_fada_collector_rejects_profile_overproduction_before_artifact_write() -> None:
    collection = SimpleNamespace(batch=_source_batch(_config(), size=3))

    with pytest.raises(RuntimeError, match="exact window allocation.*expected=2 observed=3"):
        _require_exact_collection_rows(
            collection,
            expected=2,
            scenario="static_stand",
            profile="cold_start",
        )


def test_fada_runtime_selects_request_scoped_collector_processes(tmp_path: Path) -> None:
    cfg = _paper_persistent_training_cfg(tmp_path)
    source_allocations = tuple(
        (Path(path), 1) for path in cfg.training.fada.intermediate_oracle_checkpoint_paths
    )
    runtime = build_persistent_fada_runtime(
        cfg=cfg,
        architecture=_curriculum_config(),
        paper_source_plan=FADAPaperSourcePlan(
            enabled=True,
            source_allocations=source_allocations,
        ),
        final_teacher_checkpoint=tmp_path / "final.pt",
        request_timeout_seconds=10.0,
    )
    try:
        assert runtime._worker_lifecycle == "request"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("oracle_shadow_enabled", "source_allocations", "expected_prepare_count"),
    [
        (True, (), 1),
        (False, (), 0),
        (False, (("intermediate.pt", 1),), 1),
    ],
)
def test_fada_worker_prepares_isolated_pool_only_when_shadow_is_collected(
    oracle_shadow_enabled: bool,
    source_allocations: tuple[tuple[str, int], ...],
    expected_prepare_count: int,
) -> None:
    class _Env:
        def __init__(self) -> None:
            self.prepare_count = 0

        def prepare_isolated_rollout_branch(self) -> None:
            self.prepare_count += 1

        def set_physics_envelope_guard(self, _max_abs_state: float) -> None:
            return None

        def close(self) -> None:
            return None

    environment = _Env()
    worker = PersistentFADACollectorWorker.__new__(PersistentFADACollectorWorker)
    worker.cfg = OmegaConf.create(
        {
            "training": {
                "task_name": "G1WalkFlat",
                "sim_backend": "mujoco",
                "fada": {"num_envs": 1, "oracle_shadow_enabled": oracle_shadow_enabled},
            }
        }
    )
    worker._env_cfg_override = {}
    worker._env_factory = lambda *_args, **_kwargs: environment
    worker._checkpoint_identity = None
    worker._physics_guard_max_abs = 1.0e4
    worker.source_allocations = source_allocations
    worker.root_dir = ROOT

    assert worker._materialize_collection_environment() is environment
    assert environment.prepare_count == expected_prepare_count


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
    collection_envs = [_FakeEnv(), _FakeEnv()]
    entered_envs: list[_FakeEnv] = []

    @contextmanager
    def collection_environment():
        env = collection_envs[len(entered_envs)]
        entered_envs.append(env)
        try:
            yield env
        finally:
            env.close()

    worker.collection_environment = collection_environment
    worker.student = FADAPlannerIDMPolicy(config)
    worker.final_teacher = _Oracle()
    worker.teacher_spec = object()
    worker.source_allocations = ((str(tmp_path / "intermediate.pt"), 1),)
    worker._intermediate_teacher_loader = lambda *_args, **_kwargs: _Oracle()
    worker._intermediate_teacher_reloader = lambda *_args, **_kwargs: None
    worker.intermediate_teacher = None
    worker.intermediate_teacher_checkpoint = None

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
    manifest = torch.load(output, map_location="cpu", weights_only=True)
    loaded = load_fada_source_batch(output, config=config)

    assert result.observed_weight_version == 7
    assert result.num_samples == 2
    assert manifest["schema_version"] == 5
    assert "batch" not in manifest
    assert [entry["rows"] for entry in manifest["shards"]] == [1, 1]
    assert loaded.metadata["main_windows"] == 1
    assert loaded.metadata["request_id"] == "fada-1-v7"
    assert loaded.metadata["scenario"] == FADA_ASYNC_SCENARIO
    assert loaded.metadata["checkpoint_path"] == str((tmp_path / "fada.pt").resolve())
    assert loaded.metadata["expected_weight_version"] == 7
    assert loaded.metadata["producer_pid"] == result.worker_pid
    assert [item["rollout_mode"] for item in loaded.metadata["collections"]] == [
        "planner_idm",
        "intermediate_oracle",
    ]
    assert [item["oracle_role"] for item in loaded.metadata["collections"]] == [
        "unified",
        "walking",
    ]
    assert len(entered_envs) == 2
    assert all(env.closed for env in entered_envs)


def test_fada_alternating_rolls_out_student_after_bootstrap(tmp_path: Path) -> None:
    config = _config()
    worker = PersistentFADACollectorWorker.__new__(PersistentFADACollectorWorker)
    worker.config = config
    worker.device = "cpu"
    worker.cfg = OmegaConf.create(
        {
            "training": {
                "fada": {
                    "training_schedule": "alternating_idm_then_planner",
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
    _reuse_test_collection_environment(worker, worker.env)
    worker.student = FADAPlannerIDMPolicy(config)
    worker.final_teacher = _Oracle()
    worker.teacher_spec = object()
    worker.source_allocations = ()

    class _WeightSync:
        def read_weights_into(self, _state_dict) -> int:
            return 7

    worker.weight_sync = _WeightSync()
    output = tmp_path / "alternating-iteration.pt"
    worker.collect(
        DaggerCollectRequest(
            request_id="fada-alternating-v7",
            scenario=FADA_ASYNC_SCENARIO,
            iteration=1,
            checkpoint_path=str((tmp_path / "alternating.pt").resolve()),
            output_path=str(output.resolve()),
            expected_weight_version=7,
        )
    )

    loaded = load_fada_source_batch(output, config=config)
    assert loaded.metadata["training_schedule"] == "alternating_idm_then_planner"
    assert [item["rollout_mode"] for item in loaded.metadata["collections"]] == ["planner_idm"]


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
    _reuse_test_collection_environment(worker, worker.env)
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

    worker._intermediate_teacher_loader = load_teacher
    worker._intermediate_teacher_reloader = reload_teacher
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


def test_fada_worker_isolates_each_collection_in_one_privileged_environment(
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
            self.physics_guard_calls = 0

        def set_physics_envelope_guard(self, max_abs_state: float | None) -> None:
            self.physics_guard_calls += 1
            self.physics_guard_max_abs = max_abs_state

        def close(self) -> None:
            self.close_count += 1

    class _BackendAdapter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def build_task_env_cfg_override(self) -> dict[str, object]:
            return {}

    environments: list[_Env] = []

    def env_factory(*_args, **_kwargs):
        environment = _Env()
        environments.append(environment)
        return environment

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
    worker = PersistentFADACollectorWorker(
        root_dir=str(ROOT),
        cfg_payload=cfg_payload,
        standing_curriculum_enabled=True,
        architecture=asdict(config),
        final_teacher_checkpoint="walking.pt",
        source_allocations=(),
        initial_checkpoint_path="student.pt",
        device="cpu",
        weight_sync_name="test-sync",
        weight_sync_lock=object(),
        weight_param_shapes={},
        env_factory=env_factory,
        oracle_loader=lambda *_args, **_kwargs: _Oracle(),
    )

    assert len(environments) == 1
    with worker.collection_environment() as first:
        assert first is environments[0]
        assert worker.env is first
        assert worker.standing_env is first
    assert environments[0].close_count == 1
    assert worker.env is None
    assert worker.standing_env is None

    with worker.collection_environment() as second:
        assert second is environments[1]
        assert second is not first
    assert environments[1].close_count == 1

    with pytest.raises(RuntimeError, match="collection failed"):
        with worker.collection_environment():
            raise RuntimeError("collection failed")
    assert environments[2].close_count == 1
    assert all(environment.physics_guard_calls == 1 for environment in environments)
    worker.close()
    assert all(environment.close_count == 1 for environment in environments)
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
                        "walk_cold_start_ratio": 0.2,
                        "static_cold_start_ratio": 0.5,
                    },
                }
            }
        }
    )
    worker.env = _CommandControlledEnv()
    worker.standing_env = worker.env
    _reuse_test_collection_environment(worker, worker.env)
    worker.student = FADAPlannerIDMPolicy(config)
    worker.final_teacher = _Oracle()
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
        "unified",
        "unified",
        "unified",
        "unified",
        "unified",
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


def test_enabled_worker_curriculum_does_not_require_separate_standing_environment(
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
                        "walk_command": [0.4, 0.0, 0.0],
                        "pre_switch_steps": 2,
                        "post_switch_steps": 3,
                    },
                }
            }
        }
    )
    worker.env = _CommandControlledEnv()
    worker.standing_env = None
    _reuse_test_collection_environment(worker, worker.env)
    worker.student = FADAPlannerIDMPolicy(worker.config)
    worker.final_teacher = _Oracle()
    worker.source_allocations = ()

    class _WeightSync:
        def read_weights_into(self, _state_dict) -> int:
            return 1

    worker.weight_sync = _WeightSync()
    result = worker.collect(
        DaggerCollectRequest(
            request_id="fada-0-v1",
            scenario=FADA_ASYNC_SCENARIO,
            iteration=0,
            checkpoint_path=str((tmp_path / "fada.pt").resolve()),
            output_path=str((tmp_path / "shared-env.pt").resolve()),
            expected_weight_version=1,
        )
    )
    assert result.num_samples == 3
