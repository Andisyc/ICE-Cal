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
from unilab.algos.torch.distill.fada_training_phase import FADATrainingPhase


def _write_completed_idm(path: Path, config: FADAArchitectureConfig) -> None:
    policy = FADAPlannerIDMPolicy(config)
    trainer = FADATrainer(
        policy,
        phase=FADATrainingPhase.IDM_PRETRAIN,
        optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
    )
    save_fada_checkpoint(
        path,
        policy,
        trainer,
        completed_iterations=1,
        samples_seen=1,
        runtime_config={},
        phase_completed=True,
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
                    "phase": "planner",
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
                    "pretrained_idm_path": str(tmp_path / "pretrained-idm.pt"),
                    "initial_weights_path": None,
                    "resume_path": None,
                },
            },
        }
    )
    config = _config()
    _write_completed_idm(tmp_path / "pretrained-idm.pt", config)

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
                    "training_phase": "planner",
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


@pytest.mark.parametrize(
    ("phase", "expected_ratio"),
    [("idm_pretrain", 2), ("planner", None)],
)
def test_fada_workflow_activates_role_retention_only_for_paper_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected_ratio: int | None,
) -> None:
    module = _load_train_distill()
    cfg = _paper_persistent_training_cfg(tmp_path)
    cfg.training.fada.phase = phase
    cfg.training.fada.paper_source_enabled = phase == "idm_pretrain"
    if phase == "planner":
        cfg.training.fada.intermediate_oracle_checkpoint_paths = []
        cfg.training.fada.pretrained_idm_path = str(tmp_path / "pretrained-idm.pt")
        _write_completed_idm(Path(cfg.training.fada.pretrained_idm_path), _curriculum_config())

    captured: dict[str, object] = {}
    original_replay = fada_workflow.FADAReplayBuffer

    def _recording_replay(*args, **kwargs):
        captured.update(kwargs)
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(fada_workflow, "FADAReplayBuffer", _recording_replay)
    monkeypatch.setattr(
        fada_workflow,
        "_run_fada_persistent_async",
        lambda *_args, **kwargs: {
            "suboptimal_retention_ratio": kwargs["replay"].suboptimal_retention_ratio,
        },
    )
    monkeypatch.setattr(module, "_require_teacher_policy_collection_route", lambda _cfg: None)
    monkeypatch.setattr(module, "_apply_collect_command_distribution_overrides", lambda _cfg: {})
    monkeypatch.setattr(module, "load_sac_teacher_policy", lambda *_args, **_kwargs: _Oracle())

    result = module.run_fada_training(cfg, teacher_checkpoint=tmp_path / "oracle.pt")

    assert captured.get("suboptimal_retention_ratio") == expected_ratio
    assert result["suboptimal_retention_ratio"] == expected_ratio


def test_fada_official_persistent_route_consumes_balanced_paper_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_train_distill()
    cfg = _paper_persistent_training_cfg(tmp_path)
    config = _curriculum_config()

    class _PaperRuntime:
        def __init__(self) -> None:
            self.activations: list[str] = []
            self.closed = False

        def activate_checkpoint(self, path: Path) -> int:
            self.activations.append(str(path))
            return len(self.activations)

        def collect(self, request):
            batch = _paper_role_batch(config, main_rows=12, intermediate_rows=24)
            batch = replace(
                batch,
                idm_source_role=torch.tensor(
                    ([1] * 12 if request.iteration == 0 else [1] * 3 + [0] * 9)
                    + [0] * 24,
                    dtype=torch.int64,
                ),
            )
            scenario = torch.tensor(
                [
                    *([FADA_SCENARIO_IDS["walk"]] * 6),
                    *([FADA_SCENARIO_IDS["static_stand"]] * 3),
                    *([FADA_SCENARIO_IDS["walk_to_stand"]] * 3),
                    *([FADA_SCENARIO_IDS["walk"]] * 24),
                ],
                dtype=torch.int64,
            )
            cold_start = torch.tensor(
                [True, True, True, False, False, False] + [True, True, False] + [False] * 27,
                dtype=torch.bool,
            )
            batch = replace(
                batch,
                command_scenario=scenario,
                cold_start=cold_start,
            )
            main_summaries = [
                {
                    "iteration": request.iteration,
                    "source": "optimal_or_current_policy",
                    "command_scenario": "walk",
                    "oracle_role": "walking",
                    "window_profile": "cold_start",
                    "windows": 3,
                    "env_steps": 3,
                    "rejected_done_transitions": 0,
                    "rejected_command_windows": 0,
                },
                {
                    "iteration": request.iteration,
                    "source": "optimal_or_current_policy",
                    "command_scenario": "walk",
                    "oracle_role": "walking",
                    "window_profile": "steady_state",
                    "windows": 3,
                    "env_steps": 3,
                    "rejected_done_transitions": 0,
                    "rejected_command_windows": 0,
                },
                {
                    "iteration": request.iteration,
                    "source": "optimal_or_current_policy",
                    "command_scenario": "static_stand",
                    "oracle_role": "standing",
                    "window_profile": "cold_start",
                    "windows": 2,
                    "env_steps": 2,
                    "rejected_done_transitions": 0,
                    "rejected_command_windows": 0,
                },
                {
                    "iteration": request.iteration,
                    "source": "optimal_or_current_policy",
                    "command_scenario": "static_stand",
                    "oracle_role": "standing",
                    "window_profile": "steady_state",
                    "windows": 1,
                    "env_steps": 1,
                    "rejected_done_transitions": 0,
                    "rejected_command_windows": 0,
                },
                {
                    "iteration": request.iteration,
                    "source": "optimal_or_current_policy",
                    "command_scenario": "walk_to_stand",
                    "oracle_role": "standing",
                    "window_profile": "steady_state",
                    "windows": 3,
                    "env_steps": 3,
                    "rejected_done_transitions": 0,
                    "rejected_command_windows": 0,
                },
            ]
            intermediate_summaries = [
                {
                    "iteration": request.iteration,
                    "source": "intermediate_oracle",
                    "source_checkpoint": str(path),
                    "command_scenario": "walk",
                    "oracle_role": "walking",
                    "window_profile": "steady_state",
                    "windows": 2 if index < 4 else 1,
                    "env_steps": 2 if index < 4 else 1,
                    "rejected_done_transitions": 0,
                    "rejected_command_windows": 0,
                }
                for index, path in enumerate(cfg.training.fada.intermediate_oracle_checkpoint_paths)
            ]
            save_fada_source_batch(
                request.output_path,
                batch,
                config=config,
                metadata={
                    "iteration": request.iteration,
                    "training_phase": "idm_pretrain",
                    "main_windows": 12,
                    "stand_transition_curriculum_enabled": True,
                    "v005_replay_enabled": True,
                    "scenario_allocations": {
                        "walk": 6,
                        "static_stand": 3,
                        "walk_to_stand": 3,
                    },
                    "collections": main_summaries + intermediate_summaries,
                },
            )
            return SimpleNamespace(num_samples=36)

        def close(self) -> None:
            self.closed = True

    runtime = _PaperRuntime()
    monkeypatch.setattr(module, "_require_teacher_policy_collection_route", lambda _cfg: None)
    monkeypatch.setattr(module, "_apply_collect_command_distribution_overrides", lambda _cfg: {})
    monkeypatch.setattr(module, "load_sac_teacher_policy", lambda *_args, **_kwargs: _Oracle())
    monkeypatch.setattr(module, "build_persistent_fada_runtime", lambda **_kwargs: runtime)

    result = module.run_fada_training(cfg, teacher_checkpoint=tmp_path / "oracle.pt")

    assert result["replay_size"] == 96
    assert result["replay_effective_capacity"] == 96
    assert result["replay_role_counts"] == {
        "planner_eligible": 32,
        "planner_ineligible": 64,
    }
    assert result["completed_iterations"] == 3
    assert len(runtime.activations) == 3
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
    pretrained_idm = tmp_path / "pretrained-idm.pt"
    _write_completed_idm(pretrained_idm, _config())
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
                    "phase": "planner",
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
                    "pretrained_idm_path": str(pretrained_idm),
                    "initial_weights_path": None,
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
