from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tests.algos._fada_training_test_support import (
    ROOT,
    _load_train_distill,
    _paper_role_batch,
)
from unilab.algos.torch.distill import (
    FADA_CHECKPOINT_SCHEMA_VERSION,
    FADAPaperSourcePlan,
    FADAPlannerIDMPolicy,
    FADATrainer,
    load_fada_checkpoint,
    save_fada_source_batch,
)
from unilab.algos.torch.distill.fada import FADA_SCENARIO_IDS
from unilab.algos.torch.distill.fada_training_phase import FADATrainingPhase
from unilab.algos.torch.distill.fada_workflow_setup import (
    build_fada_architecture_config,
)


def _bounded_formal_config(tmp_path: Path):
    cfg = OmegaConf.load(ROOT / "note" / "fada" / "evidence" / "fada_v007r1_final_config.yaml")
    walking = tmp_path / "G1WalkFlat" / "model_5000.pt"
    standing = tmp_path / "G1StandStill" / "model_5000.pt"
    walking.parent.mkdir(parents=True)
    standing.parent.mkdir(parents=True)
    walking.touch()
    standing.touch()
    intermediates = [tmp_path / "oracle" / f"model_{step}.pt" for step in range(240, 4801, 240)]
    intermediates[0].parent.mkdir(parents=True)
    for path in intermediates:
        path.touch()

    cfg.training.device = "cpu"
    cfg.teacher.checkpoint_path = str(walking)
    fada = cfg.training.fada
    fada.phase = "idm_pretrain"
    fada.pretrained_idm_path = None
    fada.async_request_timeout_seconds = 30.0
    fada.async_artifact_dir = str(tmp_path / "source_batches")
    fada.intermediate_oracle_checkpoint_paths = [str(path) for path in intermediates]
    fada.quality_eval_max_windows = 12
    fada.iterations = 3
    fada.windows_per_iteration = 12
    fada.num_envs = 1
    fada.replay_capacity = 96
    fada.batch_size = 12
    fada.idm_updates = 1
    fada.planner_updates = 1
    fada.max_env_steps = 12
    fada.checkpoint_path = str(tmp_path / "planner_idm_v007r1_formal.pt")
    fada.stand_transition_curriculum.standing_teacher_checkpoint_path = str(standing)
    return cfg, walking, standing, tuple(intermediates)


def _collection_summaries(iteration: int, intermediate_paths: tuple[Path, ...]):
    main = [
        {
            "iteration": iteration,
            "source": "optimal_or_current_policy",
            "command_scenario": scenario,
            "oracle_role": oracle_role,
            "window_profile": profile,
            "windows": windows,
            "env_steps": windows,
            "rejected_done_transitions": 0,
            "rejected_command_windows": 0,
        }
        for scenario, oracle_role, profile, windows in (
            ("walk", "walking", "cold_start", 3),
            ("walk", "walking", "steady_state", 3),
            ("static_stand", "standing", "cold_start", 2),
            ("static_stand", "standing", "steady_state", 1),
            ("walk_to_stand", "standing", "steady_state", 3),
        )
    ]
    intermediate = [
        {
            "iteration": iteration,
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
        for index, path in enumerate(intermediate_paths)
    ]
    return main + intermediate


def test_refactored_official_route_closes_updates_persistence_and_first_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch.manual_seed(20260823)
    module = _load_train_distill()
    cfg, walking, standing, intermediate_paths = _bounded_formal_config(tmp_path)
    architecture = build_fada_architecture_config(cfg)
    assert (
        architecture.obs_dim,
        architecture.action_dim,
        architecture.command_dim,
        architecture.history_length,
        architecture.prediction_horizon,
    ) == (66, 29, 3, 30, 6)

    class _ExternalRuntime:
        def __init__(self) -> None:
            self.activations: list[str] = []
            self.close_count = 0

        def activate_checkpoint(self, path: Path) -> int:
            self.activations.append(str(path))
            return len(self.activations)

        def collect(self, request):
            batch = _paper_role_batch(architecture, main_rows=12, intermediate_rows=24)
            batch = replace(
                batch,
                idm_source_role=torch.tensor(
                    ([1] * 12 if request.iteration == 0 else [1] * 3 + [0] * 9)
                    + [0] * 24,
                    dtype=torch.int64,
                ),
                command_scenario=torch.tensor(
                    [
                        *([FADA_SCENARIO_IDS["walk"]] * 6),
                        *([FADA_SCENARIO_IDS["static_stand"]] * 3),
                        *([FADA_SCENARIO_IDS["walk_to_stand"]] * 3),
                        *([FADA_SCENARIO_IDS["walk"]] * 24),
                    ],
                    dtype=torch.int64,
                ),
                cold_start=torch.tensor(
                    [True, True, True, False, False, False] + [True, True, False] + [False] * 27,
                    dtype=torch.bool,
                ),
            )
            save_fada_source_batch(
                request.output_path,
                batch,
                config=architecture,
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
                    "collections": _collection_summaries(
                        request.iteration,
                        intermediate_paths,
                    ),
                },
            )
            return SimpleNamespace(num_samples=36)

        def close(self) -> None:
            self.close_count += 1

    runtime = _ExternalRuntime()
    loaded_intermediates: list[Path] = []
    runtime_receipt: dict[str, object] = {}

    def _load_external_checkpoint(path, _spec, *, device):
        assert device == "cpu"
        loaded_intermediates.append(Path(path))
        return torch.nn.Identity()

    def _build_external_runtime(**kwargs):
        runtime_receipt.update(kwargs)
        return runtime

    monkeypatch.setattr(module, "load_sac_teacher_policy", _load_external_checkpoint)
    monkeypatch.setattr(module, "build_persistent_fada_runtime", _build_external_runtime)

    module.main.__wrapped__(cfg)
    result = json.loads(capsys.readouterr().out)

    assert loaded_intermediates == list(intermediate_paths)
    assert len(set(loaded_intermediates)) == 20
    assert runtime_receipt["architecture"] == architecture
    assert runtime_receipt["final_teacher_checkpoint"] == walking
    paper_source_plan = runtime_receipt["paper_source_plan"]
    assert isinstance(paper_source_plan, FADAPaperSourcePlan)
    assert tuple(paper_source_plan.checkpoint_paths) == intermediate_paths
    assert cfg.training.fada.stand_transition_curriculum.standing_teacher_checkpoint_path == str(
        standing
    )
    assert result["execution_mode"] == "persistent_async"
    assert result["training_phase"] == "idm_pretrain"
    assert result["completed_iterations"] == 3
    assert result["samples_seen"] == 108
    assert result["replay_size"] == 96
    assert result["replay_effective_capacity"] == 96
    assert result["replay_role_counts"] == {
        "planner_eligible": 32,
        "planner_ineligible": 64,
    }
    assert len(runtime.activations) == 3
    assert runtime.close_count == 1
    assert result["last_idm_loss"] is not None
    assert result["last_planner_loss"] is None

    restored_policy = FADAPlannerIDMPolicy(architecture)
    checkpoint = Path(cfg.training.fada.checkpoint_path)
    payload = load_fada_checkpoint(checkpoint, restored_policy)
    assert payload["schema_version"] == FADA_CHECKPOINT_SCHEMA_VERSION == 4
    assert payload["training_phase"] == "idm_pretrain"
    assert payload["phase_completed"] is True
    assert payload["optimizer_owner"] == "idm"
    assert "optimizer_state_dict" in payload
    assert "idm_optimizer_state_dict" not in payload
    assert "planner_optimizer_state_dict" not in payload
    assert payload["completed_iterations"] == 3
    assert payload["samples_seen"] == 108
    action = restored_policy.explore(
        torch.zeros(1, 30, 66),
        torch.zeros(1, 30, 29),
        torch.zeros(1, 3),
    )
    assert action.shape == (1, 29)
    assert torch.isfinite(action).all()
    assert not checkpoint.with_suffix(checkpoint.suffix + ".tmp").exists()
