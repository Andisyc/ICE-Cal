from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.algos.torch.distill.collector import project_student_obs
from unilab.algos.torch.distill.fada import FADAArchitectureConfig
from unilab.algos.torch.distill.fada_workflow import (
    assert_fada_source_route_contract,
    build_fada_architecture_config,
)
from unilab.algos.torch.distill.fada_workflow_setup import (
    assert_fada_training_run_contract,
)


def test_g1_fada_state_v2_projects_exact_state_and_excludes_action_command() -> None:
    source = np.stack(
        [
            np.arange(98, dtype=np.float32),
            np.arange(98, dtype=np.float32) + 1000.0,
        ]
    )
    expected = np.concatenate([source[:, :64], source[:, 96:98]], axis=1)

    observed = project_student_obs(
        source,
        projection="g1_fada_state_v2",
        expected_student_obs_dim=66,
    )

    np.testing.assert_array_equal(observed, expected)
    changed_forbidden = source.copy()
    changed_forbidden[:, 64:96] += 100_000.0
    np.testing.assert_array_equal(
        project_student_obs(
            changed_forbidden,
            projection="g1_fada_state_v2",
            expected_student_obs_dim=66,
        ),
        expected,
    )
    changed_state = source.copy()
    changed_state[:, 0] += 7.0
    changed_state[:, 97] -= 11.0
    projected_state = project_student_obs(
        changed_state,
        projection="g1_fada_state_v2",
        expected_student_obs_dim=66,
    )
    assert np.all(projected_state[:, 0] == expected[:, 0] + 7.0)
    assert np.all(projected_state[:, -1] == expected[:, -1] - 11.0)


@pytest.mark.parametrize("width", [97, 99])
def test_g1_fada_state_v2_rejects_wrong_raw_observation_width(width: int) -> None:
    with pytest.raises(ValueError, match="98"):
        project_student_obs(
            np.zeros((2, width), dtype=np.float32),
            projection="g1_fada_state_v2",
            expected_student_obs_dim=66,
        )


def test_g1_fada_state_v2_architecture_accepts_only_exact_tuple() -> None:
    config = FADAArchitectureConfig(
        obs_dim=66,
        action_dim=29,
        command_dim=3,
        observation_contract="g1_fada_state_v2",
    )
    assert config.observation_contract == "g1_fada_state_v2"

    with pytest.raises(ValueError, match="66.*29.*3"):
        FADAArchitectureConfig(
            obs_dim=98,
            action_dim=29,
            command_dim=3,
            observation_contract="g1_fada_state_v2",
        )
    with pytest.raises(ValueError, match="reserved"):
        FADAArchitectureConfig(
            obs_dim=66,
            action_dim=29,
            command_dim=3,
            observation_contract="legacy_actor_obs_v1",
        )
    with pytest.raises(ValueError, match="observation_contract"):
        FADAArchitectureConfig(
            obs_dim=66,
            action_dim=29,
            command_dim=3,
            observation_contract="unknown",
        )


def test_fada_hydra_owner_keeps_actor_width_and_builds_v2_state_width() -> None:
    cfg = OmegaConf.create(
        {
            "student": {"obs_dim": 98, "action_dim": 29},
            "training": {
                "fada": {
                    "obs_dim": 66,
                    "observation_contract": "g1_fada_state_v2",
                    "student_projection": "g1_fada_state_v2",
                    "command_dim": 3,
                    "history_length": 30,
                    "prediction_horizon": 6,
                    "hidden_dim": 128,
                    "num_heads": 4,
                    "planner_layers": 3,
                    "idm_encoder_layers": 3,
                    "idm_decoder_layers": 2,
                    "feedforward_dim": 512,
                    "dropout": 0.0,
                }
            },
        }
    )

    config = build_fada_architecture_config(cfg)

    assert cfg.student.obs_dim == 98
    assert config.obs_dim == 66
    assert config.observation_contract == "g1_fada_state_v2"

    cfg.training.fada.initial_weights_path = "checkpoints/planner_idm_v005.pt"
    cfg.training.fada.resume_path = None
    with pytest.raises(ValueError, match="fresh initialization"):
        assert_fada_source_route_contract(cfg, config)


def test_official_source_config_enables_exact_source_campaign_but_has_no_assets() -> None:
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "conf" / "distill" / "config.yaml")

    assert cfg.training.fada.enabled is False
    assert "phase" not in cfg.training.fada
    assert cfg.training.fada.paper_source_enabled is True
    assert cfg.training.fada.oracle_shadow_enabled is True
    assert cfg.training.fada.stand_transition_curriculum.enabled is True
    assert cfg.training.fada.v005_replay.enabled is True
    assert cfg.training.fada.v005_replay.walk_cold_start_ratio == 0.5
    assert cfg.training.fada.intermediate_oracle_checkpoint_paths == []
    assert "standing_teacher_checkpoint_path" not in cfg.training.fada.stand_transition_curriculum
    assert cfg.training.fada.initial_weights_path is None
    assert cfg.training.fada.resume_path is None
    assert "pretrained_idm_path" not in cfg.training.fada
    assert cfg.training.fada.async_artifact_dir == "logs/fada/planner_idm_v011/source_batches"
    assert cfg.training.fada.checkpoint_path == "logs/fada/planner_idm_v011.pt"

    cfg.training.fada.obs_dim = 98
    cfg.training.fada.observation_contract = "legacy_actor_obs_v1"
    cfg.training.fada.student_projection = "identity"
    legacy = build_fada_architecture_config(cfg)
    with pytest.raises(ValueError, match="active FADA route"):
        assert_fada_source_route_contract(cfg, legacy)


def _formal_source_run_cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "training": {
                "fada": {
                    "execution_mode": "persistent_async",
                    "paper_source_enabled": True,
                    "training_schedule": "alternating_idm_then_planner",
                    "checkpoint_path": str(tmp_path / "source.pt"),
                    "resume_path": None,
                    "initial_weights_path": None,
                    "idm_initialization_path": None,
                    "idm_updates": 2,
                    "planner_updates": 2,
                }
            }
        }
    )


@pytest.mark.parametrize("schedule", ["idm_pretrain", "planner_from_idm"])
def test_formal_source_contract_rejects_split_training_schedules(
    tmp_path: Path,
    schedule: str,
) -> None:
    cfg = _formal_source_run_cfg(tmp_path)
    cfg.training.fada.training_schedule = schedule

    with pytest.raises(ValueError, match="only supports alternating_idm_then_planner"):
        assert_fada_training_run_contract(cfg)


@pytest.mark.parametrize(
    ("idm_updates", "planner_updates", "expected"),
    [(0, 2, "idm_updates>0"), (2, 0, "planner_updates>0")],
)
def test_formal_source_contract_requires_both_optimizer_passes(
    tmp_path: Path,
    idm_updates: int,
    planner_updates: int,
    expected: str,
) -> None:
    cfg = _formal_source_run_cfg(tmp_path)
    cfg.training.fada.idm_updates = idm_updates
    cfg.training.fada.planner_updates = planner_updates

    with pytest.raises(ValueError, match=expected):
        assert_fada_training_run_contract(cfg)


def test_formal_source_contract_rejects_retired_idm_initialization(tmp_path: Path) -> None:
    cfg = _formal_source_run_cfg(tmp_path)
    cfg.training.fada.idm_initialization_path = str(tmp_path / "idm.pt")

    with pytest.raises(ValueError, match="removed training.fada.idm_initialization_path"):
        assert_fada_training_run_contract(cfg)
