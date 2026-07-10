from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch

from unilab.algos.torch.distill import build_distillation_dataset, save_distillation_dataset


def _load_audit_script() -> Any:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "deploy"
        / "check_unilab_g1_distill_dataset_audit.py"
    )
    spec = importlib.util.spec_from_file_location("check_unilab_g1_distill_dataset_audit", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _save_toy_dataset(path: Path, *, command_intents: tuple[str, ...]) -> None:
    role_labels = ("walk_flat", "walk_flat", "stand", "stand")
    commands = torch.tensor(
        [
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    save_distillation_dataset(
        path,
        build_distillation_dataset(
            torch.arange(8, dtype=torch.float32).reshape(4, 2),
            torch.arange(8, 16, dtype=torch.float32).reshape(4, 2),
            expected_student_obs_dim=2,
            expected_teacher_obs_dim=2,
            expected_teacher_action_dim=3,
            teacher_actions=torch.tensor(
                [
                    [0.1, 0.2, 0.3],
                    [0.1, 0.0, -0.1],
                    [0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            commands=commands,
            role_labels=role_labels,
            command_intents=command_intents,
            metadata={
                "source": "multitask_adapter",
                "source_count": 2,
                "source_paths": ["walk.pt", "stand.pt"],
                "source_roles": ["walk_flat", "stand"],
                "source_sample_counts": [2, 2],
                "source_metadata": [
                    {
                        "task_name": "G1WalkFlat",
                        "sim_backend": "mujoco",
                        "teacher_policy_checkpoint_path": "walk_model.pt",
                        "action_mode": "teacher_policy",
                        "command_sample_filter": "active",
                    },
                    {
                        "task_name": "G1StandStill",
                        "sim_backend": "mujoco",
                        "teacher_policy_checkpoint_path": "stand_model.pt",
                        "action_mode": "teacher_policy",
                        "command_sample_filter": "inactive",
                    },
                ],
            },
        ),
    )


def test_distill_dataset_audit_reports_role_intent_and_sources(tmp_path: Path) -> None:
    mod = _load_audit_script()
    dataset_path = tmp_path / "merged.pt"
    _save_toy_dataset(dataset_path, command_intents=("active", "active", "inactive", "inactive"))

    report = mod.audit_dataset(dataset_path, stat_sample_rows=4)

    assert report["status"] == "ok"
    assert report["num_samples"] == 4
    assert report["dims"] == {
        "student_obs_dim": 2,
        "teacher_obs_dim": 2,
        "teacher_action_dim": 3,
    }
    assert report["labels"]["role_counts"] == {"stand": 2, "walk_flat": 2}
    assert report["labels"]["command_intent_counts"] == {"active": 2, "inactive": 2}
    assert report["labels"]["role_intent_counts"] == {
        "stand|inactive": 2,
        "walk_flat|active": 2,
    }
    assert report["commands"]["label_threshold_mismatch_count"] == 0
    assert report["sources"]["source_sample_count_matches_num_samples"] is True
    assert report["sources"]["source_metadata"][0]["command_sample_filter"] == "active"
    assert report["warnings"] == []
    assert report["issues"] == []


def test_distill_dataset_audit_warns_on_command_intent_threshold_mismatch(tmp_path: Path) -> None:
    mod = _load_audit_script()
    dataset_path = tmp_path / "merged.pt"
    _save_toy_dataset(dataset_path, command_intents=("inactive", "active", "inactive", "inactive"))

    report = mod.audit_dataset(dataset_path, stat_sample_rows=4)

    assert report["status"] == "ok"
    assert report["commands"]["label_threshold_mismatch_count"] == 1
    assert report["labels"]["role_expected_intent_mismatch_count"] == 1
    assert any("threshold-recomputed" in warning for warning in report["warnings"])
    assert any("role_labels conflict" in warning for warning in report["warnings"])
    assert report["issues"] == []


def test_distill_dataset_audit_strict_fails_on_hard_schema_issue(tmp_path: Path) -> None:
    mod = _load_audit_script()
    dataset_path = tmp_path / "bad.pt"
    torch.save(
        {
            "student_obs": torch.zeros(2, 2),
            "teacher_obs": torch.zeros(3, 2),
            "num_samples": 2,
        },
        dataset_path,
    )

    report = mod.audit_dataset(dataset_path, stat_sample_rows=4)
    exit_code = mod.main([str(dataset_path), "--strict", "--stat-sample-rows", "4"])

    assert report["status"] == "issues"
    assert any("row mismatch" in issue for issue in report["issues"])
    assert exit_code == 1
