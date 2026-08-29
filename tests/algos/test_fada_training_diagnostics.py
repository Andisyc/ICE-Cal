from __future__ import annotations

from unilab.algos.torch.distill.fada_trainer import FADATrainingStats
from unilab.algos.torch.distill.fada_training_diagnostics import (
    format_fada_collection_diagnostic,
    format_fada_training_diagnostic,
)


def test_collection_diagnostic_exposes_yield_and_rejection_causes() -> None:
    line = format_fada_collection_diagnostic(
        scenario="walk",
        window_profile="steady_state",
        rollout_mode="oracle",
        windows=48_219,
        target_windows=65_536,
        env_steps=10_275,
        num_envs=64,
        rejected_done=7,
        rejected_command=11,
        rejected_scenario=13,
    )

    assert "windows=48219/65536" in line
    assert "acceptance=7.33%" in line
    assert "done=7" in line
    assert "command=11" in line
    assert "scenario=13" in line


def test_idm_diagnostic_marks_planner_as_disabled() -> None:
    line = format_fada_training_diagnostic(
        schedule="idm_pretrain",
        iteration=0,
        iterations=8,
        stats=FADATrainingStats(
            idm_loss=0.125,
            planner_loss=0.0,
            idm_grad_norm=1.25,
            planner_grad_norm=0.0,
        ),
        idm_updates=128,
        planner_updates=0,
        replay_size=65_536,
        samples_seen=65_536,
        collection_summaries=[
            {
                "windows": 65_536,
                "env_steps": 14_000,
                "rejected_done_transitions": 9,
                "rejected_command_windows": 10,
                "rejected_scenario_windows": 11,
            }
        ],
        collector_metrics={"collect_seconds": 12.5},
        checkpoint_path="/tmp/idm.pt",
    )

    assert "stage=idm_pretrain" in line
    assert "idm(loss=0.125000 grad=1.250000 updates=128)" in line
    assert "planner=disabled" in line
    assert "collect_s=12.50" in line


def test_alternating_diagnostic_prints_planner_metrics() -> None:
    line = format_fada_training_diagnostic(
        schedule="alternating_idm_then_planner",
        iteration=1,
        iterations=8,
        stats=FADATrainingStats(0.1, 0.2, 0.3, 0.4),
        idm_updates=64,
        planner_updates=64,
        replay_size=100,
        samples_seen=200,
        collection_summaries=[],
        collector_metrics={},
        checkpoint_path="/tmp/planner.pt",
    )

    assert "planner(loss=0.200000 grad=0.400000 updates=64)" in line


def test_planner_from_idm_diagnostic_marks_idm_as_frozen() -> None:
    line = format_fada_training_diagnostic(
        schedule="planner_from_idm",
        iteration=0,
        iterations=8,
        stats=FADATrainingStats(0.0, 0.2, 0.0, 0.4),
        idm_updates=0,
        planner_updates=128,
        replay_size=65_536,
        samples_seen=65_536,
        collection_summaries=[],
        collector_metrics={},
        checkpoint_path="/tmp/planner.pt",
    )

    assert "stage=planner_from_idm" in line
    assert "idm=frozen" in line
    assert "planner(loss=0.200000 grad=0.400000 updates=128)" in line
