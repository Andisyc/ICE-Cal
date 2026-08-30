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
