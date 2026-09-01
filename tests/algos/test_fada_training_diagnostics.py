from __future__ import annotations

from types import SimpleNamespace

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


def test_speed_stratified_diagnostic_exposes_configured_required_and_actual_updates() -> None:
    line = format_fada_training_diagnostic(
        schedule="alternating_idm_then_planner",
        iteration=0,
        iterations=8,
        stats=FADATrainingStats(0.1, 0.2, 0.3, 0.4),
        idm_updates=256,
        planner_updates=192,
        configured_idm_updates=128,
        configured_planner_updates=128,
        replay_coverage=SimpleNamespace(
            planner_high_rows=3000,
            planner_high_batch_quota=123,
            required_planner_updates=192,
            idm_main_high_rows=1000,
            idm_main_high_batch_quota=41,
            idm_intermediate_high_rows=6000,
            idm_intermediate_high_batch_quota=205,
            required_idm_updates=256,
        ),
        replay_size=10_000,
        samples_seen=20_000,
        collection_summaries=[],
        collector_metrics={},
        checkpoint_path="/tmp/planner.pt",
    )

    assert "planner_high=3000/123" in line
    assert "idm_main_high=1000/41" in line
    assert "idm_intermediate_high=6000/205" in line
    assert "required_updates=256/192" in line
    assert "configured_updates=128/128" in line
    assert "actual_updates=256/192" in line
