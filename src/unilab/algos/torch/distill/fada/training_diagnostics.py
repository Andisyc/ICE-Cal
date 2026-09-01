"""Compact terminal diagnostics for FADA collection and learner phases."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class FADATrainingStatsLike(Protocol):
    """Structural view needed by diagnostics, without coupling collection to trainer."""

    @property
    def idm_loss(self) -> float: ...

    @property
    def planner_loss(self) -> float: ...

    @property
    def idm_grad_norm(self) -> float: ...

    @property
    def planner_grad_norm(self) -> float: ...


class FADAReplayCoverageLike(Protocol):
    @property
    def planner_high_rows(self) -> int: ...

    @property
    def planner_high_batch_quota(self) -> int: ...

    @property
    def required_planner_updates(self) -> int: ...

    @property
    def idm_main_high_rows(self) -> int: ...

    @property
    def idm_main_high_batch_quota(self) -> int: ...

    @property
    def idm_intermediate_high_rows(self) -> int: ...

    @property
    def idm_intermediate_high_batch_quota(self) -> int: ...

    @property
    def required_idm_updates(self) -> int: ...


def format_fada_collection_diagnostic(
    *,
    scenario: str,
    window_profile: str,
    rollout_mode: str,
    windows: int,
    target_windows: int,
    env_steps: int,
    num_envs: int,
    rejected_done: int,
    rejected_command: int,
    rejected_scenario: int,
) -> str:
    attempted_rows = int(env_steps) * int(num_envs)
    acceptance = 0.0 if attempted_rows <= 0 else 100.0 * int(windows) / attempted_rows
    progress = 100.0 * int(windows) / max(int(target_windows), 1)
    return (
        "[fada-collect] "
        f"scenario={scenario} profile={window_profile} mode={rollout_mode} "
        f"progress={progress:.1f}% windows={windows}/{target_windows} "
        f"env_steps={env_steps} acceptance={acceptance:.2f}% "
        "rejected("
        f"done={rejected_done} command={rejected_command} scenario={rejected_scenario}"
        ")"
    )


@dataclass
class FADACollectionProgressReporter:
    """Print collection state at fixed progress milestones without log flooding."""

    scenario: str
    window_profile: str
    rollout_mode: str
    target_windows: int
    num_envs: int
    interval_percent: int = 10
    _next_percent: int = 0

    def report(
        self,
        *,
        windows: int,
        env_steps: int,
        rejected_done: int,
        rejected_command: int,
        rejected_scenario: int,
        force: bool = False,
    ) -> None:
        progress_percent = int(100 * int(windows) / max(int(self.target_windows), 1))
        if not force and progress_percent < self._next_percent:
            return
        print(
            format_fada_collection_diagnostic(
                scenario=self.scenario,
                window_profile=self.window_profile,
                rollout_mode=self.rollout_mode,
                windows=windows,
                target_windows=self.target_windows,
                env_steps=env_steps,
                num_envs=self.num_envs,
                rejected_done=rejected_done,
                rejected_command=rejected_command,
                rejected_scenario=rejected_scenario,
            ),
            file=sys.stderr,
            flush=True,
        )
        self._next_percent = (progress_percent // int(self.interval_percent) + 1) * int(
            self.interval_percent
        )


def _sum_collection_field(summaries: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(int(summary.get(field, 0)) for summary in summaries)


def format_fada_training_diagnostic(
    *,
    schedule: str,
    iteration: int,
    iterations: int,
    stats: FADATrainingStatsLike,
    idm_updates: int,
    planner_updates: int,
    configured_idm_updates: int | None = None,
    configured_planner_updates: int | None = None,
    replay_coverage: FADAReplayCoverageLike | None = None,
    replay_size: int,
    samples_seen: int,
    collection_summaries: Sequence[Mapping[str, Any]],
    collector_metrics: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> str:
    windows = _sum_collection_field(collection_summaries, "windows")
    env_steps = _sum_collection_field(collection_summaries, "env_steps")
    done = _sum_collection_field(collection_summaries, "rejected_done_transitions")
    command = _sum_collection_field(collection_summaries, "rejected_command_windows")
    scenario = _sum_collection_field(collection_summaries, "rejected_scenario_windows")
    planner = (
        "planner=disabled"
        if int(planner_updates) == 0
        else (
            f"planner(loss={stats.planner_loss:.6f} "
            f"grad={stats.planner_grad_norm:.6f} updates={planner_updates})"
        )
    )
    idm = (
        "idm=frozen"
        if int(idm_updates) == 0
        else (
            f"idm(loss={stats.idm_loss:.6f} grad={stats.idm_grad_norm:.6f} updates={idm_updates})"
        )
    )
    collect_seconds = float(collector_metrics.get("collect_seconds", 0.0))
    coverage = ""
    if replay_coverage is not None:
        coverage = (
            " coverage("
            f"planner_high={replay_coverage.planner_high_rows}/"
            f"{replay_coverage.planner_high_batch_quota} "
            f"idm_main_high={replay_coverage.idm_main_high_rows}/"
            f"{replay_coverage.idm_main_high_batch_quota} "
            f"idm_intermediate_high={replay_coverage.idm_intermediate_high_rows}/"
            f"{replay_coverage.idm_intermediate_high_batch_quota} "
            f"required_updates={replay_coverage.required_idm_updates}/"
            f"{replay_coverage.required_planner_updates} "
            f"configured_updates={configured_idm_updates}/{configured_planner_updates} "
            f"actual_updates={idm_updates}/{planner_updates})"
        )
    return (
        "[fada-train] "
        f"stage={schedule} iteration={iteration + 1}/{iterations} "
        f"windows={windows} env_steps={env_steps} replay={replay_size} "
        f"samples_seen={samples_seen} "
        f"{idm} {planner}{coverage} collect_s={collect_seconds:.2f} "
        f"rejected(done={done} command={command} scenario={scenario}) "
        f"checkpoint={checkpoint_path}"
    )


def print_fada_training_diagnostic(
    *,
    schedule: str,
    iteration: int,
    iterations: int,
    stats: FADATrainingStatsLike,
    idm_updates: int,
    planner_updates: int,
    configured_idm_updates: int | None = None,
    configured_planner_updates: int | None = None,
    replay_coverage: FADAReplayCoverageLike | None = None,
    replay_size: int,
    samples_seen: int,
    collection_summaries: Sequence[Mapping[str, Any]],
    collector_metrics: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> None:
    print(
        format_fada_training_diagnostic(
            schedule=schedule,
            iteration=iteration,
            iterations=iterations,
            stats=stats,
            idm_updates=idm_updates,
            planner_updates=planner_updates,
            configured_idm_updates=configured_idm_updates,
            configured_planner_updates=configured_planner_updates,
            replay_coverage=replay_coverage,
            replay_size=replay_size,
            samples_seen=samples_seen,
            collection_summaries=collection_summaries,
            collector_metrics=collector_metrics,
            checkpoint_path=checkpoint_path,
        ),
        file=sys.stderr,
        flush=True,
    )
