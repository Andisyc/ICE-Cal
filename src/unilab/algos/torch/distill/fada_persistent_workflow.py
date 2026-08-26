"""Persistent-async FADA learner unit of work."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from .async_runtime import DaggerCollectRequest
from .fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from .fada_artifact_admission import (
    fada_quality_batch,
    require_fada_curriculum_artifact,
    slice_fada_batch,
)
from .fada_async_runtime import FADA_ASYNC_SCENARIO
from .fada_checkpoint import save_fada_checkpoint
from .fada_replay import FADAReplayBuffer
from .fada_source_artifact import load_fada_source_batch
from .fada_source_evaluation import evaluate_fada_source_batch
from .fada_source_plan import FADAPaperSourcePlan
from .fada_trainer import FADATrainer
from .fada_workflow_setup import (
    FADAWorkflowDependencies,
    distill_device,
    fada_v005_replay_settings,
    resolve_fada_path,
)


def run_fada_persistent_async(
    cfg: DictConfig,
    *,
    config: FADAArchitectureConfig,
    paper_source_plan: FADAPaperSourcePlan,
    resolved_teacher: Path,
    checkpoint_path: Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer,
    replay: FADAReplayBuffer,
    start_iteration: int,
    samples_seen: int,
    dependencies: FADAWorkflowDependencies,
) -> dict[str, Any]:
    """Run the learner in the parent and all FADA rollout work in one resident child."""

    fada_cfg = cfg.training.fada
    iterations = int(fada_cfg.iterations)
    v005_enabled, planner_ratios, walk_cold_start_ratio, static_cold_start_ratio = (
        fada_v005_replay_settings(
            fada_cfg,
            batch_size=int(fada_cfg.batch_size),
        )
    )
    runtime_config = cast(dict[str, Any], OmegaConf.to_container(fada_cfg, resolve=True))
    artifact_dir_value = OmegaConf.select(
        fada_cfg, "async_artifact_dir", default="logs/fada/source_batches"
    )
    artifact_dir = resolve_fada_path(
        artifact_dir_value,
        field_name="training.fada.async_artifact_dir",
        required=True,
    )
    if artifact_dir is None:
        raise RuntimeError("FADA async artifact directory was not materialized")

    # The first publication is a complete paired checkpoint even before update 0.
    save_fada_checkpoint(
        checkpoint_path,
        policy,
        trainer,
        completed_iterations=start_iteration,
        samples_seen=samples_seen,
        runtime_config=runtime_config,
    )
    runtime = dependencies.build_persistent_fada_runtime(
        cfg=cfg,
        architecture=config,
        paper_source_plan=paper_source_plan,
        final_teacher_checkpoint=resolved_teacher,
        request_timeout_seconds=float(
            OmegaConf.select(fada_cfg, "async_request_timeout_seconds", default=3600.0)
        ),
    )
    last_stats = None
    last_quality_metrics: dict[str, float] = {}
    collection_summaries: list[dict[str, Any]] = []
    try:
        weight_version = runtime.activate_checkpoint(checkpoint_path)
        for iteration in range(start_iteration, iterations):
            artifact_path = (artifact_dir / f"iteration_{iteration:04d}.pt").resolve()
            request = DaggerCollectRequest(
                request_id=f"fada-{iteration:04d}-v{weight_version}",
                scenario=FADA_ASYNC_SCENARIO,
                iteration=iteration,
                checkpoint_path=str(checkpoint_path.resolve()),
                output_path=str(artifact_path),
                expected_weight_version=weight_version,
            )
            result = runtime.collect(request)
            loaded = load_fada_source_batch(artifact_path, config=config)
            if loaded.metadata.get("training_schedule") != "alternating_idm_then_planner":
                raise ValueError(
                    "FADA async artifact training schedule mismatch: "
                    f"observed={loaded.metadata.get('training_schedule')!r}"
                )
            if int(loaded.batch.command.shape[0]) != result.num_samples:
                raise ValueError(
                    "FADA async artifact/result sample mismatch: "
                    f"artifact={loaded.batch.command.shape[0]} result={result.num_samples}"
                )
            main_windows = int(loaded.metadata.get("main_windows", 0))
            if main_windows <= 0 or main_windows > result.num_samples:
                raise ValueError(f"invalid FADA async main_windows={main_windows}")
            summaries = loaded.metadata.get("collections")
            if not isinstance(summaries, list):
                raise ValueError("FADA async artifact collections must be a list")
            require_fada_curriculum_artifact(cfg, loaded.metadata, loaded.batch)
            collection_summaries.extend(cast(list[dict[str, Any]], summaries))
            replay.add(loaded.batch)
            samples_seen += result.num_samples

            last_stats = trainer.update_from_replay(
                replay,
                batch_size=int(fada_cfg.batch_size),
                idm_updates=int(fada_cfg.idm_updates),
                planner_updates=int(fada_cfg.planner_updates),
                device=distill_device(cfg),
                planner_scenario_ratios=(
                    planner_ratios
                    if v005_enabled
                    else None
                ),
                planner_walk_cold_start_ratio=walk_cold_start_ratio,
                planner_static_cold_start_ratio=static_cold_start_ratio,
            )
            if bool(fada_cfg.oracle_shadow_enabled):
                main_batch = slice_fada_batch(loaded.batch, main_windows)
                quality_limit = int(
                    OmegaConf.select(fada_cfg, "quality_eval_max_windows", default=4096)
                )
                quality_batch = (
                    fada_quality_batch(
                        main_batch,
                        config=config,
                        limit=quality_limit,
                        scenario_ratios=planner_ratios,
                        walk_cold_start_ratio=walk_cold_start_ratio,
                        static_cold_start_ratio=static_cold_start_ratio,
                    )
                    if v005_enabled
                    else slice_fada_batch(main_batch, quality_limit)
                )
                last_quality_metrics = evaluate_fada_source_batch(
                    policy,
                    quality_batch,
                    require_scenario_metrics=v005_enabled,
                )
                last_quality_metrics.update(
                    {
                        "rollout_rejected_done_transitions": float(
                            sum(int(item["rejected_done_transitions"]) for item in summaries)
                        ),
                        "rollout_rejected_command_windows": float(
                            sum(int(item["rejected_command_windows"]) for item in summaries)
                        ),
                    }
                )
            save_fada_checkpoint(
                checkpoint_path,
                policy,
                trainer,
                completed_iterations=iteration + 1,
                samples_seen=samples_seen,
                runtime_config=runtime_config,
                quality_metrics=last_quality_metrics,
            )
            if iteration + 1 < iterations:
                weight_version = runtime.activate_checkpoint(checkpoint_path)
    finally:
        runtime.close()

    replay_role_counts = replay.source_role_counts()
    return {
        "mode": "fada_alternating_training",
        "training_schedule": "alternating_idm_then_planner",
        "execution_mode": "persistent_async",
        "checkpoint_path": str(checkpoint_path),
        "completed_iterations": iterations,
        "samples_seen": samples_seen,
        "replay_size": len(replay),
        "replay_effective_capacity": replay.effective_capacity,
        "replay_role_counts": {
            "planner_eligible": replay_role_counts.planner_eligible,
            "planner_ineligible": replay_role_counts.planner_ineligible,
        },
        "last_idm_loss": None if last_stats is None else last_stats.idm_loss,
        "last_planner_loss": None if last_stats is None else last_stats.planner_loss,
        "quality_metrics": last_quality_metrics,
        "collections": collection_summaries,
    }
