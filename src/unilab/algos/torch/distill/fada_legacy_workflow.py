"""Legacy in-process FADA collection and learner unit of work."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from .fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from .fada_artifact_admission import slice_fada_batch as _slice_fada_batch
from .fada_checkpoint import save_fada_checkpoint
from .fada_collector import FADACollectionSpec, collect_fada_source_windows
from .fada_replay import FADAReplayBuffer
from .fada_source_evaluation import evaluate_fada_source_batch
from .fada_source_plan import FADAPaperSourcePlan
from .fada_trainer import FADATrainer
from .fada_training_diagnostics import print_fada_training_diagnostic
from .fada_workflow_setup import ROOT_DIR, FADAWorkflowDependencies, distill_device


def run_fada_legacy(
    cfg: DictConfig,
    *,
    config: FADAArchitectureConfig,
    paper_source_plan: FADAPaperSourcePlan,
    resolved_teacher: Path,
    teacher_spec: Any,
    checkpoint_path: Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer,
    replay: FADAReplayBuffer,
    start_iteration: int,
    samples_seen: int,
    dependencies: FADAWorkflowDependencies,
    create_env_fn: Any | None,
    env_cfg_override_fn: Any | None,
) -> dict[str, Any]:
    """Run the preserved legacy in-process collect-update-save lifecycle."""

    fada_cfg = cfg.training.fada
    device = distill_device(cfg)
    paper_source_enabled = paper_source_plan.enabled
    collect_oracle_shadow = bool(OmegaConf.select(fada_cfg, "oracle_shadow_enabled", default=False))
    iterations = int(fada_cfg.iterations)
    windows_per_iteration = int(fada_cfg.windows_per_iteration)
    batch_size = int(fada_cfg.batch_size)
    teacher_policy = dependencies.load_fada_oracle_policy(
        resolved_teacher,
        teacher_spec,
        device=device,
    )

    # B3: 通过 public UniLab env factory 组装 collector, 不向脚本泄漏 backend 状态.
    if create_env_fn is None:
        dependencies.ensure_registries()
        create_env_fn = dependencies.create_env
    if env_cfg_override_fn is None:
        env_cfg_override_fn = lambda owner_cfg: dependencies.backend_adapter_cls(  # noqa: E731
            owner_cfg,
            root_dir=ROOT_DIR,
            algo_name="distill",
        ).build_task_env_cfg_override()
    env = create_env_fn(
        cfg,
        num_envs=int(fada_cfg.num_envs),
        env_cfg_override=env_cfg_override_fn(cfg),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        task_name=str(OmegaConf.select(cfg, "training.task_name")),
    )
    last_stats = None
    last_quality_metrics: dict[str, float] = {}
    collection_summaries: list[dict[str, Any]] = []
    try:
        command_keys = OmegaConf.to_container(fada_cfg.command_info_keys, resolve=True)
        if not isinstance(command_keys, list) or not command_keys:
            raise ValueError("training.fada.command_info_keys must be a non-empty list")
        collection_spec = FADACollectionSpec(
            observation_key=str(fada_cfg.observation_key),
            teacher_projection=str(fada_cfg.teacher_projection),
            student_projection=str(fada_cfg.student_projection),
            student_drop_index=OmegaConf.select(cfg, "training.fada.student_drop_index"),
            command_info_keys=tuple(str(key) for key in command_keys),
            max_env_steps=OmegaConf.select(cfg, "training.fada.max_env_steps"),
            collect_oracle_shadow=collect_oracle_shadow,
        )
        for iteration in range(start_iteration, iterations):
            # B4: 收集 current-policy optimal source, 并为每个 visited state 生成 final-Oracle shadow.
            collection = collect_fada_source_windows(
                env,
                teacher_policy=teacher_policy,
                config=config,
                num_windows=windows_per_iteration,
                rollout_policy=policy if iteration > 0 else None,
                spec=collection_spec,
            )
            replay.add(collection.batch)
            samples_seen += int(collection.batch.command.shape[0])
            collection_summaries.append(
                {
                    "iteration": iteration,
                    "source": "optimal_or_current_policy",
                    "rollout_mode": collection.rollout_mode,
                    "windows": int(collection.batch.command.shape[0]),
                    "env_steps": collection.env_steps,
                    "rejected_done_transitions": collection.rejected_done_transitions,
                    "rejected_command_windows": collection.rejected_command_windows,
                }
            )

            # B5: Appendix B.2 以 2:1 总预算轮转 20 个 intermediate Oracle rollout.
            if paper_source_enabled:
                for intermediate_path, source_windows in paper_source_plan.source_allocations:
                    intermediate_policy = dependencies.load_sac_teacher_policy(
                        intermediate_path,
                        teacher_spec,
                        device=device,
                    )
                    suboptimal = collect_fada_source_windows(
                        env,
                        teacher_policy=teacher_policy,
                        config=config,
                        num_windows=source_windows,
                        rollout_teacher_policy=intermediate_policy,
                        spec=replace(collection_spec, collect_oracle_shadow=True),
                    )
                    replay.add(suboptimal.batch)
                    samples_seen += int(suboptimal.batch.command.shape[0])
                    collection_summaries.append(
                        {
                            "iteration": iteration,
                            "source": "intermediate_oracle",
                            "source_checkpoint": str(intermediate_path),
                            "rollout_mode": suboptimal.rollout_mode,
                            "windows": int(suboptimal.batch.command.shape[0]),
                            "env_steps": suboptimal.env_steps,
                            "rejected_done_transitions": suboptimal.rejected_done_transitions,
                            "rejected_command_windows": suboptimal.rejected_command_windows,
                        }
                    )
            last_stats = trainer.update_from_replay(
                replay,
                batch_size=batch_size,
                idm_updates=int(fada_cfg.idm_updates),
                planner_updates=int(fada_cfg.planner_updates),
                device=device,
            )
            if collect_oracle_shadow:
                last_quality_metrics = evaluate_fada_source_batch(
                    policy,
                    _slice_fada_batch(
                        collection.batch,
                        int(OmegaConf.select(fada_cfg, "quality_eval_max_windows", default=4096)),
                    ),
                )
                iteration_collections = [
                    summary
                    for summary in collection_summaries
                    if int(summary["iteration"]) == iteration
                ]
                last_quality_metrics.update(
                    {
                        "rollout_rejected_done_transitions": float(
                            sum(
                                int(summary["rejected_done_transitions"])
                                for summary in iteration_collections
                            )
                        ),
                        "rollout_rejected_command_windows": float(
                            sum(
                                int(summary["rejected_command_windows"])
                                for summary in iteration_collections
                            )
                        ),
                    }
                )
            runtime_config = cast(dict[str, Any], OmegaConf.to_container(fada_cfg, resolve=True))
            save_fada_checkpoint(
                checkpoint_path,
                policy,
                trainer,
                completed_iterations=iteration + 1,
                samples_seen=samples_seen,
                runtime_config=runtime_config,
                quality_metrics=last_quality_metrics,
            )
            iteration_collections = [
                summary
                for summary in collection_summaries
                if int(summary["iteration"]) == iteration
            ]
            print_fada_training_diagnostic(
                schedule="alternating_idm_then_planner",
                iteration=iteration,
                iterations=iterations,
                stats=last_stats,
                idm_updates=int(fada_cfg.idm_updates),
                planner_updates=int(fada_cfg.planner_updates),
                replay_size=len(replay),
                samples_seen=samples_seen,
                collection_summaries=iteration_collections,
                collector_metrics={},
                checkpoint_path=checkpoint_path,
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return {
        "mode": "fada_alternating_training",
        "training_schedule": "alternating_idm_then_planner",
        "checkpoint_path": str(checkpoint_path),
        "completed_iterations": iterations,
        "samples_seen": samples_seen,
        "replay_size": len(replay),
        "last_idm_loss": None if last_stats is None else last_stats.idm_loss,
        "last_planner_loss": None if last_stats is None else last_stats.planner_loss,
        "quality_metrics": last_quality_metrics,
        "collections": collection_summaries,
    }
