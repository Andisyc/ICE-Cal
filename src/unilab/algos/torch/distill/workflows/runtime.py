"""Resource lifecycle owner for one distillation workflow runtime session."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.observability.performance import DistillationPerformanceRunContext
from unilab.algos.torch.distill.workflows.diagnostics import WorkflowLoggerCallbacks
from unilab.algos.torch.distill.workflows.entry_collection import _distill_device
from unilab.logging import OffPolicyLogger


@dataclass(frozen=True)
class WorkflowRuntimeSession:
    """Own logger, persistent collector, cleanup, and performance finalization."""

    cfg: DictConfig
    operations: Any
    execution_mode: str
    run_dir: Path
    role_cfgs: dict[str, DictConfig]
    specs: tuple[Any, ...]
    scenario_specs: tuple[Any, ...] | None
    persistent_scenario_collector_factory: Callable[..., Any] | None
    performance_clock: Callable[[], float]
    run_dagger: Callable[..., Any]
    finalize_performance: Callable[..., Any]
    logger_cls: Any
    file_sha256_fn: Callable[..., str]
    config_fingerprint_fn: Callable[..., str]
    runtime_probe: Callable[[str], None]

    def _logger(self, target_iterations: int) -> OffPolicyLogger:
        return self.logger_cls(
            algo_name="distill",
            max_iterations=target_iterations,
            num_envs=int(
                OmegaConf.select(self.cfg, "training.workflow.collect_num_envs", default=64)
            ),
            env_name=str(OmegaConf.select(self.cfg, "training.task_name", default="G1WalkStand")),
            obs_dim=self.specs[0].student_obs_dim,
            action_dim=self.specs[0].teacher_action_dim,
            log_dir=str(self.run_dir),
            log_backend=str(OmegaConf.select(self.cfg, "training.logger", default="tensorboard")),
            display_title="UniLab Distillation / DAgger",
        )

    def _performance_context(self) -> DistillationPerformanceRunContext:
        resolved_config = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(resolved_config, dict):
            raise TypeError("resolved distillation config must be a mapping")
        return DistillationPerformanceRunContext(
            execution_mode=self.execution_mode,
            teacher_checkpoint_sha256=tuple(
                sorted({self.file_sha256_fn(spec.teacher_checkpoint_path) for spec in self.specs})
            ),
            config_sha256=self.config_fingerprint_fn(resolved_config),
            seed=int(self.cfg.algo.seed),
            device=_distill_device(self.cfg),
            num_envs=int(
                OmegaConf.select(self.cfg, "training.workflow.collect_num_envs", default=64)
            ),
        )

    def _persistent_collector(self, runtime_sentinel: Callable[[str], None] | None) -> Any:
        if self.execution_mode != "persistent_async":
            return None
        factory = self.persistent_scenario_collector_factory
        if factory is None:
            raise RuntimeError("persistent workflow requires a collector factory")
        collector = factory(
            cfg=self.cfg,
            role_cfgs=self.role_cfgs,
            role_specs=self.specs,
            scenario_specs=self.scenario_specs,
        )
        if not callable(getattr(collector, "close", None)):
            raise TypeError("persistent runtime factory result must provide close()")
        assert runtime_sentinel is not None
        runtime_sentinel("workflow/after_persistent_runtime_factory")
        return collector

    def run(self) -> Any:
        target_iterations = int(
            OmegaConf.select(self.cfg, "training.workflow.dagger_iterations", default=8)
        )
        runtime_sentinel = (
            self.runtime_probe
            if self.execution_mode == "persistent_async"
            else None
        )
        if runtime_sentinel is not None:
            runtime_sentinel("workflow/after_bootstrap")
        logger = self._logger(target_iterations)
        if runtime_sentinel is not None:
            runtime_sentinel("workflow/after_logger_construct")
        logger.start(status="Preparing DAgger workflow...")
        if runtime_sentinel is not None:
            runtime_sentinel("workflow/after_logger_start")
        callbacks = WorkflowLoggerCallbacks(logger=logger, runtime_sentinel=runtime_sentinel)
        operations = replace(self.operations, logger_callbacks=callbacks)
        performance_context = self._performance_context()
        collector = self._persistent_collector(runtime_sentinel)
        cleanup_duration_seconds = 0.0
        cleanup_report: Mapping[str, Any] = {
            "execution_mode": "legacy",
            "resource_scope": "per_request",
        }
        try:
            result = self.run_dagger(
                run_dir=self.run_dir,
                role_specs=self.specs,
                target_iterations=target_iterations,
                collect_role=operations.collect_dagger_role,
                aggregate_datasets=operations.aggregate_dagger_sources,
                update_student=operations.update_dagger_student,
                scenario_specs=self.scenario_specs,
                collect_scenario=(
                    operations.collect_dagger_scenario
                    if self.execution_mode == "legacy" and self.scenario_specs is not None
                    else None
                ),
                execution_mode=self.execution_mode,
                scenario_collector=collector,
                performance_context=performance_context,
                performance_clock=self.performance_clock,
                status_callback=callbacks.on_status,
                iteration_callback=callbacks.on_iteration,
                runtime_sentinel=runtime_sentinel,
            )
        except BaseException:
            logger.close()
            raise
        finally:
            if collector is not None:
                cleanup_start = float(self.performance_clock())
                collector.close()
                cleanup_duration_seconds = float(self.performance_clock()) - cleanup_start
                close_report = getattr(collector, "close_report", None)
                if not isinstance(close_report, Mapping):
                    raise ValueError("persistent runtime close() must publish close_report mapping")
                cleanup_report = close_report
        if result.completed_iterations > 0:
            self.finalize_performance(
                run_dir=self.run_dir,
                performance_context=performance_context,
                cleanup_duration_seconds=cleanup_duration_seconds,
                cleanup_report=cleanup_report,
            )
        logger.log_save(str(result.checkpoint_path))
        logger.finish(
            title="Distillation Summary",
            extra_summary=(
                f"  DAgger iterations: [yellow]{result.completed_iterations}[/]/"
                f"{target_iterations}\n"
                f"  Cumulative samples: [yellow]{result.cumulative_num_samples:,}[/]"
            ),
        )
        return result


def workflow_entry_result(
    *,
    mode: str,
    execution_mode: str,
    bootstrap_result: Any,
    dagger_result: Any,
) -> dict[str, Any]:
    return {
        "distill_source": "single_entry_workflow",
        "stage": (
            "BOOTSTRAP_COMPLETE"
            if dagger_result.completed_iterations == 0
            else f"DAGGER_ITERATION_{dagger_result.completed_iterations}_COMPLETE"
        ),
        "mode": mode,
        "execution_mode": execution_mode,
        "run_dir": str(dagger_result.run_dir),
        "manifest_path": str(dagger_result.manifest_path),
        "role_decisions": None if bootstrap_result is None else bootstrap_result.role_decisions,
        "bootstrap_dataset_path": (
            None if bootstrap_result is None else str(bootstrap_result.bootstrap_dataset_path)
        ),
        "bootstrap_num_samples": (
            None if bootstrap_result is None else bootstrap_result.bootstrap_num_samples
        ),
        "checkpoint_path": str(dagger_result.checkpoint_path),
        "completed_dagger_iterations": dagger_result.completed_iterations,
        "cumulative_num_samples": dagger_result.cumulative_num_samples,
    }
