"""Multi-role DAgger iteration lifecycle."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from unilab.algos.torch.distill.contracts.workflow import (
    ArtifactDecision,
    DaggerWorkflowResult,
    RoleArtifactSpec,
    WorkflowDatasetSource,
    WorkflowScenarioCollectionResult,
    WorkflowScenarioCollector,
    WorkflowScenarioSpec,
    WorkflowStudentUpdateResult,
    _progress,
)
from unilab.algos.torch.distill.observability.performance import (
    DistillationMetricsRecorder,
    DistillationPerformanceRunContext,
    load_distillation_metrics,
)
from unilab.algos.torch.distill.workflows.artifacts import (
    _load_json,
    _manifest_sources,
    _validate_workflow_scenarios,
    _verified_current_checkpoint,
    _write_json_atomic,
    file_sha256,
    preflight_role_artifacts,
)
from unilab.algos.torch.distill.workflows.dagger_iteration import DaggerIterationContext


@dataclass
class _PreparedDaggerWorkflow:
    run_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    active_scenarios: tuple[WorkflowScenarioSpec, ...] | None
    completed_iterations: int
    metrics_path: Path
    metrics_recorder: DistillationMetricsRecorder | None
    current_checkpoint: Path
    cumulative_sources: list[WorkflowDatasetSource]
    cumulative_num_samples: int


def _prepare_dagger_workflow(
    *,
    run_dir: str | Path,
    role_specs: Sequence[RoleArtifactSpec],
    target_iterations: int,
    scenario_specs: Sequence[WorkflowScenarioSpec] | None,
    collect_scenario: Callable[..., Any] | None,
    execution_mode: str,
    scenario_collector: WorkflowScenarioCollector | None,
    performance_context: DistillationPerformanceRunContext | None,
) -> _PreparedDaggerWorkflow:
    resolved_run_dir = Path(run_dir)
    manifest_path = resolved_run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"workflow manifest does not exist: {manifest_path}")
    if target_iterations < 0:
        raise ValueError(f"target_iterations must be non-negative, got {target_iterations}")
    manifest = _load_json(manifest_path)
    active_scenarios = (
        None if scenario_specs is None else _validate_workflow_scenarios(scenario_specs, role_specs)
    )
    if execution_mode not in {"legacy", "persistent_async"}:
        raise ValueError(
            f"execution_mode must be 'legacy' or 'persistent_async', got {execution_mode!r}"
        )
    if execution_mode == "legacy" and scenario_collector is not None:
        raise ValueError("legacy execution_mode forbids scenario_collector")
    if performance_context is not None and performance_context.execution_mode != execution_mode:
        raise ValueError("performance_context execution_mode mismatch")
    if execution_mode == "persistent_async":
        if active_scenarios is None:
            raise ValueError("persistent_async execution_mode requires scenario_specs")
        if collect_scenario is not None:
            raise ValueError("persistent_async execution_mode forbids collect_scenario")
        if scenario_collector is None:
            raise ValueError("persistent_async execution_mode requires scenario_collector")
        if performance_context is None:
            raise ValueError("persistent_async execution_mode requires performance_context")
    if active_scenarios is not None:
        expected_scenarios = [scenario.as_dict() for scenario in active_scenarios]
        if manifest.get("scenario_specs") != expected_scenarios:
            raise ValueError("workflow scenario specs do not match run manifest")
    completed = int(manifest.get("completed_dagger_iterations", 0))
    if target_iterations < completed:
        raise ValueError(
            f"target_iterations {target_iterations} is below completed iterations {completed}"
        )

    metrics_path = resolved_run_dir / "distillation_metrics.json"
    metrics_recorder: DistillationMetricsRecorder | None = None
    if performance_context is not None:
        metrics_recorder = DistillationMetricsRecorder()
        if metrics_path.is_file():
            if manifest.get("distillation_metrics_path") != str(metrics_path.resolve()):
                raise ValueError("distillation metrics manifest path mismatch")
            if manifest.get("distillation_metrics_sha256") != file_sha256(metrics_path):
                raise ValueError("distillation metrics manifest hash mismatch")
            loaded_metrics = load_distillation_metrics(metrics_path)
            if (
                loaded_metrics.records[0].identity.run_signature()
                != performance_context.run_signature()
            ):
                raise ValueError("distillation metric identity drift within one artifact")
            if manifest.get("distillation_metrics_record_count") != len(loaded_metrics.records):
                raise ValueError("distillation metrics manifest record count mismatch")
            for record in loaded_metrics.records:
                metrics_recorder.add(record)
        elif completed > 0:
            raise FileNotFoundError(
                "completed persistent workflow is missing distillation_metrics.json"
            )

    require_row_role_labels = active_scenarios is not None and any(
        scenario.kind == "role" for scenario in active_scenarios
    )
    role_preflight = preflight_role_artifacts(
        role_specs,
        require_row_role_labels=require_row_role_labels,
    )
    not_reusable = [item for item in role_preflight if item.decision is not ArtifactDecision.REUSE]
    if not_reusable:
        details = ", ".join(
            f"{item.role}={item.decision.value}:{list(item.mismatches)}" for item in not_reusable
        )
        raise ValueError(f"DAgger role artifacts no longer match the run contract: {details}")

    current_checkpoint = _verified_current_checkpoint(manifest)
    cumulative_sources = _manifest_sources(manifest)
    cumulative_num_samples = int(
        manifest.get("bootstrap_num_samples", sum(1 for _ in cumulative_sources))
    )
    return _PreparedDaggerWorkflow(
        run_dir=resolved_run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        active_scenarios=active_scenarios,
        completed_iterations=completed,
        metrics_path=metrics_path,
        metrics_recorder=metrics_recorder,
        current_checkpoint=current_checkpoint,
        cumulative_sources=cumulative_sources,
        cumulative_num_samples=cumulative_num_samples,
    )


def _commit_dagger_iteration(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    iteration: int,
    input_checkpoint_path: str,
    input_checkpoint_sha256: str,
    role_artifacts: list[dict[str, Any]],
    scenario_artifacts: list[dict[str, Any]],
    active_scenarios: tuple[WorkflowScenarioSpec, ...] | None,
    aggregate_path: Path,
    cumulative_num_samples: int,
    output_checkpoint: Path,
    updates: int,
    execution_mode: str,
    input_weight_version: int | None,
    metrics_path: Path,
    metrics_recorder: DistillationMetricsRecorder | None,
) -> None:
    iteration_record = {
        "iteration": iteration,
        "input_checkpoint_path": input_checkpoint_path,
        "input_checkpoint_sha256": input_checkpoint_sha256,
        "role_artifacts": role_artifacts,
        "aggregate_dataset_path": str(aggregate_path.resolve()),
        "aggregate_dataset_sha256": file_sha256(aggregate_path),
        "aggregate_num_samples": cumulative_num_samples,
        "checkpoint_path": str(output_checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(output_checkpoint),
        "updates": updates,
    }
    if active_scenarios is not None:
        iteration_record["scenario_artifacts"] = scenario_artifacts
    if input_weight_version is not None:
        iteration_record["collection_execution_mode"] = execution_mode
        iteration_record["input_weight_version"] = input_weight_version
    if metrics_recorder is not None:
        manifest["distillation_metrics_path"] = str(metrics_path.resolve())
        manifest["distillation_metrics_sha256"] = file_sha256(metrics_path)
        manifest["distillation_metrics_record_count"] = len(metrics_recorder.records)
    manifest.setdefault("dagger_iterations", []).append(iteration_record)
    manifest["completed_dagger_iterations"] = iteration
    manifest["stage"] = f"DAGGER_ITERATION_{iteration}_COMPLETE"
    _write_json_atomic(manifest_path, manifest)


def run_multirole_dagger_workflow(
    *,
    run_dir: str | Path,
    role_specs: Sequence[RoleArtifactSpec],
    target_iterations: int,
    collect_role: Callable[[RoleArtifactSpec, Path, int, Path], int],
    aggregate_datasets: Callable[[tuple[WorkflowDatasetSource, ...], Path], int],
    update_student: Callable[[Path, Path, Path], int | WorkflowStudentUpdateResult],
    scenario_specs: Sequence[WorkflowScenarioSpec] | None = None,
    collect_scenario: Callable[
        [WorkflowScenarioSpec, Path, int, Path],
        int | WorkflowScenarioCollectionResult,
    ]
    | None = None,
    execution_mode: str = "legacy",
    scenario_collector: WorkflowScenarioCollector | None = None,
    performance_context: DistillationPerformanceRunContext | None = None,
    performance_clock: Callable[[], float] = time.perf_counter,
    status_callback: Callable[[str], None] | None = None,
    iteration_callback: Callable[[int, int], None] | None = None,
    runtime_sentinel: Callable[[str], None] | None = None,
) -> DaggerWorkflowResult:
    prepared = _prepare_dagger_workflow(
        run_dir=run_dir,
        role_specs=role_specs,
        target_iterations=target_iterations,
        scenario_specs=scenario_specs,
        collect_scenario=collect_scenario,
        execution_mode=execution_mode,
        scenario_collector=scenario_collector,
        performance_context=performance_context,
    )
    resolved_run_dir = prepared.run_dir
    manifest_path = prepared.manifest_path
    manifest = prepared.manifest
    active_scenarios = prepared.active_scenarios
    completed = prepared.completed_iterations
    metrics_path = prepared.metrics_path
    metrics_recorder = prepared.metrics_recorder
    current_checkpoint = prepared.current_checkpoint
    cumulative_sources = prepared.cumulative_sources
    cumulative_num_samples = prepared.cumulative_num_samples

    def emit_status(message: str) -> None:
        _progress(message)
        if status_callback is not None:
            status_callback(message)

    for iteration in range(completed + 1, target_iterations + 1):
        iteration_result = DaggerIterationContext(
            iteration=iteration,
            target_iterations=target_iterations,
            run_dir=resolved_run_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            current_checkpoint=current_checkpoint,
            cumulative_sources=cumulative_sources,
            cumulative_num_samples=cumulative_num_samples,
            role_specs=role_specs,
            active_scenarios=active_scenarios,
            collect_role=collect_role,
            aggregate_datasets=aggregate_datasets,
            update_student=update_student,
            collect_scenario=collect_scenario,
            execution_mode=execution_mode,
            scenario_collector=scenario_collector,
            performance_context=performance_context,
            performance_clock=performance_clock,
            metrics_path=metrics_path,
            metrics_recorder=metrics_recorder,
            emit_status=emit_status,
            iteration_callback=iteration_callback,
            runtime_sentinel=runtime_sentinel,
            commit_iteration=_commit_dagger_iteration,
        ).run()
        current_checkpoint = iteration_result.checkpoint
        cumulative_num_samples = iteration_result.cumulative_num_samples

    return DaggerWorkflowResult(
        run_dir=resolved_run_dir,
        manifest_path=manifest_path,
        completed_iterations=target_iterations,
        checkpoint_path=current_checkpoint,
        cumulative_num_samples=cumulative_num_samples,
    )
