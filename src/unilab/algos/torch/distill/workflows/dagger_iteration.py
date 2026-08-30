"""One DAgger iteration with a single mutable lifecycle owner."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from unilab.algos.torch.distill.contracts.workflow import (
    RoleArtifactSpec,
    WorkflowDatasetSource,
    WorkflowScenarioCollectionResult,
    WorkflowScenarioCollector,
    WorkflowScenarioSpec,
    WorkflowStudentUpdateResult,
)
from unilab.algos.torch.distill.observability.performance import (
    LEGACY_REQUEST_STAGE_NAMES,
    PERSISTENT_REQUEST_STAGE_NAMES,
    WORKFLOW_ITERATION_STAGE_NAMES,
    DistillationMetricsRecorder,
    DistillationPerformanceRunContext,
    DistillationStageObservation,
    load_distillation_metrics,
)
from unilab.algos.torch.distill.runtime.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
    validate_dagger_collect_result,
)
from unilab.algos.torch.distill.workflows.artifacts import (
    create_role_artifact_manifest,
    file_sha256,
    write_role_artifact_manifest,
)


@dataclass(frozen=True)
class DaggerIterationResult:
    checkpoint: Path
    cumulative_num_samples: int


@dataclass
class DaggerIterationContext:
    iteration: int
    target_iterations: int
    run_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    current_checkpoint: Path
    cumulative_sources: list[WorkflowDatasetSource]
    cumulative_num_samples: int
    role_specs: Sequence[RoleArtifactSpec]
    active_scenarios: tuple[WorkflowScenarioSpec, ...] | None
    collect_role: Callable[[RoleArtifactSpec, Path, int, Path], int]
    aggregate_datasets: Callable[[tuple[WorkflowDatasetSource, ...], Path], int]
    update_student: Callable[[Path, Path, Path], int | WorkflowStudentUpdateResult]
    collect_scenario: Callable[..., Any] | None
    execution_mode: str
    scenario_collector: WorkflowScenarioCollector | None
    performance_context: DistillationPerformanceRunContext | None
    performance_clock: Callable[[], float]
    metrics_path: Path
    metrics_recorder: DistillationMetricsRecorder | None
    emit_status: Callable[[str], None]
    iteration_callback: Callable[[int, int], None] | None
    runtime_sentinel: Callable[[str], None] | None
    commit_iteration: Callable[..., None]

    def __post_init__(self) -> None:
        self.input_checkpoint_path = str(self.current_checkpoint.resolve())
        self.input_checkpoint_sha256 = file_sha256(self.current_checkpoint)
        self.iteration_dir = self.run_dir / "datasets" / f"dagger_iteration_{self.iteration}"
        self.iteration_dir.mkdir(parents=True, exist_ok=True)
        self.role_artifacts: list[dict[str, Any]] = []
        self.scenario_artifacts: list[dict[str, Any]] = []
        self.input_weight_version: int | None = None

    def run(self) -> DaggerIterationResult:
        if self.iteration_callback is not None:
            self.iteration_callback(self.iteration, self.target_iterations)
        self.emit_status(f"iteration={self.iteration}/{self.target_iterations} collecting datasets")
        self._activate_checkpoint()
        if self.active_scenarios is None:
            self._collect_roles()
        else:
            self._collect_scenarios()
        aggregate_path, aggregate_seconds = self._aggregate()
        output_checkpoint, updates, learner_result = self._update_student(aggregate_path)
        self._record_workflow_metrics(
            aggregate_seconds=aggregate_seconds,
            learner_result=learner_result,
        )
        self._commit(aggregate_path, output_checkpoint, updates)
        return DaggerIterationResult(output_checkpoint, self.cumulative_num_samples)

    def _activate_checkpoint(self) -> None:
        if self.execution_mode != "persistent_async":
            return
        assert self.scenario_collector is not None
        self._sentinel("before_activate_checkpoint")
        self.input_weight_version = int(
            self.scenario_collector.activate_checkpoint(self.current_checkpoint)
        )
        self._sentinel("after_activate_checkpoint")
        if self.input_weight_version < 0:
            raise ValueError(
                "persistent scenario collector returned a negative weight version: "
                f"{self.input_weight_version}"
            )

    def _collect_roles(self) -> None:
        for spec in self.role_specs:
            output_path = self.iteration_dir / f"{spec.role}.pt"
            output_spec = replace(spec, dataset_path=output_path)
            num_samples = int(
                self.collect_role(output_spec, self.current_checkpoint, self.iteration, output_path)
            )
            if num_samples <= 0:
                raise ValueError(
                    f"DAgger collector for role {spec.role!r} returned {num_samples} samples"
                )
            self.emit_status(
                f"iteration={self.iteration} collected role={spec.role} samples={num_samples}"
            )
            artifact_manifest = create_role_artifact_manifest(output_spec, num_samples=num_samples)
            write_role_artifact_manifest(output_spec.manifest_path, artifact_manifest)
            self.role_artifacts.append(asdict(artifact_manifest))
            self.cumulative_sources.append(WorkflowDatasetSource(output_path, spec.role))

    def _collect_scenarios(self) -> None:
        assert self.active_scenarios is not None
        role_specs_by_name = {spec.role: spec for spec in self.role_specs}
        if self.execution_mode == "legacy" and self.collect_scenario is None:
            raise ValueError("scenario workflow requires collect_scenario callback")
        for scenario in self.active_scenarios:
            output_path = self.iteration_dir / f"{scenario.name}.pt"
            persistent_result, legacy_result, num_samples = self._collect_one_scenario(
                scenario, output_path
            )
            if num_samples <= 0:
                raise ValueError(
                    f"DAgger collector for scenario {scenario.name!r} returned {num_samples} samples"
                )
            self.emit_status(
                f"iteration={self.iteration} collected scenario={scenario.name} "
                f"samples={num_samples}"
            )
            if persistent_result is not None or legacy_result is not None:
                self._record_collector_metrics(
                    scenario=scenario,
                    persistent_result=persistent_result,
                    legacy_result=legacy_result,
                )
            self._record_scenario_artifact(
                scenario=scenario,
                output_path=output_path,
                num_samples=num_samples,
                persistent_result=persistent_result,
                role_specs_by_name=role_specs_by_name,
            )

    def _collect_one_scenario(
        self, scenario: WorkflowScenarioSpec, output_path: Path
    ) -> tuple[
        DaggerCollectResult | None,
        WorkflowScenarioCollectionResult | None,
        int,
    ]:
        if self.execution_mode == "legacy":
            assert self.collect_scenario is not None
            raw_result = self.collect_scenario(
                scenario, self.current_checkpoint, self.iteration, output_path
            )
            legacy_result = (
                raw_result if isinstance(raw_result, WorkflowScenarioCollectionResult) else None
            )
            if self.performance_context is not None and legacy_result is None:
                raise ValueError("legacy performance_context requires rich scenario result")
            return (
                None,
                legacy_result,
                (legacy_result.num_samples if legacy_result is not None else int(raw_result)),
            )
        assert self.scenario_collector is not None
        assert self.input_weight_version is not None
        request = DaggerCollectRequest(
            request_id=f"dagger-{self.iteration}-{scenario.name}",
            scenario=scenario.name,
            iteration=self.iteration,
            checkpoint_path=self.input_checkpoint_path,
            output_path=str(output_path.resolve()),
            expected_weight_version=self.input_weight_version,
        )
        self._sentinel(f"scenario_{scenario.name}/before_collect")
        result = self.scenario_collector.collect(request)
        self._sentinel(f"scenario_{scenario.name}/after_collect")
        validate_dagger_collect_result(request, result)
        return result, None, int(result.num_samples)

    def _record_collector_metrics(
        self,
        *,
        scenario: WorkflowScenarioSpec,
        persistent_result: DaggerCollectResult | None,
        legacy_result: WorkflowScenarioCollectionResult | None,
    ) -> None:
        assert self.performance_context is not None
        assert self.metrics_recorder is not None
        if persistent_result is not None:
            payloads = persistent_result.metadata.get("performance_stage_observations")
            if not isinstance(payloads, list):
                raise ValueError("persistent result performance_stage_observations are missing")
            observations = tuple(DistillationStageObservation.from_dict(item) for item in payloads)
            worker_pid = persistent_result.worker_pid
            request_id = persistent_result.request_id
            weight_version = persistent_result.observed_weight_version
            schema_version = persistent_result.metadata.get("performance_metrics_schema_version")
        else:
            assert legacy_result is not None
            observations = legacy_result.performance_stage_observations
            worker_pid = legacy_result.worker_pid
            request_id = f"dagger-{self.iteration}-{scenario.name}"
            weight_version = None
            schema_version = legacy_result.performance_metrics_schema_version
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ValueError("collector performance_metrics_schema_version must be an integer")
        records = self.performance_context.enrich_request(
            outer_iteration=self.iteration,
            scenario=scenario.name,
            worker_pid=worker_pid,
            request_id=request_id,
            checkpoint_path=self.input_checkpoint_path,
            checkpoint_sha256=self.input_checkpoint_sha256,
            weight_version=weight_version,
            schema_version=schema_version,
            observations=observations,
        )
        for record in records:
            self.metrics_recorder.add(record)
        required = (
            LEGACY_REQUEST_STAGE_NAMES
            if self.execution_mode == "legacy"
            else PERSISTENT_REQUEST_STAGE_NAMES
        )
        self.metrics_recorder.write(self.metrics_path, required_stages=required)
        if load_distillation_metrics(self.metrics_path).records != self.metrics_recorder.records:
            raise RuntimeError("distillation metrics reload differs from recorder state")

    def _record_scenario_artifact(
        self,
        *,
        scenario: WorkflowScenarioSpec,
        output_path: Path,
        num_samples: int,
        persistent_result: DaggerCollectResult | None,
        role_specs_by_name: dict[str, RoleArtifactSpec],
    ) -> None:
        if scenario.kind == "role":
            output_spec = replace(
                role_specs_by_name[scenario.source_roles[0]], dataset_path=output_path
            )
            artifact_manifest = create_role_artifact_manifest(output_spec, num_samples=num_samples)
            write_role_artifact_manifest(output_spec.manifest_path, artifact_manifest)
            self.role_artifacts.append(asdict(artifact_manifest))
        artifact = {
            "scenario": scenario.name,
            "kind": scenario.kind,
            "source_roles": list(scenario.source_roles),
            "quota": scenario.quota,
            "dataset_path": str(output_path.resolve()),
            "dataset_sha256": file_sha256(output_path),
            "num_samples": num_samples,
            "input_checkpoint_path": self.input_checkpoint_path,
            "input_checkpoint_sha256": self.input_checkpoint_sha256,
        }
        if persistent_result is not None:
            artifact.update(
                {
                    "input_weight_version": persistent_result.observed_weight_version,
                    "collector_worker_pid": persistent_result.worker_pid,
                    "collector_metrics": dict(persistent_result.metrics),
                    "collector_metadata": dict(persistent_result.metadata),
                }
            )
        self.scenario_artifacts.append(artifact)
        self.cumulative_sources.append(
            WorkflowDatasetSource(
                output_path,
                scenario.source_roles[0],
                scenario=scenario.name,
                preserve_row_role_labels=True,
            )
        )

    def _aggregate(self) -> tuple[Path, float]:
        path = self.run_dir / "datasets" / f"dagger_iteration_{self.iteration}_aggregate.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        started = float(self.performance_clock())
        self._sentinel("before_aggregate")
        self.cumulative_num_samples = int(
            self.aggregate_datasets(tuple(self.cumulative_sources), path)
        )
        self._sentinel("after_aggregate")
        elapsed = float(self.performance_clock()) - started
        self.emit_status(
            f"iteration={self.iteration} aggregated samples={self.cumulative_num_samples} "
            f"path={path}"
        )
        if self.cumulative_num_samples <= 0 or not path.is_file():
            raise RuntimeError(
                "DAgger aggregator must create the cumulative dataset and return a positive count"
            )
        return path, elapsed

    def _update_student(
        self, aggregate_path: Path
    ) -> tuple[Path, int, WorkflowStudentUpdateResult | None]:
        output = self.run_dir / "checkpoints" / f"dagger_iteration_{self.iteration}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        self.emit_status(f"iteration={self.iteration} updating student checkpoint={output}")
        raw_result = self.update_student(aggregate_path, self.current_checkpoint, output)
        learner_result = raw_result if isinstance(raw_result, WorkflowStudentUpdateResult) else None
        updates = learner_result.updates if learner_result is not None else int(raw_result)
        if updates <= 0 or not output.is_file():
            raise RuntimeError(
                "DAgger updater must create the next checkpoint and return a positive count"
            )
        self.emit_status(
            f"iteration={self.iteration} update complete updates={updates} checkpoint={output}"
        )
        return output, updates, learner_result

    def _record_workflow_metrics(
        self,
        *,
        aggregate_seconds: float,
        learner_result: WorkflowStudentUpdateResult | None,
    ) -> None:
        if learner_result is None:
            return
        assert self.performance_context is not None
        assert self.metrics_recorder is not None
        observations = (
            DistillationStageObservation(
                stage="cumulative_aggregation",
                duration_seconds=aggregate_seconds,
                row_count=self.cumulative_num_samples,
                env_step_count=0,
                success=True,
                error=None,
                cleanup_state="not_applicable",
            ),
            *learner_result.performance_stage_observations,
        )
        records = self.performance_context.enrich_workflow_iteration(
            outer_iteration=self.iteration,
            worker_pid=os.getpid(),
            checkpoint_path=self.input_checkpoint_path,
            checkpoint_sha256=self.input_checkpoint_sha256,
            weight_version=self.input_weight_version,
            observations=observations,
        )
        for record in records:
            self.metrics_recorder.add(record)
        self.metrics_recorder.write(
            self.metrics_path, required_stages=WORKFLOW_ITERATION_STAGE_NAMES
        )
        if load_distillation_metrics(self.metrics_path).records != self.metrics_recorder.records:
            raise RuntimeError("workflow metrics reload differs from recorder state")

    def _commit(self, aggregate_path: Path, output_checkpoint: Path, updates: int) -> None:
        self.commit_iteration(
            manifest_path=self.manifest_path,
            manifest=self.manifest,
            iteration=self.iteration,
            input_checkpoint_path=self.input_checkpoint_path,
            input_checkpoint_sha256=self.input_checkpoint_sha256,
            role_artifacts=self.role_artifacts,
            scenario_artifacts=self.scenario_artifacts,
            active_scenarios=self.active_scenarios,
            aggregate_path=aggregate_path,
            cumulative_num_samples=self.cumulative_num_samples,
            output_checkpoint=output_checkpoint,
            updates=updates,
            execution_mode=self.execution_mode,
            input_weight_version=self.input_weight_version,
            metrics_path=self.metrics_path,
            metrics_recorder=self.metrics_recorder,
        )

    def _sentinel(self, suffix: str) -> None:
        if self.runtime_sentinel is not None:
            self.runtime_sentinel(f"workflow/iteration_{self.iteration}/{suffix}")
