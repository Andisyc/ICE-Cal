"""Production owner extracted from the generic distillation entrypoint: entry_workflow.py."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.contracts.workflow import (
    RoleArtifactSpec,
    WorkflowDatasetSource,
    WorkflowScenarioCollectionResult,
    WorkflowScenarioSpec,
    WorkflowStudentUpdateResult,
    config_fingerprint,
    file_sha256,
)
from unilab.algos.torch.distill.observability.performance import (
    DistillationPerformanceRunContext,
    DistillationStageObservation,
)
from unilab.algos.torch.distill.runtime.g1_worker import (
    build_persistent_g1_distillation_runtime,
)
from unilab.algos.torch.distill.workflows.artifacts import (
    adopt_legacy_role_artifact,
    finalize_workflow_performance,
    fork_workflow_run,
)
from unilab.algos.torch.distill.workflows.bootstrap import run_bootstrap_workflow
from unilab.algos.torch.distill.workflows.dagger import run_multirole_dagger_workflow
from unilab.algos.torch.distill.workflows.diagnostics import (
    WorkflowLoggerCallbacks,
    _probe_torch_serialization_runtime,
)
from unilab.algos.torch.distill.workflows.entry_collection import (
    _distill_device,
    run_collect_dataset,
)
from unilab.algos.torch.distill.workflows.entry_plan import (
    _workflow_path,
    _workflow_role_cfg,
    _workflow_role_entries,
    _workflow_scenario_specs,
    resolve_workflow_entry_plan,
)
from unilab.algos.torch.distill.workflows.entry_training import (
    run_multitask_dataset_assembly,
    run_offline_dataset_update,
)
from unilab.algos.torch.distill.workflows.runtime import (
    WorkflowRuntimeSession,
)
from unilab.algos.torch.distill.workflows.runtime import (
    workflow_entry_result as _workflow_entry_result,
)
from unilab.algos.torch.distill.workflows.transition import collect_legacy_transition_scenario
from unilab.logging import OffPolicyLogger


@dataclass(frozen=True)
class WorkflowEntryOperations:
    cfg: DictConfig
    role_cfgs: dict[str, DictConfig]
    specs: tuple[RoleArtifactSpec, ...]
    scenario_specs: tuple[WorkflowScenarioSpec, ...] | None
    execution_mode: str
    performance_clock: Callable[[], float]
    logger_callbacks: WorkflowLoggerCallbacks | None = None

    def collect_role(self, spec: RoleArtifactSpec) -> int:
        spec.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        role_cfg = OmegaConf.create(OmegaConf.to_container(self.role_cfgs[spec.role], resolve=True))
        role_cfg.training.collect_role_label = spec.role
        result = run_collect_dataset(role_cfg, dataset_path=spec.dataset_path)
        return int(result["dataset_num_samples"])

    def assemble_roles(self, dataset_paths: tuple[Path, ...], output_path: Path) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assembly_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        role_scenarios = {
            scenario.source_roles[0]: scenario.name
            for scenario in (self.scenario_specs or ())
            if scenario.kind == "role"
        }
        assembly_cfg.training.multitask_sources = [
            {
                "path": str(path),
                "role": spec.role,
                **(
                    {
                        "scenario": role_scenarios[spec.role],
                        "preserve_row_role_labels": True,
                    }
                    if spec.role in role_scenarios
                    else {}
                ),
            }
            for path, spec in zip(dataset_paths, self.specs, strict=True)
        ]
        assembly_cfg.training.multitask_expected_student_obs_dim = self.specs[0].student_obs_dim
        assembly_cfg.training.multitask_expected_teacher_obs_dim = self.specs[0].teacher_obs_dim
        assembly_cfg.training.multitask_expected_teacher_action_dim = self.specs[0].teacher_action_dim
        result = run_multitask_dataset_assembly(assembly_cfg, dataset_path=output_path)
        return int(result["dataset_num_samples"])

    def update_student(self, dataset_path: Path, checkpoint_path: Path) -> int:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        updates = int(OmegaConf.select(self.cfg, "training.workflow.bootstrap_updates", default=20000))
        run_offline_dataset_update(
            self.cfg,
            teacher_checkpoint=self.specs[0].teacher_checkpoint_path,
            dataset_path=dataset_path,
            batch_size=int(
                OmegaConf.select(self.cfg, "training.workflow.bootstrap_batch_size", default=512)
            ),
            max_updates=updates,
            checkpoint_path=checkpoint_path,
            device=_distill_device(self.cfg),
        )
        return updates

    def collect_dagger_role(
        self,
        output_spec: RoleArtifactSpec,
        checkpoint_path: Path,
        _iteration: int,
        output_path: Path,
        *,
        workflow_scenario: str | None = None,
    ) -> int | WorkflowScenarioCollectionResult:
        role_cfg = OmegaConf.create(
            OmegaConf.to_container(self.role_cfgs[output_spec.role], resolve=True)
        )
        role_cfg.training.collect_action_mode = "student_policy"
        role_cfg.training.collect_rollout_checkpoint_path = str(checkpoint_path)
        role_cfg.training.collect_role_label = output_spec.role
        if workflow_scenario is not None:
            role_cfg.training.collect_workflow_scenario = workflow_scenario
        role_cfg.training.collect_num_samples = int(
            OmegaConf.select(
                self.cfg,
                "training.workflow.dagger_samples_per_role",
                default=65536,
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        collected = run_collect_dataset(
            role_cfg,
            dataset_path=output_path,
            performance_clock=(self.performance_clock if self.execution_mode == "legacy" else None),
        )
        if self.execution_mode != "legacy":
            return int(collected["dataset_num_samples"])
        payloads = collected.get("performance_stage_observations")
        if not isinstance(payloads, list):
            raise ValueError("legacy role request performance observations are missing")
        return WorkflowScenarioCollectionResult(
            num_samples=int(collected["dataset_num_samples"]),
            worker_pid=os.getpid(),
            performance_metrics_schema_version=int(collected["performance_metrics_schema_version"]),
            performance_stage_observations=tuple(
                DistillationStageObservation.from_dict(payload) for payload in payloads
            ),
        )

    def collect_dagger_scenario(
        self,
        scenario: WorkflowScenarioSpec,
        checkpoint_path: Path,
        _iteration: int,
        output_path: Path,
    ) -> int | WorkflowScenarioCollectionResult:
        if scenario.kind == "role":
            role_spec = next(spec for spec in self.specs if spec.role == scenario.source_roles[0])
            return collect_dagger_role(
                replace(role_spec, dataset_path=output_path),
                checkpoint_path,
                _iteration,
                output_path,
                workflow_scenario=scenario.name,
            )
        return collect_legacy_transition_scenario(
            cfg=self.cfg,
            scenario=scenario,
            role_cfgs=self.role_cfgs,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            performance_clock=self.performance_clock,
        )

    def aggregate_dagger_sources(
        self,
        sources: tuple[WorkflowDatasetSource, ...],
        output_path: Path,
    ) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assembly_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        source_records = []
        for source_index, source in enumerate(sources):
            source_record = {
                "source_index": source_index,
                "path": str(source.path),
                "role": source.role,
            }
            if source.scenario is not None:
                source_record["scenario"] = source.scenario
                source_record["preserve_row_role_labels"] = source.preserve_row_role_labels
            source_records.append(source_record)
        source_snapshot_path = output_path.parent / f"{output_path.name}.sources.json"
        source_snapshot_path.write_text(
            json.dumps(
                {
                    "schema": "unilab.distill.workflow.aggregate_sources.v1",
                    "aggregate_path": str(output_path),
                    "source_count": len(source_records),
                    "sources": source_records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        assembly_cfg.training.multitask_sources = source_records
        assembly_cfg.training.multitask_expected_student_obs_dim = self.specs[0].student_obs_dim
        assembly_cfg.training.multitask_expected_teacher_obs_dim = self.specs[0].teacher_obs_dim
        assembly_cfg.training.multitask_expected_teacher_action_dim = self.specs[0].teacher_action_dim
        assembled = run_multitask_dataset_assembly(assembly_cfg, dataset_path=output_path)
        return int(assembled["dataset_num_samples"])

    def update_dagger_student(
        self,
        dataset_path: Path,
        input_checkpoint_path: Path,
        output_checkpoint_path: Path,
    ) -> WorkflowStudentUpdateResult:
        if self.logger_callbacks is None:
            raise RuntimeError("DAgger update callback requires logger callbacks")
        update_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        update_cfg.training.offline_init_checkpoint = str(input_checkpoint_path)
        update_cfg.training.offline_resume_optimizer = False
        update_cfg.training.offline_save_optimizer = False
        update_cfg.training.offline_repeat_dataset = True
        update_cfg.training.offline_shuffle = True
        update_cfg.training.offline_balance_key = str(
            OmegaConf.select(self.cfg, "training.workflow.dagger_balance_key", default="role")
        )
        update_cfg.training.offline_balanced_labels = list(
            (
                [scenario.name for scenario in self.scenario_specs]
                if self.scenario_specs is not None
                else OmegaConf.select(self.cfg, "training.workflow.dagger_balanced_labels", default=[])
            )
        )
        if self.scenario_specs is not None:
            update_cfg.training.offline_balance_key = "scenario"
            update_cfg.training.offline_balance_quotas = {
                scenario.name: scenario.quota for scenario in self.scenario_specs
            }
            update_cfg.training.offline_min_balanced_replay_passes = int(
                OmegaConf.select(
                    self.cfg,
                    "training.workflow.dagger_min_transition_replay_passes",
                    default=0,
                )
            )
            update_cfg.training.offline_min_balanced_replay_labels = list(
                OmegaConf.select(
                    self.cfg,
                    "training.workflow.dagger_min_transition_replay_labels",
                    default=["walk_to_stop"],
                )
            )
        updates = int(
            OmegaConf.select(
                self.cfg,
                "training.workflow.dagger_updates_per_iteration",
                default=128,
            )
        )
        output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_offline_dataset_update(
            update_cfg,
            teacher_checkpoint=self.specs[0].teacher_checkpoint_path,
            dataset_path=dataset_path,
            batch_size=int(
                OmegaConf.select(self.cfg, "training.workflow.dagger_batch_size", default=512)
            ),
            max_updates=updates,
            checkpoint_path=output_checkpoint_path,
            device=_distill_device(self.cfg),
            auto_expand_replay_budget=True,
            progress_callback=self.logger_callbacks.on_update_progress,
            performance_clock=self.performance_clock,
        )
        return WorkflowStudentUpdateResult(
            updates=int(result["update_count"]),
            performance_stage_observations=tuple(
                DistillationStageObservation.from_dict(observation)
                for observation in result["performance_stage_observations"]
            ),
        )


def run_single_entry_workflow(
    cfg: DictConfig,
    *,
    persistent_scenario_collector_factory: Callable[..., Any] | None = None,
    performance_clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Adapt role owner configs to the distillation workflow stage owner."""

    plan = resolve_workflow_entry_plan(
        cfg,
        persistent_factory_provided=persistent_scenario_collector_factory is not None,
    )
    execution_mode = plan.execution_mode
    run_dir = plan.run_dir
    role_cfgs = plan.role_cfgs
    specs = list(plan.role_specs)
    scenario_specs = plan.scenario_specs
    mode = plan.mode
    if execution_mode == "persistent_async" and persistent_scenario_collector_factory is None:
        persistent_scenario_collector_factory = build_persistent_g1_distillation_runtime
    if plan.adopt_legacy_artifacts:
        for spec in specs:
            if spec.dataset_path.is_file():
                adopt_legacy_role_artifact(spec)




    operations = WorkflowEntryOperations(
        cfg=cfg,
        role_cfgs=role_cfgs,
        specs=tuple(specs),
        scenario_specs=scenario_specs,
        execution_mode=execution_mode,
        performance_clock=performance_clock,
    )
    manifest_path = run_dir / "run_manifest.json"
    if mode == "fork":
        parent_run_dir = OmegaConf.select(cfg, "training.workflow.parent_run_dir")
        if parent_run_dir in (None, ""):
            raise ValueError("training.workflow.mode=fork requires parent_run_dir")
        fork_workflow_run(
            parent_run_dir=_workflow_path(parent_run_dir),
            run_dir=run_dir,
        )
        bootstrap_result = None
    elif mode == "resume" or (mode == "auto" and manifest_path.is_file()):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"workflow resume manifest does not exist: {manifest_path}")
        bootstrap_result = None
    else:
        bootstrap_result = run_bootstrap_workflow(
            run_dir=run_dir,
            role_specs=tuple(specs),
            collect_role=operations.collect_role,
            assemble_roles=operations.assemble_roles,
            update_student=operations.update_student,
            scenario_specs=scenario_specs,
        )

    dagger_result = WorkflowRuntimeSession(
        cfg=cfg,
        operations=operations,
        execution_mode=execution_mode,
        run_dir=run_dir,
        role_cfgs=role_cfgs,
        specs=tuple(specs),
        scenario_specs=scenario_specs,
        persistent_scenario_collector_factory=persistent_scenario_collector_factory,
        performance_clock=performance_clock,
        run_dagger=run_multirole_dagger_workflow,
        finalize_performance=finalize_workflow_performance,
        logger_cls=OffPolicyLogger,
        file_sha256_fn=file_sha256,
        config_fingerprint_fn=config_fingerprint,
        runtime_probe=_probe_torch_serialization_runtime,
    ).run()
    return _workflow_entry_result(
        mode=mode,
        execution_mode=execution_mode,
        bootstrap_result=bootstrap_result,
        dagger_result=dagger_result,
    )
