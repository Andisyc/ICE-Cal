"""Workflow artifact validation, persistence, resume, fork, and performance finalization."""

from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from unilab.algos.torch.distill.contracts.workflow import (
    ROLE_ARTIFACT_MANIFEST_VERSION,
    ArtifactDecision,
    RoleArtifactManifest,
    RoleArtifactPreflight,
    RoleArtifactSpec,
    WorkflowDatasetSource,
    WorkflowScenarioSpec,
    _normalize_json,
    config_fingerprint,
    file_sha256,
    resolve_walk_to_stop_role_pair,
)
from unilab.algos.torch.distill.datasets.io import (
    load_distillation_dataset,
    save_distillation_dataset,
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


def create_role_artifact_manifest(
    spec: RoleArtifactSpec,
    *,
    num_samples: int,
) -> RoleArtifactManifest:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if not spec.teacher_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"teacher checkpoint does not exist: {spec.teacher_checkpoint_path}"
        )
    if not spec.dataset_path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {spec.dataset_path}")
    return RoleArtifactManifest(
        manifest_version=ROLE_ARTIFACT_MANIFEST_VERSION,
        role=spec.role,
        task=spec.task,
        teacher_checkpoint_path=str(spec.teacher_checkpoint_path.resolve()),
        teacher_checkpoint_sha256=file_sha256(spec.teacher_checkpoint_path),
        dataset_path=str(spec.dataset_path.resolve()),
        dataset_sha256=file_sha256(spec.dataset_path),
        schema_version=int(spec.schema_version),
        student_obs_dim=int(spec.student_obs_dim),
        teacher_obs_dim=int(spec.teacher_obs_dim),
        teacher_action_dim=int(spec.teacher_action_dim),
        teacher_obs_key=spec.teacher_obs_key,
        teacher_projection=spec.teacher_projection,
        student_projection=spec.student_projection,
        student_drop_index=spec.student_drop_index,
        command_sample_filter=spec.command_sample_filter,
        command_info_key=spec.command_info_key,
        command_xy_threshold=float(spec.command_xy_threshold),
        command_yaw_threshold=float(spec.command_yaw_threshold),
        owner_config_sha256=config_fingerprint(spec.owner_config),
        num_samples=int(num_samples),
        target_height_info_key=spec.target_height_info_key,
    )


def _validate_workflow_scenarios(
    scenario_specs: Sequence[WorkflowScenarioSpec],
    role_specs: Sequence[RoleArtifactSpec],
) -> tuple[WorkflowScenarioSpec, ...]:
    scenarios = tuple(scenario_specs)
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError(f"workflow scenario names must be unique, got {names}")
    role_specs_by_role = {spec.role: spec for spec in role_specs}
    roles = set(role_specs_by_role)
    for scenario in scenarios:
        missing = sorted(set(scenario.source_roles) - roles)
        if missing:
            raise ValueError(
                f"workflow scenario {scenario.name!r} references unknown roles: {missing}"
            )
        if scenario.kind == "transition" and scenario.name == "walk_to_stop":
            resolve_walk_to_stop_role_pair(
                source_roles=scenario.source_roles,
                command_sample_filters={
                    role: role_specs_by_role[role].command_sample_filter
                    for role in scenario.source_roles
                },
                target_height_info_keys={
                    role: role_specs_by_role[role].target_height_info_key
                    for role in scenario.source_roles
                },
            )
    return scenarios


def write_role_artifact_manifest(
    path: str | Path,
    manifest: RoleArtifactManifest,
) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, resolved_path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(_normalize_json(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"workflow manifest must contain a JSON object: {path}")
    return payload


def _load_role_artifact_manifest(path: Path) -> RoleArtifactManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RoleArtifactManifest(**payload)


_COMPATIBILITY_FIELDS = (
    "manifest_version",
    "schema_version",
    "student_obs_dim",
    "teacher_obs_dim",
    "teacher_action_dim",
    "teacher_obs_key",
    "teacher_projection",
    "student_projection",
    "student_drop_index",
    "command_info_key",
    "target_height_info_key",
)


def _expected_manifest_values(spec: RoleArtifactSpec) -> dict[str, Any]:
    return {
        "manifest_version": ROLE_ARTIFACT_MANIFEST_VERSION,
        "role": spec.role,
        "task": spec.task,
        "teacher_checkpoint_path": str(spec.teacher_checkpoint_path.resolve()),
        "dataset_path": str(spec.dataset_path.resolve()),
        "schema_version": int(spec.schema_version),
        "student_obs_dim": int(spec.student_obs_dim),
        "teacher_obs_dim": int(spec.teacher_obs_dim),
        "teacher_action_dim": int(spec.teacher_action_dim),
        "teacher_obs_key": spec.teacher_obs_key,
        "teacher_projection": spec.teacher_projection,
        "student_projection": spec.student_projection,
        "student_drop_index": spec.student_drop_index,
        "command_sample_filter": spec.command_sample_filter,
        "command_info_key": spec.command_info_key,
        "target_height_info_key": spec.target_height_info_key,
        "command_xy_threshold": float(spec.command_xy_threshold),
        "command_yaw_threshold": float(spec.command_yaw_threshold),
        "owner_config_sha256": config_fingerprint(spec.owner_config),
    }


def preflight_role_artifact(
    spec: RoleArtifactSpec,
    *,
    require_row_role_labels: bool = False,
) -> RoleArtifactPreflight:
    if not spec.dataset_path.is_file():
        return RoleArtifactPreflight(
            role=spec.role,
            decision=ArtifactDecision.COLLECT,
            dataset_path=spec.dataset_path,
            manifest_path=spec.manifest_path,
            mismatches=("dataset_missing",),
        )
    if not spec.manifest_path.is_file():
        return RoleArtifactPreflight(
            role=spec.role,
            decision=ArtifactDecision.STALE,
            dataset_path=spec.dataset_path,
            manifest_path=spec.manifest_path,
            mismatches=("manifest_missing",),
        )
    try:
        manifest = _load_role_artifact_manifest(spec.manifest_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return RoleArtifactPreflight(
            role=spec.role,
            decision=ArtifactDecision.STALE,
            dataset_path=spec.dataset_path,
            manifest_path=spec.manifest_path,
            mismatches=("manifest_invalid",),
        )

    expected = _expected_manifest_values(spec)
    mismatches = [
        field_name
        for field_name, expected_value in expected.items()
        if getattr(manifest, field_name) != expected_value
    ]
    incompatible = [field_name for field_name in mismatches if field_name in _COMPATIBILITY_FIELDS]
    if incompatible:
        return RoleArtifactPreflight(
            role=spec.role,
            decision=ArtifactDecision.INCOMPATIBLE,
            dataset_path=spec.dataset_path,
            manifest_path=spec.manifest_path,
            mismatches=tuple(incompatible),
        )

    if not spec.teacher_checkpoint_path.is_file():
        mismatches.append("teacher_checkpoint_missing")
    elif file_sha256(spec.teacher_checkpoint_path) != manifest.teacher_checkpoint_sha256:
        mismatches.append("teacher_checkpoint_sha256")
    if file_sha256(spec.dataset_path) != manifest.dataset_sha256:
        mismatches.append("dataset_sha256")
    if require_row_role_labels or spec.target_height_info_key is not None:
        try:
            dataset = load_distillation_dataset(
                spec.dataset_path,
                expected_student_obs_dim=spec.student_obs_dim,
                expected_teacher_obs_dim=spec.teacher_obs_dim,
                expected_teacher_action_dim=spec.teacher_action_dim,
            )
        except (OSError, TypeError, ValueError, KeyError, pickle.UnpicklingError):
            mismatches.append("dataset_schema")
        else:
            if dataset.role_labels is None:
                if require_row_role_labels:
                    mismatches.append("role_labels")
            if spec.target_height_info_key is not None and dataset.target_height is None:
                mismatches.append("target_height")
    if mismatches:
        return RoleArtifactPreflight(
            role=spec.role,
            decision=ArtifactDecision.STALE,
            dataset_path=spec.dataset_path,
            manifest_path=spec.manifest_path,
            mismatches=tuple(mismatches),
        )
    return RoleArtifactPreflight(
        role=spec.role,
        decision=ArtifactDecision.REUSE,
        dataset_path=spec.dataset_path,
        manifest_path=spec.manifest_path,
    )


def preflight_role_artifacts(
    specs: Sequence[RoleArtifactSpec],
    *,
    require_row_role_labels: bool = False,
) -> tuple[RoleArtifactPreflight, ...]:
    roles = [spec.role for spec in specs]
    if len(roles) != len(set(roles)):
        raise ValueError(f"workflow roles must be unique, got {roles}")
    return tuple(
        preflight_role_artifact(spec, require_row_role_labels=require_row_role_labels)
        for spec in specs
    )


def adopt_legacy_role_artifact(spec: RoleArtifactSpec) -> RoleArtifactManifest:
    """Adopt or repair a legacy role dataset under the current role contract."""

    existing_manifest = spec.manifest_path.is_file()
    if existing_manifest:
        preflight = preflight_role_artifact(spec)
        if preflight.decision is not ArtifactDecision.REUSE:
            raise ValueError(
                f"cannot adopt stale role artifact {spec.role!r}: {list(preflight.mismatches)}"
            )
    dataset = load_distillation_dataset(
        spec.dataset_path,
        expected_student_obs_dim=spec.student_obs_dim,
        expected_teacher_obs_dim=spec.teacher_obs_dim,
        expected_teacher_action_dim=spec.teacher_action_dim,
        device="cpu",
    )
    if dataset.teacher_actions is None:
        raise ValueError(f"legacy role dataset has no cached teacher actions: {spec.dataset_path}")
    normalized_legacy = False
    metadata = dataset.metadata
    expected_task_name = (
        spec.owner_config.get("training", {}).get("task_name")
        if isinstance(spec.owner_config.get("training"), Mapping)
        else None
    )
    checks: dict[str, Any] = {
        "teacher_projection": spec.teacher_projection,
        "student_projection": spec.student_projection,
        "teacher_obs_key": spec.teacher_obs_key,
        "student_drop_index": spec.student_drop_index,
        "command_sample_filter": spec.command_sample_filter,
        "command_info_key": spec.command_info_key,
        "command_xy_threshold": float(spec.command_xy_threshold),
        "command_yaw_threshold": float(spec.command_yaw_threshold),
    }
    if expected_task_name not in (None, ""):
        checks["task_name"] = expected_task_name
    mismatches = [key for key, expected in checks.items() if metadata.get(key) != expected]
    if spec.target_height_info_key is not None and dataset.target_height is None:
        mismatches.append("target_height")
    teacher_path = metadata.get("teacher_policy_checkpoint_path")
    if teacher_path in (None, ""):
        mismatches.append("teacher_policy_checkpoint_path")
    elif Path(str(teacher_path)).resolve() != spec.teacher_checkpoint_path.resolve():
        mismatches.append("teacher_policy_checkpoint_path")
    if mismatches:
        raise ValueError(
            f"legacy role dataset metadata is incompatible for {spec.role!r}: {mismatches}"
        )
    if dataset.role_labels is None:
        # Legacy collectors did not persist row labels. The workflow already
        # owns this file as a role-specific artifact, so this is an explicit,
        # lossless schema migration rather than an inferred router decision.
        normalized_metadata = dict(metadata)
        normalized_metadata["legacy_role_labels_adopted"] = True
        dataset = replace(
            dataset,
            metadata=normalized_metadata,
            role_labels=(spec.role,) * dataset.num_samples,
        )
        save_distillation_dataset(spec.dataset_path, dataset)
        normalized_legacy = True
    elif any(label != spec.role for label in dataset.role_labels):
        raise ValueError(
            f"legacy role dataset labels do not match role {spec.role!r}: "
            f"{sorted(set(dataset.role_labels))}"
        )
    if existing_manifest and not normalized_legacy:
        return _load_role_artifact_manifest(spec.manifest_path)
    manifest = create_role_artifact_manifest(spec, num_samples=dataset.num_samples)
    write_role_artifact_manifest(spec.manifest_path, manifest)
    return manifest

def _verified_current_checkpoint(manifest: Mapping[str, Any]) -> Path:
    iterations = manifest.get("dagger_iterations", [])
    if iterations:
        latest = iterations[-1]
        path = Path(str(latest["checkpoint_path"]))
        expected_hash = str(latest["checkpoint_sha256"])
    else:
        path = Path(str(manifest["bootstrap_checkpoint_path"]))
        expected_hash = str(manifest["bootstrap_checkpoint_sha256"])
    if not path.is_file():
        raise FileNotFoundError(f"workflow checkpoint does not exist: {path}")
    if file_sha256(path) != expected_hash:
        raise ValueError(f"workflow checkpoint hash mismatch: {path}")
    return path


def _manifest_sources(manifest: Mapping[str, Any]) -> list[WorkflowDatasetSource]:
    sources = [
        WorkflowDatasetSource(
            path=Path(str(item["path"])),
            role=str(item["role"]),
            scenario=item.get("scenario"),
            preserve_row_role_labels=bool(item.get("preserve_row_role_labels", False)),
        )
        for item in manifest.get("bootstrap_sources", [])
    ]
    for iteration in manifest.get("dagger_iterations", []):
        if iteration.get("scenario_artifacts"):
            sources.extend(
                WorkflowDatasetSource(
                    path=Path(str(item["dataset_path"])),
                    role=str(item["source_roles"][0]),
                    scenario=str(item["scenario"]),
                    preserve_row_role_labels=True,
                )
                for item in iteration["scenario_artifacts"]
            )
            continue
        sources.extend(
            WorkflowDatasetSource(
                path=Path(str(item["dataset_path"])),
                role=str(item["role"]),
            )
            for item in iteration["role_artifacts"]
        )
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(f"workflow cumulative source does not exist: {source.path}")
    return sources

def finalize_workflow_performance(
    *,
    run_dir: str | Path,
    performance_context: DistillationPerformanceRunContext,
    cleanup_duration_seconds: float,
    cleanup_report: Mapping[str, Any],
) -> Path:
    """Persist the post-close cleanup metric and lifecycle report atomically."""

    if not math.isfinite(cleanup_duration_seconds) or cleanup_duration_seconds < 0:
        raise ValueError("cleanup_duration_seconds must be finite and non-negative")
    if not isinstance(cleanup_report, Mapping):
        raise TypeError("cleanup_report must be a mapping")
    report = dict(cleanup_report)
    if performance_context.execution_mode == "persistent_async":
        worker_pid = report.get("worker_pid")
        resource_counters = report.get("resource_counters")
        if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
            raise ValueError("persistent cleanup_report requires positive worker_pid")
        if not isinstance(resource_counters, Mapping):
            raise ValueError("persistent cleanup_report requires resource_counters mapping")
    else:
        worker_pid = os.getpid()

    resolved_run_dir = Path(run_dir).resolve()
    manifest_path = resolved_run_dir / "run_manifest.json"
    metrics_path = resolved_run_dir / "distillation_metrics.json"
    manifest = _load_json(manifest_path)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"workflow cleanup requires distillation metrics: {metrics_path}")
    existing_cleanup = manifest.get("performance_cleanup")
    if existing_cleanup is not None:
        if not isinstance(existing_cleanup, Mapping) or existing_cleanup.get("state") != "complete":
            raise ValueError("existing performance_cleanup must be complete")
        if manifest.get("distillation_metrics_path") != str(metrics_path.resolve()):
            raise ValueError("cleanup metrics manifest path mismatch")
        if manifest.get("distillation_metrics_sha256") != file_sha256(metrics_path):
            raise ValueError("cleanup metrics manifest hash mismatch")
        existing_metrics = load_distillation_metrics(metrics_path)
        if manifest.get("distillation_metrics_record_count") != len(existing_metrics.records):
            raise ValueError("cleanup metrics manifest record count mismatch")
        cleanup_records = [
            record for record in existing_metrics.records if record.stage == "cleanup"
        ]
        if len(cleanup_records) != 1:
            raise ValueError("completed cleanup requires exactly one cleanup record")
        if cleanup_records[0].identity.run_signature() != performance_context.run_signature():
            raise ValueError("existing cleanup performance context identity drift")
        if cleanup_records[0].cleanup_state != "complete":
            raise ValueError("existing cleanup record must be complete")
        return metrics_path
    iterations = manifest.get("dagger_iterations", [])
    if not isinstance(iterations, list) or not iterations:
        raise ValueError("workflow cleanup requires a completed DAgger iteration")
    last_iteration = iterations[-1]
    if not isinstance(last_iteration, Mapping):
        raise ValueError("last DAgger iteration manifest entry must be a mapping")

    loaded_metrics = load_distillation_metrics(metrics_path)
    if not loaded_metrics.records:
        raise ValueError("workflow cleanup requires non-empty distillation metrics")
    if loaded_metrics.records[0].identity.run_signature() != performance_context.run_signature():
        raise ValueError("cleanup performance context identity drift")
    recorder = DistillationMetricsRecorder()
    for record in loaded_metrics.records:
        recorder.add(record)

    input_weight_version = last_iteration.get("input_weight_version")
    cleanup_observation = DistillationStageObservation(
        stage="cleanup",
        duration_seconds=cleanup_duration_seconds,
        row_count=0,
        env_step_count=0,
        success=True,
        error=None,
        cleanup_state="complete",
    )
    recorder.add(
        performance_context.enrich_cleanup(
            outer_iteration=int(last_iteration["iteration"]),
            worker_pid=worker_pid,
            checkpoint_path=str(last_iteration["input_checkpoint_path"]),
            checkpoint_sha256=str(last_iteration["input_checkpoint_sha256"]),
            weight_version=(None if input_weight_version is None else int(input_weight_version)),
            observation=cleanup_observation,
        )
    )
    recorder.write(metrics_path, required_stages=("cleanup",))
    persisted_metrics = load_distillation_metrics(metrics_path)
    if persisted_metrics.records != recorder.records:
        raise RuntimeError("cleanup metrics reload differs from recorder state")

    manifest["performance_cleanup"] = {
        "state": "complete",
        "duration_seconds": cleanup_duration_seconds,
        "report": report,
    }
    manifest["distillation_metrics_path"] = str(metrics_path.resolve())
    manifest["distillation_metrics_sha256"] = file_sha256(metrics_path)
    manifest["distillation_metrics_record_count"] = len(recorder.records)
    _write_json_atomic(manifest_path, manifest)
    return metrics_path


def fork_workflow_run(*, parent_run_dir: str | Path, run_dir: str | Path) -> Path:
    parent_manifest_path = Path(parent_run_dir) / "run_manifest.json"
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(f"parent workflow manifest does not exist: {parent_manifest_path}")
    resolved_run_dir = Path(run_dir)
    manifest_path = resolved_run_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"fork workflow run already exists: {manifest_path}")
    parent = _load_json(parent_manifest_path)
    checkpoint = _verified_current_checkpoint(parent)
    parent_iterations = parent.get("dagger_iterations", [])
    if parent_iterations:
        latest = parent_iterations[-1]
        bootstrap_dataset_path = Path(str(latest["aggregate_dataset_path"]))
        bootstrap_dataset_sha256 = str(latest["aggregate_dataset_sha256"])
        bootstrap_num_samples = int(latest["aggregate_num_samples"])
        bootstrap_sources = [
            {
                "path": str(source.path.resolve()),
                "role": source.role,
                **({"scenario": source.scenario} if source.scenario is not None else {}),
                "preserve_row_role_labels": source.preserve_row_role_labels,
            }
            for source in _manifest_sources(parent)
        ]
    else:
        bootstrap_dataset_path = Path(str(parent["bootstrap_dataset_path"]))
        bootstrap_dataset_sha256 = str(parent["bootstrap_dataset_sha256"])
        bootstrap_num_samples = int(parent["bootstrap_num_samples"])
        bootstrap_sources = list(parent.get("bootstrap_sources", []))
    if not bootstrap_dataset_path.is_file():
        raise FileNotFoundError(f"parent workflow dataset does not exist: {bootstrap_dataset_path}")
    if file_sha256(bootstrap_dataset_path) != bootstrap_dataset_sha256:
        raise ValueError(f"parent workflow dataset hash mismatch: {bootstrap_dataset_path}")
    payload = {
        "manifest_version": 1,
        "run_id": resolved_run_dir.name,
        "mode": "fork",
        "parent_run_manifest": str(parent_manifest_path.resolve()),
        "parent_run_manifest_sha256": file_sha256(parent_manifest_path),
        "stage": "BOOTSTRAP_COMPLETE",
        "role_decisions": {item["role"]: "REUSE" for item in parent["role_artifacts"]},
        "role_artifacts": list(parent["role_artifacts"]),
        "bootstrap_sources": bootstrap_sources,
        "bootstrap_dataset_path": str(bootstrap_dataset_path.resolve()),
        "bootstrap_dataset_sha256": bootstrap_dataset_sha256,
        "bootstrap_num_samples": bootstrap_num_samples,
        "bootstrap_checkpoint_path": str(checkpoint.resolve()),
        "bootstrap_checkpoint_sha256": file_sha256(checkpoint),
        "bootstrap_updates": 0,
        "completed_dagger_iterations": 0,
        "dagger_iterations": [],
        "scenario_specs": list(parent.get("scenario_specs", [])),
    }
    _write_json_atomic(manifest_path, payload)
    return manifest_path
