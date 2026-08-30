"""Bootstrap dataset collection, assembly, and first student update."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from unilab.algos.torch.distill.contracts.workflow import (
    ArtifactDecision,
    BootstrapWorkflowResult,
    RoleArtifactSpec,
    WorkflowScenarioSpec,
)
from unilab.algos.torch.distill.workflows.artifacts import (
    _load_role_artifact_manifest,
    _validate_workflow_scenarios,
    _write_json_atomic,
    create_role_artifact_manifest,
    file_sha256,
    preflight_role_artifact,
    preflight_role_artifacts,
    write_role_artifact_manifest,
)


def run_bootstrap_workflow(
    *,
    run_dir: str | Path,
    role_specs: Sequence[RoleArtifactSpec],
    collect_role: Callable[[RoleArtifactSpec], int],
    assemble_roles: Callable[[tuple[Path, ...], Path], int],
    update_student: Callable[[Path, Path], int],
    scenario_specs: Sequence[WorkflowScenarioSpec] | None = None,
) -> BootstrapWorkflowResult:
    resolved_run_dir = Path(run_dir)
    manifest_path = resolved_run_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"workflow run already exists; use resume or fork: {manifest_path}")
    if not role_specs:
        raise ValueError("bootstrap workflow requires at least one role")
    active_scenarios = (
        None if scenario_specs is None else _validate_workflow_scenarios(scenario_specs, role_specs)
    )

    require_row_role_labels = active_scenarios is not None and any(
        scenario.kind == "role" for scenario in active_scenarios
    )
    preflight = preflight_role_artifacts(
        role_specs,
        require_row_role_labels=require_row_role_labels,
    )
    blocked = [
        result
        for result in preflight
        if result.decision in (ArtifactDecision.STALE, ArtifactDecision.INCOMPATIBLE)
    ]
    if blocked:
        details = ", ".join(
            f"{result.role}={result.decision.value}:{list(result.mismatches)}" for result in blocked
        )
        raise ValueError(f"workflow artifact preflight failed closed: {details}")

    role_decisions = {result.role: result.decision.value for result in preflight}
    for spec, result in zip(role_specs, preflight, strict=True):
        if result.decision is not ArtifactDecision.COLLECT:
            continue
        num_samples = int(collect_role(spec))
        if num_samples <= 0:
            raise ValueError(
                f"collector for role {spec.role!r} returned invalid sample count {num_samples}"
            )
        manifest = create_role_artifact_manifest(spec, num_samples=num_samples)
        write_role_artifact_manifest(spec.manifest_path, manifest)

    verified = preflight_role_artifacts(role_specs)
    not_reusable = [result for result in verified if result.decision is not ArtifactDecision.REUSE]
    if not_reusable:
        details = ", ".join(
            f"{result.role}={result.decision.value}:{list(result.mismatches)}"
            for result in not_reusable
        )
        raise RuntimeError(f"workflow role artifact verification failed: {details}")

    datasets_dir = resolved_run_dir / "datasets"
    checkpoints_dir = resolved_run_dir / "checkpoints"
    bootstrap_dataset_path = datasets_dir / "bootstrap_merged.pt"
    bootstrap_num_samples = int(
        assemble_roles(tuple(spec.dataset_path for spec in role_specs), bootstrap_dataset_path)
    )
    if bootstrap_num_samples <= 0 or not bootstrap_dataset_path.is_file():
        raise RuntimeError(
            "bootstrap assembler must create the merged dataset and return a positive sample count"
        )

    checkpoint_path = checkpoints_dir / "bootstrap_student.pt"
    bootstrap_updates = int(update_student(bootstrap_dataset_path, checkpoint_path))
    if bootstrap_updates <= 0 or not checkpoint_path.is_file():
        raise RuntimeError(
            "bootstrap updater must create the checkpoint and return a positive update count"
        )

    role_artifacts = [
        asdict(_load_role_artifact_manifest(spec.manifest_path)) for spec in role_specs
    ]
    scenario_by_role = {
        scenario.source_roles[0]: scenario.name
        for scenario in (active_scenarios or ())
        if scenario.kind == "role"
    }
    bootstrap_sources: list[dict[str, Any]] = []
    for spec in role_specs:
        source: dict[str, Any] = {
            "path": str(spec.dataset_path.resolve()),
            "role": spec.role,
        }
        if spec.role in scenario_by_role:
            source["scenario"] = scenario_by_role[spec.role]
            source["preserve_row_role_labels"] = True
        bootstrap_sources.append(source)
    manifest_payload = {
        "manifest_version": 1,
        "run_id": resolved_run_dir.name,
        "mode": "fresh",
        "stage": "BOOTSTRAP_COMPLETE",
        "role_decisions": role_decisions,
        "role_artifacts": role_artifacts,
        "bootstrap_dataset_path": str(bootstrap_dataset_path.resolve()),
        "bootstrap_dataset_sha256": file_sha256(bootstrap_dataset_path),
        "bootstrap_num_samples": bootstrap_num_samples,
        "bootstrap_checkpoint_path": str(checkpoint_path.resolve()),
        "bootstrap_checkpoint_sha256": file_sha256(checkpoint_path),
        "bootstrap_updates": bootstrap_updates,
        "completed_dagger_iterations": 0,
        "dagger_iterations": [],
        "bootstrap_sources": bootstrap_sources,
    }
    if active_scenarios is not None:
        manifest_payload["scenario_specs"] = [scenario.as_dict() for scenario in active_scenarios]
    _write_json_atomic(
        manifest_path,
        manifest_payload,
    )
    return BootstrapWorkflowResult(
        run_dir=resolved_run_dir,
        manifest_path=manifest_path,
        role_decisions=role_decisions,
        bootstrap_dataset_path=bootstrap_dataset_path,
        bootstrap_num_samples=bootstrap_num_samples,
        checkpoint_path=checkpoint_path,
        bootstrap_updates=bootstrap_updates,
    )
