"""Compatibility surface for distillation workflow owners."""

from unilab.algos.torch.distill.contracts.workflow import (
    ArtifactDecision,
    BootstrapWorkflowResult,
    DaggerWorkflowResult,
    RoleArtifactManifest,
    RoleArtifactPreflight,
    RoleArtifactSpec,
    WalkToStopRolePair,
    WorkflowDatasetSource,
    WorkflowScenarioCollectionResult,
    WorkflowScenarioCollector,
    WorkflowScenarioSpec,
    WorkflowStudentUpdateResult,
    resolve_walk_to_stop_role_pair,
)
from unilab.algos.torch.distill.workflows.artifacts import (
    adopt_legacy_role_artifact,
    config_fingerprint,
    create_role_artifact_manifest,
    file_sha256,
    finalize_workflow_performance,
    fork_workflow_run,
    preflight_role_artifact,
    preflight_role_artifacts,
    write_role_artifact_manifest,
)
from unilab.algos.torch.distill.workflows.bootstrap import run_bootstrap_workflow
from unilab.algos.torch.distill.workflows.dagger import run_multirole_dagger_workflow

__all__ = [
    "ArtifactDecision",
    "BootstrapWorkflowResult",
    "DaggerWorkflowResult",
    "RoleArtifactManifest",
    "RoleArtifactPreflight",
    "RoleArtifactSpec",
    "WalkToStopRolePair",
    "WorkflowDatasetSource",
    "WorkflowScenarioCollectionResult",
    "WorkflowScenarioCollector",
    "WorkflowScenarioSpec",
    "WorkflowStudentUpdateResult",
    "adopt_legacy_role_artifact",
    "config_fingerprint",
    "create_role_artifact_manifest",
    "file_sha256",
    "finalize_workflow_performance",
    "fork_workflow_run",
    "preflight_role_artifact",
    "preflight_role_artifacts",
    "resolve_walk_to_stop_role_pair",
    "run_bootstrap_workflow",
    "run_multirole_dagger_workflow",
    "write_role_artifact_manifest",
]
