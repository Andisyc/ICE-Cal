from __future__ import annotations

import torch


def test_dataset_owners_preserve_legacy_value_and_callable_surface() -> None:
    from unilab.algos.torch.distill import data
    from unilab.algos.torch.distill.dataset import (
        DistillationTensorDataset,
        build_distillation_dataset,
    )
    from unilab.algos.torch.distill.dataset_io import (
        load_distillation_dataset,
        save_distillation_dataset,
    )
    from unilab.algos.torch.distill.dataset_merge import build_multitask_distillation_dataset

    dataset = build_distillation_dataset(
        torch.zeros((2, 3)),
        torch.ones((2, 4)),
        expected_student_obs_dim=3,
        expected_teacher_obs_dim=4,
    )

    assert isinstance(dataset, DistillationTensorDataset)
    assert dataset.num_samples == 2
    assert data.build_distillation_dataset is build_distillation_dataset
    assert data.build_multitask_distillation_dataset is build_multitask_distillation_dataset
    assert data.save_distillation_dataset is save_distillation_dataset
    assert data.load_distillation_dataset is load_distillation_dataset


def test_collector_owners_preserve_legacy_callable_surface() -> None:
    from unilab.algos.torch.distill import collector
    from unilab.algos.torch.distill.collection_common import project_student_obs
    from unilab.algos.torch.distill.collection_standard import (
        collect_distillation_dataset_from_env,
    )
    from unilab.algos.torch.distill.collection_transition import (
        collect_transition_distillation_dataset_from_env,
    )

    assert collector.project_student_obs is project_student_obs
    assert (
        collector.collect_distillation_dataset_from_env
        is collect_distillation_dataset_from_env
    )
    assert (
        collector.collect_transition_distillation_dataset_from_env
        is collect_transition_distillation_dataset_from_env
    )


def test_standard_collector_has_an_explicit_transaction_owner() -> None:
    from unilab.algos.torch.distill.collection.standard_transaction import (
        StandardCollectionTransaction,
    )

    assert StandardCollectionTransaction.__module__.endswith(
        "collection.standard_transaction"
    )


def test_workflow_owners_preserve_legacy_callable_surface() -> None:
    from unilab.algos.torch.distill import workflow
    from unilab.algos.torch.distill.workflow_artifacts import (
        preflight_role_artifact,
        write_role_artifact_manifest,
    )
    from unilab.algos.torch.distill.workflow_bootstrap import run_bootstrap_workflow
    from unilab.algos.torch.distill.workflow_contracts import RoleArtifactSpec
    from unilab.algos.torch.distill.workflow_dagger import run_multirole_dagger_workflow

    assert workflow.RoleArtifactSpec is RoleArtifactSpec
    assert workflow.preflight_role_artifact is preflight_role_artifact
    assert workflow.write_role_artifact_manifest is write_role_artifact_manifest
    assert workflow.run_bootstrap_workflow is run_bootstrap_workflow
    assert workflow.run_multirole_dagger_workflow is run_multirole_dagger_workflow
