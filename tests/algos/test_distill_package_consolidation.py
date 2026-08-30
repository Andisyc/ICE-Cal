from __future__ import annotations

import ast
from pathlib import Path


def test_production_owner_packages_do_not_depend_on_public_aggregate() -> None:
    owner_root = Path("src/unilab/algos/torch/distill")
    owner_packages = (
        "contracts",
        "datasets",
        "collection",
        "learning",
        "workflows",
        "runtime",
        "observability",
        "fada",
    )
    offenders: list[str] = []
    for package in owner_packages:
        for path in (owner_root / package).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == (
                    "unilab.algos.torch.distill"
                ):
                    offenders.append(f"{path}:{node.lineno}")

    assert offenders == []


def test_legacy_dataset_exports_use_dataset_package_owners() -> None:
    from unilab.algos.torch.distill.dataset import (
        DistillationTensorDataset as LegacyDataset,
    )
    from unilab.algos.torch.distill.datasets.dataset import (
        DistillationTensorDataset as OwnerDataset,
    )

    assert LegacyDataset is OwnerDataset


def test_legacy_collection_exports_use_collection_package_owners() -> None:
    from unilab.algos.torch.distill.collection.transition import (
        collect_transition_distillation_dataset_from_env as owner,
    )
    from unilab.algos.torch.distill.collection_transition import (
        collect_transition_distillation_dataset_from_env as legacy,
    )

    assert legacy is owner


def test_legacy_trainer_exports_use_learning_package_owners() -> None:
    from unilab.algos.torch.distill.learning.trainer import (
        BehaviorDistillationTrainer as OwnerTrainer,
    )
    from unilab.algos.torch.distill.trainer import (
        BehaviorDistillationTrainer as LegacyTrainer,
    )

    assert LegacyTrainer is OwnerTrainer


def test_legacy_workflow_exports_use_workflow_package_owners() -> None:
    from unilab.algos.torch.distill.entry_workflow import (
        run_single_entry_workflow as legacy,
    )
    from unilab.algos.torch.distill.workflows.entry_workflow import (
        run_single_entry_workflow as owner,
    )

    assert legacy is owner


def test_legacy_runtime_exports_use_runtime_package_owners() -> None:
    from unilab.algos.torch.distill.g1_persistent_worker import (
        PersistentG1DistillationWorker as LegacyWorker,
    )
    from unilab.algos.torch.distill.runtime.g1_worker import (
        PersistentG1DistillationWorker as OwnerWorker,
    )

    assert LegacyWorker is OwnerWorker


def test_fada_public_model_is_owned_by_fada_package() -> None:
    from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
    from unilab.algos.torch.distill.fada.model import (
        FADAPlannerIDMPolicy as OwnerPolicy,
    )

    assert FADAPlannerIDMPolicy is OwnerPolicy
