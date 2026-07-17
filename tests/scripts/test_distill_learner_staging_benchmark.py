from __future__ import annotations

import json
from pathlib import Path

import torch
from scripts.deploy.benchmark_distill_learner_staging import (
    _resolve_dataset_path,
    run_staging_benchmark,
)

from unilab.algos.torch.distill.data import (
    build_distillation_dataset,
    save_distillation_dataset,
)


def _dataset(path: Path) -> Path:
    rows = 12
    dataset = build_distillation_dataset(
        torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4),
        torch.arange(rows * 5, dtype=torch.float32).reshape(rows, 5),
        teacher_actions=torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2),
        commands=torch.cat((torch.ones(4, 3), torch.zeros(8, 3))),
        role_labels=("walk_flat",) * 4 + ("stand",) * 8,
        command_intents=("active",) * 4 + ("inactive",) * 8,
        scenario_labels=("walk_flat",) * 4 + ("static_stand",) * 4 + ("walk_to_stop",) * 4,
        transition_ages=torch.tensor([-1] * 8 + [0, 1, 2, 3]),
        command_before=torch.ones(rows, 3),
        command_after=torch.cat((torch.ones(8, 3), torch.zeros(4, 3))),
    )
    save_distillation_dataset(path, dataset)
    return path


def test_resolve_dataset_path_from_workflow_manifest(tmp_path: Path) -> None:
    dataset_path = _dataset(tmp_path / "aggregate.pt")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "dagger_iterations": [
                    {
                        "iteration": 2,
                        "aggregate_dataset_path": str(dataset_path),
                    }
                ]
            }
        )
    )

    assert (
        _resolve_dataset_path(dataset_path=None, run_dir=run_dir, outer_iteration=2)
        == dataset_path.resolve()
    )


def test_staging_benchmark_is_semantically_equal_and_does_not_train(tmp_path: Path) -> None:
    dataset_path = _dataset(tmp_path / "aggregate.pt")

    report = run_staging_benchmark(
        dataset_path=dataset_path,
        device=torch.device("cpu"),
        batch_size=6,
        updates=4,
        warmup_updates=1,
        seed=7,
        balance_key="scenario",
        balanced_labels=("walk_flat", "static_stand", "walk_to_stop"),
        balance_quotas={"walk_flat": 0.5, "static_stand": 0.25, "walk_to_stop": 0.25},
    )

    assert report["training_executed"] is False
    assert report["dataset"]["rows"] == 12
    assert report["semantic_differential"] == {
        "sampled_indices_equal": True,
        "label_counts_equal": True,
        "string_labels_equal": True,
        "tensor_batches_equal": True,
        "pass": True,
    }
    assert set(report["current"]["stages_seconds"]) == {
        "label_pool_build",
        "balanced_sampling",
        "index_h2d",
        "tensor_index_select",
        "python_label_recovery",
    }
    assert report["cached_candidate"]["stages_seconds"]["label_pool_build_once"] >= 0.0


def test_staging_benchmark_resolves_command_intent_labels(tmp_path: Path) -> None:
    report = run_staging_benchmark(
        dataset_path=_dataset(tmp_path / "aggregate.pt"),
        device=torch.device("cpu"),
        batch_size=4,
        updates=1,
        warmup_updates=0,
        seed=3,
        balance_key="command_intent",
        balanced_labels=("active", "inactive"),
        balance_quotas=None,
    )

    assert report["semantic_differential"]["pass"] is True
