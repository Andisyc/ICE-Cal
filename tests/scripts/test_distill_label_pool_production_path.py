from __future__ import annotations

from pathlib import Path

import torch
from scripts.deploy.check_distill_label_pool_production_path import run_production_path_probe

from unilab.algos.torch.distill.data import (
    build_distillation_dataset,
    save_distillation_dataset,
)


def test_production_path_uses_one_pool_and_preserves_rng(tmp_path: Path) -> None:
    path = tmp_path / "aggregate.pt"
    rows = 12
    dataset = build_distillation_dataset(
        torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4),
        torch.empty(rows, 0),
        teacher_actions=torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2),
        scenario_labels=("walk_flat",) * 4 + ("static_stand",) * 4 + ("walk_to_stop",) * 4,
        transition_ages=torch.tensor([-1] * 8 + [0, 1, 2, 3]),
        command_before=torch.ones(rows, 3),
        command_after=torch.cat((torch.ones(8, 3), torch.zeros(4, 3))),
    )
    save_distillation_dataset(path, dataset)

    report = run_production_path_probe(
        dataset_path=path,
        device=torch.device("cpu"),
        batch_size=6,
        updates=5,
        seed=19,
        balance_key="scenario",
        balanced_labels=("walk_flat", "static_stand", "walk_to_stop"),
        balance_quotas={"walk_flat": 0.5, "static_stand": 0.25, "walk_to_stop": 0.25},
        shuffle=True,
    )

    assert report["training_executed"] is False
    assert report["production_cache_build_count"] == 1
    assert report["production_update_count"] == 5
    assert report["sampled_indices_digest_equal"] is True
    assert report["final_rng_state_equal"] is True
    assert report["pass"] is True
