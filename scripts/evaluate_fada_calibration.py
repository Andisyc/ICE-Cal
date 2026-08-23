"""Evaluate nominal and calibrated first-action error on a v008 labeled dataset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from unilab.algos.torch.distill import load_fada_policy_checkpoint
from unilab.algos.torch.fada_context.calibration_data import (
    load_calibration_dataset,
    load_fault_axis_catalog,
)
from unilab.algos.torch.fada_context.calibration_evaluation import (
    evaluate_held_out_calibration,
    load_calibration_full_finetune_upper_bound,
)
from unilab.algos.torch.fada_context.calibration_runtime import load_calibrated_policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--axis-catalog",
        type=Path,
        default=Path("conf/fada_context/calibration_axes/gain_delay_offset_v1.yaml"),
    )
    parser.add_argument("--full-finetune-action-chunks", type=Path, required=True)
    args = parser.parse_args()
    healthy = load_fada_policy_checkpoint(args.source_checkpoint, device="cpu").policy
    catalog = load_fault_axis_catalog(args.axis_catalog)
    dataset = load_calibration_dataset(args.dataset, healthy.config, catalog)
    batch, dataset_metadata = dataset.batch, dataset.metadata
    if dataset.axis_spec.axis_count < 2:
        raise ValueError("held-out combination evaluation is not applicable for one active axis")
    source_sha256 = _sha256(args.source_checkpoint)
    if dataset_metadata["source_tracker_sha256"] != source_sha256:
        raise ValueError("dataset source Tracker digest does not match the selected checkpoint")
    calibrated = load_calibrated_policy(
        healthy,
        args.calibration_artifact,
        expected_metadata={
            "source_tracker_sha256": source_sha256,
            "dataset_sha256": _sha256(args.dataset),
            "split_sha256": str(dataset_metadata["split_identity_sha256"]),
        },
        catalog=catalog,
        expected_axis_spec=dataset.axis_spec,
    )
    dataset_sha256 = _sha256(args.dataset)
    upper_bound = load_calibration_full_finetune_upper_bound(
        args.full_finetune_action_chunks,
        expected_metadata={
            "source_tracker_sha256": source_sha256,
            "dataset_sha256": dataset_sha256,
            "split_sha256": str(dataset_metadata["split_identity_sha256"]),
        },
    )
    report = evaluate_held_out_calibration(
        healthy,
        calibrated,
        batch,
        full_finetune=upper_bound,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
