"""Train and seal only the v007/v006 Stage 1 Direction Bank."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from unilab.algos.torch.distill import load_fada_policy_checkpoint
from unilab.algos.torch.fada_context.calibration_data import (
    load_calibration_dataset,
    load_fault_axis_catalog,
)
from unilab.algos.torch.fada_context.calibration_training import (
    CalibrationStageIdentity,
    DirectionStageConfig,
    run_direction_stage_training,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--axis-catalog",
        type=Path,
        default=Path("conf/fada_context/calibration_axes/gain_delay_offset_v1.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    loaded = load_fada_policy_checkpoint(args.source_checkpoint, device="cpu")
    batch, metadata = load_calibration_dataset(args.dataset, loaded.policy.config)
    source_sha256 = _sha256(args.source_checkpoint)
    if metadata["source_tracker_sha256"] != source_sha256:
        raise ValueError("dataset source Tracker digest does not match the selected checkpoint")
    catalog = load_fault_axis_catalog(args.axis_catalog)
    if metadata["axis_catalog_version"] != catalog.version:
        raise ValueError("dataset and configured fault-axis catalog differ")
    identity = CalibrationStageIdentity(
        source_tracker_sha256=source_sha256,
        dataset_sha256=_sha256(args.dataset),
        split_sha256=str(metadata["split_identity_sha256"]),
        axis_catalog_version=str(metadata["axis_catalog_version"]),
    )
    result = run_direction_stage_training(
        loaded.policy,
        batch,
        args.output,
        identity,
        DirectionStageConfig(
            steps_per_axis=args.stage1_steps,
            learning_rate=args.learning_rate,
        ),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
