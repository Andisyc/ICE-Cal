"""Trace Stage 1 Direction optimization without publishing a Stage artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from unilab.algos.torch.distill import load_fada_policy_checkpoint
from unilab.algos.torch.fada_context.calibration_data import (
    load_calibration_dataset,
    load_fault_axis_catalog,
)
from unilab.algos.torch.fada_context.calibration_training import (
    CalibrationStageIdentity,
    DirectionDiagnosticConfig,
    DirectionDiagnosticPoint,
    diagnose_direction_stage_training,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--axis-catalog",
        type=Path,
        default=Path("conf/fada_context/calibration_axes/gain_delay_offset_v1.yaml"),
    )
    parser.add_argument(
        "--checkpoint-step",
        action="append",
        type=int,
        dest="checkpoint_steps",
        help="Optimization step to record; repeat for an ordered checkpoint schedule.",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    loaded = load_fada_policy_checkpoint(args.source_checkpoint, device="cpu")
    catalog = load_fault_axis_catalog(args.axis_catalog)
    dataset = load_calibration_dataset(args.dataset, loaded.policy.config, catalog)
    source_sha256 = _sha256(args.source_checkpoint)
    dataset_sha256 = _sha256(args.dataset)
    if dataset.metadata["source_tracker_sha256"] != source_sha256:
        raise ValueError("dataset source Tracker digest does not match the selected checkpoint")
    identity = CalibrationStageIdentity(
        source_tracker_sha256=source_sha256,
        dataset_sha256=dataset_sha256,
        split_sha256=str(dataset.metadata["split_identity_sha256"]),
        axis_spec=dataset.axis_spec,
    )
    config = DirectionDiagnosticConfig(
        checkpoint_steps=(
            DirectionDiagnosticConfig().checkpoint_steps
            if args.checkpoint_steps is None
            else tuple(args.checkpoint_steps)
        ),
        learning_rate=args.learning_rate,
    )
    points = diagnose_direction_stage_training(
        loaded.policy,
        dataset.batch,
        identity,
        config,
    )
    payload = {
        "source_tracker_sha256": source_sha256,
        "dataset_sha256": dataset_sha256,
        "split_sha256": identity.split_sha256,
        "axis_spec": dataset.axis_spec.to_payload(),
        "learning_rate": config.learning_rate,
        "checkpoint_steps": list(config.checkpoint_steps),
        "points": [
            {
                **asdict(point),
                "axis_name": dataset.axis_spec.names[point.axis_index],
            }
            for point in points
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
