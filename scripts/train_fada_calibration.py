"""Train the v008/v007 FADA calibrator through serial S1/S2/S3 owners."""

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
    SerialCalibrationConfig,
    run_serial_calibration_training,
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
    parser.add_argument("--scale-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1-steps", type=int, default=100)
    parser.add_argument("--stage2-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    loaded = load_fada_policy_checkpoint(args.source_checkpoint, device="cpu")
    catalog = load_fault_axis_catalog(args.axis_catalog)
    dataset = load_calibration_dataset(args.dataset, loaded.policy.config, catalog)
    batch, dataset_metadata = dataset.batch, dataset.metadata
    source_sha256 = _sha256(args.source_checkpoint)
    if dataset_metadata["source_tracker_sha256"] != source_sha256:
        raise ValueError("dataset source Tracker digest does not match the selected checkpoint")
    dataset_sha256 = _sha256(args.dataset)
    split_sha256 = str(dataset_metadata["split_identity_sha256"])
    result = run_serial_calibration_training(
        loaded.policy,
        batch,
        output_dir=args.output_dir,
        source_tracker_sha256=source_sha256,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        axis_spec=dataset.axis_spec,
        scale_evidence_path=args.scale_evidence,
        config=SerialCalibrationConfig(
            stage1_steps_per_axis=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            learning_rate=args.learning_rate,
        ),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
