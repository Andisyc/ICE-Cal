"""Measure whether Stage 1 rows require one shared executed-action latent direction."""

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
    DirectionGeometryAxisReport,
    DirectionGeometryConfig,
    DirectionGeometrySplitReport,
    diagnose_direction_geometry,
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
    config = DirectionGeometryConfig()
    reports = diagnose_direction_geometry(
        loaded.policy,
        dataset.batch,
        identity,
        config,
    )
    if any(
        not isinstance(report, DirectionGeometryAxisReport)
        or not isinstance(report.training, DirectionGeometrySplitReport)
        or not isinstance(report.validation, DirectionGeometrySplitReport)
        for report in reports
    ):
        raise TypeError("direction geometry owner returned an invalid report")
    payload = {
        "supervision_scope": "executed_first_action",
        "solver": "linear_decoder_minimum_norm",
        "source_tracker_sha256": source_sha256,
        "dataset_sha256": dataset_sha256,
        "split_sha256": identity.split_sha256,
        "axis_spec": dataset.axis_spec.to_payload(),
        "reports": [
            {
                **asdict(report),
                "axis_name": dataset.axis_spec.names[report.axis_index],
            }
            for report in reports
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
