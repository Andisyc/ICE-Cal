"""Seal raw Stage 3 scale scans into a v008/v007 identity-bound artifact."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch

from unilab.algos.torch.distill import load_fada_policy_checkpoint
from unilab.algos.torch.fada_context.calibration_data import (
    load_calibration_dataset,
    load_fault_axis_catalog,
)
from unilab.algos.torch.fada_context.calibration_training import (
    CalibrationScaleEvidence,
    save_calibration_scale_evidence,
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
    parser.add_argument("--raw-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    loaded = load_fada_policy_checkpoint(args.source_checkpoint, device="cpu")
    catalog = load_fault_axis_catalog(args.axis_catalog)
    dataset = load_calibration_dataset(args.dataset, loaded.policy.config, catalog)
    dataset_metadata = dataset.metadata
    source_sha256 = _sha256(args.source_checkpoint)
    if dataset_metadata["source_tracker_sha256"] != source_sha256:
        raise ValueError("dataset source Tracker digest does not match the selected checkpoint")
    raw = torch.load(args.raw_evidence, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise ValueError("raw scale evidence must be a mapping")
    tensor_fields = (
        "coefficient_scan_grid",
        "readings",
        "candidate_scales",
        "action_errors",
    )
    if any(not isinstance(raw.get(name), torch.Tensor) for name in tensor_fields):
        raise ValueError("raw scale evidence is missing tensor fields")
    evidence = CalibrationScaleEvidence(
        coefficient_scan_grid=raw["coefficient_scan_grid"],
        readings=raw["readings"],
        candidate_scales=raw["candidate_scales"],
        action_errors=raw["action_errors"],
        axis_spec=dataset.axis_spec,
        metadata={
            "source_tracker_sha256": source_sha256,
            "dataset_sha256": _sha256(args.dataset),
            "split_sha256": str(dataset_metadata["split_identity_sha256"]),
        },
    )
    print(save_calibration_scale_evidence(args.output, evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
