"""Seal recorded v007 fault rollouts into the active calibration dataset schema."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from unilab.algos.torch.distill import load_fada_policy_checkpoint
from unilab.algos.torch.fada_context.calibration_collection import (
    load_gain_calibration_raw_rollouts,
)
from unilab.algos.torch.fada_context.calibration_data import (
    calibration_split_identity_sha256,
    load_fault_axis_catalog,
    prepare_calibration_rollout_batch,
    save_calibration_dataset,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--raw-rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--axis-catalog",
        type=Path,
        default=Path("conf/fada_context/calibration_axes/gain_delay_offset_v1.yaml"),
    )
    args = parser.parse_args()
    policy = load_fada_policy_checkpoint(args.source_checkpoint, device="cpu").policy
    source_sha256 = _sha256(args.source_checkpoint)
    raw = load_gain_calibration_raw_rollouts(
        args.raw_rollouts,
        expected_source_sha256=source_sha256,
        expected_architecture=policy.config,
    )
    catalog = load_fault_axis_catalog(args.axis_catalog)
    batch = prepare_calibration_rollout_batch(
        raw,
        policy.config,
        catalog,
    )
    save_calibration_dataset(
        args.output,
        batch,
        policy.config,
        metadata={
            "source_tracker_sha256": source_sha256,
            "axis_catalog_version": catalog.version,
            "split_identity_sha256": calibration_split_identity_sha256(batch),
            "raw_protocol_sha256": raw["metadata"]["protocol_sha256"],
            "resolved_task_backend_sha256": raw["metadata"]["resolved_task_backend_sha256"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
