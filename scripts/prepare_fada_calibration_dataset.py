"""Seal recorded v007 fault rollouts into the active calibration dataset schema."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping, cast

import torch

from unilab.algos.torch.distill import load_fada_policy_checkpoint
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
    raw = torch.load(args.raw_rollouts, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise ValueError("raw calibration rollout artifact must be a mapping")
    catalog = load_fault_axis_catalog(args.axis_catalog)
    batch = prepare_calibration_rollout_batch(
        cast(Mapping[str, Any], raw),
        policy.config,
        catalog,
    )
    save_calibration_dataset(
        args.output,
        batch,
        policy.config,
        metadata={
            "source_tracker_sha256": _sha256(args.source_checkpoint),
            "axis_catalog_version": catalog.version,
            "split_identity_sha256": calibration_split_identity_sha256(batch),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
