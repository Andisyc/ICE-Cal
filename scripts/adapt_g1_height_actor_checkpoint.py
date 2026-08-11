"""Materialize a legacy 98-D G1 SAC actor as an explicit 99-D actor checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unilab.algos.torch.offpolicy.checkpoint_adapter import (
    materialize_g1_height_actor_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Legacy 98-D SAC checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New actor-only 99-D checkpoint")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing adapter output"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_g1_height_actor_checkpoint(
        args.input,
        args.output,
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "parent_checkpoint_path": str(result.parent_checkpoint_path),
                "parent_checkpoint_sha256": result.parent_checkpoint_sha256,
                "output_checkpoint_path": str(result.output_checkpoint_path),
                "output_checkpoint_sha256": result.output_checkpoint_sha256,
                "metadata_path": str(result.metadata_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
