#!/usr/bin/env python3
"""Run bounded physical acceptance for one trained G1StandHeight checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.training.g1_stand_height_acceptance import run_acceptance  # noqa: E402


def print_report(report: Mapping[str, Any]) -> None:
    print("UniLab G1StandHeight teacher acceptance")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    for check in report["checks"]:
        print(f"[{check['level']}] {check['name']}: {check['detail']}")
    print(f"RESULT: {report['verdict']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-target-height", type=float, default=0.754)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--steps", type=int, default=800, help="Scored steps after warmup.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-height-mae", type=float, default=0.05)
    parser.add_argument("--min-double-support-fraction", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_acceptance(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        expected_sha256=args.expected_sha256,
        expected_target_height=args.expected_target_height,
        num_envs=args.num_envs,
        warmup_steps=args.warmup_steps,
        evaluation_steps=args.steps,
        seed=args.seed,
        device=args.device,
        max_height_mae=args.max_height_mae,
        min_double_support_fraction=args.min_double_support_fraction,
    )
    print_report(report)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
