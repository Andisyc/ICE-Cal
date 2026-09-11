from __future__ import annotations

import argparse
import json

from unilab.algos.torch.distill.fada.real_target_data import prepare_fada_real_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RoboJuDo G1 trajectories into a FADA real-target artifact."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--command", type=float, nargs=3, required=True, metavar=("VX", "VY", "YAW"))
    parser.add_argument("--target-domain-id", default="g1_hang_book_real")
    parser.add_argument("--condition-label", default="shoulder_hanging_book")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-phase-action-mse", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_fada_real_target(
        input_dir=args.input_dir,
        source_checkpoint_path=args.source_checkpoint,
        output_path=args.output,
        command=args.command,
        target_domain_id=args.target_domain_id,
        condition_label=args.condition_label,
        device=args.device,
        max_phase_action_mse=args.max_phase_action_mse,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
