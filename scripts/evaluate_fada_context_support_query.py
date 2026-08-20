#!/usr/bin/env python3
"""Evaluate Support-Query Context in faulted closed-loop rollouts against healthy trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.distill import load_fada_policy_checkpoint  # noqa: E402
from unilab.algos.torch.fada_context.support_query import SupportQueryContextConfig  # noqa: E402
from unilab.algos.torch.fada_context.support_query_evaluation import (  # noqa: E402
    aggregate_support_query_closed_loop_reports,
    evaluate_online_support_closed_loop,
    evaluate_support_query_closed_loop,
)
from unilab.algos.torch.fada_context.support_query_runtime import (  # noqa: E402
    create_fixed_fault_paired_environments,
    load_support_query_config,
    resolve_repo_path,
    sha256_file,
)
from unilab.algos.torch.fada_context.support_query_training import (  # noqa: E402
    prepare_context_support_query_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "conf" / "fada_context" / "support_query_left_knee_070.yaml",
    )
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--support-source",
        choices=("dataset", "online"),
        default="dataset",
        help="Use held-out dataset Support or collect no-Context Support in the live fault env.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help="Evaluation seed; repeat for multiple seeds (default: 101, 102, 103).",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.seeds is None:
        args.seeds = [101, 102, 103]
    if args.num_envs <= 0 or args.steps <= 0 or not args.seeds:
        raise ValueError("num-envs, steps, and seeds must be positive/non-empty")
    cfg = load_support_query_config(args.config, list(args.overrides), preflight=False)
    healthy_checkpoint = resolve_repo_path(ROOT_DIR, str(cfg.checkpoint_path))
    dataset_path = args.dataset.expanduser().resolve()
    context_checkpoint = args.context_checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    loaded = load_fada_policy_checkpoint(healthy_checkpoint, device=args.device)
    prepared = prepare_context_support_query_artifact(
        loaded.policy,
        SupportQueryContextConfig(
            support_length=int(cfg.collection.support_length),
            context_hidden_dim=int(cfg.context.hidden_dim),
            context_layers=int(cfg.context.num_layers),
            delta_scale=float(cfg.context.delta_scale),
        ),
        source_checkpoint_path=healthy_checkpoint,
        dataset_path=dataset_path,
        context_checkpoint_path=context_checkpoint,
        support_length=int(cfg.collection.support_length),
        query_length=int(cfg.collection.query_length),
        validation_fraction=float(cfg.training.validation_fraction),
        split_seed=int(cfg.seed),
    )
    if float(prepared.metadata.get("fault_strength", -1.0)) != 0.7:
        raise ValueError("Support dataset must use fixed left-knee strength 0.7")
    validation = prepared.validation
    policy = prepared.policy
    required_supports = args.num_envs * len(args.seeds)
    if args.support_source == "dataset" and validation.batch_size < required_supports:
        raise ValueError(
            "not enough held-out validation Supports: "
            f"required={required_supports} available={validation.batch_size}"
        )
    reports: list[dict[str, Any]] = []
    for index, seed in enumerate(args.seeds):
        support_batch = None
        if args.support_source == "dataset":
            begin = index * args.num_envs
            indices = torch.arange(begin, begin + args.num_envs, dtype=torch.int64)
            support_batch = validation.index_select(indices)
        healthy_env, fault_env = create_fixed_fault_paired_environments(
            ROOT_DIR,
            cfg,
            num_envs=args.num_envs,
            seed=seed,
        )
        try:
            if support_batch is None:
                report = evaluate_online_support_closed_loop(
                    healthy_env,
                    fault_env,
                    policy,
                    steps=args.steps,
                    device=args.device,
                )
            else:
                report = evaluate_support_query_closed_loop(
                    healthy_env,
                    fault_env,
                    policy,
                    support_batch.support,
                    support_command=support_batch.support_command,
                    steps=args.steps,
                    device=args.device,
                )
        finally:
            fault_env.close()
            healthy_env.close()
        report["seed"] = int(seed)
        report["support_pair_ids"] = (
            support_batch.pair_id.tolist() if support_batch is not None else []
        )
        reports.append(report)
        print(json.dumps({"event": "seed_completed", **report}, allow_nan=False), flush=True)

    aggregate = aggregate_support_query_closed_loop_reports(reports)
    result = {
        "schema": "unilab_fada_context_support_query_closed_loop_artifact_v2",
        "healthy_checkpoint": str(healthy_checkpoint),
        "healthy_checkpoint_sha256": prepared.source_checkpoint_sha256,
        "context_checkpoint": str(context_checkpoint),
        "context_checkpoint_sha256": sha256_file(context_checkpoint),
        "context_checkpoint_step": prepared.checkpoint_step,
        "dataset": str(dataset_path),
        "dataset_sha256": prepared.dataset_sha256,
        "checkpoint_schema": prepared.checkpoint_schema,
        "method_contract_id": prepared.method_contract_id,
        "checkpoint_identity_binding": prepared.checkpoint_identity_binding,
        "split_contract": prepared.split_contract,
        "train_split_sha256": prepared.train_split_sha256,
        "validation_split_sha256": prepared.validation_split_sha256,
        "evaluation_support_source": (
            "training_run_validation_split"
            if args.support_source == "dataset"
            else "live_fault_no_context_rollout"
        ),
        "reports": reports,
        "aggregate": aggregate,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        json.dumps(
            {"event": "evaluation_completed", "output": str(output_path), "aggregate": aggregate},
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
