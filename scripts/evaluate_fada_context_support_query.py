#!/usr/bin/env python3
"""Evaluate Support-Query Context in faulted closed-loop rollouts against healthy trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.distill import load_fada_policy_checkpoint  # noqa: E402
from unilab.algos.torch.fada_context.support_query import (  # noqa: E402
    SupportQueryBatch,
    SupportQueryContextConfig,
)
from unilab.algos.torch.fada_context.support_query_data import (  # noqa: E402
    load_support_query_dataset,
)
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
    PreparedSupportQueryTraining,
    prepare_support_query_training,
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


def _split_identity_sha256(batch: SupportQueryBatch) -> str:
    identity = (
        torch.stack((batch.pair_id, batch.support_rollout_id, batch.query_rollout_id), dim=1)
        .detach()
        .cpu()
    )
    identity = identity.index_select(0, torch.argsort(identity[:, 0])).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(identity.shape)).encode("ascii"))
    digest.update(identity.numpy().tobytes())
    return digest.hexdigest()


def _legacy_pair_split(
    batch: SupportQueryBatch,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[SupportQueryBatch, SupportQueryBatch]:
    validation_count = max(1, int(round(batch.batch_size * validation_fraction)))
    if validation_count >= batch.batch_size:
        raise ValueError("Support-Query dataset must contain at least two pairs")
    order = torch.randperm(batch.batch_size, generator=torch.Generator().manual_seed(seed))
    return (
        batch.index_select(order[validation_count:]),
        batch.index_select(order[:validation_count]),
    )


def _rollout_group_split(
    batch: SupportQueryBatch,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[SupportQueryBatch, SupportQueryBatch]:
    rollout_groups = (
        torch.stack((batch.support_rollout_id, batch.query_rollout_id), dim=1).detach().cpu()
    )
    unique_groups, inverse = torch.unique(rollout_groups, dim=0, return_inverse=True)
    group_count = int(unique_groups.shape[0])
    validation_groups = max(1, int(round(group_count * validation_fraction)))
    if group_count < 2 or validation_groups >= group_count:
        raise ValueError("Support-Query rollout split requires at least two groups")
    order = torch.randperm(group_count, generator=torch.Generator().manual_seed(seed))
    validation_mask = torch.isin(inverse, order[:validation_groups])
    train_indices = torch.nonzero(~validation_mask, as_tuple=False).flatten()
    validation_indices = torch.nonzero(validation_mask, as_tuple=False).flatten()
    return batch.index_select(train_indices), batch.index_select(validation_indices)


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Context checkpoint must be a mapping")
    return payload


def _load_context_for_evaluation(
    payload: Mapping[str, Any],
    setup: PreparedSupportQueryTraining,
    *,
    healthy_sha: str,
    dataset_sha: str,
    train_split_sha: str,
    validation_split_sha: str,
    split_seed: int,
) -> str:
    schema = int(payload.get("schema_version", -1))
    if schema not in (1, 2, 3):
        raise ValueError(f"unsupported Context checkpoint schema: {schema}")
    if payload.get("fada_architecture") != asdict(setup.policy.config):
        raise ValueError("Context checkpoint FADA architecture mismatch")
    if payload.get("context_config") != asdict(setup.policy.context_encoder.context_config):
        raise ValueError("Context checkpoint architecture mismatch")
    if payload.get("source_checkpoint_sha256") != healthy_sha:
        raise ValueError("Context checkpoint healthy source identity mismatch")
    if int(payload.get("split_seed", -1)) != split_seed:
        raise ValueError("Context checkpoint split seed mismatch")
    if schema in (2, 3):
        expected = {
            "dataset_sha256": dataset_sha,
            "train_split_sha256": train_split_sha,
            "validation_split_sha256": validation_split_sha,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f"Context checkpoint {name} mismatch")
        identity_binding = "healthy_dataset_and_splits"
    else:
        identity_binding = "legacy_v1_source_checkpoint_only"
    state = payload.get("context_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Context checkpoint is missing context_state_dict")
    setup.policy.context_encoder.load_state_dict(state, strict=True)
    setup.policy.eval()
    return identity_binding


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
    healthy_sha = sha256_file(healthy_checkpoint)
    dataset_sha = sha256_file(dataset_path)
    loaded = load_fada_policy_checkpoint(healthy_checkpoint, device=args.device)
    dataset, metadata = load_support_query_dataset(
        dataset_path,
        loaded.policy.config,
        support_length=int(cfg.collection.support_length),
        query_length=int(cfg.collection.query_length),
        allow_legacy_single_anchor=True,
    )
    if metadata.get("source_checkpoint_sha256") != healthy_sha:
        raise ValueError("Support dataset healthy checkpoint identity mismatch")
    if float(metadata.get("fault_strength", -1.0)) != 0.7:
        raise ValueError("Support dataset must use fixed left-knee strength 0.7")

    checkpoint_payload = _checkpoint_payload(context_checkpoint)
    checkpoint_schema = int(checkpoint_payload.get("schema_version", -1))
    if checkpoint_schema == 1:
        train, validation = _legacy_pair_split(
            dataset,
            validation_fraction=float(cfg.training.validation_fraction),
            seed=int(cfg.seed),
        )
        split_contract = "legacy_pair_split"
    elif checkpoint_schema in (2, 3):
        train, validation = _rollout_group_split(
            dataset,
            validation_fraction=float(cfg.training.validation_fraction),
            seed=int(cfg.seed),
        )
        split_contract = "rollout_group_split"
    else:
        raise ValueError(f"unsupported Context checkpoint schema: {checkpoint_schema}")
    required_supports = args.num_envs * len(args.seeds)
    if args.support_source == "dataset" and validation.batch_size < required_supports:
        raise ValueError(
            "not enough held-out validation Supports: "
            f"required={required_supports} available={validation.batch_size}"
        )
    setup = prepare_support_query_training(
        loaded.policy,
        SupportQueryContextConfig(
            support_length=int(cfg.collection.support_length),
            context_hidden_dim=int(cfg.context.hidden_dim),
            context_layers=int(cfg.context.num_layers),
            delta_scale=float(cfg.context.delta_scale),
        ),
        learning_rate=float(cfg.context.learning_rate),
    )
    train_split_sha = _split_identity_sha256(train)
    validation_split_sha = _split_identity_sha256(validation)
    checkpoint_identity_binding = _load_context_for_evaluation(
        checkpoint_payload,
        setup,
        healthy_sha=healthy_sha,
        dataset_sha=dataset_sha,
        train_split_sha=train_split_sha,
        validation_split_sha=validation_split_sha,
        split_seed=int(cfg.seed),
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
                    setup.policy,
                    steps=args.steps,
                    device=args.device,
                )
            else:
                report = evaluate_support_query_closed_loop(
                    healthy_env,
                    fault_env,
                    setup.policy,
                    support_batch.support,
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
        "schema": "unilab_fada_context_support_query_closed_loop_artifact_v1",
        "healthy_checkpoint": str(healthy_checkpoint),
        "healthy_checkpoint_sha256": healthy_sha,
        "context_checkpoint": str(context_checkpoint),
        "context_checkpoint_sha256": sha256_file(context_checkpoint),
        "context_checkpoint_step": int(checkpoint_payload["step"]),
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "checkpoint_schema": checkpoint_schema,
        "checkpoint_identity_binding": checkpoint_identity_binding,
        "split_contract": split_contract,
        "train_split_sha256": train_split_sha,
        "validation_split_sha256": validation_split_sha,
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
