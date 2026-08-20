#!/usr/bin/env python3
"""Validate fixed-0.7 Support-Query Context training without taking an optimizer step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.distill import load_fada_policy_checkpoint  # noqa: E402
from unilab.algos.torch.fada_context import (  # noqa: E402
    SupportQueryContextConfig,
    collect_fixed_fault_support_query,
    load_support_query_config,
    load_support_query_dataset,
    preflight_context_support_query_artifact,
    resolve_repo_path,
    run_support_query_preflight,
    save_support_query_dataset,
    sha256_file,
)
from unilab.training import (  # noqa: E402
    apply_training_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "conf" / "fada_context" / "support_query_left_knee_070.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--artifact-admission", action="store_true")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--context-checkpoint", type=Path, default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _context_config(cfg: Any) -> SupportQueryContextConfig:
    return SupportQueryContextConfig(
        support_length=int(cfg.collection.support_length),
        context_hidden_dim=int(cfg.context.hidden_dim),
        context_layers=int(cfg.context.num_layers),
        delta_scale=float(cfg.context.delta_scale),
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_support_query_config(args.config, list(args.overrides), preflight=True)
    apply_training_seed(int(cfg.seed), torch_runtime=True, cuda=False)
    checkpoint = resolve_repo_path(ROOT_DIR, str(cfg.checkpoint_path))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"healthy FADA checkpoint not found: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    loaded = load_fada_policy_checkpoint(checkpoint, device=str(cfg.device))
    artifact_admission = bool(getattr(args, "artifact_admission", False))
    dataset_arg = getattr(args, "dataset", None)
    context_checkpoint_arg = getattr(args, "context_checkpoint", None)
    if artifact_admission:
        if dataset_arg is None or context_checkpoint_arg is None:
            raise ValueError("artifact admission requires --dataset and --context-checkpoint")
        dataset_path = resolve_repo_path(ROOT_DIR, str(dataset_arg))
        context_checkpoint_path = resolve_repo_path(ROOT_DIR, str(context_checkpoint_arg))
        admitted = preflight_context_support_query_artifact(
            loaded.policy,
            _context_config(cfg),
            source_checkpoint_path=checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=context_checkpoint_path,
            support_length=int(cfg.collection.support_length),
            query_length=int(cfg.collection.query_length),
            validation_fraction=float(cfg.training.validation_fraction),
            split_seed=int(cfg.seed),
            map_location=str(cfg.device),
        )
        return {
            "schema": "unilab_fada_context_support_query_artifact_admission_v1",
            "mode": "artifact_admission",
            "status": "passed",
            "method_contract_id": admitted.method_contract_id,
            "checkpoint_schema": admitted.checkpoint_schema,
            "checkpoint_step": admitted.checkpoint_step,
            "training_started": False,
            "optimizer_steps": 0,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": checkpoint_sha,
            "dataset": str(dataset_path),
            "context_checkpoint": str(context_checkpoint_path),
            "query_provenance": {
                "pair_ids": list(admitted.pair_ids),
                "support_rollout_ids": list(admitted.support_rollout_ids),
                "query_rollout_ids": list(admitted.query_rollout_ids),
                "current_history_conditioned": True,
            },
            "collection": {"window_count": admitted.window_count},
            "tensors": {"delta_z": list(admitted.delta_z_shape)},
        }
    if dataset_arg is not None or context_checkpoint_arg is not None:
        raise ValueError("--dataset and --context-checkpoint require --artifact-admission")
    collected = collect_fixed_fault_support_query(ROOT_DIR, cfg, loaded.policy)
    artifact = resolve_repo_path(ROOT_DIR, str(cfg.collection.artifact_path))
    metadata = {
        "source_checkpoint_sha256": checkpoint_sha,
        "task_config": str(cfg.task_config),
        "fault_joint": "left_knee",
        "fault_strength": 0.7,
        "command": [0.4, 0.0, 0.0],
        "seed": int(cfg.seed),
        "training_started": False,
    }
    save_support_query_dataset(
        artifact,
        collected.batch,
        loaded.policy.config,
        support_length=int(cfg.collection.support_length),
        query_length=int(cfg.collection.query_length),
        metadata=metadata,
    )
    batch, round_trip_metadata = load_support_query_dataset(
        artifact,
        loaded.policy.config,
        support_length=int(cfg.collection.support_length),
        query_length=int(cfg.collection.query_length),
        map_location=str(cfg.device),
    )
    result = run_support_query_preflight(
        loaded.policy,
        batch,
        _context_config(cfg),
        learning_rate=float(cfg.context.learning_rate),
        minimum_zero_context_mse=float(cfg.training.minimum_zero_context_mse),
    )
    return {
        "schema": "unilab_fada_context_support_query_preflight_v3",
        "mode": "collection_preflight",
        "method_contract_id": result.method_contract_id,
        "status": "passed",
        "training_started": False,
        "optimizer_steps": 0,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "task_config": str(cfg.task_config),
        "fault": {"joint": "left_knee", "index": 3, "strength": 0.7},
        "query_provenance": {
            "pair_ids": batch.pair_id.tolist(),
            "support_rollout_ids": batch.support_rollout_id.tolist(),
            "query_rollout_ids": batch.query_rollout_id.tolist(),
            "current_history_conditioned": True,
        },
        "collection": {
            "artifact": str(artifact),
            "accepted_pairs": collected.accepted_pairs,
            "rejected_pairs": collected.rejected_pairs,
            "reset_pairs": collected.reset_pairs,
            "support_length": int(cfg.collection.support_length),
            "query_length": int(cfg.collection.query_length),
            "window_count": batch.query.window_count,
            "valid_window_count": int(batch.query.valid_window_mask.sum()),
            "anchor_min": int(batch.query.window_anchor[batch.query.valid_window_mask].min()),
            "anchor_max": int(batch.query.window_anchor[batch.query.valid_window_mask].max()),
            "round_trip_fault_strength": round_trip_metadata["fault_strength"],
        },
        "tensors": {
            "support_target_future": list(batch.support.target_future.shape),
            "support_realized_state": list(batch.support.realized_state.shape),
            "support_executed_action": list(batch.support.executed_action.shape),
            "query_observation_history": list(batch.query.observation_history.shape),
            "query_action_history": list(batch.query.action_history.shape),
            "query_planner_intent": list(batch.query.planner_intent.shape),
            "query_realized_future": list(batch.query.realized_future.shape),
            "query_executed_action": list(batch.query.executed_action.shape),
            "query_window_anchor": list(batch.query.window_anchor.shape),
            "query_valid_window_mask": list(batch.query.valid_window_mask.shape),
            "delta_z": list(result.delta_z_shape),
        },
        "supervision": {
            "zero_context_first_action_mse": result.zero_context_first_action_mse,
            "minimum_required_mse": result.minimum_required_mse,
            "context_grad_norm": result.context_grad_norm,
            "planner_frozen": result.planner_frozen,
            "idm_frozen": result.idm_frozen,
        },
    }


def main() -> int:
    args = _parse_args()
    payload = run_preflight(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print("Training boundary: ready, not started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
