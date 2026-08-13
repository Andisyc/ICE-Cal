#!/usr/bin/env python3
"""Validate fixed-0.7 Support-Query Context training without taking an optimizer step."""

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
from unilab.algos.torch.fada_context import (  # noqa: E402
    SupportQueryContextConfig,
    collect_fixed_fault_support_query,
    context_first_action_loss,
    load_support_query_config,
    load_support_query_dataset,
    parameter_snapshot,
    parameters_equal,
    prepare_support_query_training,
    resolve_repo_path,
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
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_support_query_config(args.config, list(args.overrides), preflight=True)
    apply_training_seed(int(cfg.seed), torch_runtime=True, cuda=False)
    checkpoint = resolve_repo_path(ROOT_DIR, str(cfg.checkpoint_path))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"healthy FADA checkpoint not found: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    loaded = load_fada_policy_checkpoint(checkpoint, device=str(cfg.device))
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
    planner_before = parameter_snapshot(setup.policy.planner)
    idm_before = parameter_snapshot(setup.policy.idm)
    loss = context_first_action_loss(setup.policy, batch)
    zero_context_mse = float(loss.detach())
    if not torch.isfinite(loss):
        raise ValueError("zero-Context Query action loss must be finite")
    if zero_context_mse <= float(cfg.training.minimum_zero_context_mse):
        raise ValueError(
            "zero-Context Query action loss is too small to establish supervision signal: "
            f"observed={zero_context_mse} threshold={cfg.training.minimum_zero_context_mse}"
        )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in setup.policy.context_encoder.parameters()
        if parameter.grad is not None
    ]
    context_grad_norm = float(
        torch.sqrt(sum(torch.sum(gradient.detach() ** 2) for gradient in gradients))
    ) if gradients else 0.0
    if not torch.isfinite(torch.tensor(context_grad_norm)) or context_grad_norm <= 0.0:
        raise ValueError("Context gradient norm must be finite and positive")
    if not parameters_equal(setup.policy.planner, planner_before):
        raise RuntimeError("Planner changed during preflight backward")
    if not parameters_equal(setup.policy.idm, idm_before):
        raise RuntimeError("IDM changed during preflight backward")
    if any(parameter.grad is not None for parameter in setup.policy.planner.parameters()):
        raise RuntimeError("Planner received gradients during Context backward")
    if any(parameter.grad is not None for parameter in setup.policy.idm.parameters()):
        raise RuntimeError("IDM received gradients during Context backward")
    return {
        "schema": "unilab_fada_context_support_query_preflight_v1",
        "status": "passed",
        "training_started": False,
        "optimizer_steps": 0,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "task_config": str(cfg.task_config),
        "fault": {"joint": "left_knee", "index": 3, "strength": 0.7},
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
            "delta_z": [batch.batch_size, loaded.policy.config.hidden_dim],
        },
        "supervision": {
            "zero_context_first_action_mse": zero_context_mse,
            "minimum_required_mse": float(cfg.training.minimum_zero_context_mse),
            "context_grad_norm": context_grad_norm,
            "planner_frozen": True,
            "idm_frozen": True,
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
