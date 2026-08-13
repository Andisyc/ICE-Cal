#!/usr/bin/env python3
"""Train Context Encoder from fixed-fault Support-Query first-action supervision."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.distill import load_fada_policy_checkpoint  # noqa: E402
from unilab.algos.torch.fada_context import (  # noqa: E402
    SupportQueryBatch,
    SupportQueryContextConfig,
    collect_fixed_fault_support_query,
    context_first_action_loss,
    evaluate_context_action_mse,
    load_support_query_config,
    parameter_snapshot,
    parameters_equal,
    prepare_support_query_training,
    resolve_repo_path,
    save_context_support_query_checkpoint,
    save_support_query_dataset,
    sha256_file,
    split_support_query_by_rollout,
    support_query_split_identity_sha256,
)
from unilab.training import apply_training_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "conf" / "fada_context" / "support_query_left_knee_070.yaml",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, allow_nan=False), flush=True)


def _require_fresh_run_paths(artifact: Path, output_dir: Path) -> None:
    if artifact.exists():
        raise FileExistsError(f"Support-Query dataset artifact already exists: {artifact}")
    if output_dir.exists():
        raise FileExistsError(f"Context training output directory already exists: {output_dir}")


def _sample(batch: SupportQueryBatch, batch_size: int) -> SupportQueryBatch:
    indices = torch.randint(
        batch.batch_size,
        (min(batch_size, batch.batch_size),),
        device=batch.pair_id.device,
    )
    return batch.index_select(indices)


def main() -> int:
    args = _parse_args()
    cfg: DictConfig = load_support_query_config(
        args.config, list(args.overrides), preflight=False
    )
    if not bool(cfg.boundary.optimizer_steps_allowed):
        raise ValueError("training requires boundary.optimizer_steps_allowed=true")
    if bool(cfg.boundary.training_started):
        raise ValueError("boundary.training_started is an output fact and must begin false")
    artifact = resolve_repo_path(ROOT_DIR, str(cfg.collection.artifact_path))
    output_dir = resolve_repo_path(ROOT_DIR, str(cfg.training.output_dir))
    _require_fresh_run_paths(artifact, output_dir)
    apply_training_seed(int(cfg.seed), torch_runtime=True, cuda=True)
    checkpoint = resolve_repo_path(ROOT_DIR, str(cfg.checkpoint_path))
    checkpoint_sha = sha256_file(checkpoint)
    loaded = load_fada_policy_checkpoint(checkpoint, device=str(cfg.device))
    collected = collect_fixed_fault_support_query(ROOT_DIR, cfg, loaded.policy)
    output_dir.mkdir(parents=True, exist_ok=False)
    save_support_query_dataset(
        artifact,
        collected.batch,
        loaded.policy.config,
        support_length=int(cfg.collection.support_length),
        metadata={
            "source_checkpoint_sha256": checkpoint_sha,
            "task_config": str(cfg.task_config),
            "fault_joint": "left_knee",
            "fault_strength": 0.7,
            "command": [0.4, 0.0, 0.0],
            "seed": int(cfg.seed),
            "training_started": False,
        },
    )
    dataset_sha = sha256_file(artifact)
    train_cpu, validation_cpu = split_support_query_by_rollout(
        collected.batch,
        validation_fraction=float(cfg.training.validation_fraction),
        seed=int(cfg.seed),
    )
    train = train_cpu.to(str(cfg.device))
    validation = validation_cpu.to(str(cfg.device))
    train_split_sha = support_query_split_identity_sha256(train_cpu)
    validation_split_sha = support_query_split_identity_sha256(validation_cpu)
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
    baseline_train = evaluate_context_action_mse(setup.policy, train)
    baseline_validation = evaluate_context_action_mse(setup.policy, validation)
    if baseline_train <= float(cfg.training.minimum_zero_context_mse):
        raise ValueError(
            "zero-Context training MSE is too small to establish supervision signal: "
            f"observed={baseline_train} threshold={cfg.training.minimum_zero_context_mse}"
        )
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved training config must be a mapping")
    _emit(
        "training_started",
        train_pairs=train.batch_size,
        validation_pairs=validation.batch_size,
        baseline_train_mse=baseline_train,
        baseline_validation_mse=baseline_validation,
    )
    latest_train = baseline_train
    latest_validation = baseline_validation
    best_validation = baseline_validation
    best_step = 0
    save_context_support_query_checkpoint(
        output_dir / "best.pt",
        setup,
        source_checkpoint_sha256=checkpoint_sha,
        dataset_sha256=dataset_sha,
        train_split_sha256=train_split_sha,
        validation_split_sha256=validation_split_sha,
        step=0,
        split_seed=int(cfg.seed),
        metrics={
            "baseline_train_mse": baseline_train,
            "baseline_validation_mse": baseline_validation,
            "full_train_mse": baseline_train,
            "validation_mse": baseline_validation,
        },
        resolved_config=resolved,
    )
    for step in range(1, int(cfg.training.steps) + 1):
        setup.optimizer.zero_grad(set_to_none=True)
        loss = context_first_action_loss(
            setup.policy, _sample(train, int(cfg.training.batch_size))
        )
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite Context loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            setup.policy.context_encoder.parameters(),
            float(cfg.training.gradient_clip_norm),
        )
        if not torch.isfinite(grad_norm):
            raise ValueError(f"non-finite Context gradient at step {step}")
        setup.optimizer.step()
        latest_train = float(loss.detach())
        if step == 1 or step % int(cfg.training.log_interval) == 0:
            latest_validation = evaluate_context_action_mse(setup.policy, validation)
            _emit(
                "training_step",
                step=step,
                train_first_action_mse=latest_train,
                validation_first_action_mse=latest_validation,
                context_grad_norm=float(grad_norm),
            )
            if latest_validation < best_validation:
                best_validation = latest_validation
                best_step = step
                save_context_support_query_checkpoint(
                    output_dir / "best.pt",
                    setup,
                    source_checkpoint_sha256=checkpoint_sha,
                    dataset_sha256=dataset_sha,
                    train_split_sha256=train_split_sha,
                    validation_split_sha256=validation_split_sha,
                    step=step,
                    split_seed=int(cfg.seed),
                    metrics={
                        "baseline_train_mse": baseline_train,
                        "baseline_validation_mse": baseline_validation,
                        "minibatch_train_mse": latest_train,
                        "validation_mse": latest_validation,
                    },
                    resolved_config=resolved,
                )
        if step % int(cfg.training.checkpoint_interval) == 0:
            checkpoint_train = evaluate_context_action_mse(setup.policy, train)
            checkpoint_validation = evaluate_context_action_mse(setup.policy, validation)
            save_context_support_query_checkpoint(
                output_dir / f"context_{step}.pt",
                setup,
                source_checkpoint_sha256=checkpoint_sha,
                dataset_sha256=dataset_sha,
                train_split_sha256=train_split_sha,
                validation_split_sha256=validation_split_sha,
                step=step,
                split_seed=int(cfg.seed),
                metrics={
                    "baseline_train_mse": baseline_train,
                    "baseline_validation_mse": baseline_validation,
                    "full_train_mse": checkpoint_train,
                    "validation_mse": checkpoint_validation,
                },
                resolved_config=resolved,
            )
    if not parameters_equal(setup.policy.planner, planner_before):
        raise RuntimeError("Planner changed during Context training")
    if not parameters_equal(setup.policy.idm, idm_before):
        raise RuntimeError("IDM changed during Context training")
    final_train = evaluate_context_action_mse(setup.policy, train)
    final_validation = evaluate_context_action_mse(setup.policy, validation)
    final_path = save_context_support_query_checkpoint(
        output_dir / "final.pt",
        setup,
        source_checkpoint_sha256=checkpoint_sha,
        dataset_sha256=dataset_sha,
        train_split_sha256=train_split_sha,
        validation_split_sha256=validation_split_sha,
        step=int(cfg.training.steps),
        split_seed=int(cfg.seed),
        metrics={
            "baseline_train_mse": baseline_train,
            "baseline_validation_mse": baseline_validation,
            "full_train_mse": final_train,
            "validation_mse": final_validation,
            "best_validation_mse": best_validation,
            "best_step": float(best_step),
        },
        resolved_config=resolved,
    )
    _emit("training_completed", checkpoint=str(final_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
