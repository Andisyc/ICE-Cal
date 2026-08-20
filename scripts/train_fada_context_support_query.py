#!/usr/bin/env python3
"""Train Context Encoder from fixed-fault Support-Query first-action supervision."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.distill import load_fada_policy_checkpoint  # noqa: E402
from unilab.algos.torch.fada_context import (  # noqa: E402
    SupportQueryContextConfig,
    SupportQueryTrainingLoopConfig,
    collect_fixed_fault_support_query,
    load_support_query_config,
    prepare_support_query_training,
    require_fresh_support_query_run_paths,
    resolve_repo_path,
    run_support_query_training,
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


def main() -> int:
    args = _parse_args()
    cfg: DictConfig = load_support_query_config(args.config, list(args.overrides), preflight=False)
    if not bool(cfg.boundary.optimizer_steps_allowed):
        raise ValueError("training requires boundary.optimizer_steps_allowed=true")
    if bool(cfg.boundary.training_started):
        raise ValueError("boundary.training_started is an output fact and must begin false")
    artifact = resolve_repo_path(ROOT_DIR, str(cfg.collection.artifact_path))
    output_dir = resolve_repo_path(ROOT_DIR, str(cfg.training.output_dir))
    require_fresh_support_query_run_paths(artifact, output_dir)
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
        query_length=int(cfg.collection.query_length),
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
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved training config must be a mapping")
    run_support_query_training(
        setup,
        train,
        validation,
        output_dir=output_dir,
        source_checkpoint_sha256=checkpoint_sha,
        dataset_sha256=dataset_sha,
        train_split_sha256=train_split_sha,
        validation_split_sha256=validation_split_sha,
        split_seed=int(cfg.seed),
        resolved_config=resolved,
        config=SupportQueryTrainingLoopConfig(
            steps=int(cfg.training.steps),
            batch_size=int(cfg.training.batch_size),
            log_interval=int(cfg.training.log_interval),
            checkpoint_interval=int(cfg.training.checkpoint_interval),
            gradient_clip_norm=float(cfg.training.gradient_clip_norm),
            minimum_zero_context_mse=float(cfg.training.minimum_zero_context_mse),
        ),
        emit=_emit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
