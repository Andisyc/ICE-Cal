#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Collect walking/standing teacher-policy sources and run a bounded MoE update."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

import train_distill  # noqa: E402
from unilab.algos.torch.distill import (  # noqa: E402
    BehaviorDistillationTrainer,
    MoEStudentPolicy,
    inspect_sac_teacher_checkpoint,
    load_distillation_dataset,
    run_offline_distillation_updates,
)

DEFAULT_WALKING_CHECKPOINT = (
    ROOT_DIR / "logs/fast_sac/G1WalkFlat/2026-07-09_02-48-58_mujoco/model_5000.pt"
)
DEFAULT_STANDING_CHECKPOINT = (
    ROOT_DIR / "logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt"
)
EXPECTED_COMMAND_FILTERS = {"walk_flat": "active", "stand": "inactive"}


class RaisingTeacher(torch.nn.Module):
    """Teacher sentinel proving the merged MoE update uses cached targets."""

    def __init__(self) -> None:
        super().__init__()
        self.called = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        self.called = True
        raise AssertionError("dual-teacher MoE probe must use cached teacher_actions")


def _compose_collect_cfg(
    *,
    task: str,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    num_samples: int,
    num_envs: int,
    device: str,
):
    overrides = [
        f"task={task}",
        f"teacher.checkpoint_path={Path(checkpoint_path)}",
        "teacher.load_run=-1",
        "teacher.checkpoint=-1",
        f"training.collect_dataset_path={Path(dataset_path)}",
        f"training.collect_num_samples={int(num_samples)}",
        f"training.collect_num_envs={int(num_envs)}",
        "training.collect_action_mode=teacher_policy",
        f"training.device={device}",
    ]
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "distill"), version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def _multitask_cfg(*, sources: list[dict[str, str]], merged_path: Path, device: str):
    return OmegaConf.create(
        {
            "algo": {
                "loss_type": "mse",
                "learning_rate": 0.01,
                "max_grad_norm": 10.0,
                "aux_loss_coef": 0.0,
                "role_loss_coef": 0.25,
                "role_expert_targets": {"walk_flat": 0, "stand": 1},
            },
            "student": {
                "obs_dim": 98,
                "action_dim": 29,
                "num_experts": 3,
                "expert_hidden_dims": [32],
                "router_hidden_dims": [],
                "activation": "elu",
                "squash_action": False,
                "routing_mode": "soft",
                "router_temperature": 1.0,
            },
            "teacher": {
                "obs_dim": 98,
                "action_dim": 29,
            },
            "training": {
                "device": device,
                "multitask_dataset_path": str(merged_path),
                "multitask_sources": sources,
            },
        }
    )


def _build_trainer(*, device: str) -> tuple[BehaviorDistillationTrainer, RaisingTeacher]:
    torch.manual_seed(17)
    student = MoEStudentPolicy(
        obs_dim=98,
        action_dim=29,
        num_experts=3,
        expert_hidden_dims=(32,),
        router_hidden_dims=(),
        squash_action=False,
        routing_mode="soft",
    ).to(device)
    teacher = RaisingTeacher().to(device)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=torch.optim.Adam(student.parameters(), lr=0.01),
        loss_type="mse",
        max_grad_norm=10.0,
        aux_loss_coef=0.0,
        role_loss_coef=0.25,
        role_expert_targets={"walk_flat": 0, "stand": 1},
    )
    return trainer, teacher


def _checkpoint_info(path: str | Path) -> dict[str, Any]:
    info = inspect_sac_teacher_checkpoint(path)
    if info.actor_input_dim != 98:
        raise AssertionError(f"expected 98-D teacher checkpoint, got {info.actor_input_dim}")
    return {
        "checkpoint_path": str(Path(path)),
        "actor_input_dim": info.actor_input_dim,
        "first_weight_key": info.first_weight_key,
    }


def _load_source_dataset(path: Path) -> dict[str, Any]:
    dataset = load_distillation_dataset(
        path,
        expected_student_obs_dim=98,
        expected_teacher_obs_dim=98,
        expected_teacher_action_dim=29,
    )
    if dataset.teacher_actions is None:
        raise AssertionError(f"{path} did not persist cached teacher_actions")
    return {
        "path": str(path),
        "num_samples": dataset.num_samples,
        "student_obs_dim": dataset.student_obs_dim,
        "teacher_obs_dim": dataset.teacher_obs_dim,
        "teacher_action_dim": dataset.teacher_action_dim,
        "teacher_actions_shape": list(dataset.teacher_actions.shape),
        "action_abs_max": float(dataset.metadata.get("action_abs_max", 0.0)),
        "env_steps": int(dataset.metadata.get("env_steps", 0)),
        "metadata": dict(dataset.metadata),
    }


def _assert_intent_filter_contract(
    *,
    role: str,
    collection_probe: dict[str, Any],
    source_dataset: dict[str, Any],
) -> dict[str, Any]:
    expected_filter = EXPECTED_COMMAND_FILTERS[role]
    collection_filter = str(collection_probe.get("collect_command_sample_filter"))
    metadata = dict(source_dataset["metadata"])
    dataset_filter = str(metadata.get("command_sample_filter"))
    if collection_filter != expected_filter:
        raise AssertionError(
            f"{role} collection filter {collection_filter!r} != expected {expected_filter!r}"
        )
    if dataset_filter != expected_filter:
        raise AssertionError(
            f"{role} dataset filter {dataset_filter!r} != expected {expected_filter!r}"
        )

    seen = int(metadata.get("command_seen_samples", 0))
    selected = int(metadata.get("command_selected_samples", 0))
    num_samples = int(source_dataset["num_samples"])
    if selected < num_samples or seen < selected:
        raise AssertionError(
            f"{role} command filter counts invalid: "
            f"seen={seen} selected={selected} num_samples={num_samples}"
        )
    return {
        "expected_filter": expected_filter,
        "collection_filter": collection_filter,
        "dataset_filter": dataset_filter,
        "command_seen_samples": seen,
        "command_selected_samples": selected,
    }


def run_check(
    *,
    walking_checkpoint: str | Path = DEFAULT_WALKING_CHECKPOINT,
    standing_checkpoint: str | Path = DEFAULT_STANDING_CHECKPOINT,
    work_dir: str | Path = "/private/tmp/unilab-moe6-dual-teacher",
    num_samples: int = 4,
    num_envs: int = 1,
    batch_size: int = 4,
    max_updates: int = 2,
    device: str = "cpu",
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    walking_path = Path(walking_checkpoint)
    standing_path = Path(standing_checkpoint)
    walking_dataset_path = work_dir / "walk_flat_teacher_policy.pt"
    standing_dataset_path = work_dir / "stand_teacher_policy.pt"
    merged_path = work_dir / "walk_stand_merged.pt"

    checkpoint_contracts = {
        "walk_flat": _checkpoint_info(walking_path),
        "stand": _checkpoint_info(standing_path),
    }

    collect_cfgs = {
        "walk_flat": _compose_collect_cfg(
            task="g1_walk_flat/mujoco",
            checkpoint_path=walking_path,
            dataset_path=walking_dataset_path,
            num_samples=int(num_samples),
            num_envs=int(num_envs),
            device=str(device),
        ),
        "stand": _compose_collect_cfg(
            task="g1_stand_still/mujoco",
            checkpoint_path=standing_path,
            dataset_path=standing_dataset_path,
            num_samples=int(num_samples),
            num_envs=int(num_envs),
            device=str(device),
        ),
    }
    collection = {
        "walk_flat": train_distill.run_collect_dataset(
            collect_cfgs["walk_flat"],
            dataset_path=walking_dataset_path,
        ),
        "stand": train_distill.run_collect_dataset(
            collect_cfgs["stand"],
            dataset_path=standing_dataset_path,
        ),
    }
    sources = [
        {"path": str(walking_dataset_path), "role": "walk_flat"},
        {"path": str(standing_dataset_path), "role": "stand"},
    ]
    source_datasets = {
        "walk_flat": _load_source_dataset(walking_dataset_path),
        "stand": _load_source_dataset(standing_dataset_path),
    }
    command_filter_contracts = {
        role: _assert_intent_filter_contract(
            role=role,
            collection_probe=collection[role],
            source_dataset=source_datasets[role],
        )
        for role in ("walk_flat", "stand")
    }

    assembly = train_distill.run_multitask_dataset_assembly(
        _multitask_cfg(sources=sources, merged_path=merged_path, device=str(device)),
        dataset_path=merged_path,
    )
    merged = load_distillation_dataset(
        merged_path,
        expected_student_obs_dim=98,
        expected_teacher_obs_dim=98,
        expected_teacher_action_dim=29,
        device=str(device),
    )
    if merged.role_labels is None:
        raise AssertionError("merged dataset lost role_labels")
    if merged.teacher_actions is None:
        raise AssertionError("merged dataset lost cached teacher_actions")
    role_counts = dict(sorted(Counter(merged.role_labels).items()))
    expected_role_counts = {"stand": int(num_samples), "walk_flat": int(num_samples)}
    if role_counts != expected_role_counts:
        raise AssertionError(f"unexpected role counts: {role_counts} != {expected_role_counts}")

    trainer, teacher = _build_trainer(device=str(device))
    result = run_offline_distillation_updates(
        trainer,
        merged,
        batch_size=int(batch_size),
        max_updates=int(max_updates),
    )
    if teacher.called:
        raise AssertionError("merged offline update unexpectedly called a teacher")
    if result.last_teacher_action_source != "cached":
        raise AssertionError(f"expected cached target, got {result.last_teacher_action_source!r}")

    return {
        "status": "ok",
        "probe": "g1_distill_dual_teacher_moe",
        "work_dir": str(work_dir),
        "checkpoint_contracts": checkpoint_contracts,
        "collection": collection,
        "source_datasets": source_datasets,
        "command_filter_contracts": command_filter_contracts,
        "assembly": assembly,
        "merged_dataset_path": str(merged_path),
        "merged_num_samples": merged.num_samples,
        "role_counts": role_counts,
        "teacher_action_shape": list(merged.teacher_actions.shape),
        "offline_update": {
            "update_count": result.update_count,
            "samples_seen": result.samples_seen,
            "teacher_action_source": result.last_teacher_action_source,
            "teacher_action_requires_grad": result.teacher_action_requires_grad,
            "behavior_loss": result.last_behavior_loss,
            "role_loss": result.last_role_loss,
            "role_target_count": result.last_role_target_count,
            "student_grad_norm": result.last_student_grad_norm,
            "student_action_shape": list(result.student_action_shape),
            "teacher_action_shape": list(result.teacher_action_shape),
            "expert_usage": None if result.last_expert_usage is None else list(result.last_expert_usage),
            "route_entropy": result.last_route_entropy,
        },
    }


def print_report(payload: dict[str, Any]) -> None:
    print("UniLab G1 distill dual-teacher MoE probe")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        "[PASS] g1_distill_dual_teacher_moe: "
        f"samples={payload['merged_num_samples']} "
        f"roles={payload['role_counts']} "
        f"filters={payload['command_filter_contracts']} "
        f"teacher_action_source={payload['offline_update']['teacher_action_source']} "
        f"role_loss={payload['offline_update']['role_loss']:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking-checkpoint", default=str(DEFAULT_WALKING_CHECKPOINT))
    parser.add_argument("--standing-checkpoint", default=str(DEFAULT_STANDING_CHECKPOINT))
    parser.add_argument("--work-dir", default="/private/tmp/unilab-moe6-dual-teacher")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_check(
        walking_checkpoint=args.walking_checkpoint,
        standing_checkpoint=args.standing_checkpoint,
        work_dir=args.work_dir,
        num_samples=int(args.num_samples),
        num_envs=int(args.num_envs),
        batch_size=int(args.batch_size),
        max_updates=int(args.max_updates),
        device=str(args.device),
    )
    print_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
