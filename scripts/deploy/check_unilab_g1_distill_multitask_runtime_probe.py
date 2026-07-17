#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Runtime probe for the G1 multi-task distillation MoE offline path."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
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
    build_distillation_dataset,
    load_distillation_dataset,
    run_offline_distillation_updates,
    save_distillation_dataset,
)


class RaisingTeacher(torch.nn.Module):
    """Teacher sentinel proving cached targets are used during the update."""

    def __init__(self) -> None:
        super().__init__()
        self.called = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        self.called = True
        raise AssertionError("cached-target probe must not call the teacher")


def _source_dataset(
    *,
    num_samples: int,
    student_obs_dim: int,
    teacher_obs_dim: int,
    action_dim: int,
    student_value: float,
    teacher_value: float,
    action_value: float,
    device: str,
):
    return build_distillation_dataset(
        torch.full(
            (num_samples, student_obs_dim), student_value, dtype=torch.float32, device=device
        ),
        torch.full(
            (num_samples, teacher_obs_dim), teacher_value, dtype=torch.float32, device=device
        ),
        expected_student_obs_dim=student_obs_dim,
        expected_teacher_obs_dim=teacher_obs_dim,
        expected_teacher_action_dim=action_dim,
        teacher_actions=torch.full(
            (num_samples, action_dim), action_value, dtype=torch.float32, device=device
        ),
    )


def _write_source_datasets(
    *,
    work_dir: Path,
    student_obs_dim: int,
    teacher_obs_dim: int,
    action_dim: int,
    device: str,
) -> list[dict[str, str]]:
    sources_dir = work_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("stand", 2, 1.0, 11.0, 0.10),
        ("walk_height", 3, 2.0, 12.0, -0.20),
        ("height", 1, 3.0, 13.0, 0.05),
    ]
    sources: list[dict[str, str]] = []
    for role, num_samples, student_value, teacher_value, action_value in specs:
        path = sources_dir / f"{role}.pt"
        save_distillation_dataset(
            path,
            _source_dataset(
                num_samples=num_samples,
                student_obs_dim=student_obs_dim,
                teacher_obs_dim=teacher_obs_dim,
                action_dim=action_dim,
                student_value=student_value,
                teacher_value=teacher_value,
                action_value=action_value,
                device=device,
            ),
        )
        sources.append({"path": str(path), "role": role})
    return sources


def _cfg(
    *,
    sources: list[dict[str, str]],
    merged_path: Path,
    student_obs_dim: int,
    teacher_obs_dim: int,
    action_dim: int,
    device: str,
):
    return OmegaConf.create(
        {
            "algo": {
                "loss_type": "mse",
                "learning_rate": 0.01,
                "max_grad_norm": 10.0,
                "aux_loss_coef": 0.0,
                "role_loss_coef": 0.25,
                "role_expert_targets": {"stand": 0, "walk_height": 1, "height": 2},
            },
            "student": {
                "obs_dim": student_obs_dim,
                "action_dim": action_dim,
                "num_experts": 3,
                "expert_hidden_dims": [8],
                "router_hidden_dims": [],
                "activation": "elu",
                "squash_action": False,
                "routing_mode": "soft",
                "router_temperature": 1.0,
            },
            "teacher": {
                "obs_dim": teacher_obs_dim,
                "action_dim": action_dim,
            },
            "training": {
                "device": device,
                "multitask_dataset_path": str(merged_path),
                "multitask_sources": sources,
            },
        }
    )


def _build_trainer(
    *,
    student_obs_dim: int,
    action_dim: int,
    device: str,
) -> tuple[BehaviorDistillationTrainer, RaisingTeacher]:
    torch.manual_seed(7)
    student = MoEStudentPolicy(
        obs_dim=student_obs_dim,
        action_dim=action_dim,
        num_experts=3,
        expert_hidden_dims=(8,),
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
        role_expert_targets={"stand": 0, "walk_height": 1, "height": 2},
    )
    return trainer, teacher


def run_check(
    *,
    work_dir: str | Path,
    student_obs_dim: int = 99,
    teacher_obs_dim: int = 99,
    action_dim: int = 29,
    batch_size: int = 3,
    max_updates: int = 2,
    device: str = "cpu",
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    sources = _write_source_datasets(
        work_dir=work_dir,
        student_obs_dim=int(student_obs_dim),
        teacher_obs_dim=int(teacher_obs_dim),
        action_dim=int(action_dim),
        device=str(device),
    )
    merged_path = work_dir / "merged_multitask.pt"
    cfg = _cfg(
        sources=sources,
        merged_path=merged_path,
        student_obs_dim=int(student_obs_dim),
        teacher_obs_dim=int(teacher_obs_dim),
        action_dim=int(action_dim),
        device=str(device),
    )
    assembly = train_distill.run_multitask_dataset_assembly(cfg, dataset_path=merged_path)
    dataset = load_distillation_dataset(
        merged_path,
        expected_student_obs_dim=int(student_obs_dim),
        expected_teacher_obs_dim=int(teacher_obs_dim),
        expected_teacher_action_dim=int(action_dim),
        device=str(device),
    )
    if dataset.role_labels is None:
        raise AssertionError("merged multi-task dataset must preserve role_labels")
    if dataset.teacher_actions is None:
        raise AssertionError("merged multi-task dataset must preserve cached teacher_actions")

    trainer, teacher = _build_trainer(
        student_obs_dim=int(student_obs_dim),
        action_dim=int(action_dim),
        device=str(device),
    )
    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=int(batch_size),
        max_updates=int(max_updates),
    )
    if teacher.called:
        raise AssertionError("cached-target probe unexpectedly called the teacher")
    if result.last_teacher_action_source != "cached":
        raise AssertionError(
            f"expected cached teacher target, got {result.last_teacher_action_source!r}"
        )
    if result.last_role_loss <= 0.0:
        raise AssertionError(f"expected positive role loss, got {result.last_role_loss}")
    if result.last_student_grad_norm <= 0.0:
        raise AssertionError(
            f"expected positive student grad norm, got {result.last_student_grad_norm}"
        )

    role_counts = dict(sorted(Counter(dataset.role_labels).items()))
    return {
        "status": "ok",
        "probe": "g1_distill_multitask_runtime",
        "work_dir": str(work_dir),
        "assembly": assembly,
        "merged_dataset_path": str(merged_path),
        "merged_num_samples": dataset.num_samples,
        "student_obs_dim": dataset.student_obs_dim,
        "teacher_obs_dim": dataset.teacher_obs_dim,
        "teacher_action_dim": dataset.teacher_action_dim,
        "role_counts": role_counts,
        "teacher_action_shape": list(dataset.teacher_actions.shape),
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
            "expert_usage": None
            if result.last_expert_usage is None
            else list(result.last_expert_usage),
            "route_entropy": result.last_route_entropy,
        },
    }


def print_report(payload: dict[str, Any]) -> None:
    print("UniLab G1 distill multi-task MoE runtime probe")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        "[PASS] g1_distill_multitask_runtime: "
        f"samples={payload['merged_num_samples']} "
        f"teacher_action_source={payload['offline_update']['teacher_action_source']} "
        f"role_loss={payload['offline_update']['role_loss']:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/private/tmp/unilab-moe5-runtime-probe")
    parser.add_argument("--student-obs-dim", type=int, default=99)
    parser.add_argument("--teacher-obs-dim", type=int, default=99)
    parser.add_argument("--action-dim", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-updates", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_check(
        work_dir=args.work_dir,
        student_obs_dim=int(args.student_obs_dim),
        teacher_obs_dim=int(args.teacher_obs_dim),
        action_dim=int(args.action_dim),
        batch_size=int(args.batch_size),
        max_updates=int(args.max_updates),
        device=str(args.device),
    )
    print_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
