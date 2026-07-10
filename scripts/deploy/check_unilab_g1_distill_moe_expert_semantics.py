#!/usr/bin/env python3
"""Offline MoE expert-route diagnostics for generic G1 distillation checkpoints."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.distill import (  # noqa: E402
    MoEStudentPolicy,
    diagnose_moe_expert_routes,
    load_distillation_dataset,
    load_distillation_student_policy,
    moe_diagnostics_to_dict,
)


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level, name, detail))


def _metadata_role_labels(metadata: dict[str, Any], *, num_samples: int) -> list[str] | None:
    raw = metadata.get("role_labels")
    if not isinstance(raw, list):
        return None
    labels = [str(label) for label in raw]
    if len(labels) != int(num_samples):
        raise ValueError(
            "dataset metadata role_labels length mismatch: "
            f"labels={len(labels)} samples={int(num_samples)}"
        )
    return labels


def _report_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    report = dict(metadata)
    raw_role_labels = report.pop("role_labels", None)
    if isinstance(raw_role_labels, list):
        counts = Counter(str(label) for label in raw_role_labels)
        report["role_label_counts"] = {
            role: int(counts[role])
            for role in sorted(counts)
        }
        report["role_label_count_total"] = int(sum(counts.values()))
    elif raw_role_labels is not None:
        report["role_labels_type"] = type(raw_role_labels).__name__
    return report


def _action_imitation_summary(
    policy: MoEStudentPolicy,
    student_obs: torch.Tensor,
    teacher_actions: torch.Tensor | None,
    *,
    role_labels: list[str] | None,
    hard_routing: bool,
) -> dict[str, Any] | None:
    if teacher_actions is None:
        return None
    if teacher_actions.ndim != 2:
        raise ValueError(f"teacher_actions must be rank-2, got shape {tuple(teacher_actions.shape)}")
    if teacher_actions.shape[0] != student_obs.shape[0]:
        raise ValueError(
            "teacher_actions batch size must match student_obs: "
            f"teacher_actions={teacher_actions.shape[0]} student_obs={student_obs.shape[0]}"
        )
    with torch.no_grad():
        output = policy(
            student_obs,
            hard_routing=bool(hard_routing),
            return_diagnostics=True,
        )
    student_actions = output.action.detach()
    target_actions = teacher_actions.to(
        device=student_actions.device,
        dtype=student_actions.dtype,
    )
    if target_actions.shape != student_actions.shape:
        raise ValueError(
            "teacher_actions shape must match student actions: "
            f"teacher_actions={tuple(target_actions.shape)} student_actions={tuple(student_actions.shape)}"
        )

    sq_error = (student_actions - target_actions).square().mean(dim=-1)
    abs_error = (student_actions - target_actions).abs().mean(dim=-1)
    labels = role_labels if role_labels is not None else ["all"] * int(student_obs.shape[0])
    if len(labels) != int(student_obs.shape[0]):
        raise ValueError(
            "role_labels length must match student_obs batch size: "
            f"labels={len(labels)} batch={int(student_obs.shape[0])}"
        )

    def summarize(mask: torch.Tensor) -> dict[str, Any]:
        selected_sq = sq_error.index_select(0, mask)
        selected_abs = abs_error.index_select(0, mask)
        selected_student = student_actions.index_select(0, mask)
        selected_target = target_actions.index_select(0, mask)
        return {
            "count": int(mask.numel()),
            "mse": float(selected_sq.mean().detach().cpu().item()),
            "mae": float(selected_abs.mean().detach().cpu().item()),
            "student_action_abs_max": float(selected_student.abs().max().detach().cpu().item()),
            "teacher_action_abs_max": float(selected_target.abs().max().detach().cpu().item()),
        }

    by_role: dict[str, Any] = {}
    for role in sorted(set(labels)):
        idx = torch.tensor(
            [i for i, label in enumerate(labels) if label == role],
            dtype=torch.long,
            device=student_actions.device,
        )
        by_role[str(role)] = summarize(idx)
    overall_idx = torch.arange(student_obs.shape[0], dtype=torch.long, device=student_actions.device)
    return {
        "overall": summarize(overall_idx),
        "by_role": by_role,
    }


def run_check(
    *,
    task: str,
    dataset_path: str | Path,
    student_checkpoint: str | Path,
    device: str = "cpu",
    hard_routing: bool = False,
    collapse_fraction: float = 0.90,
    fail_on_collapse: bool = False,
) -> tuple[list[Check], dict[str, Any]]:
    loaded = load_distillation_student_policy(student_checkpoint, device=device)
    policy = loaded.policy
    checks: list[Check] = []
    details: dict[str, Any] = {
        "moe_expert/task_owner": str(task),
        "moe_expert/student_checkpoint": str(student_checkpoint),
        "moe_expert/dataset_path": str(dataset_path),
        "moe_expert/device": str(device),
        "moe_expert/hard_routing": bool(hard_routing),
        "moe_expert/collapse_fraction": float(collapse_fraction),
        "moe_expert/student_model_type": loaded.distill_runtime_cfg.get("student_model_type"),
        "moe_expert/student_obs_dim": loaded.obs_dim,
        "moe_expert/student_action_dim": loaded.action_dim,
        "moe_expert/agent_steps": loaded.agent_steps,
    }
    if not isinstance(policy, MoEStudentPolicy):
        _add(checks, "FAIL", "moe_expert/student_model_type", type(policy).__name__)
        return checks, details

    expected_teacher_obs_dim = loaded.distill_runtime_cfg.get("teacher_obs_dim")
    dataset = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=int(loaded.obs_dim),
        expected_teacher_obs_dim=(
            None if expected_teacher_obs_dim is None else int(expected_teacher_obs_dim)
        ),
        device=device,
    )
    role_labels = (
        None
        if dataset.role_labels is None
        else [str(label) for label in dataset.role_labels]
    )
    if role_labels is None:
        role_labels = _metadata_role_labels(dataset.metadata, num_samples=dataset.num_samples)
    diagnostics = diagnose_moe_expert_routes(
        policy,
        dataset.student_obs,
        role_labels=role_labels,
        hard_routing=bool(hard_routing),
        collapse_fraction=float(collapse_fraction),
    )
    payload = moe_diagnostics_to_dict(diagnostics)
    action_imitation = _action_imitation_summary(
        policy,
        dataset.student_obs,
        dataset.teacher_actions,
        role_labels=role_labels,
        hard_routing=bool(hard_routing),
    )
    details.update(
        {
            "moe_expert/dataset_num_samples": dataset.num_samples,
            "moe_expert/dataset_student_obs_dim": dataset.student_obs_dim,
            "moe_expert/dataset_teacher_obs_dim": dataset.teacher_obs_dim,
            "moe_expert/dataset_metadata": _report_metadata(dict(dataset.metadata)),
            "moe_expert/role_labels_present": diagnostics.role_labels_present,
            "moe_expert/diagnostics": payload,
            "moe_expert/action_imitation": action_imitation,
        }
    )

    _add(checks, "PASS", "moe_expert/student_model_type", "moe")
    _add(checks, "PASS", "moe_expert/dataset_dim_guard", f"{dataset.student_obs_dim}")
    if diagnostics.role_labels_present:
        _add(
            checks,
            "PASS",
            "moe_expert/role_labels",
            ",".join(summary.role for summary in diagnostics.by_role),
        )
    else:
        _add(
            checks,
            "WARN",
            "moe_expert/role_labels",
            "dataset has no role_labels metadata; reporting overall collapse/entropy only",
        )

    collapse_level = "FAIL" if fail_on_collapse and diagnostics.overall.collapse_detected else "WARN"
    if diagnostics.overall.collapse_detected:
        _add(
            checks,
            collapse_level,
            "moe_expert/collapse_guard",
            f"dominant={diagnostics.overall.dominant_expert}, max_fraction={diagnostics.overall.max_fraction:.6f}",
        )
    else:
        _add(
            checks,
            "PASS",
            "moe_expert/collapse_guard",
            f"max_fraction={diagnostics.overall.max_fraction:.6f}",
        )
    _add(
        checks,
        "PASS",
        "moe_expert/route_entropy",
        f"{diagnostics.overall.mean_entropy:.6f}",
    )
    if action_imitation is None:
        _add(
            checks,
            "WARN",
            "moe_expert/action_imitation",
            "dataset has no cached teacher_actions; cannot compute student-vs-teacher action error",
        )
    else:
        _add(
            checks,
            "PASS",
            "moe_expert/action_imitation",
            f"overall_mse={action_imitation['overall']['mse']:.6f}",
        )
    return checks, details


def print_report(checks: list[Check], details: dict[str, Any]) -> None:
    print("UniLab G1 distill MoE expert semantics diagnostics")
    for key, value in details.items():
        print(f"{key}: {value}")
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="g1_stand_still/mujoco")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hard-routing", action="store_true")
    parser.add_argument("--collapse-fraction", type=float, default=0.90)
    parser.add_argument("--fail-on-collapse", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks, details = run_check(
        task=args.task,
        dataset_path=args.dataset_path,
        student_checkpoint=args.student_checkpoint,
        device=args.device,
        hard_routing=bool(args.hard_routing),
        collapse_fraction=float(args.collapse_fraction),
        fail_on_collapse=bool(args.fail_on_collapse),
    )
    print_report(checks, details)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
