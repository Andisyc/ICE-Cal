#!/usr/bin/env python3
"""Separate target, rollout-distribution, and MoE-router failure boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.algos.torch.distill import (  # noqa: E402
    MoEStudentPolicy,
    load_distillation_dataset,
    load_distillation_student_policy,
)


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().float()
    finite = torch.isfinite(value)
    finite_values = value[finite]
    if finite_values.numel() == 0:
        return {
            "shape": list(value.shape),
            "finite_fraction": 0.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "shape": list(value.shape),
        "finite_fraction": float(finite.float().mean().item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "mean": float(finite_values.mean().item()),
        "std": float(finite_values.std(unbiased=False).item()),
    }


def _label_counts(labels: tuple[str, ...] | None) -> dict[str, int] | None:
    if labels is None:
        return None
    return dict(sorted(Counter(labels).items()))


def _group_metrics(
    *,
    mask: torch.Tensor,
    labels: tuple[str, ...],
    commands: torch.Tensor,
    ages: torch.Tensor | None,
    raw_action: torch.Tensor,
    forced_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    raw_selected: torch.Tensor,
    route_probs: torch.Tensor,
    expected_experts: torch.Tensor,
    student_obs: torch.Tensor,
) -> dict[str, Any]:
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return {"count": 0}
    forced_selected = forced_actions[indices, expected_experts[indices]]
    target = teacher_actions[indices]
    raw = raw_action[indices]
    route = raw_selected[indices]
    probs = route_probs[indices]
    obs = student_obs[indices]
    sorted_probs = torch.sort(probs, dim=-1, descending=True).values
    result: dict[str, Any] = {
        "count": int(indices.numel()),
        "role_counts": _label_counts(tuple(labels[int(i)] for i in indices.tolist())),
        "raw_router_argmax_fraction": [
            float((route == expert).float().mean().item())
            for expert in range(forced_actions.shape[1])
        ],
        "raw_router_probability_mean": [
            float(value) for value in probs.mean(dim=0).detach().cpu().tolist()
        ],
        "raw_router_top2_margin_mean": float(
            (sorted_probs[:, 0] - sorted_probs[:, 1]).mean().item()
        ),
        "expected_expert_fraction": [
            float((expected_experts[indices] == expert).float().mean().item())
            for expert in range(forced_actions.shape[1])
        ],
        "raw_action_mse": float((raw - target).square().mean().item()),
        "forced_expected_action_mse": float((forced_selected - target).square().mean().item()),
        "forced_expert_action_abs_max": float(forced_selected.abs().max().item()),
        "teacher_action_abs_max": float(target.abs().max().item()),
        "student_obs": _tensor_stats(obs),
        "commands": _tensor_stats(commands[indices]),
    }
    if ages is not None:
        result["transition_age_min"] = int(ages[indices].min().item())
        result["transition_age_max"] = int(ages[indices].max().item())
    return result


def inspect_dataset(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    device: str,
) -> dict[str, Any]:
    dataset = load_distillation_dataset(dataset_path, device=device)
    loaded = load_distillation_student_policy(checkpoint_path, device=device)
    if not isinstance(loaded.policy, MoEStudentPolicy):
        raise TypeError("RT-9d probe requires a MoEStudentPolicy checkpoint")
    policy = loaded.policy.eval()
    student_obs = dataset.student_obs
    teacher_actions = dataset.teacher_actions
    commands = dataset.commands
    if teacher_actions is None or commands is None:
        raise ValueError("RT-9d probe requires teacher_actions and commands")
    if dataset.role_labels is None or dataset.command_intents is None:
        raise ValueError("RT-9d probe requires role_labels and command_intents")
    if student_obs.shape[1] != policy.obs_dim:
        raise ValueError(
            f"student obs dim mismatch: dataset={student_obs.shape[1]} policy={policy.obs_dim}"
        )

    with torch.no_grad():
        output = policy(student_obs, return_diagnostics=True)
    raw_action = output.action
    forced_actions = output.expert_actions
    raw_selected = torch.argmax(output.route_probs, dim=-1)
    targets = {"active": 0, "inactive": 1}
    expected_experts = torch.as_tensor(
        [targets[intent] for intent in dataset.command_intents],
        dtype=torch.long,
        device=student_obs.device,
    )
    labels = tuple(str(label) for label in dataset.role_labels)
    intents = tuple(str(intent) for intent in dataset.command_intents)
    ages = dataset.transition_ages

    groups: dict[str, torch.Tensor] = {
        "all": torch.ones(dataset.num_samples, dtype=torch.bool, device=student_obs.device),
    }
    for role in sorted(set(labels)):
        groups[f"role:{role}"] = torch.as_tensor(
            [label == role for label in labels], dtype=torch.bool, device=student_obs.device
        )
    for intent in sorted(set(intents)):
        groups[f"intent:{intent}"] = torch.as_tensor(
            [label == intent for label in intents], dtype=torch.bool, device=student_obs.device
        )
    if ages is not None:
        for age in sorted(set(int(value) for value in ages.detach().cpu().tolist())):
            groups[f"transition_age:{age}"] = ages == age

    metrics = {
        name: _group_metrics(
            mask=mask,
            labels=labels,
            commands=commands,
            ages=ages,
            raw_action=raw_action,
            forced_actions=forced_actions,
            teacher_actions=teacher_actions,
            raw_selected=raw_selected,
            route_probs=output.route_probs,
            expected_experts=expected_experts,
            student_obs=student_obs,
        )
        for name, mask in groups.items()
    }
    return {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "num_samples": dataset.num_samples,
        "dataset_metadata": dataset.metadata,
        "student_obs": _tensor_stats(student_obs),
        "teacher_actions": _tensor_stats(teacher_actions),
        "role_counts": _label_counts(labels),
        "command_intent_counts": _label_counts(intents),
        "deployment_targets": targets,
        "raw_router_argmax_counts": {
            str(expert): int((raw_selected == expert).sum().item())
            for expert in range(policy.num_experts)
        },
        "raw_router_route_entropy_mean": float(
            (
                -(
                    output.route_probs.clamp_min(1e-12) * output.route_probs.clamp_min(1e-12).log()
                ).sum(dim=-1)
            )
            .mean()
            .item()
        ),
        "groups": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = {}
    for item in args.dataset:
        if "=" not in item:
            raise ValueError(f"--dataset must be NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        reports[name] = inspect_dataset(
            dataset_path=Path(raw_path),
            checkpoint_path=Path(args.student_checkpoint),
            device=args.device,
        )
    print("UniLab G1 distill RT-9d policy-quality probe")
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
