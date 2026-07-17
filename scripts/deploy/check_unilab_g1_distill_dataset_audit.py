#!/usr/bin/env python3
"""Audit offline G1 distillation dataset payloads without launching simulation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"dataset payload must be a dict, got {type(payload).__name__}")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return value


def _summarize_value(value: Any, *, max_items: int = 4) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _summarize_value(item, max_items=max_items)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        values = list(value)
        if len(values) <= max_items:
            return [_summarize_value(item, max_items=max_items) for item in values]
        return {
            "count": len(values),
            "head": [_summarize_value(item, max_items=max_items) for item in values[:max_items]],
            "tail": [_summarize_value(item, max_items=max_items) for item in values[-max_items:]],
        }
    return _jsonable(value)


def _shape(value: Any) -> list[int] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return [int(dim) for dim in value.shape]


def _sample_rows(tensor: torch.Tensor, *, max_rows: int) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1)
    row_count = int(tensor.shape[0])
    if row_count <= max_rows:
        return tensor
    indices = torch.linspace(0, row_count - 1, steps=max_rows).round().long()
    return tensor.index_select(0, indices)


def _tensor_stats(tensor: Any, *, max_rows: int) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        return {
            "present": False,
            "type": type(tensor).__name__,
        }
    sampled = _sample_rows(tensor.detach().cpu(), max_rows=max_rows)
    flat = sampled.reshape(-1)
    finite = torch.isfinite(flat)
    finite_count = int(finite.sum().item())
    total_count = int(flat.numel())
    summary: dict[str, Any] = {
        "present": True,
        "shape": _shape(tensor),
        "dtype": str(tensor.dtype),
        "sampled_rows": int(sampled.shape[0]) if sampled.ndim > 0 else 1,
        "total_rows": int(tensor.shape[0]) if tensor.ndim > 0 else 1,
        "finite": finite_count == total_count,
        "finite_count": finite_count,
        "total_count": total_count,
    }
    if finite_count == 0:
        return summary
    finite_values = flat[finite].float()
    summary.update(
        {
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "abs_max": float(finite_values.abs().max().item()),
        }
    )
    return summary


def _labels(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        return None
    return tuple(str(item) for item in value)


def _counts(labels: tuple[str, ...] | None) -> dict[str, int] | None:
    if labels is None:
        return None
    counts = Counter(labels)
    return {label: int(counts[label]) for label in sorted(counts)}


def _paired_counts(
    left: tuple[str, ...] | None, right: tuple[str, ...] | None
) -> dict[str, int] | None:
    if left is None or right is None:
        return None
    counts = Counter(
        f"{left_label}|{right_label}" for left_label, right_label in zip(left, right, strict=True)
    )
    return {label: int(counts[label]) for label in sorted(counts)}


def _active_mask_from_commands(
    commands: torch.Tensor,
    *,
    xy_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    xy_norm = torch.linalg.norm(commands[:, :2], dim=1)
    yaw_abs = commands[:, 2].abs()
    return (xy_norm > float(xy_threshold)) | (yaw_abs > float(yaw_threshold))


def _intent_from_role(role: str) -> str | None:
    normalized = role.lower()
    if "stand" in normalized:
        return "inactive"
    if "walk" in normalized:
        return "active"
    return None


def _label_indices(labels: tuple[str, ...], label: str) -> torch.Tensor:
    selected = [index for index, value in enumerate(labels) if value == label]
    return torch.tensor(selected, dtype=torch.long)


def _group_stats(
    *,
    labels: tuple[str, ...] | None,
    teacher_actions: torch.Tensor | None,
    commands: torch.Tensor | None,
    max_rows: int,
) -> dict[str, Any] | None:
    if labels is None:
        return None
    result: dict[str, Any] = {}
    for label in sorted(set(labels)):
        indices = _label_indices(labels, label)
        entry: dict[str, Any] = {
            "count": int(indices.numel()),
        }
        if teacher_actions is not None:
            entry["teacher_actions"] = _tensor_stats(
                teacher_actions.index_select(0, indices),
                max_rows=max_rows,
            )
        if commands is not None:
            entry["commands"] = _tensor_stats(
                commands.index_select(0, indices),
                max_rows=max_rows,
            )
        result[label] = entry
    return result


def _source_summary(metadata: Mapping[str, Any], *, num_samples: int) -> dict[str, Any]:
    source_paths = metadata.get("source_paths")
    source_roles = metadata.get("source_roles")
    source_sample_counts = metadata.get("source_sample_counts")
    source_metadata = metadata.get("source_metadata")
    summary: dict[str, Any] = {
        "source": metadata.get("source"),
        "source_count": metadata.get("source_count"),
        "source_paths": _summarize_value(source_paths),
        "source_roles": _summarize_value(source_roles),
        "source_sample_counts": _summarize_value(source_sample_counts),
    }
    if isinstance(source_sample_counts, list | tuple):
        summary["source_sample_count_total"] = int(
            sum(int(value) for value in source_sample_counts)
        )
        summary["source_sample_count_matches_num_samples"] = summary[
            "source_sample_count_total"
        ] == int(num_samples)
    if isinstance(source_metadata, list | tuple):
        compact: list[dict[str, Any]] = []
        keep_keys = (
            "source",
            "task_name",
            "sim_backend",
            "teacher_policy_checkpoint_path",
            "teacher_obs_key",
            "teacher_projection",
            "student_projection",
            "action_mode",
            "action_abs_max",
            "num_envs",
            "env_steps",
            "command_sample_filter",
            "command_seen_samples",
            "command_selected_samples",
            "synthetic_teacher_tail",
        )
        for item in source_metadata:
            if not isinstance(item, Mapping):
                compact.append({"type": type(item).__name__, "value": _summarize_value(item)})
                continue
            compact.append({key: _summarize_value(item[key]) for key in keep_keys if key in item})
        summary["source_metadata"] = compact
    else:
        summary["source_metadata"] = _summarize_value(source_metadata)
    return summary


def audit_dataset(
    dataset_path: str | Path,
    *,
    command_xy_threshold: float = 0.05,
    command_yaw_threshold: float = 0.05,
    stat_sample_rows: int = 262144,
) -> dict[str, Any]:
    path = Path(dataset_path)
    payload = _load_payload(path)
    issues: list[str] = []
    warnings: list[str] = []

    student_obs = payload.get("student_obs")
    teacher_obs = payload.get("teacher_obs")
    teacher_actions = payload.get("teacher_actions")
    commands = payload.get("commands")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        warnings.append(f"metadata is not a mapping: {type(metadata).__name__}")
        metadata = {}

    if not isinstance(student_obs, torch.Tensor):
        issues.append("student_obs missing or not a tensor")
        num_samples = 0
    else:
        num_samples = int(student_obs.shape[0]) if student_obs.ndim > 0 else 0
        if student_obs.ndim != 2:
            issues.append(f"student_obs must be rank-2, got shape {tuple(student_obs.shape)}")

    if not isinstance(teacher_obs, torch.Tensor):
        issues.append("teacher_obs missing or not a tensor")
    else:
        if teacher_obs.ndim != 2:
            issues.append(f"teacher_obs must be rank-2, got shape {tuple(teacher_obs.shape)}")
        if isinstance(student_obs, torch.Tensor) and teacher_obs.shape[0] != student_obs.shape[0]:
            issues.append(
                "student_obs/teacher_obs row mismatch: "
                f"{int(student_obs.shape[0])} != {int(teacher_obs.shape[0])}"
            )

    if teacher_actions is not None:
        if not isinstance(teacher_actions, torch.Tensor):
            issues.append(
                f"teacher_actions must be a tensor or None, got {type(teacher_actions).__name__}"
            )
        elif teacher_actions.ndim != 2:
            issues.append(
                f"teacher_actions must be rank-2, got shape {tuple(teacher_actions.shape)}"
            )
        elif (
            isinstance(student_obs, torch.Tensor)
            and teacher_actions.shape[0] != student_obs.shape[0]
        ):
            issues.append(
                "teacher_actions row mismatch: "
                f"{int(teacher_actions.shape[0])} != {int(student_obs.shape[0])}"
            )

    if commands is not None:
        if not isinstance(commands, torch.Tensor):
            issues.append(f"commands must be a tensor or None, got {type(commands).__name__}")
        elif commands.ndim != 2 or int(commands.shape[-1]) != 3:
            issues.append(f"commands must have shape (N, 3), got {tuple(commands.shape)}")
        elif isinstance(student_obs, torch.Tensor) and commands.shape[0] != student_obs.shape[0]:
            issues.append(
                f"commands row mismatch: {int(commands.shape[0])} != {int(student_obs.shape[0])}"
            )

    role_labels = _labels(payload.get("role_labels"))
    command_intents = _labels(payload.get("command_intents"))
    if payload.get("role_labels") is not None and role_labels is None:
        issues.append("role_labels must be a list/tuple or None")
    if payload.get("command_intents") is not None and command_intents is None:
        issues.append("command_intents must be a list/tuple or None")
    if role_labels is not None and len(role_labels) != num_samples:
        issues.append(f"role_labels length mismatch: {len(role_labels)} != {num_samples}")
    if command_intents is not None and len(command_intents) != num_samples:
        issues.append(f"command_intents length mismatch: {len(command_intents)} != {num_samples}")
    if command_intents is not None:
        invalid_intents = sorted(
            {intent for intent in command_intents if intent not in {"active", "inactive"}}
        )
        if invalid_intents:
            issues.append(f"command_intents contain invalid labels: {invalid_intents}")

    payload_num_samples = payload.get("num_samples")
    if payload_num_samples is not None and int(payload_num_samples) != num_samples:
        issues.append(f"payload num_samples mismatch: {payload_num_samples} != {num_samples}")
    if isinstance(student_obs, torch.Tensor) and student_obs.ndim == 2:
        payload_student_dim = payload.get("student_obs_dim")
        if payload_student_dim is not None and int(payload_student_dim) != int(
            student_obs.shape[-1]
        ):
            issues.append(
                f"payload student_obs_dim mismatch: {payload_student_dim} != {int(student_obs.shape[-1])}"
            )
    if isinstance(teacher_obs, torch.Tensor) and teacher_obs.ndim == 2:
        payload_teacher_dim = payload.get("teacher_obs_dim")
        if payload_teacher_dim is not None and int(payload_teacher_dim) != int(
            teacher_obs.shape[-1]
        ):
            issues.append(
                f"payload teacher_obs_dim mismatch: {payload_teacher_dim} != {int(teacher_obs.shape[-1])}"
            )
    if isinstance(teacher_actions, torch.Tensor) and teacher_actions.ndim == 2:
        payload_action_dim = payload.get("teacher_action_dim")
        if payload_action_dim is not None and int(payload_action_dim) != int(
            teacher_actions.shape[-1]
        ):
            issues.append(
                "payload teacher_action_dim mismatch: "
                f"{payload_action_dim} != {int(teacher_actions.shape[-1])}"
            )

    command_summary: dict[str, Any] = {
        "xy_threshold": float(command_xy_threshold),
        "yaw_threshold": float(command_yaw_threshold),
    }
    if isinstance(commands, torch.Tensor) and commands.ndim == 2 and int(commands.shape[-1]) == 3:
        command_summary["stats"] = _tensor_stats(commands, max_rows=stat_sample_rows)
        active_mask = _active_mask_from_commands(
            commands,
            xy_threshold=command_xy_threshold,
            yaw_threshold=command_yaw_threshold,
        )
        recomputed = tuple("active" if bool(value) else "inactive" for value in active_mask.cpu())
        command_summary["recomputed_intent_counts"] = _counts(recomputed)
        if command_intents is not None and len(command_intents) == len(recomputed):
            mismatch = sum(
                1
                for expected, actual in zip(command_intents, recomputed, strict=True)
                if expected != actual
            )
            command_summary["label_threshold_mismatch_count"] = int(mismatch)
            command_summary["label_threshold_mismatch_fraction"] = (
                float(mismatch) / float(len(command_intents)) if command_intents else 0.0
            )
            if mismatch:
                warnings.append(
                    "command_intents differ from threshold-recomputed intents: "
                    f"{mismatch}/{len(command_intents)}"
                )

    role_expected_mismatch = 0
    if (
        role_labels is not None
        and command_intents is not None
        and len(role_labels) == len(command_intents)
    ):
        for role, intent in zip(role_labels, command_intents, strict=True):
            expected = _intent_from_role(role)
            if expected is not None and expected != intent:
                role_expected_mismatch += 1
        if role_expected_mismatch:
            warnings.append(
                "role_labels conflict with command_intents for stand/walk roles: "
                f"{role_expected_mismatch}/{len(role_labels)}"
            )

    report = {
        "status": "ok" if not issues else "issues",
        "dataset_path": str(path),
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "num_samples": int(num_samples),
        "dims": {
            "student_obs_dim": int(student_obs.shape[-1])
            if isinstance(student_obs, torch.Tensor) and student_obs.ndim == 2
            else None,
            "teacher_obs_dim": int(teacher_obs.shape[-1])
            if isinstance(teacher_obs, torch.Tensor) and teacher_obs.ndim == 2
            else None,
            "teacher_action_dim": int(teacher_actions.shape[-1])
            if isinstance(teacher_actions, torch.Tensor) and teacher_actions.ndim == 2
            else None,
        },
        "tensors": {
            "student_obs": _tensor_stats(student_obs, max_rows=stat_sample_rows),
            "teacher_obs": _tensor_stats(teacher_obs, max_rows=stat_sample_rows),
            "teacher_actions": _tensor_stats(teacher_actions, max_rows=stat_sample_rows),
            "commands": _tensor_stats(commands, max_rows=stat_sample_rows),
        },
        "labels": {
            "role_counts": _counts(role_labels),
            "command_intent_counts": _counts(command_intents),
            "role_intent_counts": _paired_counts(role_labels, command_intents),
            "role_expected_intent_mismatch_count": int(role_expected_mismatch),
        },
        "commands": command_summary,
        "sources": _source_summary(metadata, num_samples=num_samples),
        "groups": {
            "by_role": _group_stats(
                labels=role_labels,
                teacher_actions=teacher_actions
                if isinstance(teacher_actions, torch.Tensor)
                else None,
                commands=commands if isinstance(commands, torch.Tensor) else None,
                max_rows=stat_sample_rows,
            ),
            "by_command_intent": _group_stats(
                labels=command_intents,
                teacher_actions=teacher_actions
                if isinstance(teacher_actions, torch.Tensor)
                else None,
                commands=commands if isinstance(commands, torch.Tensor) else None,
                max_rows=stat_sample_rows,
            ),
        },
        "metadata_summary": _summarize_value(metadata),
        "warnings": warnings,
        "issues": issues,
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--command-xy-threshold", type=float, default=0.05)
    parser.add_argument("--command-yaw-threshold", type=float, default=0.05)
    parser.add_argument("--stat-sample-rows", type=int, default=262144)
    parser.add_argument(
        "--strict", action="store_true", help="exit non-zero when hard schema issues exist"
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="exit non-zero when warnings exist; implies --strict behavior for issues",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_dataset(
        args.dataset_path,
        command_xy_threshold=args.command_xy_threshold,
        command_yaw_threshold=args.command_yaw_threshold,
        stat_sample_rows=args.stat_sample_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_warning and (report["issues"] or report["warnings"]):
        return 1
    if args.strict and report["issues"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
