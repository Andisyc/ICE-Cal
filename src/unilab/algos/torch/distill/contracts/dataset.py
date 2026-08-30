from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch

from unilab.algos.torch.distill.datasets.diagnostics import (
    _ORIGINAL_TYPE,
    _TRANSITION_SCENARIOS,
    _abort_for_native_capture,
    _emit_data_runtime,
    _safe_runtime_repr,
    _scenario_label_debug_snapshot,
)


def _validate_obs_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_dim: int | None,
) -> int:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {tuple(tensor.shape)}")
    obs_dim = int(tensor.shape[-1])
    if expected_dim is not None and obs_dim != int(expected_dim):
        raise ValueError(f"{name} dim mismatch: expected {int(expected_dim)}, got {obs_dim}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return obs_dim


def _validate_action_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_dim: int | None,
) -> int:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {tuple(tensor.shape)}")
    action_dim = int(tensor.shape[-1])
    if expected_dim is not None and action_dim != int(expected_dim):
        raise ValueError(f"{name} dim mismatch: expected {int(expected_dim)}, got {action_dim}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return action_dim


def _validate_role_labels(
    role_labels: list[str] | tuple[str, ...] | None,
    *,
    num_samples: int,
) -> tuple[str, ...] | None:
    if role_labels is None:
        return None
    if len(role_labels) != int(num_samples):
        raise ValueError(
            f"role_labels length mismatch: labels={len(role_labels)} samples={int(num_samples)}"
        )
    labels = tuple(str(label) for label in role_labels)
    if any(label == "" for label in labels):
        raise ValueError("role_labels must not contain empty labels")
    return labels


def _validate_commands(
    commands: torch.Tensor | None,
    *,
    num_samples: int,
) -> torch.Tensor | None:
    return _validate_command_tensor("commands", commands, num_samples=num_samples)


def _validate_target_height(
    target_height: torch.Tensor | None,
    *,
    num_samples: int,
) -> torch.Tensor | None:
    if target_height is None:
        return None
    if target_height.ndim != 2 or int(target_height.shape[-1]) != 1:
        raise ValueError(f"target_height must have shape (N, 1), got {tuple(target_height.shape)}")
    if int(target_height.shape[0]) != int(num_samples):
        raise ValueError(
            "target_height batch size mismatch: "
            f"target_height={int(target_height.shape[0])} samples={int(num_samples)}"
        )
    if not torch.isfinite(target_height).all():
        raise ValueError("target_height must contain only finite values")
    return target_height


def _validate_command_tensor(
    name: str,
    commands: torch.Tensor | None,
    *,
    num_samples: int,
) -> torch.Tensor | None:
    if commands is None:
        return None
    if commands.ndim != 2 or int(commands.shape[-1]) != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(commands.shape)}")
    if int(commands.shape[0]) != int(num_samples):
        raise ValueError(
            f"{name} batch size mismatch: "
            f"{name}={int(commands.shape[0])} samples={int(num_samples)}"
        )
    if not torch.isfinite(commands).all():
        raise ValueError(f"{name} must contain only finite values")
    return commands


def _validate_command_intents(
    command_intents: list[str] | tuple[str, ...] | None,
    *,
    num_samples: int,
) -> tuple[str, ...] | None:
    if command_intents is None:
        return None
    if len(command_intents) != int(num_samples):
        raise ValueError(
            "command_intents length mismatch: "
            f"intents={len(command_intents)} samples={int(num_samples)}"
        )
    intents = tuple(str(intent) for intent in command_intents)
    allowed = {"active", "inactive"}
    invalid_indices = [index for index, intent in enumerate(intents) if intent not in allowed]
    if invalid_indices:
        invalid_head = [
            {
                "index": index,
                "raw_type": _ORIGINAL_TYPE(command_intents[index]).__name__,
                "raw_repr": _safe_runtime_repr(command_intents[index]),
                "normalized": intents[index],
            }
            for index in invalid_indices[:10]
        ]
        abort_requested = os.environ.get("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "0") == "1"
        _emit_data_runtime(
            "command_intent_validation/corruption_detected",
            num_samples=int(num_samples),
            command_intents_type=_ORIGINAL_TYPE(command_intents).__name__,
            command_intents_length=len(command_intents),
            invalid_count=len(invalid_indices),
            invalid_head=invalid_head,
            native_abort_requested=abort_requested,
        )
        if abort_requested:
            _abort_for_native_capture()
        raise ValueError(
            "command_intents must contain only active/inactive labels; "
            f"invalid_head={invalid_head!r}"
        )
    return intents


def _command_intents_from_commands(
    commands: torch.Tensor,
    *,
    xy_threshold: float,
    yaw_threshold: float,
) -> tuple[str, ...]:
    xy_threshold = float(xy_threshold)
    yaw_threshold = float(yaw_threshold)
    if xy_threshold < 0.0:
        raise ValueError(f"command_xy_threshold must be non-negative, got {xy_threshold}")
    if yaw_threshold < 0.0:
        raise ValueError(f"command_yaw_threshold must be non-negative, got {yaw_threshold}")
    xy_norm = torch.linalg.norm(commands[:, :2], dim=1)
    yaw_abs = commands[:, 2].abs()
    active = (xy_norm > xy_threshold) | (yaw_abs > yaw_threshold)
    return tuple("active" if bool(value) else "inactive" for value in active.detach().cpu())


def _command_intents_from_role_labels(
    role_labels: tuple[str, ...],
) -> tuple[str, ...] | None:
    intents: list[str] = []
    for role in role_labels:
        normalized = role.lower()
        if "stand" in normalized:
            intents.append("inactive")
        elif "walk" in normalized:
            intents.append("active")
        else:
            return None
    return tuple(intents)


def _validate_scenario_labels(
    scenario_labels: list[str] | tuple[str, ...] | None,
    *,
    num_samples: int,
) -> tuple[str, ...] | None:
    if scenario_labels is None:
        return None
    entry_snapshot = _scenario_label_debug_snapshot(scenario_labels)
    _emit_data_runtime(
        "scenario_validation/entry",
        num_samples=num_samples,
        scenario_labels=entry_snapshot,
    )
    if len(scenario_labels) != int(num_samples):
        _emit_data_runtime(
            "scenario_validation/failure",
            reason="length_mismatch",
            num_samples=num_samples,
            scenario_labels=entry_snapshot,
        )
        raise ValueError(
            "scenario_labels length mismatch: "
            f"labels={len(scenario_labels)} samples={int(num_samples)}"
        )
    labels = tuple(str(label) for label in scenario_labels)
    if any(label == "" for label in labels):
        _emit_data_runtime(
            "scenario_validation/failure",
            reason="empty_label",
            num_samples=num_samples,
            scenario_labels=_scenario_label_debug_snapshot(scenario_labels),
        )
        raise ValueError("scenario_labels must not contain empty labels")
    unknown = sorted(set(labels) - _TRANSITION_SCENARIOS)
    if unknown:
        _emit_data_runtime(
            "scenario_validation/failure",
            reason="unknown_label",
            num_samples=num_samples,
            unknown=unknown,
            scenario_labels=_scenario_label_debug_snapshot(scenario_labels),
        )
        raise ValueError(
            f"scenario_labels must contain only static_stand/walk_flat/walk_to_stop, got {unknown}"
        )
    _emit_data_runtime(
        "scenario_validation/success",
        num_samples=num_samples,
        scenario_labels=_scenario_label_debug_snapshot(scenario_labels),
    )
    return labels


def _validate_transition_ages(
    transition_ages: torch.Tensor | None,
    *,
    num_samples: int,
) -> torch.Tensor | None:
    if transition_ages is None:
        return None
    if transition_ages.ndim != 1:
        raise ValueError(
            f"transition_ages must have shape (N,), got {tuple(transition_ages.shape)}"
        )
    if int(transition_ages.shape[0]) != int(num_samples):
        raise ValueError(
            "transition_ages batch size mismatch: "
            f"transition_ages={int(transition_ages.shape[0])} samples={int(num_samples)}"
        )
    if transition_ages.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError(f"transition_ages must have an integer dtype, got {transition_ages.dtype}")
    if torch.any(transition_ages < -1):
        raise ValueError("transition_ages must be -1 or non-negative")
    return transition_ages


def _validate_transition_fields(
    *,
    scenario_labels: list[str] | tuple[str, ...] | None,
    transition_ages: torch.Tensor | None,
    command_before: torch.Tensor | None,
    command_after: torch.Tensor | None,
    num_samples: int,
) -> tuple[
    tuple[str, ...] | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    has_extra = any(value is not None for value in (transition_ages, command_before, command_after))
    validated_labels = _validate_scenario_labels(scenario_labels, num_samples=num_samples)
    if validated_labels is None:
        if has_extra:
            raise ValueError("transition fields require scenario_labels")
        return None, None, None, None
    if transition_ages is None:
        raise ValueError("scenario_labels require transition_ages")
    validated_ages = _validate_transition_ages(transition_ages, num_samples=num_samples)
    if validated_ages is None:
        raise RuntimeError("validated transition_ages unexpectedly missing")
    if (command_before is None) != (command_after is None):
        raise ValueError("command_before and command_after must be provided together")
    validated_before = _validate_command_tensor(
        "command_before",
        command_before,
        num_samples=num_samples,
    )
    validated_after = _validate_command_tensor(
        "command_after",
        command_after,
        num_samples=num_samples,
    )
    transition_mask = torch.tensor(
        [label == "walk_to_stop" for label in validated_labels],
        dtype=torch.bool,
        device=validated_ages.device,
    )
    static_mask = ~transition_mask
    if torch.any(validated_ages[static_mask] != -1):
        raise ValueError("static_stand/walk_flat rows must use transition_age=-1")
    if bool(transition_mask.any()):
        if validated_before is None or validated_after is None:
            raise ValueError("walk_to_stop rows require command_before and command_after")
        post_switch = transition_mask & (validated_ages >= 0)
        if torch.any(validated_after[post_switch].abs() > 1e-6):
            raise ValueError("walk_to_stop post-switch command_after must be zero")
    return validated_labels, validated_ages, validated_before, validated_after
