from __future__ import annotations

import builtins
import json
import os
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.distill.observability.debug import (
    _distill_runtime_debug_enabled,
)

_ORIGINAL_CALLABLE = callable
_ORIGINAL_ISINSTANCE = isinstance
_ORIGINAL_LIST = list
_ORIGINAL_REPR = repr
_ORIGINAL_STR = str
_ORIGINAL_TUPLE = tuple
_ORIGINAL_TYPE = type

_TRANSITION_SCENARIOS = {"static_stand", "walk_flat", "walk_to_stop"}


def _label_counts(labels: tuple[str, ...]) -> dict[str, int]:
    return {label: labels.count(label) for label in sorted(set(labels))}

def _command_intent_debug_snapshot(
    command_intents: Sequence[Any],
) -> dict[str, Any]:
    normalized = tuple(_ORIGINAL_STR(intent) for intent in command_intents)
    return {
        "type": type(command_intents).__name__,
        "length": len(normalized),
        "command_intent_counts": _label_counts(normalized),
        "invalid_head": [
            {
                "index": index,
                "type": type(command_intents[index]).__name__,
                "repr": repr(command_intents[index]),
                "normalized": intent,
            }
            for index, intent in enumerate(normalized)
            if intent not in {"active", "inactive"}
        ][:10],
    }


def _expected_command_intent_for_scenario(scenario: str | None) -> str | None:
    if scenario == "walk_flat":
        return "active"
    if scenario == "static_stand":
        return "inactive"
    return None


def _command_intent_contract_debug_snapshot(
    command_intents: Sequence[Any] | None,
    *,
    expected_intent: str | None,
) -> dict[str, Any] | None:
    if command_intents is None:
        return None
    normalized = tuple(_ORIGINAL_STR(intent) for intent in command_intents)
    snapshot = _command_intent_debug_snapshot(command_intents)
    snapshot["expected_intent"] = expected_intent
    snapshot["expected_mismatch_head"] = (
        []
        if expected_intent is None
        else [
            {
                "index": index,
                "type": _ORIGINAL_TYPE(command_intents[index]).__name__,
                "repr": _safe_runtime_repr(command_intents[index]),
                "normalized": intent,
            }
            for index, intent in enumerate(normalized)
            if intent != expected_intent
        ][:10]
    )
    return snapshot


def _multitask_source_debug_snapshot(
    *,
    source_index: int,
    path: Path,
    role: str,
    scenario: str | None,
    dataset: Any,
    error: BaseException | None = None,
) -> dict[str, Any]:
    metadata = dict(dataset.metadata)
    metadata_keys = (
        "source",
        "scenario_annotation",
        "workflow_scenario",
        "command_sample_filter",
        "command_seen_samples",
        "command_selected_samples",
        "command_intent_counts",
        "scenario_counts",
        "role_label_counts",
        "num_samples",
    )
    expected_intent = _expected_command_intent_for_scenario(scenario)
    snapshot: dict[str, Any] = {
        "source_index": int(source_index),
        "path": str(path),
        "role": role,
        "requested_scenario": scenario,
        "num_samples": dataset.num_samples,
        "student_obs_shape": tuple(dataset.student_obs.shape),
        "teacher_obs_shape": tuple(dataset.teacher_obs.shape),
        "teacher_actions_shape": (
            None if dataset.teacher_actions is None else tuple(dataset.teacher_actions.shape)
        ),
        "commands_shape": None if dataset.commands is None else tuple(dataset.commands.shape),
        "target_height_shape": (
            None if dataset.target_height is None else tuple(dataset.target_height.shape)
        ),
        "command_intents": _command_intent_contract_debug_snapshot(
            dataset.command_intents,
            expected_intent=expected_intent,
        ),
        "scenario_labels": (
            None
            if dataset.scenario_labels is None
            else _scenario_label_debug_snapshot(dataset.scenario_labels)
        ),
        "metadata": {key: metadata[key] for key in metadata_keys if key in metadata},
    }
    if error is not None:
        snapshot["error_type"] = _ORIGINAL_TYPE(error).__name__
        snapshot["error"] = _ORIGINAL_STR(error)
        snapshot["error_repr"] = _safe_runtime_repr(error)
    return snapshot


def _metadata_workflow_scenario(dataset: Any) -> str | None:
    value = dataset.metadata.get("workflow_scenario")
    if value in (None, ""):
        return None
    scenario = _ORIGINAL_STR(value)
    if scenario not in _TRANSITION_SCENARIOS:
        raise ValueError(f"dataset metadata workflow_scenario is invalid: {scenario!r}")
    return scenario


def _safe_runtime_repr(value: Any) -> str:
    try:
        return _ORIGINAL_REPR(value)
    except BaseException as error:  # pragma: no cover - defensive runtime probe
        return f"<repr-error type={_ORIGINAL_TYPE(error).__name__} repr={_ORIGINAL_REPR(error)}>"


def _scenario_label_debug_snapshot(
    scenario_labels: Sequence[Any],
    *,
    source_ranges: Sequence[Mapping[str, Any]] = (),
    force: bool = False,
) -> dict[str, Any]:
    if not force and not _distill_runtime_debug_enabled():
        return {"runtime_debug_enabled": False}

    label_counts: dict[str, int] = {}
    invalid_head: list[dict[str, Any]] = []
    boundary_entries: list[dict[str, Any]] = []
    length = len(scenario_labels)
    boundary_indices = sorted({0, 1, max(0, length - 2), max(0, length - 1)})

    for index, raw_label in enumerate(scenario_labels):
        try:
            normalized_label = _ORIGINAL_STR(raw_label)
        except BaseException as error:  # pragma: no cover - defensive runtime probe
            normalized_label = (
                "<str-error "
                f"type={_ORIGINAL_TYPE(error).__name__} "
                f"repr={_safe_runtime_repr(error)}>"
            )
        label_counts[normalized_label] = label_counts.get(normalized_label, 0) + 1
        entry = {
            "index": index,
            "raw_type": _ORIGINAL_TYPE(raw_label).__name__,
            "raw_repr": _safe_runtime_repr(raw_label),
            "normalized": normalized_label,
        }
        if index in boundary_indices:
            boundary_entries.append(entry)
        if (
            _ORIGINAL_TYPE(raw_label) is not _ORIGINAL_STR
            or normalized_label not in _TRANSITION_SCENARIOS
        ) and len(invalid_head) < 10:
            if source_ranges:
                provenance = None
                for source_range in source_ranges:
                    if source_range["global_start"] <= index < source_range["global_stop"]:
                        provenance = source_range
                        break
                enriched = {
                    "global_index": index,
                    "raw_type": entry["raw_type"],
                    "raw_repr": entry["raw_repr"],
                    "normalized": normalized_label,
                }
                if provenance is not None:
                    enriched.update(
                        {
                            "source_index": provenance["source_index"],
                            "source_row_index": index - provenance["global_start"],
                            "path": provenance["path"],
                            "role": provenance["role"],
                            "scenario": provenance["scenario"],
                        }
                    )
                invalid_head.append(enriched)
            else:
                invalid_head.append(entry)

    return {
        "type": _ORIGINAL_TYPE(scenario_labels).__name__,
        "length": length,
        "label_counts": dict(sorted(label_counts.items())),
        "boundary_entries": boundary_entries,
        "invalid_head": invalid_head,
    }


def _emit_data_runtime(stage: str, **fields: Any) -> None:
    if not _distill_runtime_debug_enabled():
        return
    is_storage = torch.is_storage
    current_int = builtins.int
    current_isinstance = builtins.isinstance
    current_str = builtins.str
    current_type = builtins.type
    current_tuple = builtins.tuple
    current_list = builtins.list
    trace = sys.gettrace()
    profile = sys.getprofile()
    snapshot = {
        "stage": stage,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "torch_is_storage_type": _ORIGINAL_TYPE(is_storage).__name__,
        "torch_is_storage_repr": _safe_runtime_repr(is_storage),
        "torch_is_storage_callable": _ORIGINAL_CALLABLE(is_storage),
        "builtins_int_type": _ORIGINAL_TYPE(current_int).__name__,
        "builtins_int_repr": _safe_runtime_repr(current_int),
        "builtins_int_callable": _ORIGINAL_CALLABLE(current_int),
        "builtins_isinstance_type": _ORIGINAL_TYPE(current_isinstance).__name__,
        "builtins_isinstance_repr": _safe_runtime_repr(current_isinstance),
        "builtins_isinstance_callable": _ORIGINAL_CALLABLE(current_isinstance),
        "builtins_isinstance_is_original": current_isinstance is _ORIGINAL_ISINSTANCE,
        "builtins_str_type": _ORIGINAL_TYPE(current_str).__name__,
        "builtins_str_repr": _safe_runtime_repr(current_str),
        "builtins_str_callable": _ORIGINAL_CALLABLE(current_str),
        "builtins_str_is_original": current_str is _ORIGINAL_STR,
        "builtins_type_type": _ORIGINAL_TYPE(current_type).__name__,
        "builtins_type_repr": _safe_runtime_repr(current_type),
        "builtins_type_callable": _ORIGINAL_CALLABLE(current_type),
        "builtins_type_is_original": current_type is _ORIGINAL_TYPE,
        "builtins_tuple_type": _ORIGINAL_TYPE(current_tuple).__name__,
        "builtins_tuple_repr": _safe_runtime_repr(current_tuple),
        "builtins_tuple_callable": _ORIGINAL_CALLABLE(current_tuple),
        "builtins_tuple_is_original": current_tuple is _ORIGINAL_TUPLE,
        "builtins_list_type": _ORIGINAL_TYPE(current_list).__name__,
        "builtins_list_repr": _safe_runtime_repr(current_list),
        "builtins_list_callable": _ORIGINAL_CALLABLE(current_list),
        "builtins_list_is_original": current_list is _ORIGINAL_LIST,
        "sys_trace_type": None if trace is None else _ORIGINAL_TYPE(trace).__name__,
        "sys_trace_repr": None if trace is None else _safe_runtime_repr(trace),
        "sys_profile_type": None if profile is None else _ORIGINAL_TYPE(profile).__name__,
        "sys_profile_repr": None if profile is None else _safe_runtime_repr(profile),
        **fields,
    }
    print(f"[distill-data-runtime] {snapshot!r}", flush=True)


def _native_abort_for_impossible_callable_error_requested(error: BaseException) -> bool:
    return (
        os.environ.get("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "0") == "1"
        and _ORIGINAL_ISINSTANCE(error, TypeError)
        and "object is not callable" in _ORIGINAL_STR(error)
    )


def _abort_for_native_capture() -> None:
    # 仅用于诊断: 在 Apport core 中保留当前 learner 进程状态.
    sys.stdout.flush()
    sys.stderr.flush()
    os.abort()
