"""Read-only runtime diagnostics for behavior distillation training."""

from __future__ import annotations

import builtins
import gc
import os
import sys
import threading
from collections import Counter
from typing import Any

import torch

from unilab.algos.torch.distill.observability.debug import (
    _distill_runtime_debug_enabled,
)

_DISTILL_RUNTIME_TRACE_INTERVAL = 100
_ORIGINAL_INT = int
_ORIGINAL_REPR = repr
_ORIGINAL_TORCH_TENSOR = torch.tensor
_ORIGINAL_TYPE = type


def _runtime_trace_update(update_number: int) -> bool:
    return _distill_runtime_debug_enabled() and (
        update_number == 1 or update_number % _DISTILL_RUNTIME_TRACE_INTERVAL == 0
    )


def _label_counts(labels: tuple[str, ...] | None) -> dict[str, int]:
    return {} if labels is None else dict(Counter(str(label) for label in labels))


def _runtime_identity_snapshot() -> dict[str, Any]:
    current_int = builtins.int
    trace = sys.gettrace()
    profile = sys.getprofile()
    return {
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "builtins_int_type": type(current_int).__name__,
        "builtins_int_repr": repr(current_int),
        "builtins_int_callable": callable(current_int),
        "builtins_int_is_original": current_int is _ORIGINAL_INT,
        "sys_trace_type": None if trace is None else type(trace).__name__,
        "sys_trace_repr": None if trace is None else repr(trace),
        "sys_profile_type": None if profile is None else type(profile).__name__,
        "sys_profile_repr": None if profile is None else repr(profile),
    }


def _tensor_runtime_snapshot(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
        "grad_fn_type": None if tensor.grad_fn is None else type(tensor.grad_fn).__name__,
        "finite": bool(torch.isfinite(tensor.detach()).all()) if tensor.numel() else True,
    }


def _safe_runtime_repr(value: Any) -> str:
    try:
        return _ORIGINAL_REPR(value)
    except BaseException as error:  # pragma: no cover - defensive runtime probe
        return f"<repr-error type={_ORIGINAL_TYPE(error).__name__} repr={_ORIGINAL_REPR(error)}>"


def _target_index_list_runtime_snapshot(target_indices: list[int]) -> dict[str, Any]:
    element_type_counts = Counter(_ORIGINAL_TYPE(value).__name__ for value in target_indices)
    invalid_head = [
        {
            "index": index,
            "raw_type": _ORIGINAL_TYPE(value).__name__,
            "raw_repr": _safe_runtime_repr(value),
        }
        for index, value in enumerate(target_indices)
        if _ORIGINAL_TYPE(value) is not _ORIGINAL_INT
    ][:16]
    length = len(target_indices)
    boundary_indices = sorted({0, 1, max(0, length - 2), max(0, length - 1)})
    boundary_entries = [
        {
            "index": index,
            "raw_type": _ORIGINAL_TYPE(target_indices[index]).__name__,
            "raw_repr": _safe_runtime_repr(target_indices[index]),
        }
        for index in boundary_indices
        if index < length
    ]
    return {
        "type": _ORIGINAL_TYPE(target_indices).__name__,
        "id": id(target_indices),
        "length": length,
        "size_bytes": sys.getsizeof(target_indices),
        "refcount": sys.getrefcount(target_indices),
        "gc_tracked": gc.is_tracked(target_indices),
        "none_count": sum(value is None for value in target_indices),
        "non_int_count": sum(
            _ORIGINAL_TYPE(value) is not _ORIGINAL_INT for value in target_indices
        ),
        "element_type_counts": dict(sorted(element_type_counts.items())),
        "boundary_entries": boundary_entries,
        "invalid_head": invalid_head,
    }


def _emit_trainer_runtime(stage: str, **fields: Any) -> None:
    if not _distill_runtime_debug_enabled():
        return
    snapshot = {"stage": stage, **_runtime_identity_snapshot(), **fields}
    print(f"[distill-trainer-runtime] {snapshot!r}", flush=True)


def append_runtime_target_index(
    *,
    target_indices: list[int],
    raw_target: Any,
    label_name: str,
    label_key: str,
    row_index: int,
    update_number: int,
    trace_row: bool,
) -> None:
    int_fn = builtins.int
    append_fn = getattr(target_indices, "append", None)
    context = {
        "update_number": update_number,
        "label_name": label_name,
        "label_key": label_key,
        "row_index": row_index,
        "raw_target_type": type(raw_target).__name__,
        "raw_target_repr": repr(raw_target),
        "target_indices_type": type(target_indices).__name__,
        "target_indices_length": len(target_indices),
        "target_indices_head": tuple(target_indices[:8]),
        "append_type": type(append_fn).__name__,
        "append_repr": repr(append_fn),
        "append_callable": callable(append_fn),
    }
    if trace_row:
        _emit_trainer_runtime("target_index/before_int", **context)
    try:
        converted_target = int_fn(raw_target)
    except Exception as error:
        _emit_trainer_runtime(
            "target_index/int_failure",
            **context,
            error_type=type(error).__name__,
            error_repr=repr(error),
        )
        raise
    if trace_row:
        _emit_trainer_runtime(
            "target_index/after_int",
            **context,
            converted_target_type=type(converted_target).__name__,
            converted_target_repr=repr(converted_target),
        )
    try:
        append_fn(converted_target)  # type: ignore[misc]
    except Exception as error:
        _emit_trainer_runtime(
            "target_index/append_failure",
            **context,
            converted_target_type=type(converted_target).__name__,
            converted_target_repr=repr(converted_target),
            error_type=type(error).__name__,
            error_repr=repr(error),
        )
        raise
    if trace_row:
        _emit_trainer_runtime(
            "target_index/after_append",
            **context,
            converted_target_type=type(converted_target).__name__,
            converted_target_repr=repr(converted_target),
            target_indices_length_after=len(target_indices),
            target_indices_head_after=tuple(target_indices[:8]),
        )
