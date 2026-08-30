"""Shared read-only switches for distillation diagnostics."""

from __future__ import annotations

import os

_DISTILL_RUNTIME_DEBUG_ENV = "UNILAB_DISTILL_RUNTIME_DEBUG"
_DISTILL_RUNTIME_DEBUG_FALSE_VALUES = {"", "0", "false", "no", "off"}


def _distill_runtime_debug_enabled() -> bool:
    value = os.environ.get(_DISTILL_RUNTIME_DEBUG_ENV, "0")
    return value.strip().lower() not in _DISTILL_RUNTIME_DEBUG_FALSE_VALUES
