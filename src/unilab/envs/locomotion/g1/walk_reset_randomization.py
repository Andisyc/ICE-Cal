"""Stateless reset-time gait decisions for G1 walking."""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype


def freeze_standing_phase(
    gait_phase: np.ndarray,
    *,
    gait_enabled: np.ndarray,
    enabled: bool,
    freeze: bool,
    stand_phase: Any,
) -> np.ndarray:
    if not (enabled and freeze):
        return gait_phase
    stand_phase_arr = np.asarray(stand_phase, dtype=get_global_dtype())
    if stand_phase_arr.shape != (2,):
        raise ValueError(f"gait_constraint.stand_phase must have shape (2,), got {stand_phase}")
    standing = np.asarray(gait_enabled <= 0.5, dtype=bool)
    if np.any(standing):
        gait_phase[standing, :] = stand_phase_arr[None, :]
    return gait_phase


def sample_gait_phase(*, num_reset: int, enabled: bool, mode: str) -> np.ndarray:
    if not enabled:
        return np.zeros((num_reset, 2), dtype=get_global_dtype())
    if mode == "independent":
        left = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        right = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.asarray(np.column_stack([left, right]), dtype=get_global_dtype())
    phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
    return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())

