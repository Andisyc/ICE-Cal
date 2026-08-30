"""Pure G1 command scheduling and stand-phase decisions."""

from __future__ import annotations

import numpy as np


def command_resample_mask(steps: np.ndarray, *, interval_steps: int) -> np.ndarray:
    if interval_steps <= 0:
        raise ValueError("interval_steps must be positive")
    step_values = np.asarray(steps)
    return (step_values > 0) & ((step_values % int(interval_steps)) == 0)


def freeze_inactive_gait_phase(
    gait_phase: np.ndarray,
    active: np.ndarray,
    stand_phase: np.ndarray,
) -> np.ndarray:
    phase = np.array(gait_phase, copy=True)
    phase[~np.asarray(active, dtype=bool)] = np.asarray(stand_phase, dtype=phase.dtype)
    return phase
