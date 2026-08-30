"""Pure G1 walk action-authority and gait-phase decisions."""

from __future__ import annotations

import numpy as np


def select_authority_actions(
    actions: np.ndarray,
    active: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    if not enabled or np.all(active):
        return actions
    executed = np.array(actions, copy=True)
    executed[~np.asarray(active, dtype=bool)] = 0.0
    return executed


def advance_gait_phase(
    gait_phase: np.ndarray,
    *,
    active: np.ndarray,
    delta: float,
    enabled: bool,
    freeze_inactive: bool,
    stand_phase: np.ndarray,
) -> np.ndarray:
    phase = np.array(gait_phase, copy=True)
    if not enabled:
        return np.zeros_like(phase)
    if freeze_inactive:
        active_mask = np.asarray(active, dtype=bool)
        phase[active_mask] = (phase[active_mask] + float(delta)) % (2 * np.pi)
        phase[~active_mask] = np.asarray(stand_phase, dtype=phase.dtype)
        return phase
    phase[:] = (phase + float(delta)) % (2 * np.pi)
    return phase
