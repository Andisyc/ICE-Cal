"""G1-owned calibration faults at the final action-execution boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class G1ActionExecutionFaultConfig:
    """One explicit action fault applied after task action authority."""

    mode: str
    gain: float

    def validate(self) -> G1ActionExecutionFaultConfig:
        if self.mode not in {"none", "gain"}:
            raise ValueError(f"unsupported G1 action-execution fault mode: {self.mode!r}")
        if not math.isfinite(float(self.gain)):
            raise ValueError("G1 action-execution fault gain must be finite")
        if float(self.gain) <= 0.0:
            raise ValueError("G1 action-execution fault gain must be positive")
        return self


def apply_action_execution_fault(
    authority_actions: np.ndarray,
    config: G1ActionExecutionFaultConfig | None,
    *,
    num_envs: int,
) -> np.ndarray:
    """Transform authority-approved actions without consulting simulator state."""

    actions = np.asarray(authority_actions)
    if actions.ndim != 2:
        raise ValueError(
            f"authority actions must be rank-2 [environment, action], got {actions.shape}"
        )
    if actions.shape[0] != int(num_envs):
        raise ValueError(
            "authority action environment batch mismatch: "
            f"expected {num_envs}, got {actions.shape[0]}"
        )
    if config is None:
        return actions
    config.validate()
    if config.mode == "none" or float(config.gain) == 1.0:
        return actions
    return np.asarray(actions * float(config.gain), dtype=actions.dtype)
