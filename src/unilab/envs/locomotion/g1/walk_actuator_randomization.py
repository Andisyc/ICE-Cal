"""Fixed actuator faults and the independent physical DR curriculum.

Random single-actuator weakening has been removed. Historical disabled config
fields remain readable, but enabling that sampling mode is an explicit error.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_actuator_strength_config(
    strength_cfg: Any | None, *, expected_actions: int
) -> Any | None:
    if strength_cfg is None:
        return None
    enabled = bool(getattr(strength_cfg, "enabled", False))
    include_in_critic = bool(getattr(strength_cfg, "include_in_critic_obs", False))
    if not enabled:
        if include_in_critic:
            raise ValueError(
                "domain_rand.actuator_strength.include_in_critic_obs requires enabled=true"
            )
        validate_grouped_domain_rand_curriculum(strength_cfg)
        return None

    if bool(getattr(strength_cfg, "curriculum_enabled", False)):
        raise ValueError("actuator-strength randomization curriculum has been removed")
    sampling_mode = str(getattr(strength_cfg, "sampling_mode", "fixed"))
    if sampling_mode == "fixed":
        multipliers = np.asarray(strength_cfg.multipliers, dtype=np.float64)
        if multipliers.shape != (expected_actions,):
            raise ValueError(
                "domain_rand.actuator_strength requires exactly "
                f"{expected_actions} multipliers, got shape {multipliers.shape}"
            )
        if not np.isfinite(multipliers).all():
            raise ValueError("domain_rand.actuator_strength multipliers must be finite")
        if np.any(multipliers <= 0.0) or np.any(multipliers > 1.0):
            raise ValueError(
                "domain_rand.actuator_strength multipliers must be in the interval (0, 1]"
            )
        if list(getattr(strength_cfg, "candidate_actuator_indices", [])):
            raise ValueError("fixed actuator strength cannot define candidate_actuator_indices")
        return strength_cfg

    raise ValueError(
        "random actuator-strength weakening has been removed; "
        "only fixed multipliers are supported"
    )


def validate_grouped_domain_rand_curriculum(cfg: Any) -> None:
    """Validate the physical DR schedule independently of knee fault sampling.

    Keep the existing schedule fields for config compatibility; the knee's
    enabled flag and multiplier schedule do not own grouped randomization.
    """

    if not bool(getattr(cfg, "group_curriculum_enabled", False)):
        return
    scales = np.asarray(cfg.group_curriculum_scales, dtype=np.float64)
    if scales.ndim != 1 or scales.size == 0 or not np.isfinite(scales).all():
        raise ValueError("group curriculum scales must be a non-empty finite list")
    if (
        scales[0] != 0.0
        or scales[-1] != 1.0
        or np.any(scales < 0.0)
        or np.any(scales > 1.0)
        or np.any(np.diff(scales) < 0.0)
    ):
        raise ValueError("group curriculum scales must ascend from 0 to 1")
    promote = float(cfg.curriculum_promote_threshold)
    demote = float(cfg.curriculum_demote_threshold)
    if not np.isfinite([promote, demote]).all() or not demote < promote:
        raise ValueError("group curriculum thresholds must satisfy down < up")
    if int(cfg.curriculum_update_episodes) <= 0:
        raise ValueError("curriculum_update_episodes must be positive")
    _validate_curriculum_progress(cfg, num_levels=len(scales))


def _validate_curriculum_progress(strength_cfg: Any, *, num_levels: int) -> None:
    progress_mode = str(
        getattr(strength_cfg, "curriculum_progress_mode", "episode_quality")
    )
    if progress_mode not in {"episode_quality", "iterations"}:
        raise ValueError("unsupported actuator strength curriculum progress mode")
    if progress_mode == "iterations":
        boundaries = np.asarray(
            strength_cfg.curriculum_iteration_boundaries, dtype=np.int64
        )
        if (
            boundaries.shape != (num_levels,)
            or boundaries[0] != 0
            or np.any(np.diff(boundaries) <= 0)
        ):
            raise ValueError(
                "iteration curriculum boundaries must align and strictly increase from 0"
            )
        max_rate = float(strength_cfg.curriculum_max_termination_rate)
        if not np.isfinite(max_rate) or not 0.0 <= max_rate <= 1.0:
            raise ValueError("curriculum_max_termination_rate must be in [0, 1]")
        if int(strength_cfg.curriculum_brake_cooldown_steps) <= 0:
            raise ValueError("curriculum_brake_cooldown_steps must be positive")
        if int(strength_cfg.curriculum_recovery_hold_steps) < 0:
            raise ValueError("curriculum_recovery_hold_steps must be non-negative")


def scale_symmetric_range(values: Any, *, center: float, scale: float) -> list[float]:
    low, high = (float(value) for value in values)
    return [center + (low - center) * scale, center + (high - center) * scale]


def sample_actuator_strength_multipliers(
    strength_cfg: Any,
    *,
    num_reset: int,
    expected_actions: int,
) -> np.ndarray:
    if validate_actuator_strength_config(strength_cfg, expected_actions=expected_actions) is None:
        raise ValueError("fixed actuator strength is not enabled")
    fixed = np.asarray(strength_cfg.multipliers, dtype=np.float64)
    return np.broadcast_to(fixed, (num_reset, expected_actions)).copy()
