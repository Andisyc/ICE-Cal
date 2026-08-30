"""Stateless actuator-strength domain-randomization decisions."""

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
        return None

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

    if sampling_mode != "single_candidate":
        raise ValueError(
            "domain_rand.actuator_strength.sampling_mode must be 'fixed' or "
            f"'single_candidate', got {sampling_mode!r}"
        )
    if list(getattr(strength_cfg, "multipliers", [])):
        raise ValueError("single_candidate actuator strength cannot define fixed multipliers")
    candidates = np.asarray(strength_cfg.candidate_actuator_indices, dtype=np.int64)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("single_candidate actuator strength requires candidate_actuator_indices")
    if np.unique(candidates).size != candidates.size:
        raise ValueError("actuator strength candidate indices must be unique")
    if np.any(candidates < 0) or np.any(candidates >= expected_actions):
        raise ValueError(
            f"actuator strength candidate indices must be in [0, {expected_actions})"
        )
    multiplier_range = np.asarray(strength_cfg.multiplier_range, dtype=np.float64)
    if multiplier_range.shape != (2,) or not np.isfinite(multiplier_range).all():
        raise ValueError("actuator strength multiplier_range must contain two finite values")
    low, high = multiplier_range.tolist()
    if low <= 0.0 or high < low or high > 1.0:
        raise ValueError("actuator strength multiplier_range must satisfy 0 < low <= high <= 1")
    nominal_probability = float(strength_cfg.nominal_probability)
    if not np.isfinite(nominal_probability) or not 0.0 <= nominal_probability <= 1.0:
        raise ValueError("actuator strength nominal_probability must be in [0, 1]")
    validate_actuator_strength_curriculum(strength_cfg, low=low, high=high)
    return strength_cfg


def validate_actuator_strength_curriculum(
    strength_cfg: Any, *, low: float, high: float
) -> None:
    if not bool(getattr(strength_cfg, "curriculum_enabled", False)):
        return
    nominal_probability = float(strength_cfg.nominal_probability)
    lows = np.asarray(strength_cfg.curriculum_multiplier_lows, dtype=np.float64)
    probabilities = np.asarray(
        strength_cfg.curriculum_nominal_probabilities, dtype=np.float64
    )
    if lows.ndim != 1 or lows.size == 0 or probabilities.shape != lows.shape:
        raise ValueError("actuator strength curriculum schedules must be non-empty and aligned")
    if not np.isfinite(lows).all() or np.any(lows < low) or np.any(lows > high):
        raise ValueError("actuator strength curriculum multiplier lows are out of range")
    if np.any(np.diff(lows) > 0.0) or lows[0] != high or lows[-1] != low:
        raise ValueError(
            "actuator strength curriculum multiplier lows must descend from high to low"
        )
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < nominal_probability)
        or np.any(probabilities > 1.0)
        or np.any(np.diff(probabilities) > 0.0)
        or probabilities[0] != 1.0
        or probabilities[-1] != nominal_probability
    ):
        raise ValueError(
            "actuator strength curriculum nominal probabilities must descend from 1"
        )
    promote = float(strength_cfg.curriculum_promote_threshold)
    demote = float(strength_cfg.curriculum_demote_threshold)
    if not np.isfinite([promote, demote]).all() or not demote < promote:
        raise ValueError("actuator strength curriculum thresholds must satisfy down < up")
    if int(strength_cfg.curriculum_update_episodes) <= 0:
        raise ValueError("actuator strength curriculum_update_episodes must be positive")
    if bool(getattr(strength_cfg, "group_curriculum_enabled", False)):
        scales = np.asarray(strength_cfg.group_curriculum_scales, dtype=np.float64)
        if scales.shape != lows.shape or not np.isfinite(scales).all():
            raise ValueError("group curriculum scales must align with actuator strength levels")
        if (
            scales[0] != 0.0
            or scales[-1] != 1.0
            or np.any(scales < 0.0)
            or np.any(scales > 1.0)
            or np.any(np.diff(scales) < 0.0)
        ):
            raise ValueError("group curriculum scales must ascend from 0 to 1")
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
            boundaries.shape != lows.shape
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
    curriculum_profile: tuple[int, float, float] | None,
) -> np.ndarray:
    if str(getattr(strength_cfg, "sampling_mode", "fixed")) == "fixed":
        fixed = np.asarray(strength_cfg.multipliers, dtype=np.float64)
        return np.broadcast_to(fixed, (num_reset, expected_actions)).copy()

    sampled = np.ones((num_reset, expected_actions), dtype=np.float64)
    low, high = np.asarray(strength_cfg.multiplier_range, dtype=np.float64).tolist()
    nominal_probability = float(strength_cfg.nominal_probability)
    if curriculum_profile is not None:
        _, low, nominal_probability = curriculum_profile
    anomaly_rows = np.flatnonzero(np.random.uniform(size=(num_reset,)) >= nominal_probability)
    if anomaly_rows.size == 0:
        return sampled
    candidates = np.asarray(strength_cfg.candidate_actuator_indices, dtype=np.int64)
    selected = np.random.choice(candidates, size=anomaly_rows.size, replace=True)
    sampled[anomaly_rows, selected] = np.random.uniform(low, high, size=anomaly_rows.size)
    return sampled

