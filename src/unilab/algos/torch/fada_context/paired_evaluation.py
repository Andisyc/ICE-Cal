"""Same-snapshot evaluation for the Phase-1 privileged residual teacher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any, cast

import numpy as np
import torch

from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat

ACTUATOR_STRENGTH_SCENARIOS = ("nominal", "left_knee")
LOWER_IS_BETTER_METRICS = (
    "final_lateral_abs_m",
    "max_lateral_abs_m",
    "final_yaw_abs_rad",
    "max_yaw_abs_rad",
    "forward_velocity_mae_mps",
    "lateral_velocity_mae_mps",
    "fall_rate",
    "truncation_rate",
)


def classify_actuator_strength(strength: np.ndarray) -> np.ndarray:
    """Classify the exact Phase-1 nominal and fixed left-knee-0.9 rows."""
    values = np.asarray(strength, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 29:
        raise ValueError(
            f"paired evaluation requires actuator strength shape (N, 29), got {values.shape}"
        )
    if not np.isfinite(values).all() or np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("paired evaluation actuator strength must be finite and in (0, 1]")

    labels = np.full((values.shape[0],), "other", dtype="<U16")
    changed = ~np.isclose(values, 1.0)
    nominal = ~np.any(changed, axis=1)
    left = changed[:, 3] & np.isclose(values[:, 3], 0.9) & (np.sum(changed, axis=1) == 1)
    labels[nominal] = "nominal"
    labels[left] = "left_knee"
    return labels


def _require_state(env: Any) -> Any:
    state = getattr(env, "state", None)
    if state is None:
        raise RuntimeError("paired evaluation requires an initialized environment state")
    return state


def _actor_observation(state: Any) -> np.ndarray:
    obs = getattr(state, "obs", None)
    if not isinstance(obs, Mapping) or "obs" not in obs:
        raise ValueError("paired evaluation requires state.obs['obs']")
    values = np.asarray(obs["obs"], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"paired actor observation must be rank 2, got {values.shape}")
    return values


def _info_array(state: Any, key: str, *, width: int) -> np.ndarray:
    info = getattr(state, "info", None)
    if not isinstance(info, Mapping) or key not in info:
        raise ValueError(f"paired evaluation requires state.info[{key!r}]")
    values = np.asarray(info[key], dtype=np.float32)
    expected = (_actor_observation(state).shape[0], width)
    if values.shape != expected:
        raise ValueError(
            f"paired evaluation state.info[{key!r}] must have shape {expected}, got {values.shape}"
        )
    return values


def _start_contract(env: Any) -> dict[str, np.ndarray]:
    state = _require_state(env)
    return {
        "actor_obs": _actor_observation(state).copy(),
        "commands": _info_array(state, "commands", width=3).copy(),
        "actuator_strength": _info_array(
            state,
            "privileged_actuator_strength",
            width=29,
        ).copy(),
        "base_pos": np.asarray(env.get_base_pos(), dtype=np.float32).copy(),
        "base_quat": np.asarray(env.get_base_quat(), dtype=np.float32).copy(),
        "local_linvel": np.asarray(env.get_local_linvel(), dtype=np.float32).copy(),
    }


def _assert_start_match(
    expected: Mapping[str, np.ndarray], actual: Mapping[str, np.ndarray]
) -> None:
    for key in expected:
        if key not in actual or not np.array_equal(expected[key], actual[key]):
            raise ValueError(f"paired branch start mismatch at {key}")


def _initial_frame_displacement(
    position: np.ndarray,
    initial_position: np.ndarray,
    initial_yaw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = position[:, :2] - initial_position[:, :2]
    cos_yaw = np.cos(initial_yaw)
    sin_yaw = np.sin(initial_yaw)
    forward = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    lateral = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    return forward, lateral


def _mean_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    return float(np.mean(selected)) if selected.size else 0.0


def _max_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    return float(np.max(selected)) if selected.size else 0.0


def _summarize_rows(rows: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    return {
        "final_lateral_abs_m": _mean_or_zero(rows["final_lateral_abs_m"], mask),
        "max_lateral_abs_m": _mean_or_zero(rows["max_lateral_abs_m"], mask),
        "final_yaw_abs_rad": _mean_or_zero(rows["final_yaw_abs_rad"], mask),
        "max_yaw_abs_rad": _mean_or_zero(rows["max_yaw_abs_rad"], mask),
        "forward_velocity_mae_mps": _mean_or_zero(rows["forward_velocity_mae_mps"], mask),
        "lateral_velocity_mae_mps": _mean_or_zero(rows["lateral_velocity_mae_mps"], mask),
        "forward_progress_m": _mean_or_zero(rows["forward_progress_m"], mask),
        "fall_rate": _mean_or_zero(rows["fell"], mask),
        "truncation_rate": _mean_or_zero(rows["truncated"], mask),
        "survival_steps_mean": _mean_or_zero(rows["survival_steps"], mask),
        "residual_l2_mean": _mean_or_zero(rows["residual_l2_mean"], mask),
        "residual_linf_max": _max_or_zero(rows["residual_linf_max"], mask),
        "clipping_element_rate": _mean_or_zero(rows["clipping_element_rate"], mask),
        "clipping_step_rate": _mean_or_zero(rows["clipping_step_rate"], mask),
    }


def _rollout_branch(
    env: Any,
    actor: Any,
    *,
    use_residual: bool,
    steps: int,
    device: str | torch.device,
    scenario_labels: np.ndarray,
) -> dict[str, Any]:
    start = _start_contract(env)
    num_envs = int(start["actor_obs"].shape[0])
    if start["base_pos"].shape != (num_envs, 3):
        raise ValueError(f"paired base position must have shape ({num_envs}, 3)")
    if start["base_quat"].shape != (num_envs, 4):
        raise ValueError(f"paired base quaternion must have shape ({num_envs}, 4)")
    if start["local_linvel"].shape != (num_envs, 3):
        raise ValueError(f"paired local linear velocity must have shape ({num_envs}, 3)")

    initial_position = start["base_pos"]
    initial_yaw = np_yaw_from_quat(start["base_quat"])
    active = np.ones((num_envs,), dtype=np.bool_)
    fell = np.zeros((num_envs,), dtype=np.float64)
    truncated_rows = np.zeros((num_envs,), dtype=np.float64)
    survival = np.zeros((num_envs,), dtype=np.float64)
    final_lateral = np.zeros((num_envs,), dtype=np.float64)
    max_lateral = np.zeros((num_envs,), dtype=np.float64)
    final_yaw = np.zeros((num_envs,), dtype=np.float64)
    max_yaw = np.zeros((num_envs,), dtype=np.float64)
    final_forward = np.zeros((num_envs,), dtype=np.float64)
    forward_velocity_error = np.zeros((num_envs,), dtype=np.float64)
    lateral_velocity_error = np.zeros((num_envs,), dtype=np.float64)
    residual_l2_sum = np.zeros((num_envs,), dtype=np.float64)
    residual_linf_max = np.zeros((num_envs,), dtype=np.float64)
    clipping_elements = np.zeros((num_envs,), dtype=np.float64)
    clipping_steps = np.zeros((num_envs,), dtype=np.float64)
    action_dim = int(actor.action_dim)

    for _ in range(steps):
        if not np.any(active):
            break
        state = _require_state(env)
        obs_np = _actor_observation(state)
        strength_np = _info_array(state, "privileged_actuator_strength", width=29)
        commands = _info_array(state, "commands", width=3)
        obs = torch.from_numpy(obs_np).to(device)
        strength = torch.from_numpy(strength_np).to(device)
        with torch.inference_mode():
            nominal = actor.nominal_action(obs)
            delta = (
                actor.residual_action(obs, strength, deterministic=True)
                if use_residual
                else torch.zeros_like(nominal)
            )
            raw_action = nominal + delta
            action = actor.fuse_action(nominal, delta)
        nominal_np = nominal.detach().cpu().numpy().astype(np.float32)
        delta_np = delta.detach().cpu().numpy().astype(np.float32)
        raw_np = raw_action.detach().cpu().numpy().astype(np.float32)
        action_np = action.detach().cpu().numpy().astype(np.float32)
        expected_shape = (num_envs, action_dim)
        if any(value.shape != expected_shape for value in (nominal_np, delta_np, action_np)):
            raise ValueError(f"paired actor action contract requires shape {expected_shape}")
        action_np[~active] = 0.0

        active_before = active.copy()
        residual_l2_sum[active_before] += np.linalg.norm(delta_np[active_before], axis=1)
        residual_linf_max[active_before] = np.maximum(
            residual_linf_max[active_before],
            np.max(np.abs(delta_np[active_before]), axis=1),
        )
        clipped = np.abs(raw_np) > 1.0
        clipping_elements[active_before] += np.sum(clipped[active_before], axis=1)
        clipping_steps[active_before] += np.any(clipped[active_before], axis=1)

        next_state = env.step(action_np)
        position = np.asarray(env.get_base_pos(), dtype=np.float32)
        yaw = np_yaw_from_quat(np.asarray(env.get_base_quat(), dtype=np.float32))
        local_linvel = np.asarray(env.get_local_linvel(), dtype=np.float32)
        forward, lateral = _initial_frame_displacement(position, initial_position, initial_yaw)
        yaw_drift = np_wrap_to_pi(yaw - initial_yaw)
        lateral_abs = np.abs(lateral)
        yaw_abs = np.abs(yaw_drift)
        final_forward[active_before] = forward[active_before]
        final_lateral[active_before] = lateral_abs[active_before]
        max_lateral[active_before] = np.maximum(
            max_lateral[active_before], lateral_abs[active_before]
        )
        final_yaw[active_before] = yaw_abs[active_before]
        max_yaw[active_before] = np.maximum(max_yaw[active_before], yaw_abs[active_before])
        forward_velocity_error[active_before] += np.abs(
            local_linvel[active_before, 0] - commands[active_before, 0]
        )
        lateral_velocity_error[active_before] += np.abs(
            local_linvel[active_before, 1] - commands[active_before, 1]
        )
        survival[active_before] += 1.0

        terminated = np.asarray(next_state.terminated, dtype=np.bool_).reshape(num_envs)
        truncated = np.asarray(next_state.truncated, dtype=np.bool_).reshape(num_envs)
        fell[active_before & terminated] = 1.0
        truncated_rows[active_before & truncated] = 1.0
        active &= ~(terminated | truncated)

    denominator = np.maximum(survival, 1.0)
    rows = {
        "final_lateral_abs_m": final_lateral,
        "max_lateral_abs_m": max_lateral,
        "final_yaw_abs_rad": final_yaw,
        "max_yaw_abs_rad": max_yaw,
        "forward_velocity_mae_mps": forward_velocity_error / denominator,
        "lateral_velocity_mae_mps": lateral_velocity_error / denominator,
        "forward_progress_m": final_forward,
        "fell": fell,
        "truncated": truncated_rows,
        "survival_steps": survival,
        "residual_l2_mean": residual_l2_sum / denominator,
        "residual_linf_max": residual_linf_max,
        "clipping_element_rate": clipping_elements / (denominator * action_dim),
        "clipping_step_rate": clipping_steps / denominator,
    }
    return {
        "overall": _summarize_rows(rows, np.ones((num_envs,), dtype=np.bool_)),
        "by_scenario": {
            scenario: _summarize_rows(rows, scenario_labels == scenario)
            for scenario in ACTUATOR_STRENGTH_SCENARIOS
        },
    }


def _metric_delta(
    nominal: Mapping[str, Any],
    teacher: Mapping[str, Any],
    *,
    improvement: bool,
) -> dict[str, Any]:
    def _one(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
        keys = LOWER_IS_BETTER_METRICS if improvement else tuple(left.keys())
        return {
            key: float(left[key] - right[key] if improvement else right[key] - left[key])
            for key in keys
        }

    return {
        "overall": _one(nominal["overall"], teacher["overall"]),
        "by_scenario": {
            scenario: _one(
                nominal["by_scenario"][scenario],
                teacher["by_scenario"][scenario],
            )
            for scenario in ACTUATOR_STRENGTH_SCENARIOS
        },
    }


def evaluate_paired_rollouts(
    env: Any,
    actor: Any,
    *,
    steps: int,
    device: str | torch.device,
    required_scenarios: Sequence[str] = ACTUATOR_STRENGTH_SCENARIOS,
) -> dict[str, Any]:
    """Evaluate nominal and residual branches from one exact environment snapshot."""
    if int(steps) <= 0:
        raise ValueError(f"paired evaluation steps must be positive, got {steps}")
    if int(getattr(actor, "priv_info_dim", -1)) != 29:
        raise ValueError("paired evaluation requires actor.priv_info_dim=29")
    start = _start_contract(env)
    if int(getattr(actor, "obs_dim", -1)) != int(start["actor_obs"].shape[1]):
        raise ValueError("paired evaluation actor observation dimension mismatch")
    if int(getattr(actor, "action_dim", -1)) != int(env.action_space.shape[0]):
        raise ValueError("paired evaluation actor action dimension mismatch")

    labels = classify_actuator_strength(start["actuator_strength"])
    unrecognized_count = int(np.count_nonzero(labels == "other"))
    if unrecognized_count:
        raise ValueError(
            "paired evaluation found "
            f"{unrecognized_count} actuator-strength rows outside the exact Phase-1 profile"
        )
    counts = {
        scenario: int(np.count_nonzero(labels == scenario)) for scenario in required_scenarios
    }
    missing = [scenario for scenario, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"missing actuator-strength scenarios: {', '.join(missing)}")

    preserve = getattr(env, "preserve_rollout_state", None)
    if not callable(preserve):
        raise TypeError("paired evaluation requires env.preserve_rollout_state()")
    rollout_context = cast(AbstractContextManager[Any], preserve())
    with rollout_context:
        branch_snapshot = env.capture_rollout_snapshot()
        branch_start = _start_contract(env)
        nominal = _rollout_branch(
            env,
            actor,
            use_residual=False,
            steps=int(steps),
            device=device,
            scenario_labels=labels,
        )
        env.restore_rollout_snapshot(branch_snapshot)
        restored_start = _start_contract(env)
        _assert_start_match(branch_start, restored_start)
        teacher = _rollout_branch(
            env,
            actor,
            use_residual=True,
            steps=int(steps),
            device=device,
            scenario_labels=labels,
        )

    return {
        "schema": "unilab_context_teacher_paired_evaluation_v1",
        "steps": int(steps),
        "num_envs": int(start["actor_obs"].shape[0]),
        "scenario_counts": dict(sorted(counts.items())),
        "pairing": {
            "exact_start_match": True,
            "autoreset_during_branches": False,
            "compared_fields": sorted(branch_start),
        },
        "nominal": nominal,
        "teacher": teacher,
        "teacher_minus_nominal": _metric_delta(nominal, teacher, improvement=False),
        "improvement_lower_is_better": _metric_delta(nominal, teacher, improvement=True),
    }


def _average_metric_groups(reports: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    def _average(path: tuple[str, ...]) -> dict[str, float]:
        first: Mapping[str, float] = reports[0][key]
        for part in path:
            first = first[part]  # type: ignore[assignment]
        return {
            metric: float(
                np.mean([_nested_metric_group(report[key], path)[metric] for report in reports])
            )
            for metric in first
        }

    return {
        "overall": _average(("overall",)),
        "by_scenario": {
            scenario: _average(("by_scenario", scenario))
            for scenario in ACTUATOR_STRENGTH_SCENARIOS
        },
    }


def _nested_metric_group(root: Mapping[str, Any], path: tuple[str, ...]) -> Mapping[str, float]:
    current: Mapping[str, Any] = root
    for part in path:
        current = current[part]
    return current


def aggregate_paired_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average seed reports while preserving scenario-resolved measurements."""
    if not reports:
        raise ValueError("paired evaluation aggregation requires at least one report")
    seeds = [int(report["seed"]) for report in reports]
    scenario_counts = {
        scenario: int(sum(int(report["scenario_counts"][scenario]) for report in reports))
        for scenario in ACTUATOR_STRENGTH_SCENARIOS
    }
    return {
        "schema": "unilab_context_teacher_paired_evaluation_aggregate_v1",
        "seed_count": len(reports),
        "seeds": seeds,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "pairing_exact_for_all_seeds": all(
            bool(report["pairing"]["exact_start_match"]) for report in reports
        ),
        "nominal": _average_metric_groups(reports, "nominal"),
        "teacher": _average_metric_groups(reports, "teacher"),
        "teacher_minus_nominal": _average_metric_groups(reports, "teacher_minus_nominal"),
        "improvement_lower_is_better": _average_metric_groups(
            reports,
            "improvement_lower_is_better",
        ),
    }


__all__ = [
    "ACTUATOR_STRENGTH_SCENARIOS",
    "LOWER_IS_BETTER_METRICS",
    "aggregate_paired_reports",
    "classify_actuator_strength",
    "evaluate_paired_rollouts",
]
