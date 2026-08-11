"""Same-snapshot comparison of a full-action teacher and the original policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any, cast

import numpy as np
import torch

from unilab.algos.torch.fada_context.full_action_formal_protocol import FORMAL_STRENGTH
from unilab.algos.torch.fada_context.paired_evaluation import (
    _actor_observation,
    _assert_start_match,
    _info_array,
    _initial_frame_displacement,
    _require_state,
    _start_contract,
)
from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _rollout(
    env: Any,
    *,
    policy: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    baseline_policy: Callable[[torch.Tensor], torch.Tensor],
    compare_to_baseline: bool,
    steps: int,
    device: str | torch.device,
) -> dict[str, Any]:
    start = _start_contract(env)
    num_envs = int(start["actor_obs"].shape[0])
    action_dim = int(env.action_space.shape[0])
    initial_position = start["base_pos"]
    initial_yaw = np_yaw_from_quat(start["base_quat"])
    active = np.ones(num_envs, dtype=np.bool_)
    fell = np.zeros(num_envs, dtype=np.float64)
    truncated_rows = np.zeros(num_envs, dtype=np.float64)
    survival = np.zeros(num_envs, dtype=np.float64)
    final_lateral = np.zeros(num_envs, dtype=np.float64)
    max_lateral = np.zeros(num_envs, dtype=np.float64)
    final_yaw = np.zeros(num_envs, dtype=np.float64)
    max_yaw = np.zeros(num_envs, dtype=np.float64)
    final_forward = np.zeros(num_envs, dtype=np.float64)
    forward_velocity_error = np.zeros(num_envs, dtype=np.float64)
    lateral_velocity_error = np.zeros(num_envs, dtype=np.float64)
    action_delta_l2 = np.zeros(num_envs, dtype=np.float64)
    action_delta_linf = np.zeros(num_envs, dtype=np.float64)
    saturation_elements = np.zeros(num_envs, dtype=np.float64)
    saturation_steps = np.zeros(num_envs, dtype=np.float64)

    for _ in range(int(steps)):
        if not np.any(active):
            break
        state = _require_state(env)
        obs_np = _actor_observation(state)
        strength_np = _info_array(state, "privileged_actuator_strength", width=29)
        commands = _info_array(state, "commands", width=3)
        obs = torch.from_numpy(obs_np).to(device)
        strength = torch.from_numpy(strength_np).to(device)
        with torch.inference_mode():
            action = policy(obs, strength)
            reference = baseline_policy(obs) if compare_to_baseline else action
        action_np = action.detach().cpu().numpy().astype(np.float32)
        reference_np = reference.detach().cpu().numpy().astype(np.float32)
        expected = (num_envs, action_dim)
        if action_np.shape != expected or reference_np.shape != expected:
            raise ValueError(f"Full-action paired policy output must have shape {expected}")
        if not np.isfinite(action_np).all():
            raise ValueError("Full-action paired policy emitted non-finite action")

        active_before = active.copy()
        delta = action_np - reference_np
        action_delta_l2[active_before] += np.linalg.norm(delta[active_before], axis=1)
        action_delta_linf[active_before] = np.maximum(
            action_delta_linf[active_before],
            np.max(np.abs(delta[active_before]), axis=1),
        )
        saturated = np.abs(action_np) >= 0.999
        saturation_elements[active_before] += np.sum(saturated[active_before], axis=1)
        saturation_steps[active_before] += np.any(saturated[active_before], axis=1)
        action_np[~active] = 0.0

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
    return {
        "overall": {
            "final_lateral_abs_m": _mean(final_lateral),
            "max_lateral_abs_m": _mean(max_lateral),
            "final_yaw_abs_rad": _mean(final_yaw),
            "max_yaw_abs_rad": _mean(max_yaw),
            "forward_velocity_mae_mps": _mean(forward_velocity_error / denominator),
            "lateral_velocity_mae_mps": _mean(lateral_velocity_error / denominator),
            "forward_progress_m": _mean(final_forward),
            "fall_rate": _mean(fell),
            "truncation_rate": _mean(truncated_rows),
            "survival_steps_mean": _mean(survival),
            "action_delta_l2_mean": _mean(action_delta_l2 / denominator),
            "action_delta_linf_max": float(np.max(action_delta_linf)),
            "action_saturation_element_rate": _mean(
                saturation_elements / (denominator * action_dim)
            ),
            "action_saturation_step_rate": _mean(saturation_steps / denominator),
        }
    }


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overall": {
            key: float(right["overall"][key] - value) for key, value in left["overall"].items()
        }
    }


def evaluate_full_action_paired_rollouts(
    env: Any,
    baseline_actor: Any,
    teacher_actor: Any,
    *,
    steps: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Run both complete policies from one exact fixed-0.9 environment snapshot."""
    if int(steps) <= 0:
        raise ValueError(f"Paired evaluation steps must be positive, got {steps}")
    start = _start_contract(env)
    expected_strength = np.broadcast_to(
        np.asarray(FORMAL_STRENGTH, dtype=np.float32),
        start["actuator_strength"].shape,
    )
    if not np.array_equal(start["actuator_strength"], expected_strength):
        raise ValueError(
            "Full-action paired evaluation requires every row to be fixed left-knee 0.9"
        )
    obs_dim = int(start["actor_obs"].shape[1])
    action_dim = int(env.action_space.shape[0])
    for label, actor in (("baseline", baseline_actor), ("teacher", teacher_actor)):
        if int(getattr(actor, "obs_dim", -1)) != obs_dim:
            raise ValueError(f"{label} actor observation dimension mismatch")
        if int(getattr(actor, "action_dim", -1)) != action_dim:
            raise ValueError(f"{label} actor action dimension mismatch")
    if int(getattr(teacher_actor, "priv_info_dim", -1)) != 29:
        raise ValueError("Teacher actor must consume exactly 29 privileged strength values")

    def baseline_policy(obs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, baseline_actor.explore(obs, deterministic=True))

    def teacher_policy(obs: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            teacher_actor.explore(obs, strength, deterministic=True),
        )

    preserve = getattr(env, "preserve_rollout_state", None)
    if not callable(preserve):
        raise TypeError("Full-action paired evaluation requires env.preserve_rollout_state()")
    with cast(AbstractContextManager[Any], preserve()):
        snapshot = env.capture_rollout_snapshot()
        branch_start = _start_contract(env)
        baseline = _rollout(
            env,
            policy=lambda obs, strength: baseline_policy(obs),
            baseline_policy=baseline_policy,
            compare_to_baseline=False,
            steps=steps,
            device=device,
        )
        env.restore_rollout_snapshot(snapshot)
        _assert_start_match(branch_start, _start_contract(env))
        teacher = _rollout(
            env,
            policy=teacher_policy,
            baseline_policy=baseline_policy,
            compare_to_baseline=True,
            steps=steps,
            device=device,
        )

    return {
        "schema": "unilab_context_full_action_paired_evaluation_v1",
        "steps": int(steps),
        "num_envs": int(start["actor_obs"].shape[0]),
        "actuator_strength": list(FORMAL_STRENGTH),
        "pairing": {"exact_start_match": True, "compared_fields": sorted(branch_start)},
        "baseline": baseline,
        "teacher": teacher,
        "teacher_minus_baseline": _delta(baseline, teacher),
    }


def aggregate_full_action_paired_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("Full-action paired aggregation requires at least one report")

    def average(branch: str) -> dict[str, Any]:
        keys = reports[0][branch]["overall"]
        return {
            "overall": {
                key: float(np.mean([report[branch]["overall"][key] for report in reports]))
                for key in keys
            }
        }

    return {
        "schema": "unilab_context_full_action_paired_aggregate_v1",
        "seed_count": len(reports),
        "seeds": [int(report["seed"]) for report in reports],
        "pairing_exact_for_all_seeds": all(
            bool(report["pairing"]["exact_start_match"]) for report in reports
        ),
        "baseline": average("baseline"),
        "teacher": average("teacher"),
        "teacher_minus_baseline": average("teacher_minus_baseline"),
    }


__all__ = [
    "aggregate_full_action_paired_reports",
    "evaluate_full_action_paired_rollouts",
]
