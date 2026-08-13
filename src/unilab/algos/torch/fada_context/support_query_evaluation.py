"""Closed-loop healthy-reference evaluation for Support-Query Context calibration."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.support_query import (
    FrozenIDMSupportQueryPolicy,
    SupportContextBatch,
)
from unilab.algos.torch.fada_context.support_query_collector import (
    collect_no_context_support,
)
from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat

TRAJECTORY_DISTANCE_METRICS = (
    "actor_observation_mse",
    "base_position_mse_m2",
    "base_yaw_mse_rad2",
    "local_velocity_mse_m2ps2",
    "joint_position_mse_rad2",
    "joint_velocity_mse_rad2ps2",
    "action_mse",
)


@dataclass(frozen=True)
class _TrajectoryTrace:
    actor_observation: np.ndarray
    base_position: np.ndarray
    base_yaw: np.ndarray
    local_velocity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    action: np.ndarray
    state_valid: np.ndarray
    action_valid: np.ndarray
    fell: np.ndarray
    truncated: np.ndarray
    survival_steps: np.ndarray


def _state_matrix(state: Any, carrier_name: str, key: str) -> np.ndarray:
    carrier = getattr(state, carrier_name, None)
    if not isinstance(carrier, Mapping) or key not in carrier:
        available = sorted(carrier) if isinstance(carrier, Mapping) else []
        raise KeyError(f"state.{carrier_name}[{key!r}] missing; available={available}")
    value = np.asarray(carrier[key], dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"state.{carrier_name}[{key!r}] must be finite rank-2")
    return value


def _features(env: Any) -> dict[str, np.ndarray]:
    state = getattr(env, "state", None)
    if state is None:
        raise RuntimeError("closed-loop evaluation requires an initialized environment")
    base_quat = np.asarray(env.get_base_quat(), dtype=np.float32)
    values = {
        "actor_observation": _state_matrix(state, "obs", "obs"),
        "base_position": np.asarray(env.get_base_pos(), dtype=np.float32),
        "base_yaw": np.asarray(np_yaw_from_quat(base_quat), dtype=np.float32)[:, None],
        "local_velocity": np.asarray(env.get_local_linvel(), dtype=np.float32),
        "joint_position": np.asarray(env.get_dof_pos(), dtype=np.float32),
        "joint_velocity": np.asarray(env.get_dof_vel(), dtype=np.float32),
        "command": _state_matrix(state, "info", "commands"),
    }
    rows = int(env.num_envs)
    for name, value in values.items():
        if value.ndim != 2 or value.shape[0] != rows or not np.isfinite(value).all():
            raise ValueError(f"closed-loop {name} must be finite rank-2 with {rows} rows")
    return values


def _strength(env: Any) -> np.ndarray:
    state = getattr(env, "state", None)
    if state is None:
        raise RuntimeError("closed-loop evaluation requires an initialized environment")
    return _state_matrix(state, "info", "privileged_actuator_strength")


def _assert_formal_strengths(healthy_env: Any, fault_env: Any) -> None:
    healthy = _strength(healthy_env)
    fault = _strength(fault_env)
    if healthy.shape != fault.shape or healthy.shape[1] != 29:
        raise ValueError("closed-loop evaluation requires matching 29-actuator strength arrays")
    expected_healthy = np.ones_like(healthy)
    expected_fault = np.ones_like(fault)
    expected_fault[:, 3] = 0.7
    if not np.array_equal(healthy, expected_healthy):
        raise ValueError("healthy reference must use actuator strength 1.0 on every joint")
    if not np.array_equal(fault, expected_fault):
        raise ValueError("fault branch must use only left-knee index 3 strength 0.7")


def _assert_exact_start(healthy_env: Any, fault_env: Any) -> list[str]:
    healthy = _features(healthy_env)
    fault = _features(fault_env)
    compared = sorted(healthy)
    for name in compared:
        if healthy[name].shape != fault[name].shape or not np.array_equal(
            healthy[name], fault[name]
        ):
            difference = float(np.max(np.abs(healthy[name] - fault[name])))
            raise ValueError(
                "healthy/fault paired initial state mismatch: "
                f"field={name} max_abs_difference={difference}"
            )
    return compared


def _rollout(
    env: Any,
    policy: FrozenIDMSupportQueryPolicy,
    delta_z: torch.Tensor,
    *,
    steps: int,
    device: str | torch.device,
) -> _TrajectoryTrace:
    start = _features(env)
    config = policy.config
    rows = int(env.num_envs)
    if start["actor_observation"].shape != (rows, config.obs_dim):
        raise ValueError("environment actor observation does not match FADA checkpoint")
    if start["command"].shape != (rows, config.command_dim):
        raise ValueError("environment command does not match FADA checkpoint")
    expected_delta = (rows, config.hidden_dim)
    if tuple(delta_z.shape) != expected_delta:
        raise ValueError(
            f"closed-loop delta_z shape mismatch: expected={expected_delta} "
            f"observed={tuple(delta_z.shape)}"
        )

    observation = start["actor_observation"]
    command = start["command"]
    observation_history = np.repeat(observation[:, None, :], config.history_length, axis=1)
    action_history = np.zeros((rows, config.history_length, config.action_dim), dtype=np.float32)
    feature_rows = {name: [value.copy()] for name, value in start.items() if name != "command"}
    actions: list[np.ndarray] = []
    state_valid = [np.ones((rows,), dtype=np.bool_)]
    action_valid: list[np.ndarray] = []
    active = np.ones((rows,), dtype=np.bool_)
    fell = np.zeros((rows,), dtype=np.bool_)
    truncated_rows = np.zeros((rows,), dtype=np.bool_)
    survival = np.zeros((rows,), dtype=np.int64)

    for _ in range(steps):
        with torch.inference_mode():
            output = policy.act_with_context(
                torch.as_tensor(observation_history, device=device),
                torch.as_tensor(action_history, device=device),
                torch.as_tensor(command, device=device),
                delta_z,
            )
        action = output.action.detach().cpu().numpy().astype(np.float32)
        if action.shape != (rows, config.action_dim) or not np.isfinite(action).all():
            raise ValueError("closed-loop policy emitted an invalid action")
        active_before = active.copy()
        action[~active_before] = 0.0
        next_state = env.step(action)
        next_features = _features(env)
        changed_command = np.any(next_features["command"] != command, axis=1)
        if np.any(changed_command & active_before):
            raise ValueError("fixed-command closed-loop evaluation observed a command change")

        actions.append(action.copy())
        action_valid.append(active_before)
        for name in feature_rows:
            feature_rows[name].append(next_features[name].copy())
        state_valid.append(active_before)
        survival[active_before] += 1

        terminated = np.asarray(next_state.terminated, dtype=np.bool_).reshape(rows)
        truncated = np.asarray(next_state.truncated, dtype=np.bool_).reshape(rows)
        fell[active_before & terminated] = True
        truncated_rows[active_before & truncated] = True
        active &= ~(terminated | truncated)
        observation = next_features["actor_observation"]
        observation_history = np.concatenate(
            (observation_history[:, 1:], observation[:, None, :]), axis=1
        )
        action_history = np.concatenate((action_history[:, 1:], action[:, None, :]), axis=1)

    return _TrajectoryTrace(
        **{name: np.stack(values) for name, values in feature_rows.items()},
        action=np.stack(actions),
        state_valid=np.stack(state_valid),
        action_valid=np.stack(action_valid),
        fell=fell,
        truncated=truncated_rows,
        survival_steps=survival,
    )


def _masked_mse(
    reference: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    *,
    angular: bool = False,
) -> float:
    if reference.shape != candidate.shape or reference.shape[:2] != valid.shape:
        raise ValueError("trajectory distance shape mismatch")
    difference = candidate - reference
    if angular:
        difference = np_wrap_to_pi(difference)
    selected = np.square(difference)[valid]
    return float(np.mean(selected)) if selected.size else float("inf")


def _distance(reference: _TrajectoryTrace, candidate: _TrajectoryTrace) -> dict[str, float]:
    state_valid = reference.state_valid & candidate.state_valid
    action_valid = reference.action_valid & candidate.action_valid
    return {
        "actor_observation_mse": _masked_mse(
            reference.actor_observation, candidate.actor_observation, state_valid
        ),
        "base_position_mse_m2": _masked_mse(
            reference.base_position, candidate.base_position, state_valid
        ),
        "base_yaw_mse_rad2": _masked_mse(
            reference.base_yaw, candidate.base_yaw, state_valid, angular=True
        ),
        "local_velocity_mse_m2ps2": _masked_mse(
            reference.local_velocity, candidate.local_velocity, state_valid
        ),
        "joint_position_mse_rad2": _masked_mse(
            reference.joint_position, candidate.joint_position, state_valid
        ),
        "joint_velocity_mse_rad2ps2": _masked_mse(
            reference.joint_velocity, candidate.joint_velocity, state_valid
        ),
        "action_mse": _masked_mse(reference.action, candidate.action, action_valid),
        "aligned_state_row_steps": float(np.sum(state_valid)),
        "aligned_action_row_steps": float(np.sum(action_valid)),
    }


def _branch_health(trace: _TrajectoryTrace) -> dict[str, float]:
    return {
        "fall_rate": float(np.mean(trace.fell)),
        "truncation_rate": float(np.mean(trace.truncated)),
        "survival_steps_mean": float(np.mean(trace.survival_steps)),
    }


def _restore(env: Any, snapshot: Any) -> None:
    restore = getattr(env, "restore_rollout_snapshot", None)
    if not callable(restore):
        raise TypeError("closed-loop evaluation requires restore_rollout_snapshot")
    restore(copy.deepcopy(snapshot))


def evaluate_support_query_closed_loop(
    healthy_env: Any,
    fault_env: Any,
    policy: FrozenIDMSupportQueryPolicy,
    support: SupportContextBatch,
    *,
    steps: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Compare fault+zero and fault+Context trajectories to a healthy trajectory."""

    if steps <= 0:
        raise ValueError("closed-loop evaluation steps must be positive")
    rows = int(healthy_env.num_envs)
    if int(fault_env.num_envs) != rows or support.batch_size != rows:
        raise ValueError("healthy, fault, and Support batch sizes must match")
    support.validate(
        policy.config,
        support_length=policy.context_encoder.context_config.support_length,
    )
    set_healthy_autoreset = getattr(healthy_env, "set_autoreset", None)
    set_fault_autoreset = getattr(fault_env, "set_autoreset", None)
    if not callable(set_healthy_autoreset) or not callable(set_fault_autoreset):
        raise TypeError("closed-loop evaluation requires set_autoreset")
    set_healthy_autoreset(False)
    set_fault_autoreset(False)
    _assert_formal_strengths(healthy_env, fault_env)
    compared_fields = _assert_exact_start(healthy_env, fault_env)

    healthy_snapshot = healthy_env.capture_rollout_snapshot()
    fault_snapshot = fault_env.capture_rollout_snapshot()
    policy.eval()
    support_device = support.to(device)
    with torch.inference_mode():
        delta_z = policy.context_encoder(support_device)
    zero_z = torch.zeros_like(delta_z)

    _restore(healthy_env, healthy_snapshot)
    healthy = _rollout(healthy_env, policy, zero_z, steps=steps, device=device)
    _restore(fault_env, fault_snapshot)
    fault_zero = _rollout(fault_env, policy, zero_z, steps=steps, device=device)
    _restore(fault_env, fault_snapshot)
    fault_context = _rollout(fault_env, policy, delta_z, steps=steps, device=device)

    zero_distance = _distance(healthy, fault_zero)
    context_distance = _distance(healthy, fault_context)
    difference = {
        name: float(context_distance[name] - zero_distance[name])
        for name in TRAJECTORY_DISTANCE_METRICS
    }
    improvement_fraction = {
        name: float(
            (zero_distance[name] - context_distance[name]) / max(zero_distance[name], 1e-12)
        )
        for name in TRAJECTORY_DISTANCE_METRICS
    }
    zero_health = _branch_health(fault_zero)
    context_health = _branch_health(fault_context)
    primary_improved = (
        context_distance["actor_observation_mse"] < zero_distance["actor_observation_mse"]
    )
    health_not_worse = (
        context_health["fall_rate"] <= zero_health["fall_rate"]
        and context_health["survival_steps_mean"] >= zero_health["survival_steps_mean"]
    )
    return {
        "schema": "unilab_fada_context_support_query_closed_loop_v1",
        "steps": int(steps),
        "num_envs": rows,
        "pairing": {
            "exact_initial_state_match": True,
            "same_command": [0.4, 0.0, 0.0],
            "compared_fields": compared_fields,
            "healthy_strength": 1.0,
            "fault_joint_index": 3,
            "fault_strength": 0.7,
            "autoreset": False,
        },
        "context": {
            "delta_z_l2_mean": float(torch.linalg.vector_norm(delta_z, dim=1).mean()),
            "delta_z_linf_max": float(delta_z.abs().max()),
        },
        "healthy": _branch_health(healthy),
        "fault_zero": zero_health,
        "fault_context": context_health,
        "fault_zero_distance_to_healthy": zero_distance,
        "fault_context_distance_to_healthy": context_distance,
        "context_minus_zero_distance": difference,
        "context_improvement_fraction": improvement_fraction,
        "verdict": {
            "primary_metric": "actor_observation_mse",
            "primary_metric_improved": primary_improved,
            "fall_and_survival_not_worse": health_not_worse,
            "context_closer_to_healthy": primary_improved and health_not_worse,
            "improved_distance_metric_count": sum(
                context_distance[name] < zero_distance[name] for name in TRAJECTORY_DISTANCE_METRICS
            ),
            "distance_metric_count": len(TRAJECTORY_DISTANCE_METRICS),
        },
    }


def evaluate_online_support_closed_loop(
    healthy_env: Any,
    fault_env: Any,
    policy: FrozenIDMSupportQueryPolicy,
    *,
    steps: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Collect fault Support online, then compare a reset fault+Context rollout."""

    if steps <= 0:
        raise ValueError("closed-loop evaluation steps must be positive")
    rows = int(healthy_env.num_envs)
    if int(fault_env.num_envs) != rows:
        raise ValueError("healthy and fault environment batch sizes must match")
    for env in (healthy_env, fault_env):
        set_autoreset = getattr(env, "set_autoreset", None)
        if not callable(set_autoreset):
            raise TypeError("closed-loop evaluation requires set_autoreset")
        set_autoreset(False)
    _assert_formal_strengths(healthy_env, fault_env)
    compared_fields = _assert_exact_start(healthy_env, fault_env)

    healthy_snapshot = healthy_env.capture_rollout_snapshot()
    fault_snapshot = fault_env.capture_rollout_snapshot()
    policy.eval()
    baseline_policy = FADAPlannerIDMPolicy(
        policy.config,
        planner=policy.planner,
        idm=policy.idm,
    ).to(device).eval()

    _restore(fault_env, fault_snapshot)
    support_initial = fault_env.state
    support = collect_no_context_support(
        fault_env,
        baseline_policy,
        support_initial,
        support_length=policy.context_encoder.context_config.support_length,
    )
    with torch.inference_mode():
        delta_z = policy.context_encoder(support.to(device))
    zero_z = torch.zeros_like(delta_z)

    _restore(healthy_env, healthy_snapshot)
    healthy = _rollout(healthy_env, policy, zero_z, steps=steps, device=device)
    _restore(fault_env, fault_snapshot)
    fault_zero = _rollout(fault_env, policy, zero_z, steps=steps, device=device)
    _restore(fault_env, fault_snapshot)
    fault_context = _rollout(fault_env, policy, delta_z, steps=steps, device=device)

    report = _build_report(
        healthy,
        fault_zero,
        fault_context,
        delta_z,
        steps=steps,
        rows=rows,
        compared_fields=compared_fields,
    )
    report["schema"] = "unilab_fada_context_online_support_closed_loop_v1"
    report["support"] = {
        "source": "same_fault_environment_no_context_rollout",
        "length": support.support_length,
        "reset_before_repaired_rollout": True,
    }
    return report


def _build_report(
    healthy: _TrajectoryTrace,
    fault_zero: _TrajectoryTrace,
    fault_context: _TrajectoryTrace,
    delta_z: torch.Tensor,
    *,
    steps: int,
    rows: int,
    compared_fields: list[str],
) -> dict[str, Any]:
    """Build the shared healthy/fault-zero/fault-Context comparison payload."""

    zero_distance = _distance(healthy, fault_zero)
    context_distance = _distance(healthy, fault_context)
    difference = {
        name: float(context_distance[name] - zero_distance[name])
        for name in TRAJECTORY_DISTANCE_METRICS
    }
    improvement_fraction = {
        name: float(
            (zero_distance[name] - context_distance[name]) / max(zero_distance[name], 1e-12)
        )
        for name in TRAJECTORY_DISTANCE_METRICS
    }
    zero_health = _branch_health(fault_zero)
    context_health = _branch_health(fault_context)
    primary_improved = (
        context_distance["actor_observation_mse"] < zero_distance["actor_observation_mse"]
    )
    health_not_worse = (
        context_health["fall_rate"] <= zero_health["fall_rate"]
        and context_health["survival_steps_mean"] >= zero_health["survival_steps_mean"]
    )
    return {
        "schema": "unilab_fada_context_support_query_closed_loop_v1",
        "steps": int(steps),
        "num_envs": rows,
        "pairing": {
            "exact_initial_state_match": True,
            "same_command": [0.4, 0.0, 0.0],
            "compared_fields": compared_fields,
            "healthy_strength": 1.0,
            "fault_joint_index": 3,
            "fault_strength": 0.7,
            "autoreset": False,
        },
        "context": {
            "delta_z_l2_mean": float(torch.linalg.vector_norm(delta_z, dim=1).mean()),
            "delta_z_linf_max": float(delta_z.abs().max()),
        },
        "healthy": _branch_health(healthy),
        "fault_zero": zero_health,
        "fault_context": context_health,
        "fault_zero_distance_to_healthy": zero_distance,
        "fault_context_distance_to_healthy": context_distance,
        "context_minus_zero_distance": difference,
        "context_improvement_fraction": improvement_fraction,
        "verdict": {
            "primary_metric": "actor_observation_mse",
            "primary_metric_improved": primary_improved,
            "fall_and_survival_not_worse": health_not_worse,
            "context_closer_to_healthy": primary_improved and health_not_worse,
            "improved_distance_metric_count": sum(
                context_distance[name] < zero_distance[name] for name in TRAJECTORY_DISTANCE_METRICS
            ),
            "distance_metric_count": len(TRAJECTORY_DISTANCE_METRICS),
        },
    }


def aggregate_support_query_closed_loop_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("closed-loop aggregation requires at least one report")

    def average_group(name: str) -> dict[str, float]:
        keys = reports[0][name]
        return {
            key: float(np.mean([float(report[name][key]) for report in reports])) for key in keys
        }

    zero_distance = average_group("fault_zero_distance_to_healthy")
    context_distance = average_group("fault_context_distance_to_healthy")
    zero_health = average_group("fault_zero")
    context_health = average_group("fault_context")
    difference = {
        name: context_distance[name] - zero_distance[name] for name in TRAJECTORY_DISTANCE_METRICS
    }
    improvement = {
        name: (zero_distance[name] - context_distance[name]) / max(zero_distance[name], 1e-12)
        for name in TRAJECTORY_DISTANCE_METRICS
    }
    primary_improved = (
        context_distance["actor_observation_mse"] < zero_distance["actor_observation_mse"]
    )
    health_not_worse = (
        context_health["fall_rate"] <= zero_health["fall_rate"]
        and context_health["survival_steps_mean"] >= zero_health["survival_steps_mean"]
    )
    return {
        "schema": "unilab_fada_context_support_query_closed_loop_aggregate_v1",
        "seed_count": len(reports),
        "seeds": [int(report["seed"]) for report in reports],
        "num_envs_total": sum(int(report["num_envs"]) for report in reports),
        "pairing_exact_for_all_seeds": all(
            bool(report["pairing"]["exact_initial_state_match"]) for report in reports
        ),
        "healthy": average_group("healthy"),
        "fault_zero": zero_health,
        "fault_context": context_health,
        "fault_zero_distance_to_healthy": zero_distance,
        "fault_context_distance_to_healthy": context_distance,
        "context_minus_zero_distance": difference,
        "context_improvement_fraction": improvement,
        "context": average_group("context"),
        "verdict": {
            "primary_metric": "actor_observation_mse",
            "primary_metric_improved": primary_improved,
            "fall_and_survival_not_worse": health_not_worse,
            "context_closer_to_healthy": primary_improved and health_not_worse,
            "improved_distance_metric_count": sum(
                context_distance[name] < zero_distance[name] for name in TRAJECTORY_DISTANCE_METRICS
            ),
            "distance_metric_count": len(TRAJECTORY_DISTANCE_METRICS),
        },
    }


__all__ = [
    "TRAJECTORY_DISTANCE_METRICS",
    "aggregate_support_query_closed_loop_reports",
    "evaluate_online_support_closed_loop",
    "evaluate_support_query_closed_loop",
]
