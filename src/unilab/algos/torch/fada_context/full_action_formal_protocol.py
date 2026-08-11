"""Formal profile and paired quality gate for the v004 full-action teacher."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf

FORMAL_TASK_CONFIG = "sac/g1_walk_flat/mujoco_context_teacher_full_action_v005"
FORMAL_EVALUATION_SEEDS = (101, 102, 103, 104, 105)
FORMAL_EVALUATION_NUM_ENVS = 256
FORMAL_EVALUATION_STEPS = 400
FORMAL_EVALUATION_COMMAND = (0.4, 0.0, 0.0)
FORMAL_AGGREGATION = "equal_seed_mean"
FORMAL_NOMINAL_CHECKPOINT_SHA256 = (
    "db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291"
)
FORMAL_STRENGTH = tuple(0.9 if index == 3 else 1.0 for index in range(29))

MIN_ERROR_REDUCTION = 0.10
MAX_FORWARD_RELATIVE_DEGRADATION = 0.02
MAX_FALL_RATE = 0.01
MAX_SATURATION_STEP_RATE = 0.01

_TRAINING_PROFILE = {
    "algo.runtime_impl": "privileged_full_action_sac",
    "algo.actor.nominal_initialization_checkpoint": (
        "checkpoints/oracles/G1WalkFlat/model_5000.pt"
    ),
    "algo.actor.priv_info_dim": 29,
    "algo.num_envs": 2048,
    "algo.batch_size": 8192,
    "algo.replay_buffer_n": 512,
    "algo.updates_per_step": 8,
    "algo.learning_starts": 10,
    "algo.policy_frequency": 4,
    "algo.max_iterations": 5000,
    "algo.save_interval": 1000,
    "algo.use_symmetry": False,
    "algo.load_run": "-1",
    "algo.actor_warm_start_checkpoint": None,
    "training.no_play": True,
    "training.no_sync_collection": False,
    "training.env_steps_per_sync": 1,
    "env.commands.vel_limit": [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]],
    "env.curriculum.enabled": False,
    "env.reset_base_qvel_limit": 0.0,
    "env.domain_rand.randomize_kp": False,
    "env.domain_rand.randomize_kd": False,
    "env.domain_rand.actuator_strength.enabled": True,
    "env.domain_rand.actuator_strength.include_in_critic_obs": True,
    "env.domain_rand.actuator_strength.multipliers": list(FORMAL_STRENGTH),
    "env.forward_progress_termination.enabled": True,
    "env.forward_progress_termination.grace_steps": 50,
    "env.forward_progress_termination.min_command_forward_speed": 0.1,
    "env.forward_progress_termination.min_average_forward_speed": 0.2,
    "reward.scales.penalty_lateral_displacement": -20.0,
    "reward.scales.penalty_yaw_drift": -10.0,
}


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _equal(actual: Any, expected: Any) -> bool:
    actual = _plain(actual)
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return bool(actual == expected)


def formal_quality_thresholds() -> dict[str, float]:
    return {
        "min_error_reduction": MIN_ERROR_REDUCTION,
        "max_forward_relative_degradation": MAX_FORWARD_RELATIVE_DEGRADATION,
        "max_fall_rate": MAX_FALL_RATE,
        "max_saturation_step_rate": MAX_SATURATION_STEP_RATE,
    }


def validate_full_action_formal_training_config(cfg: Any) -> dict[str, Any]:
    mismatches: list[str] = []
    observed: dict[str, Any] = {}
    for path, expected in _TRAINING_PROFILE.items():
        actual = OmegaConf.select(cfg, path)
        observed[path] = _plain(actual)
        if not _equal(actual, expected):
            mismatches.append(path)
    if mismatches:
        raise ValueError(
            "Full-action formal training profile mismatch at: " + ", ".join(sorted(mismatches))
        )
    return {
        "schema": "unilab_context_full_action_training_profile_v1",
        "formal_profile_match": True,
        "mismatches": [],
        "observed": observed,
    }


def validate_full_action_formal_evaluation_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "task_config": FORMAL_TASK_CONFIG,
        "num_envs_per_seed": FORMAL_EVALUATION_NUM_ENVS,
        "steps": FORMAL_EVALUATION_STEPS,
        "seeds": list(FORMAL_EVALUATION_SEEDS),
        "command": list(FORMAL_EVALUATION_COMMAND),
        "aggregation": FORMAL_AGGREGATION,
        "actuator_strength": list(FORMAL_STRENGTH),
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if not _equal(contract.get(key), expected_value)
    ]
    return {
        "schema": "unilab_context_full_action_evaluation_protocol_v1",
        "formal_protocol_match": not mismatches,
        "mismatches": sorted(mismatches),
        "thresholds": formal_quality_thresholds(),
    }


def _metric(report: Mapping[str, Any], branch: str, metric: str) -> float:
    try:
        value = float(report[branch]["overall"][metric])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing numeric metric {branch}.overall.{metric}") from error
    if not math.isfinite(value):
        raise ValueError(f"Metric {branch}.overall.{metric} must be finite")
    return value


def assess_full_action_teacher_quality(
    aggregate: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = validate_full_action_formal_evaluation_contract(evaluation_contract)
    if not protocol["formal_protocol_match"]:
        return {
            "schema": "unilab_context_full_action_quality_gate_v1",
            "quality_status": "unassessed",
            "checks": [],
            "failed_checks": [],
            "protocol_mismatches": protocol["mismatches"],
        }

    checks: list[dict[str, Any]] = []

    def add(
        name: str, actual: float | bool, limit: float | bool, relation: str, passed: bool
    ) -> None:
        checks.append(
            {
                "name": name,
                "actual": actual,
                "limit": limit,
                "relation": relation,
                "passed": bool(passed),
            }
        )

    pairing = bool(aggregate.get("pairing_exact_for_all_seeds", False))
    add("protocol.exact_pairing", pairing, True, "==", pairing)
    seeds = tuple(int(seed) for seed in aggregate.get("seeds", ()))
    seed_match = seeds == FORMAL_EVALUATION_SEEDS
    add("protocol.held_out_seeds", seed_match, True, "==", seed_match)

    for metric, name in (
        ("max_lateral_abs_m", "max_lateral_reduction"),
        ("max_yaw_abs_rad", "max_yaw_reduction"),
    ):
        baseline = _metric(aggregate, "baseline", metric)
        teacher = _metric(aggregate, "teacher", metric)
        reduction = (baseline - teacher) / baseline if baseline > 1e-12 else 0.0
        add(name, reduction, MIN_ERROR_REDUCTION, ">=", reduction >= MIN_ERROR_REDUCTION)

    baseline_forward = _metric(aggregate, "baseline", "forward_velocity_mae_mps")
    teacher_forward = _metric(aggregate, "teacher", "forward_velocity_mae_mps")
    forward_limit = baseline_forward * (1.0 + MAX_FORWARD_RELATIVE_DEGRADATION)
    add(
        "forward_velocity_non_degradation",
        teacher_forward,
        forward_limit,
        "<=",
        teacher_forward <= forward_limit,
    )
    baseline_fall = _metric(aggregate, "baseline", "fall_rate")
    teacher_fall = _metric(aggregate, "teacher", "fall_rate")
    fall_limit = min(baseline_fall, MAX_FALL_RATE)
    add("fall_rate", teacher_fall, fall_limit, "<=", teacher_fall <= fall_limit)
    saturation = _metric(aggregate, "teacher", "action_saturation_step_rate")
    add(
        "action_saturation_step_rate",
        saturation,
        MAX_SATURATION_STEP_RATE,
        "<=",
        saturation <= MAX_SATURATION_STEP_RATE,
    )
    failed = [str(check["name"]) for check in checks if not check["passed"]]
    return {
        "schema": "unilab_context_full_action_quality_gate_v1",
        "quality_status": "passed" if not failed else "failed",
        "thresholds": formal_quality_thresholds(),
        "checks": checks,
        "failed_checks": failed,
        "protocol_mismatches": [],
    }


__all__ = [
    "FORMAL_AGGREGATION",
    "FORMAL_EVALUATION_NUM_ENVS",
    "FORMAL_EVALUATION_SEEDS",
    "FORMAL_EVALUATION_STEPS",
    "FORMAL_NOMINAL_CHECKPOINT_SHA256",
    "FORMAL_STRENGTH",
    "FORMAL_TASK_CONFIG",
    "assess_full_action_teacher_quality",
    "formal_quality_thresholds",
    "validate_full_action_formal_evaluation_contract",
    "validate_full_action_formal_training_config",
]
