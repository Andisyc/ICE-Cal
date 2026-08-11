"""Accepted formal training and quality protocol for the Phase-1 teacher."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import OmegaConf

FORMAL_TASK_CONFIG = "sac/g1_walk_flat/mujoco_context_teacher_phase1"
FORMAL_EVALUATION_SEEDS = (101, 102, 103, 104, 105)
FORMAL_EVALUATION_NUM_ENVS = 256
FORMAL_EVALUATION_STEPS = 400
FORMAL_EVALUATION_COMMAND = (0.4, 0.0, 0.0)
FORMAL_AGGREGATION = "equal_seed_mean_of_per_seed_scenario_means"
FORMAL_NOMINAL_CHECKPOINT_SHA256 = (
    "db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291"
)

ANOMALY_MIN_ERROR_REDUCTION = 0.10
MAX_RELATIVE_DEGRADATION = 0.02
MAX_FALL_RATE = 0.01
MAX_CLIPPING_STEP_RATE = 0.01

_ANOMALY_SCENARIOS = ("left_knee",)
_NOMINAL_NON_DEGRADATION_METRICS = (
    "final_lateral_abs_m",
    "max_lateral_abs_m",
    "final_yaw_abs_rad",
    "max_yaw_abs_rad",
    "forward_velocity_mae_mps",
    "lateral_velocity_mae_mps",
)
_TRAINING_PROFILE = {
    "algo.runtime_impl": "privileged_residual_sac",
    "algo.actor.nominal_checkpoint_path": "checkpoints/oracles/G1WalkFlat/model_5000.pt",
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
    "env.domain_rand.actuator_strength.sampling_mode": "single_candidate",
    "env.domain_rand.actuator_strength.candidate_actuator_indices": [3],
    "env.domain_rand.actuator_strength.multiplier_range": [0.9, 0.9],
    "env.domain_rand.actuator_strength.nominal_probability": 0.5,
    "env.domain_rand.actuator_strength.include_in_critic_obs": True,
    "reward.scales.penalty_lateral_displacement": -20.0,
    "reward.scales.penalty_yaw_drift": -10.0,
}


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _equal(actual: Any, expected: Any) -> bool:
    actual = _plain(actual)
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return bool(actual == expected)


def formal_quality_thresholds() -> dict[str, float]:
    """Return the accepted numeric gate as JSON-safe values."""
    return {
        "anomaly_min_error_reduction": ANOMALY_MIN_ERROR_REDUCTION,
        "max_relative_degradation": MAX_RELATIVE_DEGRADATION,
        "max_fall_rate": MAX_FALL_RATE,
        "max_clipping_step_rate": MAX_CLIPPING_STEP_RATE,
    }


def validate_phase1_formal_training_config(cfg: Any) -> dict[str, Any]:
    """Fail closed when the composed formal training profile drifts."""
    mismatches: list[str] = []
    observed: dict[str, Any] = {}
    for path, expected in _TRAINING_PROFILE.items():
        actual = OmegaConf.select(cfg, path)
        observed[path] = _plain(actual)
        if not _equal(actual, expected):
            mismatches.append(path)
    if mismatches:
        raise ValueError(
            "Phase-1 formal training profile mismatch at: " + ", ".join(sorted(mismatches))
        )
    return {
        "schema": "unilab_context_teacher_phase1_training_profile_v1",
        "formal_profile_match": True,
        "mismatches": [],
        "observed": observed,
    }


def validate_phase1_formal_evaluation_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Report whether an evaluation invocation matches the accepted formal protocol."""
    expected = {
        "task_config": FORMAL_TASK_CONFIG,
        "num_envs_per_seed": FORMAL_EVALUATION_NUM_ENVS,
        "steps": FORMAL_EVALUATION_STEPS,
        "seeds": list(FORMAL_EVALUATION_SEEDS),
        "command": list(FORMAL_EVALUATION_COMMAND),
        "aggregation": FORMAL_AGGREGATION,
        "actuator_strength.sampling_mode": "single_candidate",
        "actuator_strength.candidate_actuator_indices": [3],
        "actuator_strength.multiplier_range": [0.9, 0.9],
        "actuator_strength.nominal_probability": 0.5,
    }
    mismatches: list[str] = []
    for path, expected_value in expected.items():
        current: Any = contract
        try:
            for part in path.split("."):
                current = current[part]
        except (KeyError, TypeError):
            mismatches.append(path)
            continue
        if not _equal(current, expected_value):
            mismatches.append(path)
    return {
        "schema": "unilab_context_teacher_phase1_evaluation_protocol_v1",
        "formal_protocol_match": not mismatches,
        "mismatches": sorted(mismatches),
        "thresholds": formal_quality_thresholds(),
    }


def _metric_group(aggregate: Mapping[str, Any], branch: str, scenario: str) -> Mapping[str, float]:
    try:
        group = aggregate[branch]["by_scenario"][scenario]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"formal quality report is missing {branch}.{scenario}") from exc
    if not isinstance(group, Mapping):
        raise ValueError(f"formal quality metric group {branch}.{scenario} must be a mapping")
    return group


def _finite_metric(group: Mapping[str, float], metric: str, *, path: str) -> float:
    try:
        value = float(group[metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"formal quality report is missing numeric metric {path}.{metric}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"formal quality metric {path}.{metric} must be finite")
    return value


def assess_phase1_teacher_quality(
    aggregate: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the accepted conjunctive per-stratum quality gate."""
    protocol_validation = validate_phase1_formal_evaluation_contract(evaluation_contract)
    if not protocol_validation["formal_protocol_match"]:
        return {
            "schema": "unilab_context_teacher_phase1_quality_gate_v1",
            "quality_status": "unassessed",
            "thresholds": formal_quality_thresholds(),
            "checks": [],
            "failed_checks": [],
            "reason": "formal evaluation protocol mismatch",
            "protocol_mismatches": protocol_validation["mismatches"],
        }
    checks: list[dict[str, Any]] = []

    def add_check(
        name: str,
        *,
        scenario: str,
        metric: str,
        actual: float | bool,
        limit: float | bool,
        relation: str,
        passed: bool,
    ) -> None:
        checks.append(
            {
                "name": name,
                "scenario": scenario,
                "metric": metric,
                "actual": actual,
                "limit": limit,
                "relation": relation,
                "passed": bool(passed),
            }
        )

    exact_pairing = bool(aggregate.get("pairing_exact_for_all_seeds", False))
    add_check(
        "protocol.exact_pairing",
        scenario="protocol",
        metric="pairing_exact_for_all_seeds",
        actual=exact_pairing,
        limit=True,
        relation="==",
        passed=exact_pairing,
    )
    seeds = tuple(int(seed) for seed in aggregate.get("seeds", ()))
    seed_match = seeds == FORMAL_EVALUATION_SEEDS and int(aggregate.get("seed_count", -1)) == len(
        FORMAL_EVALUATION_SEEDS
    )
    add_check(
        "protocol.held_out_seeds",
        scenario="protocol",
        metric="seeds",
        actual=seed_match,
        limit=True,
        relation="==",
        passed=seed_match,
    )

    for scenario in _ANOMALY_SCENARIOS:
        nominal = _metric_group(aggregate, "nominal", scenario)
        teacher = _metric_group(aggregate, "teacher", scenario)
        for metric, suffix in (
            ("max_lateral_abs_m", "max_lateral_reduction"),
            ("max_yaw_abs_rad", "max_yaw_reduction"),
        ):
            baseline = _finite_metric(nominal, metric, path=f"nominal.{scenario}")
            value = _finite_metric(teacher, metric, path=f"teacher.{scenario}")
            if baseline <= 1e-12:
                reduction = 0.0
                passed = value <= baseline + 1e-12
                relation = ">= or zero-baseline non-inferior"
            else:
                reduction = (baseline - value) / baseline
                passed = reduction + 1e-12 >= ANOMALY_MIN_ERROR_REDUCTION
                relation = ">="
            add_check(
                f"{scenario}.{suffix}",
                scenario=scenario,
                metric=metric,
                actual=float(reduction),
                limit=ANOMALY_MIN_ERROR_REDUCTION,
                relation=relation,
                passed=passed,
            )

        baseline_forward = _finite_metric(
            nominal,
            "forward_velocity_mae_mps",
            path=f"nominal.{scenario}",
        )
        teacher_forward = _finite_metric(
            teacher,
            "forward_velocity_mae_mps",
            path=f"teacher.{scenario}",
        )
        forward_limit = baseline_forward * (1.0 + MAX_RELATIVE_DEGRADATION)
        add_check(
            f"{scenario}.forward_velocity_mae_mps_non_degradation",
            scenario=scenario,
            metric="forward_velocity_mae_mps",
            actual=teacher_forward,
            limit=forward_limit,
            relation="<=",
            passed=teacher_forward <= forward_limit + 1e-12,
        )

        baseline_fall = _finite_metric(nominal, "fall_rate", path=f"nominal.{scenario}")
        teacher_fall = _finite_metric(teacher, "fall_rate", path=f"teacher.{scenario}")
        fall_limit = min(baseline_fall, MAX_FALL_RATE)
        add_check(
            f"{scenario}.fall_rate",
            scenario=scenario,
            metric="fall_rate",
            actual=teacher_fall,
            limit=fall_limit,
            relation="<=",
            passed=teacher_fall <= fall_limit + 1e-12,
        )
        clipping = _finite_metric(
            teacher,
            "clipping_step_rate",
            path=f"teacher.{scenario}",
        )
        add_check(
            f"{scenario}.clipping_step_rate",
            scenario=scenario,
            metric="clipping_step_rate",
            actual=clipping,
            limit=MAX_CLIPPING_STEP_RATE,
            relation="<=",
            passed=clipping <= MAX_CLIPPING_STEP_RATE + 1e-12,
        )

    nominal_baseline = _metric_group(aggregate, "nominal", "nominal")
    nominal_teacher = _metric_group(aggregate, "teacher", "nominal")
    for metric in _NOMINAL_NON_DEGRADATION_METRICS:
        baseline = _finite_metric(nominal_baseline, metric, path="nominal.nominal")
        value = _finite_metric(nominal_teacher, metric, path="teacher.nominal")
        limit = baseline * (1.0 + MAX_RELATIVE_DEGRADATION)
        add_check(
            f"nominal.{metric}_non_degradation",
            scenario="nominal",
            metric=metric,
            actual=value,
            limit=limit,
            relation="<=",
            passed=value <= limit + 1e-12,
        )
    nominal_fall = _finite_metric(nominal_baseline, "fall_rate", path="nominal.nominal")
    teacher_fall = _finite_metric(nominal_teacher, "fall_rate", path="teacher.nominal")
    fall_limit = min(nominal_fall, MAX_FALL_RATE)
    add_check(
        "nominal.fall_rate",
        scenario="nominal",
        metric="fall_rate",
        actual=teacher_fall,
        limit=fall_limit,
        relation="<=",
        passed=teacher_fall <= fall_limit + 1e-12,
    )
    nominal_clipping = _finite_metric(
        nominal_teacher,
        "clipping_step_rate",
        path="teacher.nominal",
    )
    add_check(
        "nominal.clipping_step_rate",
        scenario="nominal",
        metric="clipping_step_rate",
        actual=nominal_clipping,
        limit=MAX_CLIPPING_STEP_RATE,
        relation="<=",
        passed=nominal_clipping <= MAX_CLIPPING_STEP_RATE + 1e-12,
    )

    failed_checks = [str(check["name"]) for check in checks if not check["passed"]]
    return {
        "schema": "unilab_context_teacher_phase1_quality_gate_v1",
        "quality_status": "passed" if not failed_checks else "failed",
        "thresholds": formal_quality_thresholds(),
        "checks": checks,
        "failed_checks": failed_checks,
    }


__all__ = [
    "FORMAL_AGGREGATION",
    "FORMAL_EVALUATION_COMMAND",
    "FORMAL_EVALUATION_NUM_ENVS",
    "FORMAL_EVALUATION_SEEDS",
    "FORMAL_EVALUATION_STEPS",
    "FORMAL_NOMINAL_CHECKPOINT_SHA256",
    "FORMAL_TASK_CONFIG",
    "assess_phase1_teacher_quality",
    "formal_quality_thresholds",
    "validate_phase1_formal_evaluation_contract",
    "validate_phase1_formal_training_config",
]
