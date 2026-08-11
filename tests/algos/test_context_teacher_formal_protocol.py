from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

ROOT_DIR = Path(__file__).resolve().parents[2]


def _metric_group(
    *,
    max_lateral: float = 1.0,
    max_yaw: float = 1.0,
    forward_mae: float = 1.0,
    fall_rate: float = 0.005,
    clipping_step_rate: float = 0.0,
) -> dict[str, float]:
    return {
        "final_lateral_abs_m": 1.0,
        "max_lateral_abs_m": max_lateral,
        "final_yaw_abs_rad": 1.0,
        "max_yaw_abs_rad": max_yaw,
        "forward_velocity_mae_mps": forward_mae,
        "lateral_velocity_mae_mps": 1.0,
        "fall_rate": fall_rate,
        "clipping_step_rate": clipping_step_rate,
    }


def _passing_aggregate() -> dict[str, Any]:
    nominal_branch = {
        "by_scenario": {
            "nominal": _metric_group(fall_rate=0.0),
            "left_knee": _metric_group(),
        }
    }
    teacher_branch = {
        "by_scenario": {
            "nominal": {
                **_metric_group(fall_rate=0.0, clipping_step_rate=0.005),
                "final_lateral_abs_m": 1.01,
                "max_lateral_abs_m": 1.01,
                "final_yaw_abs_rad": 1.01,
                "max_yaw_abs_rad": 1.01,
                "forward_velocity_mae_mps": 1.01,
                "lateral_velocity_mae_mps": 1.01,
            },
            "left_knee": _metric_group(
                max_lateral=0.89,
                max_yaw=0.89,
                forward_mae=1.02,
                clipping_step_rate=0.01,
            ),
        }
    }
    return {
        "seed_count": 5,
        "seeds": [101, 102, 103, 104, 105],
        "pairing_exact_for_all_seeds": True,
        "nominal": nominal_branch,
        "teacher": teacher_branch,
    }


def _formal_evaluation_contract() -> dict[str, Any]:
    return {
        "task_config": "sac/g1_walk_flat/mujoco_context_teacher_phase1",
        "num_envs_per_seed": 256,
        "steps": 400,
        "seeds": [101, 102, 103, 104, 105],
        "command": [0.4, 0.0, 0.0],
        "aggregation": "equal_seed_mean_of_per_seed_scenario_means",
        "actuator_strength": {
            "sampling_mode": "single_candidate",
            "candidate_actuator_indices": [3],
            "multiplier_range": [0.9, 0.9],
            "nominal_probability": 0.5,
        },
    }


def test_formal_quality_gate_passes_only_when_every_stratum_check_passes() -> None:
    from unilab.algos.torch.fada_context.formal_protocol import (
        assess_phase1_teacher_quality,
    )

    assessment = assess_phase1_teacher_quality(
        _passing_aggregate(),
        _formal_evaluation_contract(),
    )

    assert assessment["quality_status"] == "passed"
    assert assessment["failed_checks"] == []
    assert assessment["checks"]
    assert all(check["passed"] for check in assessment["checks"])


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda report: report["teacher"]["by_scenario"]["left_knee"].__setitem__(
                "max_lateral_abs_m", 0.91
            ),
            "left_knee.max_lateral_reduction",
        ),
        (
            lambda report: report["teacher"]["by_scenario"]["nominal"].__setitem__(
                "forward_velocity_mae_mps", 1.03
            ),
            "nominal.forward_velocity_mae_mps_non_degradation",
        ),
        (
            lambda report: report["teacher"]["by_scenario"]["left_knee"].__setitem__(
                "clipping_step_rate", 0.02
            ),
            "left_knee.clipping_step_rate",
        ),
        (
            lambda report: report.__setitem__("pairing_exact_for_all_seeds", False),
            "protocol.exact_pairing",
        ),
    ],
)
def test_formal_quality_gate_fails_closed(
    mutation: Any,
    failed_check: str,
) -> None:
    from unilab.algos.torch.fada_context.formal_protocol import (
        assess_phase1_teacher_quality,
    )

    aggregate = _passing_aggregate()
    mutation(aggregate)
    assessment = assess_phase1_teacher_quality(aggregate, _formal_evaluation_contract())

    assert assessment["quality_status"] == "failed"
    assert failed_check in assessment["failed_checks"]


def test_formal_quality_gate_rejects_nonfinite_metric() -> None:
    from unilab.algos.torch.fada_context.formal_protocol import (
        assess_phase1_teacher_quality,
    )

    aggregate = _passing_aggregate()
    aggregate["teacher"]["by_scenario"]["left_knee"]["max_yaw_abs_rad"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        assess_phase1_teacher_quality(aggregate, _formal_evaluation_contract())


def test_formal_quality_gate_cannot_assess_shortened_protocol() -> None:
    from unilab.algos.torch.fada_context.formal_protocol import (
        assess_phase1_teacher_quality,
    )

    contract = _formal_evaluation_contract()
    contract["steps"] = 100

    assessment = assess_phase1_teacher_quality(_passing_aggregate(), contract)

    assert assessment["quality_status"] == "unassessed"
    assert assessment["protocol_mismatches"] == ["steps"]


def test_formal_evaluation_contract_requires_exact_protocol() -> None:
    from unilab.algos.torch.fada_context.formal_protocol import (
        validate_phase1_formal_evaluation_contract,
    )

    manifest = validate_phase1_formal_evaluation_contract(_formal_evaluation_contract())
    assert manifest["formal_protocol_match"] is True
    assert manifest["mismatches"] == []

    drifted = _formal_evaluation_contract()
    drifted["seeds"] = [1, 2, 3]
    manifest = validate_phase1_formal_evaluation_contract(drifted)
    assert manifest["formal_protocol_match"] is False
    assert "seeds" in manifest["mismatches"]


def test_phase1_training_config_matches_frozen_formal_profile() -> None:
    from unilab.algos.torch.fada_context.formal_protocol import (
        validate_phase1_formal_training_config,
    )

    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / "offpolicy"),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=["task=sac/g1_walk_flat/mujoco_context_teacher_phase1"],
        )

    manifest = validate_phase1_formal_training_config(cfg)
    assert manifest["formal_profile_match"] is True
    assert manifest["mismatches"] == []

    drifted = copy.deepcopy(cfg)
    drifted.algo.num_envs = 64
    with pytest.raises(ValueError, match="algo.num_envs"):
        validate_phase1_formal_training_config(drifted)
