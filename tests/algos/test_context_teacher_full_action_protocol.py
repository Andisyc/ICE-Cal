from __future__ import annotations

import copy
from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT_DIR = Path(__file__).resolve().parents[2]


def _metrics(lateral: float, yaw: float, forward: float, fall: float = 0.0) -> dict[str, float]:
    return {
        "max_lateral_abs_m": lateral,
        "max_yaw_abs_rad": yaw,
        "forward_velocity_mae_mps": forward,
        "fall_rate": fall,
        "action_saturation_step_rate": 0.0,
    }


def _report() -> dict:
    return {
        "seeds": [101, 102, 103, 104, 105],
        "pairing_exact_for_all_seeds": True,
        "baseline": {"overall": _metrics(1.0, 1.0, 1.0)},
        "teacher": {"overall": _metrics(0.89, 0.89, 1.02)},
    }


def _contract() -> dict:
    from unilab.algos.torch.fada_context.full_action_formal_protocol import (
        FORMAL_AGGREGATION,
        FORMAL_EVALUATION_NUM_ENVS,
        FORMAL_EVALUATION_SEEDS,
        FORMAL_EVALUATION_STEPS,
        FORMAL_STRENGTH,
        FORMAL_TASK_CONFIG,
    )

    return {
        "task_config": FORMAL_TASK_CONFIG,
        "num_envs_per_seed": FORMAL_EVALUATION_NUM_ENVS,
        "steps": FORMAL_EVALUATION_STEPS,
        "seeds": list(FORMAL_EVALUATION_SEEDS),
        "command": [0.4, 0.0, 0.0],
        "aggregation": FORMAL_AGGREGATION,
        "actuator_strength": list(FORMAL_STRENGTH),
    }


def test_full_action_formal_config_is_fixed_left_knee_only() -> None:
    from unilab.algos.torch.fada_context.full_action_formal_protocol import (
        FORMAL_TASK_CONFIG,
        validate_full_action_formal_training_config,
    )

    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"task={FORMAL_TASK_CONFIG}"])
    result = validate_full_action_formal_training_config(cfg)
    assert result["formal_profile_match"] is True
    assert cfg.algo.runtime_impl == "privileged_full_action_sac"
    assert cfg.algo.actor.nominal_action_anchor_coef == 10.0
    assert float(cfg.algo.actor_lr) == 3e-5
    assert cfg.algo.max_iterations == 1000
    assert cfg.algo.save_interval == 100
    assert list(cfg.env.domain_rand.actuator_strength.multipliers).count(0.9) == 1
    assert cfg.env.domain_rand.actuator_strength.multipliers[3] == 0.9
    assert cfg.env.forward_progress_termination.enabled is True
    assert cfg.env.forward_progress_termination.grace_steps == 50
    assert cfg.env.forward_progress_termination.min_command_forward_speed == 0.1
    assert cfg.env.forward_progress_termination.min_average_forward_speed == 0.2
    assert cfg.reward.straight_line_lateral_tolerance_m == 0.10
    assert cfg.reward.straight_line_yaw_tolerance_rad == 0.10
    assert cfg.reward.scales.penalty_lateral_corridor_violation == -20.0
    assert cfg.reward.scales.penalty_yaw_corridor_violation == -20.0


def test_full_action_quality_gate_compares_only_same_condition_branches() -> None:
    from unilab.algos.torch.fada_context.full_action_formal_protocol import (
        assess_full_action_teacher_quality,
    )

    assert assess_full_action_teacher_quality(_report(), _contract())["quality_status"] == "passed"
    failed = copy.deepcopy(_report())
    failed["teacher"]["overall"]["max_yaw_abs_rad"] = 0.95
    result = assess_full_action_teacher_quality(failed, _contract())
    assert result["quality_status"] == "failed"
    assert "max_yaw_reduction" in result["failed_checks"]
