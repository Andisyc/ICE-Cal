from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf


def _slope_cfg():
    return OmegaConf.create(
        {
            "training": {"task_name": "G1WalkFlat", "sim_backend": "mujoco"},
            "target_domain": {
                "target_domain_id": "g1_slope_15_mujoco",
                "kind": "slope",
                "task": "sac/g1_walk_flat/mujoco_fada_slope_15",
                "task_name": "G1WalkFlat",
                "backend": "mujoco",
                "command_sequence": [[0.75, 0.0, 0.0], [0.8, 0.0, 0.0], [0.85, 0.0, 0.0]],
                "slope": {
                    "angle_deg": 15.0,
                    "width_m": 0.8,
                    "approach_length_m": 1.5,
                    "surface_length_m": 8.0,
                    "entry_margin_m": 0.25,
                    "finish_margin_m": 0.5,
                },
            },
        }
    )


def test_resolves_exact_slope_target_domain() -> None:
    from unilab.algos.torch.distill.fada.target_domain import resolve_fada_target_domain

    domain = resolve_fada_target_domain(_slope_cfg())

    assert domain.target_domain_id == "g1_slope_15_mujoco"
    assert domain.kind == "slope"
    assert domain.command_sequence == ((0.75, 0.0, 0.0), (0.8, 0.0, 0.0), (0.85, 0.0, 0.0))
    assert domain.slope is not None
    assert domain.slope.angle_deg == 15.0
    assert domain.slope.width_m == 0.8
    assert domain.actuator_index is None


def test_slope_geometry_owns_entry_finish_and_foot_exit() -> None:
    from unilab.algos.torch.distill.fada.target_domain import resolve_fada_target_domain

    geometry = resolve_fada_target_domain(_slope_cfg()).slope
    assert geometry is not None
    angle = np.deg2rad(15.0)

    def point(s: float, lateral: float = 0.0) -> np.ndarray:
        return np.array([1.5 + np.cos(angle) * s, lateral, np.sin(angle) * s])

    assert not geometry.has_entered(point(0.24), np.stack((point(0.1), point(0.1))))
    assert geometry.has_entered(point(0.25), np.stack((point(0.0), point(0.0))))
    assert geometry.has_finished(point(7.5))
    assert not geometry.has_finished(point(7.49))
    assert not geometry.foot_exited(np.stack((point(1.0, 0.4), point(1.0, -0.4))))
    assert geometry.foot_exited(np.stack((point(1.0, 0.401), point(1.0))))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda cfg: cfg.target_domain.slope.pop("width_m"), "slope"),
        (lambda cfg: cfg.target_domain.command_sequence.__setitem__(0, [0.8, 0.1, 0.0]), "lateral"),
        (lambda cfg: cfg.target_domain.__setitem__("actuator_index", 3), "actuator"),
        (lambda cfg: cfg.target_domain.__setitem__("kind", "payload"), "kind"),
        (lambda cfg: cfg.__setitem__("fault", {"name": "legacy"}), "both"),
    ],
)
def test_target_domain_rejects_ambiguous_or_incomplete_configs(mutation, match: str) -> None:
    from unilab.algos.torch.distill.fada.target_domain import resolve_fada_target_domain

    cfg = _slope_cfg()
    mutation(cfg)
    with pytest.raises(ValueError, match=match):
        resolve_fada_target_domain(cfg)


def test_legacy_fault_converts_only_at_explicit_boundary() -> None:
    from unilab.algos.torch.distill.fada.target_domain import resolve_fada_target_domain

    cfg = OmegaConf.create(
        {
            "fault": {
                "name": "right_knee_090",
                "task": "sac/g1_walk_flat/mujoco_fada_target",
                "task_name": "G1WalkFlat",
                "backend": "mujoco",
                "fault_profile": "right_knee_strength_0.9",
                "command_limit": [[0.8, 0.0, 0.0], [0.8, 0.0, 0.0]],
                "actuator_index": 9,
                "actuator_strength": 0.9,
                "actuator_count": 29,
            }
        }
    )

    domain = resolve_fada_target_domain(cfg)

    assert domain.kind == "actuator_gain"
    assert domain.target_domain_id == "right_knee_090"
    assert domain.actuator_index == 9
    assert domain.command_sequence == ((0.8, 0.0, 0.0),)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("env.noise_config.level", 0.1),
        ("env.curriculum.enabled", True),
        ("env.control_config.simulate_action_latency", True),
        ("env.domain_rand.randomize_reset_pose", True),
        ("env.domain_rand.randomize_base_mass", True),
        ("env.domain_rand.randomize_body_mass", True),
        ("env.domain_rand.random_com", True),
        ("env.domain_rand.randomize_gravity", True),
        ("env.domain_rand.randomize_ground_friction", True),
        ("env.domain_rand.randomize_dof_armature", True),
        ("env.domain_rand.push_robots", True),
        ("env.domain_rand.randomize_kp", True),
        ("env.domain_rand.randomize_kd", True),
        ("env.domain_rand.randomize_dof_position_bias", True),
        ("env.domain_rand.randomize_control_delay", True),
        ("env.domain_rand.torque_rfi_fraction", 0.01),
        ("env.domain_rand.actuator_strength.enabled", True),
    ],
)
def test_nominal_slope_environment_rejects_each_nonnominal_field(path: str, value: object) -> None:
    from unilab.algos.torch.distill.fada.target_domain import (
        assert_nominal_slope_environment,
        resolve_fada_target_domain,
    )

    cfg = _slope_cfg()
    cfg.env = OmegaConf.create(
        {
            "curriculum": {"enabled": False},
            "scene": {"model_file": "src/unilab/assets/robots/g1/scene_slope_15.xml"},
            "noise_config": {"level": 0.0},
            "control_config": {"simulate_action_latency": False},
            "domain_rand": {
                "randomize_reset_pose": False,
                "randomize_base_mass": False,
                "randomize_body_mass": False,
                "random_com": False,
                "randomize_gravity": False,
                "randomize_ground_friction": False,
                "randomize_dof_armature": False,
                "push_robots": False,
                "randomize_kp": False,
                "randomize_kd": False,
                "randomize_dof_position_bias": False,
                "randomize_control_delay": False,
                "torque_rfi_fraction": 0.0,
                "actuator_strength": {"enabled": False, "multipliers": None},
            },
        }
    )
    OmegaConf.update(cfg, path, value, merge=False)

    with pytest.raises(ValueError, match=path):
        assert_nominal_slope_environment(
            cfg,
            resolve_fada_target_domain(cfg),
            task_choice="sac/g1_walk_flat/mujoco_fada_slope_15",
        )


@pytest.mark.parametrize(
    "field",
    [
        "angle_deg",
        "width_m",
        "approach_length_m",
        "surface_length_m",
        "entry_margin_m",
        "finish_margin_m",
    ],
)
def test_nominal_slope_environment_rejects_geometry_drift(field: str) -> None:
    from unilab.algos.torch.distill.fada.target_domain import (
        assert_nominal_slope_environment,
        resolve_fada_target_domain,
    )

    cfg = _slope_cfg()
    cfg.env = OmegaConf.create(
        {
            "curriculum": {"enabled": False},
            "scene": {"model_file": "src/unilab/assets/robots/g1/scene_slope_15.xml"},
            "noise_config": {"level": 0.0},
            "control_config": {"simulate_action_latency": False},
            "domain_rand": {
                "randomize_reset_pose": False,
                "randomize_base_mass": False,
                "randomize_body_mass": False,
                "random_com": False,
                "randomize_gravity": False,
                "randomize_ground_friction": False,
                "randomize_dof_armature": False,
                "push_robots": False,
                "randomize_kp": False,
                "randomize_kd": False,
                "randomize_dof_position_bias": False,
                "randomize_control_delay": False,
                "torque_rfi_fraction": 0.0,
                "actuator_strength": {"enabled": False, "multipliers": []},
            },
        }
    )
    OmegaConf.update(cfg, f"target_domain.slope.{field}", 1.0, merge=False)
    with pytest.raises(ValueError, match="canonical"):
        assert_nominal_slope_environment(
            cfg,
            resolve_fada_target_domain(cfg),
            task_choice="sac/g1_walk_flat/mujoco_fada_slope_15",
        )
