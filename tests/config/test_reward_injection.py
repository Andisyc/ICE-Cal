"""Test reward config injection system."""

from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf


def test_reward_config_loading_g1():
    """Test G1 SAC reward config loads correctly."""
    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_flat/mujoco"])
        assert hasattr(cfg, "reward")
        assert cfg.reward.scales.tracking_lin_vel == 2.0
        assert cfg.reward.scales.feet_phase == 5.0
        assert cfg.reward.scales.alive == 10.0
        assert cfg.reward.scales.pose == -0.5
        assert cfg.reward.scales.penalty_action_rate == -4.0
        assert list(cfg.reward.scales.keys()) == [
            "tracking_lin_vel",
            "tracking_ang_vel",
            "penalty_ang_vel_xy",
            "penalty_orientation",
            "penalty_action_rate",
            "pose",
            "penalty_feet_ori",
            "feet_phase",
            "alive",
        ]
        assert cfg.reward.tracking_sigma == 0.25
        assert cfg.reward.base_height_target == 0.754
        assert "gait_constraint" not in cfg.reward
        assert "mode" not in cfg.reward
        assert "commands" not in cfg.env
        assert "mode_observation" not in cfg.env
        assert cfg.interactive.action_mode == "policy"
        assert cfg.interactive.keyboard is True
        assert cfg.reward.pose_weights[2] == 5.0
        assert cfg.reward.pose_weights[8] == 5.0


def test_offpolicy_g1_env_override_preserves_upstream_walking_contract():
    """Default G1WalkFlat should stay on the upstream walking reward contract."""
    from pathlib import Path

    from unilab.training import BackendAdapter

    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_flat/mujoco"])

    override = BackendAdapter(cfg, root_dir=Path.cwd(), algo_name="sac").build_task_env_cfg_override()

    assert "commands" not in override
    assert "mode_observation" not in override
    assert "standing_reset_base_qvel_limit" not in override
    assert "mode" not in override["reward_config"]
    assert "gait_constraint" not in override["reward_config"]
    assert list(override["reward_config"]["scales"].keys()) == [
        "tracking_lin_vel",
        "tracking_ang_vel",
        "penalty_ang_vel_xy",
        "penalty_orientation",
        "penalty_action_rate",
        "pose",
        "penalty_feet_ori",
        "feet_phase",
        "alive",
    ]


def test_g1_height_sac_config_preserves_g1_walk_flat_checkpoint_contract():
    """Old SAC G1WalkFlat config must not gain height-conditioned fields."""
    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_flat/mujoco"])

    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert "mode_observation" not in cfg.env
    assert "commands" not in cfg.env
    assert "track_base_height_exp_smooth" not in cfg.reward.scales


def test_g1_height_sac_config_exposes_explicit_height_fields():
    """New SAC G1 height config is the explicit config boundary for height tracking."""
    from pathlib import Path

    from unilab.training import BackendAdapter

    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_height/mujoco"])

    assert cfg.training.task_name == "G1WalkHeight"
    assert cfg.training.sim_backend == "mujoco"
    assert "mode_observation" not in cfg.env
    assert cfg.env.commands.height_range == [0.2, 0.754]
    assert cfg.env.commands.default_height == 0.754
    assert cfg.env.commands.random_height_during_walking is True
    assert cfg.env.commands.observe_height_command is True
    assert cfg.reward.scales.track_base_height_exp_smooth == 4.0
    assert cfg.reward.base_height_target == 0.754

    override = BackendAdapter(cfg, root_dir=Path.cwd(), algo_name="sac").build_task_env_cfg_override()
    assert override["commands"]["height_range"] == [0.2, 0.754]
    assert override["commands"]["default_height"] == 0.754
    assert override["commands"]["random_height_during_walking"] is True
    assert override["commands"]["observe_height_command"] is True
    assert override["reward_config"]["scales"]["track_base_height_exp_smooth"] == 4.0


def test_offpolicy_g1_action_authority_ablation_does_not_enable_standing_path():
    """A standing-only ablation must not turn on mode observation by default."""
    from pathlib import Path

    from unilab.training import BackendAdapter

    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=sac/g1_walk_flat/mujoco",
                "+env.stand_action_authority=false",
            ],
        )

    override = BackendAdapter(cfg, root_dir=Path.cwd(), algo_name="sac").build_task_env_cfg_override()

    assert cfg.env.stand_action_authority is False
    assert "mode_observation" not in cfg.env
    assert "mode_observation" not in override
    assert override["stand_action_authority"] is False


def test_offpolicy_g1_standing_reward_is_explicit_stage_contract():
    """Standing reward belongs to the explicit mixed-mode stage, not default Walking."""
    from pathlib import Path

    from unilab.training import BackendAdapter

    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=sac/g1_walk_flat/mujoco",
            ],
        )

    override = BackendAdapter(cfg, root_dir=Path.cwd(), algo_name="sac").build_task_env_cfg_override()

    assert "mode" not in cfg.reward
    assert "mode" not in override["reward_config"]

    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        enabled_cfg = compose(
            config_name="config",
            overrides=[
                "task=sac/g1_walk_flat/mujoco",
                "+g1_walk_stage=mixed_mode",
            ],
        )

    enabled_override = BackendAdapter(
        enabled_cfg, root_dir=Path.cwd(), algo_name="sac"
    ).build_task_env_cfg_override()
    assert enabled_cfg.env.mode_observation is True
    assert enabled_cfg.reward.mode.standing_enabled is True
    assert enabled_cfg.reward.gait_constraint.enabled is True
    assert enabled_cfg.env.commands.rel_standing_envs == 0.3
    assert enabled_cfg.env.commands.rel_transition_envs == 0.2
    assert enabled_override["reward_config"]["mode"]["standing_enabled"] is True
    assert enabled_override["reward_config"]["gait_constraint"]["enabled"] is True


@pytest.mark.parametrize(
    (
        "stage",
        "standing_frac",
        "transition_frac",
        "vel_limit",
        "transition_vel_limit",
        "reset_qvel",
        "standing_reset_qvel",
        "resampling_time",
        "curriculum_enabled",
        "mode_enabled",
    ),
    [
        (
            "standing_sanity",
            1.0,
            0.0,
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.05, -0.05, -0.15], [0.25, 0.05, 0.15]],
            0.0,
            0.0,
            2.0,
            False,
            True,
        ),
        (
            "walking_sanity",
            0.0,
            0.0,
            [[-0.2, -0.1, -0.2], [0.4, 0.1, 0.2]],
            [[0.05, -0.05, -0.15], [0.25, 0.05, 0.15]],
            0.5,
            0.0,
            2.0,
            True,
            False,
        ),
        (
            "mixed_mode",
            0.3,
            0.2,
            [[-0.3, -0.2, -0.4], [0.8, 0.2, 0.4]],
            [[0.05, -0.05, -0.15], [0.25, 0.05, 0.15]],
            0.5,
            0.5,
            2.0,
            True,
            True,
        ),
    ],
)
def test_offpolicy_g1_training_stage_configs_reach_env_override(
    stage: str,
    standing_frac: float,
    transition_frac: float,
    vel_limit: list[list[float]],
    transition_vel_limit: list[list[float]],
    reset_qvel: float,
    standing_reset_qvel: float,
    resampling_time: float,
    curriculum_enabled: bool,
    mode_enabled: bool,
):
    """G1 standing/walking curriculum stages are env-owner config fragments."""
    from pathlib import Path

    from unilab.training import BackendAdapter

    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=sac/g1_walk_flat/mujoco",
                f"+g1_walk_stage={stage}",
            ],
        )

    override = BackendAdapter(cfg, root_dir=Path.cwd(), algo_name="sac").build_task_env_cfg_override()

    if mode_enabled:
        assert override["mode_observation"] is True
        assert override["reward_config"]["mode"]["enabled"] is True
        assert override["reward_config"]["gait_constraint"]["enabled"] is True
    else:
        assert "mode_observation" not in override
        assert "mode" not in override["reward_config"]
        assert "gait_constraint" not in override["reward_config"]
    assert override["stand_action_authority"] is False
    assert override["standing_reset_base_qvel_limit"] == standing_reset_qvel
    assert override["reset_base_qvel_limit"] == reset_qvel
    assert override["commands"]["resampling_time"] == resampling_time
    assert override["commands"]["rel_standing_envs"] == standing_frac
    assert override["commands"]["rel_transition_envs"] == transition_frac
    assert override["commands"]["vel_limit"] == vel_limit
    assert override["commands"]["transition_vel_limit"] == transition_vel_limit
    assert override["commands"]["small_xy_threshold"] == 0.0
    assert override["curriculum"]["enabled"] is curriculum_enabled


def test_reward_config_loading_g1_motrix():
    """Test G1 Motrix reward config loads correctly."""
    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_flat/motrix"])
        assert hasattr(cfg, "reward")
        assert cfg.reward.scales.tracking_lin_vel == 2.2
        assert cfg.reward.scales.alive == 12.0


def test_resolve_reward_dict_reads_task_reward():
    """Task-backend configs should expose the final reward mapping directly."""
    from unilab.training.reward import resolve_reward_dict

    with initialize(config_path="../../conf/ppo", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["task=go2_joystick_flat/motrix"],
        )

    reward_dict = resolve_reward_dict(cfg)

    assert reward_dict["scales"]["tracking_lin_vel"] == 1.0
    assert reward_dict["scales"]["tracking_ang_vel"] == 0.2


def test_reward_config_conversion():
    """Test reward config converts to dataclasses via registry."""
    from unilab.base import registry
    from unilab.base.registry import ensure_registries

    ensure_registries()

    # Test G1 walk config - registry auto-converts dict to G1WalkRewardConfig
    g1_dict = {
        "scales": {"tracking_lin_vel": 2.0, "alive": 10.0},
        "tracking_sigma": 0.25,
        "base_height_target": 0.754,
        "gait_frequency": 1.5,
        "feet_phase_swing_height": 0.09,
        "feet_phase_tracking_sigma": 0.008,
        "min_base_height": 0.3,
        "max_tilt_deg": 65.0,
        "close_feet_threshold": 0.15,
        "pose_weights": [0.01] * 29,
    }
    env = cast(
        Any,
        registry.make(
            "G1WalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": g1_dict},
        ),
    )
    assert hasattr(env._cfg.reward_config, "scales")
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 2.0
    env.close()

    # Test Go1 config - registry auto-converts dict to RewardConfig
    go1_dict = {
        "scales": {"tracking_lin_vel": 1.0, "base_height": -100.0},
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
    }
    env = cast(
        Any,
        registry.make(
            "Go1JoystickFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": go1_dict},
        ),
    )
    assert hasattr(env._cfg.reward_config, "scales")
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 1.0
    env.close()
