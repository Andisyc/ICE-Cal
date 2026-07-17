from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.g1.joystick import G1WalkRewardConfig
from unilab.training import BackendAdapter, create_env

pytest.importorskip("mujoco", reason="mujoco is required for G1 symmetry contract tests")

ROOT_DIR = Path(__file__).resolve().parents[4]


def _reward_config() -> G1WalkRewardConfig:
    return G1WalkRewardConfig(
        scales={"tracking_lin_vel": 2.0, "alive": 10.0},
        tracking_sigma=0.25,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        close_feet_threshold=0.15,
        pose_weights=[0.01] * 29,
    )


def test_g1_walk_flat_symmetry_contract_matches_obs_groups():
    ensure_registries()
    env = cast(
        Any,
        registry.make(
            "G1WalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": _reward_config()},
        ),
    )

    try:
        layouts = env.get_symmetry_obs_layouts()
        assert set(layouts) == {"obs", "critic"}
        for group_name, layout in layouts.items():
            assert sum(dim for _, dim in layout) == env.obs_groups_spec[group_name]
    finally:
        env.close()


def test_g1_walk_flat_symmetry_can_augment_critic_group():
    ensure_registries()
    env = cast(
        Any,
        registry.make(
            "G1WalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": _reward_config()},
        ),
    )

    try:
        augmentation = env.build_symmetry_augmentation(device="cpu")
        assert augmentation is not None

        action_dim = env.action_space.shape[0]
        obs = torch.zeros((1, env.obs_groups_spec["obs"]))
        critic = torch.zeros((1, env.obs_groups_spec["critic"]))
        actions = torch.zeros((1, action_dim))

        actor_aug, action_aug = augmentation.augment_obs_and_actions(obs, actions, obs_group="obs")
        critic_aug, critic_action_aug = augmentation.augment_obs_and_actions(
            critic,
            actions,
            obs_group="critic",
        )

        assert actor_aug.shape == (2, env.obs_groups_spec["obs"])
        assert critic_aug.shape == (2, env.obs_groups_spec["critic"])
        assert action_aug.shape == (2, action_dim)
        assert critic_action_aug.shape == (2, action_dim)
    finally:
        env.close()


def test_g1_walk_height_symmetry_keeps_height_command_scalar():
    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_height/mujoco"])
    assert "mode_observation" not in cfg.env
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="sac"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=env_override, sim_backend="mujoco")

    try:
        augmentation = env.build_symmetry_augmentation(device="cpu")
        assert augmentation is not None

        command_start = 3 + 3 + env.action_space.shape[0] * 3
        obs = torch.zeros((1, env.obs_groups_spec["obs"]))
        obs[0, command_start : command_start + 4] = torch.tensor([0.2, 0.1, 0.3, 0.754])

        mirrored = augmentation.mirror_obs(obs, obs_group="obs")

        assert env.obs_groups_spec["obs"] == 99
        torch.testing.assert_close(
            mirrored[0, command_start : command_start + 4],
            torch.tensor([0.2, -0.1, -0.3, 0.754]),
        )
    finally:
        env.close()


def test_g1_stand_still_symmetry_keeps_walking_actor_obs_dim():
    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_stand_still/mujoco"])
    assert "mode_observation" not in cfg.env
    assert "observe_height_command" not in cfg.env.commands
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="sac"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=env_override, sim_backend="mujoco")

    try:
        augmentation = env.build_symmetry_augmentation(device="cpu")
        assert augmentation is not None

        command_start = 3 + 3 + env.action_space.shape[0] * 3
        obs = torch.zeros((1, env.obs_groups_spec["obs"]))
        obs[0, command_start : command_start + 3] = torch.tensor([0.0, 0.0, 0.0])

        mirrored = augmentation.mirror_obs(obs, obs_group="obs")

        assert env.obs_groups_spec["obs"] == 98
        assert env.obs_groups_spec["critic"] == 101
        torch.testing.assert_close(
            mirrored[0, command_start : command_start + 3],
            torch.tensor([0.0, -0.0, -0.0]),
        )
    finally:
        env.close()
