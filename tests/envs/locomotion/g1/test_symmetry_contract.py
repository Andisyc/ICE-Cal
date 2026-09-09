from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.privileged_oracle_sac import (
    resolve_privileged_locomotion_sac_runtime,
)
from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.common.commands import zero_small_xy_commands
from unilab.envs.locomotion.g1.fada_privileged import build_g1_fada_privileged_layout
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


def test_fada_fixed_contact_symmetry_mirrors_privileged_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "fada-symmetry-contract")
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"
    ):
        source_cfg = compose(
            config_name="config",
            overrides=["algo=sac", "task=sac/g1_walk_flat/mujoco_fada_source"],
        )
        cfg = compose(
            config_name="config",
            overrides=["algo=sac", "task=sac/g1_walk_flat/mujoco_fada_fixed_contact"],
        )

    assert source_cfg.algo.use_symmetry is False
    assert cfg.algo.use_symmetry is True
    assert cfg.env.commands.small_xy_threshold == pytest.approx(0.4)
    threshold_probe = np.asarray([[0.4, 0.0, 0.3], [0.5, 0.0, 0.3]], dtype=np.float32)
    zero_small_xy_commands(threshold_probe, threshold=cfg.env.commands.small_xy_threshold)
    np.testing.assert_array_equal(threshold_probe[:, 0], [0.0, 0.5])

    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )
    assert runtime.supports_symmetry is True
    runtime.validate_training_config(cfg)

    ensure_registries()
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="sac"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=env_override, sim_backend="mujoco")
    try:
        augmentation = env.build_symmetry_augmentation(device="cpu")
        assert augmentation is not None
        actor_dim = env.obs_groups_spec["obs"]
        critic_dim = env.obs_groups_spec["critic"]
        action_dim = env.action_space.shape[0]
        identity = env.get_fada_privileged_checkpoint_identity()
        privileged_layout = build_g1_fada_privileged_layout(identity.body_names)
        assert critic_dim == actor_dim + privileged_layout.width

        actor_obs = torch.zeros((1, actor_dim))
        command_start = 3 + 3 + action_dim * 3
        actor_obs[0, command_start : command_start + 3] = torch.tensor([0.5, 0.2, 0.3])
        privileged = torch.zeros((1, privileged_layout.width))
        base_velocity = privileged_layout.slice_for("base_linear_velocity")
        contact_resultants = privileged_layout.slice_for("foot_contact_resultants")
        contact_flags = privileged_layout.slice_for("foot_contact_flags")
        base_com_shift = privileged_layout.slice_for("base_com_shift")
        privileged[0, base_velocity] = torch.tensor([0.5, 0.2, 0.0])
        privileged[0, contact_resultants] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        privileged[0, contact_flags] = torch.tensor([1.0, 0.0])
        privileged[0, base_com_shift] = torch.tensor([0.1, 0.2, 0.3])
        critic_obs = torch.cat([actor_obs, privileged], dim=-1)
        actions = torch.arange(action_dim, dtype=torch.float32).unsqueeze(0)

        mirrored_actor = augmentation.mirror_obs(actor_obs, obs_group="obs")
        mirrored_critic = augmentation.mirror_obs(critic_obs, obs_group="critic")
        mirrored_privileged = mirrored_critic[:, actor_dim:]
        torch.testing.assert_close(
            mirrored_actor[0, command_start : command_start + 3],
            torch.tensor([0.5, -0.2, -0.3]),
        )
        torch.testing.assert_close(
            mirrored_privileged[0, base_velocity], torch.tensor([0.5, -0.2, 0.0])
        )
        torch.testing.assert_close(
            mirrored_privileged[0, contact_resultants],
            torch.tensor([4.0, -5.0, 6.0, 1.0, -2.0, 3.0]),
        )
        torch.testing.assert_close(
            mirrored_privileged[0, contact_flags], torch.tensor([0.0, 1.0])
        )
        torch.testing.assert_close(
            mirrored_privileged[0, base_com_shift], torch.tensor([0.1, -0.2, 0.3])
        )
        torch.testing.assert_close(
            augmentation.mirror_obs(mirrored_critic, obs_group="critic"), critic_obs
        )
        torch.testing.assert_close(
            augmentation.mirror_action(augmentation.mirror_action(actions)), actions
        )

        model_kwargs = runtime.build_model_kwargs(
            obs_dim=actor_dim, critic_obs_dim=critic_dim
        )
        learner = runtime.learner_cls(
            obs_dim=actor_dim,
            critic_obs_dim=critic_dim,
            action_dim=action_dim,
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            priv_info_embed_dim=4,
            priv_mlp_hidden_dims=(8, 4),
            num_atoms=5,
            use_layer_norm=False,
            use_compile=False,
            obs_normalization=True,
            priv_info_normalization=True,
            use_symmetry=True,
            symmetry_augmentation=augmentation,
            **{
                key: value
                for key, value in model_kwargs.items()
                if key
                not in {
                    "priv_info_embed_dim",
                    "priv_info_normalization",
                    "priv_mlp_hidden_dims",
                }
            },
        )
        learner.update_critic(
            {
                "obs": actor_obs,
                "critic": critic_obs,
                "actions": torch.zeros((1, action_dim)),
                "rewards": torch.zeros(1),
                "next_obs": actor_obs.clone(),
                "next_critic": critic_obs.clone(),
                "dones": torch.zeros(1),
                "truncated": torch.zeros(1),
            }
        )
        assert float(learner.obs_normalizer.count) == 4.0
        assert float(learner.actor.priv_info_normalizer.count) == 4.0
    finally:
        env.close()
