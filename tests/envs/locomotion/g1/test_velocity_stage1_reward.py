from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.envs.locomotion.common.rewards import RewardContext, tracking_ang_vel, tracking_lin_vel
from unilab.envs.locomotion.g1.walk_config import G1RewardConfig


def test_split_tracking_preserves_legacy_and_yaw():
    ctx = RewardContext(
        info={"commands": np.tile([0.1, 0, 0.1], (3, 1))},
        linvel=np.array([[0, 0, 0], [0.05, 0, 0], [0.1, 0, 0]]),
        gyro=np.zeros((3, 3)),
        dof_pos=np.zeros((3, 29)),
    )
    np.testing.assert_allclose(tracking_lin_vel(ctx), np.exp([-0.04, -0.01, 0]))
    yaw = tracking_ang_vel(ctx).copy()
    ctx.tracking_lin_vel_sigma = 0.04
    np.testing.assert_allclose(tracking_lin_vel(ctx), np.exp([-0.25, -0.0625, 0]))
    np.testing.assert_array_equal(tracking_ang_vel(ctx), yaw)
    ctx.tracking_ang_vel_sigma = 0.04
    np.testing.assert_allclose(tracking_ang_vel(ctx), np.exp(np.full(3, -0.25)))
    ctx.info["commands"] *= -1
    ctx.linvel *= -1
    np.testing.assert_allclose(tracking_lin_vel(ctx), np.exp([-0.25, -0.0625, 0]))
    ctx.info["commands"][:] = 0
    assert np.all(np.diff(tracking_lin_vel(ctx)) < 0)


def test_stage1_config_admission_and_preservation(monkeypatch):
    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        resolve_privileged_locomotion_sac_runtime,
    )
    from unilab.training.backend_adapter import BackendAdapter

    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "velocity-stage1-test")
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(config_dir=str(root / "conf/offpolicy"), version_base="1.3"):
        parent = compose(
            config_name="config",
            overrides=[
                "task=sac/g1_walk_flat/mujoco_fada_privileged_oracle_original_height_grouped_dr_lineage"
            ],
        )
        cfg = compose(
            config_name="config",
            overrides=["task=sac/g1_walk_flat/mujoco_fada_privileged_oracle_velocity_stage1"],
        )
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )
    runtime.validate_training_config(cfg)
    assert cfg.env == parent.env
    assert cfg.reward.pose_weights == parent.reward.pose_weights
    assert cfg.reward.scales.feet_phase == parent.reward.scales.feet_phase
    assert cfg.reward.scales.penalty_action_rate == -1.0
    override = BackendAdapter(cfg, root_dir=root).build_task_env_cfg_override()
    reward = G1RewardConfig(**override["reward_config"])
    assert reward.tracking_lin_vel_sigma == 0.04
    assert reward.tracking_ang_vel_sigma == 0.25
    from types import SimpleNamespace

    from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings

    owner = SimpleNamespace(
        _reward_cfg=reward,
        _num_envs=1,
        default_angles=np.zeros(29),
        _pose_weights=np.array(reward.pose_weights),
        _terrain_relative_base_height=lambda: np.array([0.754]),
    )
    ctx = G1WalkRewardBindings._build_reward_context(
        owner,
        {"commands": np.array([[0.1, 0, 0.1]])},
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        np.array([[0, 0, -1]]),
        np.zeros((1, 29)),
        np.zeros((1, 29)),
    )
    np.testing.assert_allclose(tracking_lin_vel(ctx), [np.exp(-0.25)])
    np.testing.assert_allclose(tracking_ang_vel(ctx), [np.exp(-0.04)])
    for field in ("tracking_lin_vel_sigma", "tracking_ang_vel_sigma"):
        for invalid in (0, -1, float("nan"), float("inf")):
            with pytest.raises(ValueError, match=field):
                G1RewardConfig(**{**override["reward_config"], field: invalid})
