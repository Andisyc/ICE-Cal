from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.commands import Commands, sample_height_commands
from unilab.envs.locomotion.common.rewards import (
    RewardContext,
    track_base_height_exp_smooth,
    tracking_lin_vel,
)
from unilab.envs.locomotion.g1.joystick import G1WalkEnv


class _FakeBasePosBackend:
    def __init__(self, base_pos: np.ndarray) -> None:
        self._base_pos = np.asarray(base_pos, dtype=get_global_dtype())

    def get_base_pos(self) -> np.ndarray:
        return self._base_pos


def _height_reward_ctx(
    base_height: list[float],
    target_height: float | np.ndarray,
    *,
    sigma: float = 0.25,
) -> RewardContext:
    dtype = get_global_dtype()
    base_height_arr = np.asarray(base_height, dtype=dtype)
    num_envs = base_height_arr.shape[0]
    return RewardContext(
        info={},
        linvel=np.zeros((num_envs, 3), dtype=dtype),
        gyro=np.zeros((num_envs, 3), dtype=dtype),
        dof_pos=np.zeros((num_envs, 0), dtype=dtype),
        num_envs=num_envs,
        tracking_sigma=sigma,
        base_height_target=target_height,
        base_height=base_height_arr,
    )


def _fake_g1_reward_context(
    *,
    info: dict,
    base_pos: np.ndarray,
    fallback_height: float = 0.75,
) -> RewardContext:
    dtype = get_global_dtype()
    env = object.__new__(G1WalkEnv)
    env._num_envs = base_pos.shape[0]
    env._backend = _FakeBasePosBackend(base_pos)
    env.default_angles = np.zeros((0,), dtype=dtype)
    env._pose_weights = None
    env._reward_cfg = SimpleNamespace(tracking_sigma=0.25, base_height_target=fallback_height)
    zeros3 = np.zeros((env._num_envs, 3), dtype=dtype)
    zeros0 = np.zeros((env._num_envs, 0), dtype=dtype)
    return env._build_reward_context(info, zeros3, zeros3, zeros3, zeros0, zeros0)


def _fake_g1_obs_env(*, mode_observation: bool, height_obs: bool) -> G1WalkEnv:
    dtype = get_global_dtype()
    env = object.__new__(G1WalkEnv)
    env._num_envs = 2
    env._num_action = 29
    env.default_angles = np.zeros((29,), dtype=dtype)
    env._cfg = SimpleNamespace(
        mode_observation=mode_observation,
        commands=SimpleNamespace(
            observe_height_command=height_obs,
            default_height=0.754,
        ),
        noise_config=SimpleNamespace(
            level=0.0,
            scale_gyro=0.0,
            scale_gravity=0.0,
            scale_joint_angle=0.0,
            scale_joint_vel=0.0,
        ),
        curriculum=SimpleNamespace(enabled=False),
    )
    env._reward_cfg = SimpleNamespace(
        scales={},
        base_height_target=0.754,
        gait_constraint=SimpleNamespace(
            command_xy_threshold=0.05,
            command_yaw_threshold=0.05,
            enabled=False,
            freeze_phase_in_stand_mode=False,
            stand_phase=[np.pi, np.pi],
        ),
    )
    return env


def _obs_fixture() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype = get_global_dtype()
    info = {
        "commands": np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.1]], dtype=dtype),
        "height_commands": np.asarray([[0.7], [0.8]], dtype=dtype),
        "current_actions": np.zeros((2, 29), dtype=dtype),
        "gait_phase": np.zeros((2, 2), dtype=dtype),
    }
    linvel = np.zeros((2, 3), dtype=dtype)
    gyro = np.zeros((2, 3), dtype=dtype)
    gravity = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=dtype)
    dof_pos = np.zeros((2, 29), dtype=dtype)
    dof_vel = np.zeros((2, 29), dtype=dtype)
    return info, linvel, gyro, gravity, dof_pos, dof_vel


def test_height_command_default_fields_match_contract() -> None:
    cfg = Commands()

    assert len(cfg.height_range) == 2
    assert cfg.height_range[0] <= cfg.default_height <= cfg.height_range[1]
    assert cfg.random_height_during_walking is False
    assert cfg.observe_height_command is False


def test_height_command_default_only_shape_dtype_and_value() -> None:
    height_cmd = sample_height_commands(
        np.random.default_rng(0),
        4,
        [0.2, 0.754],
        default_height=0.754,
        random_height=False,
    )

    assert height_cmd.shape == (4, 1)
    assert height_cmd.dtype == get_global_dtype()
    np.testing.assert_allclose(
        height_cmd,
        np.full((4, 1), 0.754, dtype=get_global_dtype()),
    )


def test_height_command_random_respects_range_and_low_default_high_ordering() -> None:
    low, default, high = 0.2, 0.754, 0.9

    height_cmd = sample_height_commands(
        np.random.default_rng(1),
        32,
        [low, high],
        default_height=default,
        random_height=True,
    )

    assert height_cmd.shape == (32, 1)
    assert height_cmd.dtype == get_global_dtype()
    assert np.all(height_cmd >= low)
    assert np.all(height_cmd <= high)

    fixture = np.asarray([[low], [default], [high]], dtype=get_global_dtype())
    assert fixture[0, 0] < fixture[1, 0] < fixture[2, 0]


def test_height_reward_exp_smooth_orders_target_above_off_target() -> None:
    target = 0.75
    reward = track_base_height_exp_smooth(
        _height_reward_ctx([target - 0.1, target, target + 0.1], target)
    )

    assert reward.shape == (3,)
    assert reward.dtype == get_global_dtype()
    assert np.all(np.isfinite(reward))
    assert reward[1] == 1.0
    assert reward[0] < reward[1]
    assert reward[2] < reward[1]
    np.testing.assert_allclose(reward[0], reward[2])


def test_height_reward_exp_smooth_accepts_per_env_column_target() -> None:
    target = np.asarray([[0.7], [0.8]], dtype=get_global_dtype())
    reward = track_base_height_exp_smooth(_height_reward_ctx([0.7, 0.9], target))

    assert reward.shape == (2,)
    assert reward[0] == 1.0
    assert reward[1] < reward[0]


def test_measured_height_uses_base_pos_z_column() -> None:
    base_pos = np.asarray([[1.0, 2.0, 0.7], [-1.0, 0.5, 0.82]], dtype=get_global_dtype())
    env = object.__new__(G1WalkEnv)
    env._backend = _FakeBasePosBackend(base_pos)

    measured_height = env._terrain_relative_base_height()

    assert base_pos.shape == (2, 3)
    assert measured_height.shape == (2,)
    assert measured_height.dtype == get_global_dtype()
    np.testing.assert_allclose(measured_height, base_pos[:, 2])


def test_reward_context_uses_scalar_height_target_fallback() -> None:
    base_pos = np.asarray([[0.0, 0.0, 0.75], [0.0, 0.0, 0.85]], dtype=get_global_dtype())
    ctx = _fake_g1_reward_context(info={}, base_pos=base_pos, fallback_height=0.75)

    assert ctx.base_height_target == 0.75
    assert ctx.base_height.shape == (2,)
    np.testing.assert_allclose(ctx.base_height, base_pos[:, 2])

    reward = track_base_height_exp_smooth(ctx)
    assert reward[0] == 1.0
    assert reward[1] < reward[0]


def test_reward_context_uses_per_env_height_target_aliases() -> None:
    base_pos = np.asarray([[0.0, 0.0, 0.7], [0.0, 0.0, 1.0]], dtype=get_global_dtype())
    target = np.asarray([[0.7], [0.9]], dtype=get_global_dtype())

    for key in ("height_commands", "commands_height"):
        ctx = _fake_g1_reward_context(
            info={key: target},
            base_pos=base_pos,
            fallback_height=0.75,
        )

        assert ctx.base_height_target.shape == (2,)
        np.testing.assert_allclose(ctx.base_height_target, target[:, 0])
        reward = track_base_height_exp_smooth(ctx)
        np.testing.assert_allclose(
            reward,
            np.exp(-np.square(base_pos[:, 2] - target[:, 0]) / ctx.tracking_sigma).astype(
                get_global_dtype()
            ),
        )


def test_height_obs_old_path_preserves_checkpoint_dims() -> None:
    info, linvel, gyro, gravity, dof_pos, dof_vel = _obs_fixture()

    for mode_observation, expected_obs, expected_critic in [(False, 98, 101), (True, 99, 102)]:
        env = _fake_g1_obs_env(mode_observation=mode_observation, height_obs=False)
        obs = env._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)

        assert env.obs_groups_spec == {"obs": expected_obs, "critic": expected_critic}
        assert obs["obs"].shape == (2, expected_obs)
        assert obs["critic"].shape == (2, expected_critic)


def test_height_obs_appends_target_height_after_velocity_command() -> None:
    info, linvel, gyro, gravity, dof_pos, dof_vel = _obs_fixture()
    env = _fake_g1_obs_env(mode_observation=True, height_obs=True)

    obs = env._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)

    assert env.obs_groups_spec == {"obs": 100, "critic": 103}
    assert obs["obs"].shape == (2, 100)
    assert obs["critic"].shape == (2, 103)

    command_start = 3 + 3 + 29 + 29 + 29
    expected_command_block = np.concatenate(
        [info["commands"], info["height_commands"]], axis=1, dtype=get_global_dtype()
    )
    np.testing.assert_allclose(obs["obs"][:, command_start : command_start + 4], expected_command_block)
    np.testing.assert_allclose(
        obs["critic"][:, command_start : command_start + 4], expected_command_block
    )


def test_height_tracking_config_keeps_original_walking_reward_boundary() -> None:
    with initialize(config_path="../../../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_height/mujoco"])

    scales = OmegaConf.to_container(cfg.reward.scales, resolve=True)

    assert cfg.training.task_name == "G1WalkHeight"
    assert "mode_observation" not in cfg.env
    assert "mode" not in cfg.reward
    assert "gait_constraint" not in cfg.reward
    assert cfg.env.commands.small_xy_threshold == 0.0
    assert "rel_standing_envs" not in cfg.env.commands
    assert "rel_transition_envs" not in cfg.env.commands
    assert scales["tracking_lin_vel"] == 2.0
    assert scales["tracking_ang_vel"] == 1.5
    assert scales["feet_phase"] == 5.0
    assert scales["track_base_height_exp_smooth"] == 4.0


def test_height_command_range_stays_inside_walking_survival_height() -> None:
    with initialize(config_path="../../../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_height/mujoco"])

    height_low, height_high = cfg.env.commands.height_range

    assert height_low >= cfg.reward.min_base_height
    assert cfg.reward.min_base_height <= cfg.env.commands.default_height <= height_high


def test_height_tracking_reward_does_not_change_velocity_ranking_at_same_height() -> None:
    command = np.asarray([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=get_global_dtype())
    ctx = RewardContext(
        info={"commands": command},
        linvel=np.asarray([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=get_global_dtype()),
        gyro=np.zeros((2, 3), dtype=get_global_dtype()),
        dof_pos=np.zeros((2, 0), dtype=get_global_dtype()),
        num_envs=2,
        tracking_sigma=0.25,
        base_height_target=np.asarray([0.7, 0.7], dtype=get_global_dtype()),
        base_height=np.asarray([0.7, 0.7], dtype=get_global_dtype()),
    )

    lin = 2.0 * tracking_lin_vel(ctx)
    height = 4.0 * track_base_height_exp_smooth(ctx)
    total = lin + height

    assert height[0] == height[1]
    assert lin[0] > lin[1]
    assert total[0] > total[1]


def test_height_tracking_scale_stays_below_walking_positive_reward_budget() -> None:
    with initialize(config_path="../../../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_height/mujoco"])

    scales = OmegaConf.to_container(cfg.reward.scales, resolve=True)
    walking_positive = (
        scales["tracking_lin_vel"]
        + scales["tracking_ang_vel"]
        + scales["feet_phase"]
        + scales["alive"]
    )
    height_scale = scales["track_base_height_exp_smooth"]

    assert height_scale < walking_positive
    assert height_scale / walking_positive < 0.25
