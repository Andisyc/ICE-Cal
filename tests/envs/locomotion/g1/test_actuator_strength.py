from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from unilab.base.registry import ensure_registries
from unilab.dr import DomainRandomizationCapabilities
from unilab.dr.types import RESET_TERM_KD, RESET_TERM_KP
from unilab.envs.common.rotation import np_yaw_to_quat
from unilab.envs.locomotion.g1.joystick import (
    G1ActuatorStrengthConfig,
    G1DomainRandConfig,
    G1WalkDomainRandomizationProvider,
    G1WalkEnv,
)
from unilab.training import BackendAdapter, create_env

ROOT_DIR = Path(__file__).resolve().parents[4]


class _Spawn:
    def apply_spawn(self, env_ids, qpos_xyz, *, yaw=None):
        return qpos_xyz

    def record_episode_start(self, env_ids, qpos_xyz) -> None:
        pass


def _fake_env(strength: G1ActuatorStrengthConfig):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            commands=SimpleNamespace(
                heading_command=False,
                vel_limit=[[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]],
                small_xy_threshold=0.0,
                rel_standing_envs=0.0,
                rel_transition_envs=0.0,
            ),
            gait_phase_init_mode="offset_phase",
            reset_base_qvel_limit=0.0,
            standing_reset_base_qvel_limit=0.0,
            reward_config=SimpleNamespace(gait_constraint=None),
            domain_rand=G1DomainRandConfig(
                randomize_kp=False,
                randomize_kd=False,
                actuator_strength=strength,
            ),
        ),
        _init_qpos=np.zeros((36,), dtype=np.float32),
        _init_qvel=np.zeros((35,), dtype=np.float32),
        _spawn=_Spawn(),
        _num_action=29,
    )


def _capabilities() -> DomainRandomizationCapabilities:
    return DomainRandomizationCapabilities(
        supported_reset_terms=frozenset({RESET_TERM_KP, RESET_TERM_KD})
    )


def test_g1_actuator_strength_off_preserves_empty_gain_payload() -> None:
    base_kp = np.linspace(10.0, 38.0, 29)
    base_kd = np.linspace(1.0, 3.8, 29)
    provider = G1WalkDomainRandomizationProvider(base_kp=base_kp, base_kd=base_kd)
    env = _fake_env(G1ActuatorStrengthConfig())

    provider.validate(env, _capabilities())
    plan = provider.build_reset_plan(env, np.asarray([0], dtype=np.int32))

    assert plan.randomization is None
    assert "privileged_actuator_strength" not in plan.info_updates


def test_g1_actuator_strength_scales_only_selected_joint_gains() -> None:
    base_kp = np.linspace(10.0, 38.0, 29)
    base_kd = np.linspace(1.0, 3.8, 29)
    multipliers = np.ones((29,), dtype=np.float64)
    multipliers[3] = 0.9
    provider = G1WalkDomainRandomizationProvider(base_kp=base_kp, base_kd=base_kd)
    env = _fake_env(G1ActuatorStrengthConfig(enabled=True, multipliers=multipliers.tolist()))

    provider.validate(env, _capabilities())
    plan = provider.build_reset_plan(env, np.asarray([0, 1], dtype=np.int32))

    assert plan.randomization is not None
    expected_kp = np.broadcast_to(base_kp * multipliers, (2, 29))
    expected_kd = np.broadcast_to(base_kd * multipliers, (2, 29))
    np.testing.assert_allclose(plan.randomization.kp, expected_kp)
    np.testing.assert_allclose(plan.randomization.kd, expected_kd)
    np.testing.assert_allclose(
        plan.info_updates["privileged_actuator_strength"],
        np.broadcast_to(multipliers, (2, 29)),
    )


def test_g1_actuator_strength_rejects_incomplete_vector() -> None:
    provider = G1WalkDomainRandomizationProvider(
        base_kp=np.ones((29,), dtype=np.float64),
        base_kd=np.ones((29,), dtype=np.float64),
    )
    env = _fake_env(G1ActuatorStrengthConfig(enabled=True, multipliers=[1.0] * 28))

    with pytest.raises(ValueError, match="exactly 29 multipliers"):
        provider.validate(env, _capabilities())


def test_g1_actuator_strength_samples_nominal_and_bilateral_knee_cases() -> None:
    provider = G1WalkDomainRandomizationProvider(
        base_kp=np.ones((29,), dtype=np.float64),
        base_kd=np.ones((29,), dtype=np.float64),
    )
    env = _fake_env(
        G1ActuatorStrengthConfig(
            enabled=True,
            sampling_mode="single_candidate",
            candidate_actuator_indices=[3, 9],
            multiplier_range=[0.9, 0.9],
            nominal_probability=0.25,
        )
    )

    provider.validate(env, _capabilities())
    plan = provider.build_reset_plan(env, np.arange(512, dtype=np.int32))
    sampled = plan.info_updates["privileged_actuator_strength"]

    assert sampled.shape == (512, 29)
    assert np.all(np.sum(sampled != 1.0, axis=1) <= 1)
    assert np.any(np.all(sampled == 1.0, axis=1))
    assert np.any(np.isclose(sampled[:, 3], 0.9))
    assert np.any(np.isclose(sampled[:, 9], 0.9))
    np.testing.assert_allclose(plan.randomization.kp, sampled)
    np.testing.assert_allclose(plan.randomization.kd, sampled)


def _observation_only_env(*, include_privileged_strength: bool) -> G1WalkEnv:
    env = object.__new__(G1WalkEnv)
    env._cfg = SimpleNamespace(
        mode_observation=False,
        commands=SimpleNamespace(observe_height_command=False),
        noise_config=SimpleNamespace(
            scale_gyro=0.0,
            scale_gravity=0.0,
            scale_joint_angle=0.0,
            scale_joint_vel=0.0,
        ),
        domain_rand=SimpleNamespace(
            actuator_strength=G1ActuatorStrengthConfig(
                enabled=include_privileged_strength,
                include_in_critic_obs=include_privileged_strength,
                multipliers=[1.0] * 29 if include_privileged_strength else [],
            )
        ),
    )
    env.default_angles = np.zeros((29,), dtype=np.float32)
    env._num_action = 29
    env._obs_noise = lambda value, scale: value
    env._command_observation = lambda info, num_envs: info["commands"]
    env._gait_phase_for_observation = lambda info: info["gait_phase"]
    env._mode_observation = lambda info: np.zeros((len(info["commands"]), 0), dtype=np.float32)
    env._uses_walk_observation_profile = lambda: True
    return env


@pytest.mark.parametrize(
    ("include_privileged_strength", "critic_dim"),
    [(False, 101), (True, 130)],
)
def test_g1_actuator_strength_is_critic_only_when_explicitly_enabled(
    include_privileged_strength: bool,
    critic_dim: int,
) -> None:
    env = _observation_only_env(include_privileged_strength=include_privileged_strength)
    batch = 2
    info = {
        "commands": np.zeros((batch, 3), dtype=np.float32),
        "gait_phase": np.zeros((batch, 2), dtype=np.float32),
        "current_actions": np.zeros((batch, 29), dtype=np.float32),
    }
    strength = np.full((batch, 29), 0.9, dtype=np.float32)
    if include_privileged_strength:
        info["privileged_actuator_strength"] = strength

    obs = env._compute_obs(
        info,
        np.zeros((batch, 3), dtype=np.float32),
        np.zeros((batch, 3), dtype=np.float32),
        np.zeros((batch, 3), dtype=np.float32),
        np.zeros((batch, 29), dtype=np.float32),
        np.zeros((batch, 29), dtype=np.float32),
    )

    assert env.obs_groups_spec == {"obs": 98, "critic": critic_dim}
    assert obs["obs"].shape == (batch, 98)
    assert obs["critic"].shape == (batch, critic_dim)
    if include_privileged_strength:
        np.testing.assert_allclose(obs["critic"][:, -29:], strength)


def test_teacher_left_knee_strength_profile_is_explicit_and_isolated() -> None:
    with initialize(config_path="../../../../conf/offpolicy", version_base="1.3"):
        base_cfg = compose(config_name="config", overrides=["task=sac/g1_walk_flat/mujoco"])
        strength_cfg = compose(
            config_name="config",
            overrides=["task=sac/g1_walk_flat/mujoco_left_knee_090"],
        )

    assert OmegaConf.select(base_cfg, "env.domain_rand.actuator_strength") is None
    assert strength_cfg.env.domain_rand.actuator_strength.enabled is True
    assert len(strength_cfg.env.domain_rand.actuator_strength.multipliers) == 29
    assert strength_cfg.env.domain_rand.actuator_strength.multipliers[3] == pytest.approx(0.9)
    assert (
        sum(
            value != pytest.approx(1.0)
            for value in strength_cfg.env.domain_rand.actuator_strength.multipliers
        )
        == 1
    )
    assert strength_cfg.env.domain_rand.randomize_kp is False
    assert strength_cfg.env.domain_rand.randomize_kd is False
    assert strength_cfg.env.commands.vel_limit == [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]
    assert strength_cfg.interactive.keyboard is False


def test_context_teacher_phase1_profile_uses_nominal_and_fixed_left_knee_only() -> None:
    with initialize(config_path="../../../../conf/offpolicy", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["task=sac/g1_walk_flat/mujoco_context_teacher_phase1"],
        )

    strength = cfg.env.domain_rand.actuator_strength
    assert cfg.algo.runtime_impl == "privileged_residual_sac"
    assert cfg.algo.use_symmetry is False
    assert cfg.algo.actor.priv_info_dim == 29
    assert cfg.algo.actor.residual_scale == pytest.approx(0.2)
    assert strength.enabled is True
    assert strength.sampling_mode == "single_candidate"
    assert strength.candidate_actuator_indices == [3]
    assert strength.multiplier_range == [0.9, 0.9]
    assert strength.nominal_probability == pytest.approx(0.5)
    assert strength.include_in_critic_obs is True
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False
    assert cfg.env.commands.vel_limit == [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]
    assert cfg.reward.scales.penalty_lateral_displacement == pytest.approx(-20.0)
    assert cfg.reward.scales.penalty_yaw_drift == pytest.approx(-10.0)


def test_trajectory_precision_reward_records_episode_frame_only_when_enabled() -> None:
    provider = G1WalkDomainRandomizationProvider()
    env = _fake_env(G1ActuatorStrengthConfig())
    env._init_qpos[3] = 1.0

    disabled = provider.build_reset_plan(env, np.asarray([0], dtype=np.int32))
    assert "episode_start_base_pos" not in disabled.info_updates
    assert "episode_start_base_yaw" not in disabled.info_updates

    env.cfg.reward_config.scales = {"penalty_lateral_displacement": -20.0}
    enabled = provider.build_reset_plan(env, np.asarray([0], dtype=np.int32))
    np.testing.assert_allclose(enabled.info_updates["episode_start_base_pos"], enabled.qpos[:, :3])
    assert enabled.info_updates["episode_start_base_yaw"].shape == (1,)


def test_trajectory_precision_rewards_use_episode_start_yaw_frame() -> None:
    env = object.__new__(G1WalkEnv)
    env._num_envs = 2
    initial_position = np.asarray([[1.0, 2.0, 0.75], [-1.0, 3.0, 0.75]], dtype=np.float32)
    initial_yaw = np.asarray([np.pi / 2.0, 0.0], dtype=np.float32)
    current_position = initial_position.copy()
    current_position[0, 0] += 0.2
    current_position[1, 1] -= 0.3
    current_yaw = initial_yaw + np.asarray([0.1, -0.2], dtype=np.float32)
    env.get_base_pos = lambda: current_position
    env.get_base_quat = lambda: np_yaw_to_quat(current_yaw)
    ctx = SimpleNamespace(
        info={
            "episode_start_base_pos": initial_position,
            "episode_start_base_yaw": initial_yaw,
        }
    )

    np.testing.assert_allclose(
        env._reward_lateral_displacement(ctx),
        np.asarray([0.04, 0.09], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        env._reward_yaw_drift(ctx),
        np.asarray([0.01, 0.04], dtype=np.float32),
        atol=1e-6,
    )


def test_trajectory_corridor_penalty_is_zero_inside_and_quadratic_outside() -> None:
    error = np.asarray([-0.20, -0.10, 0.0, 0.15, 0.30], dtype=np.float32)
    np.testing.assert_allclose(
        G1WalkEnv._normalized_corridor_violation(error, 0.10),
        np.asarray([1.0, 0.0, 0.0, 0.25, 4.0], dtype=np.float32),
        atol=1e-6,
    )


def test_forward_progress_failure_uses_reset_yaw_and_exact_grace_boundary() -> None:
    from unilab.envs.locomotion.g1.joystick import compute_forward_progress_failure

    initial = np.asarray([[1.0, 2.0, 0.75], [-1.0, 3.0, 0.75]], dtype=np.float32)
    yaw = np.asarray([np.pi / 2.0, 0.0], dtype=np.float32)
    current = initial.copy()
    current[0, 1] += 0.19
    current[1, 0] += 0.20
    commands = np.asarray([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32)

    before, speed_before = compute_forward_progress_failure(
        current,
        initial,
        yaw,
        np.asarray([48, 48], dtype=np.uint32),
        commands,
        ctrl_dt=0.02,
        grace_steps=50,
        min_command_forward_speed=0.1,
        min_average_forward_speed=0.2,
    )
    at_boundary, speed_at_boundary = compute_forward_progress_failure(
        current,
        initial,
        yaw,
        np.asarray([49, 49], dtype=np.uint32),
        commands,
        ctrl_dt=0.02,
        grace_steps=50,
        min_command_forward_speed=0.1,
        min_average_forward_speed=0.2,
    )

    np.testing.assert_array_equal(before, [False, False])
    np.testing.assert_allclose(speed_before, [0.19 / 0.98, 0.20 / 0.98], atol=1e-6)
    np.testing.assert_array_equal(at_boundary, [True, False])
    np.testing.assert_allclose(speed_at_boundary, [0.19, 0.20], atol=1e-6)


def test_forward_progress_failure_ignores_nonforward_commands() -> None:
    from unilab.envs.locomotion.g1.joystick import compute_forward_progress_failure

    failure, _ = compute_forward_progress_failure(
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        np.asarray([99, 99], dtype=np.uint32),
        np.asarray([[0.0, 0.0, 0.0], [0.09, 0.0, 0.0]], dtype=np.float32),
        ctrl_dt=0.02,
        grace_steps=50,
        min_command_forward_speed=0.1,
        min_average_forward_speed=0.2,
    )
    np.testing.assert_array_equal(failure, [False, False])


def test_teacher_mujoco_reset_applies_left_knee_strength_to_runtime_pool() -> None:
    pytest.importorskip("mujoco", reason="mujoco is required for the runtime sentinel")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env is unavailable")

    ensure_registries()
    with initialize(config_path="../../../../conf/offpolicy", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["task=sac/g1_walk_flat/mujoco_left_knee_090"],
        )
    env_override = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="sac",
    ).build_task_env_cfg_override()
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=env_override,
        sim_backend="mujoco",
    )
    try:
        env.init_state()
        backend = env._backend
        nominal_kp, nominal_kd = backend.get_actuator_gains()
        runtime_kp = np.asarray(backend._pool.get_field(0, "kp"))
        runtime_kd = np.asarray(backend._pool.get_field(0, "kd"))

        np.testing.assert_allclose(runtime_kp[3], nominal_kp[3] * 0.9)
        np.testing.assert_allclose(runtime_kd[3], nominal_kd[3] * 0.9)
        np.testing.assert_allclose(runtime_kp[9], nominal_kp[9])
        np.testing.assert_allclose(runtime_kd[9], nominal_kd[9])
        np.testing.assert_allclose(
            env.state.info["privileged_actuator_strength"][0],
            cfg.env.domain_rand.actuator_strength.multipliers,
        )
    finally:
        env.close()
