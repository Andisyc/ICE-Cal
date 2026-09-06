import numpy as np
import pytest

from unilab.envs.locomotion.g1.walk_math import (
    command_phase_amplitude,
    command_phase_height_cost,
    command_phase_height_targets,
    update_command_phase_amplitude,
)


def test_command_controls_amplitude_including_turn_and_reverse():
    command = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(command_phase_amplitude(command, 0.3, 0.3), [0.0, 0.5, 0.5, 0.5])


def test_stopped_grounded_beats_lift_and_hover():
    phase = np.array([[np.pi / 2, 3 * np.pi / 2]])
    target = command_phase_height_targets(phase, np.array([0.0]), 0.09)
    assert command_phase_height_cost(np.zeros((1, 2)), target, 0.09)[0] == 0
    for actual in [[[0.0, 0.045]], [[0.045, 0.045]]]:
        assert command_phase_height_cost(np.array(actual), target, 0.09)[0] > 0


def test_correct_swing_beats_drag_and_wrong_leg():
    phase = np.array([[np.pi / 2, 3 * np.pi / 2]])
    target = command_phase_height_targets(phase, np.array([0.5]), 0.09)
    np.testing.assert_allclose(target, [[0.045, 0.0]])
    good = command_phase_height_cost(np.array([[0.045, 0.0]]), target, 0.09)[0]
    assert good == 0
    assert good < command_phase_height_cost(np.zeros((1, 2)), target, 0.09)[0]
    assert good < command_phase_height_cost(np.array([[0.0, 0.045]]), target, 0.09)[0]


def test_stopping_settles_smoothly_and_is_row_local():
    result = update_command_phase_amplitude(np.array([1.0, 0.0]), np.array([0.0, 0.0]), 0.02, 0.15)
    assert 0 < result[0] < 1
    assert result[1] == 0
    np.testing.assert_allclose(result[0], np.exp(-0.02 / 0.15))


def test_periodicity_and_support():
    phase = np.array([[0.0, np.pi], [np.pi / 2, 3 * np.pi / 2]])
    target = command_phase_height_targets(phase, np.ones(2), 0.09)
    np.testing.assert_allclose(target[0], [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(
        target, command_phase_height_targets(phase + 2 * np.pi, np.ones(2), 0.09), atol=1e-15
    )


def test_invalid_command_rejected():
    with pytest.raises(ValueError):
        command_phase_amplitude(np.array([[np.nan, 0.0, 0.0]]), 0.3, 0.3)


def test_reward_binding_penalizes_hover_and_does_not_mutate_state():
    from types import SimpleNamespace

    from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings

    class Backend:
        def get_sensor_data(self, name):
            if name.endswith("upvector"):
                return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
            return np.array([[0.0, 0.0, -0.002], [0.0, 0.0, 0.043]])

    owner = SimpleNamespace(
        _backend=Backend(),
        _num_envs=2,
        _reward_cfg=SimpleNamespace(
            feet_phase_mode="command_height_v1",
            feet_phase_swing_height=0.09,
            feet_phase_height_scale=0.09,
        ),
    )
    info = {
        "gait_phase": np.array([[np.pi / 2, 3 * np.pi / 2]] * 2),
        "command_phase_amplitude": np.zeros(2),
    }
    ctx = SimpleNamespace(info=info)
    result = G1WalkRewardBindings._reward_feet_phase(owner, ctx)
    np.testing.assert_allclose(result, [0.0, -0.5], atol=1e-12)
    np.testing.assert_array_equal(info["command_phase_amplitude"], [0.0, 0.0])


def test_reset_initializes_amplitude_from_commands():
    from types import SimpleNamespace

    from unilab.envs.locomotion.g1.walk_domain_randomization import (
        G1WalkDomainRandomizationProvider,
    )

    owner = SimpleNamespace(
        _command_gait_mask=lambda env, cmd: np.any(cmd != 0, axis=1),
        _sample_gait_phase=lambda env, n: np.zeros((n, 2)),
        _apply_standing_reset_phase=lambda *args: None,
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            commands=SimpleNamespace(),
            reward_config=SimpleNamespace(
                feet_phase_mode="command_height_v1",
                feet_phase_command_speed_scale=0.3,
                feet_phase_turn_length=0.3,
            ),
        )
    )
    result = G1WalkDomainRandomizationProvider._build_extra_info_updates_for_commands(
        owner, env, 2, np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    )
    np.testing.assert_allclose(result["command_phase_amplitude"], [0.0, 0.5])


def test_command_phase_config_composes_and_validates(monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from unilab.algos.torch.distill.fada.privileged_oracle import validate_fada_single_reward
    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        resolve_privileged_locomotion_sac_runtime,
    )

    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "command-phase-test")
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(config_dir=str(root / "conf/offpolicy"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=sac/g1_walk_flat/mujoco_fada_privileged_oracle_command_phase_grouped_dr_lineage"
            ],
        )
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )
    runtime.validate_training_config(cfg)
    assert cfg.env.commands.rel_standing_envs == 0.3
    assert cfg.env.commands.resampling_time == 4.0
    reward = OmegaConf.to_container(cfg.reward, resolve=True)
    with pytest.raises(ValueError, match="feet_phase_mode"):
        validate_fada_single_reward(
            reward_scales=reward["scales"],
            reward_config=reward,
            behavior_profile="phase_locomotion_v1",
        )


def test_new_mode_rejects_composed_terrain_before_backend_creation():
    from types import SimpleNamespace

    from unilab.envs.locomotion.g1.joystick import G1WalkEnv

    for scene in [
        SimpleNamespace(fragment_files=["extra.xml"], terrain=None),
        SimpleNamespace(fragment_files=[], terrain=object()),
    ]:
        cfg = SimpleNamespace(
            reward_config=SimpleNamespace(feet_phase_mode="command_height_v1"), scene=scene
        )
        with pytest.raises(ValueError, match="terrain or scene fragments"):
            G1WalkEnv(cfg)


def test_action_updates_amplitude_once_and_preserves_inactive_row():
    from types import SimpleNamespace

    from unilab.base.np_env import NpEnvState
    from unilab.envs.locomotion.g1.joystick import G1WalkEnv

    env = object.__new__(G1WalkEnv)
    env._num_envs = 2
    env._reward_cfg = SimpleNamespace(
        feet_phase_mode="command_height_v1",
        feet_phase_command_speed_scale=0.3,
        feet_phase_turn_length=0.3,
        feet_phase_settling_tau=0.15,
    )
    env._cfg = SimpleNamespace(
        reward_config=env._reward_cfg,
        ctrl_dt=0.02,
        control_config=SimpleNamespace(action_scale=1.0),
    )
    env.default_angles = np.zeros(2)
    env._gait_phase_delta = 0.1
    env._gait_constraint_cfg = lambda: SimpleNamespace(
        enabled=False, freeze_phase_in_stand_mode=False
    )
    env._actions_for_execution = lambda actions, info: actions
    env._debug_action_trace_enabled = lambda: False
    state = NpEnvState(
        obs={},
        reward=np.zeros(2),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.zeros(2, dtype=bool),
        info={
            "gait_phase": np.zeros((2, 2)),
            "commands": np.zeros((2, 3)),
            "command_phase_amplitude": np.array([1.0, 0.0]),
        },
    )
    env.apply_action(np.zeros((2, 2)), state)
    np.testing.assert_allclose(state.info["command_phase_amplitude"], [np.exp(-0.02 / 0.15), 0.0])
