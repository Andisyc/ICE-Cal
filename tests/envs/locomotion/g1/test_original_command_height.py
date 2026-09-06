from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.locomotion.g1.walk_math import (
    compute_feet_phase_height_targets,
    original_command_height_targets,
)
from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings


def test_original_curve_equivalence_and_timing():
    p = np.linspace(-np.pi, np.pi, 1001)
    phase = np.column_stack([p, (p + 2 * np.pi) % (2 * np.pi) - np.pi])
    commands = np.tile([0.3, 0, 0], (len(p), 1))
    actual = original_command_height_targets(phase, commands, 0.06, 0.3, 0.3)
    expected = compute_feet_phase_height_targets(phase, 0.03)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    np.testing.assert_allclose(
        original_command_height_targets(phase, commands * 0, 0.06, 0.3, 0.3), 0
    )
    np.testing.assert_allclose(
        original_command_height_targets(phase[:, ::-1], commands, 0.06, 0.3, 0.3), actual[::-1]
    )
    points = np.array([[0, np.pi], [np.pi / 2, -np.pi / 2], [np.pi, 0]])
    left, right = compute_feet_phase_height_targets(points, 0.03)
    np.testing.assert_allclose(left, [0.03, 0.015, 0], atol=1e-8)
    np.testing.assert_allclose(right, [0, 0.015, 0.03], atol=1e-8)


def test_turn_and_reverse_use_same_amplitude():
    phase = np.tile([0, np.pi], (3, 1))
    commands = np.array([[0.3, 0, 0], [-0.3, 0, 0], [0, 0, 1]])
    left, right = original_command_height_targets(phase, commands, 0.06, 0.3, 0.3)
    np.testing.assert_allclose(left, 0.03, atol=1e-8)
    np.testing.assert_allclose(right, 0)


def test_reward_matches_original_and_orders_heights():
    class Backend:
        heights = np.array([[0.03, 0], [0, 0.03], [0, 0], [0.04, 0.01]])

        def get_sensor_data(self, name):
            z = self.heights[:, 0 if name.startswith("left") else 1]
            return np.column_stack([np.zeros((4, 2)), z])

    backend = Backend()
    cfg = SimpleNamespace(
        feet_phase_mode="original_command_height_v1",
        feet_phase_swing_height=0.06,
        feet_phase_command_speed_scale=0.3,
        feet_phase_turn_length=0.3,
        feet_phase_tracking_sigma=0.04,
    )
    owner = SimpleNamespace(
        _backend=backend, _num_envs=4, _reward_cfg=cfg, _gait_reward_gate=lambda v: np.ones(4)
    )
    ctx = SimpleNamespace(
        info={"commands": np.tile([0.3, 0, 0], (4, 1)), "gait_phase": np.tile([0, np.pi], (4, 1))},
        linvel=np.zeros((4, 3)),
    )
    actual = G1WalkRewardBindings._reward_feet_phase(owner, ctx)
    cfg.feet_phase_mode, cfg.feet_phase_swing_height = "legacy", 0.03
    np.testing.assert_allclose(actual, G1WalkRewardBindings._reward_feet_phase(owner, ctx))
    assert actual[0] > actual[2] > actual[1]
    np.testing.assert_allclose(actual[0], actual[3])  # Original relative-height invariance.
    np.testing.assert_allclose(actual[2], np.exp(-(0.03**2) / 0.04))
    cfg.feet_phase_mode, cfg.feet_phase_swing_height = "original_command_height_v1", 0.06
    ctx.info["commands"][:] = 0
    stopped = G1WalkRewardBindings._reward_feet_phase(owner, ctx)
    assert stopped[2] == 1 > stopped[0]


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        original_command_height_targets(
            np.zeros((1, 2)), np.array([[np.nan, 0, 0]]), 0.06, 0.3, 0.3
        )


def test_raw_phase_is_preserved_in_actor_and_critic():
    from unilab.envs.locomotion.g1.walk_observation import assemble_walk_observation

    phase = np.array([[0.7, 0.7 - np.pi]])
    z3, joints = np.zeros((1, 3)), np.zeros((1, 29))
    obs = assemble_walk_observation(
        noisy_gyro=z3,
        noisy_gravity=z3,
        noisy_diff=joints,
        noisy_dof_vel=joints,
        gyro=z3,
        gravity=z3,
        diff=joints,
        dof_vel=joints,
        last_actions=joints,
        command_obs=z3,
        gait_phase=phase,
        mode_obs=np.zeros((1, 1)),
        linvel=z3,
        mode_observation=False,
        walk_profile=True,
        fada_privileged=True,
        privileged_strength=None,
        fada_privileged_obs=None,
        dtype=np.float32,
    )
    for value in obs.values():
        np.testing.assert_allclose(value[:, -2:], phase)


def test_zero_command_does_not_stop_clock():
    from unilab.envs.locomotion.g1.walk_control import advance_gait_phase

    phase = np.array([[.7, .7 + np.pi]])
    advanced = advance_gait_phase(
        phase, active=np.array([False]), delta=.1, enabled=True,
        freeze_inactive=False, stand_phase=np.zeros(2),
    )
    np.testing.assert_allclose(advanced, phase + .1)
    np.testing.assert_allclose(
        original_command_height_targets(advanced, np.zeros((1, 3)), .06, .3, .3), 0,
    )
