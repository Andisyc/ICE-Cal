import numpy as np

from unilab.envs.locomotion.g1.walk_math import phase_height_targets_v3


def targets(phase, speed=0.3):
    commands = np.tile([speed, 0.0, 0.0], (len(phase), 1))
    return phase_height_targets_v3(np.asarray(phase), commands, 0.06, 0.3, 0.3)


def test_full_cycle_has_alternating_support_and_zero_speed():
    p = np.arange(1000) * 2 * np.pi / 1000
    phase = np.column_stack([p, p + np.pi])
    h = targets(phase)
    assert np.all(np.any(h == 0, axis=1))
    assert np.all(h >= 0)
    np.testing.assert_allclose(h.max(axis=0), [0.03, 0.03])
    np.testing.assert_allclose(targets(phase, 0), 0)
    np.testing.assert_allclose(targets(phase + 2 * np.pi), h, atol=1e-14)
    np.testing.assert_allclose(targets(phase[:, ::-1]), h[:, ::-1])


def test_quarter_cycle_and_mid_rise_values():
    phase = [
        [0, np.pi],
        [np.pi / 4, 5 * np.pi / 4],
        [np.pi / 2, 3 * np.pi / 2],
        [np.pi, 2 * np.pi],
        [3 * np.pi / 2, 5 * np.pi / 2],
    ]
    np.testing.assert_allclose(
        targets(phase), [[0, 0], [0.015, 0], [0.03, 0], [0, 0], [0, 0.03]], atol=1e-14
    )


def test_amplitude_monotonic_bounded_and_turn_aware():
    commands = np.array(
        [[0, 0, 0], [0.3, 0, 0], [0.8, 0, 0], [1, 0, 0], [100, 0, 0], [-0.3, 0, 0], [0, 0, 1]]
    )
    phase = np.tile([np.pi / 2, 3 * np.pi / 2], (len(commands), 1))
    h = phase_height_targets_v3(phase, commands, 0.06, 0.3, 0.3)
    assert np.all(np.diff(h[:5, 0]) > 0)
    assert np.all(h < 0.06)
    np.testing.assert_allclose(h[[1, 5, 6], 0], 0.03)


def test_binding_prefers_correct_leg_and_grounded_stop():
    from types import SimpleNamespace

    from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings

    class Backend:
        heights = [0.03, 0.0]

        def get_sensor_data(self, name):
            if name.endswith("upvector"):
                return np.array([[0.0, 0.0, 1.0]])
            return np.array([[0.0, 0.0, self.heights[0 if name.startswith("left") else 1] - 0.002]])

    backend = Backend()
    owner = SimpleNamespace(
        _backend=backend,
        _num_envs=1,
        _reward_cfg=SimpleNamespace(
            feet_phase_mode="command_height_v3",
            feet_phase_swing_height=0.06,
            feet_phase_command_speed_scale=0.3,
            feet_phase_turn_length=0.3,
            feet_phase_height_scale=0.03,
        ),
    )
    ctx = SimpleNamespace(
        info={
            "commands": np.array([[0.3, 0, 0]]),
            "gait_phase": np.array([[np.pi / 2, 3 * np.pi / 2]]),
        }
    )

    def reward():
        return G1WalkRewardBindings._reward_feet_phase(owner, ctx)[0]

    assert reward() == 1
    backend.heights = [0.02, 0]
    np.testing.assert_allclose(reward(), np.exp(-1 / 18))
    backend.heights = [0, 0]
    np.testing.assert_allclose(reward(), np.exp(-0.5))
    backend.heights = [0, 0.03]
    np.testing.assert_allclose(reward(), np.exp(-1))
    ctx.info["commands"][:] = 0
    lifted = reward()
    backend.heights = [0, 0]
    assert reward() == 1 > lifted
