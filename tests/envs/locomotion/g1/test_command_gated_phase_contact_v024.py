from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base.np_env import NpEnv, NpEnvState
from unilab.envs.locomotion.g1.walk_commands import (
    canonicalize_g1_commands,
    resolve_g1_command_gait_state,
)
from unilab.envs.locomotion.g1.walk_control import (
    estimate_clipped_pd_torque,
    reset_command_gait_phase,
    step_command_gait_phase,
)
from unilab.envs.locomotion.g1.walk_reward import (
    null_foot_force_balance_l1,
    null_torque_relaxation_l2,
    phase_contact_mismatch_cost,
)


def test_command_dead_zone_is_joint_exact_and_boundary_is_non_null() -> None:
    commands = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.099, 0.0, 0.099],
            [0.1, 0.0, 0.0],
            [0.0, -0.1, 0.0],
            [0.0, 0.0, -0.1],
            [0.0, 0.0, 0.3],
            [-0.6, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    canonical = canonicalize_g1_commands(commands, xy_dead_zone=0.1, yaw_dead_zone=0.1)
    state = resolve_g1_command_gait_state(
        canonical,
        xy_dead_zone=0.1,
        yaw_dead_zone=0.1,
        linear_intensity_span=0.9,
        yaw_intensity_span=0.7,
        min_frequency=0.7,
        max_frequency=1.5,
    )

    np.testing.assert_array_equal(canonical[:2], np.zeros((2, 3)))
    np.testing.assert_array_equal(state.is_null, [True, True, False, False, False, False, False])
    np.testing.assert_allclose(state.intensity[2:5], 0.0)
    np.testing.assert_allclose(state.frequency[:2], 0.0)
    np.testing.assert_allclose(state.frequency[2:5], 0.7)
    assert state.intensity[5] > 0.0
    assert state.intensity[6] > 0.0

    downstream = resolve_g1_command_gait_state(
        np.array([[0.01, 0.0, 0.0]]),
        xy_dead_zone=0.1,
        yaw_dead_zone=0.1,
        linear_intensity_span=0.9,
        yaw_intensity_span=0.7,
        min_frequency=0.7,
        max_frequency=1.5,
    )
    np.testing.assert_array_equal(downstream.is_null, [False])


def test_command_intensity_uses_xy_and_yaw_and_saturates() -> None:
    state = resolve_g1_command_gait_state(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, -0.8],
                [0.2, 0.0, 0.2],
            ]
        ),
        xy_dead_zone=0.1,
        yaw_dead_zone=0.1,
        linear_intensity_span=0.9,
        yaw_intensity_span=0.7,
        min_frequency=0.7,
        max_frequency=1.5,
    )

    np.testing.assert_allclose(state.intensity[:5], 1.0)
    np.testing.assert_allclose(state.frequency[:5], 1.5)
    assert state.intensity[5] == pytest.approx(max(0.1 / 0.9, 0.1 / 0.7))


def test_command_gated_phase_reset_and_all_four_transitions() -> None:
    startup = np.array([0.0, np.pi])
    stand = np.array([np.pi, np.pi])
    reset = reset_command_gait_phase(
        np.array([True, False]), startup_phase=startup, stand_phase=stand
    )
    np.testing.assert_allclose(reset, [stand, startup])

    previous = np.array(
        [
            [np.pi, np.pi],
            [np.pi, np.pi],
            [1.9 * np.pi, 0.9 * np.pi],
            [0.2, np.pi + 0.2],
        ]
    )
    next_phase = step_command_gait_phase(
        previous,
        was_null=np.array([True, True, False, False]),
        is_null=np.array([True, False, False, True]),
        frequency=np.array([0.0, 0.7, 1.5, 0.0]),
        ctrl_dt=0.02,
        startup_phase=startup,
        stand_phase=stand,
    )

    np.testing.assert_allclose(next_phase[0], stand)
    np.testing.assert_allclose(next_phase[1], startup)
    delta = 2.0 * np.pi * 1.5 * 0.02
    np.testing.assert_allclose(next_phase[2], (previous[2] + delta) % (2.0 * np.pi))
    np.testing.assert_allclose(next_phase[3], stand)
    assert ((next_phase[2, 1] - next_phase[2, 0]) % (2.0 * np.pi)) == pytest.approx(np.pi)


def test_phase_contact_cost_uses_cycle_fraction_and_strict_boundaries() -> None:
    duty = 0.55
    threshold = 1.0
    fractions = np.array(
        [
            [0.10, 0.60],
            [0.50, 0.55],
            [0.55, 0.60],
            [0.50, 0.60],
        ]
    )
    phase = fractions * (2.0 * np.pi)
    left_force = np.array([2.0, 1.0, 2.0, 2.0])
    right_force = np.array([0.0, 2.0, 2.0, 2.0])

    cost = phase_contact_mismatch_cost(
        phase,
        left_force,
        right_force,
        duty_factor=duty,
        contact_force_threshold=threshold,
    )

    np.testing.assert_allclose(cost, [0.0, 1.0, 1.0, 0.5])
    assert np.all((0.0 <= cost) & (cost <= 1.0))
    null_cost = phase_contact_mismatch_cost(
        np.array([[np.pi, np.pi]]),
        np.array([2.0]),
        np.array([2.0]),
        duty_factor=duty,
        contact_force_threshold=threshold,
    )
    np.testing.assert_allclose(null_cost, 0.0)

    boundary = np.asarray(duty * (2.0 * np.pi), dtype=np.float32)
    below = np.nextafter(boundary, np.asarray(0.0, dtype=np.float32))
    strict_cost = phase_contact_mismatch_cost(
        np.array([[below, boundary]], dtype=np.float32),
        np.array([2.0]),
        np.array([0.0]),
        duty_factor=duty,
        contact_force_threshold=threshold,
    )
    np.testing.assert_allclose(strict_cost, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duty_factor": 0.5, "contact_force_threshold": 1.0},
        {"duty_factor": 1.0, "contact_force_threshold": 1.0},
        {"duty_factor": 0.55, "contact_force_threshold": 0.0},
    ],
)
def test_phase_contact_cost_rejects_invalid_semantics(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        phase_contact_mismatch_cost(np.zeros((1, 2)), np.zeros(1), np.zeros(1), **kwargs)


def test_null_quality_terms_are_exactly_gated_and_normalized() -> None:
    null = np.array([True, True, False])
    force_cost = null_foot_force_balance_l1(
        np.array([50.0, 0.0, 100.0]),
        np.array([50.0, 0.0, 0.0]),
        null,
        epsilon=1.0e-6,
    )
    np.testing.assert_allclose(force_cost, [0.0, 1.0, 0.0])

    torque_cost = null_torque_relaxation_l2(
        np.array([[10.0, 5.0], [20.0, 10.0], [20.0, 10.0]]),
        np.array([20.0, 10.0]),
        null,
    )
    np.testing.assert_allclose(torque_cost, [0.25, 1.0, 0.0])


def test_virtual_pd_torque_is_per_row_clipped_and_never_silently_zero_filled() -> None:
    target = np.array([[1.0, -1.0], [0.1, 0.2]])
    dof_pos = np.zeros_like(target)
    dof_vel = np.array([[1.0, -1.0], [0.0, 0.0]])
    kp = np.array([[100.0, 100.0], [20.0, 30.0]])
    kd = np.array([[10.0, 10.0], [2.0, 3.0]])
    tau_max = np.array([50.0, 25.0])

    torque = estimate_clipped_pd_torque(target, dof_pos, dof_vel, kp, kd, tau_max)

    np.testing.assert_allclose(torque, [[50.0, -25.0], [2.0, 6.0]])
    with pytest.raises(ValueError, match="target"):
        estimate_clipped_pd_torque(None, dof_pos, dof_vel, kp, kd, tau_max)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "old_command,new_command,expected_next_phase",
    [
        (
            np.array([[0.3, 0.0, 0.0]]),
            np.zeros((1, 3)),
            np.array([[np.pi, np.pi]]),
        ),
        (
            np.zeros((1, 3)),
            np.array([[0.3, 0.0, 0.0]]),
            np.array([[0.0, np.pi]]),
        ),
    ],
)
def test_resample_boundary_scores_old_command_and_publishes_new_command_for_next_obs(
    monkeypatch: pytest.MonkeyPatch,
    old_command: np.ndarray,
    new_command: np.ndarray,
    expected_next_phase: np.ndarray,
) -> None:
    from unilab.envs.locomotion.g1.walk_config import G1CommandGatedPhaseContactConfig
    from unilab.envs.locomotion.g1.walk_control_bindings import G1WalkControlBindings

    owner = G1WalkControlBindings()
    owner._num_envs = 1
    owner._cfg = SimpleNamespace(
        ctrl_dt=0.02,
        commands=SimpleNamespace(resampling_time=4.0, heading_command=False),
        command_gated_phase_contact=G1CommandGatedPhaseContactConfig(enabled=True),
    )
    old_state = resolve_g1_command_gait_state(
        old_command,
        xy_dead_zone=0.1,
        yaw_dead_zone=0.1,
        linear_intensity_span=0.9,
        yaw_intensity_span=0.7,
        min_frequency=0.7,
        max_frequency=1.5,
    )
    info = {
        "commands": old_state.commands,
        "command_is_null": old_state.is_null,
        "gait_phase": reset_command_gait_phase(
            old_state.is_null,
            startup_phase=np.array([0.0, np.pi]),
            stand_phase=np.array([np.pi, np.pi]),
        ),
        "steps": np.array([199], dtype=np.uint32),
    }
    monkeypatch.setattr(
        "unilab.envs.locomotion.g1.walk_control_bindings.sample_g1_walk_commands",
        lambda _env, _count: new_command.copy(),
    )

    owner._advance_command_state_for_reward(info)
    reward_command = info["commands"].copy()
    reward_phase = info["gait_phase"].copy()
    owner._commit_command_state_for_next_observation(info)

    np.testing.assert_allclose(reward_command, old_command)
    if old_state.is_null[0]:
        np.testing.assert_allclose(reward_phase, [[np.pi, np.pi]])
    else:
        assert not np.allclose(reward_phase, [[0.0, np.pi]])
    np.testing.assert_allclose(info["commands"], new_command)
    np.testing.assert_allclose(info["gait_phase"], expected_next_phase)


def test_actual_step_refreshes_torque_and_missing_target_fails_closed() -> None:
    from unilab.envs.locomotion.g1.walk_config import G1CommandGatedPhaseContactConfig
    from unilab.envs.locomotion.g1.walk_control_bindings import G1WalkControlBindings

    owner = G1WalkControlBindings()
    owner._num_envs = 1
    owner._num_action = 2
    owner._cfg = SimpleNamespace(
        command_gated_phase_contact=G1CommandGatedPhaseContactConfig(enabled=True)
    )
    owner._fada_base_kp = np.array([20.0, 30.0])
    owner._fada_base_kd = np.array([2.0, 3.0])
    owner._fada_tau_max = np.array([10.0, 10.0])
    info = {
        "command_gated_actuator_target": np.array([[1.0, 0.2]]),
        "command_gated_actuator_target_valid": np.array([True]),
        "fada_kp_scale": np.ones((1, 2)),
        "fada_kd_scale": np.ones((1, 2)),
        "torques": np.zeros((1, 2)),
    }

    owner._refresh_command_gated_torque(info, np.array([[0.8, 0.0]]), np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(info["torques"], [[2.0, 6.0]], atol=1.0e-6)
    np.testing.assert_array_equal(info["command_gated_actuator_target_valid"], [False])

    with pytest.raises(ValueError, match="valid final actuator target"):
        owner._refresh_command_gated_torque(info, np.zeros((1, 2)), np.zeros((1, 2)))


def test_public_refresh_cannot_consume_or_replay_a_v024_physics_step() -> None:
    from unilab.envs.locomotion.g1.walk_config import G1CommandGatedPhaseContactConfig
    from unilab.envs.locomotion.g1.walk_control_bindings import G1WalkControlBindings
    from unilab.envs.locomotion.g1.walk_runtime_bindings import G1WalkRuntimeBindings

    class Backend:
        def get_sensor_data(self, _name: str) -> np.ndarray:
            return np.array([[0.0, 0.0, 1.0]])

    class Owner(G1WalkControlBindings, G1WalkRuntimeBindings):
        def __init__(self) -> None:
            self._num_envs = 1
            self._num_action = 2
            self._cfg = SimpleNamespace(
                command_gated_phase_contact=G1CommandGatedPhaseContactConfig(enabled=True),
                sensor=SimpleNamespace(upvector="up"),
                domain_rand=SimpleNamespace(
                    actuator_strength=SimpleNamespace(
                        curriculum_progress_mode="episode_quality",
                        curriculum_enabled=False,
                    )
                ),
            )
            self._reward_cfg = SimpleNamespace(max_tilt_deg=45.0, min_base_height=0.1)
            self._backend = Backend()
            self._episode_tracker = None
            self._penalty_curriculum = None
            self.step_counter = 0
            self.calls = {name: 0 for name in ("advance", "torque", "reward", "commit", "obs")}

        def _command_gated_phase_contact_enabled(self) -> bool:
            return True

        def _advance_command_state_for_reward(self, _info: dict) -> None:
            self.calls["advance"] += 1

        def get_local_linvel(self) -> np.ndarray:
            return np.zeros((1, 3))

        def get_gyro(self) -> np.ndarray:
            return np.zeros((1, 3))

        def get_dof_pos(self) -> np.ndarray:
            return np.zeros((1, 2))

        def get_dof_vel(self) -> np.ndarray:
            return np.zeros((1, 2))

        def _refresh_command_gated_torque(self, info: dict, *_args) -> None:
            self.calls["torque"] += 1
            info["command_gated_actuator_target_valid"][:] = False

        def _terrain_relative_base_height(self) -> np.ndarray:
            return np.ones(1)

        def _forward_progress_failure(self, _info: dict) -> np.ndarray:
            return np.zeros(1, dtype=np.bool_)

        def _compute_reward(self, *_args) -> np.ndarray:
            self.calls["reward"] += 1
            return np.array([3.0])

        def _debug_action_trace(self, *_args, **_kwargs) -> None:
            return None

        def _commit_command_state_for_next_observation(self, _info: dict) -> None:
            self.calls["commit"] += 1

        def _compute_obs(self, *_args) -> dict[str, np.ndarray]:
            self.calls["obs"] += 1
            return {"obs": np.array([[float(self.calls["obs"])]]), "critic": np.zeros((1, 1))}

        def _write_curriculum_log(self, _info: dict) -> None:
            return None

    owner = Owner()
    owner._state = NpEnvState(
        obs={"obs": np.zeros((1, 1)), "critic": np.zeros((1, 1))},
        reward=np.array([7.0]),
        terminated=np.array([False]),
        truncated=np.array([False]),
        info={
            "steps": np.zeros(1, dtype=np.uint32),
            "commands": np.zeros((1, 3)),
            "command_is_null": np.array([True]),
            "gait_phase": np.array([[np.pi, np.pi]]),
            "command_gated_actuator_target_valid": np.array([False]),
        },
    )

    refreshed = NpEnv.refresh_state(owner)
    np.testing.assert_allclose(refreshed.reward, [7.0])
    assert owner.calls == {"advance": 0, "torque": 0, "reward": 0, "commit": 0, "obs": 1}

    owner._state.info["commands"][:] = np.array([[0.2, 0.0, 0.0]])
    refreshed_command = NpEnv.refresh_state(owner)
    np.testing.assert_allclose(refreshed_command.info["gait_phase"], [[0.0, np.pi]])
    np.testing.assert_array_equal(refreshed_command.info["command_is_null"], [False])
    assert owner.calls == {"advance": 0, "torque": 0, "reward": 0, "commit": 0, "obs": 2}

    owner._state.info["command_gated_actuator_target_valid"][:] = True
    owner._state = owner.update_state(owner._state)
    np.testing.assert_allclose(owner._state.reward, [3.0])
    assert owner.calls == {"advance": 1, "torque": 1, "reward": 1, "commit": 1, "obs": 3}

    refreshed_again = NpEnv.refresh_state(owner)
    np.testing.assert_allclose(refreshed_again.reward, [3.0])
    assert owner.calls == {"advance": 1, "torque": 1, "reward": 1, "commit": 1, "obs": 4}


def test_reset_publishes_explicit_zero_torque_without_claiming_a_valid_target() -> None:
    from unilab.envs.locomotion.g1.walk_config import G1CommandGatedPhaseContactConfig
    from unilab.envs.locomotion.g1.walk_domain_randomization import (
        G1WalkDomainRandomizationProvider,
    )

    env = SimpleNamespace(
        _num_action=2,
        cfg=SimpleNamespace(
            command_gated_phase_contact=G1CommandGatedPhaseContactConfig(enabled=True),
            commands=SimpleNamespace(
                heading_command=False,
                observe_height_command=False,
                random_height_during_walking=False,
            ),
            reward_config=SimpleNamespace(feet_phase_mode="phase_contact_v1"),
        ),
    )
    updates = G1WalkDomainRandomizationProvider()._build_extra_info_updates_for_commands(
        env, 2, np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    )

    np.testing.assert_array_equal(updates["torques"], np.zeros((2, 2)))
    np.testing.assert_array_equal(
        updates["command_gated_actuator_target_valid"], np.array([False, False])
    )
    np.testing.assert_allclose(updates["gait_phase"], [[np.pi, np.pi], [0.0, np.pi]])


def test_reward_bindings_use_public_net_force_and_finite_negative_contact_reward() -> None:
    from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings

    class Backend:
        def get_sensor_data(self, name: str) -> np.ndarray:
            sensors = {
                "left_foot_net_contact": np.array([[1.0, 0.0, 0.0, 20.0]]),
                "right_foot_net_contact": np.array([[1.0, 0.0, 0.0, 20.0]]),
            }
            return sensors[name]

    owner = G1WalkRewardBindings()
    owner._backend = Backend()
    owner._num_envs = 1
    owner._cfg = SimpleNamespace(
        command_gated_phase_contact=SimpleNamespace(
            duty_factor=0.55,
            contact_force_threshold=1.0,
            force_balance_epsilon=1.0e-6,
        )
    )
    owner._fada_tau_max = np.array([20.0, 10.0])
    ctx = SimpleNamespace(
        info={
            "commands": np.zeros((1, 3)),
            "gait_phase": np.array([[0.1 * 2 * np.pi, 0.6 * 2 * np.pi]]),
            "torques": np.array([[10.0, 5.0]]),
        }
    )

    contact = G1WalkRewardBindings._reward_phase_contact(owner, ctx)
    balance = G1WalkRewardBindings._reward_null_foot_force_balance(owner, ctx)
    torque = G1WalkRewardBindings._reward_null_torque_relaxation(owner, ctx)

    np.testing.assert_allclose(contact, [-0.5])
    np.testing.assert_allclose(balance, [0.0])
    np.testing.assert_allclose(torque, [0.25])
