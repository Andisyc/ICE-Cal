from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.g1.walk_commands import (
    command_resample_mask,
    freeze_inactive_gait_phase,
)
from unilab.envs.locomotion.g1.walk_control import (
    advance_gait_phase,
    select_authority_actions,
)
from unilab.envs.locomotion.g1.walk_observation import (
    assemble_walk_observation,
    build_obs_groups_spec,
)
from unilab.envs.locomotion.g1.walk_reward import normalized_corridor_violation


def test_g1_walk_env_composes_responsibility_specific_framework_bindings() -> None:
    from unilab.envs.locomotion.g1.joystick import G1WalkEnv
    from unilab.envs.locomotion.g1.walk_control_bindings import G1WalkControlBindings
    from unilab.envs.locomotion.g1.walk_observation_bindings import (
        G1WalkObservationBindings,
    )
    from unilab.envs.locomotion.g1.walk_runtime_bindings import G1WalkRuntimeBindings

    assert issubclass(G1WalkEnv, G1WalkObservationBindings)
    assert issubclass(G1WalkEnv, G1WalkControlBindings)
    assert issubclass(G1WalkEnv, G1WalkRuntimeBindings)


def test_walk_observation_owner_builds_actor_and_critic_dimensions() -> None:
    assert build_obs_groups_spec(
        mode_observation=False,
        height_observation=False,
        privileged_strength=False,
        fada_privileged=False,
        fada_body_count=0,
    ) == {"obs": 98, "critic": 101}
    assert build_obs_groups_spec(
        mode_observation=True,
        height_observation=True,
        privileged_strength=True,
        fada_privileged=True,
        fada_body_count=3,
    ) == {"obs": 100, "critic": 98 + 2 + 29 + 177}


def test_walk_observation_owner_assembles_actor_and_critic_without_backend_access() -> None:
    rows = 2
    zeros3 = np.zeros((rows, 3), dtype=np.float32)
    zeros2 = np.zeros((rows, 2), dtype=np.float32)
    zeros29 = np.zeros((rows, 29), dtype=np.float32)
    result = assemble_walk_observation(
        noisy_gyro=zeros3,
        noisy_gravity=zeros3,
        noisy_diff=zeros29,
        noisy_dof_vel=zeros29,
        gyro=zeros3,
        gravity=zeros3,
        diff=zeros29,
        dof_vel=zeros29,
        last_actions=zeros29,
        command_obs=zeros3,
        gait_phase=zeros2,
        mode_obs=np.ones((rows, 1), dtype=np.float32),
        linvel=zeros3,
        mode_observation=False,
        walk_profile=True,
        fada_privileged=False,
        privileged_strength=None,
        fada_privileged_obs=None,
        dtype=np.dtype(np.float32),
    )
    assert result["obs"].shape == (rows, 98)
    assert result["critic"].shape == (rows, 101)


def test_walk_control_owner_preserves_stand_authority_and_phase_rules() -> None:
    actions = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(
        select_authority_actions(actions, np.array([True, False]), enabled=True),
        np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float32),
    )
    phase = advance_gait_phase(
        np.zeros((2, 2), dtype=np.float32),
        active=np.array([True, False]),
        delta=0.25,
        enabled=True,
        freeze_inactive=True,
        stand_phase=np.array([1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(phase, [[0.25, 0.25], [1.0, 2.0]])


def test_walk_command_owner_resamples_and_freezes_only_selected_rows() -> None:
    np.testing.assert_array_equal(
        command_resample_mask(np.array([0, 4, 5, 8]), interval_steps=4),
        np.array([False, True, False, True]),
    )
    phase = np.zeros((3, 2), dtype=np.float32)
    frozen = freeze_inactive_gait_phase(
        phase,
        np.array([True, False, True]),
        np.array([0.5, 1.5], dtype=np.float32),
    )
    np.testing.assert_allclose(frozen, [[0.0, 0.0], [0.5, 1.5], [0.0, 0.0]])


def test_walk_reward_owner_preserves_squared_normalized_corridor_excess() -> None:
    np.testing.assert_allclose(
        normalized_corridor_violation(np.array([0.05, 0.2, -0.3]), 0.1),
        np.array([0.0, 1.0, 4.0]),
    )
