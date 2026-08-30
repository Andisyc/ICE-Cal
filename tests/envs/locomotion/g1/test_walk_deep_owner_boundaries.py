from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_stand_reward_terms_use_explicit_mask_and_preserve_rows() -> None:
    from unilab.envs.locomotion.g1.walk_reward import stand_action_l2

    actions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    stand_mask = np.asarray([1.0, 0.0], dtype=np.float32)

    np.testing.assert_allclose(stand_action_l2(actions, stand_mask), [5.0, 0.0])


def test_stand_height_target_accepts_scalar_and_column() -> None:
    from unilab.envs.locomotion.g1.walk_reward import resolve_stand_height_target

    np.testing.assert_allclose(resolve_stand_height_target(0.75, num_envs=2), [0.75, 0.75])
    np.testing.assert_allclose(
        resolve_stand_height_target(np.asarray([[0.7], [0.8]]), num_envs=2),
        [0.7, 0.8],
    )


def test_actuator_strength_validation_accepts_fixed_multipliers() -> None:
    from unilab.envs.locomotion.g1.walk_actuator_randomization import (
        validate_actuator_strength_config,
    )

    cfg = SimpleNamespace(
        enabled=True,
        include_in_critic_obs=False,
        sampling_mode="fixed",
        multipliers=[1.0, 0.8],
        candidate_actuator_indices=[],
    )

    assert validate_actuator_strength_config(cfg, expected_actions=2) is cfg


def test_reset_randomization_freezes_only_standing_rows() -> None:
    from unilab.envs.locomotion.g1.walk_reset_randomization import freeze_standing_phase

    phase = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    result = freeze_standing_phase(
        phase,
        gait_enabled=np.asarray([0.0, 1.0], dtype=np.float32),
        enabled=True,
        freeze=True,
        stand_phase=(np.pi, np.pi),
    )

    np.testing.assert_allclose(result[0], [np.pi, np.pi])
    np.testing.assert_allclose(result[1], [0.3, 0.4])


def test_pure_walk_owners_do_not_import_joystick_or_backend_subclasses() -> None:
    from pathlib import Path

    owner_paths = (
        Path("src/unilab/envs/locomotion/g1/walk_reward.py"),
        Path("src/unilab/envs/locomotion/g1/walk_actuator_randomization.py"),
        Path("src/unilab/envs/locomotion/g1/walk_reset_randomization.py"),
    )

    for path in owner_paths:
        source = path.read_text()
        assert "locomotion.g1.joystick" not in source
        assert "backend.mujoco" not in source
        assert "backend.motrix" not in source

