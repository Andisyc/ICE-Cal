from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider


class _Spawn:
    def apply_spawn(self, _env_ids, qpos_xyz, *, yaw=None):
        del yaw
        return qpos_xyz

    def record_episode_start(self, _env_ids, _qpos_xyz) -> None:
        return None


def _env(*, randomize_reset_pose: bool):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            commands=SimpleNamespace(vel_limit=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
            domain_rand=DomainRandConfig(randomize_reset_pose=randomize_reset_pose),
        ),
        _init_qpos=np.asarray([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]),
        _init_qvel=np.zeros((6,), dtype=np.float64),
        _num_action=0,
        _spawn=_Spawn(),
    )


def test_locomotion_reset_pose_can_be_nominal_without_changing_training_default() -> None:
    provider = LocomotionDRProvider()
    provider._sample_commands = lambda _env, count: np.zeros((count, 3))  # type: ignore[method-assign]
    provider._get_qvel_limit = lambda _env: 0.0  # type: ignore[method-assign]

    nominal = provider.build_reset_plan(_env(randomize_reset_pose=False), np.asarray([0, 1]))

    expected = np.asarray([[0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]] * 2)
    np.testing.assert_allclose(nominal.qpos, expected)
    assert DomainRandConfig().randomize_reset_pose is True
