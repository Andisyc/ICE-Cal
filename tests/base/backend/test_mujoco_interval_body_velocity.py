from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg
from unilab.dr import IntervalRandomizationPlan
from unilab.envs.locomotion.g1.joystick import G1WalkDomainRandomizationProvider


def _g1_backend(*, num_envs: int = 2) -> MuJoCoBackend:
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots/g1/scene_flat.xml")),
        num_envs,
        0.005,
        base_name="pelvis",
    )
    backend.materialize()
    return backend


def test_mujoco_applies_interval_velocity_delta_to_floating_base() -> None:
    backend = _g1_backend()
    pelvis_id = backend.get_body_id("pelvis")
    delta = np.asarray(
        [
            [[0.25, -0.50, 0.0]],
            [[-0.75, 0.125, 0.0]],
        ],
        dtype=np.float64,
    )

    assert backend.get_dr_capabilities().supports_interval_body_velocity_delta
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            body_ids=np.asarray([pelvis_id], dtype=np.int32),
            body_linear_velocity_delta=delta,
        )
    )

    np.testing.assert_allclose(backend.get_base_lin_vel(), delta[:, 0, :])


def test_mujoco_rejects_interval_velocity_delta_for_non_base_body() -> None:
    backend = _g1_backend(num_envs=1)
    torso_id = backend.get_body_id("torso_link")

    with pytest.raises(ValueError, match="floating base body"):
        backend.apply_body_linear_velocity_delta(
            np.asarray([torso_id], dtype=np.int32),
            np.zeros((1, 1, 3), dtype=np.float64),
        )


def test_fada_velocity_push_waits_for_first_full_interval() -> None:
    provider = G1WalkDomainRandomizationProvider()
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            ctrl_dt=0.02,
            fada_privileged_observation=SimpleNamespace(enabled=True),
            domain_rand=SimpleNamespace(
                push_robots=True,
                fada_push_interval_seconds=7.5,
                fada_max_push_velocity=0.8,
                push_body_name="pelvis",
            ),
        ),
        _num_envs=2,
        _backend=SimpleNamespace(get_body_id=lambda name: 1),
    )
    interval_steps = round(7.5 / 0.02)

    assert provider.build_interval_randomization_plan(env, 0) is None
    assert provider.build_interval_randomization_plan(env, interval_steps - 1) is None
    plan = provider.build_interval_randomization_plan(env, interval_steps)

    assert plan is not None
    assert plan.body_linear_velocity_delta is not None
    assert plan.body_linear_velocity_delta.shape == (2, 1, 3)
