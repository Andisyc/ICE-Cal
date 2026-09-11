from __future__ import annotations

import numpy as np

from unilab.algos.torch.distill.fada.real_target_data import (
    fit_policy_step_timeline,
    project_torso_imu,
)


def _lowcmd_event(time_ns: int, value: float) -> dict[str, object]:
    return {
        "monotonic_time_ns": time_ns,
        "payload": {"motor": {"q": [value] * 29}},
    }


def test_policy_timeline_recovers_one_step_from_each_duplicated_pair() -> None:
    events = []
    start = 1_000_000_000
    for step in range(80):
        value = float(step) * 0.01
        events.append(_lowcmd_event(start + step * 20_000_000, value))
        events.append(_lowcmd_event(start + step * 20_000_000 + 10_000_000, value))

    indices = fit_policy_step_timeline(events, control_dt=0.02)
    selected = np.asarray([events[int(index)]["payload"]["motor"]["q"][0] for index in indices])

    assert len(indices) == 80
    np.testing.assert_allclose(selected, np.arange(80) * 0.01)


def test_torso_projection_matches_deployment_identity_orientation() -> None:
    gyro, gravity = project_torso_imu(
        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 9.81]], dtype=np.float32),
    )

    np.testing.assert_allclose(gyro, [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(gravity, [[0.0, 0.0, -1.0]])
