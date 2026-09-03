from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from unilab.envs.locomotion.g1.base import G1BaseEnv

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("angle_deg", [10, 15])
def test_slope_scene_has_exact_geometry_and_ground_independent_contacts(angle_deg: int) -> None:
    scene = ROOT / f"src/unilab/assets/robots/g1/scene_slope_{angle_deg}.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    approach_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "approach")
    slope_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"slope_{angle_deg}")

    np.testing.assert_allclose(model.geom_size[approach_id], [1.25, 0.4, 0.05])
    np.testing.assert_allclose(model.geom_pos[approach_id], [0.25, 0.0, -0.05])
    np.testing.assert_allclose(model.geom_size[slope_id], [4.0, 0.4, 0.05])
    angle = np.deg2rad(angle_deg)
    expected_slope_center = [
        1.5 + 4.0 * np.cos(angle) + 0.05 * np.sin(angle),
        0.0,
        4.0 * np.sin(angle) - 0.05 * np.cos(angle),
    ]
    np.testing.assert_allclose(model.geom_pos[slope_id], expected_slope_center, atol=1e-3)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    slope_x_axis = data.geom_xmat[slope_id].reshape(3, 3)[:, 0]
    assert np.rad2deg(np.arctan2(slope_x_axis[2], slope_x_axis[0])) == pytest.approx(angle_deg)

    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, index)
        for index in range(model.nsensor)
    }
    assert {f"left_foot_contact_{index}" for index in range(4)} <= names
    assert {f"right_foot_contact_{index}" for index in range(4)} <= names
    assert {"left_foot_net_contact", "right_foot_net_contact"} <= names
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1


class _SensorBackend:
    def get_sensor_data(self, name: str) -> np.ndarray:
        values = {
            "left_foot_pos": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "right_foot_pos": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
        }
        return values[name]


class _G1BaseProbe(G1BaseEnv):
    def update_state(self) -> None:
        pass


def test_g1_task_exposes_ordered_foot_world_positions() -> None:
    env = object.__new__(_G1BaseProbe)
    env._backend = _SensorBackend()  # type: ignore[attr-defined]

    observed = env.get_foot_pos()

    assert observed.shape == (1, 2, 3)
    np.testing.assert_array_equal(observed[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(observed[0, 1], [4.0, 5.0, 6.0])


def test_g1_task_rejects_malformed_foot_sensor_shape() -> None:
    env = object.__new__(_G1BaseProbe)
    env._backend = SimpleNamespace(  # type: ignore[attr-defined]
        get_sensor_data=lambda _name: np.zeros((1, 2), dtype=np.float32)
    )

    with pytest.raises(ValueError, match="foot position sensors"):
        env.get_foot_pos()
