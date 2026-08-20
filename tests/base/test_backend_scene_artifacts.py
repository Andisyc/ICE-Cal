"""Public cold-path scene-artifact contract tests."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from unilab.base.backend.base import BackendSceneArtifacts, SimBackend
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.base.scene import SceneCfg


def test_backend_scene_artifacts_is_an_immutable_empty_projection_by_default() -> None:
    artifacts = SimBackend.get_scene_artifacts(object())

    assert artifacts == BackendSceneArtifacts()
    with pytest.raises(FrozenInstanceError):
        artifacts.model_file = "mutated.xml"  # type: ignore[misc]


def test_np_env_forwards_backend_scene_artifacts_without_private_field_probing() -> None:
    expected = BackendSceneArtifacts(
        model_file="robot.xml",
        visual_model_file="scene.xml",
        artifacts_dir="generated",
    )

    class FakeBackend:
        def get_scene_artifacts(self) -> BackendSceneArtifacts:
            return expected

    class FakeEnv(NpEnv):
        @property
        def action_space(self):
            return spaces.Box(-1.0, 1.0, shape=(1,))

        def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
            return actions

        def update_state(self, state: NpEnvState) -> NpEnvState:
            return state

    env = FakeEnv.__new__(FakeEnv)
    env._backend = FakeBackend()  # type: ignore[assignment]

    assert env.get_scene_artifacts() is expected


def test_mujoco_fragment_artifact_exists_until_backend_cleanup() -> None:
    from unilab.base.backend.mujoco.backend import MuJoCoBackend
    from unilab.envs.locomotion.g1.joystick import G1StandStillCfg

    cfg = G1StandStillCfg()
    backend = MuJoCoBackend(
        cfg.scene,
        num_envs=1,
        sim_dt=cfg.sim_dt,
        base_name=cfg.asset.base_name,
    )
    visual_model_file = backend.get_scene_artifacts().visual_model_file

    assert backend.get_scene_artifacts().model_file == cfg.scene.model_file
    assert visual_model_file is not None
    assert Path(visual_model_file).is_file()
    assert visual_model_file != cfg.scene.model_file

    backend.cleanup_scene_assets()

    assert not Path(visual_model_file).exists()


@pytest.mark.parametrize(
    ("artifacts", "expected"),
    [
        pytest.param(
            BackendSceneArtifacts(visual_model_file="materialized.xml"),
            "materialized.xml",
            id="backend-visual-model",
        ),
        pytest.param(BackendSceneArtifacts(), "configured.xml", id="configured-fallback"),
    ],
)
def test_mujoco_playback_model_resolution_preserves_visual_then_configured_priority(
    artifacts: BackendSceneArtifacts,
    expected: str,
) -> None:
    from unilab.base.backend.mujoco.playback import _visual_model_file

    class FakeEnv:
        cfg = SimpleNamespace(scene=SceneCfg(model_file="configured.xml"))

        def get_scene_artifacts(self) -> BackendSceneArtifacts:
            return artifacts

    assert _visual_model_file(FakeEnv()) == expected
