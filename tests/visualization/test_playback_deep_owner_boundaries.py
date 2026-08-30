from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _owner(name: str):
    try:
        return importlib.import_module(f"unilab.visualization.{name}")
    except ModuleNotFoundError as exc:
        raise AssertionError(f"missing playback owner module: {name}") from exc


def test_trace_owner_formats_one_environment_row() -> None:
    trace = _owner("playback_trace")

    row = trace._first_env_row(np.array([[1.0, -2.0, 3.0]], dtype=np.float32))

    assert trace._format_trace_vector(row, max_items=2) == "[+1.000,-2.000]"


def test_overlay_owner_rotates_local_xy_into_world() -> None:
    overlay = _owner("playback_overlay")
    body_xmat = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    world = overlay._local_xy_to_world_arrow(body_xmat, np.array([2.0, 0.0]))

    np.testing.assert_allclose(world, np.array([0.0, 2.0, 0.0]))


def test_viewer_owns_model_loading_and_launch_helpers() -> None:
    viewer_helpers = {
        "_resolve_focus_body_id",
        "_has_generated_terrain",
        "_default_viewer_camera_distance",
        "_available_backends_for_task",
        "_can_launch_glfw_viewer",
        "_uses_native_mujoco_viewer_launch",
        "_load_mujoco_model_file_for_viewer",
        "_load_resolved_visual_viewer_model",
        "_load_viewer_model",
    }

    overlay_path = Path("src/unilab/visualization/playback_overlay.py")
    viewer_path = Path("src/unilab/visualization/playback_viewer.py")
    overlay_tree = ast.parse(overlay_path.read_text())
    viewer_tree = ast.parse(viewer_path.read_text())
    overlay_definitions = {
        node.name for node in overlay_tree.body if isinstance(node, ast.FunctionDef)
    }
    viewer_definitions = {
        node.name for node in viewer_tree.body if isinstance(node, ast.FunctionDef)
    }

    assert viewer_helpers.isdisjoint(overlay_definitions)
    assert viewer_helpers <= viewer_definitions


def test_control_owner_builds_playback_config_without_environment_state() -> None:
    controls = _owner("playback_controls")
    args = SimpleNamespace(
        task="G1WalkFlat",
        load_run="run",
        checkpoint="model.pt",
        checkpoint_path=None,
        action_mode="policy",
        policy_obs_mode="actor",
        algo_log_name="distill",
        log_root=None,
        speed=1.5,
        start_paused=True,
        keyboard=False,
    )

    config = controls._build_playback_config(args, num_envs=2)

    assert config.task == "G1WalkFlat"
    assert config.num_envs == 2
    assert config.speed == 1.5
    assert config.start_paused is True


def test_distill_policy_owner_is_separate_from_session_factory() -> None:
    owner = _owner("playback_distill_policy")

    assert callable(owner.load_distill_playback_policy)
