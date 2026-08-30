"""MuJoCo overlay geometry and selection for interactive playback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

from unilab.visualization.playback_sessions import MotionOverlaySelection


def _quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _add_sphere_marker(scene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    scene.ngeom += 1
    return True


def _add_axis_arrow(scene, p0: np.ndarray, p1: np.ndarray, width: float, rgba: np.ndarray) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros((3,), dtype=np.float64),
        np.zeros((3,), dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scene.ngeom += 1
    return True


def _add_vector_arrow(
    scene,
    origin: np.ndarray,
    vector: np.ndarray,
    scale: float,
    width: float,
    rgba: np.ndarray,
    min_len: float = 1e-6,
) -> bool:
    vec = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vec))
    if length < min_len:
        return True
    p0 = np.asarray(origin, dtype=np.float64)
    p1 = p0 + vec * scale
    return _add_axis_arrow(scene, p0, p1, width, rgba)


def _local_xy_to_world_arrow(body_xmat: np.ndarray, local_xy: np.ndarray) -> np.ndarray:
    rot = np.asarray(body_xmat, dtype=np.float64).reshape(3, 3)
    forward = rot[:, 0].copy()
    left = rot[:, 1].copy()
    forward[2] = 0.0
    left[2] = 0.0

    forward_norm = float(np.linalg.norm(forward))
    left_norm = float(np.linalg.norm(left))
    if forward_norm < 1e-9:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward /= forward_norm
    if left_norm < 1e-9:
        left = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        left /= left_norm

    xy = np.asarray(local_xy, dtype=np.float64).reshape(2)
    return forward * xy[0] + left * xy[1]


def _render_motion_targets(
    viewer,
    motion_data,
    selected_indices: np.ndarray,
    marker_radius: float,
    marker_alpha: float,
    show_axes: bool,
    axis_length: float,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0

    if motion_data is None:
        return

    body_pos = motion_data.body_pos_w[0]
    body_quat = motion_data.body_quat_w[0]

    point_rgba = np.array([0.1, 0.85, 1.0, marker_alpha], dtype=np.float32)
    x_rgba = np.array([1.0, 0.35, 0.35, marker_alpha], dtype=np.float32)
    y_rgba = np.array([0.35, 1.0, 0.35, marker_alpha], dtype=np.float32)
    z_rgba = np.array([0.35, 0.55, 1.0, marker_alpha], dtype=np.float32)

    for idx in selected_indices:
        if idx < 0 or idx >= body_pos.shape[0]:
            continue

        p = body_pos[idx]
        if not _add_sphere_marker(scene, p, marker_radius, point_rgba):
            break

        if show_axes:
            rot = _quat_to_rotmat_wxyz(body_quat[idx])
            px = p + rot[:, 0] * axis_length
            py = p + rot[:, 1] * axis_length
            pz = p + rot[:, 2] * axis_length
            if not _add_axis_arrow(scene, p, px, marker_radius * 0.4, x_rgba):
                break
            if not _add_axis_arrow(scene, p, py, marker_radius * 0.4, y_rgba):
                break
            if not _add_axis_arrow(scene, p, pz, marker_radius * 0.4, z_rgba):
                break


def _render_reward_debug_targets(
    viewer,
    info: dict,
    selected_indices: np.ndarray,
    marker_radius: float,
    marker_alpha: float,
    show_axes: bool,
    axis_length: float,
    show_vel: bool,
    lin_vel_scale: float,
    ang_vel_scale: float,
    show_connectors: bool,
    show_global_anchor: bool,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0

    if not info:
        return

    motion_data = info.get("motion_data", None)
    robot_body_pos_w = info.get("robot_body_pos_w", None)
    robot_body_quat_w = info.get("robot_body_quat_w", None)
    robot_body_lin_vel_w = info.get("robot_body_lin_vel_w", None)
    robot_body_ang_vel_w = info.get("robot_body_ang_vel_w", None)
    ref_body_pos_w = info.get("reward_ref_body_pos_w", None)
    ref_body_quat_w = info.get("reward_ref_body_quat_w", None)

    if motion_data is None or robot_body_pos_w is None or robot_body_quat_w is None:
        return
    if ref_body_pos_w is None or ref_body_quat_w is None:
        return

    robot_pos = robot_body_pos_w[0]
    robot_quat = robot_body_quat_w[0]
    ref_pos = ref_body_pos_w[0]
    ref_quat = ref_body_quat_w[0]
    motion_pos = motion_data.body_pos_w[0]
    motion_quat = motion_data.body_quat_w[0]
    motion_lin_vel = motion_data.body_lin_vel_w[0]
    motion_ang_vel = motion_data.body_ang_vel_w[0]
    robot_lin_vel = robot_body_lin_vel_w[0] if robot_body_lin_vel_w is not None else None
    robot_ang_vel = robot_body_ang_vel_w[0] if robot_body_ang_vel_w is not None else None

    ref_rgba = np.array([0.15, 0.95, 0.95, marker_alpha], dtype=np.float32)
    robot_rgba = np.array([1.0, 0.62, 0.15, marker_alpha], dtype=np.float32)
    motion_rgba = np.array([0.55, 0.55, 1.0, marker_alpha * 0.7], dtype=np.float32)
    connector_rgba = np.array([1.0, 1.0, 1.0, marker_alpha * 0.55], dtype=np.float32)

    ref_lin_vel_rgba = np.array([0.0, 1.0, 1.0, marker_alpha], dtype=np.float32)
    robot_lin_vel_rgba = np.array([1.0, 0.95, 0.1, marker_alpha], dtype=np.float32)
    ref_ang_vel_rgba = np.array([0.45, 0.75, 1.0, marker_alpha], dtype=np.float32)
    robot_ang_vel_rgba = np.array([1.0, 0.45, 0.1, marker_alpha], dtype=np.float32)

    x_rgba = np.array([1.0, 0.35, 0.35, marker_alpha], dtype=np.float32)
    y_rgba = np.array([0.35, 1.0, 0.35, marker_alpha], dtype=np.float32)
    z_rgba = np.array([0.35, 0.55, 1.0, marker_alpha], dtype=np.float32)

    for idx in selected_indices:
        if idx < 0 or idx >= robot_pos.shape[0]:
            continue

        p_ref = ref_pos[idx]
        p_robot = robot_pos[idx]
        p_motion = motion_pos[idx]

        if not _add_sphere_marker(scene, p_ref, marker_radius, ref_rgba):
            break
        if not _add_sphere_marker(scene, p_robot, marker_radius, robot_rgba):
            break
        if not _add_sphere_marker(scene, p_motion, marker_radius * 0.85, motion_rgba):
            break

        if show_connectors:
            if not _add_axis_arrow(scene, p_ref, p_robot, marker_radius * 0.2, connector_rgba):
                break

        if show_axes:
            for p, q in (
                (p_ref, ref_quat[idx]),
                (p_robot, robot_quat[idx]),
                (p_motion, motion_quat[idx]),
            ):
                rot = _quat_to_rotmat_wxyz(q)
                px = p + rot[:, 0] * axis_length
                py = p + rot[:, 1] * axis_length
                pz = p + rot[:, 2] * axis_length
                if not _add_axis_arrow(scene, p, px, marker_radius * 0.35, x_rgba):
                    break
                if not _add_axis_arrow(scene, p, py, marker_radius * 0.35, y_rgba):
                    break
                if not _add_axis_arrow(scene, p, pz, marker_radius * 0.35, z_rgba):
                    break

        if show_vel:
            if not _add_vector_arrow(
                scene,
                p_motion,
                motion_lin_vel[idx],
                lin_vel_scale,
                marker_radius * 0.28,
                ref_lin_vel_rgba,
            ):
                break
            if robot_lin_vel is not None and not _add_vector_arrow(
                scene,
                p_robot,
                robot_lin_vel[idx],
                lin_vel_scale,
                marker_radius * 0.28,
                robot_lin_vel_rgba,
            ):
                break
            if not _add_vector_arrow(
                scene,
                p_motion,
                motion_ang_vel[idx],
                ang_vel_scale,
                marker_radius * 0.24,
                ref_ang_vel_rgba,
            ):
                break
            if robot_ang_vel is not None and not _add_vector_arrow(
                scene,
                p_robot,
                robot_ang_vel[idx],
                ang_vel_scale,
                marker_radius * 0.24,
                robot_ang_vel_rgba,
            ):
                break

    if show_global_anchor:
        anchor_idx = int(info.get("anchor_body_idx", 0))
        if 0 <= anchor_idx < robot_pos.shape[0]:
            anchor_motion_p = motion_pos[anchor_idx]
            anchor_motion_q = motion_quat[anchor_idx]
            anchor_robot_p = robot_pos[anchor_idx]
            anchor_robot_q = robot_quat[anchor_idx]

            anchor_motion_rgba = np.array([0.95, 0.2, 0.95, marker_alpha], dtype=np.float32)
            anchor_robot_rgba = np.array([1.0, 0.2, 0.2, marker_alpha], dtype=np.float32)
            anchor_radius = marker_radius * 1.35

            _add_sphere_marker(scene, anchor_motion_p, anchor_radius, anchor_motion_rgba)
            _add_sphere_marker(scene, anchor_robot_p, anchor_radius, anchor_robot_rgba)
            _add_axis_arrow(
                scene,
                anchor_motion_p,
                anchor_robot_p,
                marker_radius * 0.3,
                connector_rgba,
            )

            if show_axes:
                for p, q in ((anchor_motion_p, anchor_motion_q), (anchor_robot_p, anchor_robot_q)):
                    rot = _quat_to_rotmat_wxyz(q)
                    px = p + rot[:, 0] * (axis_length * 1.25)
                    py = p + rot[:, 1] * (axis_length * 1.25)
                    pz = p + rot[:, 2] * (axis_length * 1.25)
                    _add_axis_arrow(scene, p, px, marker_radius * 0.45, x_rgba)
                    _add_axis_arrow(scene, p, py, marker_radius * 0.45, y_rgba)
                    _add_axis_arrow(scene, p, pz, marker_radius * 0.45, z_rgba)


def _render_velocity_arrows(
    viewer,
    viz_data,
    focus_body_id: int,
    env: Any,
    *,
    height: float,
    scale: float,
    width: float,
    lateral_offset: float,
) -> None:
    state = getattr(env, "state", None)
    info = getattr(state, "info", None) if state is not None else None
    commands = info.get("commands") if isinstance(info, dict) else None
    if not isinstance(commands, np.ndarray) or commands.ndim != 2 or commands.shape[1] < 3:
        return

    try:
        local_linvel = env.get_local_linvel()
    except AttributeError:
        return
    if (
        not isinstance(local_linvel, np.ndarray)
        or local_linvel.ndim != 2
        or local_linvel.shape[1] < 2
    ):
        return

    body_xmat = np.asarray(viz_data.xmat[focus_body_id], dtype=np.float64)
    origin = np.asarray(viz_data.xpos[focus_body_id], dtype=np.float64).copy()
    origin[2] += float(height)
    side = _local_xy_to_world_arrow(body_xmat, np.array([0.0, 1.0], dtype=np.float64))

    target_vec = _local_xy_to_world_arrow(body_xmat, commands[0, :2])
    current_vec = _local_xy_to_world_arrow(body_xmat, local_linvel[0, :2])
    target_origin = origin + side * float(lateral_offset)
    current_origin = origin - side * float(lateral_offset)

    target_rgba = np.array([0.1, 0.95, 0.15, 0.9], dtype=np.float32)
    current_rgba = np.array([0.1, 0.45, 1.0, 0.9], dtype=np.float32)
    _add_vector_arrow(
        viewer.user_scn,
        target_origin,
        target_vec,
        scale,
        width,
        target_rgba,
    )
    _add_vector_arrow(
        viewer.user_scn,
        current_origin,
        current_vec,
        scale,
        width,
        current_rgba,
    )


LogFn = Callable[[str], None]


def prepare_motion_overlay_selection(
    env: Any,
    *,
    show_target_bodies: bool,
    show_reward_debug: bool,
    target_body_names: str,
    target_max_bodies: int,
    log: LogFn = print,
) -> MotionOverlaySelection:
    """Resolve body indices used by motion-target and reward-debug overlays."""

    if not (show_target_bodies or show_reward_debug):
        return MotionOverlaySelection(
            enabled=False,
            selected_indices=np.zeros((0,), dtype=np.int32),
        )

    if not (hasattr(env, "motion_loader") and hasattr(env, "motion_sampler")):
        log("WARNING: target/reward visualization only works for motion-tracking tasks.")
        return MotionOverlaySelection(
            enabled=False,
            selected_indices=np.zeros((0,), dtype=np.int32),
        )

    names = tuple(getattr(env.cfg, "body_names", ()))
    if len(names) == 0:
        log("WARNING: task has no body_names; cannot visualize targets.")
        return MotionOverlaySelection(
            enabled=False,
            selected_indices=np.zeros((0,), dtype=np.int32),
        )

    name_to_idx = {name: i for i, name in enumerate(names)}
    if target_body_names.strip():
        chosen = []
        for name in [n.strip() for n in target_body_names.split(",") if n.strip()]:
            if name in name_to_idx:
                chosen.append(name_to_idx[name])
            else:
                log(f"WARNING: body name not found in task body list: {name}")
        selected_indices = np.array(chosen, dtype=np.int32)
    else:
        selected_indices = np.arange(len(names), dtype=np.int32)

    if target_max_bodies > 0:
        selected_indices = selected_indices[:target_max_bodies]

    return MotionOverlaySelection(
        enabled=selected_indices.size > 0,
        selected_indices=selected_indices,
    )
