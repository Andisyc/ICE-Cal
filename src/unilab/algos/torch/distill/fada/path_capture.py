from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unilab.base.backend.mujoco.playback import render_mujoco_states_video
from unilab.envs.common.rotation import np_yaw_from_quat


@dataclass(frozen=True)
class FADAStageCPathTrace:
    position_xy_m: np.ndarray
    yaw_rad: np.ndarray
    origin_xy_m: np.ndarray
    heading_rad: float


class FADAStageCPathCapture:
    """Own synchronized Stage-C video frames and planar base-pose samples."""

    def __init__(self, env: Any, video_path: Path) -> None:
        self.env = env
        self.video_path = video_path
        self.states: list[np.ndarray] = []
        self.positions_xy_m: list[np.ndarray] = []
        self.yaws_rad: list[float] = []
        self.origin_xy_m: np.ndarray | None = None
        self.heading_rad: float | None = None

    def capture_initial(self) -> None:
        start_position, start_yaw = self._current_base_pose()
        origin_xy_m = start_position[:2].copy()
        self.origin_xy_m = origin_xy_m
        self.heading_rad = start_yaw
        self.positions_xy_m.append(origin_xy_m.copy())
        self.yaws_rad.append(start_yaw)
        self._capture_physics_state()

    def capture_step(self) -> None:
        if self.origin_xy_m is None or self.heading_rad is None:
            raise RuntimeError("Stage C path capture requires the reset frame first")
        base_position, base_yaw = self._current_base_pose()
        self.positions_xy_m.append(base_position[:2].copy())
        self.yaws_rad.append(base_yaw)
        self._capture_physics_state()

    def _current_base_pose(self) -> tuple[np.ndarray, float]:
        position = np.asarray(self.env.get_base_pos(), dtype=np.float64)
        quat = np.asarray(self.env.get_base_quat(), dtype=np.float64)
        if position.shape != (1, 3) or quat.shape != (1, 4):
            raise ValueError(
                "Stage C path capture requires one base pose with shapes (1, 3) and (1, 4)"
            )
        if not bool(np.all(np.isfinite(position))) or not bool(np.all(np.isfinite(quat))):
            raise ValueError("Stage C path capture requires a finite base pose")
        yaw = float(np_yaw_from_quat(quat)[0])
        if not np.isfinite(yaw):
            raise ValueError("Stage C path capture requires a finite base yaw")
        return position[0], yaw

    def _capture_physics_state(self) -> None:
        self.states.append(
            np.asarray(self.env.get_physics_state_snapshot(), dtype=np.float32).copy()
        )

    def discard_terminal_frames(self, count: int) -> None:
        if count < 0 or count >= len(self.states):
            raise ValueError(f"invalid terminal frame count: {count}")
        if count:
            del self.states[-count:]
            del self.positions_xy_m[-count:]
            del self.yaws_rad[-count:]

    def write_video(self) -> None:
        render_mujoco_states_video(
            env=self.env,
            state_list=self.states,
            output_video=self.video_path,
            camera_kwargs={"cam_tracking": True, "cam_tracking_env_idx": 0},
        )

    def path_trace(self) -> FADAStageCPathTrace:
        if self.origin_xy_m is None or self.heading_rad is None:
            raise ValueError("Stage C path capture did not observe a trajectory frame")
        return FADAStageCPathTrace(
            position_xy_m=np.asarray(self.positions_xy_m, dtype=np.float64),
            yaw_rad=np.asarray(self.yaws_rad, dtype=np.float64),
            origin_xy_m=self.origin_xy_m.copy(),
            heading_rad=self.heading_rad,
        )
