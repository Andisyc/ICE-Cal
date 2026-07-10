#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Preflight the generic G1 distillation checkpoint path up to MuJoCo viewer launch."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from unilab.visualization.interactive_playback import (  # noqa: E402
    create_distill_playback_session,
)
import play_interactive  # noqa: E402


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level, name, detail))


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _split_task_owner(task_owner: str) -> tuple[str, str]:
    parts = str(task_owner).split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("--task must be an owner route like g1_stand_still/mujoco")
    return parts[0], parts[1]


def _viewer_command(
    *,
    task_owner: str,
    load_run: str,
    checkpoint: str | None,
    action_mode: str,
    device: str | None,
) -> list[str]:
    task_name, sim_name = _split_task_owner(task_owner)
    cmd = [
        "mjpython",
        "scripts/play_interactive.py",
        "--algo",
        "distill",
        "--task",
        task_name,
        "--sim",
        sim_name,
        f"algo.load_run={load_run}",
        f"interactive.action_mode={action_mode}",
    ]
    if checkpoint is not None:
        cmd.append(f"algo.checkpoint={checkpoint}")
    if device is not None:
        cmd.append(f"training.device={device}")
    return cmd


def _transfer_physics_to_viewer_model(mj_model: Any, physics: Any) -> dict[str, Any]:
    import mujoco

    physics_arr = np.asarray(physics, dtype=np.float64)
    if physics_arr.ndim != 2 or physics_arr.shape[0] < 1:
        raise ValueError(f"Expected physics state with shape (N, D), got {physics_arr.shape}")
    viz_data = mujoco.MjData(mj_model)
    mujoco.mj_setState(
        mj_model,
        viz_data,
        physics_arr[0],
        mujoco.mjtState.mjSTATE_FULLPHYSICS,
    )
    mujoco.mj_forward(mj_model, viz_data)
    return {
        "qpos_shape": list(viz_data.qpos.shape),
        "qvel_shape": list(viz_data.qvel.shape),
        "ctrl_shape": list(viz_data.ctrl.shape),
    }


def run_check(
    *,
    task: str,
    load_run: str,
    checkpoint: str | None,
    action_mode: str,
    device: str | None,
    create_session_fn=create_distill_playback_session,
    load_viewer_model_fn=play_interactive._load_viewer_model,
    state_transfer_fn=_transfer_physics_to_viewer_model,
) -> tuple[list[Check], dict[str, Any]]:
    task_name, _sim_name = _split_task_owner(task)
    cfg = play_interactive._compose_interactive_config("distill", [f"task={task}"])
    cfg.algo.load_run = str(load_run)
    cfg.algo.checkpoint = checkpoint
    cfg.interactive.action_mode = str(action_mode)
    if device is not None:
        cfg.training.device = str(device)

    args = play_interactive._build_play_args(cfg, algo="distill")
    playback_cfg = play_interactive._build_playback_config(args, num_envs=1)
    messages: list[str] = []
    checks: list[Check] = []
    command = _viewer_command(
        task_owner=task,
        load_run=str(load_run),
        checkpoint=checkpoint,
        action_mode=str(action_mode),
        device=device,
    )
    mjpython_path = shutil.which("mjpython")
    details: dict[str, Any] = {
        "distill_viewer/task": str(cfg.training.task_name),
        "distill_viewer/task_owner": str(task),
        "distill_viewer/load_run": str(load_run),
        "distill_viewer/checkpoint": checkpoint,
        "distill_viewer/action_mode": str(action_mode),
        "distill_viewer/device": device,
        "distill_viewer/cfg_student_obs_dim": int(cfg.student.obs_dim),
        "distill_viewer/cfg_teacher_obs_dim": int(cfg.teacher.obs_dim),
        "distill_viewer/mjpython_path": mjpython_path,
        "distill_viewer/viewer_command": " ".join(command),
    }

    session = None
    try:
        session, policy_obs_mode, checkpoint_path = create_session_fn(
            playback_cfg=playback_cfg,
            cfg=cfg,
            root_dir=ROOT_DIR,
            device=device,
            log=messages.append,
        )
        session.reset()
        session.step_once()
        physics = session.physics_state()
        actions = getattr(session, "actions", None)
        actions_arr = None if actions is None else np.asarray(actions, dtype=np.float64)
        actions_abs_max = None if actions_arr is None else float(np.max(np.abs(actions_arr)))

        viewer_model = load_viewer_model_fn(
            session.env,
            use_env_visual_model=bool(args.use_env_visual_model),
        )
        transfer = state_transfer_fn(viewer_model, physics)
        details.update(
            {
                "distill_viewer/policy_obs_mode": policy_obs_mode,
                "distill_viewer/checkpoint_path": checkpoint_path,
                "distill_viewer/physics_shape": list(np.asarray(physics).shape),
                "distill_viewer/actions_shape": (
                    None if actions is None else list(np.asarray(actions).shape)
                ),
                "distill_viewer/actions_abs_max": actions_abs_max,
                "distill_viewer/viewer_model_nq": int(getattr(viewer_model, "nq", -1)),
                "distill_viewer/viewer_model_nv": int(getattr(viewer_model, "nv", -1)),
                "distill_viewer/viewer_model_nu": int(getattr(viewer_model, "nu", -1)),
                "distill_viewer/state_transfer": transfer,
                "distill_viewer/messages": messages,
            }
        )

        if str(task) == "g1_stand_still/mujoco" and str(cfg.training.task_name) == "G1StandStill":
            _add(checks, "PASS", "distill_viewer/task_owner", "g1_stand_still/mujoco")
        elif task_name.startswith("g1"):
            _add(checks, "WARN", "distill_viewer/task_owner", str(task))

        if int(cfg.student.obs_dim) == 98 and int(cfg.teacher.obs_dim) == 98:
            _add(checks, "PASS", "distill_viewer/stand_still_obs_contract", "98/98")
        else:
            _add(
                checks,
                "FAIL",
                "distill_viewer/stand_still_obs_contract",
                f"{int(cfg.student.obs_dim)}/{int(cfg.teacher.obs_dim)}",
            )

        if policy_obs_mode == "actor":
            _add(checks, "PASS", "distill_viewer/policy_obs_mode", "actor")
        else:
            _add(checks, "FAIL", "distill_viewer/policy_obs_mode", str(policy_obs_mode))

        if checkpoint_path is not None and Path(checkpoint_path).is_file():
            _add(checks, "PASS", "distill_viewer/checkpoint_path", str(checkpoint_path))
        else:
            _add(checks, "FAIL", "distill_viewer/checkpoint_path", str(checkpoint_path))

        if actions_abs_max is not None and actions_abs_max > 1.0e-6:
            _add(checks, "PASS", "distill_viewer/policy_action_nonzero", f"{actions_abs_max:.6f}")
        else:
            _add(checks, "FAIL", "distill_viewer/policy_action_nonzero", str(actions_abs_max))

        if int(getattr(viewer_model, "nq", 0)) > 0 and int(getattr(viewer_model, "nv", 0)) > 0:
            _add(
                checks,
                "PASS",
                "distill_viewer/viewer_model_loaded",
                f"nq={int(getattr(viewer_model, 'nq', -1))}, nv={int(getattr(viewer_model, 'nv', -1))}",
            )
        else:
            _add(checks, "FAIL", "distill_viewer/viewer_model_loaded", "invalid model")

        _add(checks, "PASS", "distill_viewer/state_transfer", str(transfer))
        if mjpython_path is None:
            _add(
                checks,
                "WARN",
                "distill_viewer/mjpython_available",
                "mjpython not found; use the printed command only after installing MuJoCo macOS launcher",
            )
        else:
            _add(checks, "PASS", "distill_viewer/mjpython_available", mjpython_path)
    finally:
        if session is not None:
            _close_env(session.env)

    return checks, details


def print_report(checks: list[Check], details: dict[str, Any]) -> None:
    print("UniLab G1 generic distill viewer path preflight")
    for key, value in details.items():
        print(f"{key}: {value}")
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="g1_stand_still/mujoco")
    parser.add_argument("--load-run", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--action-mode", choices=("policy",), default="policy")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks, details = run_check(
        task=args.task,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
        action_mode=args.action_mode,
        device=args.device,
    )
    print_report(checks, details)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
