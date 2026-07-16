#!/usr/bin/env python3
"""Run a bounded MuJoCo command-transition sentinel for a distill student."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.visualization.interactive_playback import (  # noqa: E402
    RslRlPlaybackConfig,
    create_distill_playback_session,
)

COMMANDS = (
    ("forward", np.asarray([0.4, 0.0, 0.0], dtype=np.float32)),
    ("lateral", np.asarray([0.0, 0.4, 0.0], dtype=np.float32)),
    ("yaw", np.asarray([0.0, 0.0, 0.4], dtype=np.float32)),
)


def _compose_cfg(task: str) -> Any:
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "distill"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"task={task}"])


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "last": float(arr[-1]),
        "mean": float(np.mean(arr)),
    }


def _height(env: Any) -> float:
    return float(np.asarray(env._backend.get_base_pos(), dtype=np.float64)[0, 2])


def _tilt_deg(env: Any) -> float:
    upvector_name = str(env.cfg.sensor.upvector)
    upvector = np.asarray(env._backend.get_sensor_data(upvector_name), dtype=np.float64)
    return float(np.rad2deg(np.arccos(np.clip(upvector[0, 2], -1.0, 1.0))))


def _speed(env: Any) -> float:
    velocity = np.asarray(env._backend.get_base_lin_vel(), dtype=np.float64)[0]
    return float(np.linalg.norm(velocity))


def _done(env: Any) -> bool:
    state = getattr(env, "state", None)
    if state is None:
        state = getattr(env, "_state", None)
    if state is None:
        return False
    terminated = np.asarray(getattr(state, "terminated", False), dtype=bool)
    truncated = np.asarray(getattr(state, "truncated", False), dtype=bool)
    return bool(np.any(terminated | truncated))


def _action_abs_max(session: Any) -> float:
    actions = getattr(session, "actions", None)
    if actions is None:
        return 0.0
    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    return float(np.max(np.abs(np.asarray(actions, dtype=np.float64))))


def _run_phase(session: Any, steps: int) -> dict[str, Any]:
    heights: list[float] = []
    tilts: list[float] = []
    speeds: list[float] = []
    actions: list[float] = []
    done_count = 0
    for _ in range(int(steps)):
        session.step_once()
        env = session.env
        heights.append(_height(env))
        tilts.append(_tilt_deg(env))
        speeds.append(_speed(env))
        actions.append(_action_abs_max(session))
        done_count += int(_done(env))
    return {
        "steps": int(steps),
        "height": _stats(heights),
        "tilt_deg": _stats(tilts),
        "speed": _stats(speeds),
        "action_abs_max": _stats(actions),
        "done_count": done_count,
    }


def run_check(
    *,
    student_checkpoint: Path,
    task: str,
    repeats: int,
    active_steps: int,
    stop_steps: int,
    device: str,
) -> dict[str, Any]:
    cfg = _compose_cfg(task)
    playback_cfg = RslRlPlaybackConfig(
        task=str(cfg.training.task_name),
        load_run="-1",
        checkpoint=None,
        checkpoint_path=str(student_checkpoint),
        action_mode="policy",
        policy_obs_mode="actor",
        algo_log_name=str(cfg.algo.algo_log_name),
        log_root=str(cfg.training.log_root) if cfg.training.log_root else None,
        num_envs=1,
    )
    session, policy_obs_mode, resolved_checkpoint = create_distill_playback_session(
        playback_cfg=playback_cfg,
        cfg=cfg,
        root_dir=ROOT_DIR,
        device=device,
    )
    episodes: list[dict[str, Any]] = []
    try:
        for index in range(int(repeats)):
            reset_ids = np.arange(1, dtype=np.int32)
            session.env.reset(reset_ids)
            session.refresh_observation()
            session.set_external_command(np.zeros(3, dtype=np.float32))
            standing = _run_phase(session, stop_steps)
            command_name, command = COMMANDS[index % len(COMMANDS)]
            session.set_external_command(command)
            walking = _run_phase(session, active_steps)
            session.set_external_command(np.zeros(3, dtype=np.float32))
            recovered = _run_phase(session, stop_steps)
            episodes.append(
                {
                    "index": index,
                    "command": command_name,
                    "standing": standing,
                    "walking": walking,
                    "stop": recovered,
                    "stop_speed_le_active": recovered["speed"]["last"]
                    <= walking["speed"]["last"] + 0.05,
                }
            )
    finally:
        close = getattr(session.env, "close", None)
        if callable(close):
            close()

    reward_cfg = cfg.reward
    all_phases = [
        phase
        for episode in episodes
        for phase in (episode["standing"], episode["walking"], episode["stop"])
    ]
    summary = {
        "min_base_height": min(phase["height"]["min"] for phase in all_phases),
        "max_tilt_deg": max(phase["tilt_deg"]["max"] for phase in all_phases),
        "total_done_count": sum(phase["done_count"] for phase in all_phases),
        "nonzero_action_phases": sum(
            phase["action_abs_max"]["max"] > 1.0e-6 for phase in all_phases
        ),
        "stop_speed_decay_pass": all(
            bool(episode["stop_speed_le_active"]) for episode in episodes
        ),
        "task_min_base_height": float(reward_cfg.min_base_height),
        "task_max_tilt_deg": float(reward_cfg.max_tilt_deg),
    }
    summary["gate_pass"] = bool(
        summary["total_done_count"] == 0
        and summary["min_base_height"] > summary["task_min_base_height"]
        and summary["max_tilt_deg"] < summary["task_max_tilt_deg"]
        and summary["stop_speed_decay_pass"]
        and summary["nonzero_action_phases"] == len(all_phases)
    )
    command_summary: dict[str, dict[str, Any]] = {}
    for name, _command in COMMANDS:
        selected = [episode for episode in episodes if episode["command"] == name]
        command_summary[name] = {
            "episodes": len(selected),
            "done_count": sum(
                phase["done_count"]
                for episode in selected
                for phase in (episode["standing"], episode["walking"], episode["stop"])
            ),
            "min_base_height": min(
                phase["height"]["min"]
                for episode in selected
                for phase in (episode["standing"], episode["walking"], episode["stop"])
            ),
            "max_tilt_deg": max(
                phase["tilt_deg"]["max"]
                for episode in selected
                for phase in (episode["standing"], episode["walking"], episode["stop"])
            ),
            "stop_speed_decay_pass": all(
                bool(episode["stop_speed_le_active"]) for episode in selected
            ),
        }
    return {
        "student_checkpoint": str(student_checkpoint.resolve()),
        "resolved_checkpoint": resolved_checkpoint,
        "task": task,
        "policy_obs_mode": policy_obs_mode,
        "repeats": int(repeats),
        "active_steps": int(active_steps),
        "stop_steps": int(stop_steps),
        "command_grid": [name for name, _command in COMMANDS],
        "episodes": episodes,
        "summary": summary,
        "command_summary": command_summary,
        "failure_indices": [
            int(episode["index"])
            for episode in episodes
            if not episode["stop_speed_le_active"]
            or any(
                phase["done_count"] > 0
                for phase in (episode["standing"], episode["walking"], episode["stop"])
            )
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True, type=Path)
    parser.add_argument("--task", default="g1_walk_flat/mujoco")
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--active-steps", type=int, default=20)
    parser.add_argument("--stop-steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.student_checkpoint.is_file():
        raise FileNotFoundError(args.student_checkpoint)
    report = run_check(
        student_checkpoint=args.student_checkpoint,
        task=args.task,
        repeats=args.repeats,
        active_steps=args.active_steps,
        stop_steps=args.stop_steps,
        device=args.device,
    )
    print("UniLab G1 distill student transition live sentinel")
    print({key: report[key] for key in ("student_checkpoint", "resolved_checkpoint", "task")})
    print({"summary": report["summary"]})
    print({"command_summary": report["command_summary"]})
    print({"failure_indices": report["failure_indices"]})
    return 0 if report["summary"]["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
