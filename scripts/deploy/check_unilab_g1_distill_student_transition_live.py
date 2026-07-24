#!/usr/bin/env python3
"""Run a bounded MuJoCo command-transition sentinel for a distill student.

Each repetition starts from a fully initialized env state. Automatic reset is
disabled so termination diagnostics describe the terminal frame rather than
the reset state that follows it.
"""

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

from unilab.training.seed import apply_training_seed  # noqa: E402
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


def _empty_stats() -> dict[str, None]:
    return {"min": None, "max": None, "last": None, "mean": None}


def _height(env: Any) -> float:
    return float(np.asarray(env._backend.get_base_pos(), dtype=np.float64)[0, 2])


def _tilt_deg(env: Any) -> float:
    upvector_name = str(env.cfg.sensor.upvector)
    upvector = np.asarray(env._backend.get_sensor_data(upvector_name), dtype=np.float64)
    return float(np.rad2deg(np.arccos(np.clip(upvector[0, 2], -1.0, 1.0))))


def _speed(env: Any) -> float:
    velocity = np.asarray(env._backend.get_base_lin_vel(), dtype=np.float64)[0]
    return float(np.linalg.norm(velocity))


def _done_counts(env: Any) -> tuple[int, int]:
    state = getattr(env, "state", None)
    if state is None:
        state = getattr(env, "_state", None)
    if state is None:
        return 0, 0
    terminated = np.asarray(getattr(state, "terminated", False), dtype=bool)
    truncated = np.asarray(getattr(state, "truncated", False), dtype=bool)
    return int(np.count_nonzero(terminated)), int(np.count_nonzero(truncated))


def _state_step(env: Any) -> int | None:
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    if not isinstance(info, dict):
        return None
    steps = info.get("steps")
    if steps is None or np.asarray(steps).size == 0:
        return None
    return int(np.asarray(steps).reshape(-1)[0])


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
    terminated_count = 0
    truncated_count = 0
    terminal_snapshot: dict[str, Any] | None = None
    for phase_step in range(int(steps)):
        session.step_once()
        env = session.env
        heights.append(_height(env))
        tilts.append(_tilt_deg(env))
        speeds.append(_speed(env))
        actions.append(_action_abs_max(session))
        step_terminated, step_truncated = _done_counts(env)
        terminated_count += step_terminated
        truncated_count += step_truncated
        if step_terminated > 0 or step_truncated > 0:
            terminal_snapshot = {
                "phase_step": int(phase_step + 1),
                "state_step": _state_step(env),
                "terminated": step_terminated > 0,
                "truncated": step_truncated > 0,
                "terminated_rows": step_terminated,
                "truncated_rows": step_truncated,
                "height": heights[-1],
                "tilt_deg": tilts[-1],
                "speed": speeds[-1],
            }
            break
    return {
        "steps": int(steps),
        "executed_steps": len(heights),
        "height": _stats(heights),
        "tilt_deg": _stats(tilts),
        "speed": _stats(speeds),
        "action_abs_max": _stats(actions),
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "done_count": terminated_count + truncated_count,
        "terminal_snapshot": terminal_snapshot,
        "skip_reason": None,
    }


def _skipped_phase(*, steps: int, reason: str) -> dict[str, Any]:
    return {
        "steps": int(steps),
        "executed_steps": 0,
        "height": _empty_stats(),
        "tilt_deg": _empty_stats(),
        "speed": _empty_stats(),
        "action_abs_max": _empty_stats(),
        "terminated_count": 0,
        "truncated_count": 0,
        "done_count": 0,
        "terminal_snapshot": None,
        "skip_reason": reason,
    }


def _phase_completed(phase: dict[str, Any]) -> bool:
    return bool(
        int(phase["executed_steps"]) == int(phase["steps"]) and int(phase["done_count"]) == 0
    )


def _phase_metric_values(phases: list[dict[str, Any]], metric: str, field: str) -> list[float]:
    values: list[float] = []
    for phase in phases:
        stats = phase.get(metric)
        value = stats.get(field) if isinstance(stats, dict) else None
        if value is not None:
            values.append(float(value))
    return values


def _stop_speed_decay_pass(walking: dict[str, Any], stopped: dict[str, Any]) -> bool:
    if not _phase_completed(walking) or not _phase_completed(stopped):
        return False
    return bool(float(stopped["speed"]["last"]) <= float(walking["speed"]["last"]) + 0.05)


def _reset_episode(session: Any) -> dict[str, float | int]:
    """Create and synchronize one clean manual-reset state for the sentinel."""

    env = session.env
    init_state = getattr(env, "init_state", None)
    if not callable(init_state):
        raise RuntimeError("transition sentinel requires env.init_state() for full reset")
    state = init_state()
    if state is None:
        state = getattr(env, "state", None)
    if state is None or not isinstance(getattr(state, "info", None), dict):
        raise RuntimeError("transition sentinel reset did not produce env state/info")

    steps = np.asarray(state.info.get("steps"))
    if steps.size == 0 or np.any(steps != 0):
        raise RuntimeError(f"transition sentinel reset steps must be zero, got {steps!r}")
    np.asarray(state.terminated, dtype=bool).fill(False)
    np.asarray(state.truncated, dtype=bool).fill(False)
    if hasattr(state, "final_observation"):
        state.final_observation = None

    session.action_obs = None
    session.actions = None
    session.step_count = 0
    session.refresh_observation()

    commands = state.info.get("commands")
    commands_arr = np.asarray(commands)
    if commands_arr.ndim != 2 or commands_arr.shape[1] < 3:
        raise RuntimeError(
            "transition sentinel reset requires state.info['commands'] with shape (N, >=3)"
        )
    session.set_external_command(np.zeros(3, dtype=commands_arr.dtype))
    command_max_abs = float(np.max(np.abs(np.asarray(state.info["commands"])[:, :3])))
    if command_max_abs != 0.0:
        raise RuntimeError(
            f"transition sentinel reset command must be zero, got max_abs={command_max_abs}"
        )
    return {
        "step_count": int(np.asarray(state.info["steps"]).reshape(-1)[0]),
        "command_max_abs": command_max_abs,
    }


def run_check(
    *,
    student_checkpoint: Path,
    task: str,
    repeats: int,
    active_steps: int,
    stop_steps: int,
    device: str,
    seed: int = 1,
) -> dict[str, Any]:
    if int(repeats) <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if int(active_steps) <= 0 or int(stop_steps) <= 0:
        raise ValueError(
            "active_steps and stop_steps must be positive, "
            f"got active_steps={active_steps} stop_steps={stop_steps}"
        )
    effective_seed = apply_training_seed(
        int(seed),
        torch_runtime=True,
        cuda=str(device).startswith("cuda"),
    )
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
    set_autoreset = getattr(session.env, "set_autoreset", None)
    if not callable(set_autoreset):
        raise RuntimeError("transition sentinel requires env.set_autoreset(False)")
    set_autoreset(False)
    episodes: list[dict[str, Any]] = []
    terminal_events: list[dict[str, Any]] = []
    try:
        for index in range(int(repeats)):
            reset_snapshot = _reset_episode(session)
            command_name, command = COMMANDS[index % len(COMMANDS)]
            standing = _run_phase(session, stop_steps)
            if standing["done_count"] > 0:
                walking = _skipped_phase(steps=active_steps, reason="standing_done")
                recovered = _skipped_phase(steps=stop_steps, reason="standing_done")
            else:
                session.set_external_command(command)
                walking = _run_phase(session, active_steps)
                if walking["done_count"] > 0:
                    recovered = _skipped_phase(steps=stop_steps, reason="walking_done")
                else:
                    session.set_external_command(np.zeros(3, dtype=np.float32))
                    recovered = _run_phase(session, stop_steps)

            stop_speed_le_active = _stop_speed_decay_pass(walking, recovered)
            episode = {
                "index": index,
                "command": command_name,
                "reset_step_count": int(reset_snapshot["step_count"]),
                "reset_command_max_abs": float(reset_snapshot["command_max_abs"]),
                "standing": standing,
                "walking": walking,
                "stop": recovered,
                "stop_speed_le_active": stop_speed_le_active,
            }
            episodes.append(episode)
            for phase_name, phase in (
                ("standing", standing),
                ("walking", walking),
                ("stop", recovered),
            ):
                terminal_snapshot = phase["terminal_snapshot"]
                if terminal_snapshot is not None:
                    terminal_events.append(
                        {
                            "episode_index": index,
                            "command": command_name,
                            "phase": phase_name,
                            **terminal_snapshot,
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
    height_mins = _phase_metric_values(all_phases, "height", "min")
    tilt_maxes = _phase_metric_values(all_phases, "tilt_deg", "max")
    if not height_mins or not tilt_maxes:
        raise RuntimeError("transition sentinel did not execute any physical phase")
    expected_phase_count = len(episodes) * 3
    summary = {
        "min_base_height": min(height_mins),
        "max_tilt_deg": max(tilt_maxes),
        "total_terminated_count": sum(int(phase["terminated_count"]) for phase in all_phases),
        "total_truncated_count": sum(int(phase["truncated_count"]) for phase in all_phases),
        "total_done_count": sum(phase["done_count"] for phase in all_phases),
        "completed_phase_count": sum(_phase_completed(phase) for phase in all_phases),
        "expected_phase_count": expected_phase_count,
        "nonzero_action_phases": sum(
            phase["action_abs_max"]["max"] is not None and phase["action_abs_max"]["max"] > 1.0e-6
            for phase in all_phases
        ),
        "stop_speed_decay_pass": all(bool(episode["stop_speed_le_active"]) for episode in episodes),
        "task_min_base_height": float(reward_cfg.min_base_height),
        "task_max_tilt_deg": float(reward_cfg.max_tilt_deg),
    }
    summary["gate_pass"] = bool(
        summary["total_done_count"] == 0
        and summary["min_base_height"] > summary["task_min_base_height"]
        and summary["max_tilt_deg"] < summary["task_max_tilt_deg"]
        and summary["stop_speed_decay_pass"]
        and summary["completed_phase_count"] == expected_phase_count
        and summary["nonzero_action_phases"] == expected_phase_count
    )
    command_summary: dict[str, dict[str, Any]] = {}
    for name, _command in COMMANDS:
        selected = [episode for episode in episodes if episode["command"] == name]
        selected_phases = [
            phase
            for episode in selected
            for phase in (episode["standing"], episode["walking"], episode["stop"])
        ]
        height_mins = _phase_metric_values(selected_phases, "height", "min")
        tilt_maxes = _phase_metric_values(selected_phases, "tilt_deg", "max")
        command_summary[name] = {
            "episodes": len(selected),
            "terminated_count": sum(int(phase["terminated_count"]) for phase in selected_phases),
            "truncated_count": sum(int(phase["truncated_count"]) for phase in selected_phases),
            "done_count": sum(int(phase["done_count"]) for phase in selected_phases),
            "min_base_height": min(height_mins) if height_mins else None,
            "max_tilt_deg": max(tilt_maxes) if tilt_maxes else None,
            "stop_speed_decay_pass": all(
                bool(episode["stop_speed_le_active"]) for episode in selected
            ),
        }
    return {
        "student_checkpoint": str(student_checkpoint.resolve()),
        "resolved_checkpoint": resolved_checkpoint,
        "task": task,
        "policy_obs_mode": policy_obs_mode,
        "seed": effective_seed,
        "repeats": int(repeats),
        "active_steps": int(active_steps),
        "stop_steps": int(stop_steps),
        "command_grid": [name for name, _command in COMMANDS],
        "episodes": episodes,
        "terminal_events": terminal_events,
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
        "termination_indices": [
            int(episode["index"])
            for episode in episodes
            if any(
                phase["terminated_count"] > 0
                for phase in (episode["standing"], episode["walking"], episode["stop"])
            )
        ],
        "truncation_indices": [
            int(episode["index"])
            for episode in episodes
            if any(
                phase["truncated_count"] > 0
                for phase in (episode["standing"], episode["walking"], episode["stop"])
            )
        ],
        "stop_decay_failure_indices": [
            int(episode["index"])
            for episode in episodes
            if not episode["stop_speed_le_active"]
            and not any(
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
    parser.add_argument("--seed", type=int, default=1)
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
        seed=args.seed,
    )
    print("UniLab G1 distill student transition live sentinel")
    print(
        {key: report[key] for key in ("student_checkpoint", "resolved_checkpoint", "task", "seed")}
    )
    print({"summary": report["summary"]})
    print({"command_summary": report["command_summary"]})
    print({"terminal_events": report["terminal_events"]})
    print(
        {
            "failure_indices": report["failure_indices"],
            "termination_indices": report["termination_indices"],
            "truncation_indices": report["truncation_indices"],
            "stop_decay_failure_indices": report["stop_decay_failure_indices"],
        }
    )
    return 0 if report["summary"]["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
