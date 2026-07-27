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

from unilab.envs.locomotion.g1.joystick import (  # noqa: E402
    LEFT_FOOT_CONTACT_SENSORS,
    RIGHT_FOOT_CONTACT_SENSORS,
    compute_aggregated_foot_contact,
)
from unilab.training.g1_stand_height_acceptance import (  # noqa: E402
    EXPECTED_ACTION_DIM,
    EXPECTED_OBS_DIM,
    TARGET_OBS_INDEX,
    RolloutSamples,
    evaluate_samples,
)
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

NOMINAL_WALK_TARGET_HEIGHT = 0.754
DEFAULT_POST_WALK_TARGET_HEIGHTS = (0.650, 0.702, 0.754)
DEFAULT_HEIGHT_RECOVERY_NOMINAL_SETTLE_STEPS = 100
DEFAULT_HEIGHT_RECOVERY_WARMUP_STEPS = 100
DEFAULT_HEIGHT_RECOVERY_EVALUATION_STEPS = 800
HEIGHT_RECOVERY_MAX_HEIGHT_MAE = 0.05
HEIGHT_RECOVERY_MIN_DOUBLE_SUPPORT_FRACTION = 0.90


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


def _action_rows(session: Any, *, rows: int) -> np.ndarray:
    actions = getattr(session, "actions", None)
    if actions is None:
        raise RuntimeError("height recovery requires policy actions for every sampled step")
    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    action_rows = np.asarray(actions, dtype=np.float32)
    expected_shape = (int(rows), EXPECTED_ACTION_DIM)
    if action_rows.shape != expected_shape:
        raise RuntimeError(
            f"height recovery actions must have shape {expected_shape}, got {action_rows.shape}"
        )
    return action_rows


def _transition_input_snapshot(
    env: Any,
    *,
    command: np.ndarray,
    target_height: float,
) -> dict[str, Any]:
    """Read the current 99-D transition contract without refreshing env state."""

    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    obs = getattr(state, "obs", None)
    if state is None or not isinstance(info, dict) or not isinstance(obs, dict):
        raise RuntimeError("transition input snapshot requires dict state.info and state.obs")
    observed_commands = np.asarray(info.get("commands"))
    if (
        observed_commands.ndim != 2
        or observed_commands.shape[0] <= 0
        or observed_commands.shape[1] < 3
    ):
        raise RuntimeError(
            "transition input snapshot requires state.info['commands'] with shape (N, >=3)"
        )
    command_arr = np.asarray(command, dtype=observed_commands.dtype)
    if command_arr.shape != (3,):
        raise ValueError(f"transition command must have shape (3,), got {command_arr.shape}")
    expected_commands = np.broadcast_to(command_arr, (observed_commands.shape[0], 3))

    observed_targets = np.asarray(info.get("height_commands"))
    if observed_targets.shape != (observed_commands.shape[0], 1):
        raise RuntimeError(
            "99-D transition input snapshot requires state.info['height_commands'] "
            f"with shape ({observed_commands.shape[0]}, 1), got {observed_targets.shape}"
        )
    observed_targets = observed_targets[:, 0]
    expected_targets = np.full(
        (observed_commands.shape[0],), float(target_height), dtype=observed_targets.dtype
    )
    actor_obs = np.asarray(obs.get("obs"), dtype=np.float32)
    if actor_obs.shape != (observed_commands.shape[0], EXPECTED_OBS_DIM):
        raise RuntimeError(
            "99-D transition input snapshot requires actor observation shape "
            f"({observed_commands.shape[0]}, {EXPECTED_OBS_DIM}), got {actor_obs.shape}"
        )

    command_max_error = float(np.max(np.abs(observed_commands[:, :3] - expected_commands)))
    target_max_error = float(np.max(np.abs(observed_targets - expected_targets)))
    target_obs_max_error = float(np.max(np.abs(actor_obs[:, TARGET_OBS_INDEX] - expected_targets)))
    return {
        "passed": bool(
            command_max_error <= 1.0e-6
            and target_max_error <= 1.0e-6
            and target_obs_max_error <= 1.0e-6
        ),
        "command_max_error": command_max_error,
        "target_max_error": target_max_error,
        "target_obs_max_error": target_obs_max_error,
        "target_height_min": float(np.min(observed_targets)),
        "target_height_max": float(np.max(observed_targets)),
    }


def _sync_transition_inputs(
    session: Any,
    *,
    command: np.ndarray,
    target_height: float,
) -> dict[str, Any]:
    """Synchronize velocity, target height, env state, and policy observation."""

    env = session.env
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    if state is None or not isinstance(info, dict):
        raise RuntimeError("transition input synchronization requires env.state.info")

    commands = info.get("commands")
    commands_arr = np.asarray(commands)
    if (
        not isinstance(commands, np.ndarray)
        or commands_arr.ndim != 2
        or commands_arr.shape[0] <= 0
        or commands_arr.shape[1] < 3
    ):
        raise RuntimeError(
            "transition input synchronization requires state.info['commands'] with shape (N, >=3)"
        )
    command_arr = np.asarray(command, dtype=commands_arr.dtype)
    if command_arr.shape != (3,):
        raise ValueError(f"transition command must have shape (3,), got {command_arr.shape}")
    commands_arr[:, :3] = np.broadcast_to(command_arr, (commands_arr.shape[0], 3))
    height_commands = info.get("height_commands")
    if not isinstance(height_commands, np.ndarray) or height_commands.shape != (
        commands_arr.shape[0],
        1,
    ):
        observed_shape = getattr(height_commands, "shape", None)
        raise RuntimeError(
            "99-D transition input synchronization requires "
            f"state.info['height_commands'] with shape ({commands_arr.shape[0]}, 1), "
            f"got {observed_shape}"
        )
    if not np.isfinite(float(target_height)):
        raise ValueError(f"target_height must be finite, got {target_height}")
    height_commands[:, 0] = np.asarray(target_height, dtype=height_commands.dtype)

    refresh_state = getattr(env, "refresh_state", None)
    if not callable(refresh_state):
        raise RuntimeError("transition input synchronization requires env.refresh_state()")
    refresh_state()
    session.refresh_observation()
    return _transition_input_snapshot(
        env,
        command=command_arr,
        target_height=target_height,
    )


def _sync_summary(snapshots: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not snapshots:
        return None
    return {
        "sync_count": len(snapshots),
        "passed": all(bool(snapshot["passed"]) for snapshot in snapshots),
        "command_max_error": max(float(item["command_max_error"]) for item in snapshots),
        "target_max_error": max(float(item["target_max_error"]) for item in snapshots),
        "target_obs_max_error": max(float(item["target_obs_max_error"]) for item in snapshots),
        "target_height_min": min(float(item["target_height_min"]) for item in snapshots),
        "target_height_max": max(float(item["target_height_max"]) for item in snapshots),
    }


def _append_height_recovery_sample(
    samples: RolloutSamples,
    session: Any,
    *,
    scored: bool,
) -> None:
    env = session.env
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    obs = getattr(state, "obs", None)
    if state is None or not isinstance(info, dict) or not isinstance(obs, dict):
        raise RuntimeError("height recovery sample requires dict env state info and obs")
    actor_obs = np.asarray(obs.get("obs"), dtype=np.float32)
    if actor_obs.ndim != 2 or actor_obs.shape[1] != EXPECTED_OBS_DIM:
        raise RuntimeError(
            f"height recovery actor obs must have shape (N, {EXPECTED_OBS_DIM}), "
            f"got {actor_obs.shape}"
        )
    rows = int(actor_obs.shape[0])
    measured_height_fn = getattr(env, "_terrain_relative_base_height", None)
    if not callable(measured_height_fn):
        raise RuntimeError("height recovery requires env._terrain_relative_base_height()")
    upvector = np.asarray(
        env._backend.get_sensor_data(env.cfg.sensor.upvector), dtype=np.float32
    ).reshape(rows, -1)
    if upvector.shape[1] < 3:
        raise RuntimeError(f"upvector sensor must expose at least 3 values, got {upvector.shape}")
    left_contact = compute_aggregated_foot_contact(env._backend, LEFT_FOOT_CONTACT_SENSORS)
    right_contact = compute_aggregated_foot_contact(env._backend, RIGHT_FOOT_CONTACT_SENSORS)
    samples.append(
        target_height=np.asarray(info.get("height_commands"), dtype=np.float32),
        measured_height=np.asarray(measured_height_fn(), dtype=np.float32),
        double_support=np.asarray(left_contact & right_contact, dtype=bool),
        tilt_deg=np.rad2deg(np.arccos(np.clip(upvector[:, 2], -1.0, 1.0))).astype(np.float32),
        terminated=np.asarray(state.terminated, dtype=bool),
        truncated=np.asarray(state.truncated, dtype=bool),
        commands=np.asarray(info.get("commands"), dtype=np.float32)[:, :3],
        target_obs=actor_obs[:, TARGET_OBS_INDEX],
        actions=_action_rows(session, rows=rows),
        scored=scored,
    )


def _run_phase(
    session: Any,
    steps: int,
    *,
    command: np.ndarray | None = None,
    target_height: float | None = None,
    samples: RolloutSamples | None = None,
    warmup_steps: int = 0,
) -> dict[str, Any]:
    if (command is None) != (target_height is None):
        raise ValueError("controlled phase requires both command and target_height")
    heights: list[float] = []
    tilts: list[float] = []
    speeds: list[float] = []
    actions: list[float] = []
    sync_snapshots: list[dict[str, Any]] = []
    terminated_count = 0
    truncated_count = 0
    terminal_snapshot: dict[str, Any] | None = None
    if command is not None and target_height is not None:
        sync_snapshots.append(
            _sync_transition_inputs(
                session,
                command=command,
                target_height=target_height,
            )
        )
    for phase_step in range(int(steps)):
        session.step_once()
        if command is not None and target_height is not None:
            sync_snapshots.append(
                _transition_input_snapshot(
                    session.env,
                    command=command,
                    target_height=target_height,
                )
            )
        env = session.env
        heights.append(_height(env))
        tilts.append(_tilt_deg(env))
        speeds.append(_speed(env))
        actions.append(_action_abs_max(session))
        if samples is not None:
            _append_height_recovery_sample(
                samples,
                session,
                scored=phase_step >= int(warmup_steps),
            )
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
        "input_sync": _sync_summary(sync_snapshots),
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
        "input_sync": None,
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

    commands_arr = np.asarray(state.info.get("commands"))
    sync = _sync_transition_inputs(
        session,
        command=np.zeros(3, dtype=commands_arr.dtype),
        target_height=NOMINAL_WALK_TARGET_HEIGHT,
    )
    state = getattr(env, "state", None)
    if state is None or not isinstance(getattr(state, "info", None), dict):
        raise RuntimeError("transition sentinel reset synchronization lost env state")
    command_max_abs = float(np.max(np.abs(np.asarray(state.info["commands"])[:, :3])))
    if command_max_abs != 0.0:
        raise RuntimeError(
            f"transition sentinel reset command must be zero, got max_abs={command_max_abs}"
        )
    return {
        "step_count": int(np.asarray(state.info["steps"]).reshape(-1)[0]),
        "command_max_abs": command_max_abs,
        "target_height": float(NOMINAL_WALK_TARGET_HEIGHT),
        "target_obs_max_error": float(sync["target_obs_max_error"]),
    }


def _controlled_phase_pass(phase: dict[str, Any]) -> bool:
    return bool(_phase_completed(phase) and _phase_input_sync_pass(phase))


def _phase_input_sync_pass(phase: dict[str, Any]) -> bool:
    sync = phase.get("input_sync")
    return bool(isinstance(sync, dict) and bool(sync.get("passed", False)))


def _phase_input_sync_detail(phase: dict[str, Any]) -> str:
    sync = phase.get("input_sync")
    if not isinstance(sync, dict):
        return "input_sync=missing"
    return (
        f"command_max_error={float(sync['command_max_error']):.9g} "
        f"target_max_error={float(sync['target_max_error']):.9g} "
        f"target_obs_max_error={float(sync['target_obs_max_error']):.9g}"
    )


def _transition_check(passed: bool, name: str, detail: str) -> dict[str, str]:
    return {
        "level": "PASS" if passed else "FAIL",
        "name": name,
        "detail": detail,
    }


def _run_height_recovery_grid(
    session: Any,
    *,
    target_heights: tuple[float, ...],
    active_steps: int,
    standing_steps: int,
    nominal_settle_steps: int,
    warmup_steps: int,
    evaluation_steps: int,
    max_tilt_deg: float,
) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    terminal_events: list[dict[str, Any]] = []
    zero_command = np.zeros(3, dtype=np.float32)
    recovery_steps = int(warmup_steps) + int(evaluation_steps)

    for command_name, command in COMMANDS:
        for target_height in target_heights:
            scenario_index = len(scenarios)
            reset_snapshot = _reset_episode(session)
            standing = _run_phase(
                session,
                standing_steps,
                command=zero_command,
                target_height=NOMINAL_WALK_TARGET_HEIGHT,
            )
            if standing["done_count"] > 0:
                walking = _skipped_phase(steps=active_steps, reason="standing_done")
            else:
                walking = _run_phase(
                    session,
                    active_steps,
                    command=command,
                    target_height=NOMINAL_WALK_TARGET_HEIGHT,
                )

            if standing["done_count"] > 0:
                settling = _skipped_phase(
                    steps=nominal_settle_steps,
                    reason="standing_done",
                )
            elif walking["done_count"] > 0:
                settling = _skipped_phase(
                    steps=nominal_settle_steps,
                    reason="walking_done",
                )
            else:
                settling = _run_phase(
                    session,
                    nominal_settle_steps,
                    command=zero_command,
                    target_height=NOMINAL_WALK_TARGET_HEIGHT,
                )

            samples = RolloutSamples()
            if standing["done_count"] > 0:
                recovery_phase = _skipped_phase(
                    steps=recovery_steps,
                    reason="standing_done",
                )
            elif walking["done_count"] > 0:
                recovery_phase = _skipped_phase(
                    steps=recovery_steps,
                    reason="walking_done",
                )
            elif settling["done_count"] > 0:
                recovery_phase = _skipped_phase(
                    steps=recovery_steps,
                    reason="settling_done",
                )
            else:
                recovery_phase = _run_phase(
                    session,
                    recovery_steps,
                    command=zero_command,
                    target_height=float(target_height),
                    samples=samples,
                    warmup_steps=warmup_steps,
                )

            recovery_report = evaluate_samples(
                samples,
                expected_target_height=float(target_height),
                max_height_mae=HEIGHT_RECOVERY_MAX_HEIGHT_MAE,
                min_double_support_fraction=HEIGHT_RECOVERY_MIN_DOUBLE_SUPPORT_FRACTION,
                max_tilt_deg=float(max_tilt_deg),
                requested_steps=recovery_steps,
                executed_steps=int(recovery_phase["executed_steps"]),
            )
            transition_checks = [
                _transition_check(
                    _phase_completed(standing),
                    "transition/standing_completed",
                    (
                        f"executed_steps={standing['executed_steps']} "
                        f"requested_steps={standing['steps']}"
                    ),
                ),
                _transition_check(
                    _phase_input_sync_pass(standing),
                    "transition/standing_input_synchronized",
                    _phase_input_sync_detail(standing),
                ),
                _transition_check(
                    _phase_completed(walking),
                    "transition/walking_completed_at_nominal_height",
                    (
                        f"target_height={NOMINAL_WALK_TARGET_HEIGHT:.6f} "
                        f"executed_steps={walking['executed_steps']} "
                        f"requested_steps={walking['steps']}"
                    ),
                ),
                _transition_check(
                    _phase_input_sync_pass(walking),
                    "transition/walking_input_synchronized",
                    _phase_input_sync_detail(walking),
                ),
                _transition_check(
                    _phase_completed(settling),
                    "transition/nominal_settle_completed",
                    (
                        f"target_height={NOMINAL_WALK_TARGET_HEIGHT:.6f} "
                        f"executed_steps={settling['executed_steps']} "
                        f"requested_steps={settling['steps']}"
                    ),
                ),
                _transition_check(
                    _phase_input_sync_pass(settling),
                    "transition/nominal_settle_input_synchronized",
                    _phase_input_sync_detail(settling),
                ),
                _transition_check(
                    _phase_completed(recovery_phase),
                    "transition/recovery_completed",
                    (
                        f"executed_steps={recovery_phase['executed_steps']} "
                        f"requested_steps={recovery_phase['steps']}"
                    ),
                ),
                _transition_check(
                    _phase_input_sync_pass(recovery_phase),
                    "transition/recovery_input_synchronized",
                    _phase_input_sync_detail(recovery_phase),
                ),
            ]
            scenario_pass = bool(
                all(check["level"] == "PASS" for check in transition_checks)
                and recovery_report["verdict"] == "PASS"
            )
            scenario = {
                "index": scenario_index,
                "command": command_name,
                "requested_target_height": float(target_height),
                "walking_target_height": float(NOMINAL_WALK_TARGET_HEIGHT),
                "reset": reset_snapshot,
                "standing": standing,
                "walking": walking,
                "nominal_settle": settling,
                "recovery_phase": recovery_phase,
                "transition_checks": transition_checks,
                "recovery": recovery_report,
                "verdict": "PASS" if scenario_pass else "FAIL",
            }
            scenarios.append(scenario)

            for phase_name, phase in (
                ("standing", standing),
                ("walking", walking),
                ("nominal_settle", settling),
                ("recovery", recovery_phase),
            ):
                terminal_snapshot = phase["terminal_snapshot"]
                if terminal_snapshot is not None:
                    terminal_events.append(
                        {
                            "scenario_index": scenario_index,
                            "command": command_name,
                            "requested_target_height": float(target_height),
                            "phase": phase_name,
                            **terminal_snapshot,
                        }
                    )

    passed_scenarios = sum(scenario["verdict"] == "PASS" for scenario in scenarios)
    return {
        "verdict": "PASS" if passed_scenarios == len(scenarios) else "FAIL",
        "nominal_walk_target_height": float(NOMINAL_WALK_TARGET_HEIGHT),
        "target_heights": [float(value) for value in target_heights],
        "nominal_settle_steps": int(nominal_settle_steps),
        "command_grid": [name for name, _command in COMMANDS],
        "warmup_steps": int(warmup_steps),
        "evaluation_steps": int(evaluation_steps),
        "thresholds": {
            "max_height_mae": HEIGHT_RECOVERY_MAX_HEIGHT_MAE,
            "min_double_support_fraction": HEIGHT_RECOVERY_MIN_DOUBLE_SUPPORT_FRACTION,
            "task_max_tilt_deg": float(max_tilt_deg),
            "termination_count": 0,
            "truncation_count": 0,
        },
        "scenario_count": len(scenarios),
        "passed_scenario_count": passed_scenarios,
        "failure_indices": [
            int(scenario["index"]) for scenario in scenarios if scenario["verdict"] != "PASS"
        ],
        "terminal_events": terminal_events,
        "scenarios": scenarios,
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
    post_walk_target_heights: tuple[float, ...] = DEFAULT_POST_WALK_TARGET_HEIGHTS,
    height_recovery_nominal_settle_steps: int = (DEFAULT_HEIGHT_RECOVERY_NOMINAL_SETTLE_STEPS),
    height_recovery_warmup_steps: int = DEFAULT_HEIGHT_RECOVERY_WARMUP_STEPS,
    height_recovery_evaluation_steps: int = DEFAULT_HEIGHT_RECOVERY_EVALUATION_STEPS,
) -> dict[str, Any]:
    if int(repeats) <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if int(active_steps) <= 0 or int(stop_steps) <= 0:
        raise ValueError(
            "active_steps and stop_steps must be positive, "
            f"got active_steps={active_steps} stop_steps={stop_steps}"
        )
    target_heights = tuple(float(value) for value in post_walk_target_heights)
    if not target_heights or not all(np.isfinite(value) for value in target_heights):
        raise ValueError(
            "post_walk_target_heights must contain at least one finite value, "
            f"got {post_walk_target_heights}"
        )
    if int(height_recovery_warmup_steps) < 0:
        raise ValueError("height_recovery_warmup_steps must be non-negative")
    if int(height_recovery_nominal_settle_steps) <= 0:
        raise ValueError("height_recovery_nominal_settle_steps must be positive")
    if int(height_recovery_evaluation_steps) <= 0:
        raise ValueError("height_recovery_evaluation_steps must be positive")
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
    zero_command = np.zeros(3, dtype=np.float32)
    try:
        for index in range(int(repeats)):
            reset_snapshot = _reset_episode(session)
            command_name, command = COMMANDS[index % len(COMMANDS)]
            standing = _run_phase(
                session,
                stop_steps,
                command=zero_command,
                target_height=NOMINAL_WALK_TARGET_HEIGHT,
            )
            if standing["done_count"] > 0:
                walking = _skipped_phase(steps=active_steps, reason="standing_done")
                recovered = _skipped_phase(steps=stop_steps, reason="standing_done")
            else:
                walking = _run_phase(
                    session,
                    active_steps,
                    command=command,
                    target_height=NOMINAL_WALK_TARGET_HEIGHT,
                )
                if walking["done_count"] > 0:
                    recovered = _skipped_phase(steps=stop_steps, reason="walking_done")
                else:
                    recovered = _run_phase(
                        session,
                        stop_steps,
                        command=zero_command,
                        target_height=NOMINAL_WALK_TARGET_HEIGHT,
                    )

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
        height_recovery = _run_height_recovery_grid(
            session,
            target_heights=target_heights,
            active_steps=int(active_steps),
            standing_steps=int(stop_steps),
            nominal_settle_steps=int(height_recovery_nominal_settle_steps),
            warmup_steps=int(height_recovery_warmup_steps),
            evaluation_steps=int(height_recovery_evaluation_steps),
            max_tilt_deg=float(cfg.reward.max_tilt_deg),
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
        "input_sync_pass": all(_phase_input_sync_pass(phase) for phase in all_phases),
        "task_min_base_height": float(reward_cfg.min_base_height),
        "task_max_tilt_deg": float(reward_cfg.max_tilt_deg),
    }
    summary["nominal_transition_gate_pass"] = bool(
        summary["total_done_count"] == 0
        and summary["min_base_height"] > summary["task_min_base_height"]
        and summary["max_tilt_deg"] < summary["task_max_tilt_deg"]
        and summary["stop_speed_decay_pass"]
        and summary["input_sync_pass"]
        and summary["completed_phase_count"] == expected_phase_count
        and summary["nonzero_action_phases"] == expected_phase_count
    )
    summary["height_recovery_gate_pass"] = bool(height_recovery["verdict"] == "PASS")
    summary["height_recovery_scenario_count"] = int(height_recovery["scenario_count"])
    summary["height_recovery_passed_scenario_count"] = int(height_recovery["passed_scenario_count"])
    summary["gate_pass"] = bool(
        summary["nominal_transition_gate_pass"] and summary["height_recovery_gate_pass"]
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
        "height_recovery": height_recovery,
        "failure_indices": [
            int(episode["index"])
            for episode in episodes
            if not episode["stop_speed_le_active"]
            or not all(
                _controlled_phase_pass(phase)
                for phase in (episode["standing"], episode["walking"], episode["stop"])
            )
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
        "input_sync_failure_indices": [
            int(episode["index"])
            for episode in episodes
            if not all(
                _controlled_phase_pass(phase)
                for phase in (episode["standing"], episode["walking"], episode["stop"])
            )
            and not any(
                phase["done_count"] > 0
                for phase in (episode["standing"], episode["walking"], episode["stop"])
            )
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True, type=Path)
    parser.add_argument("--task", default="g1_walk_height/mujoco")
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--active-steps", type=int, default=20)
    parser.add_argument("--stop-steps", type=int, default=20)
    parser.add_argument(
        "--post-walk-target-heights",
        type=float,
        nargs="+",
        default=list(DEFAULT_POST_WALK_TARGET_HEIGHTS),
    )
    parser.add_argument(
        "--height-recovery-nominal-settle-steps",
        type=int,
        default=DEFAULT_HEIGHT_RECOVERY_NOMINAL_SETTLE_STEPS,
    )
    parser.add_argument(
        "--height-recovery-warmup-steps",
        type=int,
        default=DEFAULT_HEIGHT_RECOVERY_WARMUP_STEPS,
    )
    parser.add_argument(
        "--height-recovery-evaluation-steps",
        type=int,
        default=DEFAULT_HEIGHT_RECOVERY_EVALUATION_STEPS,
    )
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
        post_walk_target_heights=tuple(args.post_walk_target_heights),
        height_recovery_nominal_settle_steps=args.height_recovery_nominal_settle_steps,
        height_recovery_warmup_steps=args.height_recovery_warmup_steps,
        height_recovery_evaluation_steps=args.height_recovery_evaluation_steps,
    )
    print("UniLab G1 distill student transition live sentinel")
    print(
        {key: report[key] for key in ("student_checkpoint", "resolved_checkpoint", "task", "seed")}
    )
    print({"summary": report["summary"]})
    print({"command_summary": report["command_summary"]})
    print({"terminal_events": report["terminal_events"]})
    height_recovery = report["height_recovery"]
    print(
        {
            "height_recovery_summary": {
                key: height_recovery[key]
                for key in (
                    "verdict",
                    "nominal_walk_target_height",
                    "target_heights",
                    "command_grid",
                    "nominal_settle_steps",
                    "warmup_steps",
                    "evaluation_steps",
                    "thresholds",
                    "scenario_count",
                    "passed_scenario_count",
                    "failure_indices",
                )
            }
        }
    )
    for scenario in height_recovery["scenarios"]:
        recovery = scenario["recovery"]
        print(
            {
                "height_recovery_scenario": {
                    "index": scenario["index"],
                    "command": scenario["command"],
                    "requested_target_height": scenario["requested_target_height"],
                    "walking_target_height": scenario["walking_target_height"],
                    "nominal_settle_steps": scenario["nominal_settle"]["steps"],
                    "verdict": scenario["verdict"],
                    "metrics": recovery["metrics"],
                    "failed_checks": [
                        check
                        for check in (*scenario["transition_checks"], *recovery["checks"])
                        if check["level"] == "FAIL"
                    ],
                }
            }
        )
    print(
        {
            "failure_indices": report["failure_indices"],
            "termination_indices": report["termination_indices"],
            "truncation_indices": report["truncation_indices"],
            "stop_decay_failure_indices": report["stop_decay_failure_indices"],
            "input_sync_failure_indices": report["input_sync_failure_indices"],
            "height_recovery_failure_indices": height_recovery["failure_indices"],
        }
    )
    return 0 if report["summary"]["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
