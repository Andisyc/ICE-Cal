"""Compare walking and standing SAC teacher authority after a walk-to-stop switch.

This is a diagnostic-only live probe. It reuses the distillation playback
session and teacher loader, but it does not update data, student weights, or
training contracts.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from unilab.algos.torch.distill import DistillationTeacherSpec, load_sac_teacher_policy
from unilab.visualization.interactive_playback import (
    RslRlPlaybackConfig,
    create_distill_playback_session,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
OBS_DIM = 98
ACTION_DIM = 29
COMMAND_START = 93
COMMAND_DIM = 3
COHORTS = ("WT", "WS", "SS")


def _compose_cfg(task: str = "g1_walk_flat/mujoco") -> Any:
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "distill"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"task={task}"])


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _actor_obs(session: Any) -> np.ndarray:
    observations = session.obs
    if hasattr(observations, "get"):
        actor = observations.get("actor")
        if actor is None:
            actor = observations.get("obs")
    else:
        actor = observations
    if actor is None:
        raise RuntimeError("playback session did not expose actor observations")
    result = _as_array(actor)
    if result.ndim != 2 or result.shape[1] != OBS_DIM:
        raise ValueError(f"expected actor obs shape (N,{OBS_DIM}), got {result.shape}")
    return result


def _kinematic_state(env: Any) -> np.ndarray:
    backend = env._backend
    return np.concatenate(
        [
            np.asarray(backend.get_base_pos(), dtype=np.float32),
            np.asarray(backend.get_base_quat(), dtype=np.float32),
            np.asarray(backend.get_dof_pos(), dtype=np.float32),
            np.asarray(backend.get_base_lin_vel(), dtype=np.float32),
            np.asarray(backend.get_base_ang_vel(), dtype=np.float32),
            np.asarray(backend.get_dof_vel(), dtype=np.float32),
        ],
        axis=1,
    )


def _qpos_qvel(env: Any) -> tuple[np.ndarray, np.ndarray]:
    backend = env._backend
    qpos = np.concatenate(
        [
            np.asarray(backend.get_base_pos(), dtype=np.float32),
            np.asarray(backend.get_base_quat(), dtype=np.float32),
            np.asarray(backend.get_dof_pos(), dtype=np.float32),
        ],
        axis=1,
    )
    qvel = np.concatenate(
        [
            np.asarray(backend.get_base_lin_vel(), dtype=np.float32),
            np.asarray(backend.get_base_ang_vel(), dtype=np.float32),
            np.asarray(backend.get_dof_vel(), dtype=np.float32),
        ],
        axis=1,
    )
    return qpos, qvel


def _capture_snapshot(env: Any) -> dict[str, Any]:
    qpos, qvel = _qpos_qvel(env)
    return {
        "qpos": qpos.copy(),
        "qvel": qvel.copy(),
        "kinematic": _kinematic_state(env).copy(),
        "info": copy.deepcopy(env.state.info),
        "terminated": np.asarray(env.state.terminated, dtype=bool).copy(),
        "truncated": np.asarray(env.state.truncated, dtype=bool).copy(),
    }


def _restore_snapshot(session: Any, snapshot: dict[str, Any]) -> float:
    env = session.env
    env._backend.set_state(
        np.asarray([0], dtype=np.int64),
        snapshot["qpos"],
        snapshot["qvel"],
    )
    env.state.info.clear()
    env.state.info.update(copy.deepcopy(snapshot["info"]))
    env.state.terminated[...] = snapshot["terminated"]
    env.state.truncated[...] = snapshot["truncated"]
    env.refresh_state()
    session.refresh_observation()
    return float(np.max(np.abs(_kinematic_state(env) - snapshot["kinematic"])))


def _set_command(session: Any, command: np.ndarray) -> tuple[bool, bool]:
    session.set_external_command(command)
    actor_obs = _actor_obs(session)
    observed_command = actor_obs[:, COMMAND_START : COMMAND_START + COMMAND_DIM]
    command_sync = bool(np.max(np.abs(observed_command - command[None, :])) <= 1.0e-5)
    gait_enabled = _as_array(session.env.state.info["gait_enabled"])
    gait_active = bool(gait_enabled[0] > 0.5)
    return command_sync, gait_active


def _disable_probe_observation_noise(env: Any) -> float | None:
    noise_cfg = getattr(env.cfg, "noise_config", None)
    if noise_cfg is None or not hasattr(noise_cfg, "level"):
        return None
    previous = float(noise_cfg.level)
    noise_cfg.level = 0.0
    return previous


def _load_teacher(checkpoint: Path, device: str) -> torch.nn.Module:
    spec = DistillationTeacherSpec(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        actor_hidden_dim=512,
        use_layer_norm=True,
        obs_normalization=True,
    )
    teacher = load_sac_teacher_policy(checkpoint, spec, device=device)
    teacher.eval()
    return teacher


def _teacher_action(teacher: torch.nn.Module, observations: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        action = teacher(torch.as_tensor(observations, dtype=torch.float32, device=device))
    result = _as_array(action)
    if result.shape != (observations.shape[0], ACTION_DIM):
        raise ValueError(f"expected teacher action shape (N,{ACTION_DIM}), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("teacher action contains non-finite values")
    return result


def _state_metrics(env: Any) -> dict[str, float | bool]:
    backend = env._backend
    base_pos = np.asarray(backend.get_base_pos(), dtype=np.float32)
    base_lin_vel = np.asarray(backend.get_base_lin_vel(), dtype=np.float32)
    base_ang_vel = np.asarray(backend.get_base_ang_vel(), dtype=np.float32)
    upvector = np.asarray(backend.get_sensor_data(env.cfg.sensor.upvector), dtype=np.float32)
    tilt_deg = float(np.rad2deg(np.arccos(np.clip(upvector[0, 2], -1.0, 1.0))))
    return {
        "base_height": float(base_pos[0, 2]),
        "tilt_deg": tilt_deg,
        "lin_vel_norm": float(np.linalg.norm(base_lin_vel[0])),
        "ang_vel_norm": float(np.linalg.norm(base_ang_vel[0])),
        "terminated": bool(np.asarray(env.state.terminated, dtype=bool)[0]),
    }


def _summary(records: list[dict[str, float | bool]]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "min_base_height": None,
            "max_tilt_deg": None,
            "max_lin_vel_norm": None,
            "max_ang_vel_norm": None,
            "first_terminated_step": None,
        }
    heights = np.asarray([float(row["base_height"]) for row in records], dtype=np.float32)
    tilts = np.asarray([float(row["tilt_deg"]) for row in records], dtype=np.float32)
    lin_vels = np.asarray([float(row["lin_vel_norm"]) for row in records], dtype=np.float32)
    ang_vels = np.asarray([float(row["ang_vel_norm"]) for row in records], dtype=np.float32)
    terminated_steps = [int(row["step"]) for row in records if bool(row["terminated"])]
    return {
        "count": len(records),
        "min_base_height": float(np.min(heights)),
        "max_tilt_deg": float(np.max(tilts)),
        "max_lin_vel_norm": float(np.max(lin_vels)),
        "max_ang_vel_norm": float(np.max(ang_vels)),
        "first_terminated_step": min(terminated_steps) if terminated_steps else None,
    }


def _run_branch(
    session: Any,
    selected_teacher: torch.nn.Module,
    comparison_teacher: torch.nn.Module,
    steps: int,
    device: str,
) -> tuple[list[dict[str, float | bool]], float, bool]:
    actor_obs = _actor_obs(session)
    selected_action = _teacher_action(selected_teacher, actor_obs, device)
    comparison_action = _teacher_action(comparison_teacher, actor_obs, device)
    switch_action_mse = float(np.mean(np.square(selected_action - comparison_action)))
    command_obs = actor_obs[:, COMMAND_START : COMMAND_START + COMMAND_DIM]
    command_sync = bool(np.max(np.abs(command_obs)) <= 1.0e-5)
    records: list[dict[str, float | bool]] = []
    for step in range(steps):
        actor_obs = _actor_obs(session)
        action = _teacher_action(selected_teacher, actor_obs, device)
        session.env.step(action)
        session.refresh_observation()
        metrics = _state_metrics(session.env)
        metrics["step"] = step
        records.append(metrics)
    return records, switch_action_mse, command_sync


def run_probe(
    *,
    walking_checkpoint: Path,
    standing_checkpoint: Path,
    walk_vx: float,
    pre_switch_steps: int,
    post_switch_steps: int,
    device: str,
) -> tuple[dict[str, bool], dict[str, Any]]:
    if pre_switch_steps < 1 or post_switch_steps < 1:
        raise ValueError("pre_switch_steps and post_switch_steps must be positive")
    cfg = _compose_cfg()
    playback_cfg = RslRlPlaybackConfig(
        task="g1_walk_flat",
        load_run="",
        checkpoint=None,
        action_mode="zero",
        policy_obs_mode="actor",
        algo_log_name="distill",
        log_root=None,
        num_envs=1,
    )
    session, _, _ = create_distill_playback_session(
        playback_cfg=playback_cfg,
        cfg=cfg,
        root_dir=ROOT_DIR,
        device=device,
    )
    walking_teacher = _load_teacher(walking_checkpoint, device)
    standing_teacher = _load_teacher(standing_checkpoint, device)
    zero_command = np.zeros((COMMAND_DIM,), dtype=np.float32)
    active_command = np.asarray([walk_vx, 0.0, 0.0], dtype=np.float32)
    try:
        previous_noise_level = _disable_probe_observation_noise(session.env)
        session.reset()
        initial_snapshot = _capture_snapshot(session.env)
        initial_obs = _actor_obs(session)
        initial_command_sync = bool(
            np.max(np.abs(initial_obs[:, COMMAND_START : COMMAND_START + COMMAND_DIM])) <= 1.0e-5
        )
        initial_gait_disabled = bool(_as_array(session.env.state.info["gait_enabled"])[0] <= 0.5)

        _restore_snapshot(session, initial_snapshot)
        static_command_sync, static_gait_active = _set_command(session, zero_command)
        static_records, static_mse, _ = _run_branch(
            session, standing_teacher, walking_teacher, post_switch_steps, device
        )

        _restore_snapshot(session, initial_snapshot)
        active_command_sync, active_gait_active = _set_command(session, active_command)
        pre_records: list[dict[str, float | bool]] = []
        for step in range(pre_switch_steps):
            actor_obs = _actor_obs(session)
            action = _teacher_action(walking_teacher, actor_obs, device)
            session.env.step(action)
            session.refresh_observation()
            metrics = _state_metrics(session.env)
            metrics["step"] = step
            pre_records.append(metrics)
        walk_snapshot = _capture_snapshot(session.env)
        pre_walk_terminated = any(bool(row["terminated"]) for row in pre_records)

        walking_restore_diff = _restore_snapshot(session, walk_snapshot)
        wt_command_sync, wt_gait_active = _set_command(session, zero_command)
        wt_records, wt_mse, _ = _run_branch(
            session, walking_teacher, standing_teacher, post_switch_steps, device
        )

        standing_restore_diff = _restore_snapshot(session, walk_snapshot)
        ws_command_sync, ws_gait_active = _set_command(session, zero_command)
        ws_records, ws_mse, _ = _run_branch(
            session, standing_teacher, walking_teacher, post_switch_steps, device
        )

        env = session.env
        min_height = float(env.cfg.reward_config.min_base_height)
        max_tilt = float(env.cfg.reward_config.max_tilt_deg)
        summaries = {
            "WT": _summary(wt_records),
            "WS": _summary(ws_records),
            "SS": _summary(static_records),
        }
        ws = summaries["WS"]
        checks = {
            "checkpoint_paths_exist": walking_checkpoint.is_file()
            and standing_checkpoint.is_file(),
            "initial_command_sync": initial_command_sync,
            "initial_gait_disabled": initial_gait_disabled,
            "static_command_sync": static_command_sync,
            "static_gait_disabled": not static_gait_active,
            "static_standing_survived": summaries["SS"]["first_terminated_step"] is None,
            "active_command_sync": active_command_sync,
            "active_walk_gait_enabled": active_gait_active,
            "pre_switch_walk_survived": not pre_walk_terminated,
            "wt_restore_exact": walking_restore_diff <= 1.0e-5,
            "ws_restore_exact": standing_restore_diff <= 1.0e-5,
            "wt_zero_command_sync": wt_command_sync,
            "ws_zero_command_sync": ws_command_sync,
            "standing_teacher_recovery": ws["first_terminated_step"] is None
            and float(ws["min_base_height"]) >= min_height
            and float(ws["max_tilt_deg"]) <= max_tilt,
        }
        details = {
            "walking_checkpoint": str(walking_checkpoint),
            "standing_checkpoint": str(standing_checkpoint),
            "teacher_obs_normalizer": {
                "walking": walking_teacher.obs_normalizer is not None,
                "standing": standing_teacher.obs_normalizer is not None,
            },
            "probe_observation_noise_level_before_disable": previous_noise_level,
            "walk_vx": float(walk_vx),
            "pre_switch_steps": int(pre_switch_steps),
            "post_switch_steps": int(post_switch_steps),
            "cohorts": list(COHORTS),
            "pre_walk_summary": _summary(pre_records),
            "branch_restore_kinematic_max_abs_diff": {
                "WT": walking_restore_diff,
                "WS": standing_restore_diff,
            },
            "switch_teacher_action_mse": {
                "WT": wt_mse,
                "WS": ws_mse,
                "SS": static_mse,
            },
            "branch_gait_active_after_zero_command": {
                "WT": wt_gait_active,
                "WS": ws_gait_active,
            },
            "min_base_height_limit": min_height,
            "max_tilt_deg_limit": max_tilt,
            "summaries": summaries,
        }
        return checks, details
    finally:
        close = getattr(session.env, "close", None)
        if callable(close):
            close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--walking-checkpoint",
        type=Path,
        default=ROOT_DIR / "logs/fast_sac/G1WalkFlat/2026-07-09_02-48-58_mujoco/model_5000.pt",
    )
    parser.add_argument(
        "--standing-checkpoint",
        type=Path,
        default=ROOT_DIR / "logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt",
    )
    parser.add_argument("--walk-vx", type=float, default=0.4)
    parser.add_argument("--pre-switch-steps", type=int, default=80)
    parser.add_argument("--post-switch-steps", type=int, default=80)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checks, details = run_probe(
        walking_checkpoint=args.walking_checkpoint,
        standing_checkpoint=args.standing_checkpoint,
        walk_vx=args.walk_vx,
        pre_switch_steps=args.pre_switch_steps,
        post_switch_steps=args.post_switch_steps,
        device=args.device,
    )
    print("UniLab G1 distill teacher recovery differential")
    for key, value in details.items():
        print(f"differential/{key}: {value}")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] differential/{name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
