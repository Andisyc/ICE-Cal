"""Same-condition zero-shot versus adapted slope evaluation."""

from __future__ import annotations

import json
import random
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.adaptation_checkpoint import (
    FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
    assert_fada_adaptation_source_checkpoint,
    load_fada_deployable_policy_checkpoint,
)
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.algos.torch.distill.fada.slope_metrics import (
    FADASlopeTrajectory,
    compare_slope_summaries,
    summarize_slope_trajectory,
)
from unilab.algos.torch.distill.fada.target_collector import (
    FADASlopeEpisodePolicy,
)
from unilab.algos.torch.distill.fada.target_data import FADA_TARGET_ARTIFACT_SCHEMA_VERSION
from unilab.algos.torch.distill.fada.target_domain import (
    assert_nominal_slope_environment,
    resolve_fada_target_domain,
)
from unilab.algos.torch.distill.fada.target_rollout import (
    apply_external_command,
    rollout_done_flags,
    rollout_terminal_reasons,
    scheduled_target_command,
    target_tracking_camera_kwargs,
)
from unilab.algos.torch.distill.workflow import file_sha256
from unilab.base.backend.mujoco.playback import render_mujoco_states_video
from unilab.envs.common.rotation import np_yaw_from_quat
from unilab.training import BackendAdapter, create_env, ensure_registries, get_hydra_runtime_choice

ROOT_DIR = Path(__file__).resolve().parents[6]


def _path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _command(cfg: DictConfig) -> np.ndarray:
    command = np.asarray(
        OmegaConf.to_container(cfg.evaluation.command, resolve=True), dtype=np.float32
    )
    if command.shape != (3,) or not bool(np.all(np.isfinite(command))):
        raise ValueError("FADA slope evaluation command must be finite and 3-D")
    if command[1] != 0.0 or command[2] != 0.0:
        raise ValueError("FADA slope evaluation requires zero lateral and yaw command")
    return command


def _rollout(
    env: Any,
    policy: Any,
    *,
    snapshot: Any,
    command: np.ndarray,
    control_steps: int,
    ramp_steps: int,
    episode_policy: FADASlopeEpisodePolicy | None,
) -> FADASlopeTrajectory:
    env.restore_rollout_snapshot(snapshot)
    controller = FADAPlaybackController(policy, device=next(policy.parameters()).device)
    base_positions: list[np.ndarray] = []
    yaws: list[float] = []
    feet_positions: list[np.ndarray] = []
    velocities: list[float] = []
    commands: list[float] = []
    states: list[np.ndarray] = []
    terminal_reason = "horizon"
    angle = np.deg2rad(episode_policy.geometry.angle_deg) if episode_policy else 0.0
    forward_axis = np.asarray([np.cos(angle), 0.0, np.sin(angle)])
    for step in range(control_steps):
        current_command = scheduled_target_command(
            (0.0, 0.0, 0.0), command, ramp_steps=ramp_steps, step=step
        )
        apply_external_command(env, current_command)
        action = controller.act(env.state.obs, current_command[None, :])
        state = env.step(action.detach().cpu().numpy().astype(np.float32))
        base = np.asarray(env.get_base_pos(), dtype=np.float64)
        quat = np.asarray(env.get_base_quat(), dtype=np.float64)
        feet = np.asarray(env.get_foot_pos(), dtype=np.float64)
        velocity = np.asarray(env.get_base_lin_vel(), dtype=np.float64)
        if (
            base.shape != (1, 3)
            or quat.shape != (1, 4)
            or feet.shape != (1, 2, 3)
            or velocity.shape != (1, 3)
        ):
            raise ValueError("FADA slope evaluation task-state shape mismatch")
        base_positions.append(base[0].copy())
        yaws.append(float(np_yaw_from_quat(quat)[0]))
        feet_positions.append(feet[0].copy())
        velocities.append(float(np.dot(velocity[0], forward_axis)))
        commands.append(float(current_command[0]))
        states.append(np.asarray(env.get_physics_state_snapshot()).copy())
        terminated, truncated = rollout_done_flags(state, num_envs=1)
        if episode_policy is not None:
            decision = episode_policy.classify(base_pos_w=base[0], feet_pos_w=feet[0], done=False)
            lifecycle_reason = rollout_terminal_reasons(state, num_envs=1)[0]
            if lifecycle_reason is not None:
                terminal_reason = lifecycle_reason
                break
            if decision.terminal_reason is not None:
                terminal_reason = decision.terminal_reason
                break
        elif bool(terminated[0]) or bool(truncated[0]):
            terminal_reason = rollout_terminal_reasons(state, num_envs=1)[0] or "horizon"
            break
    return FADASlopeTrajectory(
        base_pos_w=np.asarray(base_positions),
        base_yaw_rad=np.asarray(yaws),
        feet_pos_w=np.asarray(feet_positions),
        forward_velocity_mps=np.asarray(velocities),
        command_forward_mps=np.asarray(commands),
        physics_states=tuple(states),
        terminal_reason=terminal_reason,
        control_dt_s=float(env.cfg.ctrl_dt),
    ).validate()


def _flat_summary(trajectory: FADASlopeTrajectory) -> dict[str, Any]:
    positions = trajectory.base_pos_w
    speed_error = trajectory.forward_velocity_mps - trajectory.command_forward_mps
    return {
        "survived_horizon": trajectory.terminal_reason == "horizon",
        "terminal_reason": trajectory.terminal_reason,
        "forward_progress_m": float(positions[-1, 0] - positions[0, 0]),
        "mean_abs_lateral_m": float(np.mean(np.abs(positions[:, 1] - positions[0, 1]))),
        "forward_velocity_mae_mps": float(np.mean(np.abs(speed_error))),
        "steps": int(positions.shape[0]),
    }


def _run_pair(
    env: Any,
    source_policy: Any,
    adapted_policy: Any,
    *,
    command: np.ndarray,
    control_steps: int,
    ramp_steps: int,
    episode_policy: FADASlopeEpisodePolicy | None,
) -> tuple[FADASlopeTrajectory, FADASlopeTrajectory]:
    env.set_autoreset(False)
    env.reset_all()
    snapshot = env.capture_rollout_snapshot()
    zero = _rollout(
        env,
        source_policy,
        snapshot=snapshot,
        command=command,
        control_steps=control_steps,
        ramp_steps=ramp_steps,
        episode_policy=episode_policy,
    )
    adapted = _rollout(
        env,
        adapted_policy,
        snapshot=snapshot,
        command=command,
        control_steps=control_steps,
        ramp_steps=ramp_steps,
        episode_policy=episode_policy,
    )
    return zero, adapted


def run_fada_target_evaluation(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
    load_policy_fn: Callable[..., Any] = load_fada_deployable_policy_checkpoint,
    ensure_registries_fn: Callable[[], None] = ensure_registries,
    create_env_fn: Callable[..., Any] = create_env,
    render_fn: Callable[..., str] = render_mujoco_states_video,
) -> dict[str, Any]:
    root = Path(root_dir).resolve()
    domain = resolve_fada_target_domain(cfg)
    assert_nominal_slope_environment(cfg, domain, task_choice=get_hydra_runtime_choice(cfg, "task"))
    geometry = domain.slope
    if geometry is None:
        raise ValueError("FADA slope evaluation requires slope geometry")
    source_path = _path(cfg.evaluation.source_checkpoint_path, root)
    adapted_path = _path(cfg.evaluation.adapted_checkpoint_path, root)
    output_dir = _path(cfg.evaluation.output_dir, root)
    if output_dir.exists():
        raise FileExistsError(f"FADA evaluation output already exists: {output_dir}")
    for label, path in (("source", source_path), ("adapted", adapted_path)):
        if not path.is_file():
            raise FileNotFoundError(f"FADA evaluation {label} checkpoint not found: {path}")
        expected = OmegaConf.select(cfg, f"evaluation.expected_{label}_checkpoint_sha256")
        observed = file_sha256(path)
        if expected is not None and str(expected) != observed:
            raise ValueError(f"FADA evaluation {label} checkpoint SHA-256 mismatch")
    source = assert_fada_adaptation_source_checkpoint(
        load_policy_fn(source_path, device=str(cfg.evaluation.device))
    )
    adapted = load_policy_fn(adapted_path, device=str(cfg.evaluation.device))
    if adapted.checkpoint.get("schema_version") != FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("FADA evaluation adapted checkpoint must use fada-adapted/v3")
    if adapted.checkpoint.get("target_domain_id") != domain.target_domain_id:
        raise ValueError("FADA adapted checkpoint target-domain identity does not match evaluation")
    if (
        adapted.checkpoint.get("target_artifact_schema_version")
        != FADA_TARGET_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("FADA slope evaluation requires a schema-v3 slope target artifact")
    if source.policy.config != adapted.policy.config:
        raise ValueError("FADA evaluation checkpoint architectures do not match")
    adapted_source = adapted.checkpoint.get("source_checkpoint_sha256")
    if adapted_source is not None and adapted_source != file_sha256(source_path):
        raise ValueError("FADA adapted checkpoint source lineage does not match zero-shot source")
    command = _command(cfg)
    control_steps = int(cfg.evaluation.control_steps)
    ramp_steps = int(cfg.evaluation.ramp_steps)
    if control_steps <= 0 or ramp_steps < 0:
        raise ValueError("FADA evaluation control_steps/ramp_steps are invalid")
    if not bool(cfg.evaluation.record_video):
        raise ValueError("FADA slope evaluation requires evaluation.record_video=true")
    seed = int(cfg.evaluation.seed)
    ensure_registries_fn()
    _seed_all(seed)
    env = create_env_fn(
        cfg,
        num_envs=1,
        env_cfg_override=BackendAdapter(
            cfg, root_dir=root, algo_name="sac"
        ).build_task_env_cfg_override(),
        sim_backend=domain.backend,
    )
    flat_env = None
    try:
        command_tuple = tuple(float(value) for value in command)
        episode_policy = FADASlopeEpisodePolicy(geometry, (command_tuple,))
        zero, adapted_rollout = _run_pair(
            env,
            source.policy,
            adapted.policy,
            command=command,
            control_steps=control_steps,
            ramp_steps=ramp_steps,
            episode_policy=episode_policy,
        )
        zero_summary = summarize_slope_trajectory(zero, geometry)
        adapted_summary = summarize_slope_trajectory(adapted_rollout, geometry)
        metrics: dict[str, Any] = {
            "zero_shot": zero_summary,
            "adapted": adapted_summary,
            "improvement": compare_slope_summaries(zero_summary, adapted_summary),
        }
        flat_pair = None
        if bool(cfg.evaluation.run_flat_regression):
            flat_override = BackendAdapter(
                cfg, root_dir=root, algo_name="sac"
            ).build_task_env_cfg_override()
            flat_override.setdefault("scene", {})["model_file"] = str(
                root / "src/unilab/assets/robots/g1/scene_flat.xml"
            )
            _seed_all(seed)
            flat_env = create_env_fn(
                cfg,
                num_envs=1,
                env_cfg_override=flat_override,
                sim_backend=domain.backend,
            )
            flat_pair = _run_pair(
                flat_env,
                source.policy,
                adapted.policy,
                command=command,
                control_steps=control_steps,
                ramp_steps=ramp_steps,
                episode_policy=None,
            )
            metrics["flat_regression"] = {
                "zero_shot": _flat_summary(flat_pair[0]),
                "adapted": _flat_summary(flat_pair[1]),
            }
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}-", dir=output_dir.parent
        ) as tmp:
            stage = Path(tmp) / output_dir.name
            stage.mkdir()
            render_fn(
                env=env,
                state_list=zero.physics_states,
                output_video=stage / "zero_shot.mp4",
                camera_kwargs=target_tracking_camera_kwargs(),
            )
            render_fn(
                env=env,
                state_list=adapted_rollout.physics_states,
                output_video=stage / "adapted.mp4",
                camera_kwargs=target_tracking_camera_kwargs(),
            )
            files = ["zero_shot.mp4", "adapted.mp4", "metrics.json"]
            if flat_pair is not None:
                render_fn(
                    env=flat_env,
                    state_list=flat_pair[0].physics_states,
                    output_video=stage / "zero_shot_flat.mp4",
                    camera_kwargs=target_tracking_camera_kwargs(),
                )
                render_fn(
                    env=flat_env,
                    state_list=flat_pair[1].physics_states,
                    output_video=stage / "adapted_flat.mp4",
                    camera_kwargs=target_tracking_camera_kwargs(),
                )
                files.extend(("zero_shot_flat.mp4", "adapted_flat.mp4"))
            (stage / "metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
            (stage / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "fada-slope-evaluation/v1",
                        "target_domain_id": domain.target_domain_id,
                        "source_checkpoint_sha256": file_sha256(source_path),
                        "adapted_checkpoint_sha256": file_sha256(adapted_path),
                        "files": files,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            stage.replace(output_dir)
    finally:
        env.close()
        if flat_env is not None:
            flat_env.close()
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "metrics_path": str(output_dir / "metrics.json"),
        "metrics": metrics,
    }
