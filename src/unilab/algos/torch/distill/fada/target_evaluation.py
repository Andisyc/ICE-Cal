"""Same-condition zero-shot versus adapted slope evaluation."""

from __future__ import annotations

import json
import random
import tempfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
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
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.observation import project_fada_observation_tensor
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.algos.torch.distill.fada.slope_metrics import (
    FADASlopeTrajectory,
    aggregate_improvement_summaries,
    aggregate_numeric_summaries,
    aggregate_policy_summaries,
    compact_slope_summary,
    compare_slope_summaries,
    summarize_slope_trajectory,
)
from unilab.algos.torch.distill.fada.target_collector import (
    FADASlopeEpisodePolicy,
    concat_fada_target_batches,
    fada_target_batch_from_window,
)
from unilab.algos.torch.distill.fada.target_data import (
    FADA_TARGET_ARTIFACT_SCHEMA_VERSION,
    FADATargetBatch,
)
from unilab.algos.torch.distill.fada.target_domain import (
    FADASlopeGeometry,
    FADATargetDomainSpec,
    assert_nominal_slope_environment,
    resolve_fada_target_domain,
)
from unilab.algos.torch.distill.fada.target_evaluation_diagnostics import (
    compare_fada_rollout_diagnostics,
    summarize_fada_own_rollout,
)
from unilab.algos.torch.distill.fada.target_rollout import (
    apply_external_command,
    rollout_done_flags,
    rollout_terminal_reasons,
    scheduled_target_command,
    target_tracking_camera_kwargs,
)
from unilab.algos.torch.distill.fada.windows import FADACausalTransition
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


@dataclass(frozen=True)
class FADAEvaluationRollout:
    trajectory: FADASlopeTrajectory
    target_batch: FADATargetBatch


def _evaluation_commands(
    cfg: DictConfig,
    domain: FADATargetDomainSpec,
) -> tuple[tuple[tuple[float, ...], ...], int]:
    num_trials = int(cfg.evaluation.num_trials)
    available = domain.command_sequence
    if not 2 <= num_trials <= len(available):
        raise ValueError(
            "FADA slope evaluation num_trials must be between 2 and the target-domain "
            f"command count ({len(available)}), got {num_trials}"
        )
    commands = tuple(tuple(float(value) for value in command) for command in available[:num_trials])
    if len(set(commands)) != len(commands):
        raise ValueError("FADA slope evaluation commands must be unique")
    representative_speed = float(cfg.evaluation.representative_forward_speed_mps)
    if not np.isfinite(representative_speed):
        raise ValueError("FADA representative forward speed must be finite")
    representative = min(
        range(len(commands)),
        key=lambda index: abs(commands[index][0] - representative_speed),
    )
    return commands, representative


def _raw_actor_observation(state: Any) -> Any:
    observations = state.obs
    for key in ("actor", "obs", "policy"):
        if key in observations:
            return observations[key]
    raise KeyError(f"FADA evaluation actor observation not found; available={sorted(observations)}")


def _project_observation(state: Any, policy: FADAPlannerIDMPolicy) -> np.ndarray:
    raw = torch.as_tensor(_raw_actor_observation(state), dtype=torch.float32)
    projected = project_fada_observation_tensor(
        raw,
        observation_contract=policy.config.observation_contract,
    )
    if tuple(projected.shape) != (1, policy.config.obs_dim):
        raise ValueError(
            "FADA evaluation projected observation shape mismatch: "
            f"expected={(1, policy.config.obs_dim)} observed={tuple(projected.shape)}"
        )
    return projected[0].cpu().numpy().copy()


def _rollout(
    env: Any,
    policy: FADAPlannerIDMPolicy,
    *,
    snapshot: Any,
    command: np.ndarray,
    control_steps: int,
    ramp_steps: int,
    episode_policy: FADASlopeEpisodePolicy | None,
) -> FADAEvaluationRollout:
    env.restore_rollout_snapshot(snapshot)
    controller = FADAPlaybackController(policy, device=next(policy.parameters()).device)
    base_positions: list[np.ndarray] = []
    yaws: list[float] = []
    feet_positions: list[np.ndarray] = []
    velocities: list[float] = []
    commands: list[float] = []
    states: list[np.ndarray] = []
    batches: list[FADATargetBatch] = []
    previous_action = np.zeros(policy.config.action_dim, dtype=np.float32)
    records: deque[FADACausalTransition] = deque(
        maxlen=policy.config.history_length + policy.config.prediction_horizon - 1
    )
    terminal_reason = "horizon"
    angle = np.deg2rad(episode_policy.geometry.angle_deg) if episode_policy else 0.0
    forward_axis = np.asarray([np.cos(angle), 0.0, np.sin(angle)])
    for step in range(control_steps):
        current_command = scheduled_target_command(
            (0.0, 0.0, 0.0), command, ramp_steps=ramp_steps, step=step
        )
        apply_external_command(env, current_command)
        current_observation = _project_observation(env.state, policy)
        action = controller.act(env.state.obs, current_command[None, :])
        state = env.step(action.detach().cpu().numpy().astype(np.float32))
        executed_action = action.detach().cpu().numpy().astype(np.float32)[0]
        next_observation = _project_observation(state, policy)
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
        done = bool(terminated[0]) or bool(truncated[0])
        lifecycle_reason = rollout_terminal_reasons(state, num_envs=1)[0]
        semantic_reason = None
        if episode_policy is not None:
            semantic_reason = episode_policy.classify(
                base_pos_w=base[0], feet_pos_w=feet[0], done=False
            ).terminal_reason
        step_terminal_reason = lifecycle_reason or semantic_reason
        if step_terminal_reason is None and not done:
            records.append(
                FADACausalTransition(
                    observation=current_observation,
                    previous_action=previous_action.copy(),
                    command=current_command.copy(),
                    executed_action=executed_action.copy(),
                    next_observation=next_observation,
                    episode_id=0,
                    timestep=step,
                )
            )
            if len(records) == records.maxlen:
                window = fada_target_batch_from_window(tuple(records), policy.config)
                if window is not None:
                    batches.append(window)
        previous_action = executed_action.copy()
        if step_terminal_reason is not None or done:
            terminal_reason = step_terminal_reason or "environment_termination"
            break
    if not batches:
        raise RuntimeError(
            "FADA evaluation rollout produced no complete causal windows; "
            "increase control_steps or inspect early termination"
        )
    return FADAEvaluationRollout(
        trajectory=FADASlopeTrajectory(
            base_pos_w=np.asarray(base_positions),
            base_yaw_rad=np.asarray(yaws),
            feet_pos_w=np.asarray(feet_positions),
            forward_velocity_mps=np.asarray(velocities),
            command_forward_mps=np.asarray(commands),
            physics_states=tuple(states),
            terminal_reason=terminal_reason,
            control_dt_s=float(env.cfg.ctrl_dt),
        ).validate(),
        target_batch=concat_fada_target_batches(batches, policy.config),
    )


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
    source_policy: FADAPlannerIDMPolicy,
    adapted_policy: FADAPlannerIDMPolicy,
    *,
    command: np.ndarray,
    control_steps: int,
    ramp_steps: int,
    episode_policy: FADASlopeEpisodePolicy | None,
) -> tuple[FADAEvaluationRollout, FADAEvaluationRollout]:
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


def _compare_flat_summaries(zero_shot: dict[str, Any], adapted: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_abs_lateral_m": float(zero_shot["mean_abs_lateral_m"])
        - float(adapted["mean_abs_lateral_m"]),
        "forward_velocity_mae_mps": float(zero_shot["forward_velocity_mae_mps"])
        - float(adapted["forward_velocity_mae_mps"]),
        "forward_progress_m": float(adapted["forward_progress_m"])
        - float(zero_shot["forward_progress_m"]),
        "steps": float(adapted["steps"]) - float(zero_shot["steps"]),
    }


def _trial_record(
    *,
    index: int,
    command: tuple[float, ...],
    zero: FADAEvaluationRollout,
    adapted: FADAEvaluationRollout,
    zero_trajectory: dict[str, Any],
    adapted_trajectory: dict[str, Any],
    trajectory_improvement: dict[str, float],
    source_policy: FADAPlannerIDMPolicy,
    adapted_policy: FADAPlannerIDMPolicy,
) -> dict[str, Any]:
    zero_diagnostic = summarize_fada_own_rollout(source_policy, zero.target_batch)
    adapted_diagnostic = summarize_fada_own_rollout(adapted_policy, adapted.target_batch)
    return {
        "trial_index": index,
        "command": list(command),
        "zero_shot": {
            "trajectory": zero_trajectory,
            "idm": zero_diagnostic,
        },
        "adapted": {
            "trajectory": adapted_trajectory,
            "idm": adapted_diagnostic,
        },
        "improvement": {
            "trajectory": trajectory_improvement,
            "idm": compare_fada_rollout_diagnostics(zero_diagnostic, adapted_diagnostic),
        },
    }


def _aggregate_trial_records(trials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "zero_shot": {
            "trajectory": aggregate_policy_summaries(
                [trial["zero_shot"]["trajectory"] for trial in trials]
            ),
            "idm": aggregate_numeric_summaries([trial["zero_shot"]["idm"] for trial in trials]),
        },
        "adapted": {
            "trajectory": aggregate_policy_summaries(
                [trial["adapted"]["trajectory"] for trial in trials]
            ),
            "idm": aggregate_numeric_summaries([trial["adapted"]["idm"] for trial in trials]),
        },
        "improvement": {
            "trajectory": aggregate_improvement_summaries(
                [trial["improvement"]["trajectory"] for trial in trials]
            ),
            "idm": aggregate_improvement_summaries(
                [trial["improvement"]["idm"] for trial in trials]
            ),
        },
    }


def _evaluate_command_pairs(
    env: Any,
    source_policy: FADAPlannerIDMPolicy,
    adapted_policy: FADAPlannerIDMPolicy,
    *,
    commands: tuple[tuple[float, ...], ...],
    control_steps: int,
    ramp_steps: int,
    geometry: FADASlopeGeometry | None,
    representative_index: int,
) -> tuple[
    list[dict[str, Any]],
    tuple[FADAEvaluationRollout, FADAEvaluationRollout],
]:
    trials: list[dict[str, Any]] = []
    representative_pair = None
    for index, command in enumerate(commands):
        command_array = np.asarray(command, dtype=np.float32)
        episode_policy = (
            FADASlopeEpisodePolicy(geometry, (command,)) if geometry is not None else None
        )
        zero, adapted = _run_pair(
            env,
            source_policy,
            adapted_policy,
            command=command_array,
            control_steps=control_steps,
            ramp_steps=ramp_steps,
            episode_policy=episode_policy,
        )
        if geometry is None:
            zero_summary = _flat_summary(zero.trajectory)
            adapted_summary = _flat_summary(adapted.trajectory)
            comparison = _compare_flat_summaries(zero_summary, adapted_summary)
        else:
            zero_summary = compact_slope_summary(
                summarize_slope_trajectory(zero.trajectory, geometry)
            )
            adapted_summary = compact_slope_summary(
                summarize_slope_trajectory(adapted.trajectory, geometry)
            )
            comparison = compare_slope_summaries(zero_summary, adapted_summary)
        trials.append(
            _trial_record(
                index=index,
                command=command,
                zero=zero,
                adapted=adapted,
                zero_trajectory=zero_summary,
                adapted_trajectory=adapted_summary,
                trajectory_improvement=comparison,
                source_policy=source_policy,
                adapted_policy=adapted_policy,
            )
        )
        if index == representative_index:
            representative_pair = (zero, adapted)
    if representative_pair is None:
        raise ValueError("FADA representative trial index is outside the command sequence")
    return trials, representative_pair


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
    commands, representative_index = _evaluation_commands(cfg, domain)
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
        slope_trials, representative_slope = _evaluate_command_pairs(
            env,
            source.policy,
            adapted.policy,
            commands=commands,
            control_steps=control_steps,
            ramp_steps=ramp_steps,
            geometry=geometry,
            representative_index=representative_index,
        )
        metrics: dict[str, Any] = {
            "trial_count": len(commands),
            "representative_trial_index": representative_index,
            "trials": slope_trials,
            "aggregate": _aggregate_trial_records(slope_trials),
        }
        representative_flat = None
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
            flat_trials, representative_flat = _evaluate_command_pairs(
                flat_env,
                source.policy,
                adapted.policy,
                commands=commands,
                control_steps=control_steps,
                ramp_steps=ramp_steps,
                geometry=None,
                representative_index=representative_index,
            )
            metrics["flat_regression"] = {
                "trials": flat_trials,
                "aggregate": _aggregate_trial_records(flat_trials),
            }
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}-", dir=output_dir.parent
        ) as tmp:
            stage = Path(tmp) / output_dir.name
            stage.mkdir()
            render_fn(
                env=env,
                state_list=representative_slope[0].trajectory.physics_states,
                output_video=stage / "zero_shot.mp4",
                camera_kwargs=target_tracking_camera_kwargs(),
            )
            render_fn(
                env=env,
                state_list=representative_slope[1].trajectory.physics_states,
                output_video=stage / "adapted.mp4",
                camera_kwargs=target_tracking_camera_kwargs(),
            )
            files = ["zero_shot.mp4", "adapted.mp4", "metrics.json"]
            if representative_flat is not None:
                render_fn(
                    env=flat_env,
                    state_list=representative_flat[0].trajectory.physics_states,
                    output_video=stage / "zero_shot_flat.mp4",
                    camera_kwargs=target_tracking_camera_kwargs(),
                )
                render_fn(
                    env=flat_env,
                    state_list=representative_flat[1].trajectory.physics_states,
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
                        "schema_version": "fada-slope-evaluation/v2",
                        "target_domain_id": domain.target_domain_id,
                        "trial_count": len(commands),
                        "representative_trial_index": representative_index,
                        "representative_command": list(commands[representative_index]),
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
