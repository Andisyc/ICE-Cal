from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import msgpack
import numpy as np
import torch

from unilab.algos.torch.distill.fada.adaptation_checkpoint import (
    assert_fada_adaptation_source_checkpoint,
)
from unilab.algos.torch.distill.fada.checkpoint import load_fada_policy_checkpoint
from unilab.algos.torch.distill.fada.target_data import (
    FADA_REAL_TARGET_ARTIFACT_SCHEMA_VERSION,
    FADATargetBatch,
    save_fada_target_artifact,
)
from unilab.algos.torch.distill.fada.windows import (
    FADACausalTransition,
    build_fada_causal_window,
)
from unilab.algos.torch.distill.workflow import file_sha256

G1_FADA_DEFAULT_DOF_POS = np.asarray(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RoboJudoTrajectory:
    path: Path
    lowstate: tuple[Mapping[str, Any], ...]
    torso_imu: tuple[Mapping[str, Any], ...]
    lowcmd: tuple[Mapping[str, Any], ...]
    duration_seconds: float


@dataclass(frozen=True)
class RealTargetEpisode:
    windows: tuple[Any, ...]
    source_metadata: Mapping[str, Any]
    num_policy_steps: int
    accepted_policy_steps: int


def load_robojudo_msgpack_episode(path: str | Path) -> RoboJudoTrajectory:
    source = Path(path)
    by_kind: dict[str, list[Mapping[str, Any]]] = {
        "lowstate": [],
        "torso_imu": [],
        "lowcmd": [],
    }
    header: Mapping[str, Any] | None = None
    summary: Mapping[str, Any] | None = None
    with source.open("rb") as stream:
        for event in msgpack.Unpacker(stream, raw=False):
            if not isinstance(event, Mapping):
                raise ValueError(f"RoboJuDo trajectory event must be a mapping: {source}")
            kind = event.get("kind")
            if kind == "header":
                if header is not None:
                    raise ValueError(f"RoboJuDo trajectory contains multiple headers: {source}")
                header = event.get("payload")
            elif kind == "summary":
                summary = event.get("payload")
            elif kind in by_kind:
                by_kind[str(kind)].append(event)
    if not isinstance(header, Mapping) or header.get("schema_version") != 1:
        raise ValueError(f"unsupported RoboJuDo trajectory schema: {source}")
    metadata = header.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("num_dofs") != 29:
        raise ValueError(f"RoboJuDo trajectory must contain 29-DoF G1 data: {source}")
    if not isinstance(summary, Mapping):
        raise ValueError(f"RoboJuDo trajectory is missing its final summary: {source}")
    if summary.get("dropped") != 0:
        raise ValueError(f"RoboJuDo trajectory contains dropped events: {source}")
    counts = summary.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError(f"RoboJuDo trajectory summary counts are missing: {source}")
    for kind, events in by_kind.items():
        if not events or counts.get(kind) != len(events):
            raise ValueError(f"RoboJuDo trajectory {kind} count mismatch: {source}")
    return RoboJudoTrajectory(
        path=source,
        lowstate=tuple(by_kind["lowstate"]),
        torso_imu=tuple(by_kind["torso_imu"]),
        lowcmd=tuple(by_kind["lowcmd"]),
        duration_seconds=float(summary["duration_seconds"]),
    )


def _event_times(events: Sequence[Mapping[str, Any]]) -> np.ndarray:
    times = np.asarray([event["monotonic_time_ns"] for event in events], dtype=np.int64)
    if times.ndim != 1 or len(times) < 2 or bool(np.any(np.diff(times) <= 0)):
        raise ValueError("RoboJuDo event timestamps must be strictly increasing")
    return times


def fit_policy_step_timeline(
    lowcmd: Sequence[Mapping[str, Any]],
    *,
    control_dt: float,
    max_alignment_error_seconds: float = 0.008,
) -> np.ndarray:
    """Recover the 50 Hz policy lattice from the duplicated 100 Hz LowCmd stream."""

    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt must be finite and positive")
    times_ns = _event_times(lowcmd)
    times = (times_ns - times_ns[0]).astype(np.float64) * 1.0e-9
    q = np.asarray([event["payload"]["motor"]["q"] for event in lowcmd], dtype=np.float32)
    if q.shape != (len(lowcmd), 29) or not bool(np.isfinite(q).all()):
        raise ValueError("RoboJuDo LowCmd q must be finite [N, 29]")
    changed = np.max(np.abs(np.diff(q, axis=0)), axis=1) >= 1.0e-7
    change_times = times[1:][changed]
    if len(change_times) < 4:
        raise ValueError("RoboJuDo LowCmd stream has too few policy-action changes")
    observed_period = float(np.median(np.diff(change_times)))
    if not 0.75 * control_dt <= observed_period <= 1.5 * control_dt:
        raise ValueError("RoboJuDo LowCmd stream has an invalid policy period")
    change_indices = np.flatnonzero(changed).astype(np.int64) + 1
    selected: list[int] = []
    if float(times[int(change_indices[0])] - times[0]) >= 0.75 * observed_period:
        selected.append(0)
    selected.append(int(change_indices[0]))
    previous = int(change_indices[0])
    for current_raw in change_indices[1:]:
        current = int(current_raw)
        elapsed = float(times[current] - times[previous])
        intervals = max(1, int(round(elapsed / observed_period)))
        for offset in range(1, intervals):
            target = times[previous] + elapsed * (offset / intervals)
            right = int(np.searchsorted(times, target, side="left"))
            right = min(right, len(times) - 1)
            left = max(right - 1, 0)
            nearest = left if abs(times[left] - target) <= abs(times[right] - target) else right
            if abs(times[nearest] - target) <= float(max_alignment_error_seconds):
                selected.append(nearest)
        selected.append(current)
        previous = current
    indices = np.asarray(sorted(set(selected)), dtype=np.int64)
    if len(indices) < 66 or len(np.unique(indices)) != len(indices):
        raise ValueError("RoboJuDo LowCmd stream cannot form a unique 50 Hz policy timeline")
    return indices


def align_sensor_before_action(
    sensor_events: Sequence[Mapping[str, Any]],
    action_times_ns: np.ndarray,
    *,
    max_lag_seconds: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    sensor_times = _event_times(sensor_events)
    indices = np.searchsorted(sensor_times, action_times_ns, side="right") - 1
    if bool(np.any(indices < 0)):
        raise ValueError("RoboJuDo sensor stream starts after the first policy action")
    lag = (action_times_ns - sensor_times[indices]).astype(np.float64) * 1.0e-9
    if bool(np.any(lag < 0.0)) or float(np.max(lag)) > max_lag_seconds:
        raise ValueError("RoboJuDo sensor-to-action alignment exceeds the allowed lag")
    return indices.astype(np.int64), np.asarray(lag * 1.0e3, dtype=np.float64)


def project_torso_imu(
    quaternions_wxyz: np.ndarray,
    gyroscopes: np.ndarray,
    accelerometers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    gyro = np.asarray(gyroscopes, dtype=np.float32)
    accel = np.asarray(accelerometers, dtype=np.float32)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("torso quaternion must have shape [N, 4]")
    if gyro.shape != (len(quaternions), 3) or accel.shape != gyro.shape:
        raise ValueError("torso gyroscope and accelerometer must have shape [N, 3]")
    if not bool(np.isfinite(quaternions).all() and np.isfinite(gyro).all() and np.isfinite(accel).all()):
        raise ValueError("torso IMU values must be finite")
    norms = np.linalg.norm(quaternions, axis=1)
    if bool(np.any((norms < 0.9) | (norms > 1.1))) or bool(
        np.any(np.linalg.norm(accel, axis=1) < 1.0e-3)
    ):
        raise ValueError("torso IMU contains an invalid sample")
    q = quaternions / norms[:, None]
    w, x, y, z = (q[:, index] for index in range(4))
    reference_yaw = float(
        np.arctan2(2.0 * (w[0] * z[0] + x[0] * y[0]), 1.0 - 2.0 * (y[0] ** 2 + z[0] ** 2))
    )
    up = np.column_stack(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    )
    c, s = math.cos(reference_yaw), math.sin(reference_yaw)
    aligned = np.column_stack(
        (c * up[:, 0] + s * up[:, 1], -s * up[:, 0] + c * up[:, 1], up[:, 2])
    )
    return gyro.copy(), np.asarray(-aligned, dtype=np.float32)


def reconstruct_fada_state_base(
    lowstate_payloads: Sequence[Mapping[str, Any]],
    torso_payloads: Sequence[Mapping[str, Any]],
    *,
    default_dof_pos: np.ndarray = G1_FADA_DEFAULT_DOF_POS,
) -> np.ndarray:
    q = np.asarray([item["motor"]["q"] for item in lowstate_payloads], dtype=np.float32)
    dq = np.asarray([item["motor"]["dq"] for item in lowstate_payloads], dtype=np.float32)
    quat = np.asarray([item["quaternion"] for item in torso_payloads], dtype=np.float64)
    gyro = np.asarray([item["gyroscope"] for item in torso_payloads], dtype=np.float32)
    accel = np.asarray([item["accelerometer"] for item in torso_payloads], dtype=np.float32)
    policy_gyro, gravity = project_torso_imu(quat, gyro, accel)
    default = np.asarray(default_dof_pos, dtype=np.float32)
    if q.shape != dq.shape or q.shape[1:] != (29,) or default.shape != (29,):
        raise ValueError("G1 real target joint arrays must use the 29-DoF policy order")
    state = np.concatenate((policy_gyro * 0.25, gravity, q - default, dq * 0.05), axis=1)
    if state.shape != (len(q), 64) or not bool(np.isfinite(state).all()):
        raise ValueError("G1 real target state base must be finite [N, 64]")
    return np.asarray(state, dtype=np.float32)


@torch.no_grad()
def recover_gait_phase_offset(
    policy: Any,
    state_base: np.ndarray,
    actions: np.ndarray,
    command: np.ndarray,
    *,
    control_dt: float,
    gait_frequency: float,
    valid_mask: np.ndarray | None = None,
    search_steps: int = 48,
    evaluation_windows: int = 12,
) -> tuple[float, float]:
    history_length = int(policy.config.history_length)
    if len(state_base) <= history_length or actions.shape != (len(state_base), 29):
        raise ValueError("real target episode is too short for gait-phase recovery")
    valid = (
        np.ones(len(state_base), dtype=np.bool_)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=np.bool_)
    )
    if valid.shape != (len(state_base),):
        raise ValueError("gait-phase valid_mask must match the episode length")
    eligible = np.asarray(
        [
            index
            for index in range(history_length, len(state_base))
            if bool(np.all(valid[index - history_length + 1 : index + 1]))
        ],
        dtype=np.int64,
    )
    if not len(eligible):
        raise ValueError("real target has no valid history for gait-phase recovery")
    eval_count = min(int(evaluation_windows), len(eligible))
    eval_indices = eligible[
        np.linspace(0, len(eligible) - 1, num=eval_count, dtype=np.int64)
    ]
    history_offsets = np.arange(history_length - 1, -1, -1, dtype=np.int64)
    history_indices = eval_indices[:, None] - history_offsets[None, :]
    previous = np.zeros_like(actions)
    previous[1:] = actions[:-1]
    base_history = state_base[history_indices]
    action_history = previous[history_indices]
    targets = actions[eval_indices]
    delta = 2.0 * np.pi * float(gait_frequency) * float(control_dt)
    device = next(policy.parameters()).device

    def losses(candidates: np.ndarray) -> np.ndarray:
        result: list[np.ndarray] = []
        for start in range(0, len(candidates), 16):
            chunk = candidates[start : start + 16]
            phase = np.mod(
                chunk[:, None, None] + delta * history_indices[None, :, :], 2.0 * np.pi
            )
            phase_pair = np.stack((phase, np.mod(phase + np.pi, 2.0 * np.pi)), axis=-1)
            obs = np.concatenate(
                (
                    np.broadcast_to(base_history, (len(chunk), *base_history.shape)),
                    phase_pair.astype(np.float32),
                ),
                axis=-1,
            ).reshape(-1, history_length, 66)
            act = np.broadcast_to(
                action_history, (len(chunk), *action_history.shape)
            ).reshape(-1, history_length, 29)
            cmd = np.broadcast_to(command, (len(chunk) * eval_count, 3))
            predicted = policy(
                torch.from_numpy(obs).to(device),
                torch.from_numpy(np.asarray(act, dtype=np.float32).copy()).to(device),
                torch.from_numpy(np.asarray(cmd, dtype=np.float32).copy()).to(device),
            ).action.detach().cpu().numpy()
            target = np.broadcast_to(targets, (len(chunk), *targets.shape)).reshape(-1, 29)
            mse = np.mean(np.square(predicted - target), axis=1).reshape(len(chunk), eval_count)
            result.append(np.mean(mse, axis=1))
        return np.concatenate(result)

    coarse = np.linspace(0.0, 2.0 * np.pi, num=int(search_steps), endpoint=False)
    coarse_loss = losses(coarse)
    best = float(coarse[int(np.argmin(coarse_loss))])
    width = 2.0 * np.pi / int(search_steps)
    refined = np.mod(np.linspace(best - width, best + width, num=17), 2.0 * np.pi)
    refined_loss = losses(refined)
    index = int(np.argmin(refined_loss))
    return float(refined[index]), float(refined_loss[index])


def _stack_windows(windows: Sequence[Any]) -> FADATargetBatch:
    return FADATargetBatch(
        observation_history=torch.from_numpy(
            np.stack([window.observation_history for window in windows]).astype(np.float32)
        ),
        action_history=torch.from_numpy(
            np.stack([window.action_history for window in windows]).astype(np.float32)
        ),
        command=torch.from_numpy(
            np.stack([window.command for window in windows]).astype(np.float32)
        ),
        realized_future=torch.from_numpy(
            np.stack([window.realized_future for window in windows]).astype(np.float32)
        ),
        executed_action_chunk=torch.from_numpy(
            np.stack([window.executed_action_chunk for window in windows]).astype(np.float32)
        ),
        episode_id=torch.tensor([window.episode_id for window in windows], dtype=torch.int64),
        start_timestep=torch.tensor(
            [window.start_timestep for window in windows], dtype=torch.int64
        ),
    )


def build_real_target_episode(
    trajectory: RoboJudoTrajectory,
    policy: Any,
    *,
    episode_id: int,
    command: Sequence[float],
    control_dt: float,
    gait_frequency: float,
    max_phase_action_mse: float,
) -> RealTargetEpisode:
    cmd = np.asarray(command, dtype=np.float32)
    if cmd.shape != (3,) or not bool(np.isfinite(cmd).all()):
        raise ValueError("real target command must be finite with shape [3]")
    policy_indices = fit_policy_step_timeline(trajectory.lowcmd, control_dt=control_dt)
    selected_cmd = [trajectory.lowcmd[int(index)] for index in policy_indices]
    action_times = np.asarray(
        [event["monotonic_time_ns"] for event in selected_cmd], dtype=np.int64
    )
    state_indices, state_lag = align_sensor_before_action(trajectory.lowstate, action_times)
    imu_indices, imu_lag = align_sensor_before_action(trajectory.torso_imu, action_times)
    valid_sensor = (state_lag <= 5.0) & (imu_lag <= 5.0)
    lowstate_payloads = [trajectory.lowstate[int(index)]["payload"] for index in state_indices]
    torso_payloads = [trajectory.torso_imu[int(index)]["payload"] for index in imu_indices]
    state_base = reconstruct_fada_state_base(lowstate_payloads, torso_payloads)
    targets = np.asarray(
        [event["payload"]["motor"]["q"] for event in selected_cmd], dtype=np.float32
    )
    actions = targets - G1_FADA_DEFAULT_DOF_POS[None, :]
    phase_offset, phase_mse = recover_gait_phase_offset(
        policy,
        state_base,
        actions,
        cmd,
        control_dt=control_dt,
        gait_frequency=gait_frequency,
        valid_mask=valid_sensor,
    )
    if phase_mse > float(max_phase_action_mse):
        raise ValueError(
            f"gait phase recovery is not trustworthy for {trajectory.path.name}: "
            f"action_mse={phase_mse:.6f} limit={max_phase_action_mse:.6f}"
        )
    phase = np.mod(
        phase_offset
        + np.arange(len(state_base), dtype=np.float32)
        * (2.0 * np.pi * float(gait_frequency) * float(control_dt)),
        2.0 * np.pi,
    )
    observations = np.concatenate(
        (state_base, np.column_stack((phase, np.mod(phase + np.pi, 2.0 * np.pi)))), axis=1
    ).astype(np.float32)
    history_length = int(policy.config.history_length)
    horizon = int(policy.config.prediction_horizon)
    transitions = [
        FADACausalTransition(
            observation=observations[timestep],
            previous_action=actions[timestep - 1],
            command=cmd,
            executed_action=actions[timestep],
            next_observation=observations[timestep + 1],
            episode_id=int(episode_id),
            timestep=timestep,
        )
        for timestep in range(history_length, len(observations) - 1)
    ]
    span = history_length + horizon - 1
    windows = tuple(
        window
        for start in range(0, len(transitions) - span + 1)
        if bool(
            np.all(
                valid_sensor[
                    history_length + start : history_length + start + span + 1
                ]
            )
        )
        if (
            window := build_fada_causal_window(
                transitions[start : start + span],
                history_length=history_length,
                prediction_horizon=horizon,
            )
        )
        is not None
    )
    if not windows:
        raise ValueError(f"real target episode produced no causal windows: {trajectory.path}")
    return RealTargetEpisode(
        windows=windows,
        num_policy_steps=len(observations),
        accepted_policy_steps=int(np.sum(valid_sensor)),
        source_metadata={
            "name": trajectory.path.name,
            "sha256": file_sha256(trajectory.path),
            "num_policy_steps": len(observations),
            "rejected_sensor_steps": int(np.sum(~valid_sensor)),
            "num_windows": len(windows),
            "phase_offset": phase_offset,
            "phase_action_mse": phase_mse,
            "max_sensor_lag_ms": float(max(np.max(state_lag), np.max(imu_lag))),
        },
    )


def prepare_fada_real_target(
    *,
    input_dir: str | Path,
    source_checkpoint_path: str | Path,
    output_path: str | Path,
    command: Sequence[float],
    target_domain_id: str,
    condition_label: str,
    device: str = "cpu",
    control_dt: float = 0.02,
    gait_frequency: float = 1.5,
    max_phase_action_mse: float = 0.02,
) -> dict[str, Any]:
    source_path = Path(source_checkpoint_path)
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"FADA real target artifact already exists: {target}")
    loaded = assert_fada_adaptation_source_checkpoint(
        load_fada_policy_checkpoint(source_path, device=device),
        expected_behavior_profile="configured_gait",
    )
    config = loaded.policy.config
    if (
        config.obs_dim,
        config.action_dim,
        config.command_dim,
        config.history_length,
        config.prediction_horizon,
        config.observation_contract,
    ) != (66, 29, 3, 30, 6, "g1_fada_state_v2"):
        raise ValueError("real target preparation requires the active 66/29/3 H30 K6 checkpoint")
    sources = sorted(Path(input_dir).glob("*.msgpack"))
    if not sources:
        raise FileNotFoundError(f"no RoboJuDo .msgpack trajectories found in {input_dir}")
    loaded.policy.eval()
    episodes = [
        build_real_target_episode(
            load_robojudo_msgpack_episode(path),
            loaded.policy,
            episode_id=index,
            command=command,
            control_dt=control_dt,
            gait_frequency=gait_frequency,
            max_phase_action_mse=max_phase_action_mse,
        )
        for index, path in enumerate(sources)
    ]
    all_windows = tuple(window for episode in episodes for window in episode.windows)
    batch = _stack_windows(all_windows).validate(config)
    source_sha = file_sha256(source_path)
    fingerprint_payload = {
        "source_checkpoint_sha256": source_sha,
        "target_domain_id": target_domain_id,
        "condition_label": condition_label,
        "command": list(map(float, command)),
        "control_dt": float(control_dt),
        "gait_frequency": float(gait_frequency),
        "source_trajectories": [dict(episode.source_metadata) for episode in episodes],
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = {
        "policy_checkpoint_sha256": source_sha,
        "config_fingerprint": config_fingerprint,
        "task": "G1WalkFlat",
        "num_envs": 1,
        "num_windows": len(all_windows),
        "target_domain_id": target_domain_id,
        "target_domain_kind": "real_robot_load",
        "command_sequence": [list(map(float, command))],
        "observation_contract": config.observation_contract,
        "episode_count": len(episodes),
        "accepted_steps": sum(episode.accepted_policy_steps for episode in episodes),
        "robot": "g1",
        "condition_label": condition_label,
        "control_dt": float(control_dt),
        "source_trajectories": [dict(episode.source_metadata) for episode in episodes],
    }
    save_fada_target_artifact(
        target,
        batch,
        config=config,
        metadata=metadata,
        schema_version=FADA_REAL_TARGET_ARTIFACT_SCHEMA_VERSION,
    )
    return {
        "output_path": str(target),
        "source_checkpoint_sha256": source_sha,
        "episode_count": len(episodes),
        "policy_steps": sum(episode.num_policy_steps for episode in episodes),
        "num_windows": len(all_windows),
        "episodes": [dict(episode.source_metadata) for episode in episodes],
    }
