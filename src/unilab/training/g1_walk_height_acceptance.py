"""G1WalkHeight checkpoint identity and bounded physical acceptance owner.

Status: active live-acceptance probe with synthetic contract coverage.
Upstream: ``check_unilab_g1_walk_height_teacher.py``.
Downstream: immutable checkpoint qualification for two-teacher distillation.
Evidence: contract-confirmed; real checkpoint quality remains live-only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.teacher import (
    DistillationTeacherSpec,
    load_sac_teacher_policy,
)
from unilab.base.registry import ensure_registries
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.common import create_env
from unilab.training.seed import apply_training_seed

ROOT_DIR = Path(__file__).resolve().parents[3]

EXPECTED_TASK_NAME = "G1WalkHeight"
EXPECTED_SIM_BACKEND = "mujoco"
EXPECTED_ALGO = "sac"
EXPECTED_OBS_DIM = 99
EXPECTED_CRITIC_OBS_DIM = 102
EXPECTED_ACTION_DIM = 29
COMMAND_OBS_SLICE = slice(93, 96)
TARGET_HEIGHT_OBS_INDEX = 96
DEFAULT_PROBES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("forward_slow", (0.2, 0.0, 0.0)),
    ("forward_nominal", (0.5, 0.0, 0.0)),
    ("lateral", (0.0, 0.2, 0.0)),
    ("yaw", (0.0, 0.0, 0.4)),
)


@dataclass(frozen=True)
class WalkRunIdentity:
    run_dir: Path
    run_config_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    max_tilt_deg: float
    effective_seed: int
    config: dict[str, Any] = field(repr=False)


@dataclass
class WalkRolloutSamples:
    command: list[np.ndarray] = field(default_factory=list)
    measured_linvel: list[np.ndarray] = field(default_factory=list)
    measured_gyro: list[np.ndarray] = field(default_factory=list)
    target_height: list[np.ndarray] = field(default_factory=list)
    measured_height: list[np.ndarray] = field(default_factory=list)
    tilt_deg: list[np.ndarray] = field(default_factory=list)
    terminated: list[np.ndarray] = field(default_factory=list)
    truncated: list[np.ndarray] = field(default_factory=list)
    command_obs: list[np.ndarray] = field(default_factory=list)
    target_height_obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    scored: list[np.ndarray] = field(default_factory=list)

    def append(self, *, scored: bool, **rows: np.ndarray) -> None:
        command = _matrix(rows["command"], name="command", columns=3)
        count = int(command.shape[0])
        matrices = {
            "measured_linvel": _matrix(rows["measured_linvel"], name="measured_linvel", columns=3),
            "measured_gyro": _matrix(rows["measured_gyro"], name="measured_gyro", columns=3),
            "command_obs": _matrix(rows["command_obs"], name="command_obs", columns=3),
            "actions": _matrix(rows["actions"], name="actions", columns=EXPECTED_ACTION_DIM),
        }
        vectors = {
            name: _vector(
                rows[name],
                name=name,
                dtype=bool if name in {"terminated", "truncated"} else np.float32,
            )
            for name in (
                "target_height",
                "measured_height",
                "tilt_deg",
                "terminated",
                "truncated",
                "target_height_obs",
            )
        }
        for name, value in {**matrices, **vectors}.items():
            if int(value.shape[0]) != count:
                raise ValueError(f"{name} rows={value.shape[0]} do not match command rows={count}")
        self.command.append(command.copy())
        for name, value in matrices.items():
            cast(list[np.ndarray], getattr(self, name)).append(value.copy())
        for name, value in vectors.items():
            cast(list[np.ndarray], getattr(self, name)).append(value.copy())
        self.scored.append(np.full(count, bool(scored), dtype=bool))

    def concat(self, name: str) -> np.ndarray:
        chunks = cast(list[np.ndarray], getattr(self, name))
        if chunks:
            return np.concatenate(chunks, axis=0)
        if name in {"command", "measured_linvel", "measured_gyro", "command_obs"}:
            return np.empty((0, 3), dtype=np.float32)
        if name == "actions":
            return np.empty((0, EXPECTED_ACTION_DIM), dtype=np.float32)
        return np.empty((0,), dtype=np.float32)


def _matrix(value: Any, *, name: str, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {array.shape}")
    return array


def _vector(value: Any, *, name: str, dtype: Any = np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (N,) or (N, 1), got {array.shape}")
    return array


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"run_config.json field {name!r} must be an object")
    return value


def load_run_identity(
    *,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    expected_sha256: str,
    expected_target_height: float,
) -> WalkRunIdentity:
    """Validate immutable artifact and nominal 99-D WalkHeight training identity."""

    resolved_run_dir = Path(run_dir).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    run_config_path = resolved_run_dir / "run_config.json"
    if not resolved_run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {resolved_run_dir}")
    if not run_config_path.is_file():
        raise FileNotFoundError(f"run_config.json does not exist: {run_config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if checkpoint.parent != resolved_run_dir:
        raise ValueError("checkpoint must be directly inside the accepted run directory")
    normalized_hash = str(expected_sha256).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_hash) is None:
        raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters")
    observed_hash = _file_sha256(checkpoint)
    if observed_hash != normalized_hash:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected={normalized_hash} observed={observed_hash}"
        )

    root = _require_mapping(json.loads(run_config_path.read_text(encoding="utf-8")), name="root")
    config = dict(_require_mapping(root.get("config"), name="config"))
    training = _require_mapping(config.get("training"), name="config.training")
    algo = _require_mapping(config.get("algo"), name="config.algo")
    env = _require_mapping(config.get("env"), name="config.env")
    commands = _require_mapping(env.get("commands"), name="config.env.commands")
    reward = _require_mapping(config.get("reward"), name="config.reward")
    if str(training.get("task_name")) != EXPECTED_TASK_NAME:
        raise ValueError(f"expected task {EXPECTED_TASK_NAME}, got {training.get('task_name')}")
    if str(training.get("sim_backend")) != EXPECTED_SIM_BACKEND:
        raise ValueError(
            f"expected backend {EXPECTED_SIM_BACKEND}, got {training.get('sim_backend')}"
        )
    if str(algo.get("algo")) != EXPECTED_ALGO:
        raise ValueError(f"expected algorithm {EXPECTED_ALGO}, got {algo.get('algo')}")
    observed_range = np.asarray(commands.get("height_range"), dtype=np.float64)
    expected_range = np.full(2, float(expected_target_height), dtype=np.float64)
    if observed_range.shape != (2,) or not np.allclose(
        observed_range, expected_range, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(
            "WalkHeight acceptance requires the nominal fixed target range: "
            f"expected={expected_range.tolist()} observed={observed_range.tolist()}"
        )
    if not np.isclose(
        float(commands.get("default_height", float("nan"))),
        float(expected_target_height),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("WalkHeight default target mismatch")
    if commands.get("observe_height_command") is not True:
        raise ValueError("WalkHeight acceptance requires observe_height_command=true")
    if commands.get("random_height_during_walking") is not False:
        raise ValueError(
            "WalkHeight nominal acceptance requires random_height_during_walking=false"
        )
    max_tilt_deg = float(reward.get("max_tilt_deg", float("nan")))
    if not np.isfinite(max_tilt_deg) or max_tilt_deg <= 0.0:
        raise ValueError(f"config.reward.max_tilt_deg must be positive, got {max_tilt_deg}")
    run = _require_mapping(root.get("run", {}), name="run")
    seed = int(run.get("effective_seed", algo.get("seed", 1)))
    return WalkRunIdentity(
        run_dir=resolved_run_dir,
        run_config_path=run_config_path,
        checkpoint_path=checkpoint,
        checkpoint_sha256=observed_hash,
        max_tilt_deg=max_tilt_deg,
        effective_seed=seed,
        config=config,
    )


def load_policy(
    checkpoint_path: str | Path,
    cfg: DictConfig,
    *,
    obs_dim: int,
    action_dim: int,
    device: str,
) -> Callable[[np.ndarray], np.ndarray]:
    import torch

    teacher = load_sac_teacher_policy(
        checkpoint_path,
        DistillationTeacherSpec(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dim=int(cfg.algo.actor_hidden_dim),
            use_layer_norm=bool(cfg.algo.use_layer_norm),
            obs_normalization=bool(cfg.algo.obs_normalization),
        ),
        device=device,
    )

    def policy(obs: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
        with torch.inference_mode():
            return teacher(tensor).cpu().numpy().astype(np.float32, copy=False)

    return policy


def _runtime_sample(env: Any, state: Any, actions: np.ndarray) -> dict[str, np.ndarray]:
    obs = _matrix(state.obs["obs"], name="actor observation", columns=EXPECTED_OBS_DIM)
    upvector = np.asarray(
        env._backend.get_sensor_data(env.cfg.sensor.upvector), dtype=np.float32
    ).reshape(int(env.num_envs), -1)
    return {
        "command": _matrix(state.info["commands"], name="commands", columns=3),
        "measured_linvel": _matrix(env.get_local_linvel(), name="local linvel", columns=3),
        "measured_gyro": _matrix(env.get_gyro(), name="gyro", columns=3),
        "target_height": _vector(state.info["height_commands"], name="height_commands"),
        "measured_height": _vector(
            env._terrain_relative_base_height(), name="terrain_relative_base_height"
        ),
        "tilt_deg": np.rad2deg(np.arccos(np.clip(upvector[:, 2], -1.0, 1.0))),
        "terminated": np.asarray(state.terminated, dtype=bool),
        "truncated": np.asarray(state.truncated, dtype=bool),
        "command_obs": obs[:, COMMAND_OBS_SLICE],
        "target_height_obs": obs[:, TARGET_HEIGHT_OBS_INDEX],
        "actions": np.asarray(actions, dtype=np.float32),
    }


def _add(checks: list[dict[str, Any]], passed: bool, name: str, detail: str) -> None:
    checks.append({"level": "PASS" if passed else "FAIL", "name": name, "detail": detail})


def evaluate_samples(
    samples: WalkRolloutSamples,
    *,
    expected_command: tuple[float, float, float],
    expected_target_height: float,
    max_linear_velocity_error: float,
    max_yaw_velocity_error: float,
    max_height_mae: float,
    max_tilt_deg: float,
    requested_steps: int,
    executed_steps: int,
) -> dict[str, Any]:
    command = samples.concat("command")
    linvel = samples.concat("measured_linvel")
    gyro = samples.concat("measured_gyro")
    target_height = samples.concat("target_height")
    measured_height = samples.concat("measured_height")
    tilt = samples.concat("tilt_deg")
    terminated = samples.concat("terminated").astype(bool, copy=False)
    truncated = samples.concat("truncated").astype(bool, copy=False)
    command_obs = samples.concat("command_obs")
    height_obs = samples.concat("target_height_obs")
    actions = samples.concat("actions")
    scored = samples.concat("scored").astype(bool, copy=False)
    expected = np.asarray(expected_command, dtype=np.float32)
    finite = (
        all(
            np.all(np.isfinite(value))
            for value in (command, linvel, gyro, target_height, measured_height, tilt, actions)
        )
        and target_height.size > 0
    )
    scored_count = int(np.count_nonzero(scored))
    linear_error = (
        float(np.mean(np.linalg.norm(linvel[scored, :2] - command[scored, :2], axis=1)))
        if scored_count
        else float("inf")
    )
    yaw_error = (
        float(np.mean(np.abs(gyro[scored, 2] - command[scored, 2])))
        if scored_count
        else float("inf")
    )
    height_mae = (
        float(np.mean(np.abs(measured_height[scored] - target_height[scored])))
        if scored_count
        else float("inf")
    )
    command_identity_error = (
        float(np.max(np.abs(command - expected[None, :]))) if command.size else float("inf")
    )
    command_obs_error = (
        float(np.max(np.abs(command_obs - command))) if command.size else float("inf")
    )
    height_identity_error = (
        float(np.max(np.abs(target_height - expected_target_height)))
        if target_height.size
        else float("inf")
    )
    height_obs_error = (
        float(np.max(np.abs(height_obs - target_height))) if target_height.size else float("inf")
    )
    termination_count = int(np.count_nonzero(terminated))
    truncation_count = int(np.count_nonzero(truncated))
    observed_max_tilt = float(np.max(tilt)) if tilt.size else float("inf")
    checks: list[dict[str, Any]] = []
    _add(
        checks,
        executed_steps == requested_steps,
        "rollout/completed_window",
        f"executed_steps={executed_steps} requested_steps={requested_steps}",
    )
    _add(checks, finite, "rollout/finite", f"finite={finite}")
    _add(
        checks,
        termination_count == 0 and truncation_count == 0,
        "rollout/no_termination",
        f"terminated_rows={termination_count} truncated_rows={truncation_count}",
    )
    _add(
        checks,
        command_identity_error <= 1.0e-6,
        "rollout/fixed_command_identity",
        f"max_error={command_identity_error:.9g}",
    )
    _add(
        checks,
        command_obs_error <= 1.0e-6,
        "rollout/command_obs_roundtrip",
        f"indices=93:96 max_error={command_obs_error:.9g}",
    )
    _add(
        checks,
        height_identity_error <= 1.0e-6 and height_obs_error <= 1.0e-6,
        "rollout/target_height_roundtrip",
        f"target_error={height_identity_error:.9g} obs_error={height_obs_error:.9g}",
    )
    _add(
        checks,
        scored_count > 0 and linear_error <= max_linear_velocity_error,
        "quality/linear_velocity_error",
        f"mean_l2={linear_error:.9g} limit={max_linear_velocity_error:.9g} rows={scored_count}",
    )
    _add(
        checks,
        scored_count > 0 and yaw_error <= max_yaw_velocity_error,
        "quality/yaw_velocity_error",
        f"mae={yaw_error:.9g} limit={max_yaw_velocity_error:.9g} rows={scored_count}",
    )
    _add(
        checks,
        scored_count > 0 and height_mae <= max_height_mae,
        "quality/height_mae",
        f"mae={height_mae:.9g} limit={max_height_mae:.9g} rows={scored_count}",
    )
    _add(
        checks,
        observed_max_tilt < max_tilt_deg,
        "quality/tilt_below_limit",
        f"max_tilt_deg={observed_max_tilt:.9g} task_limit_deg={max_tilt_deg:.9g}",
    )
    return {
        "verdict": "FAIL" if any(item["level"] == "FAIL" for item in checks) else "PASS",
        "metrics": {
            "scored_sample_count": scored_count,
            "linear_velocity_error": linear_error,
            "yaw_velocity_error": yaw_error,
            "height_mae": height_mae,
            "max_tilt_deg": observed_max_tilt,
            "termination_count": termination_count,
            "truncation_count": truncation_count,
            "command_identity_max_error": command_identity_error,
            "command_obs_max_error": command_obs_error,
            "target_height_obs_max_error": height_obs_error,
        },
        "checks": checks,
    }


def run_probe(
    *,
    identity: WalkRunIdentity,
    probe_name: str,
    command: tuple[float, float, float],
    expected_target_height: float,
    num_envs: int,
    warmup_steps: int,
    evaluation_steps: int,
    seed: int,
    device: str,
    max_linear_velocity_error: float,
    max_yaw_velocity_error: float,
    max_height_mae: float,
    create_env_fn: Callable[..., Any] = create_env,
    load_policy_fn: Callable[..., Callable[[np.ndarray], np.ndarray]] = load_policy,
    ensure_registries_fn: Callable[[], None] = ensure_registries,
) -> dict[str, Any]:
    cfg = OmegaConf.create(identity.config)
    cfg.env.commands.vel_limit = [list(command), list(command)]
    cfg.env.commands.rel_standing_envs = 0.0
    cfg.env.commands.rel_transition_envs = 0.0
    cfg.env.commands.heading_command = False
    cfg.env.commands.resampling_time = 0.0
    apply_training_seed(seed, torch_runtime=True, cuda=device.startswith("cuda"))
    ensure_registries_fn()
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name=EXPECTED_ALGO
    ).build_task_env_cfg_override()
    env = create_env_fn(
        cfg,
        num_envs=num_envs,
        env_cfg_override=env_override,
        sim_backend=EXPECTED_SIM_BACKEND,
    )
    samples = WalkRolloutSamples()
    requested_steps = warmup_steps + evaluation_steps
    executed_steps = 0
    try:
        if env.obs_groups_spec != {"obs": EXPECTED_OBS_DIM, "critic": EXPECTED_CRITIC_OBS_DIM}:
            raise ValueError(f"G1WalkHeight observation contract mismatch: {env.obs_groups_spec}")
        if env.action_space.shape != (EXPECTED_ACTION_DIM,):
            raise ValueError(f"G1WalkHeight action contract mismatch: {env.action_space.shape}")
        policy = load_policy_fn(
            identity.checkpoint_path,
            cfg,
            obs_dim=EXPECTED_OBS_DIM,
            action_dim=EXPECTED_ACTION_DIM,
            device=device,
        )
        env.set_autoreset(False)
        state = env.init_state()
        steps = state.info.get("steps")
        if not isinstance(steps, np.ndarray) or steps.shape != (num_envs,):
            raise ValueError("environment state.info['steps'] must have shape (num_envs,)")
        steps.fill(0)
        for step_index in range(requested_steps):
            actor_obs = np.asarray(state.obs["obs"], dtype=np.float32)
            actions = np.asarray(policy(actor_obs), dtype=np.float32)
            if actions.shape != (num_envs, EXPECTED_ACTION_DIM) or not np.all(np.isfinite(actions)):
                raise ValueError("policy actions must be finite with shape (num_envs, 29)")
            state = env.step(actions)
            executed_steps += 1
            row = _runtime_sample(env, state, actions)
            samples.append(**row, scored=step_index >= warmup_steps)
            if np.any(row["terminated"]) or np.any(row["truncated"]):
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    report = evaluate_samples(
        samples,
        expected_command=command,
        expected_target_height=expected_target_height,
        max_linear_velocity_error=max_linear_velocity_error,
        max_yaw_velocity_error=max_yaw_velocity_error,
        max_height_mae=max_height_mae,
        max_tilt_deg=identity.max_tilt_deg,
        requested_steps=requested_steps,
        executed_steps=executed_steps,
    )
    report.update({"probe_name": probe_name, "command": list(command)})
    return report


def run_acceptance(
    *,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    expected_sha256: str,
    expected_target_height: float,
    num_envs: int,
    warmup_steps: int,
    evaluation_steps: int,
    seed: int | None,
    device: str,
    max_linear_velocity_error: float = 0.25,
    max_yaw_velocity_error: float = 0.35,
    max_height_mae: float = 0.08,
    probes: tuple[tuple[str, tuple[float, float, float]], ...] = DEFAULT_PROBES,
    run_probe_fn: Callable[..., dict[str, Any]] = run_probe,
    **probe_dependencies: Any,
) -> dict[str, Any]:
    if num_envs <= 0 or warmup_steps < 0 or evaluation_steps <= 0:
        raise ValueError("num_envs/evaluation_steps must be positive and warmup_steps non-negative")
    for value, name in (
        (max_linear_velocity_error, "max_linear_velocity_error"),
        (max_yaw_velocity_error, "max_yaw_velocity_error"),
        (max_height_mae, "max_height_mae"),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    identity = load_run_identity(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        expected_sha256=expected_sha256,
        expected_target_height=expected_target_height,
    )
    effective_seed = identity.effective_seed if seed is None else int(seed)
    reports: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for name, command in probes:
        report = run_probe_fn(
            identity=identity,
            probe_name=name,
            command=command,
            expected_target_height=expected_target_height,
            num_envs=num_envs,
            warmup_steps=warmup_steps,
            evaluation_steps=evaluation_steps,
            seed=effective_seed,
            device=device,
            max_linear_velocity_error=max_linear_velocity_error,
            max_yaw_velocity_error=max_yaw_velocity_error,
            max_height_mae=max_height_mae,
            **probe_dependencies,
        )
        reports.append(report)
        checks.extend({**item, "name": f"probe/{name}/{item['name']}"} for item in report["checks"])
    return {
        "verdict": "PASS" if all(report["verdict"] == "PASS" for report in reports) else "FAIL",
        "identity": {
            "run_dir": str(identity.run_dir),
            "run_config_path": str(identity.run_config_path),
            "checkpoint_path": str(identity.checkpoint_path),
            "checkpoint_sha256": identity.checkpoint_sha256,
            "task_name": EXPECTED_TASK_NAME,
            "sim_backend": EXPECTED_SIM_BACKEND,
            "algo": EXPECTED_ALGO,
        },
        "contract": {
            "actor_obs_dim": EXPECTED_OBS_DIM,
            "critic_obs_dim": EXPECTED_CRITIC_OBS_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "command_obs_indices": [93, 96],
            "target_height_obs_index": TARGET_HEIGHT_OBS_INDEX,
            "probe_commands": {name: list(command) for name, command in probes},
        },
        "thresholds": {
            "max_linear_velocity_error": max_linear_velocity_error,
            "max_yaw_velocity_error": max_yaw_velocity_error,
            "max_height_mae": max_height_mae,
            "task_max_tilt_deg": identity.max_tilt_deg,
        },
        "probe_reports": reports,
        "checks": checks,
    }
