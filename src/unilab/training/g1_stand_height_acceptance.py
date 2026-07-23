"""G1StandHeight checkpoint identity and bounded physical acceptance owner.

Status: live acceptance probe. This owner reads one checkpoint and its
``run_config.json`` sidecar, then runs deterministic policy inference in the
official MuJoCo environment. It never trains, resumes, adapts, or rewrites the
checkpoint.
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
from unilab.envs.locomotion.g1.joystick import (
    LEFT_FOOT_CONTACT_SENSORS,
    RIGHT_FOOT_CONTACT_SENSORS,
    compute_aggregated_foot_contact,
)
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.common import create_env
from unilab.training.seed import apply_training_seed

ROOT_DIR = Path(__file__).resolve().parents[3]

EXPECTED_TASK_NAME = "G1StandHeight"
EXPECTED_SIM_BACKEND = "mujoco"
EXPECTED_ALGO = "sac"
EXPECTED_OBS_DIM = 99
EXPECTED_CRITIC_OBS_DIM = 102
EXPECTED_ACTION_DIM = 29
TARGET_OBS_INDEX = 96


@dataclass(frozen=True)
class RunIdentity:
    run_dir: Path
    run_config_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    task_name: str
    sim_backend: str
    algo: str
    target_height: float
    max_tilt_deg: float
    effective_seed: int
    config: dict[str, Any] = field(repr=False)


@dataclass
class RolloutSamples:
    """Per-env rollout rows before acceptance aggregation."""

    target_height: list[np.ndarray] = field(default_factory=list)
    measured_height: list[np.ndarray] = field(default_factory=list)
    double_support: list[np.ndarray] = field(default_factory=list)
    tilt_deg: list[np.ndarray] = field(default_factory=list)
    terminated: list[np.ndarray] = field(default_factory=list)
    truncated: list[np.ndarray] = field(default_factory=list)
    commands: list[np.ndarray] = field(default_factory=list)
    target_obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    scored: list[np.ndarray] = field(default_factory=list)

    def append(
        self,
        *,
        target_height: np.ndarray,
        measured_height: np.ndarray,
        double_support: np.ndarray,
        tilt_deg: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        commands: np.ndarray,
        target_obs: np.ndarray,
        actions: np.ndarray,
        scored: bool,
    ) -> None:
        target = _vector(target_height, name="target_height")
        rows = int(target.shape[0])
        vectors = {
            "measured_height": _vector(measured_height, name="measured_height"),
            "double_support": _vector(double_support, name="double_support", dtype=bool),
            "tilt_deg": _vector(tilt_deg, name="tilt_deg"),
            "terminated": _vector(terminated, name="terminated", dtype=bool),
            "truncated": _vector(truncated, name="truncated", dtype=bool),
            "target_obs": _vector(target_obs, name="target_obs"),
        }
        for name, value in vectors.items():
            if int(value.shape[0]) != rows:
                raise ValueError(f"{name} rows={value.shape[0]} do not match target rows={rows}")

        command_rows = np.asarray(commands, dtype=np.float32)
        action_rows = np.asarray(actions, dtype=np.float32)
        if command_rows.shape != (rows, 3):
            raise ValueError(f"commands must have shape ({rows}, 3), got {command_rows.shape}")
        if action_rows.shape != (rows, EXPECTED_ACTION_DIM):
            raise ValueError(
                f"actions must have shape ({rows}, {EXPECTED_ACTION_DIM}), got {action_rows.shape}"
            )

        self.target_height.append(target.copy())
        self.measured_height.append(vectors["measured_height"].copy())
        self.double_support.append(vectors["double_support"].copy())
        self.tilt_deg.append(vectors["tilt_deg"].copy())
        self.terminated.append(vectors["terminated"].copy())
        self.truncated.append(vectors["truncated"].copy())
        self.commands.append(command_rows.copy())
        self.target_obs.append(vectors["target_obs"].copy())
        self.actions.append(action_rows.copy())
        self.scored.append(np.full(rows, bool(scored), dtype=bool))

    def concat(self, name: str) -> np.ndarray:
        chunks = cast(list[np.ndarray], getattr(self, name))
        if not chunks:
            if name == "commands":
                return np.empty((0, 3), dtype=np.float32)
            if name == "actions":
                return np.empty((0, EXPECTED_ACTION_DIM), dtype=np.float32)
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(chunks, axis=0)


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


def _require_fixed_target(config: Mapping[str, Any], *, expected_target_height: float) -> None:
    env = _require_mapping(config.get("env"), name="config.env")
    commands = _require_mapping(env.get("commands"), name="config.env.commands")
    height_range = commands.get("height_range")
    if not isinstance(height_range, (list, tuple)) or len(height_range) != 2:
        raise ValueError("config.env.commands.height_range must contain two values")
    observed = np.asarray(height_range, dtype=np.float64)
    expected = np.full(2, float(expected_target_height), dtype=np.float64)
    if not np.allclose(observed, expected, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "Stage-1 acceptance requires a fixed target range: "
            f"expected={expected.tolist()} observed={observed.tolist()}"
        )
    default_height = commands.get("default_height")
    if default_height is None or not np.isclose(
        float(default_height), float(expected_target_height), rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(
            "Stage-1 default height mismatch: "
            f"expected={expected_target_height} observed={default_height}"
        )


def load_run_identity(
    *,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    expected_sha256: str,
    expected_target_height: float,
) -> RunIdentity:
    """Validate immutable artifact and training-config identity before rollout."""

    resolved_run_dir = Path(run_dir).resolve()
    resolved_checkpoint = Path(checkpoint_path).resolve()
    run_config_path = resolved_run_dir / "run_config.json"
    if not resolved_run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {resolved_run_dir}")
    if not run_config_path.is_file():
        raise FileNotFoundError(f"run_config.json does not exist: {run_config_path}")
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved_checkpoint}")
    if resolved_checkpoint.parent != resolved_run_dir:
        raise ValueError(
            "checkpoint must be directly inside the accepted run directory: "
            f"run_dir={resolved_run_dir} checkpoint={resolved_checkpoint}"
        )

    normalized_expected_hash = str(expected_sha256).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_expected_hash) is None:
        raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters")
    observed_hash = _file_sha256(resolved_checkpoint)
    if observed_hash != normalized_expected_hash:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected={normalized_expected_hash} observed={observed_hash}"
        )

    payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    root = _require_mapping(payload, name="root")
    config = dict(_require_mapping(root.get("config"), name="config"))
    training = _require_mapping(config.get("training"), name="config.training")
    algo_cfg = _require_mapping(config.get("algo"), name="config.algo")
    reward = _require_mapping(config.get("reward"), name="config.reward")
    task_name = str(training.get("task_name"))
    sim_backend = str(training.get("sim_backend"))
    algo = str(algo_cfg.get("algo"))
    if task_name != EXPECTED_TASK_NAME:
        raise ValueError(f"expected task {EXPECTED_TASK_NAME}, got {task_name}")
    if sim_backend != EXPECTED_SIM_BACKEND:
        raise ValueError(f"expected backend {EXPECTED_SIM_BACKEND}, got {sim_backend}")
    if algo != EXPECTED_ALGO:
        raise ValueError(f"expected algorithm {EXPECTED_ALGO}, got {algo}")
    _require_fixed_target(config, expected_target_height=expected_target_height)

    max_tilt_deg = float(reward.get("max_tilt_deg", float("nan")))
    if not np.isfinite(max_tilt_deg) or max_tilt_deg <= 0.0:
        raise ValueError(
            f"config.reward.max_tilt_deg must be positive and finite, got {max_tilt_deg}"
        )
    run = _require_mapping(root.get("run", {}), name="run")
    seed_value = run.get("effective_seed", algo_cfg.get("seed", 1))

    return RunIdentity(
        run_dir=resolved_run_dir,
        run_config_path=run_config_path,
        checkpoint_path=resolved_checkpoint,
        checkpoint_sha256=observed_hash,
        task_name=task_name,
        sim_backend=sim_backend,
        algo=algo,
        target_height=float(expected_target_height),
        max_tilt_deg=max_tilt_deg,
        effective_seed=int(seed_value),
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
    """Load the exact SAC actor path used by distillation teacher collection."""

    import torch

    spec = DistillationTeacherSpec(
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        actor_hidden_dim=int(cfg.algo.actor_hidden_dim),
        use_layer_norm=bool(cfg.algo.use_layer_norm),
        obs_normalization=bool(cfg.algo.obs_normalization),
    )
    teacher = load_sac_teacher_policy(checkpoint_path, spec, device=device)

    def _policy(obs: np.ndarray) -> np.ndarray:
        actor_obs = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
        with torch.inference_mode():
            actions = teacher(actor_obs)
        return actions.detach().cpu().numpy().astype(np.float32, copy=False)

    return _policy


def _runtime_sample(env: Any, state: Any, actions: np.ndarray) -> dict[str, np.ndarray]:
    obs = np.asarray(state.obs["obs"], dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != EXPECTED_OBS_DIM:
        raise ValueError(
            f"actor observation must have shape (N, {EXPECTED_OBS_DIM}), got {obs.shape}"
        )
    info = state.info
    commands = np.asarray(info.get("commands"), dtype=np.float32)
    target_height = _vector(info.get("height_commands"), name="height_commands")
    measured_height = _vector(
        env._terrain_relative_base_height(), name="terrain_relative_base_height"
    )
    upvector = np.asarray(
        env._backend.get_sensor_data(env.cfg.sensor.upvector), dtype=np.float32
    ).reshape(int(env.num_envs), -1)
    if upvector.shape[1] < 3:
        raise ValueError(f"upvector sensor must expose at least 3 values, got {upvector.shape}")
    tilt_deg = np.rad2deg(np.arccos(np.clip(upvector[:, 2], -1.0, 1.0))).astype(np.float32)
    left_contact = compute_aggregated_foot_contact(env._backend, LEFT_FOOT_CONTACT_SENSORS)
    right_contact = compute_aggregated_foot_contact(env._backend, RIGHT_FOOT_CONTACT_SENSORS)
    return {
        "target_height": target_height,
        "measured_height": measured_height,
        "double_support": np.asarray(left_contact & right_contact, dtype=bool),
        "tilt_deg": tilt_deg,
        "terminated": np.asarray(state.terminated, dtype=bool),
        "truncated": np.asarray(state.truncated, dtype=bool),
        "commands": commands,
        "target_obs": obs[:, TARGET_OBS_INDEX],
        "actions": np.asarray(actions, dtype=np.float32),
    }


def _add_check(checks: list[dict[str, Any]], passed: bool, name: str, detail: str) -> None:
    checks.append({"level": "PASS" if passed else "FAIL", "name": name, "detail": detail})


def evaluate_samples(
    samples: RolloutSamples,
    *,
    expected_target_height: float,
    max_height_mae: float,
    min_double_support_fraction: float,
    max_tilt_deg: float,
    requested_steps: int,
    executed_steps: int,
) -> dict[str, Any]:
    """Aggregate rollout rows against the Stage-1 teacher gate."""

    target = samples.concat("target_height")
    measured = samples.concat("measured_height")
    support = samples.concat("double_support").astype(bool, copy=False)
    tilt = samples.concat("tilt_deg")
    terminated = samples.concat("terminated").astype(bool, copy=False)
    truncated = samples.concat("truncated").astype(bool, copy=False)
    commands = samples.concat("commands")
    target_obs = samples.concat("target_obs")
    actions = samples.concat("actions")
    scored = samples.concat("scored").astype(bool, copy=False)
    scored_count = int(np.count_nonzero(scored))

    finite = bool(
        target.size > 0
        and np.all(np.isfinite(target))
        and np.all(np.isfinite(measured))
        and np.all(np.isfinite(tilt))
        and np.all(np.isfinite(commands))
        and np.all(np.isfinite(target_obs))
        and np.all(np.isfinite(actions))
    )
    if scored_count:
        height_mae = float(np.mean(np.abs(measured[scored] - target[scored])))
        double_support_fraction = float(np.mean(support[scored]))
        target_min = float(np.min(target[scored]))
        target_max = float(np.max(target[scored]))
    else:
        height_mae = float("inf")
        double_support_fraction = 0.0
        target_min = float("nan")
        target_max = float("nan")
    max_observed_tilt = float(np.max(tilt)) if tilt.size else float("inf")
    termination_count = int(np.count_nonzero(terminated))
    truncation_count = int(np.count_nonzero(truncated))
    commands_max_abs = float(np.max(np.abs(commands))) if commands.size else float("inf")
    target_obs_max_error = (
        float(np.max(np.abs(target_obs - target))) if target.size else float("inf")
    )
    target_identity_max_error = (
        float(np.max(np.abs(target - float(expected_target_height))))
        if target.size
        else float("inf")
    )

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        int(executed_steps) == int(requested_steps),
        "rollout/completed_window",
        f"executed_steps={executed_steps} requested_steps={requested_steps}",
    )
    _add_check(checks, finite, "rollout/finite", f"finite={finite}")
    _add_check(
        checks,
        termination_count == 0 and truncation_count == 0,
        "rollout/no_termination",
        f"terminated_rows={termination_count} truncated_rows={truncation_count}",
    )
    _add_check(
        checks,
        commands_max_abs <= 1.0e-6,
        "rollout/zero_velocity_command",
        f"commands_max_abs={commands_max_abs:.9g}",
    )
    _add_check(
        checks,
        target_identity_max_error <= 1.0e-6,
        "rollout/fixed_target_identity",
        f"target_max_error={target_identity_max_error:.9g}",
    )
    _add_check(
        checks,
        target_obs_max_error <= 1.0e-6,
        "rollout/target_obs_roundtrip",
        f"obs_index={TARGET_OBS_INDEX} max_error={target_obs_max_error:.9g}",
    )
    _add_check(
        checks,
        scored_count > 0 and height_mae <= float(max_height_mae),
        "quality/height_mae",
        f"mae={height_mae:.9g} limit={max_height_mae:.9g} rows={scored_count}",
    )
    _add_check(
        checks,
        scored_count > 0 and double_support_fraction >= float(min_double_support_fraction),
        "quality/double_support_fraction",
        (
            f"fraction={double_support_fraction:.9g} "
            f"minimum={min_double_support_fraction:.9g} rows={scored_count}"
        ),
    )
    _add_check(
        checks,
        max_observed_tilt < float(max_tilt_deg),
        "quality/tilt_below_limit",
        f"max_tilt_deg={max_observed_tilt:.9g} task_limit_deg={max_tilt_deg:.9g}",
    )

    return {
        "verdict": "FAIL" if any(check["level"] == "FAIL" for check in checks) else "PASS",
        "metrics": {
            "all_sample_count": int(target.shape[0]),
            "scored_sample_count": scored_count,
            "target_height_min": target_min,
            "target_height_max": target_max,
            "height_mae": height_mae,
            "double_support_fraction": double_support_fraction,
            "max_tilt_deg": max_observed_tilt,
            "termination_count": termination_count,
            "truncation_count": truncation_count,
            "commands_max_abs": commands_max_abs,
            "target_obs_max_error": target_obs_max_error,
            "target_identity_max_error": target_identity_max_error,
        },
        "checks": checks,
    }


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
    max_height_mae: float = 0.05,
    min_double_support_fraction: float = 0.90,
    create_env_fn: Callable[..., Any] = create_env,
    load_policy_fn: Callable[..., Callable[[np.ndarray], np.ndarray]] = load_policy,
    ensure_registries_fn: Callable[[], None] = ensure_registries,
) -> dict[str, Any]:
    """Run one immutable checkpoint through the bounded Stage-1 gate."""

    if int(num_envs) <= 0:
        raise ValueError("num_envs must be positive")
    if int(warmup_steps) < 0:
        raise ValueError("warmup_steps must be non-negative")
    if int(evaluation_steps) <= 0:
        raise ValueError("evaluation_steps must be positive")
    if float(max_height_mae) <= 0.0:
        raise ValueError("max_height_mae must be positive")
    if not 0.0 <= float(min_double_support_fraction) <= 1.0:
        raise ValueError("min_double_support_fraction must be in [0, 1]")

    identity = load_run_identity(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        expected_sha256=expected_sha256,
        expected_target_height=expected_target_height,
    )
    effective_seed = identity.effective_seed if seed is None else int(seed)
    apply_training_seed(effective_seed, torch_runtime=True, cuda=device.startswith("cuda"))
    cfg = OmegaConf.create(identity.config)
    ensure_registries_fn()
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name=EXPECTED_ALGO
    ).build_task_env_cfg_override()
    env = create_env_fn(
        cfg,
        num_envs=int(num_envs),
        env_cfg_override=env_override,
        sim_backend=EXPECTED_SIM_BACKEND,
    )
    samples = RolloutSamples()
    requested_steps = int(warmup_steps) + int(evaluation_steps)
    executed_steps = 0
    try:
        if env.obs_groups_spec != {
            "obs": EXPECTED_OBS_DIM,
            "critic": EXPECTED_CRITIC_OBS_DIM,
        }:
            raise ValueError(
                "G1StandHeight observation contract mismatch: "
                f"expected={{'obs': {EXPECTED_OBS_DIM}, 'critic': {EXPECTED_CRITIC_OBS_DIM}}} "
                f"observed={env.obs_groups_spec}"
            )
        action_shape = env.action_space.shape
        if action_shape != (EXPECTED_ACTION_DIM,):
            raise ValueError(
                f"G1StandHeight action contract mismatch: expected={(EXPECTED_ACTION_DIM,)} "
                f"observed={action_shape}"
            )
        policy = load_policy_fn(
            identity.checkpoint_path,
            cfg,
            obs_dim=EXPECTED_OBS_DIM,
            action_dim=EXPECTED_ACTION_DIM,
            device=device,
        )
        env.set_autoreset(False)
        state = env.init_state()
        steps_array = state.info.get("steps")
        if not isinstance(steps_array, np.ndarray) or steps_array.shape != (int(num_envs),):
            raise ValueError("environment state.info['steps'] must have shape (num_envs,)")
        steps_array.fill(0)

        for step_index in range(requested_steps):
            actor_obs = np.asarray(state.obs["obs"], dtype=np.float32)
            actions = np.asarray(policy(actor_obs), dtype=np.float32)
            if actions.shape != (int(num_envs), EXPECTED_ACTION_DIM):
                raise ValueError(
                    "policy action shape mismatch: "
                    f"expected={(int(num_envs), EXPECTED_ACTION_DIM)} observed={actions.shape}"
                )
            if not np.all(np.isfinite(actions)):
                raise ValueError("policy produced non-finite actions; refusing to step MuJoCo")
            state = env.step(actions)
            executed_steps += 1
            row = _runtime_sample(env, state, actions)
            samples.append(**row, scored=step_index >= int(warmup_steps))
            if np.any(row["terminated"]) or np.any(row["truncated"]):
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    report = evaluate_samples(
        samples,
        expected_target_height=float(expected_target_height),
        max_height_mae=float(max_height_mae),
        min_double_support_fraction=float(min_double_support_fraction),
        max_tilt_deg=identity.max_tilt_deg,
        requested_steps=requested_steps,
        executed_steps=executed_steps,
    )
    report.update(
        {
            "identity": {
                "run_dir": str(identity.run_dir),
                "run_config_path": str(identity.run_config_path),
                "checkpoint_path": str(identity.checkpoint_path),
                "checkpoint_sha256": identity.checkpoint_sha256,
                "task_name": identity.task_name,
                "sim_backend": identity.sim_backend,
                "algo": identity.algo,
            },
            "contract": {
                "actor_obs_dim": EXPECTED_OBS_DIM,
                "critic_obs_dim": EXPECTED_CRITIC_OBS_DIM,
                "action_dim": EXPECTED_ACTION_DIM,
                "target_obs_index": TARGET_OBS_INDEX,
            },
            "rollout": {
                "num_envs": int(num_envs),
                "warmup_steps": int(warmup_steps),
                "evaluation_steps": int(evaluation_steps),
                "requested_steps": requested_steps,
                "executed_steps": executed_steps,
                "seed": effective_seed,
                "device": device,
                "autoreset": False,
            },
            "thresholds": {
                "expected_target_height": float(expected_target_height),
                "max_height_mae": float(max_height_mae),
                "min_double_support_fraction": float(min_double_support_fraction),
                "task_max_tilt_deg": identity.max_tilt_deg,
                "termination_count": 0,
                "truncation_count": 0,
            },
        }
    )
    return report
