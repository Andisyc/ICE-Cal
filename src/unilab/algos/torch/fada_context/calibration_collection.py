"""Gain-only calibration rollout collection and raw artifact admission."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import (
    CALIBRATION_METHOD_CONTRACT_ID,
    CALIBRATION_TRAINING_CONTRACT_ID,
    CalibrationAxisSpec,
    FaultAxisCatalog,
)

GAIN_CALIBRATION_RAW_SCHEMA = "unilab_fada_gain_calibration_raw_rollouts_v2"
_LEGACY_GAIN_CALIBRATION_RAW_SCHEMA = "unilab_fada_gain_calibration_raw_rollouts_v1"
_LEGACY_METHOD_CONTRACT_ID = "FADA-CONTEXT-METHOD-v007"
_LEGACY_TRAINING_CONTRACT_ID = "FADA-CONTEXT-TRAIN-v006"
_LEGACY_AXIS_CATALOG_VERSION = "gain-delay-offset-v1"
_LEGACY_AXIS_NAMES = ("gain", "delay", "offset")
_APPROVED_POINTS = tuple(
    (round(-1.0 + 2.0 * index / 31.0, 9), round(0.8 + 0.4 * index / 31.0, 9))
    for index in range(32)
)
_APPROVED_SPLITS = (("train", 0, 101), ("validation", 1, 201))
_HEX_DIGITS = frozenset("0123456789abcdef")
_RESERVED_AXIS_METADATA_KEYS = frozenset(
    {
        "active_axes",
        "axis_catalog_version",
        "axis_count",
        "axis_names",
        "axis_spec",
        "catalog_version",
    }
)


@dataclass(frozen=True)
class GainCalibrationPoint:
    c_true: float
    gain: float

    def validate(self) -> GainCalibrationPoint:
        values = np.asarray((self.c_true, self.gain), dtype=np.float64)
        if not bool(np.isfinite(values).all()):
            raise ValueError("gain calibration point values must be finite")
        if self.gain <= 0.0:
            raise ValueError("gain calibration physical gain must be positive")
        return self


@dataclass(frozen=True)
class GainCalibrationSplit:
    name: str
    split_id: int
    seed: int

    def validate(self) -> GainCalibrationSplit:
        if not self.name or self.split_id < 0 or self.seed < 0:
            raise ValueError("gain calibration split name, id, and seed must be valid")
        return self


@dataclass(frozen=True)
class GainCalibrationScenarioSpec:
    point: GainCalibrationPoint
    split: GainCalibrationSplit
    fixed_command: tuple[float, ...]
    accepted_rows: int
    max_environment_steps: int
    observation_key: str = "obs"
    command_key: str = "commands"

    def validate(self) -> GainCalibrationScenarioSpec:
        self.point.validate()
        self.split.validate()
        command = np.asarray(self.fixed_command, dtype=np.float32)
        if command.ndim != 1 or command.size == 0 or not bool(np.isfinite(command).all()):
            raise ValueError("gain calibration fixed command must be a finite vector")
        if self.accepted_rows <= 0 or self.max_environment_steps <= 0:
            raise ValueError("gain calibration row quota and environment limit must be positive")
        if not self.observation_key or not self.command_key:
            raise ValueError("gain calibration observation and command keys are required")
        return self


@dataclass(frozen=True)
class GainCalibrationCollectionProtocol:
    version: str
    task_config: str
    task_name: str
    sim_backend: str
    observation_key: str
    command_key: str
    fixed_command: tuple[float, ...]
    points: tuple[GainCalibrationPoint, ...]
    splits: tuple[GainCalibrationSplit, ...]
    accepted_rows_per_scenario: int
    max_environment_steps_per_scenario: int

    def validate_approved(self) -> GainCalibrationCollectionProtocol:
        observed_points = tuple(
            (float(point.c_true), float(point.gain))
            for point in self.points
            if isinstance(point, GainCalibrationPoint)
        )
        if observed_points != _APPROVED_POINTS or len(observed_points) != len(self.points):
            raise ValueError(
                f"protocol does not match the exact approved gain grid {_APPROVED_POINTS}"
            )
        observed_splits = tuple(
            (split.name, int(split.split_id), int(split.seed))
            for split in self.splits
            if isinstance(split, GainCalibrationSplit)
        )
        if observed_splits != _APPROVED_SPLITS or len(observed_splits) != len(self.splits):
            raise ValueError("protocol does not match the approved train/validation splits")
        if (
            self.version != "gain-smoke-v2"
            or self.task_config != "g1_walk_flat/mujoco"
            or self.task_name != "G1WalkFlat"
            or self.sim_backend != "mujoco"
            or self.observation_key != "obs"
            or self.command_key != "commands"
            or tuple(float(value) for value in self.fixed_command) != (0.4, 0.0, 0.0)
            or int(self.accepted_rows_per_scenario) != 32
            or int(self.max_environment_steps_per_scenario) != 512
        ):
            raise ValueError("protocol does not match the approved gain smoke identity")
        return self


@dataclass(frozen=True)
class GainCalibrationRawIdentity:
    source_checkpoint_sha256: str
    source_checkpoint_path: str
    protocol_sha256: str
    resolved_task_backend_sha256: str
    axis_catalog_version: str

    def validate(self, axis_spec: CalibrationAxisSpec) -> GainCalibrationRawIdentity:
        for name in (
            "source_checkpoint_sha256",
            "protocol_sha256",
            "resolved_task_backend_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
                raise ValueError(f"raw rollout {name} must be a lowercase SHA256")
        if not self.source_checkpoint_path:
            raise ValueError("raw rollout source checkpoint path is required")
        if self.axis_catalog_version != axis_spec.catalog_version:
            raise ValueError("raw rollout axis catalog version mismatch")
        return self


@dataclass(frozen=True)
class GainCalibrationScenarioResult:
    rows: Mapping[str, Any]
    environment_steps: int
    rejected_transactions: int
    next_rollout_id: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_mapping(value: Mapping[str, Any]) -> str:
    normalized = _canonical_json_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("canonical provenance payload must be a mapping")
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_json_value(asdict(value))
    if OmegaConf.is_config(value):
        return _canonical_json_value(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return {str(key): _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"provenance payload is not JSON-safe: {type(value).__name__}")


def canonicalize_resolved_task_backend_payload(
    resolved_config: Any,
    base_env_override: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Materialize the exact JSON-safe task/backend identity and its digest."""

    normalized = _canonical_json_value(
        {
            "resolved_distill_config": resolved_config,
            "base_env_override": base_env_override,
        }
    )
    if not isinstance(normalized, dict):
        raise TypeError("resolved task/backend provenance must be a mapping")
    return normalized, sha256_canonical_mapping(normalized)


def load_gain_calibration_protocol(
    path: str | Path,
) -> tuple[GainCalibrationCollectionProtocol, bytes, str]:
    source = Path(path).expanduser().resolve()
    raw_bytes = source.read_bytes()
    protocol = _protocol_from_bytes(raw_bytes)
    return protocol, raw_bytes, hashlib.sha256(raw_bytes).hexdigest()


def _state_matrix(state: Any, carrier_name: str, key: str) -> np.ndarray:
    carrier = getattr(state, carrier_name, None)
    if not isinstance(carrier, Mapping) or key not in carrier:
        raise ValueError(f"state.{carrier_name}[{key!r}] is missing")
    value = np.asarray(carrier[key], dtype=np.float32)
    if value.ndim != 2 or not bool(np.isfinite(value).all()):
        raise ValueError(f"state.{carrier_name}[{key!r}] must be finite rank-2")
    return value


def _single_done(state: Any) -> bool:
    terminated = np.asarray(getattr(state, "terminated"), dtype=np.bool_)
    truncated = np.asarray(getattr(state, "truncated"), dtype=np.bool_)
    if terminated.shape != (1,) or truncated.shape != (1,):
        raise ValueError("calibration collection done flags must have one environment row")
    return bool(terminated[0] or truncated[0])


def _left_padded_history(
    observations: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    config: FADAArchitectureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    current = observations[-1]
    observation_history = list(observations[-config.history_length :])
    if len(observation_history) < config.history_length:
        observation_history = [current.copy()] * (
            config.history_length - len(observation_history)
        ) + observation_history
    action_history = list(actions[-config.history_length :])
    if len(action_history) < config.history_length:
        action_history = [np.zeros((config.action_dim,), dtype=np.float32)] * (
            config.history_length - len(action_history)
        ) + action_history
    return (
        np.asarray(observation_history, dtype=np.float32)[None],
        np.asarray(action_history, dtype=np.float32)[None],
    )


def _policy_query(
    policy: FADAPlannerIDMPolicy,
    observation_history: np.ndarray,
    action_history: np.ndarray,
    command: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        device = next(policy.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.inference_mode():
        output = policy(
            torch.as_tensor(observation_history, device=device),
            torch.as_tensor(action_history, device=device),
            torch.as_tensor(command, device=device),
        )
    intent = output.predicted_future.detach().cpu().numpy().astype(np.float32)
    chunk = output.action_chunk.detach().cpu().numpy().astype(np.float32)
    action = output.action.detach().cpu().numpy().astype(np.float32)
    expected = (
        (1, policy.config.prediction_horizon, policy.config.obs_dim),
        (1, policy.config.prediction_horizon, policy.config.action_dim),
        (1, policy.config.action_dim),
    )
    if (intent.shape, chunk.shape, action.shape) != expected:
        raise ValueError(
            "frozen Planner-IDM output shape mismatch: "
            f"observed={(intent.shape, chunk.shape, action.shape)} expected={expected}"
        )
    if not bool(
        np.isfinite(intent).all() and np.isfinite(chunk).all() and np.isfinite(action).all()
    ):
        raise ValueError("frozen Planner-IDM produced non-finite output")
    if not np.array_equal(action, chunk[:, 0]):
        raise ValueError("frozen Planner-IDM first action does not match action chunk index zero")
    return intent, chunk, action


def _stack_pending(pending: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tensor_names = tuple(name for name in pending[0] if name != "axis_name")
    return {
        **{name: torch.cat([row[name] for row in pending], dim=0) for name in tensor_names},
        "axis_name": [str(row["axis_name"]) for row in pending],
    }


def collect_gain_calibration_scenario(
    env: Any,
    policy: FADAPlannerIDMPolicy,
    spec: GainCalibrationScenarioSpec,
    *,
    rollout_id_start: int,
    axis_spec: CalibrationAxisSpec,
) -> GainCalibrationScenarioResult:
    """Collect one all-or-nothing episode transaction for a gain/split scenario."""

    spec.validate()
    if policy.training:
        raise ValueError("calibration collection requires an eval-mode frozen policy")
    if policy.config.history_length != 30 or policy.config.prediction_horizon != 6:
        raise ValueError("gain calibration collection requires H=30 and K=6")
    if "gain" not in axis_spec.names:
        raise ValueError("gain calibration collection requires an active gain axis")
    gain_axis_index = axis_spec.names.index("gain")
    if int(getattr(env, "num_envs", -1)) != 1:
        raise ValueError("gain calibration collection requires exactly one environment")
    set_autoreset = getattr(env, "set_autoreset", None)
    reset_all = getattr(env, "reset_all", None)
    if not callable(set_autoreset) or not callable(reset_all):
        raise TypeError("calibration environment must expose set_autoreset and reset_all")
    set_autoreset(False)
    fixed_command = np.asarray(spec.fixed_command, dtype=np.float32)[None]
    rollout_id = int(rollout_id_start)
    rejected = 0
    environment_steps = 0
    query_attempts = 0

    state = reset_all()
    observation = _state_matrix(state, "obs", spec.observation_key)
    command = _state_matrix(state, "info", spec.command_key)
    if observation.shape != (1, policy.config.obs_dim):
        raise ValueError("environment observation dimension does not match the checkpoint")
    if command.shape != fixed_command.shape or not np.array_equal(command, fixed_command):
        raise ValueError("environment reset command does not match the fixed smoke command")
    observations = [observation[0].copy()]
    actions: list[np.ndarray] = []
    pending: list[dict[str, Any]] = []

    while environment_steps < spec.max_environment_steps:
        query_attempts += 1
        if query_attempts > spec.max_environment_steps * 2:
            break
        observation_history, action_history = _left_padded_history(
            observations, actions, policy.config
        )
        try:
            intent, chunk, nominal = _policy_query(
                policy, observation_history, action_history, command
            )
        except ValueError:
            rejected += 1
            pending.clear()
            observations.clear()
            actions.clear()
            rollout_id += 1
            state = reset_all()
            observation = _state_matrix(state, "obs", spec.observation_key)
            command = _state_matrix(state, "info", spec.command_key)
            if command.shape != fixed_command.shape or not np.array_equal(command, fixed_command):
                raise ValueError("environment reset command does not match the fixed smoke command")
            observations.append(observation[0].copy())
            continue

        next_state = env.step(nominal)
        environment_steps += 1
        next_observation = _state_matrix(next_state, "obs", spec.observation_key)
        next_command = _state_matrix(next_state, "info", spec.command_key)
        current = _state_matrix(next_state, "info", "current_actions")
        authority = _state_matrix(next_state, "info", "authority_actions")
        executed = _state_matrix(next_state, "info", "executed_actions")
        if not np.array_equal(current, nominal):
            raise ValueError("environment current_actions no longer exposes the nominal action")
        expected_executed = authority * float(spec.point.gain)
        if not np.allclose(executed, expected_executed, rtol=1.0e-6, atol=1.0e-7):
            raise ValueError("environment executed_actions does not match authority action gain")
        valid_transaction = (
            not _single_done(next_state)
            and next_command.shape == fixed_command.shape
            and np.array_equal(next_command, fixed_command)
        )
        if valid_transaction and len(actions) >= policy.config.history_length:
            coefficients = torch.zeros((1, axis_spec.axis_count), dtype=torch.float32)
            coefficients[0, gain_axis_index] = spec.point.c_true
            pending.append(
                {
                    "observation_history": torch.from_numpy(observation_history.copy()),
                    "action_history": torch.from_numpy(action_history.copy()),
                    "command": torch.from_numpy(command.copy()),
                    "nominal_action_chunk": torch.from_numpy(chunk.copy()),
                    "c_true": coefficients,
                    "is_held_out_combination": torch.zeros((1,), dtype=torch.bool),
                    "injected_strength": torch.tensor([spec.point.gain], dtype=torch.float32),
                    "planner_intent": torch.from_numpy(intent.copy()),
                    "rollout_id": torch.tensor([rollout_id], dtype=torch.int64),
                    "seed": torch.tensor([spec.split.seed], dtype=torch.int64),
                    "split_id": torch.tensor([spec.split.split_id], dtype=torch.int64),
                    "executed_action": torch.from_numpy(executed.copy()),
                    "axis_name": "gain",
                }
            )
        if not valid_transaction:
            rejected += 1
            pending.clear()
            observations.clear()
            actions.clear()
            rollout_id += 1
            state = reset_all()
            observation = _state_matrix(state, "obs", spec.observation_key)
            command = _state_matrix(state, "info", spec.command_key)
            if command.shape != fixed_command.shape or not np.array_equal(command, fixed_command):
                raise ValueError("environment reset command does not match the fixed smoke command")
            observations.append(observation[0].copy())
            continue
        actions.append(nominal[0].copy())
        observations.append(next_observation[0].copy())
        command = next_command
        if len(pending) == spec.accepted_rows:
            return GainCalibrationScenarioResult(
                rows=_stack_pending(pending),
                environment_steps=environment_steps,
                rejected_transactions=rejected,
                next_rollout_id=rollout_id + 1,
            )
    raise RuntimeError(
        "gain calibration scenario exhausted its environment-step budget: "
        f"accepted={len(pending)} requested={spec.accepted_rows} "
        f"steps={environment_steps}"
    )


def _concat_row_trees(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tensor_names = tuple(name for name in rows[0] if name != "axis_name")
    return {
        **{name: torch.cat([row[name] for row in rows], dim=0) for name in tensor_names},
        "axis_name": [name for row in rows for name in row["axis_name"]],
    }


def build_gain_calibration_raw_artifact(
    rows: Mapping[str, Any],
    config: FADAArchitectureConfig,
    protocol: GainCalibrationCollectionProtocol,
    identity: GainCalibrationRawIdentity,
    axis_spec: CalibrationAxisSpec,
    *,
    protocol_bytes: bytes,
    resolved_task_backend_payload: Mapping[str, Any],
) -> dict[str, Any]:
    protocol.validate_approved()
    identity.validate(axis_spec)
    if _protocol_from_bytes(protocol_bytes) != protocol:
        raise ValueError("embedded protocol bytes do not match the approved protocol object")
    observed_protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    normalized_payload = _canonical_json_value(resolved_task_backend_payload)
    if not isinstance(normalized_payload, dict):
        raise TypeError("resolved task/backend provenance must be a mapping")
    observed_task_backend_sha256 = sha256_canonical_mapping(normalized_payload)
    if (
        observed_protocol_sha256 != identity.protocol_sha256
        or observed_task_backend_sha256 != identity.resolved_task_backend_sha256
    ):
        raise ValueError("gain calibration raw rollout provenance digest mismatch")
    _validate_resolved_task_backend_payload(normalized_payload, protocol)
    metadata = asdict(identity)
    metadata.pop("axis_catalog_version")
    return {
        "schema_version": GAIN_CALIBRATION_RAW_SCHEMA,
        "method_contract_id": CALIBRATION_METHOD_CONTRACT_ID,
        "training_contract_id": CALIBRATION_TRAINING_CONTRACT_ID,
        "architecture": asdict(config),
        "axis_spec": axis_spec.to_payload(),
        "protocol_bytes": protocol_bytes,
        "resolved_task_backend_payload": normalized_payload,
        "metadata": metadata,
        **rows,
    }


def collect_gain_calibration_rollouts(
    policy: FADAPlannerIDMPolicy,
    protocol: GainCalibrationCollectionProtocol,
    environment_factory: Callable[[GainCalibrationPoint, GainCalibrationSplit], Any],
    *,
    catalog: FaultAxisCatalog,
    identity: GainCalibrationRawIdentity,
    protocol_bytes: bytes,
    resolved_task_backend_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect the complete approved grid and close every factory-owned environment."""

    protocol.validate_approved()
    axis_spec = CalibrationAxisSpec.from_catalog(catalog)
    identity.validate(axis_spec)
    if policy.training:
        raise ValueError("calibration collection requires an eval-mode frozen policy")
    snapshot = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    scenario_rows: list[Mapping[str, Any]] = []
    next_rollout_id = 0
    for split in protocol.splits:
        for point in protocol.points:
            env = environment_factory(point, split)
            try:
                result = collect_gain_calibration_scenario(
                    env,
                    policy,
                    GainCalibrationScenarioSpec(
                        point=point,
                        split=split,
                        fixed_command=protocol.fixed_command,
                        accepted_rows=protocol.accepted_rows_per_scenario,
                        max_environment_steps=protocol.max_environment_steps_per_scenario,
                        observation_key=protocol.observation_key,
                        command_key=protocol.command_key,
                    ),
                    rollout_id_start=next_rollout_id,
                    axis_spec=axis_spec,
                )
                scenario_rows.append(result.rows)
                next_rollout_id = result.next_rollout_id
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
    current = policy.state_dict()
    if current.keys() != snapshot.keys() or any(
        not torch.equal(current[name].detach().cpu(), value) for name, value in snapshot.items()
    ):
        raise RuntimeError("frozen Planner-IDM parameters or buffers changed during collection")
    return build_gain_calibration_raw_artifact(
        _concat_row_trees(scenario_rows),
        policy.config,
        protocol,
        identity,
        axis_spec,
        protocol_bytes=protocol_bytes,
        resolved_task_backend_payload=resolved_task_backend_payload,
    )


def _protocol_from_payload(payload: Any) -> GainCalibrationCollectionProtocol:
    if not isinstance(payload, Mapping):
        raise ValueError("raw rollout protocol identity must be a mapping")
    try:
        protocol = GainCalibrationCollectionProtocol(
            version=str(payload["version"]),
            task_config=str(payload["task_config"]),
            task_name=str(payload["task_name"]),
            sim_backend=str(payload["sim_backend"]),
            observation_key=str(payload["observation_key"]),
            command_key=str(payload["command_key"]),
            fixed_command=tuple(float(value) for value in payload["fixed_command"]),
            points=tuple(GainCalibrationPoint(**point) for point in payload["points"]),
            splits=tuple(GainCalibrationSplit(**split) for split in payload["splits"]),
            accepted_rows_per_scenario=int(payload["accepted_rows_per_scenario"]),
            max_environment_steps_per_scenario=int(payload["max_environment_steps_per_scenario"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw rollout protocol identity is malformed") from exc
    return protocol.validate_approved()


def _protocol_from_bytes(raw_bytes: Any) -> GainCalibrationCollectionProtocol:
    if not isinstance(raw_bytes, bytes) or not raw_bytes:
        raise ValueError("raw rollout exact protocol bytes are missing")
    try:
        decoded = raw_bytes.decode("utf-8")
        payload = OmegaConf.to_container(OmegaConf.create(decoded), resolve=True)
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ValueError("raw rollout exact protocol bytes are malformed") from exc
    return _protocol_from_payload(payload)


def _validate_resolved_task_backend_payload(
    payload: Mapping[str, Any],
    protocol: GainCalibrationCollectionProtocol,
) -> None:
    resolved = payload.get("resolved_distill_config")
    base_override = payload.get("base_env_override")
    if not isinstance(resolved, Mapping) or not isinstance(base_override, Mapping):
        raise ValueError("resolved task/backend provenance material is incomplete")
    training = resolved.get("training")
    if not isinstance(training, Mapping) or (
        str(training.get("task_name")) != protocol.task_name
        or str(training.get("sim_backend")) != protocol.sim_backend
    ):
        raise ValueError("resolved task/backend provenance does not match the protocol")
    commands = base_override.get("commands")
    if not isinstance(commands, Mapping):
        raise ValueError("resolved task/backend provenance is missing fixed commands")
    expected_limits = [list(protocol.fixed_command), list(protocol.fixed_command)]
    if commands.get("vel_limit") != expected_limits:
        raise ValueError("resolved task/backend provenance fixed command mismatch")
    if "action_execution_fault" in base_override:
        raise ValueError("resolved task/backend provenance must precede per-point gain injection")


def _validate_gain_calibration_raw_artifact(
    artifact: Mapping[str, Any],
    *,
    catalog: FaultAxisCatalog,
    legacy: bool,
    expected_source_sha256: str | None = None,
    expected_architecture: FADAArchitectureConfig | None = None,
) -> Mapping[str, Any]:
    expected_schema = _LEGACY_GAIN_CALIBRATION_RAW_SCHEMA if legacy else GAIN_CALIBRATION_RAW_SCHEMA
    if artifact.get("schema_version") != expected_schema:
        raise ValueError("unsupported gain calibration raw rollout schema")
    expected_method = _LEGACY_METHOD_CONTRACT_ID if legacy else CALIBRATION_METHOD_CONTRACT_ID
    expected_training = _LEGACY_TRAINING_CONTRACT_ID if legacy else CALIBRATION_TRAINING_CONTRACT_ID
    if artifact.get("method_contract_id") != expected_method:
        raise ValueError("gain calibration raw rollout method Contract mismatch")
    if artifact.get("training_contract_id") != expected_training:
        raise ValueError("gain calibration raw rollout training Contract mismatch")
    try:
        config = FADAArchitectureConfig(**artifact["architecture"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("gain calibration raw rollout architecture is malformed") from exc
    if expected_architecture is not None and config != expected_architecture:
        raise ValueError("gain calibration raw rollout architecture identity mismatch")
    active_axis_spec = CalibrationAxisSpec.from_catalog(catalog)
    if legacy:
        if (
            catalog.version != _LEGACY_AXIS_CATALOG_VERSION
            or catalog.names != _LEGACY_AXIS_NAMES
            or artifact.get("axis_count") != len(_LEGACY_AXIS_NAMES)
            or tuple(artifact.get("axis_names", ())) != _LEGACY_AXIS_NAMES
        ):
            raise ValueError("legacy gain calibration raw rollout axis catalog mismatch")
    else:
        if artifact.get("axis_spec") != active_axis_spec.to_payload():
            raise ValueError("gain calibration raw rollout axis spec mismatch")
        duplicated_identity = sorted(
            (_RESERVED_AXIS_METADATA_KEYS - {"axis_spec"}).intersection(artifact)
        )
        if duplicated_identity:
            raise ValueError(
                f"active raw rollout contains duplicate axis identity: {duplicated_identity}"
            )
    raw_identity = artifact.get("metadata")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("gain calibration raw rollout metadata must be a mapping")
    try:
        identity_payload = dict(raw_identity)
        if not legacy:
            reserved = sorted(_RESERVED_AXIS_METADATA_KEYS.intersection(identity_payload))
            if reserved:
                raise ValueError("active raw rollout metadata contains reserved axis identity")
            identity_payload["axis_catalog_version"] = active_axis_spec.catalog_version
        identity = GainCalibrationRawIdentity(**identity_payload)
        if legacy:
            for name in (
                "source_checkpoint_sha256",
                "protocol_sha256",
                "resolved_task_backend_sha256",
            ):
                value = getattr(identity, name)
                if len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
                    raise ValueError(f"legacy raw rollout {name} must be a lowercase SHA256")
            if not identity.source_checkpoint_path:
                raise ValueError("legacy raw rollout source checkpoint path is required")
            if identity.axis_catalog_version != _LEGACY_AXIS_CATALOG_VERSION:
                raise ValueError("legacy raw rollout catalog version mismatch")
        else:
            identity.validate(active_axis_spec)
    except (TypeError, ValueError) as exc:
        raise ValueError("gain calibration raw rollout metadata identity is malformed") from exc
    protocol_bytes = artifact.get("protocol_bytes")
    if not isinstance(protocol_bytes, bytes):
        raise ValueError("raw rollout exact protocol bytes are missing")
    protocol = _protocol_from_bytes(protocol_bytes)
    resolved_payload = artifact.get("resolved_task_backend_payload")
    if not isinstance(resolved_payload, Mapping):
        raise ValueError("resolved task/backend provenance material is missing")
    normalized_payload = _canonical_json_value(resolved_payload)
    if not isinstance(normalized_payload, dict) or normalized_payload != resolved_payload:
        raise ValueError("resolved task/backend provenance material is not canonical")
    observed_protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    observed_task_backend_sha256 = sha256_canonical_mapping(normalized_payload)
    if (
        observed_protocol_sha256 != identity.protocol_sha256
        or observed_task_backend_sha256 != identity.resolved_task_backend_sha256
    ):
        raise ValueError("gain calibration raw rollout provenance digest mismatch")
    _validate_resolved_task_backend_payload(normalized_payload, protocol)
    if (
        expected_source_sha256 is not None
        and identity.source_checkpoint_sha256 != expected_source_sha256
    ):
        raise ValueError("gain calibration raw rollout source checkpoint SHA256 mismatch")
    tensor_names = (
        "observation_history",
        "action_history",
        "command",
        "nominal_action_chunk",
        "c_true",
        "is_held_out_combination",
        "injected_strength",
        "planner_intent",
        "rollout_id",
        "seed",
        "split_id",
        "executed_action",
    )
    missing = [name for name in tensor_names if not isinstance(artifact.get(name), torch.Tensor)]
    if missing:
        raise ValueError(f"gain calibration raw rollout is missing tensor fields: {missing}")
    batch = int(artifact["observation_history"].shape[0])
    expected_shapes = {
        "observation_history": (batch, config.history_length, config.obs_dim),
        "action_history": (batch, config.history_length, config.action_dim),
        "command": (batch, config.command_dim),
        "nominal_action_chunk": (batch, config.prediction_horizon, config.action_dim),
        "c_true": (batch, active_axis_spec.axis_count),
        "is_held_out_combination": (batch,),
        "injected_strength": (batch,),
        "planner_intent": (batch, config.prediction_horizon, config.obs_dim),
        "rollout_id": (batch,),
        "seed": (batch,),
        "split_id": (batch,),
        "executed_action": (batch, config.action_dim),
    }
    for name, shape in expected_shapes.items():
        tensor = artifact[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"gain calibration raw rollout {name} shape mismatch")
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"gain calibration raw rollout {name} must be finite")
    axis_name = artifact.get("axis_name")
    if not isinstance(axis_name, (list, tuple)) or axis_name != ["gain"] * batch:
        raise ValueError("gain calibration raw rollout must bind every row to gain")
    if "gain" not in active_axis_spec.names:
        raise ValueError("gain calibration raw rollout catalog lacks gain")
    gain_axis_index = active_axis_spec.names.index("gain")
    if artifact["is_held_out_combination"].dtype != torch.bool or bool(
        artifact["is_held_out_combination"].any()
    ):
        raise ValueError("gain-only smoke rows cannot be held-out combinations")
    omitted_axes = torch.ones(active_axis_spec.axis_count, dtype=torch.bool)
    omitted_axes[gain_axis_index] = False
    if not bool((artifact["c_true"][:, omitted_axes] == 0.0).all()):
        raise ValueError("gain-only smoke rows cannot label omitted axes")
    fixed = torch.tensor(protocol.fixed_command, dtype=artifact["command"].dtype)
    if not torch.equal(artifact["command"], fixed[None].expand(batch, -1)):
        raise ValueError("gain calibration raw rollout command identity mismatch")
    expected_total = (
        len(protocol.points) * len(protocol.splits) * protocol.accepted_rows_per_scenario
    )
    if batch != expected_total:
        raise ValueError(
            f"gain calibration raw rollout row count mismatch: expected={expected_total} got={batch}"
        )
    for split in protocol.splits:
        for point in protocol.points:
            mask = (
                (artifact["split_id"] == split.split_id)
                & (artifact["seed"] == split.seed)
                & (artifact["c_true"][:, gain_axis_index] == point.c_true)
                & (artifact["injected_strength"] == point.gain)
            )
            if int(mask.sum()) != protocol.accepted_rows_per_scenario:
                raise ValueError("gain calibration raw rollout scenario quota mismatch")
            if torch.unique(artifact["rollout_id"][mask]).numel() != 1:
                raise ValueError("gain calibration scenario crosses rollout identities")
    train_ids = set(artifact["rollout_id"][artifact["split_id"] == 0].tolist())
    validation_ids = set(artifact["rollout_id"][artifact["split_id"] == 1].tolist())
    if not train_ids.isdisjoint(validation_ids):
        raise ValueError("gain calibration train and validation rollout identities overlap")
    return artifact


def validate_gain_calibration_raw_artifact(
    artifact: Mapping[str, Any],
    *,
    catalog: FaultAxisCatalog,
    expected_source_sha256: str | None = None,
    expected_architecture: FADAArchitectureConfig | None = None,
) -> Mapping[str, Any]:
    """Validate only the active raw v2 envelope used by current writers."""

    return _validate_gain_calibration_raw_artifact(
        artifact,
        catalog=catalog,
        legacy=False,
        expected_source_sha256=expected_source_sha256,
        expected_architecture=expected_architecture,
    )


def _load_legacy_gain_calibration_raw_gateway(
    artifact: Mapping[str, Any],
    *,
    catalog: FaultAxisCatalog,
    expected_source_sha256: str | None,
    expected_architecture: FADAArchitectureConfig | None,
) -> Mapping[str, Any]:
    """Read the exact historical v1 donor envelope for one-time dataset resealing."""

    return _validate_gain_calibration_raw_artifact(
        artifact,
        catalog=catalog,
        legacy=True,
        expected_source_sha256=expected_source_sha256,
        expected_architecture=expected_architecture,
    )


def save_gain_calibration_raw_rollouts(
    path: str | Path,
    artifact: Mapping[str, Any],
    *,
    catalog: FaultAxisCatalog,
) -> Path:
    validate_gain_calibration_raw_artifact(artifact, catalog=catalog)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(dict(artifact), temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_gain_calibration_raw_rollouts(
    path: str | Path,
    *,
    catalog: FaultAxisCatalog,
    expected_source_sha256: str | None = None,
    expected_architecture: FADAArchitectureConfig | None = None,
) -> Mapping[str, Any]:
    serialized = Path(path).expanduser().resolve().read_bytes()
    payload = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("gain calibration raw rollout artifact must be a mapping")
    if payload.get("schema_version") == _LEGACY_GAIN_CALIBRATION_RAW_SCHEMA:
        return _load_legacy_gain_calibration_raw_gateway(
            payload,
            catalog=catalog,
            expected_source_sha256=expected_source_sha256,
            expected_architecture=expected_architecture,
        )
    return validate_gain_calibration_raw_artifact(
        payload,
        catalog=catalog,
        expected_source_sha256=expected_source_sha256,
        expected_architecture=expected_architecture,
    )
