from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from unilab.algos.torch.fada_context.gain_collection_types import (
    GainCalibrationCollectionProtocol,
    GainCalibrationPoint,
    GainCalibrationSplit,
)


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
