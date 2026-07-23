"""Explicit actor-only checkpoint adapters for off-policy policy migrations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

G1_HEIGHT_ACTOR_ADAPTER_ID = "g1_height_actor_obs_98_to_99_v1"
G1_HEIGHT_SOURCE_OBS_DIM = 98
G1_HEIGHT_TARGET_OBS_DIM = 99
G1_HEIGHT_INSERTION_INDEX = 96
G1_HEIGHT_DEFAULT_TARGET = 0.754


@dataclass(frozen=True)
class G1HeightActorCheckpointResult:
    """Paths and hashes produced by an immutable adapter materialization."""

    parent_checkpoint_path: Path
    parent_checkpoint_sha256: str
    output_checkpoint_path: Path
    output_checkpoint_sha256: str
    metadata_path: Path


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _first_rank2_weight(
    state: Mapping[str, torch.Tensor],
) -> tuple[str, torch.Tensor]:
    for key, value in state.items():
        if key.endswith("weight") and isinstance(value, torch.Tensor) and value.ndim == 2:
            return str(key), value
    raise ValueError("source actor state has no rank-2 weight")


def _clone_tensor_state(state: Mapping[str, torch.Tensor], *, name: str) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} state entry {key!r} is not a tensor")
        cloned[str(key)] = value.detach().clone()
    return cloned


def adapt_g1_height_actor_state(
    actor_state: Mapping[str, torch.Tensor],
    *,
    insertion_index: int = G1_HEIGHT_INSERTION_INDEX,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Insert the G1 target-height input while preserving the legacy actor function."""

    first_weight_key, first_weight = _first_rank2_weight(actor_state)
    source_dim = int(first_weight.shape[1])
    if source_dim != G1_HEIGHT_SOURCE_OBS_DIM:
        raise ValueError(
            "G1 height adapter source actor input dim must be "
            f"{G1_HEIGHT_SOURCE_OBS_DIM}, got {source_dim} ({first_weight_key})"
        )
    if not 0 <= int(insertion_index) <= source_dim:
        raise ValueError(
            f"G1 height adapter insertion index must be in [0, {source_dim}], "
            f"got {insertion_index}"
        )

    adapted = _clone_tensor_state(actor_state, name="actor")
    expanded = first_weight.new_zeros((int(first_weight.shape[0]), source_dim + 1))
    expanded[:, :insertion_index] = first_weight[:, :insertion_index]
    expanded[:, insertion_index + 1 :] = first_weight[:, insertion_index:]
    adapted[first_weight_key] = expanded

    metadata: dict[str, Any] = {
        "adapter_id": G1_HEIGHT_ACTOR_ADAPTER_ID,
        "migration_scope": "actor_and_optional_obs_normalizer_only",
        "source_obs_dim": G1_HEIGHT_SOURCE_OBS_DIM,
        "target_obs_dim": G1_HEIGHT_TARGET_OBS_DIM,
        "insertion_index": int(insertion_index),
        "first_weight_key": first_weight_key,
        "inserted_normalizer_mean": G1_HEIGHT_DEFAULT_TARGET,
        "inserted_normalizer_variance": 1.0,
        "inserted_normalizer_std": 1.0,
    }
    return adapted, metadata


def adapt_g1_height_normalizer_state(
    normalizer_state: Mapping[str, torch.Tensor],
    *,
    insertion_index: int = G1_HEIGHT_INSERTION_INDEX,
) -> dict[str, torch.Tensor]:
    """Expand EmpiricalNormalization state for the inserted target-height input."""

    adapted = _clone_tensor_state(normalizer_state, name="obs_normalizer")
    inserted_values = {
        "_mean": G1_HEIGHT_DEFAULT_TARGET,
        "_var": 1.0,
        "_std": 1.0,
    }
    for key, inserted_value in inserted_values.items():
        if key not in normalizer_state:
            raise ValueError(f"obs_normalizer state is missing {key}")
        source = normalizer_state[key]
        if tuple(source.shape) != (1, G1_HEIGHT_SOURCE_OBS_DIM):
            raise ValueError(
                f"obs_normalizer {key} must have shape "
                f"(1, {G1_HEIGHT_SOURCE_OBS_DIM}), got {tuple(source.shape)}"
            )
        expanded = source.new_empty((1, G1_HEIGHT_TARGET_OBS_DIM))
        expanded[:, :insertion_index] = source[:, :insertion_index]
        expanded[:, insertion_index] = inserted_value
        expanded[:, insertion_index + 1 :] = source[:, insertion_index:]
        adapted[key] = expanded
    if "count" not in normalizer_state:
        raise ValueError("obs_normalizer state is missing count")
    return adapted


def _adapt_checkpoint_payload(
    checkpoint: Mapping[str, Any],
    *,
    parent_checkpoint_path: Path,
    parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    actor_state = checkpoint.get("actor")
    if not isinstance(actor_state, Mapping):
        raise ValueError("source SAC checkpoint does not contain actor state")
    adapted_actor, metadata = adapt_g1_height_actor_state(actor_state)
    metadata.update(
        {
            "parent_checkpoint_path": str(parent_checkpoint_path.resolve()),
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
        }
    )
    payload: dict[str, Any] = {
        "actor": adapted_actor,
        "actor_obs_adapter": metadata,
    }
    normalizer_state = checkpoint.get("obs_normalizer")
    if normalizer_state is not None:
        if not isinstance(normalizer_state, Mapping):
            raise ValueError("source obs_normalizer state must be a mapping")
        payload["obs_normalizer"] = adapt_g1_height_normalizer_state(normalizer_state)
    return payload


def _atomic_torch_save(payload: Mapping[str, Any], output_path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def materialize_g1_height_actor_checkpoint(
    parent_checkpoint_path: str | Path,
    output_checkpoint_path: str | Path,
    *,
    overwrite: bool = False,
) -> G1HeightActorCheckpointResult:
    """Write an actor-only 99-D checkpoint and a hash-bearing JSON sidecar."""

    parent = Path(parent_checkpoint_path).resolve()
    output = Path(output_checkpoint_path).resolve()
    metadata_path = output.with_suffix(f"{output.suffix}.migration.json")
    if not parent.is_file():
        raise FileNotFoundError(f"parent checkpoint does not exist: {parent}")
    if parent == output:
        raise ValueError("output checkpoint must not overwrite the parent checkpoint")
    if not overwrite and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"adapter output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    parent_sha256 = _file_sha256(parent)
    checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("source SAC checkpoint payload must be a mapping")
    payload = _adapt_checkpoint_payload(
        checkpoint,
        parent_checkpoint_path=parent,
        parent_checkpoint_sha256=parent_sha256,
    )
    _atomic_torch_save(payload, output)
    output_sha256 = _file_sha256(output)
    sidecar = {
        **payload["actor_obs_adapter"],
        "output_checkpoint_path": str(output),
        "output_checkpoint_sha256": output_sha256,
    }
    try:
        _atomic_write_json(sidecar, metadata_path)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return G1HeightActorCheckpointResult(
        parent_checkpoint_path=parent,
        parent_checkpoint_sha256=parent_sha256,
        output_checkpoint_path=output,
        output_checkpoint_sha256=output_sha256,
        metadata_path=metadata_path,
    )


def _validate_state_dict_compatibility(
    target: Mapping[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> None:
    target_keys = set(target)
    source_keys = set(source)
    if target_keys != source_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise ValueError(
            f"{name} state keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    mismatches = {
        key: (tuple(source[key].shape), tuple(target[key].shape))
        for key in sorted(target_keys)
        if tuple(source[key].shape) != tuple(target[key].shape)
    }
    if mismatches:
        raise ValueError(f"{name} state shape mismatch: {mismatches}")


def load_g1_height_actor_warm_start(
    learner: Any,
    parent_checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load only an adapted actor into a fresh 99-D learner."""

    parent = Path(parent_checkpoint_path).resolve()
    if not parent.is_file():
        raise FileNotFoundError(f"parent checkpoint does not exist: {parent}")
    checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("source SAC checkpoint payload must be a mapping")
    payload = _adapt_checkpoint_payload(
        checkpoint,
        parent_checkpoint_path=parent,
        parent_checkpoint_sha256=_file_sha256(parent),
    )

    actor = getattr(learner, "actor", None)
    if not isinstance(actor, nn.Module):
        raise ValueError("actor warm start requires learner.actor to be an nn.Module")
    target_actor_state = actor.state_dict()
    adapted_actor_state = payload["actor"]
    _validate_state_dict_compatibility(
        target_actor_state,
        adapted_actor_state,
        name="actor warm start",
    )
    _, target_first_weight = _first_rank2_weight(target_actor_state)
    if int(target_first_weight.shape[1]) != G1_HEIGHT_TARGET_OBS_DIM:
        raise ValueError(
            "actor warm start target input dim must be "
            f"{G1_HEIGHT_TARGET_OBS_DIM}, got {int(target_first_weight.shape[1])}"
        )

    adapted_normalizer_state = payload.get("obs_normalizer")
    target_normalizer = getattr(learner, "obs_normalizer", None)
    if adapted_normalizer_state is not None:
        if not isinstance(target_normalizer, nn.Module) or isinstance(target_normalizer, nn.Identity):
            raise ValueError(
                "source checkpoint contains obs_normalizer but target learner has no active "
                "obs_normalizer"
            )
        _validate_state_dict_compatibility(
            target_normalizer.state_dict(),
            adapted_normalizer_state,
            name="obs_normalizer warm start",
        )

    actor.load_state_dict(adapted_actor_state, strict=True)
    if adapted_normalizer_state is not None:
        target_normalizer.load_state_dict(adapted_normalizer_state, strict=True)
    metadata = dict(payload["actor_obs_adapter"])
    learner.actor_warm_start_metadata = metadata
    return metadata
