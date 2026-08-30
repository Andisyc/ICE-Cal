"""Distillation workflow value types and deterministic contract helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from unilab.algos.torch.distill.observability.performance import DistillationStageObservation
from unilab.algos.torch.distill.runtime.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
)

ROLE_ARTIFACT_MANIFEST_VERSION = 1

def _progress(message: str) -> None:
    if os.environ.get("UNILAB_DISTILL_PROGRESS", "0").lower() not in {
        "",
        "0",
        "false",
        "off",
    }:
        print(f"[distill-workflow] {message}", flush=True)


class WorkflowScenarioCollector(Protocol):
    """Activate one student barrier, then collect scenarios against it."""

    def activate_checkpoint(self, checkpoint_path: Path) -> int: ...

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult: ...


@dataclass(frozen=True)
class WorkflowScenarioCollectionResult:
    """Return legacy scenario rows plus owner-local performance observations."""

    num_samples: int
    worker_pid: int
    performance_metrics_schema_version: int
    performance_stage_observations: tuple[DistillationStageObservation, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_samples, bool)
            or not isinstance(self.num_samples, int)
            or self.num_samples <= 0
        ):
            raise ValueError("num_samples must be a positive integer")
        if (
            isinstance(self.worker_pid, bool)
            or not isinstance(self.worker_pid, int)
            or self.worker_pid <= 0
        ):
            raise ValueError("worker_pid must be a positive integer")


@dataclass(frozen=True)
class WorkflowStudentUpdateResult:
    """Return update count plus learner-owner performance observations."""

    updates: int
    performance_stage_observations: tuple[DistillationStageObservation, ...]

    def __post_init__(self) -> None:
        if isinstance(self.updates, bool) or not isinstance(self.updates, int) or self.updates <= 0:
            raise ValueError("updates must be a positive integer")


class ArtifactDecision(str, Enum):
    REUSE = "REUSE"
    COLLECT = "COLLECT"
    STALE = "STALE"
    INCOMPATIBLE = "INCOMPATIBLE"


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _normalize_json(config),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RoleArtifactSpec:
    role: str
    task: str
    teacher_checkpoint_path: Path
    dataset_path: Path
    schema_version: int
    student_obs_dim: int
    teacher_obs_dim: int
    teacher_action_dim: int
    teacher_obs_key: str
    teacher_projection: str
    student_projection: str
    student_drop_index: int | None
    command_sample_filter: str
    command_info_key: str
    command_xy_threshold: float
    command_yaw_threshold: float
    owner_config: Mapping[str, Any]
    target_height_info_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "teacher_checkpoint_path", Path(self.teacher_checkpoint_path))
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        if not self.role:
            raise ValueError("role must be non-empty")
        if not self.task:
            raise ValueError("task must be non-empty")

    @property
    def manifest_path(self) -> Path:
        return self.dataset_path.with_suffix(self.dataset_path.suffix + ".manifest.json")

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "task": self.task,
            "teacher_checkpoint_path": self.teacher_checkpoint_path,
            "dataset_path": self.dataset_path,
            "schema_version": self.schema_version,
            "student_obs_dim": self.student_obs_dim,
            "teacher_obs_dim": self.teacher_obs_dim,
            "teacher_action_dim": self.teacher_action_dim,
            "teacher_obs_key": self.teacher_obs_key,
            "teacher_projection": self.teacher_projection,
            "student_projection": self.student_projection,
            "student_drop_index": self.student_drop_index,
            "command_sample_filter": self.command_sample_filter,
            "command_info_key": self.command_info_key,
            "command_xy_threshold": self.command_xy_threshold,
            "command_yaw_threshold": self.command_yaw_threshold,
            "target_height_info_key": self.target_height_info_key,
            "owner_config": self.owner_config,
        }


@dataclass(frozen=True)
class RoleArtifactManifest:
    manifest_version: int
    role: str
    task: str
    teacher_checkpoint_path: str
    teacher_checkpoint_sha256: str
    dataset_path: str
    dataset_sha256: str
    schema_version: int
    student_obs_dim: int
    teacher_obs_dim: int
    teacher_action_dim: int
    teacher_obs_key: str
    teacher_projection: str
    student_projection: str
    student_drop_index: int | None
    command_sample_filter: str
    command_info_key: str
    command_xy_threshold: float
    command_yaw_threshold: float
    owner_config_sha256: str
    num_samples: int
    target_height_info_key: str | None = None


@dataclass(frozen=True)
class RoleArtifactPreflight:
    role: str
    decision: ArtifactDecision
    dataset_path: Path
    manifest_path: Path
    mismatches: tuple[str, ...] = ()


@dataclass(frozen=True)
class BootstrapWorkflowResult:
    run_dir: Path
    manifest_path: Path
    role_decisions: dict[str, str]
    bootstrap_dataset_path: Path
    bootstrap_num_samples: int
    checkpoint_path: Path
    bootstrap_updates: int


@dataclass(frozen=True)
class WorkflowDatasetSource:
    path: Path
    role: str
    scenario: str | None = None
    preserve_row_role_labels: bool = False


@dataclass(frozen=True)
class WorkflowScenarioSpec:
    name: str
    kind: str
    source_roles: tuple[str, ...]
    quota: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "source_roles", tuple(str(role) for role in self.source_roles))
        object.__setattr__(self, "quota", float(self.quota))
        if not self.name:
            raise ValueError("workflow scenario name must be non-empty")
        if self.kind not in {"role", "transition"}:
            raise ValueError(
                f"workflow scenario kind must be 'role' or 'transition', got {self.kind!r}"
            )
        if not self.source_roles:
            raise ValueError(f"workflow scenario {self.name!r} requires source_roles")
        if self.kind == "role" and len(self.source_roles) != 1:
            raise ValueError("role workflow scenarios require exactly one source role")
        if not math.isfinite(self.quota) or self.quota <= 0.0:
            raise ValueError("workflow scenario quota must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "source_roles": list(self.source_roles),
            "quota": self.quota,
        }


@dataclass(frozen=True)
class WalkToStopRolePair:
    walking_role: str
    standing_role: str
    target_height_info_key: str | None


def resolve_walk_to_stop_role_pair(
    *,
    source_roles: Sequence[str],
    command_sample_filters: Mapping[str, str],
    target_height_info_keys: Mapping[str, str | None],
) -> WalkToStopRolePair:
    """Resolve the sole semantic role pair for a walk-to-stop collection."""

    roles = tuple(str(role) for role in source_roles)
    if len(roles) != 2 or len(set(roles)) != 2:
        raise ValueError(f"walk_to_stop scenario requires exactly two source roles, got {roles}")

    missing_filters = tuple(role for role in roles if role not in command_sample_filters)
    missing_height_keys = tuple(role for role in roles if role not in target_height_info_keys)
    if missing_filters or missing_height_keys:
        missing = tuple(dict.fromkeys((*missing_filters, *missing_height_keys)))
        raise ValueError(f"walk_to_stop scenario references unknown roles: {missing}")

    filters_by_role = {role: str(command_sample_filters[role]) for role in roles}
    role_by_filter: dict[str, str] = {}
    for role, command_filter in filters_by_role.items():
        if command_filter not in {"active", "inactive"}:
            raise ValueError(
                "walk_to_stop scenario has unsupported command sample filter: "
                f"role={role!r} filter={command_filter!r}"
            )
        if command_filter in role_by_filter:
            raise ValueError(
                "walk_to_stop scenario requires one active role and one inactive role, "
                f"got {filters_by_role}"
            )
        role_by_filter[command_filter] = role

    walking_role = role_by_filter["active"]
    standing_role = role_by_filter["inactive"]
    walking_height_key = target_height_info_keys[walking_role]
    standing_height_key = target_height_info_keys[standing_role]
    if walking_height_key != standing_height_key:
        raise ValueError(
            "walk_to_stop roles must agree on target-height info key: "
            f"walk={walking_height_key!r} stand={standing_height_key!r}"
        )
    return WalkToStopRolePair(
        walking_role=walking_role,
        standing_role=standing_role,
        target_height_info_key=walking_height_key,
    )


@dataclass(frozen=True)
class DaggerWorkflowResult:
    run_dir: Path
    manifest_path: Path
    completed_iterations: int
    checkpoint_path: Path
    cumulative_num_samples: int
