"""DAgger distillation 的结构化性能指标契约与持久化 owner.

Status: active schema and persistent run-artifact owner.
Upstream: persistent G1 worker, role/transition collectors, parent workflow connector.
Downstream: run-local JSON artifact and HP-4 comparison tooling.
Evidence: S1/S2/S3 schema and owner tests; E61/E65/E67 S4 MuJoCo timing and
legacy/persistent A/B runtime-confirmed.
Gap: end-to-end stable speedup is not proven (``NO_STABLE_SPEEDUP``); HP-6
production gate and physical policy acceptance remain pending.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DISTILLATION_METRICS_SCHEMA_VERSION = 1

DISTILLATION_STAGE_NAMES = frozenset(
    {
        "cold_start",
        "weight_sync",
        "env_init",
        "env_reset",
        "student_inference",
        "teacher_inference",
        "env_step",
        "tensor_pack",
        "artifact_write",
        "cumulative_aggregation",
        "learner_batch_staging",
        "learner_forward",
        "learner_backward",
        "optimizer_step",
        "checkpoint_save",
        "cleanup",
        "total_elapsed",
    }
)

COLLECTOR_REQUEST_STAGE_NAMES = (
    "teacher_inference",
    "student_inference",
    "env_step",
    "tensor_pack",
)

PERSISTENT_REQUEST_STAGE_NAMES = (
    "weight_sync",
    *COLLECTOR_REQUEST_STAGE_NAMES,
    "artifact_write",
    "total_elapsed",
)

LEGACY_REQUEST_STAGE_NAMES = (
    "cold_start",
    *COLLECTOR_REQUEST_STAGE_NAMES,
    "artifact_write",
    "total_elapsed",
)

WORKFLOW_ITERATION_STAGE_NAMES = (
    "cumulative_aggregation",
    "learner_batch_staging",
    "learner_forward",
    "learner_backward",
    "optimizer_step",
    "checkpoint_save",
)

CLEANUP_STAGE_NAMES = ("cleanup",)

_EXECUTION_MODES = frozenset({"legacy", "persistent_async"})
_CLEANUP_STATES = frozenset({"not_applicable", "pending", "complete", "failed"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_nonempty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_sha256(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _require_nonnegative_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class DistillationPerformanceRunContext:
    """Own immutable run identity before request-local identity enrichment."""

    execution_mode: str
    teacher_checkpoint_sha256: tuple[str, ...]
    config_sha256: str
    seed: int
    device: str
    num_envs: int

    def __post_init__(self) -> None:
        if self.execution_mode not in _EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {sorted(_EXECUTION_MODES)}, "
                f"got {self.execution_mode!r}"
            )
        if not self.teacher_checkpoint_sha256:
            raise ValueError("teacher_checkpoint_sha256 must be non-empty")
        for teacher_hash in self.teacher_checkpoint_sha256:
            _require_sha256(teacher_hash, field_name="teacher_checkpoint_sha256")
        if len(set(self.teacher_checkpoint_sha256)) != len(self.teacher_checkpoint_sha256):
            raise ValueError("teacher_checkpoint_sha256 must not contain duplicates")
        object.__setattr__(
            self,
            "teacher_checkpoint_sha256",
            tuple(sorted(self.teacher_checkpoint_sha256)),
        )
        _require_sha256(self.config_sha256, field_name="config_sha256")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        _require_nonempty(self.device, field_name="device")
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int):
            raise ValueError("num_envs must be a positive integer")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")

    def run_signature(self) -> tuple[Any, ...]:
        """Return the exact fields shared by every request in one artifact."""

        return (
            self.execution_mode,
            self.teacher_checkpoint_sha256,
            self.config_sha256,
            self.seed,
            self.device,
            self.num_envs,
        )

    def enrich_request(
        self,
        *,
        outer_iteration: int,
        scenario: str,
        worker_pid: int,
        request_id: str,
        checkpoint_path: str,
        checkpoint_sha256: str,
        weight_version: int | None,
        schema_version: int,
        observations: Sequence[DistillationStageObservation],
    ) -> tuple[DistillationStageMetric, ...]:
        """Attach parent-owned run/request identity to one exact worker sequence."""

        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != DISTILLATION_METRICS_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported distillation request metrics schema_version: {schema_version!r}"
            )
        expected_stages = (
            LEGACY_REQUEST_STAGE_NAMES
            if self.execution_mode == "legacy"
            else PERSISTENT_REQUEST_STAGE_NAMES
        )
        observed_stages = tuple(observation.stage for observation in observations)
        if observed_stages != expected_stages:
            raise ValueError(
                f"{self.execution_mode} request performance stage order mismatch: "
                f"expected={expected_stages} observed={observed_stages}"
            )
        identity = DistillationMetricIdentity(
            execution_mode=self.execution_mode,
            outer_iteration=outer_iteration,
            scenario=scenario,
            worker_pid=worker_pid,
            request_id=request_id,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            weight_version=weight_version,
            teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            config_sha256=self.config_sha256,
            seed=self.seed,
            device=self.device,
            num_envs=self.num_envs,
        )
        return tuple(
            DistillationStageMetric.from_observation(identity, observation)
            for observation in observations
        )

    def enrich_workflow_iteration(
        self,
        *,
        outer_iteration: int,
        worker_pid: int,
        checkpoint_path: str,
        checkpoint_sha256: str,
        weight_version: int | None,
        observations: Sequence[DistillationStageObservation],
    ) -> tuple[DistillationStageMetric, ...]:
        """Attach parent identity to exact aggregation and learner owner stages."""

        observed_stages = tuple(observation.stage for observation in observations)
        if observed_stages != WORKFLOW_ITERATION_STAGE_NAMES:
            raise ValueError(
                "workflow iteration performance stage order mismatch: "
                f"expected={WORKFLOW_ITERATION_STAGE_NAMES} observed={observed_stages}"
            )
        identity = DistillationMetricIdentity(
            execution_mode=self.execution_mode,
            outer_iteration=outer_iteration,
            scenario="__workflow__",
            worker_pid=worker_pid,
            request_id=f"dagger-{outer_iteration}-workflow",
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            weight_version=weight_version,
            teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            config_sha256=self.config_sha256,
            seed=self.seed,
            device=self.device,
            num_envs=self.num_envs,
        )
        return tuple(
            DistillationStageMetric.from_observation(identity, observation)
            for observation in observations
        )

    def enrich_cleanup(
        self,
        *,
        outer_iteration: int,
        worker_pid: int,
        checkpoint_path: str,
        checkpoint_sha256: str,
        weight_version: int | None,
        observation: DistillationStageObservation,
    ) -> DistillationStageMetric:
        """Attach final lifecycle identity after the collector owner has closed."""

        if observation.stage != "cleanup":
            raise ValueError(
                "cleanup performance stage mismatch: "
                f"expected='cleanup' observed={observation.stage!r}"
            )
        identity = DistillationMetricIdentity(
            execution_mode=self.execution_mode,
            outer_iteration=outer_iteration,
            scenario="__cleanup__",
            worker_pid=worker_pid,
            request_id=f"dagger-{outer_iteration}-cleanup",
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            weight_version=weight_version,
            teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            config_sha256=self.config_sha256,
            seed=self.seed,
            device=self.device,
            num_envs=self.num_envs,
        )
        return DistillationStageMetric.from_observation(identity, observation)


@dataclass(frozen=True)
class DistillationMetricIdentity:
    """Identify one measured stage without inferring identity from filenames."""

    execution_mode: str
    outer_iteration: int
    scenario: str
    worker_pid: int
    request_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    weight_version: int | None
    teacher_checkpoint_sha256: tuple[str, ...]
    config_sha256: str
    seed: int
    device: str
    num_envs: int

    def __post_init__(self) -> None:
        if self.execution_mode not in _EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {sorted(_EXECUTION_MODES)}, "
                f"got {self.execution_mode!r}"
            )
        _require_nonnegative_int(self.outer_iteration, field_name="outer_iteration")
        _require_nonempty(self.scenario, field_name="scenario")
        if isinstance(self.worker_pid, bool) or not isinstance(self.worker_pid, int):
            raise ValueError("worker_pid must be a positive integer")
        if self.worker_pid <= 0:
            raise ValueError("worker_pid must be a positive integer")
        _require_nonempty(self.request_id, field_name="request_id")
        if not isinstance(self.checkpoint_path, str):
            raise ValueError("checkpoint_path must be an absolute canonical path")
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.is_absolute():
            raise ValueError("checkpoint_path must be absolute")
        if str(checkpoint_path.resolve()) != self.checkpoint_path:
            raise ValueError("checkpoint_path must be an absolute canonical path")
        _require_sha256(self.checkpoint_sha256, field_name="checkpoint_sha256")
        if self.execution_mode == "legacy":
            if self.weight_version is not None:
                raise ValueError("legacy weight_version must be None")
        elif self.weight_version is None:
            raise ValueError("persistent_async weight_version must be non-negative")
        else:
            _require_nonnegative_int(self.weight_version, field_name="weight_version")
        if not self.teacher_checkpoint_sha256:
            raise ValueError("teacher_checkpoint_sha256 must be non-empty")
        for teacher_hash in self.teacher_checkpoint_sha256:
            _require_sha256(teacher_hash, field_name="teacher_checkpoint_sha256")
        if len(set(self.teacher_checkpoint_sha256)) != len(self.teacher_checkpoint_sha256):
            raise ValueError("teacher_checkpoint_sha256 must not contain duplicates")
        _require_sha256(self.config_sha256, field_name="config_sha256")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        _require_nonempty(self.device, field_name="device")
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int):
            raise ValueError("num_envs must be a positive integer")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")

    def run_signature(self) -> tuple[Any, ...]:
        """Return fields that must remain constant inside one run artifact."""

        return (
            self.execution_mode,
            self.teacher_checkpoint_sha256,
            self.config_sha256,
            self.seed,
            self.device,
            self.num_envs,
        )

    def record_key(self, stage: str) -> tuple[Any, ...]:
        """Return the idempotency key for one stage observation."""

        return (
            self.outer_iteration,
            self.scenario,
            self.worker_pid,
            self.request_id,
            self.checkpoint_sha256,
            self.weight_version,
            stage,
        )

    def request_key(self) -> tuple[Any, ...]:
        return (self.outer_iteration, self.scenario, self.request_id)

    def request_signature(self) -> tuple[Any, ...]:
        return (
            self.worker_pid,
            self.checkpoint_path,
            self.checkpoint_sha256,
            self.weight_version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "outer_iteration": self.outer_iteration,
            "scenario": self.scenario,
            "worker_pid": self.worker_pid,
            "request_id": self.request_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "weight_version": self.weight_version,
            "teacher_checkpoint_sha256": list(self.teacher_checkpoint_sha256),
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "device": self.device,
            "num_envs": self.num_envs,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DistillationMetricIdentity:
        return cls(
            execution_mode=payload["execution_mode"],
            outer_iteration=payload["outer_iteration"],
            scenario=payload["scenario"],
            worker_pid=payload["worker_pid"],
            request_id=payload["request_id"],
            checkpoint_path=payload["checkpoint_path"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            weight_version=payload.get("weight_version"),
            teacher_checkpoint_sha256=tuple(payload["teacher_checkpoint_sha256"]),
            config_sha256=payload["config_sha256"],
            seed=payload["seed"],
            device=payload["device"],
            num_envs=payload["num_envs"],
        )


@dataclass(frozen=True)
class DistillationStageObservation:
    """Store one owner-local stage fact before parent identity enrichment."""

    stage: str
    duration_seconds: float
    row_count: int
    env_step_count: int
    success: bool
    error: str | None
    cleanup_state: str

    def __post_init__(self) -> None:
        if self.stage not in DISTILLATION_STAGE_NAMES:
            raise ValueError(
                f"unknown distillation metric stage {self.stage!r}; "
                f"expected one of {sorted(DISTILLATION_STAGE_NAMES)}"
            )
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, int | float)
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be finite and non-negative")
        _require_nonnegative_int(self.row_count, field_name="row_count")
        _require_nonnegative_int(self.env_step_count, field_name="env_step_count")
        if self.cleanup_state not in _CLEANUP_STATES:
            raise ValueError(
                f"cleanup_state must be one of {sorted(_CLEANUP_STATES)}, "
                f"got {self.cleanup_state!r}"
            )
        if self.success and self.error is not None:
            raise ValueError("successful metric record must not contain error")
        if not self.success and not self.error:
            raise ValueError("failed metric record must contain error")
        if not self.success and self.cleanup_state != "failed":
            raise ValueError("failed metric record must use cleanup_state='failed'")

    @property
    def rows_per_second(self) -> float | None:
        if self.row_count == 0 or self.duration_seconds == 0:
            return None
        return self.row_count / self.duration_seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "duration_seconds": self.duration_seconds,
            "row_count": self.row_count,
            "env_step_count": self.env_step_count,
            "rows_per_second": self.rows_per_second,
            "success": self.success,
            "error": self.error,
            "cleanup_state": self.cleanup_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DistillationStageObservation:
        success = payload["success"]
        if not isinstance(success, bool):
            raise ValueError("success must be a boolean")
        observation = cls(
            stage=payload["stage"],
            duration_seconds=payload["duration_seconds"],
            row_count=payload["row_count"],
            env_step_count=payload["env_step_count"],
            success=success,
            error=payload.get("error"),
            cleanup_state=payload["cleanup_state"],
        )
        persisted_rate = payload.get("rows_per_second")
        expected_rate = observation.rows_per_second
        if persisted_rate is None:
            if expected_rate is not None:
                raise ValueError("rows_per_second is missing or inconsistent")
        elif (
            isinstance(persisted_rate, bool)
            or not isinstance(persisted_rate, int | float)
            or expected_rate is None
            or not math.isclose(persisted_rate, expected_rate, rel_tol=1e-12, abs_tol=0.0)
        ):
            raise ValueError("rows_per_second is inconsistent with rows and duration")
        return observation


class DistillationStageObservationAccumulator:
    """Accumulate owner-local stage durations before identity enrichment."""

    def __init__(self, *, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._durations: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        if stage not in DISTILLATION_STAGE_NAMES:
            raise ValueError(f"unknown distillation metric stage {stage!r}")
        start = float(self._clock())
        try:
            yield
        finally:
            duration = float(self._clock()) - start
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(f"stage {stage!r} clock duration must be finite and non-negative")
            self._durations[stage] = self._durations.get(stage, 0.0) + duration

    def observation(
        self,
        *,
        stage: str,
        row_count: int,
        env_step_count: int,
        cleanup_state: str = "not_applicable",
    ) -> DistillationStageObservation:
        return DistillationStageObservation(
            stage=stage,
            duration_seconds=self._durations.get(stage, 0.0),
            row_count=row_count,
            env_step_count=env_step_count,
            success=True,
            error=None,
            cleanup_state=cleanup_state,
        )


@dataclass(frozen=True)
class DistillationStageMetric:
    """Attach immutable parent identity to one validated stage observation."""

    identity: DistillationMetricIdentity
    stage: str
    duration_seconds: float
    row_count: int
    env_step_count: int
    success: bool
    error: str | None
    cleanup_state: str

    def __post_init__(self) -> None:
        self.as_observation()

    @property
    def rows_per_second(self) -> float | None:
        return self.as_observation().rows_per_second

    def as_observation(self) -> DistillationStageObservation:
        return DistillationStageObservation(
            stage=self.stage,
            duration_seconds=self.duration_seconds,
            row_count=self.row_count,
            env_step_count=self.env_step_count,
            success=self.success,
            error=self.error,
            cleanup_state=self.cleanup_state,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.as_dict(),
            **self.as_observation().as_dict(),
        }

    @classmethod
    def from_observation(
        cls,
        identity: DistillationMetricIdentity,
        observation: DistillationStageObservation,
    ) -> DistillationStageMetric:
        return cls(
            identity=identity,
            stage=observation.stage,
            duration_seconds=observation.duration_seconds,
            row_count=observation.row_count,
            env_step_count=observation.env_step_count,
            success=observation.success,
            error=observation.error,
            cleanup_state=observation.cleanup_state,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DistillationStageMetric:
        return cls.from_observation(
            DistillationMetricIdentity.from_dict(payload["identity"]),
            DistillationStageObservation.from_dict(payload),
        )


@dataclass(frozen=True)
class DistillationMetricsArtifact:
    """Validated run-local metric records loaded from one JSON artifact."""

    schema_version: int
    records: tuple[DistillationStageMetric, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != DISTILLATION_METRICS_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported distillation metrics schema_version: {self.schema_version}"
            )
        if not self.records:
            raise ValueError("distillation metrics artifact must contain records")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "duration_unit": "seconds",
            "records": [record.as_dict() for record in self.records],
        }


class DistillationMetricsRecorder:
    """Validate, de-duplicate, and atomically persist stage metric records."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._records_by_key: dict[tuple[Any, ...], DistillationStageMetric] = {}
        self._run_signature: tuple[Any, ...] | None = None
        self._request_signatures: dict[tuple[Any, ...], tuple[Any, ...]] = {}

    @property
    def records(self) -> tuple[DistillationStageMetric, ...]:
        return tuple(self._records_by_key.values())

    def add(self, record: DistillationStageMetric) -> None:
        signature = record.identity.run_signature()
        if self._run_signature is None:
            self._run_signature = signature
        elif signature != self._run_signature:
            raise ValueError("distillation metric identity drift within one artifact")
        request_key = record.identity.request_key()
        request_signature = record.identity.request_signature()
        existing_request_signature = self._request_signatures.get(request_key)
        if existing_request_signature is None:
            self._request_signatures[request_key] = request_signature
        elif existing_request_signature != request_signature:
            raise ValueError("distillation metric request identity drift")
        key = record.identity.record_key(record.stage)
        existing = self._records_by_key.get(key)
        if existing is not None:
            if existing != record:
                raise ValueError("duplicate distillation metric record has incompatible content")
            return
        self._records_by_key[key] = record

    @contextmanager
    def measure(
        self,
        *,
        identity: DistillationMetricIdentity,
        stage: str,
        row_count: int,
        env_step_count: int,
        cleanup_state: str,
    ) -> Iterator[None]:
        """Measure one block with an injected monotonic clock and record failures."""

        start = float(self._clock())
        try:
            yield
        except Exception as exc:
            duration = float(self._clock()) - start
            self.add(
                DistillationStageMetric(
                    identity=identity,
                    stage=stage,
                    duration_seconds=duration,
                    row_count=row_count,
                    env_step_count=env_step_count,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    cleanup_state="failed",
                )
            )
            raise
        else:
            duration = float(self._clock()) - start
            self.add(
                DistillationStageMetric(
                    identity=identity,
                    stage=stage,
                    duration_seconds=duration,
                    row_count=row_count,
                    env_step_count=env_step_count,
                    success=True,
                    error=None,
                    cleanup_state=cleanup_state,
                )
            )

    def validate_required_stages(self, required_stages: Sequence[str]) -> None:
        expected = {str(stage) for stage in required_stages}
        unknown = expected - DISTILLATION_STAGE_NAMES
        if unknown:
            raise ValueError(f"unknown required distillation stages: {sorted(unknown)}")
        observed = {record.stage for record in self.records}
        missing = expected - observed
        if missing:
            raise ValueError(f"missing required stages: {sorted(missing)}")

    def write(
        self,
        path: str | Path,
        *,
        required_stages: Sequence[str] = (),
    ) -> Path:
        self.validate_required_stages(required_stages)
        artifact = DistillationMetricsArtifact(
            schema_version=DISTILLATION_METRICS_SCHEMA_VERSION,
            records=self.records,
        )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(artifact.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
        return output_path


def load_distillation_metrics(path: str | Path) -> DistillationMetricsArtifact:
    """Load and revalidate one run-local distillation metrics artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("duration_unit") != "seconds":
        raise ValueError("distillation metrics duration_unit must be 'seconds'")
    records = tuple(DistillationStageMetric.from_dict(record) for record in payload["records"])
    artifact = DistillationMetricsArtifact(
        schema_version=payload["schema_version"],
        records=records,
    )
    recorder = DistillationMetricsRecorder()
    for record in artifact.records:
        recorder.add(record)
    return artifact
