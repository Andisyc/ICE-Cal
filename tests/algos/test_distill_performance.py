from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from unilab.algos.torch.distill.data import make_fake_distillation_dataset
from unilab.algos.torch.distill.models import MLPStudentPolicy
from unilab.algos.torch.distill.offline import run_offline_distillation_updates
from unilab.algos.torch.distill.performance import (
    DISTILLATION_METRICS_SCHEMA_VERSION,
    LEGACY_REQUEST_STAGE_NAMES,
    PERSISTENT_REQUEST_STAGE_NAMES,
    DistillationMetricIdentity,
    DistillationMetricsRecorder,
    DistillationPerformanceRunContext,
    DistillationStageMetric,
    DistillationStageObservation,
    load_distillation_metrics,
)
from unilab.algos.torch.distill.trainer import BehaviorDistillationTrainer

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


class _FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _IncrementingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 1.0
        return current


def _identity(**overrides: object) -> DistillationMetricIdentity:
    values: dict[str, object] = {
        "execution_mode": "persistent_async",
        "outer_iteration": 1,
        "scenario": "walk_flat",
        "worker_pid": 17,
        "request_id": "dagger-1-walk_flat",
        "checkpoint_path": "/private/tmp/student.pt",
        "checkpoint_sha256": _HASH_A,
        "weight_version": 3,
        "teacher_checkpoint_sha256": (_HASH_B, _HASH_C),
        "config_sha256": _HASH_C,
        "seed": 11,
        "device": "cpu",
        "num_envs": 2,
    }
    values.update(overrides)
    return DistillationMetricIdentity(**values)


def test_distillation_metrics_fake_clock_roundtrip(tmp_path: Path) -> None:
    recorder = DistillationMetricsRecorder(clock=_FakeClock(10.0, 10.25))

    with recorder.measure(
        identity=_identity(),
        stage="teacher_inference",
        row_count=8,
        env_step_count=2,
        cleanup_state="complete",
    ):
        pass

    record = recorder.records[0]
    assert record.duration_seconds == pytest.approx(0.25)
    assert record.rows_per_second == pytest.approx(32.0)
    output_path = recorder.write(
        tmp_path / "distillation_metrics.json",
        required_stages=("teacher_inference",),
    )
    loaded = load_distillation_metrics(output_path)
    assert loaded.schema_version == DISTILLATION_METRICS_SCHEMA_VERSION
    assert loaded.records == recorder.records


def test_offline_learner_owner_emits_exact_stage_observations(tmp_path: Path) -> None:
    student = MLPStudentPolicy(obs_dim=4, action_dim=2, hidden_dims=(8,))
    teacher = torch.nn.Linear(4, 2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=torch.optim.Adam(student.parameters(), lr=1e-3),
    )
    dataset = make_fake_distillation_dataset(
        num_samples=2,
        student_obs_dim=4,
        teacher_obs_dim=4,
        seed=3,
    )

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=2,
        max_updates=1,
        checkpoint_path=tmp_path / "student.pt",
        performance_clock=_IncrementingClock(),
    )

    observations = result.performance_stage_observations
    assert tuple(item.stage for item in observations) == (
        "learner_batch_staging",
        "learner_forward",
        "learner_backward",
        "optimizer_step",
        "checkpoint_save",
    )
    assert [item.duration_seconds for item in observations] == pytest.approx([1.0] * 5)
    assert [item.row_count for item in observations] == [2] * 5


def test_distillation_stage_observation_roundtrip_and_identity_enrichment() -> None:
    observation = DistillationStageObservation(
        stage="artifact_write",
        duration_seconds=0.25,
        row_count=8,
        env_step_count=0,
        success=True,
        error=None,
        cleanup_state="not_applicable",
    )

    assert DistillationStageObservation.from_dict(observation.as_dict()) == observation
    metric = DistillationStageMetric.from_observation(_identity(), observation)
    assert metric.identity == _identity()
    assert metric.stage == "artifact_write"
    assert metric.rows_per_second == pytest.approx(32.0)


def _persistent_request_observations() -> tuple[DistillationStageObservation, ...]:
    return tuple(
        DistillationStageObservation(
            stage=stage,
            duration_seconds=float(index + 1) / 10.0,
            row_count=8 if stage not in {"weight_sync", "env_step"} else 0,
            env_step_count=3 if stage in {"env_step", "total_elapsed"} else 0,
            success=True,
            error=None,
            cleanup_state="pending" if stage == "total_elapsed" else "not_applicable",
        )
        for index, stage in enumerate(PERSISTENT_REQUEST_STAGE_NAMES)
    )


def _legacy_request_observations() -> tuple[DistillationStageObservation, ...]:
    return tuple(
        DistillationStageObservation(
            stage=stage,
            duration_seconds=float(index + 1) / 10.0,
            row_count=8 if stage not in {"cold_start", "env_step"} else 0,
            env_step_count=3 if stage in {"env_step", "total_elapsed"} else 0,
            success=True,
            error=None,
            cleanup_state="complete" if stage == "total_elapsed" else "not_applicable",
        )
        for index, stage in enumerate(LEGACY_REQUEST_STAGE_NAMES)
    )


def test_performance_run_context_enriches_exact_legacy_request_identity() -> None:
    context = DistillationPerformanceRunContext(
        execution_mode="legacy",
        teacher_checkpoint_sha256=(_HASH_B,),
        config_sha256=_HASH_A,
        seed=11,
        device="cpu",
        num_envs=2,
    )

    records = context.enrich_request(
        outer_iteration=1,
        scenario="static_stand",
        worker_pid=71,
        request_id="dagger-1-static_stand",
        checkpoint_path="/private/tmp/student.pt",
        checkpoint_sha256=_HASH_C,
        weight_version=None,
        schema_version=DISTILLATION_METRICS_SCHEMA_VERSION,
        observations=_legacy_request_observations(),
    )

    assert tuple(record.stage for record in records) == LEGACY_REQUEST_STAGE_NAMES
    assert {record.identity.execution_mode for record in records} == {"legacy"}
    assert {record.identity.weight_version for record in records} == {None}


def test_performance_run_context_enriches_exact_persistent_request_identity() -> None:
    context = DistillationPerformanceRunContext(
        execution_mode="persistent_async",
        teacher_checkpoint_sha256=(_HASH_C, _HASH_B),
        config_sha256=_HASH_A,
        seed=11,
        device="cpu",
        num_envs=2,
    )

    records = context.enrich_request(
        outer_iteration=2,
        scenario="walk_to_stop",
        worker_pid=71,
        request_id="dagger-2-walk_to_stop",
        checkpoint_path="/private/tmp/student.pt",
        checkpoint_sha256=_HASH_C,
        weight_version=9,
        schema_version=DISTILLATION_METRICS_SCHEMA_VERSION,
        observations=_persistent_request_observations(),
    )

    assert tuple(record.stage for record in records) == PERSISTENT_REQUEST_STAGE_NAMES
    assert {record.identity.outer_iteration for record in records} == {2}
    assert {record.identity.scenario for record in records} == {"walk_to_stop"}
    assert {record.identity.request_id for record in records} == {"dagger-2-walk_to_stop"}
    assert {record.identity.checkpoint_sha256 for record in records} == {_HASH_C}
    assert {record.identity.weight_version for record in records} == {9}
    assert {record.identity.teacher_checkpoint_sha256 for record in records} == {(_HASH_B, _HASH_C)}
    assert [record.duration_seconds for record in records] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )


@pytest.mark.parametrize(
    ("schema_version", "observations", "match"),
    [
        (2, _persistent_request_observations(), "schema_version"),
        (
            1,
            tuple(reversed(_persistent_request_observations())),
            "stage order",
        ),
        (1, _persistent_request_observations()[:-1], "stage order"),
    ],
)
def test_performance_run_context_rejects_malformed_request_observations(
    schema_version: int,
    observations: tuple[DistillationStageObservation, ...],
    match: str,
) -> None:
    context = DistillationPerformanceRunContext(
        execution_mode="persistent_async",
        teacher_checkpoint_sha256=(_HASH_B,),
        config_sha256=_HASH_A,
        seed=11,
        device="cpu",
        num_envs=2,
    )

    with pytest.raises(ValueError, match=match):
        context.enrich_request(
            outer_iteration=1,
            scenario="stand",
            worker_pid=71,
            request_id="dagger-1-stand",
            checkpoint_path="/private/tmp/student.pt",
            checkpoint_sha256=_HASH_C,
            weight_version=4,
            schema_version=schema_version,
            observations=observations,
        )


def test_performance_run_context_rejects_duplicate_or_empty_teacher_identity() -> None:
    with pytest.raises(ValueError, match="teacher_checkpoint_sha256"):
        DistillationPerformanceRunContext(
            execution_mode="persistent_async",
            teacher_checkpoint_sha256=(),
            config_sha256=_HASH_A,
            seed=11,
            device="cpu",
            num_envs=2,
        )
    with pytest.raises(ValueError, match="duplicates"):
        DistillationPerformanceRunContext(
            execution_mode="persistent_async",
            teacher_checkpoint_sha256=(_HASH_B, _HASH_B),
            config_sha256=_HASH_A,
            seed=11,
            device="cpu",
            num_envs=2,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"checkpoint_sha256": "bad"}, "checkpoint_sha256"),
        ({"execution_mode": "legacy", "weight_version": 1}, "weight_version"),
        ({"execution_mode": "persistent_async", "weight_version": None}, "weight_version"),
        ({"num_envs": 0}, "num_envs"),
        ({"checkpoint_path": "/private/tmp/dir/../student.pt"}, "canonical"),
    ],
)
def test_distillation_metric_identity_rejects_invalid_fields(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _identity(**overrides)


@pytest.mark.parametrize(
    ("field", "value"),
    [("duration_seconds", -0.1), ("row_count", -1), ("env_step_count", -1)],
)
def test_distillation_stage_metric_rejects_negative_values(
    field: str,
    value: float | int,
) -> None:
    kwargs: dict[str, object] = {
        "identity": _identity(),
        "stage": "env_step",
        "duration_seconds": 0.1,
        "row_count": 2,
        "env_step_count": 1,
        "success": True,
        "error": None,
        "cleanup_state": "not_applicable",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        DistillationStageMetric(**kwargs)


def test_distillation_metrics_reject_identity_drift() -> None:
    recorder = DistillationMetricsRecorder()
    first = DistillationStageMetric(
        identity=_identity(),
        stage="env_reset",
        duration_seconds=0.1,
        row_count=2,
        env_step_count=0,
        success=True,
        error=None,
        cleanup_state="complete",
    )
    recorder.add(first)
    drifted = replace(
        first,
        identity=replace(first.identity, config_sha256=_HASH_B),
        stage="env_step",
    )
    with pytest.raises(ValueError, match="identity drift"):
        recorder.add(drifted)


def test_distillation_metrics_reject_request_checkpoint_drift() -> None:
    recorder = DistillationMetricsRecorder()
    first = DistillationStageMetric(
        identity=_identity(),
        stage="env_reset",
        duration_seconds=0.1,
        row_count=2,
        env_step_count=0,
        success=True,
        error=None,
        cleanup_state="complete",
    )
    recorder.add(first)
    drifted = replace(
        first,
        identity=replace(
            first.identity,
            checkpoint_path="/private/tmp/other.pt",
            checkpoint_sha256=_HASH_B,
        ),
        stage="env_step",
    )
    with pytest.raises(ValueError, match="request identity drift"):
        recorder.add(drifted)


def test_distillation_metrics_reject_incompatible_duplicate() -> None:
    recorder = DistillationMetricsRecorder()
    record = DistillationStageMetric(
        identity=_identity(),
        stage="artifact_write",
        duration_seconds=0.1,
        row_count=8,
        env_step_count=0,
        success=True,
        error=None,
        cleanup_state="complete",
    )
    recorder.add(record)
    recorder.add(record)
    assert recorder.records == (record,)
    with pytest.raises(ValueError, match="duplicate"):
        recorder.add(replace(record, duration_seconds=0.2))


def test_distillation_metrics_reject_missing_required_stage(tmp_path: Path) -> None:
    recorder = DistillationMetricsRecorder()
    recorder.add(
        DistillationStageMetric(
            identity=_identity(),
            stage="env_reset",
            duration_seconds=0.1,
            row_count=2,
            env_step_count=0,
            success=True,
            error=None,
            cleanup_state="complete",
        )
    )
    with pytest.raises(ValueError, match="missing required stages"):
        recorder.write(tmp_path / "metrics.json", required_stages=("env_step",))


def test_distillation_metrics_load_rejects_derived_rate_drift(tmp_path: Path) -> None:
    recorder = DistillationMetricsRecorder(clock=_FakeClock(1.0, 1.5))
    with recorder.measure(
        identity=_identity(),
        stage="tensor_pack",
        row_count=4,
        env_step_count=0,
        cleanup_state="complete",
    ):
        pass
    path = recorder.write(tmp_path / "metrics.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["rows_per_second"] = 999.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rows_per_second"):
        load_distillation_metrics(path)


def test_distillation_metrics_load_rejects_coercible_identity_type(
    tmp_path: Path,
) -> None:
    recorder = DistillationMetricsRecorder(clock=_FakeClock(1.0, 1.5))
    with recorder.measure(
        identity=_identity(),
        stage="env_step",
        row_count=4,
        env_step_count=1,
        cleanup_state="complete",
    ):
        pass
    path = recorder.write(tmp_path / "metrics.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["identity"]["num_envs"] = 1.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="num_envs"):
        load_distillation_metrics(path)


def test_distillation_metrics_measure_records_error_before_reraising() -> None:
    recorder = DistillationMetricsRecorder(clock=_FakeClock(1.0, 1.5))

    with pytest.raises(RuntimeError, match="probe failure"):
        with recorder.measure(
            identity=_identity(),
            stage="total_elapsed",
            row_count=0,
            env_step_count=0,
            cleanup_state="complete",
        ):
            raise RuntimeError("probe failure")

    record = recorder.records[0]
    assert record.success is False
    assert record.error == "RuntimeError: probe failure"
    assert record.cleanup_state == "failed"
