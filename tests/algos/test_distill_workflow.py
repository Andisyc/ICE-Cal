from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from unilab.algos.torch.distill.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
    PersistentDaggerCollectorRunner,
)
from unilab.algos.torch.distill.data import (
    build_distillation_dataset,
    load_distillation_dataset,
    save_distillation_dataset,
)
from unilab.algos.torch.distill.offline import required_balanced_replay_updates
from unilab.algos.torch.distill.performance import (
    DISTILLATION_METRICS_SCHEMA_VERSION,
    LEGACY_REQUEST_STAGE_NAMES,
    PERSISTENT_REQUEST_STAGE_NAMES,
    WORKFLOW_ITERATION_STAGE_NAMES,
    DistillationPerformanceRunContext,
    DistillationStageObservation,
    load_distillation_metrics,
)
from unilab.algos.torch.distill.workflow import (
    ArtifactDecision,
    RoleArtifactSpec,
    WorkflowScenarioCollectionResult,
    WorkflowScenarioSpec,
    WorkflowStudentUpdateResult,
    adopt_legacy_role_artifact,
    create_role_artifact_manifest,
    file_sha256,
    finalize_workflow_performance,
    fork_workflow_run,
    preflight_role_artifacts,
    run_bootstrap_workflow,
    run_multirole_dagger_workflow,
    write_role_artifact_manifest,
)

_CONFIG_HASH = "a" * 64


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _performance_context(
    specs: tuple[RoleArtifactSpec, ...],
    *,
    config_sha256: str = _CONFIG_HASH,
    execution_mode: str = "persistent_async",
) -> DistillationPerformanceRunContext:
    return DistillationPerformanceRunContext(
        execution_mode=execution_mode,
        teacher_checkpoint_sha256=tuple(
            sorted({file_sha256(spec.teacher_checkpoint_path) for spec in specs})
        ),
        config_sha256=config_sha256,
        seed=7,
        device="cpu",
        num_envs=2,
    )


def _legacy_performance_result(num_samples: int = 3) -> WorkflowScenarioCollectionResult:
    observations = tuple(
        DistillationStageObservation(
            stage=stage,
            duration_seconds=float(index + 1) / 10.0,
            row_count=num_samples if stage not in {"cold_start", "env_step"} else 0,
            env_step_count=1 if stage in {"env_step", "total_elapsed"} else 0,
            success=True,
            error=None,
            cleanup_state="complete" if stage == "total_elapsed" else "not_applicable",
        )
        for index, stage in enumerate(LEGACY_REQUEST_STAGE_NAMES)
    )
    return WorkflowScenarioCollectionResult(
        num_samples=num_samples,
        worker_pid=os.getpid(),
        performance_metrics_schema_version=DISTILLATION_METRICS_SCHEMA_VERSION,
        performance_stage_observations=observations,
    )


def _performance_metadata() -> dict[str, object]:
    observations = tuple(
        DistillationStageObservation(
            stage=stage,
            duration_seconds=float(index + 1) / 10.0,
            row_count=3 if stage not in {"weight_sync", "env_step"} else 0,
            env_step_count=1 if stage in {"env_step", "total_elapsed"} else 0,
            success=True,
            error=None,
            cleanup_state="pending" if stage == "total_elapsed" else "not_applicable",
        )
        for index, stage in enumerate(PERSISTENT_REQUEST_STAGE_NAMES)
    )
    return {
        "performance_metrics_schema_version": DISTILLATION_METRICS_SCHEMA_VERSION,
        "performance_stage_observations": [observation.as_dict() for observation in observations],
    }


def _spec(tmp_path: Path, role: str = "stand") -> RoleArtifactSpec:
    teacher = _write(tmp_path / f"{role}_teacher.pt", f"{role}-teacher-v1".encode())
    return RoleArtifactSpec(
        role=role,
        task=f"g1_{role}/mujoco",
        teacher_checkpoint_path=teacher,
        dataset_path=tmp_path / f"{role}.pt",
        schema_version=1,
        student_obs_dim=98,
        teacher_obs_dim=98,
        teacher_action_dim=29,
        teacher_obs_key="obs",
        teacher_projection="identity",
        student_projection="identity",
        student_drop_index=None,
        command_sample_filter="inactive" if role == "stand" else "active",
        command_info_key="commands",
        command_xy_threshold=0.05,
        command_yaw_threshold=0.05,
        owner_config={
            "training": {"task_name": f"G1{role.title().replace('_', '')}"},
            "task": f"g1_{role}/mujoco",
            "ctrl_dt": 0.02,
        },
    )


def _materialize(spec: RoleArtifactSpec, *, num_samples: int = 8) -> None:
    _write(spec.dataset_path, f"dataset:{spec.role}".encode())
    manifest = create_role_artifact_manifest(spec, num_samples=num_samples)
    write_role_artifact_manifest(spec.manifest_path, manifest)


def _materialize_with_role_labels(spec: RoleArtifactSpec, *, num_samples: int = 8) -> None:
    dataset = build_distillation_dataset(
        torch.zeros(num_samples, spec.student_obs_dim),
        torch.zeros(num_samples, spec.teacher_obs_dim),
        expected_student_obs_dim=spec.student_obs_dim,
        expected_teacher_obs_dim=spec.teacher_obs_dim,
        expected_teacher_action_dim=spec.teacher_action_dim,
        teacher_actions=torch.zeros(num_samples, spec.teacher_action_dim),
        role_labels=(spec.role,) * num_samples,
    )
    save_distillation_dataset(spec.dataset_path, dataset)
    manifest = create_role_artifact_manifest(spec, num_samples=num_samples)
    write_role_artifact_manifest(spec.manifest_path, manifest)


def test_role_artifact_preflight_collects_only_missing_role(tmp_path: Path) -> None:
    stand = _spec(tmp_path, "stand")
    walk = _spec(tmp_path, "walk_flat")
    height = _spec(tmp_path, "height")
    _materialize(stand)
    _materialize(walk)

    results = preflight_role_artifacts((stand, walk, height))

    assert {result.role: result.decision for result in results} == {
        "stand": ArtifactDecision.REUSE,
        "walk_flat": ArtifactDecision.REUSE,
        "height": ArtifactDecision.COLLECT,
    }


def test_role_artifact_preflight_detects_stale_teacher_and_dataset(tmp_path: Path) -> None:
    teacher_changed = _spec(tmp_path / "teacher_changed")
    _materialize(teacher_changed)
    teacher_changed.teacher_checkpoint_path.write_bytes(b"stand-teacher-v2")

    dataset_changed = _spec(tmp_path / "dataset_changed")
    _materialize(dataset_changed)
    dataset_changed.dataset_path.write_bytes(b"changed-dataset")

    teacher_result = preflight_role_artifacts((teacher_changed,))[0]
    dataset_result = preflight_role_artifacts((dataset_changed,))[0]

    assert teacher_result.decision is ArtifactDecision.STALE
    assert "teacher_checkpoint_sha256" in teacher_result.mismatches
    assert dataset_result.decision is ArtifactDecision.STALE
    assert "dataset_sha256" in dataset_result.mismatches


def test_role_artifact_preflight_rejects_schema_incompatibility(tmp_path: Path) -> None:
    original = _spec(tmp_path)
    _materialize(original)
    changed_contract = RoleArtifactSpec(
        **{
            **original.as_dict(),
            "student_obs_dim": 99,
            "owner_config": original.owner_config,
        }
    )

    result = preflight_role_artifacts((changed_contract,))[0]

    assert result.decision is ArtifactDecision.INCOMPATIBLE
    assert result.mismatches == ("student_obs_dim",)


def test_role_artifact_preflight_does_not_reuse_file_without_manifest(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.dataset_path, b"untracked-dataset")

    result = preflight_role_artifacts((spec,))[0]

    assert result.decision is ArtifactDecision.STALE
    assert result.mismatches == ("manifest_missing",)


def test_scenario_preflight_rejects_manifest_without_row_role_labels(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "walk_flat")
    dataset = build_distillation_dataset(
        torch.zeros(4, 98),
        torch.zeros(4, 98),
        expected_student_obs_dim=98,
        expected_teacher_obs_dim=98,
        expected_teacher_action_dim=29,
        teacher_actions=torch.zeros(4, 29),
    )
    save_distillation_dataset(spec.dataset_path, dataset)
    write_role_artifact_manifest(
        spec.manifest_path,
        create_role_artifact_manifest(spec, num_samples=dataset.num_samples),
    )

    result = preflight_role_artifacts(
        (spec,),
        require_row_role_labels=True,
    )[0]

    assert result.decision is ArtifactDecision.STALE
    assert result.mismatches == ("role_labels",)


def test_explicit_legacy_adoption_validates_dataset_before_writing_manifest(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, "stand")
    dataset = build_distillation_dataset(
        torch.zeros(4, 98),
        torch.zeros(4, 98),
        teacher_actions=torch.zeros(4, 29),
        metadata={
            "task_name": "G1Stand",
            "teacher_policy_checkpoint_path": str(spec.teacher_checkpoint_path.resolve()),
            "teacher_projection": "identity",
            "student_projection": "identity",
            "teacher_obs_key": "obs",
            "student_drop_index": None,
            "command_sample_filter": "inactive",
            "command_info_key": "commands",
            "command_xy_threshold": 0.05,
            "command_yaw_threshold": 0.05,
        },
    )
    save_distillation_dataset(spec.dataset_path, dataset)

    adopt_legacy_role_artifact(spec)
    adopt_legacy_role_artifact(spec)

    assert spec.manifest_path.is_file()
    assert preflight_role_artifacts((spec,))[0].decision is ArtifactDecision.REUSE
    adopted = load_distillation_dataset(spec.dataset_path)
    assert adopted.role_labels == ("stand",) * adopted.num_samples
    assert adopted.metadata["legacy_role_labels_adopted"] is True
    assert (
        preflight_role_artifacts((spec,), require_row_role_labels=True)[0].decision
        is ArtifactDecision.REUSE
    )


def test_legacy_adoption_repairs_manifest_written_before_role_label_migration(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, "stand")
    dataset = build_distillation_dataset(
        torch.zeros(4, 98),
        torch.zeros(4, 98),
        teacher_actions=torch.zeros(4, 29),
        metadata={
            "task_name": "G1Stand",
            "teacher_policy_checkpoint_path": str(spec.teacher_checkpoint_path.resolve()),
            "teacher_projection": "identity",
            "student_projection": "identity",
            "teacher_obs_key": "obs",
            "student_drop_index": None,
            "command_sample_filter": "inactive",
            "command_info_key": "commands",
            "command_xy_threshold": 0.05,
            "command_yaw_threshold": 0.05,
        },
    )
    save_distillation_dataset(spec.dataset_path, dataset)
    write_role_artifact_manifest(
        spec.manifest_path,
        create_role_artifact_manifest(spec, num_samples=dataset.num_samples),
    )

    adopt_legacy_role_artifact(spec)

    result = preflight_role_artifacts((spec,), require_row_role_labels=True)[0]
    assert result.decision is ArtifactDecision.REUSE


def test_required_balanced_replay_updates_scales_with_transition_rows() -> None:
    dataset = build_distillation_dataset(
        torch.zeros(64, 98),
        torch.zeros(64, 98),
        teacher_actions=torch.zeros(64, 29),
        scenario_labels=("walk_to_stop",) * 32 + ("static_stand",) * 16 + ("walk_flat",) * 16,
        transition_ages=torch.tensor([0] * 32 + [-1] * 32),
        command_before=torch.zeros(64, 3),
        command_after=torch.zeros(64, 3),
    )

    required = required_balanced_replay_updates(
        dataset,
        balance_key="scenario",
        batch_size=8,
        balanced_labels=("walk_flat", "static_stand", "walk_to_stop"),
        balance_quotas={"walk_flat": 0.25, "static_stand": 0.25, "walk_to_stop": 0.5},
        replay_labels=("walk_to_stop",),
        replay_passes=2,
    )

    assert required == 16


def test_bootstrap_workflow_collects_only_missing_roles_and_owns_paths(
    tmp_path: Path,
) -> None:
    stand = _spec(tmp_path / "artifacts", "stand")
    walk = _spec(tmp_path / "artifacts", "walk_flat")
    _materialize(stand, num_samples=8)
    calls: list[tuple[str, object]] = []

    def collect(spec: RoleArtifactSpec) -> int:
        calls.append(("collect", spec.role))
        _write(spec.dataset_path, b"new-walk-dataset")
        return 12

    def assemble(dataset_paths: tuple[Path, ...], output_path: Path) -> int:
        calls.append(("assemble", tuple(dataset_paths)))
        _write(output_path, b"merged")
        return 20

    def update(dataset_path: Path, checkpoint_path: Path) -> int:
        calls.append(("update", dataset_path))
        _write(checkpoint_path, b"student")
        return 5

    run_dir = tmp_path / "run"
    result = run_bootstrap_workflow(
        run_dir=run_dir,
        role_specs=(stand, walk),
        collect_role=collect,
        assemble_roles=assemble,
        update_student=update,
    )

    assert calls[0] == ("collect", "walk_flat")
    assert calls[1][0] == "assemble"
    assert calls[2][0] == "update"
    assert result.role_decisions == {"stand": "REUSE", "walk_flat": "COLLECT"}
    assert result.bootstrap_num_samples == 20
    assert result.bootstrap_updates == 5
    assert result.checkpoint_path == run_dir / "checkpoints" / "bootstrap_student.pt"
    assert result.checkpoint_path.is_file()
    assert (run_dir / "run_manifest.json").is_file()


def test_bootstrap_workflow_fails_closed_on_stale_role(tmp_path: Path) -> None:
    stand = _spec(tmp_path, "stand")
    _materialize(stand)
    stand.teacher_checkpoint_path.write_bytes(b"new-teacher")

    try:
        run_bootstrap_workflow(
            run_dir=tmp_path / "run",
            role_specs=(stand,),
            collect_role=lambda _spec: 1,
            assemble_roles=lambda _paths, _output: 1,
            update_student=lambda _dataset, _checkpoint: 1,
        )
    except ValueError as exc:
        assert "STALE" in str(exc)
        assert "stand" in str(exc)
    else:
        raise AssertionError("stale role artifact must fail closed")


def _bootstrap_two_role_run(tmp_path: Path) -> tuple[Path, tuple[RoleArtifactSpec, ...]]:
    specs = (_spec(tmp_path / "artifacts", "stand"), _spec(tmp_path / "artifacts", "walk_flat"))
    for spec in specs:
        _materialize_with_role_labels(spec, num_samples=4)
    run_dir = tmp_path / "run"
    run_bootstrap_workflow(
        run_dir=run_dir,
        role_specs=specs,
        collect_role=lambda _spec: (_ for _ in ()).throw(AssertionError("must reuse")),
        assemble_roles=lambda _paths, output: _write(output, b"bootstrap-data") and 8,
        update_student=lambda _dataset, checkpoint: _write(checkpoint, b"student-0") and 2,
    )
    return run_dir, specs


def test_multirole_dagger_uses_previous_round_and_cumulative_data(tmp_path: Path) -> None:
    run_dir, specs = _bootstrap_two_role_run(tmp_path)
    rollout_inputs: list[tuple[int, str, str]] = []
    aggregate_sizes: list[int] = []

    def collect(
        spec: RoleArtifactSpec,
        checkpoint_path: Path,
        iteration: int,
        output_path: Path,
    ) -> int:
        rollout_inputs.append((iteration, spec.role, checkpoint_path.name))
        _write(output_path, f"{iteration}:{spec.role}".encode())
        return 3

    def aggregate(paths: tuple[Path, ...], output_path: Path) -> int:
        aggregate_sizes.append(len(paths))
        _write(output_path, f"sources={len(paths)}".encode())
        return len(paths) * 3

    def update(
        _dataset_path: Path,
        input_checkpoint_path: Path,
        output_checkpoint_path: Path,
    ) -> int:
        _write(output_checkpoint_path, input_checkpoint_path.read_bytes() + b"+update")
        return 2

    result = run_multirole_dagger_workflow(
        run_dir=run_dir,
        role_specs=specs,
        target_iterations=2,
        collect_role=collect,
        aggregate_datasets=aggregate,
        update_student=update,
    )

    assert rollout_inputs == [
        (1, "stand", "bootstrap_student.pt"),
        (1, "walk_flat", "bootstrap_student.pt"),
        (2, "stand", "dagger_iteration_1.pt"),
        (2, "walk_flat", "dagger_iteration_1.pt"),
    ]
    assert aggregate_sizes == [4, 6]
    assert result.completed_iterations == 2
    assert result.checkpoint_path.name == "dagger_iteration_2.pt"


def test_multirole_dagger_resume_runs_only_missing_round_and_fork_preserves_parent(
    tmp_path: Path,
) -> None:
    run_dir, specs = _bootstrap_two_role_run(tmp_path)

    def collect(
        spec: RoleArtifactSpec,
        _checkpoint_path: Path,
        iteration: int,
        output_path: Path,
    ) -> int:
        _write(output_path, f"{iteration}:{spec.role}".encode())
        return 2

    def aggregate(paths: tuple[Path, ...], output_path: Path) -> int:
        _write(output_path, b"aggregate")
        return len(paths) * 2

    def update(_dataset: Path, input_checkpoint: Path, output_checkpoint: Path) -> int:
        _write(output_checkpoint, input_checkpoint.read_bytes() + b"+")
        return 1

    run_multirole_dagger_workflow(
        run_dir=run_dir,
        role_specs=specs,
        target_iterations=1,
        collect_role=collect,
        aggregate_datasets=aggregate,
        update_student=update,
    )
    resumed_calls: list[int] = []

    def resume_collect(
        spec: RoleArtifactSpec,
        checkpoint_path: Path,
        iteration: int,
        output_path: Path,
    ) -> int:
        del spec, checkpoint_path
        resumed_calls.append(iteration)
        _write(output_path, b"resume")
        return 2

    run_multirole_dagger_workflow(
        run_dir=run_dir,
        role_specs=specs,
        target_iterations=2,
        collect_role=resume_collect,
        aggregate_datasets=aggregate,
        update_student=update,
    )
    assert resumed_calls == [2, 2]

    parent_manifest_before = (run_dir / "run_manifest.json").read_bytes()
    fork_dir = tmp_path / "fork"
    fork_workflow_run(parent_run_dir=run_dir, run_dir=fork_dir)
    fork_manifest = json.loads((fork_dir / "run_manifest.json").read_text())

    assert (run_dir / "run_manifest.json").read_bytes() == parent_manifest_before
    assert fork_manifest["mode"] == "fork"
    assert fork_manifest["parent_run_manifest"] == str((run_dir / "run_manifest.json").resolve())
    assert fork_manifest["completed_dagger_iterations"] == 0


def test_multirole_dagger_scenario_manifest_and_quota_sources(tmp_path: Path) -> None:
    scenarios = (
        WorkflowScenarioSpec("walk_flat", "role", ("walk_flat",), 0.5),
        WorkflowScenarioSpec("static_stand", "role", ("stand",), 0.25),
        WorkflowScenarioSpec("walk_to_stop", "transition", ("walk_flat", "stand"), 0.25),
    )
    specs = (_spec(tmp_path / "artifacts", "stand"), _spec(tmp_path / "artifacts", "walk_flat"))
    for spec in specs:
        _materialize_with_role_labels(spec, num_samples=4)
    run_dir = tmp_path / "scenario_run"

    run_bootstrap_workflow(
        run_dir=run_dir,
        role_specs=specs,
        scenario_specs=scenarios,
        collect_role=lambda _spec: (_ for _ in ()).throw(AssertionError("must reuse")),
        assemble_roles=lambda _paths, output: _write(output, b"bootstrap-data") and 8,
        update_student=lambda _dataset, checkpoint: _write(checkpoint, b"student-0") and 2,
    )
    collected: list[tuple[int, str, str]] = []
    aggregate_sizes: list[int] = []

    def collect_scenario(
        scenario: WorkflowScenarioSpec,
        checkpoint_path: Path,
        iteration: int,
        output_path: Path,
    ) -> int:
        collected.append((iteration, scenario.name, checkpoint_path.name))
        _write(output_path, f"{iteration}:{scenario.name}".encode())
        return 3

    def aggregate(sources, output_path: Path) -> int:
        aggregate_sizes.append(len(sources))
        _write(output_path, f"sources={len(sources)}".encode())
        return len(sources) * 3

    def update(_dataset: Path, input_checkpoint: Path, output_checkpoint: Path) -> int:
        _write(output_checkpoint, input_checkpoint.read_bytes() + b"+")
        return 2

    result = run_multirole_dagger_workflow(
        run_dir=run_dir,
        role_specs=specs,
        target_iterations=2,
        scenario_specs=scenarios,
        collect_role=lambda *_args: (_ for _ in ()).throw(
            AssertionError("scenario callback should own collection")
        ),
        collect_scenario=collect_scenario,
        aggregate_datasets=aggregate,
        update_student=update,
    )

    assert collected == [
        (1, "walk_flat", "bootstrap_student.pt"),
        (1, "static_stand", "bootstrap_student.pt"),
        (1, "walk_to_stop", "bootstrap_student.pt"),
        (2, "walk_flat", "dagger_iteration_1.pt"),
        (2, "static_stand", "dagger_iteration_1.pt"),
        (2, "walk_to_stop", "dagger_iteration_1.pt"),
    ]
    assert aggregate_sizes == [5, 8]
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["scenario_specs"] == [scenario.as_dict() for scenario in scenarios]
    assert len(manifest["dagger_iterations"][0]["scenario_artifacts"]) == 3
    assert "collection_execution_mode" not in manifest["dagger_iterations"][0]
    assert "input_weight_version" not in manifest["dagger_iterations"][0]
    assert "distillation_metrics_path" not in manifest
    assert not (run_dir / "distillation_metrics.json").exists()
    assert result.completed_iterations == 2

    parent_source_hashes = {
        source["path"]: file_sha256(Path(source["path"]))
        for source in manifest["bootstrap_sources"]
    }
    for iteration in manifest["dagger_iterations"]:
        parent_source_hashes.update(
            {
                source["dataset_path"]: file_sha256(Path(source["dataset_path"]))
                for source in iteration["scenario_artifacts"]
            }
        )
    fork_dir = tmp_path / "scenario_fork"
    fork_workflow_run(parent_run_dir=run_dir, run_dir=fork_dir)
    fork_manifest = json.loads((fork_dir / "run_manifest.json").read_text())
    assert [source["scenario"] for source in fork_manifest["bootstrap_sources"]] == [
        "static_stand",
        "walk_flat",
        "walk_flat",
        "static_stand",
        "walk_to_stop",
        "walk_flat",
        "static_stand",
        "walk_to_stop",
    ]
    assert all(source["preserve_row_role_labels"] for source in fork_manifest["bootstrap_sources"])
    assert {path: file_sha256(Path(path)) for path in parent_source_hashes} == parent_source_hashes

    with pytest.raises(ValueError, match="scenario specs do not match"):
        run_multirole_dagger_workflow(
            run_dir=run_dir,
            role_specs=specs,
            target_iterations=2,
            scenario_specs=(
                WorkflowScenarioSpec("walk_flat", "role", ("walk_flat",), 0.25),
                *scenarios[1:],
            ),
            collect_role=lambda *_args: 1,
            collect_scenario=collect_scenario,
            aggregate_datasets=aggregate,
            update_student=update,
        )


def test_multirole_dagger_legacy_rich_result_writes_request_metrics(
    tmp_path: Path,
) -> None:
    scenarios = (
        WorkflowScenarioSpec("walk_flat", "role", ("walk_flat",), 0.5),
        WorkflowScenarioSpec("static_stand", "role", ("stand",), 0.5),
    )
    specs = (
        _spec(tmp_path / "artifacts", "stand"),
        _spec(tmp_path / "artifacts", "walk_flat"),
    )
    for spec in specs:
        _materialize_with_role_labels(spec, num_samples=4)
    run_dir = tmp_path / "legacy_metrics_run"
    run_bootstrap_workflow(
        run_dir=run_dir,
        role_specs=specs,
        scenario_specs=scenarios,
        collect_role=lambda _spec: (_ for _ in ()).throw(AssertionError("must reuse")),
        assemble_roles=lambda _paths, output: _write(output, b"bootstrap") and 8,
        update_student=lambda _dataset, checkpoint: _write(checkpoint, b"student") and 1,
    )

    def collect_scenario(_scenario, _checkpoint, _iteration, output_path):
        _write(output_path, b"rows")
        return _legacy_performance_result()

    def update_student(_dataset, checkpoint, output):
        _write(output, checkpoint.read_bytes() + b"+")
        observations = tuple(
            DistillationStageObservation(
                stage=stage,
                duration_seconds=float(index + 1) / 10.0,
                row_count=6,
                env_step_count=0,
                success=True,
                error=None,
                cleanup_state="not_applicable",
            )
            for index, stage in enumerate(WORKFLOW_ITERATION_STAGE_NAMES[1:])
        )
        return WorkflowStudentUpdateResult(
            updates=1,
            performance_stage_observations=observations,
        )

    run_multirole_dagger_workflow(
        run_dir=run_dir,
        role_specs=specs,
        target_iterations=1,
        scenario_specs=scenarios,
        collect_role=lambda *_args: 1,
        collect_scenario=collect_scenario,
        aggregate_datasets=lambda sources, output: (
            _write(output, b"aggregate") and len(sources) * 3
        ),
        update_student=update_student,
        execution_mode="legacy",
        performance_context=_performance_context(specs, execution_mode="legacy"),
    )
    finalize_workflow_performance(
        run_dir=run_dir,
        performance_context=_performance_context(specs, execution_mode="legacy"),
        cleanup_duration_seconds=0.5,
        cleanup_report={
            "execution_mode": "legacy",
            "resource_scope": "per_request",
        },
    )
    metrics_before_resume = (run_dir / "distillation_metrics.json").read_bytes()
    manifest_before_resume = (run_dir / "run_manifest.json").read_bytes()
    finalize_workflow_performance(
        run_dir=run_dir,
        performance_context=_performance_context(specs, execution_mode="legacy"),
        cleanup_duration_seconds=9.0,
        cleanup_report={
            "execution_mode": "legacy",
            "resource_scope": "per_request",
        },
    )
    assert (run_dir / "distillation_metrics.json").read_bytes() == metrics_before_resume
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before_resume

    metrics = load_distillation_metrics(run_dir / "distillation_metrics.json")
    request_record_count = len(scenarios) * len(LEGACY_REQUEST_STAGE_NAMES)
    assert len(metrics.records) == request_record_count + len(WORKFLOW_ITERATION_STAGE_NAMES) + 1
    assert {record.identity.execution_mode for record in metrics.records} == {"legacy"}
    assert {record.identity.weight_version for record in metrics.records} == {None}
    workflow_records = [
        record for record in metrics.records if record.identity.scenario == "__workflow__"
    ]
    assert tuple(record.stage for record in workflow_records) == (WORKFLOW_ITERATION_STAGE_NAMES)
    cleanup_record = metrics.records[-1]
    assert cleanup_record.stage == "cleanup"
    assert cleanup_record.cleanup_state == "complete"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["performance_cleanup"] == {
        "state": "complete",
        "duration_seconds": 0.5,
        "report": {
            "execution_mode": "legacy",
            "resource_scope": "per_request",
        },
    }
    assert manifest["distillation_metrics_record_count"] == len(metrics.records)


def test_finalize_persistent_workflow_requires_resource_counters(tmp_path: Path) -> None:
    specs = (_spec(tmp_path / "artifacts", "stand"),)
    with pytest.raises(ValueError, match="positive worker_pid"):
        finalize_workflow_performance(
            run_dir=tmp_path / "missing",
            performance_context=_performance_context(specs),
            cleanup_duration_seconds=0.1,
            cleanup_report={},
        )


class _RecordingScenarioCollector:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.version = 40

    def activate_checkpoint(self, checkpoint_path: Path) -> int:
        self.version += 1
        self.events.append(("activate", checkpoint_path.name, self.version))
        return self.version

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        self.events.append(
            (
                "collect",
                request.iteration,
                request.scenario,
                Path(request.checkpoint_path).name,
                request.expected_weight_version,
            )
        )
        _write(Path(request.output_path), request.request_id.encode())
        return DaggerCollectResult(
            request_id=request.request_id,
            scenario=request.scenario,
            iteration=request.iteration,
            checkpoint_path=request.checkpoint_path,
            output_path=request.output_path,
            expected_weight_version=request.expected_weight_version,
            observed_weight_version=request.expected_weight_version,
            num_samples=3,
            worker_pid=os.getpid(),
            metrics={"collect_seconds": 0.25},
            metadata=_performance_metadata(),
        )


class _SpawnedWorkflowCollectorService:
    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        output_path = Path(request.output_path)
        assert output_path.parent.is_dir()
        output_path.write_bytes(request.request_id.encode())
        return DaggerCollectResult(
            request_id=request.request_id,
            scenario=request.scenario,
            iteration=request.iteration,
            checkpoint_path=request.checkpoint_path,
            output_path=request.output_path,
            expected_weight_version=request.expected_weight_version,
            observed_weight_version=request.expected_weight_version,
            num_samples=3,
            worker_pid=os.getpid(),
            metrics={"collect_seconds": 0.5},
            metadata=_performance_metadata(),
        )

    def close(self) -> None:
        pass


def _build_spawned_workflow_collector_service() -> _SpawnedWorkflowCollectorService:
    return _SpawnedWorkflowCollectorService()


def test_multirole_dagger_persistent_execution_preserves_outer_iteration_barrier(
    tmp_path: Path,
) -> None:
    scenarios = (
        WorkflowScenarioSpec("walk_flat", "role", ("walk_flat",), 0.5),
        WorkflowScenarioSpec("static_stand", "role", ("stand",), 0.25),
        WorkflowScenarioSpec("walk_to_stop", "transition", ("walk_flat", "stand"), 0.25),
    )
    specs = (
        _spec(tmp_path / "artifacts", "stand"),
        _spec(tmp_path / "artifacts", "walk_flat"),
    )
    for spec in specs:
        _materialize_with_role_labels(spec, num_samples=4)
    run_dir = tmp_path / "persistent_run"
    run_bootstrap_workflow(
        run_dir=run_dir,
        role_specs=specs,
        scenario_specs=scenarios,
        collect_role=lambda _spec: (_ for _ in ()).throw(AssertionError("must reuse")),
        assemble_roles=lambda _paths, output: _write(output, b"bootstrap-data") and 8,
        update_student=lambda _dataset, checkpoint: _write(checkpoint, b"student-0") and 2,
    )
    events: list[tuple] = []
    collector = _RecordingScenarioCollector(events)

    def aggregate(sources, output_path: Path) -> int:
        _write(output_path, f"sources={len(sources)}".encode())
        return len(sources) * 3

    def update(_dataset: Path, input_checkpoint: Path, output_checkpoint: Path) -> int:
        events.append(("update", input_checkpoint.name, output_checkpoint.name))
        _write(output_checkpoint, input_checkpoint.read_bytes() + b"+")
        return 2

    run_multirole_dagger_workflow(
        run_dir=run_dir,
        role_specs=specs,
        target_iterations=2,
        scenario_specs=scenarios,
        collect_role=lambda *_args: (_ for _ in ()).throw(
            AssertionError("scenario service should own collection")
        ),
        execution_mode="persistent_async",
        scenario_collector=collector,
        performance_context=_performance_context(specs),
        aggregate_datasets=aggregate,
        update_student=update,
    )
    finalize_workflow_performance(
        run_dir=run_dir,
        performance_context=_performance_context(specs),
        cleanup_duration_seconds=0.25,
        cleanup_report={
            "worker_pid": os.getpid(),
            "resource_counters": {"env_builds": 1, "student_inits": 1},
        },
    )

    assert events == [
        ("activate", "bootstrap_student.pt", 41),
        ("collect", 1, "walk_flat", "bootstrap_student.pt", 41),
        ("collect", 1, "static_stand", "bootstrap_student.pt", 41),
        ("collect", 1, "walk_to_stop", "bootstrap_student.pt", 41),
        ("update", "bootstrap_student.pt", "dagger_iteration_1.pt"),
        ("activate", "dagger_iteration_1.pt", 42),
        ("collect", 2, "walk_flat", "dagger_iteration_1.pt", 42),
        ("collect", 2, "static_stand", "dagger_iteration_1.pt", 42),
        ("collect", 2, "walk_to_stop", "dagger_iteration_1.pt", 42),
        ("update", "dagger_iteration_1.pt", "dagger_iteration_2.pt"),
    ]
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert [item["input_weight_version"] for item in manifest["dagger_iterations"]] == [41, 42]
    assert all(
        artifact["input_weight_version"] == iteration["input_weight_version"]
        for iteration in manifest["dagger_iterations"]
        for artifact in iteration["scenario_artifacts"]
    )
    assert all(
        artifact["collector_metrics"] == {"collect_seconds": 0.25}
        for iteration in manifest["dagger_iterations"]
        for artifact in iteration["scenario_artifacts"]
    )
    metrics_path = run_dir / "distillation_metrics.json"
    metrics_before_resume = metrics_path.read_bytes()
    metrics = load_distillation_metrics(metrics_path)
    assert len(metrics.records) == 2 * len(scenarios) * len(PERSISTENT_REQUEST_STAGE_NAMES) + 1
    assert metrics.records[-1].stage == "cleanup"
    assert metrics.records[-1].identity.weight_version == 42
    assert {record.identity.checkpoint_sha256 for record in metrics.records[:21]} == {
        file_sha256(run_dir / "checkpoints" / "bootstrap_student.pt")
    }
    assert {record.identity.weight_version for record in metrics.records[:21]} == {41}
    assert manifest["distillation_metrics_path"] == str(metrics_path.resolve())
    assert manifest["distillation_metrics_sha256"] == file_sha256(metrics_path)
    assert manifest["distillation_metrics_record_count"] == len(metrics.records)
    assert not (run_dir / "distillation_metrics.json.tmp").exists()

    events.clear()
    run_multirole_dagger_workflow(
        run_dir=run_dir,
        role_specs=specs,
        target_iterations=2,
        scenario_specs=scenarios,
        collect_role=lambda *_args: 1,
        execution_mode="persistent_async",
        scenario_collector=collector,
        performance_context=_performance_context(specs),
        aggregate_datasets=aggregate,
        update_student=update,
    )
    assert events == []
    assert metrics_path.read_bytes() == metrics_before_resume

    with pytest.raises(ValueError, match="identity drift"):
        run_multirole_dagger_workflow(
            run_dir=run_dir,
            role_specs=specs,
            target_iterations=2,
            scenario_specs=scenarios,
            collect_role=lambda *_args: 1,
            execution_mode="persistent_async",
            scenario_collector=collector,
            performance_context=_performance_context(
                specs,
                config_sha256="b" * 64,
            ),
            aggregate_datasets=aggregate,
            update_student=update,
        )


def test_multirole_dagger_connects_persistent_async_runner(tmp_path: Path) -> None:
    scenarios = (
        WorkflowScenarioSpec("stand", "role", ("stand",), 0.5),
        WorkflowScenarioSpec("walk_flat", "role", ("walk_flat",), 0.5),
    )
    specs = (
        _spec(tmp_path / "artifacts", "stand"),
        _spec(tmp_path / "artifacts", "walk_flat"),
    )
    for spec in specs:
        _materialize_with_role_labels(spec, num_samples=4)
    run_dir = tmp_path / "spawned_persistent_run"
    run_bootstrap_workflow(
        run_dir=run_dir,
        role_specs=specs,
        scenario_specs=scenarios,
        collect_role=lambda _spec: (_ for _ in ()).throw(AssertionError("must reuse")),
        assemble_roles=lambda _paths, output: _write(output, b"bootstrap-data") and 8,
        update_student=lambda _dataset, checkpoint: _write(checkpoint, b"student-0") and 2,
    )
    activated: list[str] = []

    def activate(checkpoint_path: Path) -> int:
        activated.append(checkpoint_path.name)
        return 7

    runner = PersistentDaggerCollectorRunner(
        worker_factory=_build_spawned_workflow_collector_service,
        checkpoint_activator=activate,
    )
    try:
        run_multirole_dagger_workflow(
            run_dir=run_dir,
            role_specs=specs,
            target_iterations=1,
            scenario_specs=scenarios,
            collect_role=lambda *_args: (_ for _ in ()).throw(
                AssertionError("scenario service should own collection")
            ),
            execution_mode="persistent_async",
            scenario_collector=runner,
            performance_context=_performance_context(specs),
            aggregate_datasets=lambda sources, output: (
                _write(output, b"aggregate") and len(sources) * 3
            ),
            update_student=lambda _dataset, checkpoint, output: (
                _write(output, checkpoint.read_bytes() + b"+") and 2
            ),
        )
    finally:
        runner.close()

    assert activated == ["bootstrap_student.pt"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    iteration = manifest["dagger_iterations"][0]
    assert iteration["input_weight_version"] == 7
    assert {artifact["collector_worker_pid"] for artifact in iteration["scenario_artifacts"]} != {
        os.getpid()
    }
    assert {artifact["input_weight_version"] for artifact in iteration["scenario_artifacts"]} == {7}
    assert (run_dir / "distillation_metrics.json").is_file()


def test_multirole_dagger_execution_mode_rejects_half_open_collectors(tmp_path: Path) -> None:
    run_dir, specs = _bootstrap_two_role_run(tmp_path)
    scenarios = (
        WorkflowScenarioSpec("stand", "role", ("stand",), 0.5),
        WorkflowScenarioSpec("walk_flat", "role", ("walk_flat",), 0.5),
    )
    events: list[tuple] = []
    common = {
        "run_dir": run_dir,
        "role_specs": specs,
        "target_iterations": 1,
        "scenario_specs": scenarios,
        "collect_role": lambda *_args: 1,
        "aggregate_datasets": lambda *_args: 1,
        "update_student": lambda *_args: 1,
    }

    with pytest.raises(ValueError, match="legacy.*scenario_collector"):
        run_multirole_dagger_workflow(
            **common,
            scenario_collector=_RecordingScenarioCollector(events),
        )
    with pytest.raises(ValueError, match="persistent_async.*collect_scenario"):
        run_multirole_dagger_workflow(
            **common,
            execution_mode="persistent_async",
            collect_scenario=lambda *_args: 1,
            scenario_collector=_RecordingScenarioCollector(events),
        )
    with pytest.raises(ValueError, match="persistent_async.*requires scenario_collector"):
        run_multirole_dagger_workflow(
            **common,
            execution_mode="persistent_async",
        )
    with pytest.raises(ValueError, match="persistent_async.*requires performance_context"):
        run_multirole_dagger_workflow(
            **common,
            execution_mode="persistent_async",
            scenario_collector=_RecordingScenarioCollector(events),
        )
    with pytest.raises(ValueError, match="execution_mode must be"):
        run_multirole_dagger_workflow(
            **common,
            execution_mode="unknown",
        )

    no_scenario_common = dict(common)
    no_scenario_common.pop("scenario_specs")
    with pytest.raises(ValueError, match="persistent_async.*requires scenario_specs"):
        run_multirole_dagger_workflow(
            **no_scenario_common,
            execution_mode="persistent_async",
            scenario_collector=_RecordingScenarioCollector(events),
        )
