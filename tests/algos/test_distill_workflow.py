from __future__ import annotations

import json
from pathlib import Path

import torch

from unilab.algos.torch.distill.data import (
    build_distillation_dataset,
    save_distillation_dataset,
)
from unilab.algos.torch.distill.workflow import (
    ArtifactDecision,
    RoleArtifactSpec,
    adopt_legacy_role_artifact,
    create_role_artifact_manifest,
    fork_workflow_run,
    preflight_role_artifacts,
    run_bootstrap_workflow,
    run_multirole_dagger_workflow,
    write_role_artifact_manifest,
)


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


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

    assert spec.manifest_path.is_file()
    assert preflight_role_artifacts((spec,))[0].decision is ArtifactDecision.REUSE


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
        _materialize(spec, num_samples=4)
    run_dir = tmp_path / "run"
    run_bootstrap_workflow(
        run_dir=run_dir,
        role_specs=specs,
        collect_role=lambda _spec: (_ for _ in ()).throw(AssertionError("must reuse")),
        assemble_roles=lambda _paths, output: (_write(output, b"bootstrap-data") and 8),
        update_student=lambda _dataset, checkpoint: (_write(checkpoint, b"student-0") and 2),
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
    assert fork_manifest["parent_run_manifest"] == str(
        (run_dir / "run_manifest.json").resolve()
    )
    assert fork_manifest["completed_dagger_iterations"] == 0
