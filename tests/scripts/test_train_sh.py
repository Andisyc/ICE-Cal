from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _run_train_sh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "train.sh", "--dry-run", *args],
        cwd=ROOT_DIR,
        capture_output=True,
        check=check,
        env=os.environ.copy(),
        text=True,
    )


def test_train_sh_fresh_owns_one_paired_time_sorted_identity() -> None:
    result = _run_train_sh(
        "--workflow-mode",
        "fresh",
        "--run-name",
        "pytest-fresh",
        "--execution-mode",
        "persistent_async",
        "algo.seed=7",
    )

    output = result.stdout
    match = re.search(
        r"\[train\.sh\] run_dir=(?P<run>.+/logs/distill_workflow/"
        r"(?P<stem>\d{8}-\d{6}_pytest-fresh))\n"
        r"\[train\.sh\] artifact_dir=(?P<artifact>.+/logs/distill_role_artifacts/"
        r"(?P=stem))\n",
        output,
    )
    assert match is not None
    assert not Path(match.group("run")).exists()
    assert not Path(match.group("artifact")).exists()
    assert "[train.sh] workflow_mode=fresh" in output
    assert "--algo distill" in output
    assert "workflow=g1_walk_stand" in output
    assert "[train.sh] workflow_enabled=owner-cli" in output
    assert "training.workflow.enabled=true" not in output
    assert "training.workflow.mode=fresh" in output
    assert "training.workflow.execution_mode=persistent_async" in output
    assert "algo.seed=7" in output


def test_train_sh_resume_uses_only_the_explicit_existing_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "existing-run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{}")
    artifact_dir = tmp_path / "existing-artifacts"
    artifact_dir.mkdir()

    result = _run_train_sh(
        "--workflow-mode",
        "resume",
        "--resume-run",
        str(run_dir),
        "--artifact-dir",
        str(artifact_dir),
        "training.workflow.dagger_iterations=8",
    )

    output = result.stdout
    assert "[train.sh] workflow_mode=resume" in output
    assert f"[train.sh] run_dir={run_dir}" in output
    assert f"[train.sh] artifact_dir={artifact_dir}" in output
    assert "training.workflow.mode=resume" in output
    assert f"training.workflow.run_dir={run_dir}" in output
    assert f"training.workflow.artifact_dir={artifact_dir}" in output
    assert "training.workflow.dagger_iterations=8" in output
    assert "training.workflow.execution_mode=" not in output


def test_train_sh_resume_derives_only_a_standard_paired_artifact_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "logs" / "distill_workflow" / "known-run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{}")
    artifact_dir = tmp_path / "logs" / "distill_role_artifacts" / "known-run"
    artifact_dir.mkdir(parents=True)

    result = _run_train_sh(
        "--workflow-mode",
        "resume",
        "--resume-run",
        str(run_dir),
    )

    assert f"[train.sh] run_dir={run_dir}" in result.stdout
    assert f"[train.sh] artifact_dir={artifact_dir}" in result.stdout


def test_train_sh_resume_fails_closed_without_manifest(tmp_path: Path) -> None:
    missing_manifest = tmp_path / "no-manifest"
    missing_manifest.mkdir()

    result = _run_train_sh(
        "--workflow-mode",
        "resume",
        "--resume-run",
        str(missing_manifest),
        "--artifact-dir",
        str(tmp_path),
        check=False,
    )

    assert result.returncode == 2
    assert "run_manifest.json" in result.stderr


def test_train_sh_rejects_duplicate_route_overrides() -> None:
    result = _run_train_sh(
        "--workflow-mode",
        "fresh",
        "training.workflow.mode=resume",
        check=False,
    )

    assert result.returncode == 2
    assert "route-defining Hydra override" in result.stderr
