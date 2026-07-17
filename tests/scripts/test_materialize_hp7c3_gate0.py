from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_materializer():
    path = Path("scripts/deploy/materialize_hp7c3_gate0.py")
    spec = importlib.util.spec_from_file_location("materialize_hp7c3_gate0", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_oracle_reads_materializer_identity_schema() -> None:
    module = _load_materializer()
    source = module.oracle_source()

    compile(source, "hp7c3_bounded_persistent_oracle_v6.py", "exec")
    assert 'freeze["compose"]["observed_sha256"]' in source
    assert 'freeze["compose"]["sha256"]' not in source
    assert 'freeze["hard_artifacts"]' in source
    assert 'freeze["dependency_identity"]' in source
    assert 'freeze["gpu_query"]["stdout"]' in source
    assert 'freeze["supervisor"]["sha256"]' in source
    assert "scenario weight-version identity mismatch" in source
    assert "required workflow metric stages missing" in source
    assert "cleanup metric contract mismatch" in source
    assert 'freeze["telemetry"]' in source


def test_materializer_anchors_runtime_scope_instead_of_current_head() -> None:
    module = _load_materializer()

    assert module.RUNTIME_ANCHOR_HEAD == "4fd2f67c08bb5372221ee1347561145b27238a75"
    assert module.RUNTIME_SCOPE == (
        "src",
        "scripts/train_distill.py",
        "conf/distill",
        "pyproject.toml",
        "uv.lock",
    )
    assert not hasattr(module, "EXPECTED_HEAD")


def test_supervisor_owns_exact_bounded_command_without_running_it(tmp_path: Path) -> None:
    module = _load_materializer()
    run_dir = tmp_path / "run"
    source = module.supervisor_source(tmp_path, run_dir)

    assert "uv run --no-sync train --algo distill" in source
    assert "training.workflow.execution_mode=persistent_async" in source
    assert "training.workflow.dagger_iterations=1" in source
    assert "training.workflow.dagger_updates_per_iteration=512" in source
    assert "nvidia-smi" in source
    assert "/usr/bin/time -v" in source
    assert str(run_dir) in source
