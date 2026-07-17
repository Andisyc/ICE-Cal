from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def _load_connector() -> ModuleType:
    path = Path("scripts/deploy/materialize_formal_dagger_gate0.py")
    spec = importlib.util.spec_from_file_location("materialize_formal_gate0", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(*args: str, root: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def test_connector_materializes_only_no_training_gate0_artifacts(tmp_path: Path) -> None:
    mod = _load_connector()
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", "-q", root=root)
    source = root / "workflow.py"
    source.write_text("formal runtime owner\n")
    artifacts: dict[str, str] = {}
    for name in mod.REQUIRED_HARD_ARTIFACTS:
        path = root / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = str(path)
    _git("add", "workflow.py", root=root)
    _git(
        "-c",
        "user.name=FT0 Test",
        "-c",
        "user.email=ft0@example.invalid",
        "commit",
        "-qm",
        "fixture",
        root=root,
    )
    head = _git("rev-parse", "HEAD", root=root)

    run_dir = root / "logs" / "formal_dagger_r1"
    spec_path = root / "formal_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "repo_root": str(root),
                "parent_run_dir": str(root / "parent_iteration_3"),
                "run_dir": str(run_dir),
                "parent_iteration": 3,
                "dagger_iterations": 4,
                "configured_update_floor": 512,
                "effective_updates_by_iteration": [12320, 12352, 12384, 12416],
                "seed": 0,
                "device": "cuda:0",
                "collect_num_envs": 16,
                "samples_per_role": 512,
                "batch_size": 512,
                "execution_mode": "persistent_async",
                "source_paths": {"workflow": str(source)},
                "hard_artifact_paths": artifacts,
            }
        )
    )
    observations = mod.Gate0Observations(
        head=head,
        runtime_diff_clean=True,
        compose_returncode=0,
        compose_stdout="training:\n  workflow:\n    enabled: true\n",
        compose_stderr="",
        dependency_identity={"uv_version": "uv test", "torch": "test"},
        gpu_query={"returncode": 0, "stdout": "0, GPU-test", "stderr": ""},
    )
    loaded = mod.load_materialization_spec(spec_path)
    assert loaded.identity.effective_updates_by_iteration == (
        12320,
        12352,
        12384,
        12416,
    )

    result = mod.materialize_from_spec(spec_path, observations=observations)

    assert result["accepted"] is True
    assert result["training_executed"] is False
    assert result["preflight_returncode"] == 0
    freeze = json.loads(Path(result["freeze_path"]).read_text())
    assert freeze["command"]["lineage"]["source"] == "original_parent_iteration_3"
    assert freeze["compose"]["returncode"] == 0
    assert freeze["dependency_identity"] == observations.dependency_identity
    assert freeze["gpu_query"] == observations.gpu_query
    assert Path(freeze["materialization_paths"]["compose"]).is_file()
    assert Path(freeze["materialization_paths"]["supervisor"]).is_file()
    assert Path(freeze["materialization_paths"]["oracle"]).is_file()
    assert not run_dir.exists()


def test_connector_refuses_incomplete_hard_artifact_identity(tmp_path: Path) -> None:
    mod = _load_connector()
    spec_path = tmp_path / "incomplete.json"
    spec_path.write_text(
        json.dumps(
            {
                "repo_root": str(tmp_path),
                "parent_run_dir": str(tmp_path / "parent_iteration_3"),
                "run_dir": str(tmp_path / "formal_r1"),
                "parent_iteration": 3,
                "dagger_iterations": 1,
                "configured_update_floor": 512,
                "effective_updates_by_iteration": [12320],
                "seed": 0,
                "device": "cuda:0",
                "collect_num_envs": 16,
                "samples_per_role": 512,
                "batch_size": 512,
                "execution_mode": "persistent_async",
                "source_paths": {},
                "hard_artifact_paths": {},
            }
        )
    )

    try:
        mod.load_materialization_spec(spec_path)
    except ValueError as error:
        assert "missing hard artifact identities" in str(error)
    else:
        raise AssertionError("incomplete formal identity must fail closed")


def test_connector_runtime_cleanliness_includes_untracked_owner_files() -> None:
    mod = _load_connector()

    source = inspect.getsource(mod.observe_gate0)

    assert '"status"' in source
    assert '"--porcelain"' in source
    assert '"--untracked-files=all"' in source


def test_repository_formal_two_round_spec_has_exact_reviewed_identity() -> None:
    mod = _load_connector()
    loaded = mod.load_materialization_spec(
        Path("note/distillation/plans/formal_dagger_2round_r1.spec.json")
    )

    identity = mod.build_formal_command_identity(loaded.identity)

    assert identity["training_executed"] is False
    assert identity["lineage"] == {
        "parent_iteration": 3,
        "source": "original_parent_iteration_3",
        "r6_sentinel_promoted": False,
    }
    assert identity["workload"]["dagger_iterations"] == 2
    assert identity["workload"]["effective_updates_by_iteration"] == [12320, 12352]
    assert identity["workload"]["total_effective_updates"] == 24672
    assert identity["output_paths"]["run_dir"].endswith(
        "g1_walk_stand_formal_dagger_2round_20260717_r1"
    )
