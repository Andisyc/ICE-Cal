from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import torch

from unilab.algos.torch.distill.data import (
    build_distillation_dataset,
    save_distillation_dataset,
)
from unilab.algos.torch.distill.formal_identity import FormalDaggerIdentitySpec


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
        workload_identity={
            "parent_rows": 851968,
            "aggregate_rows_by_iteration": [853504, 855040, 856576, 858112],
            "required_updates_by_iteration": [12320, 12352, 12384, 12416],
            "effective_updates_by_iteration": [12320, 12352, 12384, 12416],
            "total_effective_updates": 49472,
        },
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
    assert freeze["observed_workload"] == observations.workload_identity
    assert Path(freeze["materialization_paths"]["compose"]).is_file()
    assert Path(freeze["materialization_paths"]["supervisor"]).is_file()
    assert Path(freeze["materialization_paths"]["oracle"]).is_file()
    assert not run_dir.exists()


def test_connector_rejects_observed_workload_schedule_mismatch(tmp_path: Path) -> None:
    mod = _load_connector()
    expected = [12320, 12352]
    observed = [12320, 12384]

    failures = mod.validate_observed_workload(
        expected_schedule=expected,
        expected_total=24672,
        observed={
            "effective_updates_by_iteration": observed,
            "total_effective_updates": sum(observed),
        },
    )

    assert failures == [
        "effective update schedule mismatch: observed=[12320, 12384] expected=[12320, 12352]",
        "total effective updates mismatch: observed=24704 expected=24672",
    ]


def test_connector_recomputes_schedule_from_real_aggregate_labels(tmp_path: Path) -> None:
    mod = _load_connector()
    aggregate_path = tmp_path / "parent_aggregate.pt"
    dataset = build_distillation_dataset(
        torch.zeros(3, 2),
        torch.zeros(3, 2),
        scenario_labels=("walk_flat", "static_stand", "walk_to_stop"),
        transition_ages=torch.tensor([-1, -1, 0]),
        command_before=torch.zeros(3, 3),
        command_after=torch.zeros(3, 3),
    )
    save_distillation_dataset(aggregate_path, dataset)
    spec = mod.MaterializationSpec(
        identity=FormalDaggerIdentitySpec(
            repo_root=tmp_path,
            parent_run_dir=tmp_path / "parent_iteration_3",
            run_dir=tmp_path / "formal_r1",
            parent_iteration=3,
            dagger_iterations=2,
            configured_update_floor=1,
            effective_updates_by_iteration=(2, 4),
            seed=0,
            device="cuda:0",
            collect_num_envs=1,
            samples_per_role=4,
            batch_size=20,
            execution_mode="persistent_async",
        ),
        source_paths={},
        hard_artifact_paths={"parent_aggregate": aggregate_path},
    )
    compose = """
training:
  workflow:
    dagger_batch_size: 20
    dagger_balance_key: scenario
    dagger_min_transition_replay_passes: 2
    dagger_min_transition_replay_labels: [walk_to_stop]
    scenarios:
      - {name: walk_flat, quota: 0.50}
      - {name: static_stand, quota: 0.25}
      - {name: walk_to_stop, quota: 0.25}
"""

    observed = mod.compute_observed_workload(spec, compose)

    assert observed == {
        "parent_rows": 3,
        "aggregate_rows_by_iteration": [15, 27],
        "required_updates_by_iteration": [2, 4],
        "effective_updates_by_iteration": [2, 4],
        "total_effective_updates": 6,
    }


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
        Path("note/distillation/plans/formal_dagger_2round_r2.spec.json")
    )

    identity = mod.bind_hard_artifact_environment(
        mod.build_formal_command_identity(loaded.identity), loaded
    )

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
        "g1_walk_stand_formal_dagger_2round_20260717_r2"
    )


def test_connector_real_owner_compose_exits_zero() -> None:
    mod = _load_connector()
    loaded = mod.load_materialization_spec(
        Path("note/distillation/plans/formal_dagger_2round_r2.spec.json")
    )
    identity = mod.bind_hard_artifact_environment(
        mod.build_formal_command_identity(loaded.identity), loaded
    )

    result = subprocess.run(
        mod._compose_argv(identity),
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **identity["env"]},
    )

    assert result.returncode == 0, result.stderr
    assert "dagger_iterations: 2" in result.stdout
    assert "run_dir: /ssd1/cyx/UniLab/logs/distill_workflow/" in result.stdout
