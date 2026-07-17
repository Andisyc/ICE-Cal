#!/usr/bin/env python3
"""Materialize a formal DAgger FT-0 identity without executing training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unilab.algos.torch.distill.formal_identity import (
    FormalDaggerIdentitySpec,
    build_formal_command_identity,
    build_formal_freeze_document,
    build_formal_oracle_source,
    build_formal_supervisor_source,
)

REQUIRED_HARD_ARTIFACTS = frozenset(
    {
        "parent_manifest",
        "parent_checkpoint",
        "parent_aggregate",
        "walk_teacher",
        "stand_teacher",
        "walk_dataset",
        "stand_dataset",
    }
)
RUNTIME_SCOPE = ("src", "scripts/train_distill.py", "conf/distill", "pyproject.toml", "uv.lock")


@dataclass(frozen=True)
class Gate0Observations:
    """No-training observations captured before materializing the freeze."""

    head: str
    runtime_diff_clean: bool
    compose_returncode: int
    compose_stdout: str
    compose_stderr: str
    dependency_identity: dict[str, Any]
    gpu_query: dict[str, Any]


@dataclass(frozen=True)
class MaterializationSpec:
    identity: FormalDaggerIdentitySpec
    source_paths: dict[str, Path]
    hard_artifact_paths: dict[str, Path]


def load_materialization_spec(path: Path) -> MaterializationSpec:
    """Parse the reviewed JSON spec and reject incomplete formal identities."""

    payload = json.loads(path.read_text())
    hard_artifact_paths = {
        name: Path(value).resolve()
        for name, value in payload.pop("hard_artifact_paths").items()
    }
    missing = sorted(REQUIRED_HARD_ARTIFACTS - hard_artifact_paths.keys())
    if missing:
        raise ValueError(f"missing hard artifact identities: {missing}")
    source_paths = {
        name: Path(value).resolve() for name, value in payload.pop("source_paths").items()
    }
    payload["effective_updates_by_iteration"] = tuple(
        int(value) for value in payload["effective_updates_by_iteration"]
    )
    for field in ("repo_root", "parent_run_dir", "run_dir"):
        payload[field] = Path(payload[field]).resolve()
    return MaterializationSpec(
        identity=FormalDaggerIdentitySpec(**payload),
        source_paths=source_paths,
        hard_artifact_paths=hard_artifact_paths,
    )


def _compose_argv(command_identity: dict[str, Any]) -> list[str]:
    argv = list(command_identity["argv"])
    override_start = argv.index("workflow=g1_walk_stand")
    return [*argv[:override_start], "--cfg", "job", "--resolve", *argv[override_start:]]


def observe_gate0(spec: MaterializationSpec) -> Gate0Observations:
    """Collect Git, compose, dependency, and GPU facts without training."""

    root = spec.identity.repo_root
    command_identity = build_formal_command_identity(spec.identity)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    runtime_status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *RUNTIME_SCOPE,
        ],
        cwd=root,
        text=True,
    )
    compose = subprocess.run(
        _compose_argv(command_identity), cwd=root, check=False, capture_output=True, text=True
    )

    import mujoco
    import torch

    import unilab

    dependency_identity = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_path": str(Path(torch.__file__).resolve()),
        "mujoco_version": getattr(mujoco, "__version__", None),
        "mujoco_path": str(Path(mujoco.__file__).resolve()),
        "unilab_path": str(Path(unilab.__file__).resolve()),
        "uv_version": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR"),
        "UV_PROJECT_ENVIRONMENT": os.environ.get("UV_PROJECT_ENVIRONMENT"),
    }
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return Gate0Observations(
        head=head,
        runtime_diff_clean=not runtime_status.strip(),
        compose_returncode=compose.returncode,
        compose_stdout=compose.stdout,
        compose_stderr=compose.stderr,
        dependency_identity=dependency_identity,
        gpu_query={"returncode": gpu.returncode, "stdout": gpu.stdout.strip(), "stderr": gpu.stderr},
    )


def materialize_from_spec(
    spec_path: Path, *, observations: Gate0Observations | None = None
) -> dict[str, Any]:
    """Write FT-0 artifacts and run only the generated oracle preflight."""

    materialization = load_materialization_spec(spec_path)
    identity = build_formal_command_identity(materialization.identity)
    observed = observations or observe_gate0(materialization)
    paths = {name: Path(path) for name, path in identity["materialization_paths"].items()}

    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite FT-0 artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    paths["compose"].write_text(observed.compose_stdout)
    paths["supervisor"].write_text(build_formal_supervisor_source(identity))
    paths["oracle"].write_text(build_formal_oracle_source())

    sources = {
        **materialization.source_paths,
        "formal_spec": spec_path.resolve(),
        "generated_compose": paths["compose"],
        "generated_supervisor": paths["supervisor"],
        "generated_oracle": paths["oracle"],
    }
    freeze = build_formal_freeze_document(
        identity,
        repo_root=materialization.identity.repo_root,
        head=observed.head,
        source_paths=sources,
        hard_artifact_paths=materialization.hard_artifact_paths,
        runtime_diff_clean=observed.runtime_diff_clean,
    )
    extra_failures: list[str] = []
    if observed.compose_returncode != 0:
        extra_failures.append(f"Hydra compose failed: {observed.compose_returncode}")
    if observed.compose_stderr:
        extra_failures.append("Hydra compose stderr is not empty")
    if not observed.compose_stdout.strip():
        extra_failures.append("Hydra compose output is empty")
    if observed.gpu_query.get("returncode") != 0:
        extra_failures.append("GPU identity query failed")
    freeze["failures"].extend(extra_failures)
    freeze["accepted"] = not freeze["failures"]
    freeze["compose"] = {
        "path": str(paths["compose"]),
        "returncode": observed.compose_returncode,
        "stderr": observed.compose_stderr,
    }
    freeze["dependency_identity"] = observed.dependency_identity
    freeze["gpu_query"] = observed.gpu_query
    paths["freeze"].write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")

    preflight = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            str(paths["oracle"]),
            "--freeze",
            str(paths["freeze"]),
            "--result",
            str(paths["preflight"]),
            "--preflight",
        ],
        cwd=materialization.identity.repo_root,
        check=False,
    )
    preflight_payload = (
        json.loads(paths["preflight"].read_text()) if paths["preflight"].is_file() else {}
    )
    return {
        "accepted": preflight.returncode == 0 and bool(preflight_payload.get("accepted")),
        "training_executed": False,
        "freeze_path": str(paths["freeze"]),
        "preflight_path": str(paths["preflight"]),
        "preflight_returncode": preflight.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_from_spec(args.spec.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
