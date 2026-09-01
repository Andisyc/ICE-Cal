"""Run the bounded v007 FADA source-coverage discriminator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.async_config import teacher_spec
from unilab.algos.torch.distill.fada.oracle import load_fada_oracle_policy
from unilab.algos.torch.distill.fada_collector import FADACollectionSpec
from unilab.algos.torch.distill.fada_source_diagnostics import run_fada_coverage_diagnostic
from unilab.algos.torch.distill.fada_training import load_fada_policy_checkpoint
from unilab.base.registry import ensure_registries
from unilab.training import BackendAdapter, create_env

ROOT_DIR = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="g1_walk_flat/mujoco_fada_privileged_planner")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--command", type=float, nargs=3, default=(0.4, 0.0, 0.0))
    parser.add_argument("--override", action="append", default=[])
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkout_identity() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
    ).stdout
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    bound_sources = (
        ROOT_DIR / "scripts" / "diagnose_fada_source_coverage.py",
        ROOT_DIR / "src" / "unilab" / "algos" / "torch" / "distill" / "fada_source_diagnostics.py",
        ROOT_DIR / "src" / "unilab" / "algos" / "torch" / "distill" / "fada_collector.py",
    )
    return {
        "head": revision,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_bytes": len(diff),
        "status": status,
        "bound_source_sha256": {
            str(path.relative_to(ROOT_DIR)): _file_sha256(path) for path in bound_sources
        },
    }


def _compose_cfg(task: str, overrides: list[str]) -> Any:
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "distill"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[f"task={task}", "training.play_only=true", *overrides],
        )


def main() -> int:
    args = _parser().parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FADA checkpoint does not exist: {checkpoint}")
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(f"walking Oracle checkpoint does not exist: {teacher_checkpoint}")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.max_steps > 500:
        raise ValueError("--max-steps must not exceed 500")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = _compose_cfg(args.task, list(args.override))
    loaded = load_fada_policy_checkpoint(checkpoint, device=args.device)
    config = loaded.policy.config
    fada_cfg = cfg.training.fada
    teacher = load_fada_oracle_policy(
        teacher_checkpoint,
        teacher_spec(cfg),
        device=args.device,
    )
    ensure_registries(packages=("unilab.envs.locomotion.g1",))
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="distill"
    ).build_play_env_cfg_override()
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=env_override,
        sim_backend=str(cfg.training.sim_backend),
        task_name=str(cfg.training.task_name),
    )
    try:
        prepare_shadow = getattr(env, "prepare_isolated_rollout_branch", None)
        if not callable(prepare_shadow):
            raise TypeError(
                "coverage diagnostic requires env.prepare_isolated_rollout_branch()"
            )
        prepare_shadow()
        report = run_fada_coverage_diagnostic(
            env,
            student_policy=loaded.policy,
            teacher_policy=teacher,
            config=config,
            command=args.command,
            max_steps=args.max_steps,
            spec=FADACollectionSpec(
                observation_key=str(fada_cfg.observation_key),
                teacher_projection=str(fada_cfg.teacher_projection),
                student_projection=str(fada_cfg.student_projection),
                student_drop_index=getattr(fada_cfg, "student_drop_index", None),
                command_info_keys=tuple(str(key) for key in fada_cfg.command_info_keys),
                collect_oracle_shadow=True,
            ),
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    payload = {
        "schema_version": "fada-source-coverage-diagnostic/v1",
        "identity": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_checkpoint_sha256": _file_sha256(teacher_checkpoint),
            "task": args.task,
            "sim_backend": str(cfg.training.sim_backend),
            "seed": args.seed,
            "command": [float(item) for item in args.command],
            "max_steps": args.max_steps,
            "overrides": list(args.override),
            "architecture": asdict(config),
            "checkout": _checkout_identity(),
            "effective_config": OmegaConf.to_container(cfg, resolve=True),
            "environment_override": OmegaConf.to_container(
                OmegaConf.create(env_override), resolve=True
            ),
        },
        "report": asdict(report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "verdict": report.verdict,
                "failure_reproduced": report.failure_reproduced,
                "identity_valid": report.identity_valid,
                "steps": len(report.steps),
                "coverage_gap_step_indices": report.coverage_gap_step_indices,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if report.verdict == "COVERAGE_GAP" else 2


if __name__ == "__main__":
    raise SystemExit(main())
