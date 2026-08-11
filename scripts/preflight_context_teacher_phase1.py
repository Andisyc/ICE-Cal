#!/usr/bin/env python3
"""Validate the formal Phase-1 teacher route without starting training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.train_offpolicy import build_runner  # noqa: E402
from unilab.algos.torch.fada_context.formal_protocol import (  # noqa: E402
    FORMAL_NOMINAL_CHECKPOINT_SHA256,
    FORMAL_TASK_CONFIG,
    validate_phase1_formal_training_config,
)
from unilab.algos.torch.fada_context.privileged_residual_sac import (  # noqa: E402
    PrivilegedResidualSACActor,
    PrivilegedResidualSACLearner,
)
from unilab.algos.torch.offpolicy.double_buffer_runner import (  # noqa: E402
    DoubleBufferOffPolicyRunner,
)
from unilab.training import ensure_registries  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--planned-log-dir", type=Path, required=True)
    parser.add_argument("--task-config", default=FORMAL_TASK_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _compose_cfg(args: argparse.Namespace) -> Any:
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / "offpolicy"),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=[f"task={args.task_config}"])
    OmegaConf.update(cfg, "training.device", str(args.device), merge=False)
    OmegaConf.update(
        cfg,
        "training.log_dir",
        str(args.planned_log_dir.expanduser().resolve()),
        merge=False,
    )
    return cfg


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Build and inspect the formal runner without entering its lifecycle."""
    if str(args.task_config) != FORMAL_TASK_CONFIG:
        raise ValueError(
            f"formal Phase-1 preflight requires task config {FORMAL_TASK_CONFIG!r}, "
            f"got {args.task_config!r}"
        )
    planned_log_dir = args.planned_log_dir.expanduser().resolve()
    if planned_log_dir.exists():
        raise FileExistsError(
            f"planned formal training log directory already exists: {planned_log_dir}"
        )
    cfg = _compose_cfg(args)
    config_manifest = validate_phase1_formal_training_config(cfg)

    ensure_registries()
    runner = build_runner("sac", cfg)
    try:
        if not isinstance(runner, DoubleBufferOffPolicyRunner):
            raise TypeError(
                f"formal runner must be DoubleBufferOffPolicyRunner, got {type(runner)}"
            )
        if not isinstance(runner.learner, PrivilegedResidualSACLearner):
            raise TypeError(
                f"formal learner must be PrivilegedResidualSACLearner, got {type(runner.learner)}"
            )
        actor = runner.learner.actor
        if not isinstance(actor, PrivilegedResidualSACActor):
            raise TypeError(f"formal actor must be PrivilegedResidualSACActor, got {type(actor)}")
        dimensions = {
            "obs_dim": int(runner.obs_dim),
            "critic_obs_dim": int(runner.critic_obs_dim),
            "action_dim": int(runner.action_dim),
            "priv_info_dim": int(actor.priv_info_dim),
        }
        expected_dimensions = {
            "obs_dim": 98,
            "critic_obs_dim": 130,
            "action_dim": 29,
            "priv_info_dim": 29,
        }
        if dimensions != expected_dimensions:
            raise ValueError(
                f"formal Phase-1 dimensions drifted: expected {expected_dimensions}, got {dimensions}"
            )
        nominal_sha = str(actor.nominal_checkpoint_sha256)
        if nominal_sha != FORMAL_NOMINAL_CHECKPOINT_SHA256:
            raise ValueError(
                "formal nominal checkpoint SHA-256 mismatch: "
                f"expected {FORMAL_NOMINAL_CHECKPOINT_SHA256}, got {nominal_sha}"
            )
        if getattr(runner, "_collector_process", None) is not None:
            raise RuntimeError("preflight must not start a collector process")
        return {
            "schema": "unilab_context_teacher_phase1_preflight_v1",
            "status": "passed",
            "training_started": False,
            "collector_started": False,
            "planned_launch": {
                "device": str(args.device),
                "log_dir": str(planned_log_dir),
                "task_config": str(args.task_config),
            },
            "config": config_manifest,
            "runtime": {
                "runner": type(runner).__name__,
                "learner": type(runner.learner).__name__,
                "actor": type(actor).__name__,
                "sync_collection": bool(runner.sync_collection),
                "env_steps_per_sync": int(runner.env_steps_per_sync),
                "replay_prefetch_mode": str(runner.replay_prefetch_mode),
                "dimensions": dimensions,
                "nominal_checkpoint_path": str(actor.nominal_checkpoint_path),
                "nominal_checkpoint_sha256": nominal_sha,
            },
        }
    finally:
        runner.close()


def main() -> int:
    args = _parse_args()
    payload = run_preflight(args)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        _write_json_atomic(output_path, payload)
        print(f"Phase-1 preflight written to: {output_path}")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("Training boundary: ready, not started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
