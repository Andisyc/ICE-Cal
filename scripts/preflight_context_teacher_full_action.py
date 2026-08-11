#!/usr/bin/env python3
"""Validate the formal full-action teacher route without starting training."""

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
from unilab.algos.torch.fada_context.full_action_formal_protocol import (  # noqa: E402
    FORMAL_NOMINAL_CHECKPOINT_SHA256,
    FORMAL_STRENGTH,
    FORMAL_TASK_CONFIG,
    validate_full_action_formal_training_config,
)
from unilab.algos.torch.fada_context.privileged_full_action_sac import (  # noqa: E402
    PrivilegedFullActionSACActor,
    PrivilegedFullActionSACLearner,
)
from unilab.algos.torch.offpolicy.double_buffer_runner import (  # noqa: E402
    DoubleBufferOffPolicyRunner,
)
from unilab.training import ensure_registries  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--planned-log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _compose(device: str, log_dir: Path) -> Any:
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / "offpolicy"),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=[f"task={FORMAL_TASK_CONFIG}"])
    OmegaConf.update(cfg, "training.device", str(device), merge=False)
    OmegaConf.update(cfg, "training.log_dir", str(log_dir), merge=False)
    return cfg


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    planned_log_dir = args.planned_log_dir.expanduser().resolve()
    if planned_log_dir.exists():
        raise FileExistsError(f"Planned formal log directory already exists: {planned_log_dir}")
    cfg = _compose(str(args.device), planned_log_dir)
    config = validate_full_action_formal_training_config(cfg)
    ensure_registries()
    runner = build_runner("sac", cfg)
    try:
        if not isinstance(runner, DoubleBufferOffPolicyRunner):
            raise TypeError(f"Expected DoubleBufferOffPolicyRunner, got {type(runner)}")
        if not isinstance(runner.learner, PrivilegedFullActionSACLearner):
            raise TypeError(f"Expected PrivilegedFullActionSACLearner, got {type(runner.learner)}")
        actor = runner.learner.actor
        if not isinstance(actor, PrivilegedFullActionSACActor):
            raise TypeError(f"Expected PrivilegedFullActionSACActor, got {type(actor)}")
        dimensions = {
            "obs_dim": int(runner.obs_dim),
            "critic_obs_dim": int(runner.critic_obs_dim),
            "priv_info_dim": int(actor.priv_info_dim),
            "action_dim": int(runner.action_dim),
        }
        expected = {"obs_dim": 98, "critic_obs_dim": 130, "priv_info_dim": 29, "action_dim": 29}
        if dimensions != expected:
            raise ValueError(
                f"Full-action dimensions drifted: expected {expected}, got {dimensions}"
            )
        if actor.nominal_initialization_sha256 != FORMAL_NOMINAL_CHECKPOINT_SHA256:
            raise ValueError("Nominal initialization checkpoint SHA-256 mismatch")
        optimized = {
            id(parameter)
            for group in runner.learner.actor_optimizer.param_groups
            for parameter in group["params"]
        }
        actor_parameters = {id(parameter) for parameter in actor.parameters()}
        if optimized != actor_parameters:
            raise ValueError(
                "Actor optimizer does not own exactly all full-action actor parameters"
            )
        return {
            "schema": "unilab_context_full_action_preflight_v1",
            "status": "passed",
            "training_started": False,
            "collector_started": False,
            "planned_launch": {
                "device": str(args.device),
                "log_dir": str(planned_log_dir),
                "task_config": FORMAL_TASK_CONFIG,
            },
            "config": config,
            "runtime": {
                "runner": type(runner).__name__,
                "learner": type(runner.learner).__name__,
                "actor": type(actor).__name__,
                "dimensions": dimensions,
                "strength": list(FORMAL_STRENGTH),
                "nominal_initialization_path": actor.nominal_initialization_path,
                "nominal_initialization_sha256": actor.nominal_initialization_sha256,
                "actor_parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
                "full_action_output": True,
                "residual_fusion": False,
            },
        }
    finally:
        runner.close()


def main() -> int:
    args = _parse_args()
    payload = run_preflight(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print("Training boundary: ready, not started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
