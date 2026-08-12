#!/usr/bin/env python3
"""Collect paired MuJoCo data and prepare Context training without updating parameters."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.distill import load_fada_policy_checkpoint  # noqa: E402
from unilab.algos.torch.fada_context.differentiable_rollout import (  # noqa: E402
    trajectory_context_loss,
)
from unilab.algos.torch.fada_context.fault_dynamics import (  # noqa: E402
    fault_dynamics_loss,
)
from unilab.algos.torch.fada_context.training_setup import (  # noqa: E402
    ContextTrainingSetupConfig,
    prepare_context_training,
)
from unilab.algos.torch.fada_context.trajectory_collector import (  # noqa: E402
    PairedTrajectoryCollectionConfig,
    collect_paired_context_trajectories,
)
from unilab.algos.torch.fada_context.trajectory_data import (  # noqa: E402
    load_context_trajectory_dataset,
    save_context_trajectory_dataset,
)
from unilab.training import (  # noqa: E402
    BackendAdapter,
    apply_training_seed,
    create_env,
    ensure_registries,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "conf" / "fada_context" / "differentiable_trajectory.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-list overrides, for example collection.num_samples=2",
    )
    return parser.parse_args()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compose_task(task_config: str) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"
    ):
        return compose(config_name="config", overrides=[f"task={task_config}"])


def _context_config(path: Path, overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError("Context preflight config must be a mapping")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if not isinstance(cfg, DictConfig):
        raise TypeError("merged Context preflight config must remain a mapping")
    if bool(cfg.boundary.optimizer_steps_allowed) or bool(cfg.boundary.training_started):
        raise ValueError("preflight requires optimizer_steps_allowed=false and training_started=false")
    return cfg


def _nominal_override(fault_override: dict[str, Any]) -> dict[str, Any]:
    nominal = copy.deepcopy(fault_override)
    strength = nominal["domain_rand"]["actuator_strength"]
    multipliers = list(strength["multipliers"])
    strength["multipliers"] = [1.0] * len(multipliers)
    return nominal


def _parameter_snapshot(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().cpu().clone() for parameter in module.parameters())


def _unchanged(module: torch.nn.Module, before: tuple[torch.Tensor, ...]) -> bool:
    return all(
        torch.equal(parameter.detach().cpu(), original)
        for parameter, original in zip(module.parameters(), before, strict=True)
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = args.config.expanduser().resolve()
    cfg = _context_config(cfg_path, list(args.overrides))
    checkpoint_path = _resolve(ROOT_DIR, str(cfg.checkpoint_path))
    artifact_path = _resolve(ROOT_DIR, str(cfg.collection.artifact_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"FADA checkpoint not found: {checkpoint_path}")
    apply_training_seed(int(cfg.seed), torch_runtime=True, cuda=False)
    loaded = load_fada_policy_checkpoint(checkpoint_path, device=str(cfg.device))
    task_cfg = _compose_task(str(cfg.task_config))
    fault_override = cast(
        dict[str, Any],
        BackendAdapter(task_cfg, root_dir=ROOT_DIR, algo_name="sac").build_task_env_cfg_override(),
    )
    nominal_override = _nominal_override(fault_override)
    ensure_registries()
    nominal_env = create_env(
        task_cfg,
        num_envs=int(cfg.collection.num_envs),
        env_cfg_override=nominal_override,
        sim_backend="mujoco",
    )
    fault_env = create_env(
        task_cfg,
        num_envs=int(cfg.collection.num_envs),
        env_cfg_override=fault_override,
        sim_backend="mujoco",
    )
    try:
        nominal_env.init_state()
        fault_env.init_state()
        collected = collect_paired_context_trajectories(
            nominal_env,
            fault_env,
            loaded.policy,
            PairedTrajectoryCollectionConfig(
                num_samples=int(cfg.collection.num_samples),
                reference_horizon=int(cfg.collection.reference_horizon),
                max_reset_batches=int(cfg.collection.max_reset_batches),
            ),
        )
    finally:
        nominal_env.close()
        fault_env.close()

    checkpoint_sha = _sha256(checkpoint_path)
    metadata = {
        "task_config": str(cfg.task_config),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "fault": "left_knee_actuator_strength_0.9",
        "healthy_strength": 1.0,
        "fault_strength": 0.9,
        "seed": int(cfg.seed),
        "training_started": False,
    }
    save_context_trajectory_dataset(
        artifact_path, collected.dataset, loaded.policy.config, metadata=metadata
    )
    dataset, loaded_metadata = load_context_trajectory_dataset(
        artifact_path, loaded.policy.config, map_location=str(cfg.device)
    )
    setup = prepare_context_training(
        loaded.policy,
        ContextTrainingSetupConfig(
            context_hidden_dim=int(cfg.context.hidden_dim),
            context_num_layers=int(cfg.context.num_layers),
            residual_scale=float(cfg.context.residual_scale),
            dynamics_hidden_dims=tuple(int(value) for value in cfg.dynamics.hidden_dims),
            dynamics_ensemble_size=int(cfg.dynamics.ensemble_size),
            context_learning_rate=float(cfg.context.learning_rate),
            dynamics_learning_rate=float(cfg.dynamics.learning_rate),
        ),
    )
    planner_before = _parameter_snapshot(setup.policy.planner)
    idm_before = _parameter_snapshot(setup.policy.idm)
    context_before = _parameter_snapshot(setup.policy.context_encoder)
    dynamics_before = _parameter_snapshot(setup.dynamics)

    dynamics_loss = fault_dynamics_loss(
        setup.dynamics,
        dataset.fault_transition_batch(),
        rollout_horizon=min(int(cfg.dynamics.rollout_horizon), dataset.fault_action.shape[1]),
        multi_step_weight=float(cfg.dynamics.multi_step_weight),
    )
    dynamics_loss.total.backward()
    dynamics_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in setup.dynamics.parameters()
        if parameter.grad is not None
    )
    setup.dynamics_optimizer.zero_grad(set_to_none=True)

    rollout = setup.rollout(
        dataset.observation_history,
        dataset.action_history,
        dataset.command,
        horizon=dataset.healthy_reference.shape[1],
    )
    context_loss = trajectory_context_loss(
        rollout,
        dataset.healthy_reference,
        latent_weight=float(cfg.context.latent_weight),
        action_smoothness_weight=float(cfg.context.action_smoothness_weight),
        uncertainty_weight=float(cfg.context.uncertainty_weight),
    )
    context_loss.total.backward()
    context_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in setup.policy.context_encoder.parameters()
        if parameter.grad is not None
    )
    unchanged = {
        "planner": _unchanged(setup.policy.planner, planner_before),
        "idm": _unchanged(setup.policy.idm, idm_before),
        "context": _unchanged(setup.policy.context_encoder, context_before),
        "dynamics": _unchanged(setup.dynamics, dynamics_before),
    }
    if dynamics_grad <= 0.0 or context_grad <= 0.0 or not all(unchanged.values()):
        raise RuntimeError("preflight gradient or no-update ownership contract failed")
    return {
        "schema": "unilab_fada_context_differentiable_preflight_v1",
        "status": "passed",
        "training_started": False,
        "optimizer_steps": 0,
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "dataset": {
            "path": str(artifact_path),
            "samples": len(dataset.pair_id),
            "observation_history_shape": list(dataset.observation_history.shape),
            "healthy_reference_shape": list(dataset.healthy_reference.shape),
            "fault_state_shape": list(dataset.fault_state.shape),
            "fault_action_shape": list(dataset.fault_action.shape),
            "metadata_round_trip": dict(loaded_metadata) == metadata,
            "rejected_done_samples": collected.rejected_done_samples,
        },
        "gradient_probe": {
            "dynamics_loss": float(dynamics_loss.total.detach()),
            "dynamics_grad_l1": dynamics_grad,
            "context_loss": float(context_loss.total.detach()),
            "context_grad_l1": context_grad,
            "all_parameters_unchanged": unchanged,
        },
    }


def main() -> int:
    args = _parse_args()
    payload = run_preflight(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print("Training boundary: ready, optimizer steps = 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
