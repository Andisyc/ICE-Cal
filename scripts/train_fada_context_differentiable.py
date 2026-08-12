#!/usr/bin/env python3
"""Train fault dynamics and then Context against paired healthy trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.preflight_fada_context_differentiable import (  # noqa: E402
    _compose_task,
    _nominal_override,
    _resolve,
)
from unilab.algos.torch.distill import load_fada_policy_checkpoint  # noqa: E402
from unilab.algos.torch.fada_context.differentiable_rollout import (  # noqa: E402
    trajectory_context_loss,
)
from unilab.algos.torch.fada_context.fault_dynamics import (  # noqa: E402
    FaultTransitionBatch,
    fault_dynamics_loss,
)
from unilab.algos.torch.fada_context.training_setup import (  # noqa: E402
    ContextTrainingSetupConfig,
    PreparedContextTraining,
    prepare_context_training,
)
from unilab.algos.torch.fada_context.trajectory_collector import (  # noqa: E402
    PairedTrajectoryCollectionConfig,
    collect_paired_context_trajectories,
)
from unilab.algos.torch.fada_context.trajectory_data import (  # noqa: E402
    ContextTrajectoryDataset,
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
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _load_config(path: Path, overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.load(path.expanduser().resolve())
    if not isinstance(cfg, DictConfig):
        raise TypeError("training config must be a mapping")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if not isinstance(cfg, DictConfig):
        raise TypeError("merged training config must remain a mapping")
    for name in ("batch_size", "dynamics_steps", "context_steps", "log_interval"):
        if int(getattr(cfg.training, name)) <= 0:
            raise ValueError(f"training.{name} must be positive")
    fraction = float(cfg.training.validation_fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("training.validation_fraction must be in (0, 0.5)")
    return cfg


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, allow_nan=False), flush=True)


def _select(dataset: ContextTrajectoryDataset, indices: torch.Tensor) -> ContextTrajectoryDataset:
    return ContextTrajectoryDataset(
        observation_history=dataset.observation_history[indices],
        action_history=dataset.action_history[indices],
        command=dataset.command[indices],
        healthy_reference=dataset.healthy_reference[indices],
        fault_state=dataset.fault_state[indices],
        fault_action=dataset.fault_action[indices],
        pair_id=dataset.pair_id[indices],
    )


def _split(
    dataset: ContextTrajectoryDataset, validation_fraction: float, seed: int
) -> tuple[ContextTrajectoryDataset, ContextTrajectoryDataset]:
    count = len(dataset.pair_id)
    validation_count = max(1, int(round(count * validation_fraction)))
    if validation_count >= count:
        raise ValueError("dataset must contain at least two samples")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    return _select(dataset, order[validation_count:]), _select(dataset, order[:validation_count])


def _sample(dataset: ContextTrajectoryDataset, batch_size: int) -> ContextTrajectoryDataset:
    indices = torch.randint(len(dataset.pair_id), (min(batch_size, len(dataset.pair_id)),))
    return _select(dataset, indices)


def _dynamics_validation(setup: PreparedContextTraining, dataset: ContextTrajectoryDataset, cfg: Any) -> float:
    with torch.no_grad():
        loss = fault_dynamics_loss(
            setup.dynamics,
            dataset.fault_transition_batch(),
            rollout_horizon=min(int(cfg.dynamics.rollout_horizon), dataset.fault_action.shape[1]),
            multi_step_weight=float(cfg.dynamics.multi_step_weight),
        )
    return float(loss.total)


def _context_validation(setup: PreparedContextTraining, dataset: ContextTrajectoryDataset, cfg: Any) -> float:
    with torch.no_grad():
        rollout = setup.rollout(
            dataset.observation_history,
            dataset.action_history,
            dataset.command,
            horizon=dataset.healthy_reference.shape[1],
        )
        loss = trajectory_context_loss(
            rollout,
            dataset.healthy_reference,
            latent_weight=float(cfg.context.latent_weight),
            action_smoothness_weight=float(cfg.context.action_smoothness_weight),
            uncertainty_weight=float(cfg.context.uncertainty_weight),
        )
    return float(loss.total)


def _save_checkpoint(
    path: Path,
    setup: PreparedContextTraining,
    cfg: DictConfig,
    *,
    phase: str,
    step: int,
    checkpoint_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "phase": phase,
            "step": step,
            "fada_architecture": asdict(setup.policy.config),
            "source_checkpoint_sha256": checkpoint_sha256,
            "context_config": asdict(setup.policy.context_encoder.config),
            "dynamics_config": asdict(setup.dynamics.config),
            "context_state_dict": setup.policy.context_encoder.state_dict(),
            "dynamics_state_dict": setup.dynamics.state_dict(),
            "context_optimizer_state_dict": setup.context_optimizer.state_dict(),
            "dynamics_optimizer_state_dict": setup.dynamics_optimizer.state_dict(),
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def _collect(cfg: DictConfig, policy: torch.nn.Module) -> ContextTrajectoryDataset:
    task_cfg = _compose_task(str(cfg.task_config))
    fault_override = BackendAdapter(
        task_cfg, root_dir=ROOT_DIR, algo_name="sac"
    ).build_task_env_cfg_override()
    nominal_override = _nominal_override(fault_override)
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
        result = collect_paired_context_trajectories(
            nominal_env,
            fault_env,
            policy,  # type: ignore[arg-type]
            PairedTrajectoryCollectionConfig(
                num_samples=int(cfg.collection.num_samples),
                reference_horizon=int(cfg.collection.reference_horizon),
                max_reset_batches=int(cfg.collection.max_reset_batches),
            ),
        )
    finally:
        nominal_env.close()
        fault_env.close()
    _emit(
        "collection_completed",
        samples=result.accepted_samples,
        rejected_done_samples=result.rejected_done_samples,
        reset_batches=result.reset_batches,
    )
    return result.dataset


def main() -> int:
    args = _parse_args()
    cfg = _load_config(args.config, list(args.overrides))
    apply_training_seed(int(cfg.seed), torch_runtime=True, cuda=True)
    ensure_registries()
    checkpoint_path = _resolve(ROOT_DIR, str(cfg.checkpoint_path))
    checkpoint_sha = _sha256(checkpoint_path)
    loaded = load_fada_policy_checkpoint(checkpoint_path, device=str(cfg.device))
    dataset = _collect(cfg, loaded.policy).to(str(cfg.device))
    artifact_path = _resolve(ROOT_DIR, str(cfg.collection.artifact_path))
    save_context_trajectory_dataset(
        artifact_path,
        dataset,
        loaded.policy.config,
        metadata={
            "source_checkpoint_sha256": checkpoint_sha,
            "task_config": str(cfg.task_config),
            "fault": "left_knee_actuator_strength_0.9",
            "seed": int(cfg.seed),
        },
    )
    train, validation = _split(dataset, float(cfg.training.validation_fraction), int(cfg.seed))
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
    output_dir = _resolve(ROOT_DIR, str(cfg.training.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    _emit("training_started", phase="dynamics", train_samples=len(train.pair_id))
    for step in range(1, int(cfg.training.dynamics_steps) + 1):
        batch = _sample(train, int(cfg.training.batch_size))
        setup.dynamics_optimizer.zero_grad(set_to_none=True)
        loss = fault_dynamics_loss(
            setup.dynamics,
            batch.fault_transition_batch(),
            rollout_horizon=min(int(cfg.dynamics.rollout_horizon), batch.fault_action.shape[1]),
            multi_step_weight=float(cfg.dynamics.multi_step_weight),
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            setup.dynamics.parameters(), float(cfg.training.gradient_clip_norm)
        )
        setup.dynamics_optimizer.step()
        if step == 1 or step % int(cfg.training.log_interval) == 0:
            _emit(
                "training_step",
                phase="dynamics",
                step=step,
                train_loss=float(loss.total.detach()),
                validation_loss=_dynamics_validation(setup, validation, cfg),
            )
        if step % int(cfg.training.checkpoint_interval) == 0:
            _save_checkpoint(
                output_dir / f"dynamics_{step}.pt",
                setup,
                cfg,
                phase="dynamics",
                step=step,
                checkpoint_sha256=checkpoint_sha,
            )

    setup.dynamics.eval()
    for parameter in setup.dynamics.parameters():
        parameter.requires_grad_(False)
    _emit("training_started", phase="context", train_samples=len(train.pair_id))
    for step in range(1, int(cfg.training.context_steps) + 1):
        batch = _sample(train, int(cfg.training.batch_size))
        setup.context_optimizer.zero_grad(set_to_none=True)
        rollout = setup.rollout(
            batch.observation_history,
            batch.action_history,
            batch.command,
            horizon=batch.healthy_reference.shape[1],
        )
        loss = trajectory_context_loss(
            rollout,
            batch.healthy_reference,
            latent_weight=float(cfg.context.latent_weight),
            action_smoothness_weight=float(cfg.context.action_smoothness_weight),
            uncertainty_weight=float(cfg.context.uncertainty_weight),
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            setup.policy.context_encoder.parameters(), float(cfg.training.gradient_clip_norm)
        )
        setup.context_optimizer.step()
        if step == 1 or step % int(cfg.training.log_interval) == 0:
            _emit(
                "training_step",
                phase="context",
                step=step,
                train_loss=float(loss.total.detach()),
                validation_loss=_context_validation(setup, validation, cfg),
            )
        if step % int(cfg.training.checkpoint_interval) == 0:
            _save_checkpoint(
                output_dir / f"context_{step}.pt",
                setup,
                cfg,
                phase="context",
                step=step,
                checkpoint_sha256=checkpoint_sha,
            )
    _save_checkpoint(
        output_dir / "final.pt",
        setup,
        cfg,
        phase="complete",
        step=int(cfg.training.context_steps),
        checkpoint_sha256=checkpoint_sha,
    )
    _emit("training_completed", output=str(output_dir / "final.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
