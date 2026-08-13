from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.support_query_collector import (
    SupportQueryCollectionConfig,
    SupportQueryCollectionResult,
    collect_support_query_pairs,
)
from unilab.training import BackendAdapter, apply_training_seed, create_env, ensure_registries


def resolve_repo_path(root_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root_dir / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_support_query_config(
    path: Path,
    overrides: list[str],
    *,
    preflight: bool,
) -> DictConfig:
    cfg = OmegaConf.load(path.expanduser().resolve())
    if not isinstance(cfg, DictConfig):
        raise TypeError("Support-Query config must be a mapping")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if not isinstance(cfg, DictConfig):
        raise TypeError("merged Support-Query config must remain a mapping")
    positive = {
        "collection.num_envs": cfg.collection.num_envs,
        "collection.num_pairs": cfg.collection.num_pairs,
        "collection.support_length": cfg.collection.support_length,
        "collection.query_length": cfg.collection.query_length,
        "collection.max_reset_pairs": cfg.collection.max_reset_pairs,
        "context.hidden_dim": cfg.context.hidden_dim,
        "context.num_layers": cfg.context.num_layers,
        "training.batch_size": cfg.training.batch_size,
        "training.steps": cfg.training.steps,
    }
    for name, value in positive.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < float(cfg.training.validation_fraction) < 0.5:
        raise ValueError("training.validation_fraction must be in (0, 0.5)")
    if preflight and (
        bool(cfg.boundary.optimizer_steps_allowed) or bool(cfg.boundary.training_started)
    ):
        raise ValueError(
            "preflight requires boundary.optimizer_steps_allowed=false and "
            "boundary.training_started=false"
        )
    return cfg


def _compose_task(root_dir: Path, task_config: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(root_dir / "conf" / "offpolicy"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"task={task_config}"])


def _fixed_fault_override(root_dir: Path, task_cfg: DictConfig) -> dict[str, Any]:
    override = cast(
        dict[str, Any],
        BackendAdapter(task_cfg, root_dir=root_dir, algo_name="sac").build_task_env_cfg_override(),
    )
    strength = override["domain_rand"]["actuator_strength"]
    multipliers = [float(value) for value in strength["multipliers"]]
    expected = [1.0] * 29
    expected[3] = 0.7
    if not bool(strength["enabled"]) or multipliers != expected:
        raise ValueError(
            "formal Support-Query fault must be exactly left-knee index 3 strength 0.7"
        )
    if bool(override["control_config"].get("simulate_action_latency", False)):
        raise ValueError("Support-Query collection requires simulate_action_latency=false")
    expected_command = [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]
    if override["commands"]["vel_limit"] != expected_command:
        raise ValueError("formal Support-Query command must be fixed straight-line [0.4, 0.0, 0.0]")
    return override


def parameter_snapshot(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().cpu().clone() for parameter in module.parameters())


def parameters_equal(module: torch.nn.Module, snapshot: tuple[torch.Tensor, ...]) -> bool:
    return all(
        torch.equal(parameter.detach().cpu(), original)
        for parameter, original in zip(module.parameters(), snapshot, strict=True)
    )


def collect_fixed_fault_support_query(
    root_dir: Path,
    cfg: DictConfig,
    policy: FADAPlannerIDMPolicy,
) -> SupportQueryCollectionResult:
    task_cfg = _compose_task(root_dir, str(cfg.task_config))
    fault_override = _fixed_fault_override(root_dir, task_cfg)
    ensure_registries()
    env = create_env(
        task_cfg,
        num_envs=int(cfg.collection.num_envs),
        env_cfg_override=fault_override,
        sim_backend="mujoco",
    )
    try:
        env.init_state()
        return collect_support_query_pairs(
            env,
            policy,
            SupportQueryCollectionConfig(
                num_pairs=int(cfg.collection.num_pairs),
                support_length=int(cfg.collection.support_length),
                query_length=int(cfg.collection.query_length),
                max_reset_pairs=int(cfg.collection.max_reset_pairs),
            ),
        )
    finally:
        env.close()


def create_fixed_fault_paired_environments(
    root_dir: Path,
    cfg: DictConfig,
    *,
    num_envs: int,
    seed: int,
) -> tuple[Any, Any]:
    """Create exact-seed healthy/fault environments differing only in actuator strength."""

    if num_envs <= 0:
        raise ValueError("paired environment count must be positive")
    task_cfg = _compose_task(root_dir, str(cfg.task_config))
    fault_override = _fixed_fault_override(root_dir, task_cfg)
    healthy_override = copy.deepcopy(fault_override)
    healthy_override["domain_rand"]["actuator_strength"]["multipliers"] = [1.0] * 29
    ensure_registries()

    apply_training_seed(seed, torch_runtime=True, cuda=True)
    healthy_env = create_env(
        task_cfg,
        num_envs=num_envs,
        env_cfg_override=healthy_override,
        sim_backend="mujoco",
    )
    fault_env: Any | None = None
    try:
        healthy_env.init_state()
        apply_training_seed(seed, torch_runtime=True, cuda=True)
        fault_env = create_env(
            task_cfg,
            num_envs=num_envs,
            env_cfg_override=fault_override,
            sim_backend="mujoco",
        )
        fault_env.init_state()
    except Exception:
        if fault_env is not None:
            fault_env.close()
        healthy_env.close()
        raise
    assert fault_env is not None
    return healthy_env, fault_env
