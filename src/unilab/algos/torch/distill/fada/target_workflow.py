"""Strict Stage-C dispatch between typed target-domain owners."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from unilab.algos.torch.distill.fada.adaptation_checkpoint import (
    load_fada_deployable_policy_checkpoint,
)
from unilab.algos.torch.distill.fada.target_actuator_workflow import (
    FADATargetPreflight,
    _align_paired_batches,
    _assert_identity,
    _env_override,
    _save_delta,
    _slice_target_batch,
    file_sha256,
)
from unilab.algos.torch.distill.fada.target_actuator_workflow import (
    preflight_fada_target_collection as preflight_fada_actuator_collection,
)
from unilab.algos.torch.distill.fada.target_actuator_workflow import (
    run_fada_target_collection as run_fada_actuator_collection,
)
from unilab.algos.torch.distill.fada.target_collector import collect_fada_target_windows
from unilab.algos.torch.distill.fada.target_data import save_fada_target_artifact
from unilab.algos.torch.distill.fada.target_domain import resolve_fada_target_domain
from unilab.algos.torch.distill.fada.target_slope_workflow import (
    preflight_fada_slope_collection,
    run_fada_slope_collection,
)
from unilab.training import create_env, ensure_registries

ROOT_DIR = Path(__file__).resolve().parents[6]


def preflight_fada_target_collection(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> Any:
    domain = resolve_fada_target_domain(cfg)
    if domain.kind == "slope":
        return preflight_fada_slope_collection(cfg, root_dir=root_dir)
    if domain.kind == "actuator_gain":
        return preflight_fada_actuator_collection(cfg, root_dir=root_dir)
    raise ValueError(f"unsupported FADA target domain kind: {domain.kind}")


def run_fada_target_collection(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
    load_policy_fn: Callable[..., Any] = load_fada_deployable_policy_checkpoint,
    ensure_registries_fn: Callable[[], None] = ensure_registries,
    create_env_fn: Callable[..., Any] = create_env,
    collect_fn: Callable[..., Any] = collect_fada_target_windows,
    save_fn: Callable[..., Path] = save_fada_target_artifact,
) -> dict[str, Any]:
    domain = resolve_fada_target_domain(cfg)
    if domain.kind == "slope":
        return run_fada_slope_collection(
            cfg,
            root_dir=root_dir,
            load_policy_fn=load_policy_fn,
            ensure_registries_fn=ensure_registries_fn,
            create_env_fn=create_env_fn,
            save_fn=save_fn,
        )
    if domain.kind == "actuator_gain":
        return run_fada_actuator_collection(
            cfg,
            root_dir=root_dir,
            load_policy_fn=load_policy_fn,
            ensure_registries_fn=ensure_registries_fn,
            create_env_fn=create_env_fn,
            collect_fn=collect_fn,
            save_fn=save_fn,
        )
    raise ValueError(f"unsupported FADA target domain kind: {domain.kind}")
