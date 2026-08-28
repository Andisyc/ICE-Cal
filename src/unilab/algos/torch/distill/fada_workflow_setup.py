"""Hydra-to-runtime validation and dependency setup for FADA training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import torch
from omegaconf import DictConfig, OmegaConf

from .fada import FADAArchitectureConfig
from .fada_async_config import fada_training_schedule
from .fada_async_runtime import allocate_fada_command_scenarios
from .fada_observation import (
    FADA_G1_STATE_OBSERVATION_CONTRACT,
    assert_fada_active_route_contract,
    assert_fada_projection_matches_contract,
)
from .fada_oracle import load_fada_oracle_policy as _default_load_fada_oracle_policy
from .fada_source_plan import FADAPaperSourcePlan, build_fada_paper_source_plan

ROOT_DIR = Path(__file__).resolve().parents[5]


@dataclass(frozen=True)
class FADAWorkflowDependencies:
    """Composition-root dependencies used by the FADA training owner."""

    require_teacher_policy_collection_route: Callable[[DictConfig], None]
    apply_collect_command_distribution_overrides: Callable[[DictConfig], Mapping[str, Any]]
    resolve_teacher_checkpoint: Callable[..., tuple[Path | None, Path | None]]
    build_teacher_spec: Callable[[DictConfig], Any]
    build_persistent_fada_runtime: Callable[..., Any]
    ensure_registries: Callable[[], None]
    create_env: Callable[..., Any]
    backend_adapter_cls: Callable[..., Any]
    load_fada_oracle_policy: Callable[..., torch.nn.Module] = _default_load_fada_oracle_policy


def _distill_device(cfg: DictConfig) -> str:
    device = OmegaConf.select(cfg, "training.device", default="cpu")
    return "cpu" if device in (None, "") else str(device)


def build_fada_architecture_config(cfg: DictConfig) -> FADAArchitectureConfig:
    """Translate the single Hydra FADA owner into the paper architecture contract."""

    fada = cfg.training.fada
    return FADAArchitectureConfig(
        obs_dim=int(OmegaConf.select(fada, "obs_dim", default=cfg.student.obs_dim)),
        action_dim=int(cfg.student.action_dim),
        command_dim=int(fada.command_dim),
        observation_contract=str(
            OmegaConf.select(
                fada,
                "observation_contract",
                default="legacy_actor_obs_v1",
            )
        ),
        history_length=int(fada.history_length),
        prediction_horizon=int(fada.prediction_horizon),
        hidden_dim=int(fada.hidden_dim),
        num_heads=int(fada.num_heads),
        planner_layers=int(fada.planner_layers),
        idm_encoder_layers=int(fada.idm_encoder_layers),
        idm_decoder_layers=int(fada.idm_decoder_layers),
        feedforward_dim=int(fada.feedforward_dim),
        dropout=float(fada.dropout),
    )


def assert_fada_training_run_contract(cfg: DictConfig) -> None:
    """Validate one fresh v011 alternating run before runtime or persistence mutation."""

    fada_cfg = cfg.training.fada
    if OmegaConf.select(fada_cfg, "phase", default=None) is not None:
        raise ValueError("v011 removed training.fada.phase; one run alternates IDM then Planner")
    if "pretrained_idm_path" in fada_cfg:
        raise ValueError("v011 removed training.fada.pretrained_idm_path")
    if any(
        OmegaConf.select(fada_cfg, name, default=None) not in (None, "")
        for name in ("resume_path", "initial_weights_path")
    ):
        raise ValueError("v010 resume_path and initial_weights_path must both be null")
    paper_source_enabled = bool(OmegaConf.select(fada_cfg, "paper_source_enabled", default=False))
    training_schedule = fada_training_schedule(fada_cfg)
    checkpoint = _fada_path(
        OmegaConf.select(fada_cfg, "checkpoint_path"),
        field_name="training.fada.checkpoint_path",
        required=True,
    )
    if checkpoint is None:
        raise RuntimeError("FADA output checkpoint path was not materialized")
    if checkpoint.expanduser().exists():
        raise FileExistsError(
            f"v010 requires a fresh output checkpoint path, found existing: {checkpoint}"
        )
    if not paper_source_enabled:
        raise ValueError("v011 alternating training requires paper_source_enabled=true")
    if training_schedule == "idm_pretrain":
        if str(OmegaConf.select(fada_cfg, "execution_mode")) != "persistent_async":
            raise ValueError("IDM pretraining requires execution_mode='persistent_async'")
        if int(OmegaConf.select(fada_cfg, "planner_updates", default=-1)) != 0:
            raise ValueError("IDM pretraining requires planner_updates=0")


def assert_fada_source_route_contract(
    cfg: DictConfig,
    config: FADAArchitectureConfig,
) -> None:
    """Close projection and fresh-start identity before source lifecycle mutation."""

    fada_cfg = cfg.training.fada
    projection = str(OmegaConf.select(fada_cfg, "student_projection"))
    if OmegaConf.select(fada_cfg, "observation_contract", default=None) is None:
        # Compatibility for historical programmatic test fixtures without an identity field.
        assert_fada_projection_matches_contract(
            observation_contract=config.observation_contract,
            projection=projection,
        )
    else:
        assert_fada_active_route_contract(
            observation_contract=config.observation_contract,
            projection=projection,
        )
    if config.observation_contract == FADA_G1_STATE_OBSERVATION_CONTRACT and any(
        OmegaConf.select(fada_cfg, name) not in (None, "")
        for name in ("initial_weights_path", "resume_path")
    ):
        raise ValueError(
            "g1_fada_state_v2 source training requires fresh initialization; "
            "initial_weights_path and resume_path must both be null"
        )


def _fada_execution_mode(cfg: DictConfig) -> str:
    fada_cfg = cfg.training.fada
    execution_mode = str(OmegaConf.select(fada_cfg, "execution_mode", default="legacy"))
    if execution_mode not in {"legacy", "persistent_async"}:
        raise ValueError(
            "training.fada.execution_mode must be 'legacy' or 'persistent_async', "
            f"got {execution_mode!r}"
        )
    curriculum_enabled = bool(
        OmegaConf.select(
            fada_cfg,
            "stand_transition_curriculum.enabled",
            default=False,
        )
    )
    if curriculum_enabled and execution_mode != "persistent_async":
        raise ValueError(
            "training.fada.stand_transition_curriculum requires "
            "training.fada.execution_mode=persistent_async"
        )
    v005_enabled = bool(OmegaConf.select(fada_cfg, "v005_replay.enabled", default=False))
    if v005_enabled and not curriculum_enabled:
        raise ValueError("training.fada.v005_replay requires stand_transition_curriculum.enabled")
    if v005_enabled and execution_mode != "persistent_async":
        raise ValueError(
            "training.fada.v005_replay requires training.fada.execution_mode=persistent_async"
        )
    return execution_mode


def _fada_path(value: Any, *, field_name: str, required: bool) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"{field_name} must be set when training.fada.enabled=true")
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT_DIR / path


def _paper_source_plan(cfg: DictConfig) -> FADAPaperSourcePlan:
    """Resolve Hydra paths and delegate Appendix B.2 rules to the FADA owner."""

    fada = cfg.training.fada
    enabled = bool(OmegaConf.select(fada, "paper_source_enabled", default=False))
    if not enabled:
        return FADAPaperSourcePlan(enabled=False, source_allocations=())
    # B1: composition root resolves repository-relative paths into one explicit source namespace.
    raw_value = OmegaConf.select(fada, "intermediate_oracle_checkpoint_paths", default=[])
    raw = (
        OmegaConf.to_container(raw_value, resolve=True)
        if OmegaConf.is_config(raw_value)
        else raw_value
    )
    if not isinstance(raw, list):
        raise ValueError("intermediate_oracle_checkpoint_paths must be a list")
    paths = tuple(
        path
        for value in raw
        if (path := _fada_path(value, field_name="intermediate oracle checkpoint", required=True))
        is not None
    )
    # B2: FADA owner validates paper constants and returns the sealed allocation.
    return build_fada_paper_source_plan(
        enabled=enabled,
        oracle_shadow_enabled=bool(OmegaConf.select(fada, "oracle_shadow_enabled", default=False)),
        checkpoint_paths=paths,
        configured_checkpoint_count=int(
            OmegaConf.select(fada, "intermediate_oracle_count", default=20)
        ),
        suboptimal_data_ratio=float(OmegaConf.select(fada, "suboptimal_data_ratio", default=0.0)),
        optimal_windows=int(OmegaConf.select(fada, "windows_per_iteration")),
        resume_path=OmegaConf.select(fada, "resume_path"),
    )


def _fada_v005_replay_settings(
    fada_cfg: DictConfig,
    *,
    batch_size: int,
) -> tuple[bool, dict[str, float], float, float]:
    enabled = bool(OmegaConf.select(fada_cfg, "v005_replay.enabled", default=False))
    ratios_cfg = OmegaConf.select(fada_cfg, "v005_replay.planner_scenario_ratios")
    ratios_value = (
        OmegaConf.to_container(ratios_cfg, resolve=True)
        if OmegaConf.is_config(ratios_cfg)
        else ratios_cfg
    )
    if ratios_value is not None and not isinstance(ratios_value, Mapping):
        raise ValueError("v005 planner_scenario_ratios must be a mapping")
    ratio_mapping = cast(
        Mapping[str, Any],
        ratios_value or {"walk": 0.5, "static_stand": 0.25, "walk_to_stand": 0.25},
    )
    ratios = {str(name): float(value) for name, value in ratio_mapping.items()}
    walk_cold_ratio = float(
        OmegaConf.select(fada_cfg, "v005_replay.walk_cold_start_ratio", default=0.5)
    )
    static_cold_ratio = float(
        OmegaConf.select(fada_cfg, "v005_replay.static_cold_start_ratio", default=0.5)
    )
    if not enabled:
        return False, ratios, walk_cold_ratio, static_cold_ratio
    expected_ratios = {"walk": 0.5, "static_stand": 0.25, "walk_to_stand": 0.25}
    if set(ratios) != set(expected_ratios) or any(
        not math.isclose(ratios[name], expected, rel_tol=0.0, abs_tol=1.0e-12)
        for name, expected in expected_ratios.items()
    ):
        raise ValueError(
            "v005 Planner scenario ratios are fixed at "
            "walk/static_stand/walk_to_stand=0.5/0.25/0.25"
        )
    if not math.isclose(walk_cold_ratio, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("v005 walk_cold_start_ratio is fixed at 0.5")
    if not math.isclose(static_cold_ratio, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("v005 static_cold_start_ratio is fixed at 0.5")
    allocations = dict(allocate_fada_command_scenarios(int(batch_size), ratios))
    walk_count = int(allocations.get("walk", 0))
    static_count = int(allocations.get("static_stand", 0))
    if walk_count < 2 or static_count < 2:
        raise ValueError("v005 Planner batch must allocate walk/static cold-start and steady rows")
    return True, ratios, walk_cold_ratio, static_cold_ratio


# Public owner names; legacy underscored names remain on the composition facade.


distill_device = _distill_device
fada_execution_mode = _fada_execution_mode
resolve_fada_path = _fada_path
paper_source_plan = _paper_source_plan
fada_v005_replay_settings = _fada_v005_replay_settings
