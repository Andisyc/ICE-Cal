"""Resolved Hydra-to-workflow plan helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.contracts.workflow import RoleArtifactSpec, WorkflowScenarioSpec

ROOT_DIR = Path(__file__).resolve().parents[6]

def _workflow_role_entries(cfg: DictConfig) -> list[dict[str, Any]]:
    entries = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.workflow.roles", default=[]),
        resolve=True,
    )
    if not isinstance(entries, list) or not entries:
        raise ValueError("training.workflow.roles must be a non-empty list")
    return [dict(cast(dict[str, Any], entry)) for entry in entries]


def _workflow_scenario_specs(
    cfg: DictConfig,
    role_names: set[str],
) -> tuple[WorkflowScenarioSpec, ...] | None:
    entries = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.workflow.scenarios", default=[]),
        resolve=True,
    )
    if entries in (None, []):
        return None
    if not isinstance(entries, list):
        raise ValueError("training.workflow.scenarios must be a list")
    specs: list[WorkflowScenarioSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("every workflow scenario entry must be a mapping")
        name = str(entry.get("name", entry.get("scenario", "")))
        source_roles = entry.get("source_roles")
        if source_roles is None and entry.get("role") not in (None, ""):
            source_roles = [entry["role"]]
        specs.append(
            WorkflowScenarioSpec(
                name=name,
                kind=str(entry.get("kind", "role")),
                source_roles=tuple(str(role) for role in (source_roles or ())),
                quota=float(entry.get("quota", 1.0)),
            )
        )
    missing = sorted({role for spec in specs for role in spec.source_roles} - set(role_names))
    if missing:
        raise ValueError(f"workflow scenarios reference unknown roles: {missing}")
    return tuple(specs)


def _workflow_path(value: Any, *, root: Path = ROOT_DIR) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _workflow_role_cfg(cfg: DictConfig, entry: dict[str, Any]) -> DictConfig:
    task = str(entry.get("task", ""))
    task_path = Path(task)
    if not task or task_path.is_absolute() or ".." in task_path.parts:
        raise ValueError(f"workflow role task must be a relative owner selector, got {task!r}")
    owner_path = ROOT_DIR / "conf" / "distill" / "task" / task_path.with_suffix(".yaml")
    if not owner_path.is_file():
        raise FileNotFoundError(f"workflow role task owner does not exist: {owner_path}")

    base = OmegaConf.load(ROOT_DIR / "conf" / "distill" / "config.yaml")
    if "defaults" in base:
        del base["defaults"]
    role_cfg = OmegaConf.merge(
        base,
        OmegaConf.load(owner_path),
        {
            "algo": OmegaConf.to_container(cfg.algo, resolve=True),
            "student": OmegaConf.to_container(cfg.student, resolve=True),
            "training": {
                "device": OmegaConf.select(cfg, "training.device"),
                "collect_num_samples": int(
                    entry.get(
                        "collect_num_samples",
                        OmegaConf.select(
                            cfg,
                            "training.workflow.collect_num_samples",
                            default=262144,
                        ),
                    )
                ),
                "collect_num_envs": int(
                    entry.get(
                        "collect_num_envs",
                        OmegaConf.select(cfg, "training.workflow.collect_num_envs", default=64),
                    )
                ),
                "collect_action_mode": "teacher_policy",
            },
            "teacher": {"checkpoint_path": str(entry.get("teacher_checkpoint_path", ""))},
        },
    )
    if not str(OmegaConf.select(role_cfg, "teacher.checkpoint_path", default="")):
        raise ValueError(f"workflow role {entry.get('role')!r} requires teacher_checkpoint_path")
    role_cfg.teacher.checkpoint_path = str(
        _workflow_path(OmegaConf.select(role_cfg, "teacher.checkpoint_path"))
    )
    if "command_sample_filter" in entry:
        role_cfg.training.collect_command_sample_filter = str(entry["command_sample_filter"])
    return cast(DictConfig, role_cfg)


def _workflow_owner_fingerprint_cfg(role_cfg: DictConfig) -> dict[str, Any]:
    return {
        "training": {
            "task_name": str(role_cfg.training.task_name),
            "sim_backend": str(role_cfg.training.sim_backend),
            "collect_action_mode": str(role_cfg.training.collect_action_mode),
            "collect_command_sample_filter": str(role_cfg.training.collect_command_sample_filter),
            "collect_target_height_info_key": OmegaConf.select(
                role_cfg, "training.collect_target_height_info_key"
            ),
            "collect_command_xy_threshold": float(role_cfg.training.collect_command_xy_threshold),
            "collect_command_yaw_threshold": float(role_cfg.training.collect_command_yaw_threshold),
        },
        "env": OmegaConf.to_container(role_cfg.env, resolve=True),
        "teacher": {
            "algo_type": str(role_cfg.teacher.algo_type),
            "obs_dim": int(role_cfg.teacher.obs_dim),
            "action_dim": int(role_cfg.teacher.action_dim),
            "actor_hidden_dim": int(role_cfg.teacher.actor_hidden_dim),
            "use_layer_norm": bool(role_cfg.teacher.use_layer_norm),
            "obs_normalization": bool(role_cfg.teacher.obs_normalization),
        },
        "student": {
            "obs_dim": int(role_cfg.student.obs_dim),
            "action_dim": int(role_cfg.student.action_dim),
        },
    }


@dataclass(frozen=True)
class WorkflowEntryPlan:
    execution_mode: str
    run_dir: Path
    role_cfgs: dict[str, DictConfig]
    role_specs: tuple[RoleArtifactSpec, ...]
    scenario_specs: tuple[WorkflowScenarioSpec, ...] | None
    mode: str
    adopt_legacy_artifacts: bool


def resolve_workflow_entry_plan(
    cfg: DictConfig,
    *,
    persistent_factory_provided: bool,
) -> WorkflowEntryPlan:
    execution_mode = str(
        OmegaConf.select(cfg, "training.workflow.execution_mode", default="legacy")
    )
    if execution_mode not in {"legacy", "persistent_async"}:
        raise ValueError(
            "training.workflow.execution_mode must be 'legacy' or "
            f"'persistent_async', got {execution_mode!r}"
        )
    if execution_mode == "legacy" and persistent_factory_provided:
        raise ValueError("legacy execution_mode forbids persistent_scenario_collector_factory")

    entries = _workflow_role_entries(cfg)
    configured_run_dir = OmegaConf.select(cfg, "training.workflow.run_dir")
    run_dir = (
        ROOT_DIR / "logs" / "distill_workflow" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if configured_run_dir in (None, "")
        else _workflow_path(configured_run_dir)
    )
    artifact_dir = _workflow_path(
        OmegaConf.select(
            cfg,
            "training.workflow.artifact_dir",
            default="logs/distill_role_artifacts",
        )
    )
    role_cfgs: dict[str, DictConfig] = {}
    specs: list[RoleArtifactSpec] = []
    for entry in entries:
        role = str(entry.get("role", ""))
        if not role:
            raise ValueError("every training.workflow.roles entry requires a role")
        role_cfg = _workflow_role_cfg(cfg, entry)
        dataset_value = entry.get("dataset_path")
        dataset_path = (
            _workflow_path(dataset_value)
            if dataset_value not in (None, "")
            else artifact_dir / f"{role}.pt"
        )
        role_cfgs[role] = role_cfg
        specs.append(
            RoleArtifactSpec(
                role=role,
                task=str(entry["task"]),
                teacher_checkpoint_path=Path(str(role_cfg.teacher.checkpoint_path)),
                dataset_path=dataset_path,
                schema_version=int(
                    OmegaConf.select(cfg, "training.workflow.schema_version", default=1)
                ),
                student_obs_dim=int(role_cfg.student.obs_dim),
                teacher_obs_dim=int(role_cfg.teacher.obs_dim),
                teacher_action_dim=int(role_cfg.teacher.action_dim),
                teacher_obs_key=str(role_cfg.training.collect_teacher_obs_key),
                teacher_projection=str(role_cfg.training.collect_teacher_projection),
                student_projection=str(role_cfg.training.collect_student_projection),
                student_drop_index=(
                    None
                    if OmegaConf.select(role_cfg, "training.collect_student_drop_index") is None
                    else int(role_cfg.training.collect_student_drop_index)
                ),
                command_sample_filter=str(role_cfg.training.collect_command_sample_filter),
                command_info_key=str(role_cfg.training.collect_command_info_key),
                command_xy_threshold=float(role_cfg.training.collect_command_xy_threshold),
                command_yaw_threshold=float(role_cfg.training.collect_command_yaw_threshold),
                owner_config=_workflow_owner_fingerprint_cfg(role_cfg),
                target_height_info_key=OmegaConf.select(
                    role_cfg, "training.collect_target_height_info_key"
                ),
            )
        )

    scenarios = _workflow_scenario_specs(cfg, {spec.role for spec in specs})
    if execution_mode == "persistent_async" and scenarios is None:
        raise ValueError("persistent_async execution_mode requires training.workflow.scenarios")
    mode = str(OmegaConf.select(cfg, "training.workflow.mode", default="auto"))
    if mode not in {"auto", "fresh", "resume", "fork"}:
        raise ValueError(f"Unsupported training.workflow.mode: {mode!r}")
    return WorkflowEntryPlan(
        execution_mode=execution_mode,
        run_dir=run_dir,
        role_cfgs=role_cfgs,
        role_specs=tuple(specs),
        scenario_specs=scenarios,
        mode=mode,
        adopt_legacy_artifacts=bool(
            OmegaConf.select(cfg, "training.workflow.adopt_legacy_artifacts", default=False)
        ),
    )
