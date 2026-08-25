"""Persistent UniLab collector process for FADA Planner-IDM DAgger."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.base.registry import ensure_registries
from unilab.ipc import SharedWeightSync
from unilab.training import BackendAdapter, create_env

from .async_runtime import DaggerCollectRequest, DaggerCollectResult
from .fada import FADA_COMMAND_SCENARIOS, FADAArchitectureConfig
from .fada_async_collection import FADA_ASYNC_SCENARIO, collect_fada_iteration
from .fada_async_config import (
    allocate_fada_command_scenarios,
    curriculum_and_allocations,
    fada_runtime_device,
    stand_transition_curriculum_cfg,
    standing_owner_cfg,
    teacher_spec,
    v005_replay_cfg,
)
from .fada_checkpoint import load_fada_policy_checkpoint
from .fada_collection_contract import FADACollectionSpec
from .fada_source_plan import FADAPaperSourcePlan
from .persistent_runtime import PersistentDistillationRuntime
from .teacher import load_sac_teacher_policy, reload_sac_teacher_policy_

# Compatibility aliases for callers of the former monolithic module.
_fada_runtime_device = fada_runtime_device
_teacher_spec = teacher_spec
_stand_transition_curriculum_cfg = stand_transition_curriculum_cfg
_v005_replay_cfg = v005_replay_cfg
_standing_owner_cfg = standing_owner_cfg
_curriculum_and_allocations = curriculum_and_allocations


class PersistentFADACollectorWorker:
    """Keep walking/standing environments, Oracles, and Planner-IDM student resident."""

    def __init__(
        self,
        *,
        root_dir: str,
        cfg_payload: Mapping[str, Any],
        standing_cfg_payload: Mapping[str, Any] | None,
        architecture: Mapping[str, Any],
        final_teacher_checkpoint: str,
        standing_teacher_checkpoint: str | None,
        source_allocations: Sequence[tuple[str, int]],
        initial_checkpoint_path: str,
        device: str,
        weight_sync_name: str,
        weight_sync_lock: Any,
        weight_param_shapes: Mapping[str, torch.Size],
        env_factory: Callable[..., Any] = create_env,
        teacher_loader: Callable[..., torch.nn.Module] = load_sac_teacher_policy,
        teacher_reloader: Callable[..., None] = reload_sac_teacher_policy_,
    ) -> None:
        # B1: 解析 worker-owned config 与 source identity, 产出不可变 collection 配置.
        self.weight_sync: SharedWeightSync | None = None
        self.env: Any | None = None
        self.standing_env: Any | None = None
        self.root_dir = Path(root_dir)
        self.cfg = OmegaConf.create(dict(cfg_payload))
        self.standing_cfg = (
            None if standing_cfg_payload is None else OmegaConf.create(dict(standing_cfg_payload))
        )
        self.config = FADAArchitectureConfig(**dict(architecture))
        self.device = str(device)
        self.source_allocations = tuple(
            (str(path), int(windows)) for path, windows in source_allocations
        )
        self.teacher_spec = _teacher_spec(self.cfg)
        self._teacher_loader = teacher_loader
        self._teacher_reloader = teacher_reloader
        self.intermediate_teacher: torch.nn.Module | None = None
        self.intermediate_teacher_checkpoint: str | None = None

        try:
            # B2: 恢复 student, shared-weight barrier 与 Oracle, 产出 resident policy resources.
            self.student = load_fada_policy_checkpoint(
                initial_checkpoint_path, device=self.device
            ).policy
            self.weight_sync = SharedWeightSync(
                dict(weight_param_shapes),
                create=False,
                shm_name=weight_sync_name,
                lock=weight_sync_lock,
            )
            self.local_weight_version = self.weight_sync.read_weights_into(
                self.student.state_dict()
            )
            self.final_teacher = self._teacher_loader(
                final_teacher_checkpoint,
                self.teacher_spec,
                device=self.device,
            )
            self.standing_teacher = (
                None
                if standing_teacher_checkpoint is None
                else self._teacher_loader(
                    standing_teacher_checkpoint,
                    self.teacher_spec,
                    device=self.device,
                )
            )

            # A spawned worker starts with an empty registry. FADA owns the G1 route, so
            # importing unrelated optional robot families here would make their extras
            # accidental hard dependencies of Planner-IDM training.
            # B3: 创建 walking/standing environment, 产出 collect() 使用的 resident env owners.
            ensure_registries(packages=("unilab.envs.locomotion.g1",))
            env_override = BackendAdapter(
                self.cfg,
                root_dir=self.root_dir,
                algo_name="distill",
            ).build_task_env_cfg_override()
            fada_cfg = self.cfg.training.fada
            self.env = env_factory(
                self.cfg,
                num_envs=int(fada_cfg.num_envs),
                env_cfg_override=env_override,
                sim_backend=str(self.cfg.training.sim_backend),
                task_name=str(self.cfg.training.task_name),
            )
            if self.standing_cfg is not None:
                standing_override = BackendAdapter(
                    self.standing_cfg,
                    root_dir=self.root_dir,
                    algo_name="distill",
                ).build_task_env_cfg_override()
                self.standing_env = env_factory(
                    self.standing_cfg,
                    num_envs=int(fada_cfg.num_envs),
                    env_cfg_override=standing_override,
                    sim_backend=str(self.standing_cfg.training.sim_backend),
                    task_name=str(self.standing_cfg.training.task_name),
                )

            # B4: 物理包线守卫 opt-in, student 驱动的极端/非有限物理状态在回流
            # 下一个原生 step 之前被 sanitize + 标记 terminated, 防止原生层 SIGSEGV.
            physics_guard_max_abs = float(
                OmegaConf.select(fada_cfg, "physics_guard_max_abs", default=1.0e4)
            )
            for resident_env in (self.env, self.standing_env):
                if resident_env is not None:
                    resident_env.set_physics_envelope_guard(physics_guard_max_abs)
        except BaseException:
            self._close_resources(raise_errors=False)
            raise

    def _close_resources(self, *, raise_errors: bool) -> None:
        """Close every materialized resident resource once, including partial construction."""

        errors: list[BaseException] = []
        environments = (self.env, self.standing_env)
        closed_ids: set[int] = set()
        for environment in environments:
            if environment is None or id(environment) in closed_ids:
                continue
            close = getattr(environment, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    errors.append(exc)
            closed_ids.add(id(environment))
        self.env = None
        self.standing_env = None
        self.intermediate_teacher = None
        self.intermediate_teacher_checkpoint = None
        if self.weight_sync is not None:
            try:
                self.weight_sync.close()
            except BaseException as exc:
                errors.append(exc)
            self.weight_sync = None
        if errors and raise_errors:
            raise errors[0]

    def _collection_spec(self) -> FADACollectionSpec:
        fada_cfg = self.cfg.training.fada
        command_keys = OmegaConf.to_container(fada_cfg.command_info_keys, resolve=True)
        if not isinstance(command_keys, list) or not command_keys:
            raise ValueError("training.fada.command_info_keys must be a non-empty list")
        return FADACollectionSpec(
            observation_key=str(fada_cfg.observation_key),
            teacher_projection=str(fada_cfg.teacher_projection),
            student_projection=str(fada_cfg.student_projection),
            student_drop_index=OmegaConf.select(fada_cfg, "student_drop_index"),
            command_info_keys=tuple(str(key) for key in command_keys),
            max_env_steps=OmegaConf.select(fada_cfg, "max_env_steps"),
        )

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        return collect_fada_iteration(self, request)

    def close(self) -> None:
        self._close_resources(raise_errors=True)


def _build_persistent_fada_worker(**kwargs: Any) -> PersistentFADACollectorWorker:
    return PersistentFADACollectorWorker(**kwargs)


def build_persistent_fada_runtime(
    *,
    cfg: DictConfig,
    architecture: FADAArchitectureConfig,
    paper_source_plan: FADAPaperSourcePlan,
    final_teacher_checkpoint: str | Path,
    request_timeout_seconds: float,
) -> PersistentDistillationRuntime:
    """Build the FADA-specific UniLab persistent collector runtime."""

    root_dir = Path(__file__).resolve().parents[5]
    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_payload, dict):
        raise ValueError("composed FADA config must resolve to a mapping")

    def load_student(path: Path) -> torch.nn.Module:
        return load_fada_policy_checkpoint(path, device="cpu").policy

    curriculum, _ = _curriculum_and_allocations(cfg.training.fada, architecture)
    standing_teacher_checkpoint: str | None = None
    standing_cfg_payload: dict[str, Any] | None = None
    if bool(curriculum.enabled):
        configured = OmegaConf.select(curriculum, "standing_teacher_checkpoint_path")
        if configured in (None, ""):
            raise ValueError(
                "training.fada.stand_transition_curriculum.standing_teacher_checkpoint_path "
                "is required when the curriculum is enabled"
            )
        standing_path = Path(str(configured)).expanduser().resolve()
        if not standing_path.is_file():
            raise FileNotFoundError(f"standing Oracle checkpoint does not exist: {standing_path}")
        standing_teacher_checkpoint = str(standing_path)
        standing_cfg = _standing_owner_cfg(
            root_dir=root_dir,
            cfg=cfg,
            task_selector=str(curriculum.standing_task),
        )
        resolved_standing_cfg = OmegaConf.to_container(standing_cfg, resolve=True)
        if not isinstance(resolved_standing_cfg, dict):
            raise ValueError("standing FADA owner config must resolve to a mapping")
        standing_cfg_payload = cast(dict[str, Any], resolved_standing_cfg)

    return PersistentDistillationRuntime(
        student_loader=load_student,
        worker_factory=_build_persistent_fada_worker,
        worker_kwargs={
            "root_dir": str(root_dir),
            "cfg_payload": cfg_payload,
            "standing_cfg_payload": standing_cfg_payload,
            "architecture": asdict(architecture),
            "final_teacher_checkpoint": str(Path(final_teacher_checkpoint).resolve()),
            "standing_teacher_checkpoint": standing_teacher_checkpoint,
            "source_allocations": [
                (str(path.resolve()), windows)
                for path, windows in paper_source_plan.source_allocations
            ],
            "device": _fada_runtime_device(cfg),
        },
        worker_kwargs_factory=lambda checkpoint_path: {
            "initial_checkpoint_path": str(checkpoint_path.resolve())
        },
        request_timeout_seconds=float(request_timeout_seconds),
    )


__all__ = [
    "FADA_COMMAND_SCENARIOS",
    "FADA_ASYNC_SCENARIO",
    "PersistentFADACollectorWorker",
    "allocate_fada_command_scenarios",
    "build_persistent_fada_runtime",
]
