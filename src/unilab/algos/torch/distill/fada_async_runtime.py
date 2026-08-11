"""Persistent UniLab collector process for FADA Planner-IDM DAgger."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, cast

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.base.registry import ensure_registries
from unilab.ipc import SharedWeightSync
from unilab.training import BackendAdapter, create_env

from .async_runtime import DaggerCollectRequest, DaggerCollectResult
from .fada import (
    FADA_COMMAND_SCENARIOS,
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
    FADASourceBatch,
)
from .fada_collector import (
    FADACollectionResult,
    FADACollectionSpec,
    collect_fada_source_windows,
)
from .fada_training import (
    FADAPaperSourcePlan,
    load_fada_policy_checkpoint,
    save_fada_source_batch,
)
from .persistent_runtime import PersistentDistillationRuntime
from .teacher import DistillationTeacherSpec, load_sac_teacher_policy

FADA_ASYNC_SCENARIO = "fada_iteration"


def _fada_runtime_device(cfg: DictConfig) -> str:
    configured = OmegaConf.select(cfg, "training.device", default="cpu")
    return "cpu" if configured in (None, "") else str(configured)


def allocate_fada_command_scenarios(
    total_windows: int,
    ratios: Mapping[str, float],
) -> tuple[tuple[str, int], ...]:
    """按稳定最大余数法产出总数精确的 scenario window 配额.

    函数名说明:
        该 helper 只拥有配额整数化, 不决定 command 或 Oracle 语义.

    主链路:
        上游: persistent FADA worker 的 curriculum config.
        下游: 每轮 scenario collector 调用和 artifact 配额校验.

    语义:
        ratios 必须有限、非负且和为 1; 每个正比例 scenario 至少获得一个 window.
    """

    # B1: 校验总预算和比例域, 产出固定顺序的 raw quota.
    if int(total_windows) <= 0:
        raise ValueError(f"total_windows must be positive, got {total_windows}")
    unknown = set(ratios) - set(FADA_COMMAND_SCENARIOS)
    if unknown:
        raise ValueError(f"unknown FADA command scenarios: {sorted(unknown)}")
    values = [float(ratios.get(name, 0.0)) for name in FADA_COMMAND_SCENARIOS]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("FADA command scenario ratios must be finite and non-negative")
    if abs(sum(values) - 1.0) > 1.0e-6:
        raise ValueError(f"FADA command scenario ratios must sum to 1, got {sum(values)}")
    positive_count = sum(value > 0.0 for value in values)
    if int(total_windows) < positive_count:
        raise ValueError(
            "total_windows must give every positive-ratio scenario at least one window: "
            f"total={total_windows} positive_scenarios={positive_count}"
        )

    # B2: 用稳定最大余数整数化, 产出和 total_windows 完全一致的 counts.
    raw = [int(total_windows) * value for value in values]
    counts = [int(value) for value in raw]
    for index, value in enumerate(values):
        if value > 0.0 and counts[index] == 0:
            counts[index] = 1
    while sum(counts) > int(total_windows):
        candidates = [index for index, count in enumerate(counts) if count > 1]
        if not candidates:
            raise ValueError("unable to preserve positive FADA scenario allocations")
        index = min(candidates, key=lambda item: (raw[item] - counts[item], -item))
        counts[index] -= 1
    while sum(counts) < int(total_windows):
        index = max(
            range(len(counts)),
            key=lambda item: (raw[item] - counts[item], -item),
        )
        counts[index] += 1
    # B3: 删除零配额 scenario, 产出 worker 可直接执行的有序 allocation.
    return tuple(
        (name, count)
        for name, count in zip(FADA_COMMAND_SCENARIOS, counts, strict=True)
        if count > 0
    )


def _concat_source_batches(
    batches: Sequence[FADASourceBatch],
    config: FADAArchitectureConfig,
) -> FADASourceBatch:
    if not batches:
        raise ValueError("FADA async collector produced no source batches")
    return FADASourceBatch(
        **{
            field: torch.cat([getattr(batch, field) for batch in batches], dim=0)
            for field in FADASourceBatch.__dataclass_fields__
        }
    ).validate(config)


def _summary(
    collection: FADACollectionResult,
    *,
    iteration: int,
    source: str,
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "iteration": int(iteration),
        "source": source,
        "rollout_mode": collection.rollout_mode,
        "windows": int(collection.batch.command.shape[0]),
        "env_steps": int(collection.env_steps),
        "rejected_done_transitions": int(collection.rejected_done_transitions),
        "rejected_command_windows": int(collection.rejected_command_windows),
        "rejected_scenario_windows": int(collection.rejected_scenario_windows),
        "command_scenario": collection.command_scenario,
        "oracle_role": collection.oracle_role,
        "window_profile": collection.window_profile,
    }
    if source_checkpoint is not None:
        result["source_checkpoint"] = source_checkpoint
    return result


def _teacher_spec(cfg: DictConfig) -> DistillationTeacherSpec:
    algo_type = str(cfg.teacher.algo_type)
    if algo_type != "sac":
        raise ValueError(f"Unsupported FADA teacher algo_type: {algo_type!r}")
    return DistillationTeacherSpec(
        obs_dim=int(cfg.teacher.obs_dim),
        action_dim=int(cfg.teacher.action_dim),
        algo_type=cast(Literal["sac"], algo_type),
        actor_hidden_dim=int(cfg.teacher.actor_hidden_dim),
        use_layer_norm=bool(cfg.teacher.use_layer_norm),
        obs_normalization=bool(cfg.teacher.obs_normalization),
    )


def _stand_transition_curriculum_cfg(fada_cfg: DictConfig) -> DictConfig:
    configured = OmegaConf.select(fada_cfg, "stand_transition_curriculum")
    defaults = OmegaConf.create(
        {
            "enabled": False,
            "standing_teacher_checkpoint_path": None,
            "standing_task": "g1_stand_still/mujoco",
            "walk_ratio": 1.0,
            "static_stand_ratio": 0.0,
            "walk_to_stand_ratio": 0.0,
            "walk_command": [0.4, 0.0, 0.0],
            "pre_switch_steps": 30,
            "post_switch_steps": 36,
        }
    )
    if configured is None:
        return defaults
    return cast(DictConfig, OmegaConf.merge(defaults, configured))


def _v005_replay_cfg(fada_cfg: DictConfig) -> DictConfig:
    configured = OmegaConf.select(fada_cfg, "v005_replay")
    defaults = OmegaConf.create(
        {
            "enabled": False,
            "static_cold_start_ratio": 0.5,
            "planner_scenario_ratios": {
                "walk": 0.5,
                "static_stand": 0.25,
                "walk_to_stand": 0.25,
            },
        }
    )
    return (
        defaults if configured is None else cast(DictConfig, OmegaConf.merge(defaults, configured))
    )


def _collect_exact_cold_start_windows(
    env: Any,
    *,
    teacher_policy: torch.nn.Module,
    standing_teacher_policy: torch.nn.Module,
    rollout_policy: FADAPlannerIDMPolicy | None,
    config: FADAArchitectureConfig,
    num_windows: int,
    spec: FADACollectionSpec,
) -> FADACollectionResult:
    """Collect reset-aligned windows in batches no larger than the resident env count."""

    batches: list[FADASourceBatch] = []
    results: list[FADACollectionResult] = []
    remaining = int(num_windows)
    while remaining > 0:
        current = min(remaining, int(env.num_envs))
        result = collect_fada_source_windows(
            env,
            teacher_policy=teacher_policy,
            standing_teacher_policy=standing_teacher_policy,
            rollout_policy=rollout_policy,
            config=config,
            num_windows=current,
            spec=replace(spec, command_scenario="static_stand", cold_start_windows=True),
        )
        batches.append(result.batch)
        results.append(result)
        remaining -= current
    return FADACollectionResult(
        batch=_concat_source_batches(batches, config),
        env_steps=sum(result.env_steps for result in results),
        rejected_done_transitions=sum(result.rejected_done_transitions for result in results),
        rejected_command_windows=sum(result.rejected_command_windows for result in results),
        rollout_mode=results[0].rollout_mode,
        command_scenario="static_stand",
        oracle_role="standing",
        rejected_scenario_windows=sum(result.rejected_scenario_windows for result in results),
        window_profile="cold_start",
    )


def _standing_owner_cfg(
    *,
    root_dir: Path,
    cfg: DictConfig,
    task_selector: str,
) -> DictConfig:
    """Compose the dedicated static-standing environment owner configuration."""

    task_path = Path(task_selector)
    if not task_selector or task_path.is_absolute() or ".." in task_path.parts:
        raise ValueError(
            f"standing curriculum task must be a relative owner selector, got {task_selector!r}"
        )
    owner_path = root_dir / "conf" / "distill" / "task" / task_path.with_suffix(".yaml")
    if not owner_path.is_file():
        raise FileNotFoundError(f"standing curriculum task owner does not exist: {owner_path}")

    base = cast(
        DictConfig,
        OmegaConf.load(root_dir / "conf" / "distill" / "config.yaml"),
    )
    if "defaults" in base:
        del base["defaults"]
    standing_cfg = cast(
        DictConfig,
        OmegaConf.merge(
            base,
            OmegaConf.load(owner_path),
            {
                "algo": OmegaConf.to_container(cfg.algo, resolve=True),
                "student": OmegaConf.to_container(cfg.student, resolve=True),
                "training": {
                    "device": OmegaConf.select(cfg, "training.device"),
                    "fada": OmegaConf.to_container(cfg.training.fada, resolve=True),
                },
            },
        ),
    )
    if str(standing_cfg.training.task_name) != "G1StandStill":
        raise ValueError(
            "standing curriculum task owner must resolve to G1StandStill, got "
            f"{standing_cfg.training.task_name!r}"
        )
    if str(standing_cfg.training.sim_backend) != str(cfg.training.sim_backend):
        raise ValueError(
            "standing and walking FADA environments must use the same simulation backend"
        )
    return standing_cfg


def _curriculum_and_allocations(
    fada_cfg: DictConfig,
    config: FADAArchitectureConfig,
) -> tuple[DictConfig, tuple[tuple[str, int], ...]]:
    curriculum = _stand_transition_curriculum_cfg(fada_cfg)
    if not bool(curriculum.enabled):
        return curriculum, (("walk", int(fada_cfg.windows_per_iteration)),)
    if config.command_dim != 3:
        raise ValueError("standing curriculum requires FADA command_dim=3")
    command_keys = OmegaConf.to_container(fada_cfg.command_info_keys, resolve=True)
    if command_keys != ["commands"]:
        raise ValueError("standing curriculum requires command_info_keys=['commands']")
    walk_command = [float(value) for value in curriculum.walk_command]
    if (
        len(walk_command) != 3
        or not all(math.isfinite(value) for value in walk_command)
        or not any(abs(value) > 1.0e-6 for value in walk_command)
    ):
        raise ValueError("standing curriculum walk_command must be finite, active, and 3-D")
    if int(curriculum.pre_switch_steps) < config.history_length:
        raise ValueError("standing curriculum pre_switch_steps must be at least history_length")
    if int(curriculum.post_switch_steps) < config.prediction_horizon:
        raise ValueError(
            "standing curriculum post_switch_steps must be at least prediction_horizon"
        )
    allocations = allocate_fada_command_scenarios(
        int(fada_cfg.windows_per_iteration),
        {
            "walk": float(curriculum.walk_ratio),
            "static_stand": float(curriculum.static_stand_ratio),
            "walk_to_stand": float(curriculum.walk_to_stand_ratio),
        },
    )
    return curriculum, allocations


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
        """在一个 weight-version barrier 内产出完整 scenario source artifact.

        函数名说明:
            该 worker owner 负责 collection 生命周期和 Oracle role 路由, 不更新 learner.

        主链路:
            上游: parent learner 发布的 DaggerCollectRequest 与 SharedWeightSync version.
            下游: schema-validated FADA source artifact, 随后由 parent replay consumer 读取.

        语义:
            main source 可分为 walk/static_stand/walk_to_stand; intermediate Oracle 永远只属于 walk.
        """

        # B1: 同步唯一 student weight version, 产出已校验的 scenario allocations.
        if request.scenario != FADA_ASYNC_SCENARIO:
            raise ValueError(f"unsupported FADA async scenario: {request.scenario!r}")
        if self.weight_sync is None:
            raise RuntimeError("FADA collector worker is closed")
        started = time.perf_counter()
        self.local_weight_version = self.weight_sync.read_weights_into(self.student.state_dict())
        sync_finished = time.perf_counter()
        fada_cfg = self.cfg.training.fada
        common = self._collection_spec()

        curriculum, allocations = _curriculum_and_allocations(fada_cfg, self.config)
        curriculum_enabled = bool(curriculum.enabled)
        replay_cfg = _v005_replay_cfg(fada_cfg)
        v005_enabled = bool(replay_cfg.enabled)
        cold_start_ratio = float(replay_cfg.static_cold_start_ratio)
        if v005_enabled and not math.isfinite(cold_start_ratio):
            raise ValueError("v005 static_cold_start_ratio must be finite")
        if v005_enabled and not 0.0 < cold_start_ratio < 1.0:
            raise ValueError("v005 static_cold_start_ratio must be strictly between 0 and 1")
        standing_teacher = getattr(self, "standing_teacher", None)
        if curriculum_enabled:
            if standing_teacher is None:
                raise ValueError("enabled standing curriculum requires a loaded standing Oracle")
            if (
                any(scenario == "static_stand" for scenario, _ in allocations)
                and getattr(self, "standing_env", None) is None
            ):
                raise ValueError(
                    "enabled static standing curriculum requires a G1StandStill environment"
                )

        # B2: 按 scenario-authoritative Oracle 收集 main source, 再追加 walking intermediate source.
        batches: list[FADASourceBatch] = []
        summaries: list[dict[str, Any]] = []
        main_windows = 0
        for scenario, scenario_windows in allocations:
            scenario_env = self.standing_env if scenario == "static_stand" else self.env
            scenario_spec = replace(
                common,
                collect_oracle_shadow=bool(fada_cfg.oracle_shadow_enabled),
                transition_walk_command=tuple(float(value) for value in curriculum.walk_command),
                transition_pre_switch_steps=int(curriculum.pre_switch_steps),
                transition_post_switch_steps=int(curriculum.post_switch_steps),
                command_scenario=cast(Any, scenario),
            )
            profiles: tuple[tuple[bool, int], ...] = ((False, scenario_windows),)
            if v005_enabled and scenario == "static_stand":
                cold_windows = int(math.floor(scenario_windows * cold_start_ratio + 0.5))
                steady_windows = int(scenario_windows) - cold_windows
                if cold_windows <= 0 or steady_windows <= 0:
                    raise ValueError(
                        "v005 static allocation must contain cold-start and steady-state windows"
                    )
                profiles = ((True, cold_windows), (False, steady_windows))
            for cold_start, profile_windows in profiles:
                if cold_start:
                    main = _collect_exact_cold_start_windows(
                        scenario_env,
                        teacher_policy=self.final_teacher,
                        standing_teacher_policy=cast(torch.nn.Module, standing_teacher),
                        rollout_policy=None if request.iteration == 0 else self.student,
                        config=self.config,
                        num_windows=profile_windows,
                        spec=scenario_spec,
                    )
                else:
                    main = collect_fada_source_windows(
                        scenario_env,
                        teacher_policy=self.final_teacher,
                        standing_teacher_policy=standing_teacher,
                        rollout_policy=None if request.iteration == 0 else self.student,
                        config=self.config,
                        num_windows=profile_windows,
                        spec=scenario_spec,
                    )
                batches.append(main.batch)
                main_windows += int(main.batch.command.shape[0])
                summaries.append(
                    _summary(
                        main,
                        iteration=request.iteration,
                        source="optimal_or_current_policy",
                    )
                )

        # Intermediate Oracles are loaded one at a time so 20 source identities do not
        # become 20 resident GPU models. The environment and final Oracle remain resident.
        for source_path, source_windows in self.source_allocations:
            intermediate = self._teacher_loader(
                source_path,
                self.teacher_spec,
                device=self.device,
            )
            collection = collect_fada_source_windows(
                self.env,
                teacher_policy=self.final_teacher,
                rollout_teacher_policy=intermediate,
                config=self.config,
                num_windows=source_windows,
                spec=replace(
                    common,
                    collect_oracle_shadow=True,
                    planner_eligible=not v005_enabled,
                ),
            )
            batches.append(collection.batch)
            summaries.append(
                _summary(
                    collection,
                    iteration=request.iteration,
                    source="intermediate_oracle",
                    source_checkpoint=source_path,
                )
            )
            del intermediate

        # B3: 合并并原子写出带配额/角色证据的 artifact, 产出 parent barrier receipt.
        collected = time.perf_counter()
        batch = _concat_source_batches(batches, self.config)
        save_fada_source_batch(
            request.output_path,
            batch,
            config=self.config,
            metadata={
                "iteration": request.iteration,
                "main_windows": main_windows,
                "stand_transition_curriculum_enabled": curriculum_enabled,
                "v005_replay_enabled": v005_enabled,
                "scenario_allocations": dict(allocations),
                "collections": summaries,
            },
        )
        written = time.perf_counter()
        return DaggerCollectResult(
            request_id=request.request_id,
            scenario=request.scenario,
            iteration=request.iteration,
            checkpoint_path=request.checkpoint_path,
            output_path=request.output_path,
            expected_weight_version=request.expected_weight_version,
            observed_weight_version=self.local_weight_version,
            num_samples=int(batch.command.shape[0]),
            worker_pid=os.getpid(),
            metrics={
                "weight_sync_seconds": sync_finished - started,
                "collect_seconds": collected - sync_finished,
                "artifact_write_seconds": written - collected,
            },
            metadata={
                "main_windows": main_windows,
                "scenario_allocations": dict(allocations),
            },
        )

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
