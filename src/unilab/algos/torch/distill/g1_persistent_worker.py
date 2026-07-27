"""Production G1 resource worker for persistent DAgger collection.

状态: HP-3b2 production factory wired and bounded MuJoCo lifecycle confirmed.
上游: ``train_distill.py`` persistent execution-mode factory selection.
下游: exact resource cache, collector/data owners, dataset artifacts, merged HP-4 observations.
证据: S1/S2 config, lifecycle, differential, metrics connector tests, and S4 lifecycle.
边界: parent workflow owns identity/persistence; physical quality and bounded timing remain separate gates.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, cast

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.ipc import SharedWeightSync
from unilab.training import BackendAdapter, create_env, ensure_registries

from .async_runtime import DaggerCollectRequest, DaggerCollectResult
from .collector import (
    collect_distillation_dataset_from_env,
    collect_transition_distillation_dataset_from_env,
)
from .dagger import resolve_command_intent_rollout_policies
from .data import save_distillation_dataset
from .moe_student import MoEStudentPolicy
from .performance import (
    DISTILLATION_METRICS_SCHEMA_VERSION,
    DistillationStageObservation,
)
from .persistent_resources import (
    PersistentResourceCache,
    PersistentResourceIdentity,
)
from .persistent_runtime import PersistentDistillationRuntime
from .playback import load_distillation_student_policy
from .teacher import DistillationTeacherSpec, load_sac_teacher_policy
from .workflow import (
    RoleArtifactSpec,
    WorkflowScenarioSpec,
    config_fingerprint,
    file_sha256,
)


class _TeacherResource:
    def __init__(self, policy: torch.nn.Module) -> None:
        self.policy = policy
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _teacher_spec(cfg: DictConfig) -> DistillationTeacherSpec:
    algo_type = str(cfg.teacher.algo_type)
    if algo_type != "sac":
        raise ValueError(f"Unsupported distillation teacher algo_type: {algo_type!r}")
    validated_algo_type = cast(Literal["sac"], algo_type)
    return DistillationTeacherSpec(
        obs_dim=int(cfg.teacher.obs_dim),
        action_dim=int(cfg.teacher.action_dim),
        algo_type=validated_algo_type,
        actor_hidden_dim=int(cfg.teacher.actor_hidden_dim),
        use_layer_norm=bool(cfg.teacher.use_layer_norm),
        obs_normalization=bool(cfg.teacher.obs_normalization),
    )


def _teacher_spec_fingerprint(cfg: DictConfig) -> str:
    return config_fingerprint(
        {
            "obs_dim": int(cfg.teacher.obs_dim),
            "action_dim": int(cfg.teacher.action_dim),
            "algo_type": str(cfg.teacher.algo_type),
            "actor_hidden_dim": int(cfg.teacher.actor_hidden_dim),
            "use_layer_norm": bool(cfg.teacher.use_layer_norm),
            "obs_normalization": bool(cfg.teacher.obs_normalization),
        }
    )


class PersistentG1DistillationWorker:
    """Keep student, role teachers, and exact-compatible G1 envs resident."""

    def __init__(
        self,
        *,
        root_dir: str,
        role_cfgs: Mapping[str, Mapping[str, Any]],
        role_specs: Sequence[Mapping[str, Any]],
        scenario_specs: Sequence[Mapping[str, Any]],
        workflow_cfg: Mapping[str, Any],
        initial_checkpoint_path: str,
        device: str,
        weight_sync_name: str,
        weight_sync_lock: Any,
        weight_param_shapes: Mapping[str, torch.Size],
        lifecycle_report_queue: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.device = str(device)
        self.role_cfgs = {str(role): OmegaConf.create(dict(cfg)) for role, cfg in role_cfgs.items()}
        self.role_specs = {str(item["role"]): dict(item) for item in role_specs}
        self.scenario_specs = {str(item["name"]): dict(item) for item in scenario_specs}
        self.workflow_cfg = dict(workflow_cfg)
        loaded_student = load_distillation_student_policy(
            initial_checkpoint_path,
            device=self.device,
        )
        self.student = loaded_student.policy
        self.student_runtime_cfg = dict(loaded_student.distill_runtime_cfg)
        self.weight_sync = SharedWeightSync(
            dict(weight_param_shapes),
            create=False,
            shm_name=weight_sync_name,
            lock=weight_sync_lock,
        )
        self.local_weight_version = self.weight_sync.read_weights_into(self.student.state_dict())
        self.student_init_count = 1
        self.lifecycle_report_queue = lifecycle_report_queue
        self._clock = clock
        ensure_registries()

        self._identity_by_role: dict[str, PersistentResourceIdentity] = {}
        self._env_override_by_key: dict[str, Any] = {}
        self._role_by_key: dict[str, str] = {}
        for role, cfg in self.role_cfgs.items():
            spec = self.role_specs[role]
            env_override = BackendAdapter(
                cfg,
                root_dir=self.root_dir,
                algo_name="distill",
            ).build_task_env_cfg_override()
            env_override_payload = OmegaConf.to_container(
                OmegaConf.create(env_override),
                resolve=True,
            )
            checkpoint_path = Path(str(cfg.teacher.checkpoint_path)).resolve()
            identity = PersistentResourceIdentity(
                task_owner=str(spec["task"]),
                task_name=str(cfg.training.task_name),
                sim_backend=str(cfg.training.sim_backend),
                env_cfg_fingerprint=config_fingerprint({"env_cfg_override": env_override_payload}),
                num_envs=int(self.workflow_cfg["collect_num_envs"]),
                teacher_checkpoint_path=str(checkpoint_path),
                teacher_checkpoint_sha256=file_sha256(checkpoint_path),
                teacher_spec_fingerprint=_teacher_spec_fingerprint(cfg),
            )
            self._identity_by_role[role] = identity
            self._env_override_by_key[identity.cache_key] = env_override
            self._role_by_key[identity.cache_key] = role

        self.resources = PersistentResourceCache(
            teacher_factory=self._create_teacher,
            env_factory=self._create_env,
        )

    def _create_teacher(self, identity: PersistentResourceIdentity) -> _TeacherResource:
        role = self._role_by_key[identity.cache_key]
        cfg = self.role_cfgs[role]
        return _TeacherResource(
            load_sac_teacher_policy(
                identity.teacher_checkpoint_path,
                _teacher_spec(cfg),
                device=self.device,
            )
        )

    def _create_env(self, identity: PersistentResourceIdentity) -> Any:
        role = self._role_by_key[identity.cache_key]
        cfg = self.role_cfgs[role]
        return create_env(
            cfg,
            num_envs=identity.num_envs,
            env_cfg_override=self._env_override_by_key[identity.cache_key],
            sim_backend=identity.sim_backend,
            task_name=identity.task_name,
        )

    def _sync_student(self, expected_version: int) -> float:
        start = self._clock()
        observed_version = self.weight_sync.read_weights_into(self.student.state_dict())
        if observed_version != int(expected_version):
            raise ValueError(
                "persistent G1 worker weight version mismatch: "
                f"expected={expected_version} observed={observed_version}"
            )
        self.local_weight_version = observed_version
        return self._clock() - start

    def _metadata(
        self,
        *,
        request: DaggerCollectRequest,
        roles: Sequence[str],
    ) -> dict[str, Any]:
        identities = [self._identity_by_role[role] for role in roles]
        return {
            "workflow_scenario": request.scenario,
            "input_checkpoint_path": request.checkpoint_path,
            "input_weight_version": request.expected_weight_version,
            "resource_cache_keys": [identity.cache_key for identity in identities],
            "teacher_checkpoint_paths": [
                identity.teacher_checkpoint_path for identity in identities
            ],
            "teacher_checkpoint_sha256": [
                identity.teacher_checkpoint_sha256 for identity in identities
            ],
        }

    def _collect_role(
        self,
        request: DaggerCollectRequest,
        scenario: Mapping[str, Any],
    ):
        role = str(scenario["source_roles"][0])
        cfg = self.role_cfgs[role]
        identity = self._identity_by_role[role]

        def collect(env: Any, teacher: _TeacherResource, initial_reset: Any):
            drop_index = OmegaConf.select(cfg, "training.collect_student_drop_index")
            max_env_steps = OmegaConf.select(cfg, "training.collect_max_env_steps")
            return collect_distillation_dataset_from_env(
                env,
                num_samples=int(self.workflow_cfg["dagger_samples_per_role"]),
                expected_student_obs_dim=int(cfg.student.obs_dim),
                expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
                teacher_obs_key=str(cfg.training.collect_teacher_obs_key),
                teacher_projection=str(cfg.training.collect_teacher_projection),
                student_projection=str(cfg.training.collect_student_projection),
                student_drop_index=None if drop_index is None else int(drop_index),
                action_mode="student_policy",
                teacher_policy=teacher.policy,
                rollout_policy=self.student,
                command_sample_filter=str(cfg.training.collect_command_sample_filter),
                command_info_key=str(cfg.training.collect_command_info_key),
                target_height_info_key=OmegaConf.select(
                    cfg, "training.collect_target_height_info_key"
                ),
                command_xy_threshold=float(cfg.training.collect_command_xy_threshold),
                command_yaw_threshold=float(cfg.training.collect_command_yaw_threshold),
                max_env_steps=None if max_env_steps is None else int(max_env_steps),
                role_label=role,
                metadata=self._metadata(request=request, roles=(role,)),
                initial_reset=initial_reset,
                performance_clock=self._clock,
            )

        return self.resources.run_request(identity, collect)

    def _collect_transition(
        self,
        request: DaggerCollectRequest,
        scenario: Mapping[str, Any],
    ):
        if request.scenario != "walk_to_stop":
            raise ValueError(f"unsupported persistent transition scenario: {request.scenario!r}")
        source_roles = tuple(str(value) for value in scenario["source_roles"])
        roles_by_filter: dict[str, list[str]] = {"active": [], "inactive": []}
        for role in source_roles:
            command_filter = str(self.role_cfgs[role].training.collect_command_sample_filter)
            if command_filter in roles_by_filter:
                roles_by_filter[command_filter].append(role)
        if any(len(roles) != 1 for roles in roles_by_filter.values()):
            raise ValueError(
                "persistent walk_to_stop requires one active and one inactive role, "
                f"got {roles_by_filter}"
            )
        walk_role = roles_by_filter["active"][0]
        stand_role = roles_by_filter["inactive"][0]
        walk_cfg = self.role_cfgs[walk_role]
        stand_cfg = self.role_cfgs[stand_role]
        walk_target_height_info_key = OmegaConf.select(
            walk_cfg, "training.collect_target_height_info_key"
        )
        stand_target_height_info_key = OmegaConf.select(
            stand_cfg, "training.collect_target_height_info_key"
        )
        if walk_target_height_info_key != stand_target_height_info_key:
            raise ValueError("persistent transition roles must agree on target-height info key")
        walk_identity = self._identity_by_role[walk_role]
        stand_bundle = self.resources.acquire(self._identity_by_role[stand_role])
        rollout_policy: torch.nn.Module | None = self.student
        rollout_policies_by_intent: dict[str, torch.nn.Module] | None = None
        if isinstance(self.student, MoEStudentPolicy):
            rollout_policies_by_intent, _targets = resolve_command_intent_rollout_policies(
                self.student,
                self.student_runtime_cfg,
            )
            rollout_policy = None

        def collect(env: Any, walking_teacher: _TeacherResource, initial_reset: Any):
            return collect_transition_distillation_dataset_from_env(
                env,
                num_samples=int(self.workflow_cfg["dagger_samples_per_role"]),
                expected_student_obs_dim=int(walk_cfg.student.obs_dim),
                expected_teacher_obs_dim=int(walk_cfg.teacher.obs_dim),
                walking_teacher_policy=walking_teacher.policy,
                standing_teacher_policy=stand_bundle.teacher.policy,
                rollout_policy=rollout_policy,
                rollout_policies_by_intent=rollout_policies_by_intent,
                pre_switch_steps=int(self.workflow_cfg["transition_pre_switch_steps"]),
                nominal_settle_steps=int(self.workflow_cfg["transition_nominal_settle_steps"]),
                min_post_switch_steps=int(self.workflow_cfg["transition_min_post_switch_steps"]),
                walk_command=tuple(self.workflow_cfg["transition_walk_command"]),
                walk_commands=self.workflow_cfg.get("transition_walk_commands", ()),
                nominal_walk_target_height=self.workflow_cfg.get("transition_walk_target_height"),
                post_switch_target_heights=self.workflow_cfg.get(
                    "transition_post_switch_target_heights", ()
                ),
                teacher_obs_key=str(walk_cfg.training.collect_teacher_obs_key),
                teacher_projection=str(walk_cfg.training.collect_teacher_projection),
                student_projection=str(walk_cfg.training.collect_student_projection),
                student_drop_index=OmegaConf.select(
                    walk_cfg, "training.collect_student_drop_index"
                ),
                command_info_key=str(walk_cfg.training.collect_command_info_key),
                target_height_info_key=walk_target_height_info_key,
                walking_role_label=walk_role,
                standing_role_label=stand_role,
                scenario_label=request.scenario,
                max_env_steps=self.workflow_cfg["transition_max_env_steps"],
                metadata=self._metadata(
                    request=request,
                    roles=(walk_role, stand_role),
                ),
                initial_reset=initial_reset,
                performance_clock=self._clock,
            )

        return self.resources.run_request(walk_identity, collect)

    @staticmethod
    def _collector_performance_observations(
        dataset: Any,
    ) -> tuple[DistillationStageObservation, ...]:
        schema_version = dataset.metadata.get("performance_metrics_schema_version")
        if schema_version != DISTILLATION_METRICS_SCHEMA_VERSION:
            raise ValueError(
                "persistent collector performance schema mismatch: "
                f"expected={DISTILLATION_METRICS_SCHEMA_VERSION} "
                f"observed={schema_version!r}"
            )
        payloads = dataset.metadata.get("performance_stage_observations")
        if not isinstance(payloads, list):
            raise ValueError("persistent collector performance observations are missing")
        observations = tuple(
            DistillationStageObservation.from_dict(payload) for payload in payloads
        )
        expected_stages = (
            "teacher_inference",
            "student_inference",
            "env_step",
            "tensor_pack",
        )
        observed_stages = tuple(observation.stage for observation in observations)
        if observed_stages != expected_stages:
            raise ValueError(
                "persistent collector performance stage order mismatch: "
                f"expected={expected_stages} observed={observed_stages}"
            )
        return observations

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        request_start = self._clock()
        weight_sync_seconds = self._sync_student(request.expected_weight_version)
        scenario = self.scenario_specs.get(request.scenario)
        if scenario is None:
            raise ValueError(f"unknown persistent workflow scenario: {request.scenario!r}")
        before = self.resources.counters()
        collect_start = self._clock()
        if str(scenario["kind"]) == "role":
            dataset = self._collect_role(request, scenario)
        else:
            dataset = self._collect_transition(request, scenario)
        collect_seconds = self._clock() - collect_start
        collector_observations = self._collector_performance_observations(dataset)
        write_start = self._clock()
        save_distillation_dataset(request.output_path, dataset)
        artifact_write_seconds = self._clock() - write_start
        total_elapsed_seconds = self._clock() - request_start
        after = self.resources.counters()
        env_step_count = dataset.metadata.get("env_steps", 0)
        stage_observations = (
            DistillationStageObservation(
                stage="weight_sync",
                duration_seconds=weight_sync_seconds,
                row_count=0,
                env_step_count=0,
                success=True,
                error=None,
                cleanup_state="not_applicable",
            ),
            *collector_observations,
            DistillationStageObservation(
                stage="artifact_write",
                duration_seconds=artifact_write_seconds,
                row_count=dataset.num_samples,
                env_step_count=0,
                success=True,
                error=None,
                cleanup_state="not_applicable",
            ),
            DistillationStageObservation(
                stage="total_elapsed",
                duration_seconds=total_elapsed_seconds,
                row_count=dataset.num_samples,
                env_step_count=env_step_count,
                success=True,
                error=None,
                cleanup_state="pending",
            ),
        )
        return DaggerCollectResult(
            request_id=request.request_id,
            scenario=request.scenario,
            iteration=request.iteration,
            checkpoint_path=request.checkpoint_path,
            output_path=request.output_path,
            expected_weight_version=request.expected_weight_version,
            observed_weight_version=self.local_weight_version,
            num_samples=dataset.num_samples,
            worker_pid=os.getpid(),
            metrics={
                "weight_sync_seconds": weight_sync_seconds,
                "collect_seconds": collect_seconds,
                "artifact_write_seconds": artifact_write_seconds,
                "student_init_count": float(self.student_init_count),
                "teacher_init_count": float(after["teacher_init_count"]),
                "env_init_count": float(after["env_init_count"]),
                "cache_hit_count": float(after["cache_hit_count"]),
                "request_reset_count": float(after["reset_count"] - before["reset_count"]),
            },
            metadata={
                **self._metadata(
                    request=request,
                    roles=tuple(str(value) for value in scenario["source_roles"]),
                ),
                "resource_counters": after,
                "dataset_role_counts": dict(dataset.metadata.get("role_counts") or {}),
                "dataset_command_intent_counts": dict(
                    dataset.metadata.get("command_intent_counts") or {}
                ),
                "performance_stage_observations": [
                    observation.as_dict() for observation in stage_observations
                ],
                "performance_metrics_schema_version": (DISTILLATION_METRICS_SCHEMA_VERSION),
            },
        )

    def close(self) -> None:
        try:
            self.resources.close()
        finally:
            try:
                self.weight_sync.close()
            finally:
                if self.lifecycle_report_queue is not None:
                    self.lifecycle_report_queue.put(
                        {
                            "worker_pid": os.getpid(),
                            "student_init_count": self.student_init_count,
                            "resource_counters": self.resources.counters(),
                            "cache_keys": list(self.resources.cache_keys),
                        }
                    )


def _build_persistent_g1_worker(**kwargs: Any) -> PersistentG1DistillationWorker:
    return PersistentG1DistillationWorker(**kwargs)


def build_persistent_g1_distillation_runtime(
    *,
    cfg: DictConfig,
    role_cfgs: Mapping[str, DictConfig],
    role_specs: Sequence[RoleArtifactSpec],
    scenario_specs: Sequence[WorkflowScenarioSpec] | None,
) -> PersistentDistillationRuntime:
    """Build the production persistent G1 scenario runtime owner."""

    if scenario_specs is None:
        raise ValueError("persistent G1 runtime requires scenario_specs")
    device_value = OmegaConf.select(cfg, "training.device", default="cpu")
    device = "cpu" if device_value in (None, "") else str(device_value)
    workflow_cfg = {
        key: OmegaConf.select(cfg, f"training.workflow.{key}")
        for key in (
            "collect_num_envs",
            "dagger_samples_per_role",
            "transition_pre_switch_steps",
            "transition_nominal_settle_steps",
            "transition_min_post_switch_steps",
            "transition_walk_command",
            "transition_walk_commands",
            "transition_walk_target_height",
            "transition_post_switch_target_heights",
            "transition_max_env_steps",
        )
    }
    worker_kwargs = {
        "root_dir": str(Path(__file__).resolve().parents[5]),
        "role_cfgs": {
            role: OmegaConf.to_container(role_cfg, resolve=True)
            for role, role_cfg in role_cfgs.items()
        },
        "role_specs": [{"role": spec.role, "task": spec.task} for spec in role_specs],
        "scenario_specs": [scenario.as_dict() for scenario in scenario_specs],
        "workflow_cfg": OmegaConf.to_container(
            OmegaConf.create(workflow_cfg),
            resolve=True,
        ),
        "device": device,
    }

    def load_student(path: Path) -> torch.nn.Module:
        return load_distillation_student_policy(path, device=device).policy

    lifecycle_report_queue = mp.get_context("spawn").Queue(maxsize=1)
    worker_kwargs["lifecycle_report_queue"] = lifecycle_report_queue
    return PersistentDistillationRuntime(
        student_loader=load_student,
        worker_factory=_build_persistent_g1_worker,
        worker_kwargs=worker_kwargs,
        worker_kwargs_factory=lambda checkpoint_path: {
            "initial_checkpoint_path": str(checkpoint_path.resolve())
        },
        lifecycle_report_queue=lifecycle_report_queue,
    )
