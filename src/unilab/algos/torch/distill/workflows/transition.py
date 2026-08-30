"""Legacy transition-scenario collection owner."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.collection.transition import (
    collect_transition_distillation_dataset_from_env,
)
from unilab.algos.torch.distill.contracts.workflow import (
    WorkflowScenarioCollectionResult,
    WorkflowScenarioSpec,
    resolve_walk_to_stop_role_pair,
)
from unilab.algos.torch.distill.datasets.io import save_distillation_dataset
from unilab.algos.torch.distill.learning.dagger import (
    resolve_command_intent_rollout_policies,
)
from unilab.algos.torch.distill.learning.moe_student import MoEStudentPolicy
from unilab.algos.torch.distill.learning.playback import load_distillation_student_policy
from unilab.algos.torch.distill.learning.teacher import load_sac_teacher_policy
from unilab.algos.torch.distill.observability.performance import (
    COLLECTOR_REQUEST_STAGE_NAMES,
    DISTILLATION_METRICS_SCHEMA_VERSION,
    DistillationStageObservation,
)
from unilab.algos.torch.distill.workflows.entry_collection import (
    _distill_device,
)
from unilab.algos.torch.distill.workflows.entry_training import (
    build_teacher_spec,
)
from unilab.training import BackendAdapter, create_env, ensure_registries

ROOT_DIR = Path(__file__).resolve().parents[6]


def collect_legacy_transition_scenario(
    *,
    cfg: DictConfig,
    scenario: WorkflowScenarioSpec,
    role_cfgs: Mapping[str, DictConfig],
    checkpoint_path: Path,
    output_path: Path,
    performance_clock: Callable[[], float],
) -> WorkflowScenarioCollectionResult:
    if scenario.name != "walk_to_stop":
        raise ValueError(f"unsupported transition workflow scenario: {scenario.name!r}")
    role_pair = resolve_walk_to_stop_role_pair(
        source_roles=scenario.source_roles,
        command_sample_filters={
            role: str(role_cfgs[role].training.collect_command_sample_filter)
            for role in scenario.source_roles
            if role in role_cfgs
        },
        target_height_info_keys={
            role: OmegaConf.select(role_cfgs[role], "training.collect_target_height_info_key")
            for role in scenario.source_roles
            if role in role_cfgs
        },
    )
    walk_role = role_pair.walking_role
    stand_role = role_pair.standing_role
    request_start = float(performance_clock())
    device = _distill_device(cfg)
    walk_cfg = role_cfgs[walk_role]
    stand_cfg = role_cfgs[stand_role]
    loaded_student = load_distillation_student_policy(checkpoint_path, device=device)
    student = loaded_student.policy
    rollout_policy: torch.nn.Module | None = student
    rollout_policies_by_intent: dict[str, torch.nn.Module] | None = None
    rollout_policy_metadata: dict[str, Any] = {}
    if isinstance(student, MoEStudentPolicy):
        rollout_policies_by_intent, expert_targets = resolve_command_intent_rollout_policies(
            student,
            loaded_student.distill_runtime_cfg,
        )
        rollout_policy = None
        rollout_policy_metadata = {
            "rollout_policy_expert_targets": expert_targets,
            "rollout_policy_source": "checkpoint_command_intent_experts",
        }
    walking_teacher = load_sac_teacher_policy(
        walk_cfg.teacher.checkpoint_path,
        build_teacher_spec(walk_cfg),
        device=device,
    )
    standing_teacher = load_sac_teacher_policy(
        stand_cfg.teacher.checkpoint_path,
        build_teacher_spec(stand_cfg),
        device=device,
    )
    ensure_registries()
    env = create_env(
        walk_cfg,
        num_envs=int(OmegaConf.select(cfg, "training.workflow.collect_num_envs", default=64)),
        env_cfg_override=BackendAdapter(
            walk_cfg,
            root_dir=ROOT_DIR,
            algo_name="distill",
        ).build_task_env_cfg_override(),
        sim_backend=str(walk_cfg.training.sim_backend),
        task_name=str(walk_cfg.training.task_name),
    )
    cold_start_seconds = float(performance_clock()) - request_start
    try:
        transition_max_env_steps = OmegaConf.select(
            cfg,
            "training.workflow.transition_max_env_steps",
            default=0,
        )
        transition_walk_commands = OmegaConf.select(
            cfg,
            "training.workflow.transition_walk_commands",
            default=[],
        )
        transition_walk_target_height = OmegaConf.select(
            cfg,
            "training.workflow.transition_walk_target_height",
            default=None,
        )
        transition_post_switch_target_heights = OmegaConf.select(
            cfg,
            "training.workflow.transition_post_switch_target_heights",
            default=[],
        )
        dataset = collect_transition_distillation_dataset_from_env(
            env,
            num_samples=int(
                OmegaConf.select(
                    cfg,
                    "training.workflow.dagger_samples_per_role",
                    default=65536,
                )
            ),
            expected_student_obs_dim=int(walk_cfg.student.obs_dim),
            expected_teacher_obs_dim=int(walk_cfg.teacher.obs_dim),
            walking_teacher_policy=walking_teacher,
            standing_teacher_policy=standing_teacher,
            rollout_policy=rollout_policy,
            rollout_policies_by_intent=rollout_policies_by_intent,
            pre_switch_steps=int(
                OmegaConf.select(
                    cfg,
                    "training.workflow.transition_pre_switch_steps",
                    default=8,
                )
            ),
            nominal_settle_steps=int(
                OmegaConf.select(
                    cfg,
                    "training.workflow.transition_nominal_settle_steps",
                    default=0,
                )
            ),
            min_post_switch_steps=int(
                OmegaConf.select(
                    cfg,
                    "training.workflow.transition_min_post_switch_steps",
                    default=0,
                )
            ),
            walk_command=tuple(
                float(value)
                for value in OmegaConf.select(
                    cfg,
                    "training.workflow.transition_walk_command",
                    default=[0.4, 0.0, 0.0],
                )
            ),
            walk_commands=[
                [float(value) for value in command] for command in transition_walk_commands
            ],
            nominal_walk_target_height=(
                None
                if transition_walk_target_height in (None, "")
                else float(transition_walk_target_height)
            ),
            post_switch_target_heights=[
                float(value) for value in transition_post_switch_target_heights
            ],
            teacher_obs_key=str(walk_cfg.training.collect_teacher_obs_key),
            teacher_projection=str(walk_cfg.training.collect_teacher_projection),
            student_projection=str(walk_cfg.training.collect_student_projection),
            student_drop_index=OmegaConf.select(
                walk_cfg,
                "training.collect_student_drop_index",
            ),
            command_info_key=str(walk_cfg.training.collect_command_info_key),
            target_height_info_key=role_pair.target_height_info_key,
            walking_role_label=walk_role,
            standing_role_label=stand_role,
            scenario_label=scenario.name,
            max_env_steps=(
                None if transition_max_env_steps in (None, "") else int(transition_max_env_steps)
            ),
            metadata={
                "workflow_scenario": scenario.name,
                **rollout_policy_metadata,
            },
            performance_clock=performance_clock,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_start = float(performance_clock())
        save_distillation_dataset(output_path, dataset)
        artifact_write_seconds = float(performance_clock()) - write_start
        collector_payloads = dataset.metadata.get("performance_stage_observations")
        if not isinstance(collector_payloads, list):
            raise ValueError("legacy transition performance observations are missing")
        collector_observations = tuple(
            DistillationStageObservation.from_dict(payload) for payload in collector_payloads
        )
        collector_stages = tuple(item.stage for item in collector_observations)
        if collector_stages != COLLECTOR_REQUEST_STAGE_NAMES:
            raise ValueError(
                "legacy transition performance stage order mismatch: "
                f"expected={COLLECTOR_REQUEST_STAGE_NAMES} "
                f"observed={collector_stages}"
            )
        return WorkflowScenarioCollectionResult(
            num_samples=dataset.num_samples,
            worker_pid=os.getpid(),
            performance_metrics_schema_version=(DISTILLATION_METRICS_SCHEMA_VERSION),
            performance_stage_observations=(
                DistillationStageObservation(
                    stage="cold_start",
                    duration_seconds=cold_start_seconds,
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
                    duration_seconds=float(performance_clock()) - request_start,
                    row_count=dataset.num_samples,
                    env_step_count=int(dataset.metadata.get("env_steps", 0)),
                    success=True,
                    error=None,
                    cleanup_state="pending",
                ),
            ),
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
