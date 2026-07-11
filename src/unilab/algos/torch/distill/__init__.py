"""Generic behavior distillation helpers for torch locomotion policies."""

from .checkpoint import load_distillation_checkpoint, save_distillation_checkpoint
from .collector import (
    collect_distillation_dataset_from_env,
    command_active_mask,
    project_student_obs,
    project_teacher_obs,
)
from .dagger import IterativeDaggerRunResult, run_iterative_dagger_updates
from .data import (
    DistillationTensorDataset,
    build_distillation_dataset,
    build_multitask_distillation_dataset,
    load_distillation_dataset,
    make_fake_distillation_dataset,
    save_distillation_dataset,
)
from .models import MLPStudentPolicy
from .moe_diagnostics import (
    MoEExpertDiagnostics,
    MoERoleRouteSummary,
    diagnose_moe_expert_routes,
    moe_diagnostics_to_dict,
)
from .moe_student import MoEStudentOutput, MoEStudentPolicy
from .offline import OfflineDistillationRunResult, run_offline_distillation_updates
from .playback import LoadedDistillationStudentPolicy, load_distillation_student_policy
from .teacher import (
    DistillationTeacherCheckpointInfo,
    DistillationTeacherSpec,
    LoadedTeacherPolicy,
    inspect_sac_teacher_checkpoint,
    load_sac_teacher_policy,
    validate_sac_teacher_checkpoint_contract,
)
from .trainer import (
    BehaviorDistillationStats,
    BehaviorDistillationTrainer,
    DistillationBatch,
)

__all__ = [
    "BehaviorDistillationStats",
    "BehaviorDistillationTrainer",
    "DistillationBatch",
    "DistillationTeacherSpec",
    "DistillationTeacherCheckpointInfo",
    "DistillationTensorDataset",
    "IterativeDaggerRunResult",
    "LoadedTeacherPolicy",
    "LoadedDistillationStudentPolicy",
    "MLPStudentPolicy",
    "MoEExpertDiagnostics",
    "MoERoleRouteSummary",
    "MoEStudentOutput",
    "MoEStudentPolicy",
    "OfflineDistillationRunResult",
    "build_distillation_dataset",
    "build_multitask_distillation_dataset",
    "collect_distillation_dataset_from_env",
    "command_active_mask",
    "diagnose_moe_expert_routes",
    "inspect_sac_teacher_checkpoint",
    "load_distillation_checkpoint",
    "load_distillation_dataset",
    "load_distillation_student_policy",
    "load_sac_teacher_policy",
    "make_fake_distillation_dataset",
    "moe_diagnostics_to_dict",
    "project_student_obs",
    "project_teacher_obs",
    "run_offline_distillation_updates",
    "run_iterative_dagger_updates",
    "save_distillation_checkpoint",
    "save_distillation_dataset",
    "validate_sac_teacher_checkpoint_contract",
]
