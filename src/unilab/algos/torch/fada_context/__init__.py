"""FADA in-context execution calibration training components."""

from unilab.algos.torch.fada_context.formal_protocol import (
    assess_phase1_teacher_quality,
    validate_phase1_formal_evaluation_contract,
    validate_phase1_formal_training_config,
)
from unilab.algos.torch.fada_context.paired_evaluation import (
    aggregate_paired_reports,
    evaluate_paired_rollouts,
)
from unilab.algos.torch.fada_context.privileged_residual_sac import (
    PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE,
    PrivilegedResidualSACActor,
    PrivilegedResidualSACLearner,
    load_privileged_residual_actor_checkpoint,
    resolve_privileged_residual_sac_runtime,
)
from unilab.algos.torch.fada_context.support_query import (
    ContextActionOutput,
    ContextQueryBatch,
    FADASupportContextEncoder,
    FrozenIDMSupportQueryPolicy,
    SupportContextBatch,
    SupportQueryBatch,
    SupportQueryContextConfig,
    context_first_action_loss,
)
from unilab.algos.torch.fada_context.support_query_collector import (
    SupportQueryCollectionConfig,
    SupportQueryCollectionResult,
    collect_support_query_pairs,
)
from unilab.algos.torch.fada_context.support_query_data import (
    SUPPORT_QUERY_DATASET_SCHEMA_VERSION,
    load_support_query_dataset,
    save_support_query_dataset,
    split_support_query_by_rollout,
    support_query_split_identity_sha256,
)
from unilab.algos.torch.fada_context.support_query_evaluation import (
    TRAJECTORY_DISTANCE_METRICS,
    aggregate_support_query_closed_loop_reports,
    evaluate_support_query_closed_loop,
)
from unilab.algos.torch.fada_context.support_query_runtime import (
    collect_fixed_fault_support_query,
    create_fixed_fault_paired_environments,
    load_support_query_config,
    parameter_snapshot,
    parameters_equal,
    resolve_repo_path,
    sha256_file,
)
from unilab.algos.torch.fada_context.support_query_training import (
    CONTEXT_SUPPORT_QUERY_CHECKPOINT_SCHEMA_VERSION,
    PreparedSupportQueryTraining,
    evaluate_context_action_mse,
    load_context_support_query_checkpoint,
    prepare_support_query_training,
    save_context_support_query_checkpoint,
)

__all__ = [
    "PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE",
    "SUPPORT_QUERY_DATASET_SCHEMA_VERSION",
    "CONTEXT_SUPPORT_QUERY_CHECKPOINT_SCHEMA_VERSION",
    "ContextActionOutput",
    "ContextQueryBatch",
    "FADASupportContextEncoder",
    "FrozenIDMSupportQueryPolicy",
    "PreparedSupportQueryTraining",
    "SupportContextBatch",
    "SupportQueryBatch",
    "SupportQueryCollectionConfig",
    "SupportQueryCollectionResult",
    "SupportQueryContextConfig",
    "TRAJECTORY_DISTANCE_METRICS",
    "PrivilegedResidualSACActor",
    "PrivilegedResidualSACLearner",
    "assess_phase1_teacher_quality",
    "collect_support_query_pairs",
    "collect_fixed_fault_support_query",
    "create_fixed_fault_paired_environments",
    "context_first_action_loss",
    "evaluate_context_action_mse",
    "evaluate_support_query_closed_loop",
    "load_context_support_query_checkpoint",
    "load_privileged_residual_actor_checkpoint",
    "load_support_query_dataset",
    "load_support_query_config",
    "parameter_snapshot",
    "parameters_equal",
    "prepare_support_query_training",
    "resolve_privileged_residual_sac_runtime",
    "resolve_repo_path",
    "save_support_query_dataset",
    "split_support_query_by_rollout",
    "support_query_split_identity_sha256",
    "sha256_file",
    "save_context_support_query_checkpoint",
    "aggregate_paired_reports",
    "aggregate_support_query_closed_loop_reports",
    "evaluate_paired_rollouts",
    "validate_phase1_formal_evaluation_contract",
    "validate_phase1_formal_training_config",
]
