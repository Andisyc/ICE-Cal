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

__all__ = [
    "PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE",
    "PrivilegedResidualSACActor",
    "PrivilegedResidualSACLearner",
    "assess_phase1_teacher_quality",
    "load_privileged_residual_actor_checkpoint",
    "resolve_privileged_residual_sac_runtime",
    "aggregate_paired_reports",
    "evaluate_paired_rollouts",
    "validate_phase1_formal_evaluation_contract",
    "validate_phase1_formal_training_config",
]
