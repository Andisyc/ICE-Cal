"""FADA in-context execution calibration training components."""

from unilab.algos.torch.fada_context.differentiable_rollout import (
    DifferentiableContextRollout,
    DifferentiableContextRolloutOutput,
    TrajectoryContextLoss,
    trajectory_context_loss,
)
from unilab.algos.torch.fada_context.fault_dynamics import (
    FaultDynamicsConfig,
    FaultDynamicsEnsemble,
    FaultDynamicsLoss,
    FaultDynamicsPrediction,
    FaultTransitionBatch,
    fault_dynamics_loss,
)

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
from unilab.algos.torch.fada_context.trajectory_context import (
    ContextEncoderConfig,
    ContextPolicyOutput,
    FADATrajectoryContextEncoder,
    FrozenPlannerIDMContextPolicy,
)

__all__ = [
    "PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE",
    "ContextEncoderConfig",
    "ContextPolicyOutput",
    "DifferentiableContextRollout",
    "DifferentiableContextRolloutOutput",
    "FADATrajectoryContextEncoder",
    "FaultDynamicsConfig",
    "FaultDynamicsEnsemble",
    "FaultDynamicsLoss",
    "FaultDynamicsPrediction",
    "FaultTransitionBatch",
    "FrozenPlannerIDMContextPolicy",
    "PrivilegedResidualSACActor",
    "PrivilegedResidualSACLearner",
    "assess_phase1_teacher_quality",
    "load_privileged_residual_actor_checkpoint",
    "resolve_privileged_residual_sac_runtime",
    "aggregate_paired_reports",
    "evaluate_paired_rollouts",
    "fault_dynamics_loss",
    "trajectory_context_loss",
    "TrajectoryContextLoss",
    "validate_phase1_formal_evaluation_contract",
    "validate_phase1_formal_training_config",
]
