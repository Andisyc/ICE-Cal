from __future__ import annotations

from dataclasses import dataclass

import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.differentiable_rollout import (
    DifferentiableContextRollout,
)
from unilab.algos.torch.fada_context.fault_dynamics import (
    FaultDynamicsConfig,
    FaultDynamicsEnsemble,
)
from unilab.algos.torch.fada_context.trajectory_context import (
    ContextEncoderConfig,
    FADATrajectoryContextEncoder,
    FrozenPlannerIDMContextPolicy,
)


@dataclass(frozen=True)
class ContextTrainingSetupConfig:
    context_hidden_dim: int = 128
    context_num_layers: int = 2
    residual_scale: float = 0.1
    dynamics_hidden_dims: tuple[int, ...] = (256, 256)
    dynamics_ensemble_size: int = 5
    context_learning_rate: float = 3.0e-4
    dynamics_learning_rate: float = 3.0e-4

    def __post_init__(self) -> None:
        if self.context_learning_rate <= 0.0 or self.dynamics_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")


@dataclass(frozen=True)
class PreparedContextTraining:
    policy: FrozenPlannerIDMContextPolicy
    dynamics: FaultDynamicsEnsemble
    rollout: DifferentiableContextRollout
    context_optimizer: torch.optim.Optimizer
    dynamics_optimizer: torch.optim.Optimizer


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def prepare_context_training(
    nominal_policy: FADAPlannerIDMPolicy,
    config: ContextTrainingSetupConfig,
) -> PreparedContextTraining:
    """Construct training owners without loading data or taking an optimizer step."""

    fada = nominal_policy.config
    context = FADATrajectoryContextEncoder(
        ContextEncoderConfig.from_fada(
            fada,
            hidden_dim=config.context_hidden_dim,
            num_layers=config.context_num_layers,
            residual_scale=config.residual_scale,
        )
    ).to(next(nominal_policy.parameters()).device)
    policy = FrozenPlannerIDMContextPolicy(
        nominal_policy.planner,
        nominal_policy.idm,
        context,
    )
    dynamics = FaultDynamicsEnsemble(
        FaultDynamicsConfig(
            state_dim=fada.obs_dim,
            action_dim=fada.action_dim,
            hidden_dims=config.dynamics_hidden_dims,
        ),
        ensemble_size=config.dynamics_ensemble_size,
    ).to(next(nominal_policy.parameters()).device)
    rollout = DifferentiableContextRollout(policy, dynamics)
    context_optimizer = torch.optim.Adam(
        context.parameters(), lr=config.context_learning_rate
    )
    dynamics_optimizer = torch.optim.Adam(
        dynamics.parameters(), lr=config.dynamics_learning_rate
    )
    context_ids = {id(parameter) for parameter in context.parameters()}
    dynamics_ids = {id(parameter) for parameter in dynamics.parameters()}
    if _optimizer_parameter_ids(context_optimizer) != context_ids:
        raise RuntimeError("Context optimizer does not own exactly Context parameters")
    if _optimizer_parameter_ids(dynamics_optimizer) != dynamics_ids:
        raise RuntimeError("dynamics optimizer does not own exactly dynamics parameters")
    if context_ids & dynamics_ids:
        raise RuntimeError("Context and dynamics parameters must be disjoint")
    if any(parameter.requires_grad for parameter in policy.planner.parameters()):
        raise RuntimeError("Planner must be frozen before Context training")
    if any(parameter.requires_grad for parameter in policy.idm.parameters()):
        raise RuntimeError("IDM must be frozen before Context training")
    return PreparedContextTraining(
        policy=policy,
        dynamics=dynamics,
        rollout=rollout,
        context_optimizer=context_optimizer,
        dynamics_optimizer=dynamics_optimizer,
    )
