from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from unilab.algos.torch.fada_context.fault_dynamics import FaultDynamicsEnsemble
from unilab.algos.torch.fada_context.trajectory_context import FrozenPlannerIDMContextPolicy


@dataclass(frozen=True)
class DifferentiableContextRolloutOutput:
    delta_z: torch.Tensor
    predicted_trajectory: torch.Tensor
    actions: torch.Tensor
    model_disagreement: torch.Tensor


@dataclass(frozen=True)
class TrajectoryContextLoss:
    total: torch.Tensor
    tracking: torch.Tensor
    latent: torch.Tensor
    action_smoothness: torch.Tensor
    uncertainty: torch.Tensor


@contextmanager
def _temporarily_frozen(module: nn.Module) -> Iterator[None]:
    original = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original, strict=True):
            parameter.requires_grad_(requires_grad)


class DifferentiableContextRollout(nn.Module):
    """Unroll a fixed Context residual through frozen Planner-IDM and fault dynamics."""

    def __init__(
        self,
        policy: FrozenPlannerIDMContextPolicy,
        dynamics: FaultDynamicsEnsemble,
    ) -> None:
        super().__init__()
        if dynamics.config.state_dim != policy.config.obs_dim:
            raise ValueError("fault dynamics state_dim must equal Planner observation dimension")
        if dynamics.config.action_dim != policy.config.action_dim:
            raise ValueError("fault dynamics action_dim must equal IDM action dimension")
        self.policy = policy
        self.dynamics = dynamics

    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
        *,
        horizon: int,
    ) -> DifferentiableContextRolloutOutput:
        if int(horizon) <= 0:
            raise ValueError("horizon must be positive")
        delta_z = self.policy.context_encoder(observation_history, action_history, command)
        current_observation_history = observation_history
        current_action_history = action_history
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        disagreements: list[torch.Tensor] = []
        with _temporarily_frozen(self.dynamics):
            for _ in range(int(horizon)):
                policy_output = self.policy(
                    current_observation_history,
                    current_action_history,
                    command,
                    delta_z=delta_z,
                )
                action = policy_output.action
                prediction = self.dynamics(current_observation_history[:, -1], action)
                next_state = prediction.next_state
                states.append(next_state)
                actions.append(action)
                disagreements.append(prediction.disagreement)
                current_observation_history = torch.cat(
                    (current_observation_history[:, 1:], next_state[:, None, :]),
                    dim=1,
                )
                current_action_history = torch.cat(
                    (current_action_history[:, 1:], action[:, None, :]),
                    dim=1,
                )
        return DifferentiableContextRolloutOutput(
            delta_z=delta_z,
            predicted_trajectory=torch.stack(states, dim=1),
            actions=torch.stack(actions, dim=1),
            model_disagreement=torch.stack(disagreements, dim=1),
        )


def trajectory_context_loss(
    rollout: DifferentiableContextRolloutOutput,
    reference_trajectory: torch.Tensor,
    *,
    latent_weight: float = 0.0,
    action_smoothness_weight: float = 0.0,
    uncertainty_weight: float = 0.0,
) -> TrajectoryContextLoss:
    if reference_trajectory.shape != rollout.predicted_trajectory.shape:
        raise ValueError(
            "reference_trajectory shape mismatch: expected "
            f"{tuple(rollout.predicted_trajectory.shape)}, "
            f"got {tuple(reference_trajectory.shape)}"
        )
    if not torch.isfinite(reference_trajectory).all():
        raise ValueError("reference_trajectory must be finite")
    weights = (latent_weight, action_smoothness_weight, uncertainty_weight)
    if any(float(weight) < 0.0 for weight in weights):
        raise ValueError("trajectory loss weights must be non-negative")

    tracking = F.mse_loss(rollout.predicted_trajectory, reference_trajectory)
    latent = rollout.delta_z.square().mean()
    if rollout.actions.shape[1] > 1:
        action_smoothness = torch.diff(rollout.actions, dim=1).square().mean()
    else:
        action_smoothness = rollout.actions.new_zeros(())
    uncertainty = rollout.model_disagreement.mean()
    total = (
        tracking
        + float(latent_weight) * latent
        + float(action_smoothness_weight) * action_smoothness
        + float(uncertainty_weight) * uncertainty
    )
    return TrajectoryContextLoss(
        total=total,
        tracking=tracking,
        latent=latent,
        action_smoothness=action_smoothness,
        uncertainty=uncertainty,
    )

