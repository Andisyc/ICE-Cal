from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class FaultDynamicsConfig:
    state_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)

    def __post_init__(self) -> None:
        if int(self.state_dim) <= 0 or int(self.action_dim) <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if not self.hidden_dims or any(int(width) <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain only positive widths")


@dataclass(frozen=True)
class FaultTransitionBatch:
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor

    def validate(self, config: FaultDynamicsConfig) -> None:
        if self.state.ndim != 3 or self.state.shape[-1] != config.state_dim:
            raise ValueError("state must have shape (B, T, state_dim)")
        batch_size, horizon = self.state.shape[:2]
        if self.action.shape != (batch_size, horizon, config.action_dim):
            raise ValueError("action must align with state as (B, T, action_dim)")
        if self.next_state.shape != self.state.shape:
            raise ValueError("next_state must have the same shape as state")
        tensors = (self.state, self.action, self.next_state)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("fault transition tensors must share one device")
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("fault transition tensors must be finite")
        if self.state.shape[1] > 1 and not torch.equal(
            self.next_state[:, :-1], self.state[:, 1:]
        ):
            raise ValueError(
                "fault transition sequence must be contiguous: "
                "next_state[:, t] must equal state[:, t + 1]"
            )


@dataclass(frozen=True)
class FaultDynamicsPrediction:
    next_state: torch.Tensor
    member_next_state: torch.Tensor
    disagreement: torch.Tensor


@dataclass(frozen=True)
class FaultDynamicsLoss:
    total: torch.Tensor
    one_step: torch.Tensor
    multi_step: torch.Tensor


class FaultDynamicsModel(nn.Module):
    """Predict one faulty-robot state increment from a deployable state/action pair."""

    def __init__(self, config: FaultDynamicsConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        width = config.state_dim + config.action_dim
        for hidden_dim in config.hidden_dims:
            layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
            width = hidden_dim
        layers.append(nn.Linear(width, config.state_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state + self.network(torch.cat((state, action), dim=-1))


class FaultDynamicsEnsemble(nn.Module):
    def __init__(self, config: FaultDynamicsConfig, *, ensemble_size: int = 5) -> None:
        super().__init__()
        if int(ensemble_size) < 2:
            raise ValueError("ensemble_size must be at least two")
        self.config = config
        self.members = nn.ModuleList(
            FaultDynamicsModel(config) for _ in range(int(ensemble_size))
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> FaultDynamicsPrediction:
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError("state must have shape (B, state_dim)")
        if action.shape != (state.shape[0], self.config.action_dim):
            raise ValueError("action must have shape (B, action_dim)")
        member_next = torch.stack([member(state, action) for member in self.members], dim=0)
        mean_next = member_next.mean(dim=0)
        disagreement = member_next.var(dim=0, unbiased=False).mean(dim=-1)
        return FaultDynamicsPrediction(
            next_state=mean_next,
            member_next_state=member_next,
            disagreement=disagreement,
        )


def fault_dynamics_loss(
    ensemble: FaultDynamicsEnsemble,
    batch: FaultTransitionBatch,
    *,
    rollout_horizon: int,
    multi_step_weight: float = 1.0,
) -> FaultDynamicsLoss:
    batch.validate(ensemble.config)
    if not 1 <= int(rollout_horizon) <= batch.state.shape[1]:
        raise ValueError("rollout_horizon must be within the transition sequence")
    member_predictions = torch.stack(
        [
            member(batch.state, batch.action)
            for member in ensemble.members
        ],
        dim=0,
    )
    one_step = F.mse_loss(
        member_predictions,
        batch.next_state.unsqueeze(0).expand_as(member_predictions),
    )

    member_states = [batch.state[:, 0] for _ in ensemble.members]
    rollout_losses: list[torch.Tensor] = []
    for step in range(int(rollout_horizon)):
        for index, member in enumerate(ensemble.members):
            member_states[index] = member(member_states[index], batch.action[:, step])
            rollout_losses.append(F.mse_loss(member_states[index], batch.next_state[:, step]))
    multi_step = torch.stack(rollout_losses).mean()
    total = one_step + float(multi_step_weight) * multi_step
    return FaultDynamicsLoss(total=total, one_step=one_step, multi_step=multi_step)
