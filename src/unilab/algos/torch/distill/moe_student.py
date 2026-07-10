from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .models import MLPStudentPolicy, _activation


@dataclass(frozen=True)
class MoEStudentOutput:
    action: torch.Tensor
    router_logits: torch.Tensor
    route_probs: torch.Tensor
    expert_actions: torch.Tensor
    expert_usage: torch.Tensor
    selected_expert: torch.Tensor | None


def _mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    activation: str,
) -> nn.Sequential:
    activation_cls = _activation(activation)
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        out_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_cls())
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, int(output_dim)))
    return nn.Sequential(*layers)


class MoEStudentPolicy(nn.Module):
    """Small action-space MoE student for offline behavior-distillation probes."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        num_experts: int = 3,
        expert_hidden_dims: Sequence[int] = (256, 256),
        router_hidden_dims: Sequence[int] = (),
        activation: str = "elu",
        squash_action: bool = True,
        routing_mode: Literal["soft", "hard"] = "soft",
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_experts = int(num_experts)
        self.routing_mode = routing_mode
        self.router_temperature = float(router_temperature)
        if self.num_experts < 2:
            raise ValueError(f"num_experts must be >= 2, got {self.num_experts}")
        if self.routing_mode not in ("soft", "hard"):
            raise ValueError(f"Unsupported MoE routing_mode: {self.routing_mode!r}")
        if self.router_temperature <= 0.0:
            raise ValueError(
                f"router_temperature must be positive, got {self.router_temperature}"
            )

        self.router = _mlp(
            input_dim=self.obs_dim,
            output_dim=self.num_experts,
            hidden_dims=router_hidden_dims,
            activation=activation,
        )
        self.experts = nn.ModuleList(
            MLPStudentPolicy(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dims=expert_hidden_dims,
                activation=activation,
                squash_action=squash_action,
            )
            for _ in range(self.num_experts)
        )

    def _validate_obs(self, obs: torch.Tensor) -> None:
        if obs.ndim != 2:
            raise ValueError(f"Student obs must be rank-2, got shape {tuple(obs.shape)}")
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Student obs dim mismatch: expected {self.obs_dim}, got {obs.shape[-1]}"
            )

    def forward(
        self,
        obs: torch.Tensor,
        *,
        hard_routing: bool | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | MoEStudentOutput:
        self._validate_obs(obs)
        router_logits = self.router(obs)
        route_probs = F.softmax(router_logits / self.router_temperature, dim=-1)
        expert_actions = torch.stack([expert(obs) for expert in self.experts], dim=1)

        use_hard = self.routing_mode == "hard" if hard_routing is None else hard_routing
        selected_expert: torch.Tensor | None = None
        if use_hard:
            selected_expert = torch.argmax(route_probs, dim=-1)
            route_weights = F.one_hot(
                selected_expert,
                num_classes=self.num_experts,
            ).to(dtype=expert_actions.dtype)
            expert_usage = torch.bincount(
                selected_expert,
                minlength=self.num_experts,
            ).to(dtype=expert_actions.dtype)
        else:
            route_weights = route_probs
            expert_usage = route_probs.sum(dim=0)

        action = torch.sum(expert_actions * route_weights.unsqueeze(-1), dim=1)
        if not return_diagnostics:
            return action
        return MoEStudentOutput(
            action=action,
            router_logits=router_logits,
            route_probs=route_probs,
            expert_actions=expert_actions,
            expert_usage=expert_usage,
            selected_expert=selected_expert,
        )

    @torch.no_grad()
    def explore(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self(obs)

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        action = self(obs)
        if isinstance(action, MoEStudentOutput):
            return action.action
        return action

    def as_export_module(self) -> nn.Module:
        return self
