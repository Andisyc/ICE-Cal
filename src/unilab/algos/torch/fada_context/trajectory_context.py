from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from unilab.algos.torch.distill.fada import (
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlanner,
)


@dataclass(frozen=True)
class ContextEncoderConfig:
    obs_dim: int
    action_dim: int
    command_dim: int
    history_length: int
    prediction_horizon: int
    hidden_dim: int = 128
    num_layers: int = 2
    residual_scale: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "obs_dim",
            "action_dim",
            "command_dim",
            "history_length",
            "prediction_horizon",
            "hidden_dim",
            "num_layers",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < float(self.residual_scale) <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")

    @classmethod
    def from_fada(
        cls,
        config: FADAArchitectureConfig,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        residual_scale: float = 0.1,
    ) -> ContextEncoderConfig:
        return cls(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            command_dim=config.command_dim,
            history_length=config.history_length,
            prediction_horizon=config.prediction_horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            residual_scale=residual_scale,
        )


@dataclass(frozen=True)
class ContextPolicyOutput:
    delta_z: torch.Tensor
    nominal_future: torch.Tensor
    repaired_future: torch.Tensor
    action_chunk: torch.Tensor

    @property
    def action(self) -> torch.Tensor:
        return self.action_chunk[:, 0]


def _validate_context_inputs(
    config: ContextEncoderConfig,
    observation_history: torch.Tensor,
    action_history: torch.Tensor,
    command: torch.Tensor,
) -> None:
    if observation_history.ndim != 3 or observation_history.shape[1:] != (
        config.history_length,
        config.obs_dim,
    ):
        raise ValueError(
            "observation_history shape mismatch: expected "
            f"(B, {config.history_length}, {config.obs_dim}), "
            f"got {tuple(observation_history.shape)}"
        )
    batch_size = observation_history.shape[0]
    expected_action = (batch_size, config.history_length, config.action_dim)
    if action_history.shape != expected_action:
        raise ValueError(
            f"action_history shape mismatch: expected {expected_action}, "
            f"got {tuple(action_history.shape)}"
        )
    expected_command = (batch_size, config.command_dim)
    if command.shape != expected_command:
        raise ValueError(
            f"command shape mismatch: expected {expected_command}, got {tuple(command.shape)}"
        )
    tensors = (observation_history, action_history, command)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("Context inputs must share one device")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("Context inputs must share one dtype")
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("Context inputs must be finite")


class FADATrajectoryContextEncoder(nn.Module):
    """Encode deployable observation/action history into a Planner-future residual."""

    def __init__(self, config: ContextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        frame_dim = config.obs_dim + config.action_dim + config.command_dim
        self.history_encoder = nn.GRU(
            input_size=frame_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.residual_head = nn.Linear(
            config.hidden_dim,
            config.prediction_horizon * config.obs_dim,
        )

    def forward(self, *deployable_tensors: torch.Tensor) -> torch.Tensor:
        if len(deployable_tensors) != 3:
            raise TypeError(
                "Context Encoder accepts exactly three deployable tensors: "
                "observation_history, action_history, and command"
            )
        observation_history, action_history, command = deployable_tensors
        _validate_context_inputs(self.config, observation_history, action_history, command)
        command_history = command[:, None, :].expand(-1, self.config.history_length, -1)
        frames = torch.cat((observation_history, action_history, command_history), dim=-1)
        _, hidden = self.history_encoder(frames)
        residual = self.residual_head(hidden[-1]).reshape(
            observation_history.shape[0],
            self.config.prediction_horizon,
            self.config.obs_dim,
        )
        return torch.tanh(residual) * self.config.residual_scale


class FrozenPlannerIDMContextPolicy(nn.Module):
    """Inject Context residuals into a frozen FADA Planner-IDM action path."""

    def __init__(
        self,
        planner: FADAPlanner,
        idm: FADAInverseDynamicsModel,
        context_encoder: FADATrajectoryContextEncoder,
    ) -> None:
        super().__init__()
        if planner.config != idm.config:
            raise ValueError("Planner and IDM must share one FADA architecture config")
        expected = ContextEncoderConfig.from_fada(
            planner.config,
            hidden_dim=context_encoder.config.hidden_dim,
            num_layers=context_encoder.config.num_layers,
            residual_scale=context_encoder.config.residual_scale,
        )
        if context_encoder.config != expected:
            raise ValueError("Context Encoder dimensions must match the Planner-IDM config")
        self.planner = planner.eval()
        self.idm = idm.eval()
        self.context_encoder = context_encoder
        for module in (self.planner, self.idm):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @property
    def config(self) -> FADAArchitectureConfig:
        return self.planner.config

    def train(self, mode: bool = True) -> FrozenPlannerIDMContextPolicy:
        super().train(mode)
        self.planner.eval()
        self.idm.eval()
        return self

    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
        *,
        delta_z: torch.Tensor | None = None,
    ) -> ContextPolicyOutput:
        nominal_future = self.planner(observation_history, command)
        resolved_delta_z = (
            self.context_encoder(observation_history, action_history, command)
            if delta_z is None
            else delta_z
        )
        if resolved_delta_z.shape != nominal_future.shape:
            raise ValueError(
                f"delta_z shape mismatch: expected {tuple(nominal_future.shape)}, "
                f"got {tuple(resolved_delta_z.shape)}"
            )
        if not torch.isfinite(resolved_delta_z).all():
            raise ValueError("delta_z must be finite")
        repaired_future = nominal_future + resolved_delta_z
        action_chunk = self.idm(observation_history, action_history, repaired_future)
        return ContextPolicyOutput(
            delta_z=resolved_delta_z,
            nominal_future=nominal_future,
            repaired_future=repaired_future,
            action_chunk=action_chunk,
        )
