from __future__ import annotations

import torch
from torch import nn

from unilab.algos.torch.distill.fada import _validate_finite


class DirectionBank(nn.Module):
    def __init__(self, *, axis_count: int, prediction_horizon: int, latent_dim: int) -> None:
        super().__init__()
        if min(axis_count, prediction_horizon, latent_dim) <= 0:
            raise ValueError("Direction Bank dimensions must be positive")
        self.axis_count = int(axis_count)
        self.prediction_horizon = int(prediction_horizon)
        self.latent_dim = int(latent_dim)
        self.directions: nn.Parameter = nn.Parameter(
            torch.zeros(axis_count, prediction_horizon, latent_dim)
        )
        self.normalization_scale: torch.Tensor
        self.register_buffer("normalization_scale", torch.ones(axis_count))

    def normalize_(self) -> DirectionBank:
        for axis_index in range(self.axis_count):
            self.normalize_axis_(axis_index)
        return self

    def normalize_axis_(self, axis_index: int) -> DirectionBank:
        if axis_index < 0 or axis_index >= self.axis_count:
            raise ValueError("axis_index is outside Direction Bank")
        with torch.no_grad():
            norm = self.directions[axis_index].norm()
            if bool(norm <= 0) or not bool(torch.isfinite(norm)):
                raise ValueError("Direction Bank cannot publish zero or non-finite directions")
            self.directions[axis_index].div_(norm)
            self.normalization_scale[axis_index].mul_(norm)
        return self

    def compose(
        self,
        latent: torch.Tensor,
        coefficients: torch.Tensor,
        scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tuple(self.directions.shape) != (
            self.axis_count,
            self.prediction_horizon,
            self.latent_dim,
        ):
            raise ValueError("Direction Bank directions must be [axis, horizon, latent]")
        if latent.ndim != 3 or tuple(latent.shape[1:]) != (
            self.prediction_horizon,
            self.latent_dim,
        ):
            raise ValueError("latent shape must be [batch, prediction_horizon, latent_dim]")
        if coefficients.ndim != 2 or coefficients.shape[-1] != self.axis_count:
            raise ValueError("coefficients shape must be [batch, axis_count]")
        _validate_finite("latent", latent)
        _validate_finite("coefficients", coefficients)
        if scales is None:
            scales = coefficients * self.normalization_scale.to(coefficients)[None]
        if scales.shape != coefficients.shape:
            raise ValueError("scales and coefficients must have the same shape")
        _validate_finite("scales", scales)
        return latent + torch.einsum("bm,mkd->bkd", scales, self.directions.to(latent))


class CoefficientEncoder(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        axis_count: int,
        hidden_dim: int = 128,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or layers <= 0:
            raise ValueError("Coefficient Encoder dimensions must be positive")
        self.history_length = 30
        self.axis_count = int(axis_count)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, self.history_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4 if hidden_dim % 4 == 0 else 1,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.readout = nn.Linear(hidden_dim, axis_count)

    def forward(self, state_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        if state_history.ndim != 3 or action_history.ndim != 3:
            raise ValueError("histories must be rank-3")
        if (
            state_history.shape[1] != self.history_length
            or action_history.shape[1] != self.history_length
        ):
            raise ValueError("history length must be 30")
        if state_history.shape[0] != action_history.shape[0]:
            raise ValueError("state/action history batch sizes must match")
        _validate_finite("state_history", state_history)
        _validate_finite("action_history", action_history)
        tokens = self.state_embedding(state_history) + self.action_embedding(action_history)
        tokens = self.encoder(tokens + self.position.to(dtype=tokens.dtype, device=tokens.device))
        output = self.readout(tokens.mean(dim=1))
        _validate_finite("coefficient_readout", output)
        return output
