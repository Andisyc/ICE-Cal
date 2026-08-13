from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from unilab.algos.torch.distill.fada import (
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlanner,
)


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class SupportContextBatch:
    target_future: torch.Tensor
    realized_state: torch.Tensor
    executed_action: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.target_future.shape[0])

    @property
    def support_length(self) -> int:
        return int(self.target_future.shape[1])

    def validate(
        self,
        config: FADAArchitectureConfig,
        *,
        support_length: int | None = None,
    ) -> SupportContextBatch:
        if self.target_future.ndim != 4:
            raise ValueError("support target_future must be rank-4 [batch, support, horizon, obs]")
        batch_size, observed_length = self.target_future.shape[:2]
        expected_target = (
            batch_size,
            observed_length,
            config.prediction_horizon,
            config.obs_dim,
        )
        if tuple(self.target_future.shape) != expected_target:
            raise ValueError(
                f"support target_future shape mismatch: expected={expected_target} "
                f"observed={tuple(self.target_future.shape)}"
            )
        if support_length is not None and observed_length != int(support_length):
            raise ValueError(
                f"support length mismatch: expected={support_length} observed={observed_length}"
            )
        expected_state = (batch_size, observed_length, config.obs_dim)
        if tuple(self.realized_state.shape) != expected_state:
            raise ValueError(
                f"support realized_state shape mismatch: expected={expected_state} "
                f"observed={tuple(self.realized_state.shape)}"
            )
        expected_action = (batch_size, observed_length, config.action_dim)
        if tuple(self.executed_action.shape) != expected_action:
            raise ValueError(
                f"support executed_action shape mismatch: expected={expected_action} "
                f"observed={tuple(self.executed_action.shape)}"
            )
        devices = {value.device for value in self.tensors()}
        dtypes = {value.dtype for value in self.tensors()}
        if len(devices) != 1:
            raise ValueError("support tensors must share one device")
        if len(dtypes) != 1:
            raise ValueError("support tensors must share one dtype")
        for name, value in zip(
            ("target_future", "realized_state", "executed_action"),
            self.tensors(),
            strict=True,
        ):
            _finite(f"support {name}", value)
        return self

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.target_future, self.realized_state, self.executed_action

    def to(self, device: str | torch.device) -> SupportContextBatch:
        return SupportContextBatch(*(value.to(device) for value in self.tensors()))

    def index_select(self, indices: torch.Tensor) -> SupportContextBatch:
        return SupportContextBatch(*(value.index_select(0, indices) for value in self.tensors()))


@dataclass(frozen=True)
class ContextQueryBatch:
    observation_history: torch.Tensor
    action_history: torch.Tensor
    command: torch.Tensor
    planner_intent: torch.Tensor
    realized_future: torch.Tensor
    executed_action_chunk: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.observation_history.shape[0])

    def validate(self, config: FADAArchitectureConfig) -> ContextQueryBatch:
        batch_size = self.batch_size
        expected = {
            "observation_history": (batch_size, config.history_length, config.obs_dim),
            "action_history": (batch_size, config.history_length, config.action_dim),
            "command": (batch_size, config.command_dim),
            "planner_intent": (batch_size, config.prediction_horizon, config.obs_dim),
            "realized_future": (batch_size, config.prediction_horizon, config.obs_dim),
            "executed_action_chunk": (
                batch_size,
                config.prediction_horizon,
                config.action_dim,
            ),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"query {name} shape mismatch: expected={shape} observed={tuple(value.shape)}"
                )
            _finite(f"query {name}", value)
        devices = {value.device for value in self.tensors()}
        dtypes = {value.dtype for value in self.tensors()}
        if len(devices) != 1:
            raise ValueError("query tensors must share one device")
        if len(dtypes) != 1:
            raise ValueError("query tensors must share one dtype")
        return self

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.observation_history,
            self.action_history,
            self.command,
            self.planner_intent,
            self.realized_future,
            self.executed_action_chunk,
        )

    def to(self, device: str | torch.device) -> ContextQueryBatch:
        return ContextQueryBatch(*(value.to(device) for value in self.tensors()))

    def index_select(self, indices: torch.Tensor) -> ContextQueryBatch:
        return ContextQueryBatch(*(value.index_select(0, indices) for value in self.tensors()))


@dataclass(frozen=True)
class SupportQueryBatch:
    support: SupportContextBatch
    query: ContextQueryBatch
    support_command: torch.Tensor
    pair_id: torch.Tensor
    support_rollout_id: torch.Tensor
    query_rollout_id: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.support.batch_size

    def validate(
        self,
        config: FADAArchitectureConfig,
        *,
        support_length: int | None = None,
    ) -> SupportQueryBatch:
        self.support.validate(config, support_length=support_length)
        self.query.validate(config)
        if self.query.batch_size != self.support.batch_size:
            raise ValueError("Support and Query batch sizes must match")
        expected_command = (self.batch_size, config.command_dim)
        if tuple(self.support_command.shape) != expected_command:
            raise ValueError(
                f"support_command shape mismatch: expected={expected_command} "
                f"observed={tuple(self.support_command.shape)}"
            )
        _finite("support_command", self.support_command)
        batch_devices = {
            self.support.target_future.device,
            self.query.observation_history.device,
            self.support_command.device,
        }
        batch_dtypes = {
            self.support.target_future.dtype,
            self.query.observation_history.dtype,
            self.support_command.dtype,
        }
        if len(batch_devices) != 1:
            raise ValueError("Support, Query, and support_command must share one device")
        if len(batch_dtypes) != 1:
            raise ValueError("Support, Query, and support_command must share one dtype")
        if not torch.equal(self.support_command, self.query.command):
            raise ValueError("Support and Query commands must match exactly")
        for name in ("pair_id", "support_rollout_id", "query_rollout_id"):
            value = getattr(self, name)
            if value.shape != (self.batch_size,) or value.dtype != torch.int64:
                raise ValueError(f"{name} must be int64 with shape [batch]")
            if value.device != self.support.target_future.device:
                raise ValueError(f"{name} must share the Support/Query device")
        if torch.any(self.support_rollout_id == self.query_rollout_id):
            raise ValueError("Support and Query must use different rollout ids")
        return self

    def to(self, device: str | torch.device) -> SupportQueryBatch:
        return SupportQueryBatch(
            support=self.support.to(device),
            query=self.query.to(device),
            support_command=self.support_command.to(device),
            pair_id=self.pair_id.to(device),
            support_rollout_id=self.support_rollout_id.to(device),
            query_rollout_id=self.query_rollout_id.to(device),
        )

    def index_select(self, indices: torch.Tensor) -> SupportQueryBatch:
        return SupportQueryBatch(
            support=self.support.index_select(indices),
            query=self.query.index_select(indices),
            support_command=self.support_command.index_select(0, indices),
            pair_id=self.pair_id.index_select(0, indices),
            support_rollout_id=self.support_rollout_id.index_select(0, indices),
            query_rollout_id=self.query_rollout_id.index_select(0, indices),
        )


@dataclass(frozen=True)
class SupportQueryContextConfig:
    support_length: int
    context_hidden_dim: int = 128
    context_layers: int = 2
    delta_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.support_length <= 0:
            raise ValueError("support_length must be positive")
        if self.context_hidden_dim <= 0 or self.context_layers <= 0:
            raise ValueError("Context dimensions must be positive")
        if not 0.0 < self.delta_scale <= 1.0:
            raise ValueError("delta_scale must be in (0, 1]")


class FADASupportContextEncoder(nn.Module):
    """Encode one full Support rollout into one condition-level IDM latent residual."""

    def __init__(
        self,
        fada_config: FADAArchitectureConfig,
        context_config: SupportQueryContextConfig,
    ) -> None:
        super().__init__()
        self.fada_config = fada_config
        self.context_config = context_config
        frame_dim = (
            fada_config.prediction_horizon * fada_config.obs_dim
            + fada_config.obs_dim
            + fada_config.action_dim
        )
        self.frame_projection = nn.Linear(frame_dim, context_config.context_hidden_dim)
        self.sequence_encoder = nn.GRU(
            input_size=context_config.context_hidden_dim,
            hidden_size=context_config.context_hidden_dim,
            num_layers=context_config.context_layers,
            batch_first=True,
        )
        self.delta_head = nn.Linear(context_config.context_hidden_dim, fada_config.hidden_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, support: SupportContextBatch) -> torch.Tensor:
        support.validate(self.fada_config, support_length=self.context_config.support_length)
        target = support.target_future.flatten(start_dim=2)
        frames = torch.cat((target, support.realized_state, support.executed_action), dim=-1)
        encoded_frames = F.gelu(self.frame_projection(frames))
        _, hidden = self.sequence_encoder(encoded_frames)
        return torch.tanh(self.delta_head(hidden[-1])) * self.context_config.delta_scale


@dataclass(frozen=True)
class ContextActionOutput:
    delta_z: torch.Tensor
    query_latent: torch.Tensor
    repaired_latent: torch.Tensor
    action_chunk: torch.Tensor

    @property
    def action(self) -> torch.Tensor:
        return self.action_chunk[:, 0]


class FrozenIDMSupportQueryPolicy(nn.Module):
    """Train Context through a frozen healthy Planner-IDM latent/action boundary."""

    def __init__(
        self,
        planner: FADAPlanner,
        idm: FADAInverseDynamicsModel,
        context_encoder: FADASupportContextEncoder,
    ) -> None:
        super().__init__()
        if planner.config != idm.config or planner.config != context_encoder.fada_config:
            raise ValueError("Planner, IDM, and Context must share one FADA architecture")
        self.planner = planner.eval()
        self.idm = idm.eval()
        self.context_encoder = context_encoder
        for module in (self.planner, self.idm):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @property
    def config(self) -> FADAArchitectureConfig:
        return self.idm.config

    def train(self, mode: bool = True) -> FrozenIDMSupportQueryPolicy:
        super().train(mode)
        self.planner.eval()
        self.idm.eval()
        return self

    def reconstruct_query(self, batch: SupportQueryBatch) -> ContextActionOutput:
        batch.validate(
            self.config,
            support_length=self.context_encoder.context_config.support_length,
        )
        delta_z = self.context_encoder(batch.support)
        query_latent = self.idm.encode_latent(
            batch.query.observation_history,
            batch.query.action_history,
            batch.query.realized_future,
        )
        repaired_latent = query_latent + delta_z[:, None, :]
        return ContextActionOutput(
            delta_z=delta_z,
            query_latent=query_latent,
            repaired_latent=repaired_latent,
            action_chunk=self.idm.decode_latent(repaired_latent),
        )

    def act_with_context(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
        delta_z: torch.Tensor,
    ) -> ContextActionOutput:
        planner_intent = self.planner(observation_history, command)
        latent = self.idm.encode_latent(observation_history, action_history, planner_intent)
        expected_delta = (observation_history.shape[0], self.config.hidden_dim)
        if tuple(delta_z.shape) != expected_delta:
            raise ValueError(
                f"delta_z shape mismatch: expected={expected_delta} observed={tuple(delta_z.shape)}"
            )
        _finite("delta_z", delta_z)
        repaired = latent + delta_z[:, None, :]
        return ContextActionOutput(
            delta_z=delta_z,
            query_latent=latent,
            repaired_latent=repaired,
            action_chunk=self.idm.decode_latent(repaired),
        )


def context_first_action_loss(
    policy: FrozenIDMSupportQueryPolicy,
    batch: SupportQueryBatch,
) -> torch.Tensor:
    output = policy.reconstruct_query(batch)
    return F.mse_loss(output.action, batch.query.executed_action_chunk[:, 0])
