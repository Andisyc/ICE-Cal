from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterator

import torch
from torch import nn

from .fada import FADAPlannerIDMPolicy, first_action_mse
from .fada_target_data import FADATargetBatch


@dataclass(frozen=True)
class FADALoRAConfig:
    """Paper LoRA values plus one explicit repository-owned injection manifest."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError(f"LoRA rank must be a positive integer, got {self.rank!r}")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0.0:
            raise ValueError(f"LoRA alpha must be finite and positive, got {self.alpha!r}")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout!r}")
        if any(not isinstance(name, str) or not name for name in self.target_modules):
            raise ValueError("LoRA target module names must be non-empty strings")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("LoRA target module manifest must contain unique names")


class FADALoRALinear(nn.Module):
    """A frozen Linear owner plus a standard input-dropout low-rank branch."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, **factory_kwargs)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, **factory_kwargs)
        self.adapter_dropout = nn.Dropout(float(dropout))
        self.scaling = float(alpha) / int(rank)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the effective weight to PyTorch Transformer eval fast paths."""

        return self.base.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_B(self.lora_A(self.adapter_dropout(inputs)))
        return self.base(inputs) + residual * self.scaling


@dataclass(frozen=True)
class FADAAdaptedPolicy:
    policy: FADAPlannerIDMPolicy
    lora_config: FADALoRAConfig


def _discover_lora_targets(policy: FADAPlannerIDMPolicy) -> tuple[str, ...]:
    targets: list[str] = []
    for name, module in policy.idm.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if name.endswith("self_attn.out_proj") or name.endswith("multihead_attn.out_proj"):
            continue
        targets.append(name)
    return tuple(targets)


def _replace_submodule(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, separator, child_name = name.rpartition(".")
    if not child_name:
        raise ValueError(f"invalid LoRA target module name {name!r}")
    parent = root.get_submodule(parent_name) if separator else root
    setattr(parent, child_name, replacement)


def inject_fada_idm_lora(
    policy: FADAPlannerIDMPolicy,
    config: FADALoRAConfig,
) -> FADAAdaptedPolicy:
    """Freeze Planner/base IDM and inject zero-delta adapters at the exact manifest."""

    if any(isinstance(module, FADALoRALinear) for module in policy.idm.modules()):
        raise ValueError("FADA policy already contains LoRA adapters")
    discovered = _discover_lora_targets(policy)
    if not discovered:
        raise ValueError("FADA IDM exposes no directly invoked Linear LoRA targets")
    if config.target_modules and tuple(config.target_modules) != discovered:
        raise ValueError(
            "LoRA target module manifest does not match the complete IDM injection owner: "
            f"expected={discovered} observed={tuple(config.target_modules)}"
        )
    resolved = replace(config, target_modules=discovered)

    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for name in discovered:
        module = policy.idm.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"LoRA target manifest entry is not Linear: {name}")
        _replace_submodule(
            policy.idm,
            name,
            FADALoRALinear(
                module,
                rank=resolved.rank,
                alpha=resolved.alpha,
                dropout=resolved.dropout,
            ),
        )
    assert_fada_adaptation_parameter_ownership(policy, resolved)
    return FADAAdaptedPolicy(policy=policy, lora_config=resolved)


def fada_adapter_named_parameters(
    policy: FADAPlannerIDMPolicy,
) -> Iterator[tuple[str, nn.Parameter]]:
    for name, parameter in policy.named_parameters():
        if name.endswith(("lora_A.weight", "lora_B.weight")):
            yield name, parameter


def fada_adapter_parameters(policy: FADAPlannerIDMPolicy) -> Iterator[nn.Parameter]:
    for _name, parameter in fada_adapter_named_parameters(policy):
        yield parameter


def assert_fada_adaptation_parameter_ownership(
    policy: FADAPlannerIDMPolicy,
    config: FADALoRAConfig,
) -> None:
    adapters = dict(fada_adapter_named_parameters(policy))
    if not adapters:
        raise ValueError("FADA adaptation requires at least one LoRA parameter")
    observed_targets = tuple(
        name.removeprefix("idm.").removesuffix(".lora_A.weight")
        for name in adapters
        if name.endswith("lora_A.weight")
    )
    if observed_targets != tuple(config.target_modules):
        raise ValueError(
            "LoRA parameter ownership does not match the persisted target manifest: "
            f"expected={config.target_modules} observed={observed_targets}"
        )
    trainable = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    if trainable != set(adapters):
        raise ValueError(
            "FADA adaptation must expose only adapter parameters as trainable: "
            f"expected={sorted(adapters)} observed={sorted(trainable)}"
        )
    if any(parameter.requires_grad for parameter in policy.planner.parameters()):
        raise ValueError("FADA adaptation Planner parameters must remain frozen")


@dataclass(frozen=True)
class FADATargetSplit:
    train_indices: torch.Tensor
    validation_indices: torch.Tensor


def split_fada_target_batch(
    batch: FADATargetBatch,
    *,
    validation_fraction: float,
    seed: int,
) -> FADATargetSplit:
    """Split by episode identity so overlapping lifecycle windows never cross splits."""

    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"split seed must be a non-negative integer, got {seed!r}")
    episodes = torch.unique(batch.episode_id.detach().to("cpu"), sorted=True)
    if int(episodes.numel()) < 2:
        raise ValueError("FADA target split requires at least two episodes")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    episode_order = episodes[torch.randperm(int(episodes.numel()), generator=generator)]
    validation_count = max(1, min(int(episodes.numel()) - 1, round(len(episodes) * fraction)))
    validation_episodes = episode_order[:validation_count]
    validation_mask = torch.zeros_like(batch.episode_id, dtype=torch.bool)
    for episode in validation_episodes:
        validation_mask |= batch.episode_id == episode.to(batch.episode_id.device)
    validation_indices = torch.nonzero(validation_mask, as_tuple=False).flatten().to(torch.int64)
    train_indices = torch.nonzero(~validation_mask, as_tuple=False).flatten().to(torch.int64)
    if train_indices.numel() == 0 or validation_indices.numel() == 0:
        raise ValueError("FADA target split must produce non-empty train and validation rows")
    return FADATargetSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def select_fada_target_rows(batch: FADATargetBatch, indices: torch.Tensor) -> FADATargetBatch:
    if indices.ndim != 1 or indices.dtype != torch.int64 or indices.numel() == 0:
        raise ValueError("FADA target row indices must be non-empty rank-1 int64")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("FADA target row indices must be unique")
    if int(indices.min()) < 0 or int(indices.max()) >= int(batch.episode_id.shape[0]):
        raise ValueError("FADA target row index is out of range")
    return FADATargetBatch(
        **{
            field: getattr(batch, field).index_select(0, indices.to(getattr(batch, field).device))
            for field in FADATargetBatch.__dataclass_fields__
        }
    )


def _batch_to_device(batch: FADATargetBatch, device: torch.device) -> FADATargetBatch:
    return FADATargetBatch(
        **{
            field: getattr(batch, field).to(device)
            for field in FADATargetBatch.__dataclass_fields__
        }
    )


def fada_adaptation_loss(
    policy: FADAPlannerIDMPolicy,
    batch: FADATargetBatch,
) -> torch.Tensor:
    device = next(policy.parameters()).device
    current = _batch_to_device(batch.validate(policy.config), device)
    predicted = policy.idm(
        current.observation_history,
        current.action_history,
        current.realized_future,
    )
    return first_action_mse(predicted, current.executed_action_chunk)


@dataclass(frozen=True)
class FADAAdaptationStats:
    loss: float
    grad_norm: float
    optimizer_steps: int


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> list[int]:
    return [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]


class FADAAdaptationTrainer:
    """Own exactly one LoRA-only target-domain update transaction."""

    def __init__(
        self,
        policy: FADAPlannerIDMPolicy,
        optimizer: torch.optim.Optimizer,
        *,
        lora_config: FADALoRAConfig,
        max_grad_norm: float | None = None,
    ) -> None:
        assert_fada_adaptation_parameter_ownership(policy, lora_config)
        adapter_ids = [id(parameter) for parameter in fada_adapter_parameters(policy)]
        optimizer_ids = _optimizer_parameter_ids(optimizer)
        if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != set(adapter_ids):
            raise ValueError(
                "FADA adaptation optimizer must own only adapter parameters exactly once"
            )
        if max_grad_norm is not None and (
            not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0.0
        ):
            raise ValueError("max_grad_norm must be finite and positive when provided")
        self.policy = policy
        self.optimizer = optimizer
        self.lora_config = lora_config
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

    def update(self, batch: FADATargetBatch) -> FADAAdaptationStats:
        self.policy.planner.eval()
        self.policy.idm.train()
        adapters = list(fada_adapter_parameters(self.policy))
        self.optimizer.zero_grad(set_to_none=True)
        loss = fada_adaptation_loss(self.policy, batch)
        loss.backward()
        if any(parameter.grad is None for parameter in adapters):
            raise RuntimeError("FADA adaptation produced a missing adapter gradient")
        if any(
            parameter.grad is not None
            for name, parameter in self.policy.named_parameters()
            if not name.endswith(("lora_A.weight", "lora_B.weight"))
        ):
            raise RuntimeError("FADA adaptation accumulated a gradient on a frozen parameter")
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(adapters, self.max_grad_norm)
        squared = sum(
            float(torch.sum(torch.square(parameter.grad.detach())))
            for parameter in adapters
            if parameter.grad is not None
        )
        grad_norm = math.sqrt(squared)
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError(
                f"FADA adaptation gradient norm must be finite and positive: {grad_norm}"
            )
        self.optimizer.step()
        return FADAAdaptationStats(
            loss=float(loss.detach()),
            grad_norm=grad_norm,
            optimizer_steps=1,
        )
