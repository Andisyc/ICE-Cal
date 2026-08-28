"""Ordered IDM and fixed-IDM Planner optimization owner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .fada import (
    FADAPlannerIDMPolicy,
    FADASourceBatch,
    idm_source_loss,
    planner_source_loss,
)
from .fada_replay import FADAReplayBuffer


@dataclass(frozen=True)
class FADATrainingStats:
    idm_loss: float
    planner_loss: float
    idm_grad_norm: float
    planner_grad_norm: float


def _grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2))
    return math.sqrt(total)


class FADATrainer:
    """Own the ordered Eq. 4.2 IDM pass and fixed-IDM Eq. 4.3 Planner pass."""

    def __init__(
        self,
        policy: FADAPlannerIDMPolicy,
        *,
        idm_optimizer: torch.optim.Optimizer,
        planner_optimizer: torch.optim.Optimizer,
        max_grad_norm: float | None = None,
    ) -> None:
        self.policy = policy
        self.idm_optimizer = idm_optimizer
        self.planner_optimizer = planner_optimizer
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self._require_exact_optimizer_owner(idm_optimizer, policy.idm, owner="idm")
        self._require_exact_optimizer_owner(planner_optimizer, policy.planner, owner="planner")

    @staticmethod
    def _require_exact_optimizer_owner(
        optimizer: torch.optim.Optimizer,
        module: nn.Module,
        *,
        owner: str,
    ) -> None:
        expected = {id(parameter) for parameter in module.parameters()}
        actual = [
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        ]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"{owner} optimizer must own exactly {owner} parameters")

    def _clip(self, module: nn.Module) -> None:
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)

    def _update_idm(self, batch: FADASourceBatch) -> tuple[float, float]:
        self.policy.idm.train()
        self.idm_optimizer.zero_grad(set_to_none=True)
        loss = idm_source_loss(self.policy.idm, batch)
        loss.backward()
        self._clip(self.policy.idm)
        grad_norm = _grad_norm(self.policy.idm.parameters())
        self.idm_optimizer.step()
        return float(loss.detach()), grad_norm

    def _update_planner(self, batch: FADASourceBatch) -> tuple[float, float]:
        self.policy.idm.eval()
        self.policy.planner.train()
        self.idm_optimizer.zero_grad(set_to_none=True)
        self.planner_optimizer.zero_grad(set_to_none=True)
        loss = planner_source_loss(self.policy.planner, self.policy.idm, batch)
        loss.backward()
        self._clip(self.policy.planner)
        grad_norm = _grad_norm(self.policy.planner.parameters())
        self.planner_optimizer.step()
        if any(parameter.grad is not None for parameter in self.policy.idm.parameters()):
            raise RuntimeError("Planner pass accumulated gradients on fixed IDM parameters")
        return float(loss.detach()), grad_norm

    def update(
        self,
        batch: FADASourceBatch,
        *,
        idm_updates: int = 1,
        planner_updates: int = 1,
    ) -> FADATrainingStats:
        if int(idm_updates) <= 0 or int(planner_updates) < 0:
            raise ValueError("IDM updates must be positive and Planner updates non-negative")
        batch.validate(self.policy.config)
        idm_loss_value = 0.0
        idm_grad_norm = 0.0
        for _ in range(int(idm_updates)):
            idm_loss_value, idm_grad_norm = self._update_idm(batch)
        planner_loss_value = 0.0
        planner_grad_norm = 0.0
        for _ in range(int(planner_updates)):
            planner_loss_value, planner_grad_norm = self._update_planner(batch)

        return FADATrainingStats(
            idm_loss=idm_loss_value,
            planner_loss=planner_loss_value,
            idm_grad_norm=idm_grad_norm,
            planner_grad_norm=planner_grad_norm,
        )

    def update_from_replay(
        self,
        replay: FADAReplayBuffer,
        *,
        batch_size: int,
        idm_updates: int,
        planner_updates: int,
        device: str | torch.device,
        generator: torch.Generator | None = None,
        planner_scenario_ratios: Mapping[str, float] | None = None,
        planner_walk_cold_start_ratio: float = 0.5,
        planner_static_cold_start_ratio: float = 0.5,
    ) -> FADATrainingStats:
        """Run one phase while drawing a fresh replay sample for every update."""

        if int(idm_updates) <= 0 or int(planner_updates) < 0:
            raise ValueError("IDM updates must be positive and Planner updates non-negative")
        idm_loss_value = 0.0
        idm_grad_norm = 0.0
        for _ in range(int(idm_updates)):
            batch = replay.sample(batch_size, generator=generator, device=device)
            idm_loss_value, idm_grad_norm = self._update_idm(batch)
        planner_loss_value = 0.0
        planner_grad_norm = 0.0
        for _ in range(int(planner_updates)):
            batch = (
                replay.sample(batch_size, generator=generator, device=device)
                if planner_scenario_ratios is None
                else replay.sample_planner(
                    batch_size,
                    scenario_ratios=planner_scenario_ratios,
                    walk_cold_start_ratio=planner_walk_cold_start_ratio,
                    static_cold_start_ratio=planner_static_cold_start_ratio,
                    generator=generator,
                    device=device,
                )
            )
            planner_loss_value, planner_grad_norm = self._update_planner(batch)
        return FADATrainingStats(
            idm_loss=idm_loss_value,
            planner_loss=planner_loss_value,
            idm_grad_norm=idm_grad_norm,
            planner_grad_norm=planner_grad_norm,
        )
