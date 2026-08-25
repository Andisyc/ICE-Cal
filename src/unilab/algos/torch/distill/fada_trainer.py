"""Phase-exclusive IDM or fixed-IDM Planner optimization owner."""

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
from .fada_training_phase import FADATrainingPhase, canonical_module_sha256


@dataclass(frozen=True)
class FADATrainingStats:
    idm_loss: float | None
    planner_loss: float | None
    idm_grad_norm: float | None
    planner_grad_norm: float | None


def _grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2))
    return math.sqrt(total)


class FADATrainer:
    """Own exactly one Eq. 4.2 IDM or fixed-IDM Eq. 4.3 Planner phase."""

    def __init__(
        self,
        policy: FADAPlannerIDMPolicy,
        *,
        phase: FADATrainingPhase,
        optimizer: torch.optim.Optimizer,
        pretrained_idm_sha256: str | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        self.policy = policy
        self.phase = FADATrainingPhase(phase)
        self.optimizer = optimizer
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.pretrained_idm_sha256 = pretrained_idm_sha256
        expected_parameters = (
            tuple(policy.idm.parameters())
            if self.phase is FADATrainingPhase.IDM_PRETRAIN
            else tuple(policy.planner.parameters())
        )
        actual_parameters = tuple(
            parameter for group in optimizer.param_groups for parameter in group["params"]
        )
        expected_ids = tuple(id(parameter) for parameter in expected_parameters)
        actual_ids = tuple(id(parameter) for parameter in actual_parameters)
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
            raise ValueError(
                f"{self.phase.value} optimizer must own exactly {self.phase.optimizer_owner} parameters"
            )
        if self.phase is FADATrainingPhase.PLANNER:
            observed = canonical_module_sha256(policy.idm)
            if pretrained_idm_sha256 is None or observed != pretrained_idm_sha256:
                raise ValueError("Planner phase requires the admitted pretrained IDM identity")
            for parameter in policy.idm.parameters():
                parameter.requires_grad_(False)
            policy.idm.eval()
        elif pretrained_idm_sha256 is not None:
            raise ValueError("IDM-pretrain phase must not declare a pretrained IDM identity")

    def _clip(self, module: nn.Module) -> None:
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)

    def _update_idm(self, batch: FADASourceBatch) -> tuple[float, float]:
        self.policy.idm.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = idm_source_loss(self.policy.idm, batch)
        loss.backward()
        self._clip(self.policy.idm)
        grad_norm = _grad_norm(self.policy.idm.parameters())
        self.optimizer.step()
        return float(loss.detach()), grad_norm

    def _update_planner(self, batch: FADASourceBatch) -> tuple[float, float]:
        self.policy.idm.eval()
        self.policy.planner.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = planner_source_loss(self.policy.planner, self.policy.idm, batch)
        loss.backward()
        self._clip(self.policy.planner)
        grad_norm = _grad_norm(self.policy.planner.parameters())
        self.optimizer.step()
        if any(parameter.grad is not None for parameter in self.policy.idm.parameters()):
            raise RuntimeError("Planner pass accumulated gradients on fixed IDM parameters")
        self.assert_phase_integrity()
        return float(loss.detach()), grad_norm

    def assert_phase_integrity(self) -> None:
        if self.phase is not FADATrainingPhase.PLANNER:
            return
        self.policy.idm.eval()
        if any(parameter.requires_grad for parameter in self.policy.idm.parameters()):
            raise RuntimeError("Planner phase requires every IDM parameter to remain frozen")
        observed = canonical_module_sha256(self.policy.idm)
        if observed != self.pretrained_idm_sha256:
            raise RuntimeError("Planner phase changed the sealed pretrained IDM identity")

    def update(
        self,
        batch: FADASourceBatch,
        *,
        updates: int = 1,
    ) -> FADATrainingStats:
        if int(updates) <= 0:
            raise ValueError("FADA phase updates must be positive")
        batch.validate(self.policy.config)
        idm_loss_value = None
        idm_grad_norm = None
        planner_loss_value = None
        planner_grad_norm = None
        for _ in range(int(updates)):
            if self.phase is FADATrainingPhase.IDM_PRETRAIN:
                idm_loss_value, idm_grad_norm = self._update_idm(batch)
            else:
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
        updates: int,
        device: str | torch.device,
        generator: torch.Generator | None = None,
        planner_scenario_ratios: Mapping[str, float] | None = None,
        planner_walk_cold_start_ratio: float = 0.5,
        planner_static_cold_start_ratio: float = 0.5,
    ) -> FADATrainingStats:
        """Run one phase while drawing a fresh replay sample for every update."""

        if int(updates) <= 0:
            raise ValueError("FADA phase updates must be positive")
        idm_loss_value = None
        idm_grad_norm = None
        planner_loss_value = None
        planner_grad_norm = None
        for _ in range(int(updates)):
            if self.phase is FADATrainingPhase.IDM_PRETRAIN:
                batch = replay.sample(batch_size, generator=generator, device=device)
                idm_loss_value, idm_grad_norm = self._update_idm(batch)
            else:
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
