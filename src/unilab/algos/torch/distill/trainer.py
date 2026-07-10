from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DistillationBatch:
    """Student and teacher observations sampled for one behavior-distillation step."""

    student_obs: torch.Tensor
    teacher_obs: torch.Tensor
    role_labels: tuple[str, ...] | None = None
    teacher_actions: torch.Tensor | None = None
    commands: torch.Tensor | None = None
    command_intents: tuple[str, ...] | None = None


@dataclass(frozen=True)
class BehaviorDistillationStats:
    loss: float
    student_grad_norm: float
    update_count: int
    student_action_shape: tuple[int, ...]
    teacher_action_shape: tuple[int, ...]
    teacher_action_requires_grad: bool
    behavior_loss: float = 0.0
    aux_loss: float = 0.0
    role_loss: float = 0.0
    role_target_count: int = 0
    command_intent_loss: float = 0.0
    command_intent_target_count: int = 0
    expert_usage: tuple[float, ...] | None = None
    route_entropy: float | None = None
    teacher_action_source: str = "teacher"


class BehaviorDistillationTrainer:
    """Train a student actor to match detached teacher actions."""

    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_type: Literal["mse", "huber", "l1"] = "mse",
        max_grad_norm: float | None = None,
        aux_loss_coef: float = 0.0,
        role_loss_coef: float = 0.0,
        role_expert_targets: Mapping[str, int] | None = None,
        command_intent_loss_coef: float = 0.0,
        command_intent_expert_targets: Mapping[str, int] | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.max_grad_norm = max_grad_norm
        self.aux_loss_coef = float(aux_loss_coef)
        self.role_loss_coef = float(role_loss_coef)
        self.role_expert_targets = {
            str(role): int(expert_idx)
            for role, expert_idx in dict(role_expert_targets or {}).items()
        }
        self.command_intent_loss_coef = float(command_intent_loss_coef)
        self.command_intent_expert_targets = {
            str(intent): int(expert_idx)
            for intent, expert_idx in dict(command_intent_expert_targets or {}).items()
        }
        if self.aux_loss_coef < 0.0:
            raise ValueError(f"aux_loss_coef must be non-negative, got {aux_loss_coef}")
        if self.role_loss_coef < 0.0:
            raise ValueError(f"role_loss_coef must be non-negative, got {role_loss_coef}")
        if self.role_loss_coef > 0.0 and not self.role_expert_targets:
            raise ValueError("role_expert_targets must be non-empty when role_loss_coef > 0")
        if self.command_intent_loss_coef < 0.0:
            raise ValueError(
                "command_intent_loss_coef must be non-negative, "
                f"got {command_intent_loss_coef}"
            )
        if self.command_intent_loss_coef > 0.0 and not self.command_intent_expert_targets:
            raise ValueError(
                "command_intent_expert_targets must be non-empty when "
                "command_intent_loss_coef > 0"
            )
        self.update_count = 0
        self.teacher.eval()

    def _teacher_action(self, teacher_obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.teacher(teacher_obs)
        if isinstance(action, tuple):
            action = action[0]
        return cast(torch.Tensor, action).detach()

    @staticmethod
    def _cached_teacher_action(teacher_actions: torch.Tensor) -> torch.Tensor:
        if teacher_actions.ndim != 2:
            raise ValueError(f"teacher_actions must be rank-2, got shape {tuple(teacher_actions.shape)}")
        if not torch.isfinite(teacher_actions).all():
            raise ValueError("teacher_actions must contain only finite values")
        return teacher_actions.detach()

    def _loss(self, student_action: torch.Tensor, teacher_action: torch.Tensor) -> torch.Tensor:
        if student_action.shape != teacher_action.shape:
            raise ValueError(
                "student/teacher action shape mismatch: "
                f"student={tuple(student_action.shape)} teacher={tuple(teacher_action.shape)}"
            )
        if self.loss_type == "mse":
            return F.mse_loss(student_action, teacher_action)
        if self.loss_type == "huber":
            return F.smooth_l1_loss(student_action, teacher_action)
        if self.loss_type == "l1":
            return F.l1_loss(student_action, teacher_action)
        raise ValueError(f"Unsupported behavior distillation loss: {self.loss_type!r}")

    @staticmethod
    def _grad_norm(module: nn.Module) -> float:
        total = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            total += float(param.grad.detach().pow(2).sum().item())
        return float(total**0.5)

    def _student_action_and_aux(
        self,
        student_obs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[float, ...] | None,
        float | None,
        torch.Tensor | None,
    ]:
        try:
            student_output: Any = self.student(student_obs, return_diagnostics=True)
        except TypeError:
            student_output = self.student(student_obs)
        if isinstance(student_output, torch.Tensor):
            return student_output, student_output.new_zeros(()), None, None, None

        student_action = getattr(student_output, "action", None)
        if not isinstance(student_action, torch.Tensor):
            raise TypeError("student output diagnostics must expose a tensor `action`")

        aux_loss = student_action.new_zeros(())
        route_entropy: float | None = None
        route_probs = getattr(student_output, "route_probs", None)
        router_logits = getattr(student_output, "router_logits", None)
        if isinstance(route_probs, torch.Tensor):
            if route_probs.ndim != 2:
                raise ValueError(
                    f"route_probs must be rank-2, got shape {tuple(route_probs.shape)}"
                )
            num_experts = int(route_probs.shape[-1])
            target = torch.full(
                (num_experts,),
                1.0 / float(num_experts),
                dtype=route_probs.dtype,
                device=route_probs.device,
            )
            aux_loss = F.mse_loss(route_probs.mean(dim=0), target, reduction="sum")
            safe_probs = route_probs.clamp_min(1e-8)
            route_entropy = float(
                (-(safe_probs * safe_probs.log()).sum(dim=-1).mean()).detach().item()
            )

        expert_usage: tuple[float, ...] | None = None
        usage = getattr(student_output, "expert_usage", None)
        if isinstance(usage, torch.Tensor):
            expert_usage = tuple(float(value) for value in usage.detach().cpu().reshape(-1))

        if router_logits is not None and not isinstance(router_logits, torch.Tensor):
            raise TypeError("student output diagnostics `router_logits` must be a tensor")
        return student_action, aux_loss, expert_usage, route_entropy, router_logits

    def _role_router_loss(
        self,
        *,
        role_labels: tuple[str, ...] | None,
        router_logits: torch.Tensor | None,
        batch_size: int,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if self.role_loss_coef <= 0.0:
            return like.new_zeros(()), 0
        if role_labels is None:
            raise ValueError("role_labels are required when role_loss_coef > 0")
        if len(role_labels) != int(batch_size):
            raise ValueError(
                "role_labels length mismatch: "
                f"labels={len(role_labels)} batch={int(batch_size)}"
            )
        if router_logits is None:
            raise TypeError("role-conditioned router loss requires MoE router logits")
        if router_logits.ndim != 2:
            raise ValueError(f"router_logits must be rank-2, got shape {tuple(router_logits.shape)}")

        target_indices: list[int] = []
        for role in role_labels:
            role_key = str(role)
            if role_key not in self.role_expert_targets:
                raise ValueError(f"unmapped role label for role-conditioned loss: {role_key!r}")
            target_indices.append(int(self.role_expert_targets[role_key]))
        targets = torch.tensor(target_indices, dtype=torch.long, device=router_logits.device)
        if int(targets.min().item()) < 0 or int(targets.max().item()) >= int(router_logits.shape[-1]):
            raise ValueError(
                "role_expert_targets index out of range: "
                f"targets={sorted(set(target_indices))} num_experts={int(router_logits.shape[-1])}"
            )
        return F.cross_entropy(router_logits, targets), int(targets.numel())

    def _command_intent_router_loss(
        self,
        *,
        command_intents: tuple[str, ...] | None,
        router_logits: torch.Tensor | None,
        batch_size: int,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if self.command_intent_loss_coef <= 0.0:
            return like.new_zeros(()), 0
        if command_intents is None:
            raise ValueError("command_intents are required when command_intent_loss_coef > 0")
        if len(command_intents) != int(batch_size):
            raise ValueError(
                "command_intents length mismatch: "
                f"intents={len(command_intents)} batch={int(batch_size)}"
            )
        if router_logits is None:
            raise TypeError("command-intent router loss requires MoE router logits")
        if router_logits.ndim != 2:
            raise ValueError(f"router_logits must be rank-2, got shape {tuple(router_logits.shape)}")

        target_indices: list[int] = []
        for intent in command_intents:
            intent_key = str(intent)
            if intent_key not in self.command_intent_expert_targets:
                raise ValueError(
                    f"unmapped command intent for command-intent loss: {intent_key!r}"
                )
            target_indices.append(int(self.command_intent_expert_targets[intent_key]))
        targets = torch.tensor(target_indices, dtype=torch.long, device=router_logits.device)
        if int(targets.min().item()) < 0 or int(targets.max().item()) >= int(router_logits.shape[-1]):
            raise ValueError(
                "command_intent_expert_targets index out of range: "
                f"targets={sorted(set(target_indices))} num_experts={int(router_logits.shape[-1])}"
            )
        return F.cross_entropy(router_logits, targets), int(targets.numel())

    def update(self, batch: DistillationBatch) -> BehaviorDistillationStats:
        if batch.teacher_actions is None and batch.student_obs.shape[0] != batch.teacher_obs.shape[0]:
            raise ValueError(
                "student/teacher batch size mismatch: "
                f"student={batch.student_obs.shape[0]} teacher={batch.teacher_obs.shape[0]}"
            )
        if batch.teacher_actions is not None and batch.student_obs.shape[0] != batch.teacher_actions.shape[0]:
            raise ValueError(
                "student/teacher action batch size mismatch: "
                f"student={batch.student_obs.shape[0]} teacher_actions={batch.teacher_actions.shape[0]}"
            )

        self.student.train()
        teacher_action_source = "teacher"
        if batch.teacher_actions is None:
            teacher_action = self._teacher_action(batch.teacher_obs)
        else:
            teacher_action = self._cached_teacher_action(batch.teacher_actions)
            teacher_action_source = "cached"
        student_action, aux_loss, expert_usage, route_entropy, router_logits = (
            self._student_action_and_aux(batch.student_obs)
        )
        role_loss, role_target_count = self._role_router_loss(
            role_labels=batch.role_labels,
            router_logits=router_logits,
            batch_size=int(batch.student_obs.shape[0]),
            like=student_action,
        )
        command_intent_loss, command_intent_target_count = self._command_intent_router_loss(
            command_intents=batch.command_intents,
            router_logits=router_logits,
            batch_size=int(batch.student_obs.shape[0]),
            like=student_action,
        )
        behavior_loss = self._loss(student_action, teacher_action)
        loss = (
            behavior_loss
            + self.aux_loss_coef * aux_loss
            + self.role_loss_coef * role_loss
            + self.command_intent_loss_coef * command_intent_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        grad_norm = self._grad_norm(self.student)
        self.optimizer.step()
        self.update_count += 1

        return BehaviorDistillationStats(
            loss=float(loss.detach().item()),
            student_grad_norm=grad_norm,
            update_count=self.update_count,
            student_action_shape=tuple(student_action.shape),
            teacher_action_shape=tuple(teacher_action.shape),
            teacher_action_requires_grad=bool(teacher_action.requires_grad),
            behavior_loss=float(behavior_loss.detach().item()),
            aux_loss=float(aux_loss.detach().item()),
            role_loss=float(role_loss.detach().item()),
            role_target_count=role_target_count,
            command_intent_loss=float(command_intent_loss.detach().item()),
            command_intent_target_count=command_intent_target_count,
            expert_usage=expert_usage,
            route_entropy=route_entropy,
            teacher_action_source=teacher_action_source,
        )
