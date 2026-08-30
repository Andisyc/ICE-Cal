from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from unilab.algos.torch.distill.learning.diagnostics import (
    _ORIGINAL_TORCH_TENSOR,
    _ORIGINAL_TYPE,
    _distill_runtime_debug_enabled,
    _emit_trainer_runtime,
    _label_counts,
    _runtime_trace_update,
    _safe_runtime_repr,
    _target_index_list_runtime_snapshot,
    _tensor_runtime_snapshot,
    append_runtime_target_index,
)
from unilab.algos.torch.distill.learning.routing import (
    decode_student_output,
    resolve_label_target_indices,
)
from unilab.algos.torch.distill.observability.performance import (
    DistillationStageObservationAccumulator,
)


@dataclass(frozen=True)
class DistillationBatch:
    """Student and teacher observations sampled for one behavior-distillation step."""

    student_obs: torch.Tensor
    teacher_obs: torch.Tensor
    role_labels: tuple[str, ...] | None = None
    teacher_actions: torch.Tensor | None = None
    commands: torch.Tensor | None = None
    target_height: torch.Tensor | None = None
    command_intents: tuple[str, ...] | None = None
    scenario_labels: tuple[str, ...] | None = None
    transition_ages: torch.Tensor | None = None
    command_before: torch.Tensor | None = None
    command_after: torch.Tensor | None = None


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
    behavior_action_shape: tuple[int, ...] = ()
    behavior_action_source: str = "student_action"
    behavior_target_count: int = 0


@dataclass(frozen=True)
class _TrainerForwardPass:
    loss: torch.Tensor
    teacher_action: torch.Tensor
    teacher_action_source: str
    student_action: torch.Tensor
    behavior_action: torch.Tensor
    behavior_action_source: str
    behavior_target_count: int
    selected_expert_targets: torch.Tensor | None
    behavior_loss: torch.Tensor
    aux_loss: torch.Tensor
    role_loss: torch.Tensor
    role_target_count: int
    command_intent_loss: torch.Tensor
    command_intent_target_count: int
    expert_usage: tuple[float, ...] | None
    route_entropy: float | None


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
        student_init_metadata: Mapping[str, Any] | None = None,
        expert_behavior_loss_source: Literal["auto", "none", "role", "command_intent"] = "auto",
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
        self.student_init_metadata = dict(student_init_metadata or {})
        self.expert_behavior_loss_source = str(expert_behavior_loss_source)
        if self.aux_loss_coef < 0.0:
            raise ValueError(f"aux_loss_coef must be non-negative, got {aux_loss_coef}")
        if self.expert_behavior_loss_source not in ("auto", "none", "role", "command_intent"):
            raise ValueError(
                "expert_behavior_loss_source must be one of auto/none/role/command_intent, "
                f"got {expert_behavior_loss_source!r}"
            )
        if self.role_loss_coef < 0.0:
            raise ValueError(f"role_loss_coef must be non-negative, got {role_loss_coef}")
        if self.role_loss_coef > 0.0 and not self.role_expert_targets:
            raise ValueError("role_expert_targets must be non-empty when role_loss_coef > 0")
        if self.command_intent_loss_coef < 0.0:
            raise ValueError(
                f"command_intent_loss_coef must be non-negative, got {command_intent_loss_coef}"
            )
        if self.command_intent_loss_coef > 0.0 and not self.command_intent_expert_targets:
            raise ValueError(
                "command_intent_expert_targets must be non-empty when command_intent_loss_coef > 0"
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
            raise ValueError(
                f"teacher_actions must be rank-2, got shape {tuple(teacher_actions.shape)}"
            )
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
        torch.Tensor | None,
    ]:
        try:
            student_output: Any = self.student(student_obs, return_diagnostics=True)
        except TypeError:
            student_output = self.student(student_obs)
        return decode_student_output(student_output)

    def _target_indices_from_labels(
        self,
        *,
        labels: tuple[str, ...] | None,
        targets: Mapping[str, int],
        batch_size: int,
        num_experts: int,
        label_name: str,
        required: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        update_number = self.update_count + 1
        trace_update = _runtime_trace_update(update_number)

        def append_target(
            target_indices: list[int], raw_target: Any, label_key: str, row_index: int
        ) -> None:
            self._append_runtime_target_index(
                target_indices=target_indices,
                raw_target=raw_target,
                label_name=f"expert_behavior_{label_name}",
                label_key=label_key,
                row_index=row_index,
                update_number=update_number,
                trace_row=(
                    trace_update
                    and labels is not None
                    and (row_index < 2 or row_index == len(labels) - 1)
                ),
            )

        resolved_indices = resolve_label_target_indices(
            labels=labels,
            targets=targets,
            batch_size=batch_size,
            num_experts=num_experts,
            label_name=label_name,
            required=required,
            append_target=append_target,
            validate_range=False,
        )
        if resolved_indices is None:
            return None
        target_indices = list(resolved_indices)
        tensor_fn = torch.tensor
        try:
            target_tensor = tensor_fn(target_indices, dtype=torch.long, device=device)
        except Exception as error:
            _emit_trainer_runtime(
                "target_index/tensor_failure",
                update_number=update_number,
                label_name=label_name,
                batch_size=batch_size,
                num_experts=num_experts,
                device_repr=_safe_runtime_repr(device),
                dtype_repr=_safe_runtime_repr(torch.long),
                target_mapping=[
                    {
                        "label": _safe_runtime_repr(target_label),
                        "raw_type": _ORIGINAL_TYPE(raw_target).__name__,
                        "raw_repr": _safe_runtime_repr(raw_target),
                    }
                    for target_label, raw_target in targets.items()
                ],
                target_indices=_target_index_list_runtime_snapshot(target_indices),
                torch_tensor_type=_ORIGINAL_TYPE(tensor_fn).__name__,
                torch_tensor_repr=_safe_runtime_repr(tensor_fn),
                torch_tensor_callable=callable(tensor_fn),
                torch_tensor_is_original=tensor_fn is _ORIGINAL_TORCH_TENSOR,
                error_type=_ORIGINAL_TYPE(error).__name__,
                error_repr=_safe_runtime_repr(error),
            )
            raise
        if int(target_tensor.min().item()) < 0 or int(target_tensor.max().item()) >= int(
            num_experts
        ):
            raise ValueError(
                f"{label_name}_expert_targets index out of range: "
                f"targets={sorted(set(target_indices))} num_experts={int(num_experts)}"
            )
        return target_tensor

    def _append_runtime_target_index(
        self,
        *,
        target_indices: list[int],
        raw_target: Any,
        label_name: str,
        label_key: str,
        row_index: int,
        update_number: int,
        trace_row: bool,
    ) -> None:
        append_runtime_target_index(
            target_indices=target_indices,
            raw_target=raw_target,
            label_name=label_name,
            label_key=label_key,
            row_index=row_index,
            update_number=update_number,
            trace_row=trace_row,
        )

    def _expert_behavior_action(
        self,
        *,
        student_action: torch.Tensor,
        expert_actions: torch.Tensor | None,
        role_labels: tuple[str, ...] | None,
        command_intents: tuple[str, ...] | None,
    ) -> tuple[torch.Tensor, str, int, torch.Tensor | None]:
        if self.expert_behavior_loss_source == "none" or expert_actions is None:
            return student_action, "student_action", 0, None
        if expert_actions.ndim != 3:
            raise ValueError(
                f"expert_actions must be rank-3, got shape {tuple(expert_actions.shape)}"
            )
        if expert_actions.shape[0] != student_action.shape[0]:
            raise ValueError(
                "expert_actions batch size mismatch: "
                f"expert_actions={int(expert_actions.shape[0])} student={int(student_action.shape[0])}"
            )
        if expert_actions.shape[-1] != student_action.shape[-1]:
            raise ValueError(
                "expert_actions action dim mismatch: "
                f"expert_actions={int(expert_actions.shape[-1])} student={int(student_action.shape[-1])}"
            )

        batch_size = int(student_action.shape[0])
        num_experts = int(expert_actions.shape[1])
        command_targets = self._target_indices_from_labels(
            labels=command_intents,
            targets=self.command_intent_expert_targets,
            batch_size=batch_size,
            num_experts=num_experts,
            label_name="command_intent",
            required=self.expert_behavior_loss_source == "command_intent",
            device=expert_actions.device,
        )
        role_targets = self._target_indices_from_labels(
            labels=role_labels,
            targets=self.role_expert_targets,
            batch_size=batch_size,
            num_experts=num_experts,
            label_name="role",
            required=self.expert_behavior_loss_source == "role",
            device=expert_actions.device,
        )
        if command_targets is not None and role_targets is not None:
            if not torch.equal(command_targets, role_targets):
                raise ValueError(
                    "command_intent_expert_targets conflict with role_expert_targets "
                    "for expert behavior loss"
                )

        selected_targets: torch.Tensor | None
        source: str
        if self.expert_behavior_loss_source == "command_intent":
            selected_targets = command_targets
            source = "command_intent_expert"
        elif self.expert_behavior_loss_source == "role":
            selected_targets = role_targets
            source = "role_expert"
        elif command_targets is not None:
            selected_targets = command_targets
            source = "command_intent_expert"
        elif role_targets is not None:
            selected_targets = role_targets
            source = "role_expert"
        else:
            return student_action, "student_action", 0, None
        if selected_targets is None:
            return student_action, "student_action", 0, None

        row_indices = torch.arange(batch_size, device=expert_actions.device)
        selected_action = expert_actions[row_indices, selected_targets]
        return selected_action, source, int(selected_targets.numel()), selected_targets

    def _clear_inactive_expert_grads(self, selected_targets: torch.Tensor | None) -> None:
        if selected_targets is None:
            return
        experts = cast(Any, self.student).experts
        active_experts = {int(index) for index in selected_targets.detach().unique().tolist()}
        for expert_index, expert in enumerate(experts):
            if expert_index not in active_experts:
                for parameter in expert.parameters():
                    parameter.grad = None

    def _role_router_loss(
        self,
        *,
        role_labels: tuple[str, ...] | None,
        router_logits: torch.Tensor | None,
        batch_size: int,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        update_number = self.update_count + 1
        trace_update = _runtime_trace_update(update_number)
        if self.role_loss_coef <= 0.0:
            return like.new_zeros(()), 0
        if role_labels is None:
            raise ValueError("role_labels are required when role_loss_coef > 0")
        if len(role_labels) != int(batch_size):
            raise ValueError(
                f"role_labels length mismatch: labels={len(role_labels)} batch={int(batch_size)}"
            )
        if router_logits is None:
            raise TypeError("role-conditioned router loss requires MoE router logits")
        if router_logits.ndim != 2:
            raise ValueError(
                f"router_logits must be rank-2, got shape {tuple(router_logits.shape)}"
            )

        if trace_update:
            _emit_trainer_runtime(
                "role/entry",
                update_number=update_number,
                batch_size=batch_size,
                label_counts=_label_counts(role_labels),
                expert_targets=dict(self.role_expert_targets),
                router_logits=_tensor_runtime_snapshot(router_logits),
            )
        target_indices: list[int] = []
        for row_index, role in enumerate(role_labels):
            role_key = str(role)
            if role_key not in self.role_expert_targets:
                raise ValueError(f"unmapped role label for role-conditioned loss: {role_key!r}")
            self._append_runtime_target_index(
                target_indices=target_indices,
                raw_target=self.role_expert_targets[role_key],
                label_name="role",
                label_key=role_key,
                row_index=row_index,
                update_number=update_number,
                trace_row=(trace_update and (row_index < 2 or row_index == len(role_labels) - 1)),
            )
        targets = torch.tensor(target_indices, dtype=torch.long, device=router_logits.device)
        if trace_update:
            _emit_trainer_runtime(
                "role/after_target_tensor",
                update_number=update_number,
                target_indices=tuple(target_indices),
                targets=_tensor_runtime_snapshot(targets),
            )
        if int(targets.min().item()) < 0 or int(targets.max().item()) >= int(
            router_logits.shape[-1]
        ):
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
        update_number = self.update_count + 1
        trace_update = _runtime_trace_update(update_number)
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
            raise ValueError(
                f"router_logits must be rank-2, got shape {tuple(router_logits.shape)}"
            )

        if trace_update:
            _emit_trainer_runtime(
                "command_intent/entry",
                update_number=update_number,
                batch_size=batch_size,
                label_counts=_label_counts(command_intents),
                expert_targets=dict(self.command_intent_expert_targets),
                router_logits=_tensor_runtime_snapshot(router_logits),
            )
        target_indices: list[int] = []
        for row_index, intent in enumerate(command_intents):
            intent_key = str(intent)
            if intent_key not in self.command_intent_expert_targets:
                raise ValueError(f"unmapped command intent for command-intent loss: {intent_key!r}")
            self._append_runtime_target_index(
                target_indices=target_indices,
                raw_target=self.command_intent_expert_targets[intent_key],
                label_name="command_intent",
                label_key=intent_key,
                row_index=row_index,
                update_number=update_number,
                trace_row=(
                    trace_update and (row_index < 2 or row_index == len(command_intents) - 1)
                ),
            )
        targets = torch.tensor(target_indices, dtype=torch.long, device=router_logits.device)
        if trace_update:
            _emit_trainer_runtime(
                "command_intent/after_target_tensor",
                update_number=update_number,
                target_indices=tuple(target_indices),
                targets=_tensor_runtime_snapshot(targets),
            )
        if int(targets.min().item()) < 0 or int(targets.max().item()) >= int(
            router_logits.shape[-1]
        ):
            raise ValueError(
                "command_intent_expert_targets index out of range: "
                f"targets={sorted(set(target_indices))} num_experts={int(router_logits.shape[-1])}"
            )
        return F.cross_entropy(router_logits, targets), int(targets.numel())

    @staticmethod
    def _validate_update_batch(batch: DistillationBatch) -> None:
        if (
            batch.teacher_actions is None
            and batch.student_obs.shape[0] != batch.teacher_obs.shape[0]
        ):
            raise ValueError(
                "student/teacher batch size mismatch: "
                f"student={batch.student_obs.shape[0]} teacher={batch.teacher_obs.shape[0]}"
            )
        if (
            batch.teacher_actions is not None
            and batch.student_obs.shape[0] != batch.teacher_actions.shape[0]
        ):
            raise ValueError(
                "student/teacher action batch size mismatch: "
                f"student={batch.student_obs.shape[0]} "
                f"teacher_actions={batch.teacher_actions.shape[0]}"
            )

    def _forward_update(
        self,
        batch: DistillationBatch,
        *,
        update_number: int,
        trace_update: bool,
        performance: DistillationStageObservationAccumulator | None,
    ) -> _TrainerForwardPass:
        span = nullcontext() if performance is None else performance.measure("learner_forward")
        with span:
            teacher_action_source = "teacher"
            if batch.teacher_actions is None:
                teacher_action = self._teacher_action(batch.teacher_obs)
            else:
                teacher_action = self._cached_teacher_action(batch.teacher_actions)
                teacher_action_source = "cached"
            (
                student_action,
                aux_loss,
                expert_usage,
                route_entropy,
                router_logits,
                expert_actions,
            ) = self._student_action_and_aux(batch.student_obs)
            if trace_update:
                _emit_trainer_runtime(
                    "trainer/after_student_forward",
                    update_number=update_number,
                    teacher_action_source=teacher_action_source,
                    teacher_action=_tensor_runtime_snapshot(teacher_action),
                    student_action=_tensor_runtime_snapshot(student_action),
                    router_logits=_tensor_runtime_snapshot(router_logits),
                    expert_actions=_tensor_runtime_snapshot(expert_actions),
                )
            role_loss, role_target_count = self._role_router_loss(
                role_labels=batch.role_labels,
                router_logits=router_logits,
                batch_size=int(batch.student_obs.shape[0]),
                like=student_action,
            )
            if trace_update:
                _emit_trainer_runtime(
                    "trainer/after_role_loss",
                    update_number=update_number,
                    role_target_count=role_target_count,
                    role_loss=_tensor_runtime_snapshot(role_loss),
                )
            command_intent_loss, command_intent_target_count = self._command_intent_router_loss(
                command_intents=batch.command_intents,
                router_logits=router_logits,
                batch_size=int(batch.student_obs.shape[0]),
                like=student_action,
            )
            if trace_update:
                _emit_trainer_runtime(
                    "trainer/after_command_intent_loss",
                    update_number=update_number,
                    command_intent_target_count=command_intent_target_count,
                    command_intent_loss=_tensor_runtime_snapshot(command_intent_loss),
                )
            (
                behavior_action,
                behavior_action_source,
                behavior_target_count,
                selected_expert_targets,
            ) = self._expert_behavior_action(
                student_action=student_action,
                expert_actions=expert_actions,
                role_labels=batch.role_labels,
                command_intents=batch.command_intents,
            )
            if trace_update:
                _emit_trainer_runtime(
                    "trainer/after_behavior_action",
                    update_number=update_number,
                    behavior_action_source=behavior_action_source,
                    behavior_target_count=behavior_target_count,
                    behavior_action=_tensor_runtime_snapshot(behavior_action),
                    selected_expert_targets=_tensor_runtime_snapshot(selected_expert_targets),
                )
            behavior_loss = self._loss(behavior_action, teacher_action)
            loss = (
                behavior_loss
                + self.aux_loss_coef * aux_loss
                + self.role_loss_coef * role_loss
                + self.command_intent_loss_coef * command_intent_loss
            )
            if trace_update:
                _emit_trainer_runtime(
                    "trainer/after_loss",
                    update_number=update_number,
                    behavior_loss=_tensor_runtime_snapshot(behavior_loss),
                    aux_loss=_tensor_runtime_snapshot(aux_loss),
                    role_loss=_tensor_runtime_snapshot(role_loss),
                    command_intent_loss=_tensor_runtime_snapshot(command_intent_loss),
                    total_loss=_tensor_runtime_snapshot(loss),
                )
        return _TrainerForwardPass(
            loss=loss,
            teacher_action=teacher_action,
            teacher_action_source=teacher_action_source,
            student_action=student_action,
            behavior_action=behavior_action,
            behavior_action_source=behavior_action_source,
            behavior_target_count=behavior_target_count,
            selected_expert_targets=selected_expert_targets,
            behavior_loss=behavior_loss,
            aux_loss=aux_loss,
            role_loss=role_loss,
            role_target_count=role_target_count,
            command_intent_loss=command_intent_loss,
            command_intent_target_count=command_intent_target_count,
            expert_usage=expert_usage,
            route_entropy=route_entropy,
        )

    def _backward_update(
        self,
        forward: _TrainerForwardPass,
        *,
        update_number: int,
        trace_update: bool,
        performance: DistillationStageObservationAccumulator | None,
    ) -> float:
        span = nullcontext() if performance is None else performance.measure("learner_backward")
        with span:
            if trace_update:
                _emit_trainer_runtime(
                    "trainer/before_backward",
                    update_number=update_number,
                    total_loss=_tensor_runtime_snapshot(forward.loss),
                )
            self.optimizer.zero_grad(set_to_none=True)
            forward.loss.backward()
            self._clear_inactive_expert_grads(forward.selected_expert_targets)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            grad_norm = self._grad_norm(self.student)
            if trace_update:
                _emit_trainer_runtime(
                    "trainer/after_backward", update_number=update_number, grad_norm=grad_norm
                )
        return grad_norm

    def _optimizer_update(
        self,
        *,
        update_number: int,
        trace_update: bool,
        performance: DistillationStageObservationAccumulator | None,
    ) -> None:
        span = nullcontext() if performance is None else performance.measure("optimizer_step")
        with span:
            if trace_update:
                _emit_trainer_runtime(
                    "trainer/before_optimizer",
                    update_number=update_number,
                    optimizer_type=type(self.optimizer).__name__,
                )
            self.optimizer.step()
            if trace_update:
                _emit_trainer_runtime(
                    "trainer/after_optimizer",
                    update_number=update_number,
                    optimizer_type=type(self.optimizer).__name__,
                )

    def _build_update_stats(
        self, forward: _TrainerForwardPass, *, grad_norm: float
    ) -> BehaviorDistillationStats:
        return BehaviorDistillationStats(
            loss=float(forward.loss.detach().item()),
            student_grad_norm=grad_norm,
            update_count=self.update_count,
            student_action_shape=tuple(forward.student_action.shape),
            teacher_action_shape=tuple(forward.teacher_action.shape),
            teacher_action_requires_grad=bool(forward.teacher_action.requires_grad),
            behavior_loss=float(forward.behavior_loss.detach().item()),
            aux_loss=float(forward.aux_loss.detach().item()),
            role_loss=float(forward.role_loss.detach().item()),
            role_target_count=forward.role_target_count,
            command_intent_loss=float(forward.command_intent_loss.detach().item()),
            command_intent_target_count=forward.command_intent_target_count,
            expert_usage=forward.expert_usage,
            route_entropy=forward.route_entropy,
            teacher_action_source=forward.teacher_action_source,
            behavior_action_shape=tuple(forward.behavior_action.shape),
            behavior_action_source=forward.behavior_action_source,
            behavior_target_count=forward.behavior_target_count,
        )

    def update(
        self,
        batch: DistillationBatch,
        *,
        performance: DistillationStageObservationAccumulator | None = None,
    ) -> BehaviorDistillationStats:
        update_number = self.update_count + 1
        trace_update = _runtime_trace_update(update_number)
        if trace_update:
            _emit_trainer_runtime(
                "trainer/update_entry",
                update_number=update_number,
                student_obs=_tensor_runtime_snapshot(batch.student_obs),
                teacher_obs=_tensor_runtime_snapshot(batch.teacher_obs),
                teacher_actions=_tensor_runtime_snapshot(batch.teacher_actions),
                role_label_counts=_label_counts(batch.role_labels),
                command_intent_counts=_label_counts(batch.command_intents),
                optimizer_type=type(self.optimizer).__name__,
                student_type=type(self.student).__name__,
            )
        self._validate_update_batch(batch)
        self.student.train()
        forward = self._forward_update(
            batch,
            update_number=update_number,
            trace_update=trace_update,
            performance=performance,
        )
        grad_norm = self._backward_update(
            forward,
            update_number=update_number,
            trace_update=trace_update,
            performance=performance,
        )
        self._optimizer_update(
            update_number=update_number,
            trace_update=trace_update,
            performance=performance,
        )
        self.update_count += 1
        if trace_update:
            _emit_trainer_runtime(
                "trainer/update_complete",
                update_number=update_number,
                trainer_update_count=self.update_count,
                grad_norm=grad_norm,
            )

        return self._build_update_stats(forward, grad_norm=grad_norm)
