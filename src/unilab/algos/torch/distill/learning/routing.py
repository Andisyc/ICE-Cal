"""Pure label routing decisions for behavior distillation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F

TargetAppender = Callable[[list[int], Any, str, int], None]


def decode_student_output(
    student_output: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[float, ...] | None,
    float | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if isinstance(student_output, torch.Tensor):
        return student_output, student_output.new_zeros(()), None, None, None, None

    student_action = getattr(student_output, "action", None)
    if not isinstance(student_action, torch.Tensor):
        raise TypeError("student output diagnostics must expose a tensor `action`")
    aux_loss = student_action.new_zeros(())
    route_entropy: float | None = None
    route_probs = getattr(student_output, "route_probs", None)
    router_logits = getattr(student_output, "router_logits", None)
    expert_actions = getattr(student_output, "expert_actions", None)
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
    if expert_actions is not None and not isinstance(expert_actions, torch.Tensor):
        raise TypeError("student output diagnostics `expert_actions` must be a tensor")
    return student_action, aux_loss, expert_usage, route_entropy, router_logits, expert_actions


def resolve_label_target_indices(
    *,
    labels: tuple[str, ...] | None,
    targets: Mapping[str, int],
    batch_size: int,
    num_experts: int,
    label_name: str,
    required: bool,
    append_target: TargetAppender | None = None,
    validate_range: bool = True,
) -> tuple[int, ...] | None:
    """Resolve one expert index per row without owning tensors or trainer state."""
    if not targets:
        if required:
            raise ValueError(f"{label_name}_expert_targets must be non-empty")
        return None
    if labels is None:
        if required:
            raise ValueError(f"{label_name} labels are required for expert behavior loss")
        return None
    if len(labels) != int(batch_size):
        raise ValueError(
            f"{label_name} length mismatch: labels={len(labels)} batch={int(batch_size)}"
        )

    target_indices: list[int] = []
    for row_index, label in enumerate(labels):
        label_key = str(label)
        if label_key not in targets:
            if required:
                raise ValueError(
                    f"unmapped {label_name} label for expert behavior loss: {label_key!r}"
                )
            return None
        raw_target = targets[label_key]
        if append_target is None:
            target_indices.append(int(raw_target))
        else:
            append_target(target_indices, raw_target, label_key, row_index)

    if validate_range and target_indices and (
        min(target_indices) < 0 or max(target_indices) >= int(num_experts)
    ):
        raise ValueError(
            f"{label_name}_expert_targets index out of range: "
            f"targets={sorted(set(target_indices))} num_experts={int(num_experts)}"
        )
    return tuple(target_indices)


def materialize_target_indices(
    target_indices: tuple[int, ...], *, device: torch.device
) -> torch.Tensor:
    return torch.tensor(target_indices, dtype=torch.long, device=device)
