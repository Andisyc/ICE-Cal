from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from unilab.algos.torch.distill.learning.moe_student import MoEStudentOutput, MoEStudentPolicy


@dataclass(frozen=True)
class MoERoleRouteSummary:
    role: str
    count: int
    expert_usage: tuple[float, ...]
    expert_fraction: tuple[float, ...]
    mean_entropy: float
    dominant_expert: int
    max_fraction: float
    collapse_detected: bool


@dataclass(frozen=True)
class MoEExpertDiagnostics:
    num_samples: int
    num_experts: int
    overall: MoERoleRouteSummary
    by_role: tuple[MoERoleRouteSummary, ...]
    role_labels_present: bool


def _entropy(route_probs: torch.Tensor) -> torch.Tensor:
    safe_probs = route_probs.clamp_min(1e-8)
    return -(safe_probs * safe_probs.log()).sum(dim=-1)


def _usage(route_probs: torch.Tensor, selected_expert: torch.Tensor | None) -> torch.Tensor:
    if selected_expert is None:
        return route_probs.sum(dim=0)
    return torch.bincount(selected_expert, minlength=route_probs.shape[-1]).to(route_probs.dtype)


def _summary(
    *,
    role: str,
    route_probs: torch.Tensor,
    selected_expert: torch.Tensor | None,
    collapse_fraction: float,
) -> MoERoleRouteSummary:
    count = int(route_probs.shape[0])
    if count <= 0:
        raise ValueError(f"role {role!r} has no samples")
    usage = _usage(route_probs, selected_expert).detach().cpu()
    fractions = usage / max(float(count), 1.0)
    dominant = int(torch.argmax(fractions).item())
    max_fraction = float(fractions[dominant].item())
    return MoERoleRouteSummary(
        role=str(role),
        count=count,
        expert_usage=tuple(float(value) for value in usage.reshape(-1)),
        expert_fraction=tuple(float(value) for value in fractions.reshape(-1)),
        mean_entropy=float(_entropy(route_probs).mean().detach().cpu().item()),
        dominant_expert=dominant,
        max_fraction=max_fraction,
        collapse_detected=max_fraction >= float(collapse_fraction),
    )


def diagnose_moe_expert_routes(
    policy: MoEStudentPolicy,
    student_obs: torch.Tensor,
    *,
    role_labels: list[str] | tuple[str, ...] | None = None,
    hard_routing: bool | None = None,
    collapse_fraction: float = 0.90,
) -> MoEExpertDiagnostics:
    """Summarize MoE router usage by optional semantic role labels."""

    if not isinstance(policy, MoEStudentPolicy):
        raise TypeError("diagnose_moe_expert_routes requires a MoEStudentPolicy")
    if student_obs.ndim != 2:
        raise ValueError(f"student_obs must be rank-2, got shape {tuple(student_obs.shape)}")
    if not torch.isfinite(student_obs).all():
        raise ValueError("student_obs must contain only finite values")
    if not 0.0 < float(collapse_fraction) <= 1.0:
        raise ValueError(f"collapse_fraction must be in (0, 1], got {collapse_fraction}")

    with torch.no_grad():
        output = policy(
            student_obs,
            hard_routing=hard_routing,
            return_diagnostics=True,
        )
    if not isinstance(output, MoEStudentOutput):
        raise TypeError("MoE policy did not return diagnostics")

    route_probs = output.route_probs.detach()
    selected_expert = None if output.selected_expert is None else output.selected_expert.detach()
    if role_labels is None:
        labels = ["all"] * int(student_obs.shape[0])
        role_labels_present = False
    else:
        labels = [str(label) for label in role_labels]
        role_labels_present = True
    if len(labels) != int(student_obs.shape[0]):
        raise ValueError(
            "role_labels length must match student_obs batch size: "
            f"labels={len(labels)} batch={int(student_obs.shape[0])}"
        )

    overall = _summary(
        role="all",
        route_probs=route_probs,
        selected_expert=selected_expert,
        collapse_fraction=float(collapse_fraction),
    )
    by_role: list[MoERoleRouteSummary] = []
    for role in sorted(set(labels)):
        idx = torch.tensor(
            [i for i, label in enumerate(labels) if label == role],
            dtype=torch.long,
            device=route_probs.device,
        )
        role_selected = None if selected_expert is None else selected_expert.index_select(0, idx)
        by_role.append(
            _summary(
                role=role,
                route_probs=route_probs.index_select(0, idx),
                selected_expert=role_selected,
                collapse_fraction=float(collapse_fraction),
            )
        )

    return MoEExpertDiagnostics(
        num_samples=int(student_obs.shape[0]),
        num_experts=int(policy.num_experts),
        overall=overall,
        by_role=tuple(by_role),
        role_labels_present=role_labels_present,
    )


def moe_diagnostics_to_dict(diagnostics: MoEExpertDiagnostics) -> dict[str, Any]:
    def role_to_dict(summary: MoERoleRouteSummary) -> dict[str, Any]:
        return {
            "role": summary.role,
            "count": summary.count,
            "expert_usage": list(summary.expert_usage),
            "expert_fraction": list(summary.expert_fraction),
            "mean_entropy": summary.mean_entropy,
            "dominant_expert": summary.dominant_expert,
            "max_fraction": summary.max_fraction,
            "collapse_detected": summary.collapse_detected,
        }

    return {
        "num_samples": diagnostics.num_samples,
        "num_experts": diagnostics.num_experts,
        "role_labels_present": diagnostics.role_labels_present,
        "overall": role_to_dict(diagnostics.overall),
        "by_role": [role_to_dict(summary) for summary in diagnostics.by_role],
    }
