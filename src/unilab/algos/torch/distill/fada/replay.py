"""FADA replay retention and sampling owner."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Mapping

import torch

from unilab.algos.torch.distill.fada.model import (
    FADA_COMMAND_SCENARIOS,
    FADA_SCENARIO_IDS,
    FADAArchitectureConfig,
    FADASourceBatch,
)
from unilab.algos.torch.distill.fada.source_artifact import batch_to_device


@dataclass(frozen=True)
class FADAReplayRoleCounts:
    """Read-only retained-row evidence for the admitted paper source roles."""

    planner_eligible: int
    planner_ineligible: int


class FADAReplayBuffer:
    """Bounded source-window replay with one validated tensor owner per field."""

    def __init__(
        self,
        config: FADAArchitectureConfig,
        *,
        capacity: int,
        suboptimal_retention_ratio: int | None = None,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if suboptimal_retention_ratio is not None:
            ratio = int(suboptimal_retention_ratio)
            if ratio <= 0 or float(ratio) != float(suboptimal_retention_ratio):
                raise ValueError(
                    "suboptimal_retention_ratio must be a positive integer, "
                    f"got {suboptimal_retention_ratio}"
                )
            if int(capacity) < ratio + 1:
                raise ValueError(
                    "capacity must fit one complete replay role-ratio block: "
                    f"capacity={capacity} ratio=1:{ratio}"
                )
        self.config = config
        self.capacity = int(capacity)
        self.suboptimal_retention_ratio = (
            None if suboptimal_retention_ratio is None else int(suboptimal_retention_ratio)
        )
        self._batch: FADASourceBatch | None = None

    def __len__(self) -> int:
        return 0 if self._batch is None else int(self._batch.command.shape[0])

    @property
    def effective_capacity(self) -> int:
        ratio = self.suboptimal_retention_ratio
        if ratio is None:
            return self.capacity
        return (self.capacity // (ratio + 1)) * (ratio + 1)

    def source_role_counts(self) -> FADAReplayRoleCounts:
        if self._batch is None:
            return FADAReplayRoleCounts(planner_eligible=0, planner_ineligible=0)
        planner_eligible = int(self._batch.planner_eligible.sum())
        return FADAReplayRoleCounts(
            planner_eligible=planner_eligible,
            planner_ineligible=len(self) - planner_eligible,
        )

    def _retained_indices(self, merged: FADASourceBatch) -> torch.Tensor:
        size = int(merged.command.shape[0])
        ratio = self.suboptimal_retention_ratio
        if ratio is None or size <= self.capacity:
            start = max(size - self.capacity, 0)
            return torch.arange(start, size, dtype=torch.int64)

        planner_eligible_capacity = self.capacity // (ratio + 1)
        planner_ineligible_capacity = planner_eligible_capacity * ratio
        planner_eligible_indices = torch.nonzero(merged.planner_eligible, as_tuple=False).flatten()
        planner_ineligible_indices = torch.nonzero(
            ~merged.planner_eligible, as_tuple=False
        ).flatten()
        if planner_eligible_indices.numel() < planner_eligible_capacity:
            raise ValueError(
                "paper FADA replay overflow lacks Planner-eligible main rows: "
                f"required={planner_eligible_capacity} "
                f"observed={planner_eligible_indices.numel()}"
            )
        if planner_ineligible_indices.numel() < planner_ineligible_capacity:
            raise ValueError(
                "paper FADA replay overflow lacks Planner-ineligible intermediate rows: "
                f"required={planner_ineligible_capacity} "
                f"observed={planner_ineligible_indices.numel()}"
            )
        selected = torch.cat(
            (
                planner_eligible_indices[-planner_eligible_capacity:],
                planner_ineligible_indices[-planner_ineligible_capacity:],
            )
        )
        return selected.sort().values

    def add(self, batch: FADASourceBatch) -> None:
        # B1: 校验 causal window, 产出可进入 replay 的 CPU batch.
        incoming = batch_to_device(batch.validate(self.config), torch.device("cpu"))
        if self._batch is None:
            merged = incoming
        else:
            merged = FADASourceBatch(
                **{
                    field: torch.cat([getattr(self._batch, field), getattr(incoming, field)], dim=0)
                    for field in FADASourceBatch.__dataclass_fields__
                }
            )
        retained_indices = self._retained_indices(merged)
        candidate = FADASourceBatch(
            **{
                field: getattr(merged, field).index_select(0, retained_indices).contiguous()
                for field in FADASourceBatch.__dataclass_fields__
            }
        ).validate(self.config)
        self._batch = candidate

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> FADASourceBatch:
        if self._batch is None or len(self) == 0:
            raise ValueError("cannot sample an empty FADA replay buffer")
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        indices = torch.randint(len(self), (int(batch_size),), generator=generator)
        sampled = FADASourceBatch(
            **{
                field: getattr(self._batch, field).index_select(0, indices)
                for field in FADASourceBatch.__dataclass_fields__
            }
        )
        return batch_to_device(sampled, torch.device(device)).validate(self.config)

    def sample_planner(
        self,
        batch_size: int,
        *,
        scenario_ratios: Mapping[str, float],
        walk_cold_start_ratio: float,
        static_cold_start_ratio: float,
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> FADASourceBatch:
        """Sample one exact scenario-balanced Planner batch from eligible replay rows."""

        if self._batch is None or len(self) == 0:
            raise ValueError("cannot sample an empty FADA replay buffer")
        scenario_counts = _allocate_ratio_counts(
            int(batch_size),
            scenario_ratios,
            ordered_names=FADA_COMMAND_SCENARIOS,
            label="Planner scenario",
        )
        selected: list[torch.Tensor] = []
        for scenario, count in scenario_counts:
            scenario_mask = self._batch.planner_eligible & (
                self._batch.command_scenario == FADA_SCENARIO_IDS[scenario]
            )
            if scenario in {"walk", "static_stand"}:
                cold_start_ratio = (
                    float(walk_cold_start_ratio)
                    if scenario == "walk"
                    else float(static_cold_start_ratio)
                )
                cold_counts = _allocate_ratio_counts(
                    count,
                    {
                        "cold_start": cold_start_ratio,
                        "steady_state": 1.0 - cold_start_ratio,
                    },
                    ordered_names=("cold_start", "steady_state"),
                    label=f"{scenario} Planner profile",
                )
                for profile, profile_count in cold_counts:
                    profile_mask = (
                        scenario_mask & self._batch.cold_start
                        if profile == "cold_start"
                        else scenario_mask & ~self._batch.cold_start
                    )
                    selected.append(
                        _sample_mask_indices(
                            profile_mask,
                            profile_count,
                            generator=generator,
                            label=f"{scenario}/{profile}",
                        )
                    )
            else:
                selected.append(
                    _sample_mask_indices(
                        scenario_mask,
                        count,
                        generator=generator,
                        label=scenario,
                    )
                )
        indices = torch.cat(selected)
        indices = indices.index_select(0, torch.randperm(indices.numel(), generator=generator))
        sampled = FADASourceBatch(
            **{
                field: getattr(self._batch, field).index_select(0, indices)
                for field in FADASourceBatch.__dataclass_fields__
            }
        )
        return batch_to_device(sampled, torch.device(device)).validate(self.config)


def _allocate_ratio_counts(
    total: int,
    ratios: Mapping[str, float],
    *,
    ordered_names: Sequence[str],
    label: str,
) -> tuple[tuple[str, int], ...]:
    if int(total) <= 0:
        raise ValueError(f"{label} total must be positive, got {total}")
    unknown = set(ratios) - set(ordered_names)
    if unknown:
        raise ValueError(f"{label} ratios contain unknown labels: {sorted(unknown)}")
    values = [float(ratios.get(name, 0.0)) for name in ordered_names]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"{label} ratios must be finite and non-negative")
    if abs(sum(values) - 1.0) > 1.0e-6:
        raise ValueError(f"{label} ratios must sum to 1, got {sum(values)}")
    positive = sum(value > 0.0 for value in values)
    if int(total) < positive:
        raise ValueError(f"{label} total={total} cannot cover {positive} positive strata")
    raw = [int(total) * value for value in values]
    counts = [int(value) for value in raw]
    for index, value in enumerate(values):
        if value > 0.0 and counts[index] == 0:
            counts[index] = 1
    while sum(counts) > int(total):
        candidates = [index for index, count in enumerate(counts) if count > 1]
        if not candidates:
            raise ValueError(f"{label} allocation cannot preserve positive strata")
        counts[min(candidates, key=lambda item: (raw[item] - counts[item], -item))] -= 1
    while sum(counts) < int(total):
        counts[max(range(len(counts)), key=lambda item: (raw[item] - counts[item], -item))] += 1
    return tuple(
        (name, count) for name, count in zip(ordered_names, counts, strict=True) if count > 0
    )


def _sample_mask_indices(
    mask: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None,
    label: str,
) -> torch.Tensor:
    candidates = torch.nonzero(mask, as_tuple=False).flatten()
    if candidates.numel() == 0:
        raise ValueError(f"Planner replay is missing required stratum {label!r}")
    draws = torch.randint(candidates.numel(), (int(count),), generator=generator)
    return candidates.index_select(0, draws)
