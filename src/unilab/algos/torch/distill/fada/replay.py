"""FADA replay retention and sampling owner."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Mapping

import torch

from unilab.algos.torch.distill.fada.model import (
    FADA_COMMAND_SCENARIOS,
    FADA_SCENARIO_IDS,
    FADAArchitectureConfig,
    FADASourceBatch,
)
from unilab.algos.torch.distill.fada.source_artifact import (
    LoadedFADASourceArtifact,
    batch_to_device,
)


@dataclass(frozen=True)
class FADAReplayRoleCounts:
    """Read-only retained-row evidence for the admitted paper source roles."""

    planner_eligible: int
    planner_ineligible: int


def _slice_batch(batch: FADASourceBatch, start: int, end: int) -> FADASourceBatch:
    return FADASourceBatch(
        **{
            field: getattr(batch, field)[start:end]
            for field in FADASourceBatch.__dataclass_fields__
        }
    )


def planner_sample_indices(
    batch_size: int,
    *,
    planner_eligible: torch.Tensor,
    command_scenario: torch.Tensor,
    cold_start: torch.Tensor,
    scenario_ratios: Mapping[str, float],
    walk_cold_start_ratio: float,
    static_cold_start_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    scenario_counts = _allocate_ratio_counts(
        int(batch_size),
        scenario_ratios,
        ordered_names=FADA_COMMAND_SCENARIOS,
        label="Planner scenario",
    )
    selected: list[torch.Tensor] = []
    for scenario, count in scenario_counts:
        scenario_mask = planner_eligible & (command_scenario == FADA_SCENARIO_IDS[scenario])
        if scenario in {"walk", "static_stand"}:
            cold_ratio = (
                float(walk_cold_start_ratio)
                if scenario == "walk"
                else float(static_cold_start_ratio)
            )
            cold_counts = _allocate_ratio_counts(
                count,
                {"cold_start": cold_ratio, "steady_state": 1.0 - cold_ratio},
                ordered_names=("cold_start", "steady_state"),
                label=f"{scenario} Planner profile",
            )
            for profile, profile_count in cold_counts:
                profile_mask = (
                    scenario_mask & cold_start
                    if profile == "cold_start"
                    else scenario_mask & ~cold_start
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
    return indices.index_select(0, torch.randperm(indices.numel(), generator=generator))


class FADAReplayBuffer:
    """Bounded replay that owns independent CPU shards instead of one monolith."""

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
        self._chunks: tuple[FADASourceBatch, ...] = ()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def _batch(self) -> FADASourceBatch | None:
        """Compatibility view for diagnostics/tests; runtime sampling stays sharded."""

        if not self._chunks:
            return None
        if len(self._chunks) == 1:
            return self._chunks[0]
        return FADASourceBatch(
            **{
                field: torch.cat([getattr(chunk, field) for chunk in self._chunks], dim=0)
                for field in FADASourceBatch.__dataclass_fields__
            }
        ).validate(self.config)

    @property
    def effective_capacity(self) -> int:
        ratio = self.suboptimal_retention_ratio
        if ratio is None:
            return self.capacity
        return (self.capacity // (ratio + 1)) * (ratio + 1)

    def source_role_counts(self) -> FADAReplayRoleCounts:
        planner_eligible = sum(int(chunk.planner_eligible.sum()) for chunk in self._chunks)
        return FADAReplayRoleCounts(
            planner_eligible=planner_eligible,
            planner_ineligible=len(self) - planner_eligible,
        )

    def _retained_indices(self, roles: Sequence[torch.Tensor]) -> torch.Tensor:
        size = sum(int(role.shape[0]) for role in roles)
        ratio = self.suboptimal_retention_ratio
        if ratio is None or size <= self.capacity:
            start = max(size - self.capacity, 0)
            return torch.arange(start, size, dtype=torch.int64)

        planner_eligible_capacity = self.capacity // (ratio + 1)
        planner_ineligible_capacity = planner_eligible_capacity * ratio
        planner_eligible = torch.cat(tuple(roles))
        planner_eligible_indices = torch.nonzero(planner_eligible, as_tuple=False).flatten()
        planner_ineligible_indices = torch.nonzero(~planner_eligible, as_tuple=False).flatten()
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

    def _retain_chunks(
        self,
        chunks: Sequence[FADASourceBatch],
        indices: torch.Tensor,
    ) -> tuple[FADASourceBatch, ...]:
        retained: list[FADASourceBatch] = []
        offset = 0
        for chunk in chunks:
            rows = int(chunk.command.shape[0])
            left = int(torch.searchsorted(indices, torch.tensor(offset)).item())
            right = int(torch.searchsorted(indices, torch.tensor(offset + rows)).item())
            local = indices[left:right] - offset
            offset += rows
            if local.numel() == 0:
                continue
            first = int(local[0])
            last = int(local[-1])
            if first == 0 and last + 1 == rows and local.numel() == rows:
                candidate = chunk
            elif last - first + 1 == local.numel():
                candidate = FADASourceBatch(
                    **{
                        field: getattr(chunk, field)[first : last + 1].clone()
                        for field in FADASourceBatch.__dataclass_fields__
                    }
                )
            else:
                candidate = FADASourceBatch(
                    **{
                        field: getattr(chunk, field).index_select(0, local)
                        for field in FADASourceBatch.__dataclass_fields__
                    }
                )
            retained.append(candidate.validate(self.config))
        return tuple(retained)

    def add(self, batch: FADASourceBatch) -> None:
        self.add_many((batch,))

    def add_many(self, batches: Iterable[FADASourceBatch]) -> None:
        """Copy caller-owned batches into one atomic replay transaction."""

        self._add_many(batches, copy_incoming=True)

    def _add_many(
        self,
        batches: Iterable[FADASourceBatch],
        *,
        copy_incoming: bool,
    ) -> None:
        """Atomically admit one collection transaction without global tensor copies."""

        incoming = tuple(
            FADASourceBatch(
                **{
                    field: getattr(batch, field).detach().to("cpu").clone()
                    for field in FADASourceBatch.__dataclass_fields__
                }
            ).validate(self.config)
            if copy_incoming
            else batch_to_device(batch.validate(self.config), torch.device("cpu"))
            for batch in batches
        )
        if not incoming:
            raise ValueError("cannot add an empty FADA replay transaction")
        candidates = (*self._chunks, *incoming)
        retained_indices = self._retained_indices(
            tuple(batch.planner_eligible for batch in candidates)
        )
        retained = self._retain_chunks(candidates, retained_indices)
        self._chunks = retained
        self._size = int(retained_indices.numel())

    def add_artifact(self, artifact: LoadedFADASourceArtifact) -> None:
        """Atomically retain only selected artifact shards, loading them one at a time."""

        incoming_roles = artifact.planner_role_vectors()
        role_chunks = (
            *(chunk.planner_eligible for chunk in self._chunks),
            *incoming_roles,
        )
        retained_indices = self._retained_indices(role_chunks)
        old_size = len(self)
        old_end = int(torch.searchsorted(retained_indices, torch.tensor(old_size)).item())
        retained: list[FADASourceBatch] = list(
            self._retain_chunks(self._chunks, retained_indices[:old_end])
        )
        incoming_indices = retained_indices[old_end:] - old_size
        offset = 0
        for index, roles in enumerate(incoming_roles):
            rows = int(roles.shape[0])
            left = int(torch.searchsorted(incoming_indices, torch.tensor(offset)).item())
            right = int(torch.searchsorted(incoming_indices, torch.tensor(offset + rows)).item())
            local = incoming_indices[left:right] - offset
            offset += rows
            if local.numel() == 0:
                continue
            batch = artifact.load_batch(index)
            retained.extend(self._retain_chunks((batch,), local))
        self._chunks = tuple(retained)
        self._size = int(retained_indices.numel())

    def _select_global_indices(self, indices: torch.Tensor) -> FADASourceBatch:
        outputs: dict[str, torch.Tensor] = {}
        first = self._chunks[0]
        for field in FADASourceBatch.__dataclass_fields__:
            tensor = getattr(first, field)
            outputs[field] = torch.empty(
                (indices.numel(), *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
            )
        offset = 0
        for chunk in self._chunks:
            rows = int(chunk.command.shape[0])
            positions = torch.nonzero(
                (indices >= offset) & (indices < offset + rows), as_tuple=False
            ).flatten()
            if positions.numel() > 0:
                local = indices.index_select(0, positions) - offset
                for field in FADASourceBatch.__dataclass_fields__:
                    outputs[field].index_copy_(
                        0,
                        positions,
                        getattr(chunk, field).index_select(0, local),
                    )
            offset += rows
        return FADASourceBatch(**outputs).validate(self.config)

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> FADASourceBatch:
        if len(self) == 0:
            raise ValueError("cannot sample an empty FADA replay buffer")
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        indices = torch.randint(len(self), (int(batch_size),), generator=generator)
        sampled = self._select_global_indices(indices)
        return batch_to_device(sampled, torch.device(device)).validate(self.config)

    def _metadata_field(self, field: str) -> torch.Tensor:
        return torch.cat([getattr(chunk, field) for chunk in self._chunks])

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

        if len(self) == 0:
            raise ValueError("cannot sample an empty FADA replay buffer")
        planner_eligible = self._metadata_field("planner_eligible")
        command_scenario = self._metadata_field("command_scenario")
        cold_start = self._metadata_field("cold_start")
        indices = planner_sample_indices(
            int(batch_size),
            planner_eligible=planner_eligible,
            command_scenario=command_scenario,
            cold_start=cold_start,
            scenario_ratios=scenario_ratios,
            walk_cold_start_ratio=walk_cold_start_ratio,
            static_cold_start_ratio=static_cold_start_ratio,
            generator=generator,
        )
        sampled = self._select_global_indices(indices)
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
