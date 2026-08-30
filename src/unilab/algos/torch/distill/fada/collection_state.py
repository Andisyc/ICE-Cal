"""Mutable batch accumulation owned by one FADA collection transaction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FADAWindowAccumulator:
    """Compact accepted windows while preserving their append order and count."""

    config: Any
    compact_size: int
    merge: Callable[[Sequence[Any], Any], Any]
    pending_batches: list[Any] = field(default_factory=list)
    compacted_batches: list[Any] = field(default_factory=list)
    window_count: int = 0

    def append(self, batch: Any) -> None:
        self.pending_batches.append(batch)
        self.window_count += int(batch.observation_history.shape[0])
        if len(self.pending_batches) >= int(self.compact_size):
            self.compacted_batches.append(self.merge(tuple(self.pending_batches), self.config))
            self.pending_batches.clear()

    def finalize(self) -> Any:
        return self.merge([*self.compacted_batches, *self.pending_batches], self.config)
