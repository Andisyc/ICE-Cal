"""Paper-exact FADA source identity and allocation owner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

FADA_INTERMEDIATE_ORACLE_COUNT = 20


@dataclass(frozen=True)
class FADAPaperSourcePlan:
    """Validated Appendix B.2 intermediate-Oracle identities and window allocation."""

    enabled: bool
    source_allocations: tuple[tuple[Path, int], ...]

    @property
    def checkpoint_paths(self) -> tuple[Path, ...]:
        return tuple(path for path, _ in self.source_allocations)


def build_fada_paper_source_plan(
    *,
    enabled: bool,
    oracle_shadow_enabled: bool,
    checkpoint_paths: Sequence[str | Path],
    configured_checkpoint_count: int,
    suboptimal_data_ratio: float,
    optimal_windows: int,
    resume_path: str | Path | None,
) -> FADAPaperSourcePlan:
    """Own the paper-exact source identities, ratio, and per-Oracle allocation."""

    if not enabled:
        return FADAPaperSourcePlan(enabled=False, source_allocations=())

    # B1: freeze Appendix B.2 invariants before any environment or optimizer mutation.
    if not oracle_shadow_enabled:
        raise ValueError("paper-exact FADA source training requires oracle_shadow_enabled=true")
    if resume_path not in (None, ""):
        raise ValueError(
            "paper-exact FADA resume is disabled until replay persistence is implemented; "
            "restart the source campaign instead"
        )
    if int(configured_checkpoint_count) != FADA_INTERMEDIATE_ORACLE_COUNT:
        raise ValueError(
            "paper-exact FADA requires intermediate_oracle_count=20, got "
            f"{configured_checkpoint_count}"
        )
    if float(suboptimal_data_ratio) != 2.0:
        raise ValueError(
            f"paper-exact FADA requires suboptimal_data_ratio=2.0, got {suboptimal_data_ratio}"
        )
    if int(optimal_windows) <= 0:
        raise ValueError(f"optimal_windows must be positive, got {optimal_windows}")

    # B2: seal exactly 20 unique readable identities from one caller-resolved namespace.
    paths = tuple(Path(path) for path in checkpoint_paths)
    if len(paths) != FADA_INTERMEDIATE_ORACLE_COUNT or len(set(paths)) != len(paths):
        raise ValueError(
            "paper-exact FADA requires exactly 20 unique intermediate Oracle checkpoints, "
            f"got {len(paths)}"
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"intermediate Oracle checkpoints do not exist: {missing}")

    # B3: distribute the exact 2:1 source budget while ensuring every Oracle contributes.
    total = int(round(int(optimal_windows) * float(suboptimal_data_ratio)))
    if total < len(paths):
        raise ValueError(
            "paper-exact FADA suboptimal budget must allocate at least one window to each "
            f"intermediate Oracle; got total={total} sources={len(paths)}"
        )
    quotient, remainder = divmod(total, len(paths))
    allocations = tuple(
        (path, quotient + (1 if index < remainder else 0)) for index, path in enumerate(paths)
    )
    return FADAPaperSourcePlan(enabled=True, source_allocations=allocations)
