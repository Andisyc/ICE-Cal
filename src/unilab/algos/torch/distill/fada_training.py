from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .fada import (
    FADA_COMMAND_SCENARIOS,
    FADA_SCENARIO_IDS,
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlanner,
    FADAPlannerIDMPolicy,
    FADASourceBatch,
    idm_source_loss,
    planner_source_loss,
)

FADA_CHECKPOINT_SCHEMA_VERSION = 2
FADA_SOURCE_BATCH_SCHEMA_VERSION = 2
FADA_INTERMEDIATE_ORACLE_COUNT = 20
FADA_V005_REQUIRED_QUALITY_METRICS = (
    "scenario/walk/planner_idm_oracle_action_mse",
    "scenario/static_stand/planner_idm_oracle_action_mse",
    "scenario/walk_to_stand/planner_idm_oracle_action_mse",
    "scenario/static_stand/cold_start_fraction",
    "scenario/static_stand/cold_start_planner_mse",
    "scenario/static_stand/steady_state_planner_mse",
)


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


@dataclass(frozen=True)
class LoadedFADAPlannerIDMPolicy:
    """Inference-ready FADA policy reconstructed from one strict checkpoint."""

    policy: FADAPlannerIDMPolicy
    checkpoint: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedFADASourceBatch:
    """One validated collector artifact and its iteration metadata."""

    batch: FADASourceBatch
    metadata: Mapping[str, Any]


def _batch_to_device(batch: FADASourceBatch, device: torch.device) -> FADASourceBatch:
    return FADASourceBatch(
        observation_history=batch.observation_history.to(device),
        action_history=batch.action_history.to(device),
        command=batch.command.to(device),
        realized_future=batch.realized_future.to(device),
        executed_action_chunk=batch.executed_action_chunk.to(device),
        oracle_future=batch.oracle_future.to(device),
        oracle_action_chunk=batch.oracle_action_chunk.to(device),
        oracle_shadow_valid=batch.oracle_shadow_valid.to(device),
        oracle_first_action=batch.oracle_first_action.to(device),
        command_scenario=batch.command_scenario.to(device),
        planner_eligible=batch.planner_eligible.to(device),
        cold_start=batch.cold_start.to(device),
    )


def save_fada_source_batch(
    path: str | Path,
    batch: FADASourceBatch,
    *,
    config: FADAArchitectureConfig,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically persist one CPU causal-window artifact from the collector process."""

    validated = _batch_to_device(batch.validate(config), torch.device("cpu"))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "schema_version": FADA_SOURCE_BATCH_SCHEMA_VERSION,
            "architecture": asdict(config),
            "batch": {
                field: getattr(validated, field) for field in FADASourceBatch.__dataclass_fields__
            },
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(target)
    return target


def load_fada_source_batch(
    path: str | Path,
    *,
    config: FADAArchitectureConfig,
) -> LoadedFADASourceBatch:
    """Load and validate one collector artifact before it enters learner replay."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        FADA_SOURCE_BATCH_SCHEMA_VERSION
    ):
        raise ValueError("unsupported or malformed FADA source batch schema")
    expected = asdict(config)
    if payload.get("architecture") != expected:
        raise ValueError(
            "FADA source batch architecture mismatch: "
            f"expected={expected} observed={payload.get('architecture')}"
        )
    tensors = payload.get("batch")
    if not isinstance(tensors, dict) or set(tensors) != set(FADASourceBatch.__dataclass_fields__):
        raise ValueError("FADA source batch tensor fields are incomplete")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("FADA source batch metadata must be a mapping")
    batch = FADASourceBatch(**tensors).validate(config)
    return LoadedFADASourceBatch(batch=batch, metadata=metadata)


class FADAReplayBuffer:
    """Bounded source-window replay with one validated tensor owner per field."""

    def __init__(self, config: FADAArchitectureConfig, *, capacity: int) -> None:
        if int(capacity) <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.config = config
        self.capacity = int(capacity)
        self._batch: FADASourceBatch | None = None

    def __len__(self) -> int:
        return 0 if self._batch is None else int(self._batch.command.shape[0])

    def add(self, batch: FADASourceBatch) -> None:
        # B1: 校验 causal window, 产出可进入 replay 的 CPU batch.
        incoming = _batch_to_device(batch.validate(self.config), torch.device("cpu"))
        if self._batch is None:
            merged = incoming
        else:
            merged = FADASourceBatch(
                **{
                    field: torch.cat([getattr(self._batch, field), getattr(incoming, field)], dim=0)
                    for field in FADASourceBatch.__dataclass_fields__
                }
            )
        start = max(int(merged.command.shape[0]) - self.capacity, 0)
        self._batch = FADASourceBatch(
            **{
                field: getattr(merged, field)[start:].contiguous()
                for field in FADASourceBatch.__dataclass_fields__
            }
        ).validate(self.config)

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
        return _batch_to_device(sampled, torch.device(device)).validate(self.config)

    def sample_planner(
        self,
        batch_size: int,
        *,
        scenario_ratios: Mapping[str, float],
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
            if scenario == "static_stand":
                cold_counts = _allocate_ratio_counts(
                    count,
                    {
                        "cold_start": float(static_cold_start_ratio),
                        "steady_state": 1.0 - float(static_cold_start_ratio),
                    },
                    ordered_names=("cold_start", "steady_state"),
                    label="static Planner profile",
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
                            label=f"static_stand/{profile}",
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
        return _batch_to_device(sampled, torch.device(device)).validate(self.config)


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


@dataclass(frozen=True)
class FADATrainingStats:
    idm_loss: float
    planner_loss: float
    idm_grad_norm: float
    planner_grad_norm: float


@torch.no_grad()
def evaluate_fada_source_batch(
    policy: FADAPlannerIDMPolicy,
    batch: FADASourceBatch,
    *,
    require_scenario_metrics: bool = False,
) -> dict[str, float]:
    """Measure the three adjacent source-quality boundaries on one sealed batch."""

    # B1: 在 policy device 上重放 causal rows, 产出 true-future IDM boundary error.
    device = next(policy.parameters()).device
    current = _batch_to_device(batch.validate(policy.config), device)
    trajectory_action = policy.idm(
        current.observation_history,
        current.action_history,
        current.realized_future,
    )[:, 0]
    trajectory_mse = torch.mean(
        torch.square(trajectory_action - current.executed_action_chunk[:, 0])
    )

    # B2: 单独测量 final-Oracle shadow support; 没有 valid row 时 fail-closed.
    valid = current.oracle_shadow_valid
    if not bool(valid.any()):
        raise ValueError("FADA quality evaluation requires at least one valid Oracle-shadow row")
    shadow_action = policy.idm(
        current.observation_history[valid],
        current.action_history[valid],
        current.oracle_future[valid],
    )[:, 0]
    shadow_mse = torch.mean(torch.square(shadow_action - current.oracle_action_chunk[valid, 0]))

    # B3: 测量 Planner 经 IDM 的 Oracle-action error 与 future support drift, 交给 checkpoint consumer.
    output = policy(
        current.observation_history,
        current.action_history,
        current.command,
    )
    planner_action_mse = torch.mean(torch.square(output.action - current.oracle_first_action))
    planner_future_realized_mse = torch.mean(
        torch.square(output.predicted_future - current.realized_future)
    )
    metrics = {
        "trajectory_idm_action_mse": float(trajectory_mse),
        "oracle_shadow_idm_action_mse": float(shadow_mse),
        "planner_idm_oracle_action_mse": float(planner_action_mse),
        "planner_future_realized_mse": float(planner_future_realized_mse),
        "oracle_shadow_valid_fraction": float(valid.float().mean()),
    }
    if require_scenario_metrics:
        eligible = current.planner_eligible
        if not bool(eligible.any()):
            raise ValueError("FADA scenario quality requires Planner-eligible rows")
        for scenario in FADA_COMMAND_SCENARIOS:
            mask = eligible & (current.command_scenario == FADA_SCENARIO_IDS[scenario])
            if not bool(mask.any()):
                raise ValueError(f"FADA scenario quality is missing {scenario!r} rows")
            metrics[f"scenario/{scenario}/row_fraction"] = float(mask.float().mean())
            metrics[f"scenario/{scenario}/planner_idm_oracle_action_mse"] = float(
                torch.mean(torch.square(output.action[mask] - current.oracle_first_action[mask]))
            )
        static = eligible & (current.command_scenario == FADA_SCENARIO_IDS["static_stand"])
        cold = static & current.cold_start
        steady = static & ~current.cold_start
        if not bool(cold.any()) or not bool(steady.any()):
            raise ValueError("FADA scenario quality requires static cold-start and steady rows")
        metrics["scenario/static_stand/cold_start_fraction"] = float(
            cold.float().sum() / static.float().sum()
        )
        metrics["scenario/static_stand/cold_start_planner_mse"] = float(
            torch.mean(torch.square(output.action[cold] - current.oracle_first_action[cold]))
        )
        metrics["scenario/static_stand/steady_state_planner_mse"] = float(
            torch.mean(torch.square(output.action[steady] - current.oracle_first_action[steady]))
        )
    if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
        raise ValueError(f"FADA quality metrics must be finite: {metrics}")
    return metrics


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

    def _clip(self, module: nn.Module) -> None:
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)

    def _update_idm(self, batch: FADASourceBatch) -> tuple[float, float]:
        self.idm_optimizer.zero_grad(set_to_none=True)
        loss = idm_source_loss(self.policy.idm, batch)
        loss.backward()
        self._clip(self.policy.idm)
        grad_norm = _grad_norm(self.policy.idm.parameters())
        self.idm_optimizer.step()
        return float(loss.detach()), grad_norm

    def _update_planner(self, batch: FADASourceBatch) -> tuple[float, float]:
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
        # B1: 先执行 teacher-forced IDM pass, 产出只属于 IDM 的梯度和 loss.
        if int(idm_updates) <= 0 or int(planner_updates) <= 0:
            raise ValueError("idm_updates and planner_updates must both be positive")
        batch.validate(self.policy.config)
        idm_loss_value = 0.0
        idm_grad_norm = 0.0
        for _ in range(int(idm_updates)):
            idm_loss_value, idm_grad_norm = self._update_idm(batch)

        # B2: 再固定 IDM 参数执行 Planner pass, 产出只属于 Planner 的更新.
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
        planner_static_cold_start_ratio: float = 0.5,
    ) -> FADATrainingStats:
        """Run ordered passes while drawing a fresh replay sample for every update."""

        if int(idm_updates) <= 0 or int(planner_updates) <= 0:
            raise ValueError("idm_updates and planner_updates must both be positive")
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


def save_fada_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer,
    *,
    completed_iterations: int,
    samples_seen: int,
    runtime_config: Mapping[str, Any],
    quality_metrics: Mapping[str, float] | None = None,
) -> Path:
    """Atomically persist the paired FADA module and optimizer identity."""

    metrics = dict(quality_metrics or {})
    v005_enabled = bool(dict(runtime_config.get("v005_replay") or {}).get("enabled", False))
    if v005_enabled and int(completed_iterations) > 0:
        missing = [name for name in FADA_V005_REQUIRED_QUALITY_METRICS if name not in metrics]
        if missing:
            raise ValueError(f"v005 FADA checkpoint quality metrics are missing: {missing}")
        invalid = [
            name
            for name, value in metrics.items()
            if not isinstance(value, (int, float)) or not math.isfinite(float(value))
        ]
        if invalid:
            raise ValueError(f"v005 FADA checkpoint quality metrics are non-finite: {invalid}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "schema_version": FADA_CHECKPOINT_SCHEMA_VERSION,
        "architecture": asdict(policy.config),
        "planner_state_dict": policy.planner.state_dict(),
        "idm_state_dict": policy.idm.state_dict(),
        "planner_optimizer_state_dict": trainer.planner_optimizer.state_dict(),
        "idm_optimizer_state_dict": trainer.idm_optimizer.state_dict(),
        "completed_iterations": int(completed_iterations),
        "samples_seen": int(samples_seen),
        "runtime_config": dict(runtime_config),
        "quality_metrics": metrics,
    }
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_fada_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore one exact FADA architecture and optionally its paired optimizers."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported or malformed FADA checkpoint schema")
    expected = asdict(policy.config)
    if payload.get("architecture") != expected:
        raise ValueError(
            f"FADA checkpoint architecture mismatch: expected={expected} "
            f"observed={payload.get('architecture')}"
        )
    policy.planner.load_state_dict(payload["planner_state_dict"], strict=True)
    policy.idm.load_state_dict(payload["idm_state_dict"], strict=True)
    if trainer is not None:
        trainer.planner_optimizer.load_state_dict(payload["planner_optimizer_state_dict"])
        trainer.idm_optimizer.load_state_dict(payload["idm_optimizer_state_dict"])
    return payload


def load_fada_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedFADAPlannerIDMPolicy:
    """Construct an inference-only Planner-IDM policy from checkpoint-owned architecture."""

    # B1: 只读取 tensor 与基础类型, 并在模型构造前关闭 schema/architecture 不匹配.
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported or malformed FADA checkpoint schema")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("FADA checkpoint must contain an architecture mapping")
    try:
        config = FADAArchitectureConfig(**architecture)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid FADA checkpoint architecture: {architecture}") from exc

    # B2: 严格恢复 Planner 与 IDM, 产出 eval 模式的唯一 playback policy owner.
    policy = FADAPlannerIDMPolicy(config).to(device)
    policy.planner.load_state_dict(payload["planner_state_dict"], strict=True)
    policy.idm.load_state_dict(payload["idm_state_dict"], strict=True)
    policy.eval()
    return LoadedFADAPlannerIDMPolicy(policy=policy, checkpoint=payload)
