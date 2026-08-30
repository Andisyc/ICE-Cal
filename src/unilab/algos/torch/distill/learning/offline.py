from __future__ import annotations

import builtins
import math
import os
import sys
import threading
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.distill.contracts.checkpoint import save_distillation_checkpoint
from unilab.algos.torch.distill.datasets.dataset import DistillationTensorDataset
from unilab.algos.torch.distill.learning.trainer import (
    BehaviorDistillationTrainer,
    DistillationBatch,
)
from unilab.algos.torch.distill.observability.debug import (
    _distill_runtime_debug_enabled,
)
from unilab.algos.torch.distill.observability.performance import (
    DistillationStageObservation,
    DistillationStageObservationAccumulator,
)

_DISTILL_OFFLINE_TRACE_INTERVAL = 100


def _offline_label_counts(labels: tuple[str, ...] | None) -> dict[str, int]:
    return {} if labels is None else dict(Counter(str(label) for label in labels))


def _offline_batch_runtime_snapshot(
    *,
    batch: DistillationBatch,
    update_number: int,
) -> dict[str, Any]:
    return {
        "update_number": update_number,
        "student_obs_shape": tuple(batch.student_obs.shape),
        "teacher_obs_shape": tuple(batch.teacher_obs.shape),
        "role_label_counts": _offline_label_counts(batch.role_labels),
        "command_intent_counts": _offline_label_counts(batch.command_intents),
    }


def _emit_offline_runtime(stage: str, **fields: Any) -> None:
    if not _distill_runtime_debug_enabled():
        return
    current_int = builtins.int
    trace = sys.gettrace()
    profile = sys.getprofile()
    snapshot = {
        "stage": stage,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "builtins_int_type": type(current_int).__name__,
        "builtins_int_repr": repr(current_int),
        "builtins_int_callable": callable(current_int),
        "sys_trace_type": None if trace is None else type(trace).__name__,
        "sys_trace_repr": None if trace is None else repr(trace),
        "sys_profile_type": None if profile is None else type(profile).__name__,
        "sys_profile_repr": None if profile is None else repr(profile),
        **fields,
    }
    print(f"[distill-offline-runtime] {snapshot!r}", flush=True)


@dataclass(frozen=True)
class OfflineDistillationRunResult:
    """Summary of a bounded offline behavior-distillation update loop."""

    update_count: int
    samples_seen: int
    last_loss: float
    last_student_grad_norm: float
    student_obs_shape: tuple[int, ...]
    teacher_obs_shape: tuple[int, ...]
    student_action_shape: tuple[int, ...]
    teacher_action_shape: tuple[int, ...]
    teacher_action_requires_grad: bool
    last_teacher_action_source: str
    checkpoint_path: Path | None
    losses: tuple[float, ...]
    student_grad_norms: tuple[float, ...]
    last_behavior_loss: float
    last_behavior_action_shape: tuple[int, ...]
    last_behavior_action_source: str
    last_behavior_target_count: int
    last_aux_loss: float
    last_role_loss: float
    last_role_target_count: int
    last_command_intent_loss: float
    last_command_intent_target_count: int
    last_expert_usage: tuple[float, ...] | None
    last_route_entropy: float | None
    balance_key: str
    batch_label_counts: tuple[dict[str, int], ...]
    last_balance_label_counts: dict[str, int]
    performance_stage_observations: tuple[DistillationStageObservation, ...] = ()


@dataclass
class _OfflineUpdateState:
    """Mutable observations produced by one bounded offline update loop."""

    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    batch_label_counts: list[dict[str, int]] = field(default_factory=list)
    samples_seen: int = 0
    last_batch: DistillationBatch | None = None
    last_stats: Any | None = None

    def record(self, batch: DistillationBatch, stats: Any, label_counts: dict[str, int]) -> None:
        self.samples_seen += int(batch.student_obs.shape[0])
        self.losses.append(stats.loss)
        self.grad_norms.append(stats.student_grad_norm)
        self.batch_label_counts.append(label_counts)
        self.last_batch = batch
        self.last_stats = stats

    def result(
        self,
        *,
        update_count: int,
        checkpoint_path: Path | None,
        balance_key: str,
        performance_observations: tuple[DistillationStageObservation, ...],
    ) -> OfflineDistillationRunResult:
        if self.last_batch is None or self.last_stats is None:
            raise ValueError("offline distillation loop did not execute any update")
        batch = self.last_batch
        stats = self.last_stats
        return OfflineDistillationRunResult(
            update_count=update_count,
            samples_seen=self.samples_seen,
            last_loss=self.losses[-1],
            last_student_grad_norm=self.grad_norms[-1],
            student_obs_shape=tuple(batch.student_obs.shape),
            teacher_obs_shape=tuple(batch.teacher_obs.shape),
            student_action_shape=stats.student_action_shape,
            teacher_action_shape=stats.teacher_action_shape,
            teacher_action_requires_grad=stats.teacher_action_requires_grad,
            last_teacher_action_source=stats.teacher_action_source,
            checkpoint_path=checkpoint_path,
            losses=tuple(self.losses),
            student_grad_norms=tuple(self.grad_norms),
            last_behavior_loss=stats.behavior_loss,
            last_behavior_action_shape=stats.behavior_action_shape,
            last_behavior_action_source=stats.behavior_action_source,
            last_behavior_target_count=stats.behavior_target_count,
            last_aux_loss=stats.aux_loss,
            last_role_loss=stats.role_loss,
            last_role_target_count=stats.role_target_count,
            last_command_intent_loss=stats.command_intent_loss,
            last_command_intent_target_count=stats.command_intent_target_count,
            last_expert_usage=stats.expert_usage,
            last_route_entropy=stats.route_entropy,
            balance_key=balance_key,
            batch_label_counts=tuple(self.batch_label_counts),
            last_balance_label_counts=(
                self.batch_label_counts[-1] if self.batch_label_counts else {}
            ),
            performance_stage_observations=performance_observations,
        )


@dataclass(frozen=True)
class BalancedLabelIndexPools:
    """为一次 offline invocation 持有 immutable CPU balanced-row pools.

    Status: active, HP-7c owner path locally verified.
    Upstream: run_offline_distillation_updates 从当前 loaded dataset 构建一次.
    Downstream: 每个 update 的 balanced sampler 复用 pools, 不预生成 schedule.
    Evidence: S1/S2 contract-confirmed; CUDA 与 bounded persistent live pending.
    Gap: 不证明端到端 speedup, default-on 或 promotion.
    """

    source_labels: tuple[str, ...]
    balance_key: str
    selected_labels: tuple[str, ...]
    row_indices: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.selected_labels:
            raise ValueError("offline balanced sampler requires at least one label")
        if len(set(self.selected_labels)) != len(self.selected_labels):
            raise ValueError(f"offline balanced labels must be unique: {self.selected_labels}")
        if len(self.selected_labels) != len(self.row_indices):
            raise ValueError("offline balanced label pools must match selected labels")
        for label, indices in zip(self.selected_labels, self.row_indices, strict=True):
            if indices.device.type != "cpu" or indices.dtype != torch.int64:
                raise ValueError(
                    "offline balanced label pools must be CPU int64 tensors: "
                    f"label={label!r} device={indices.device} dtype={indices.dtype}"
                )
            if indices.ndim != 1 or not indices.is_contiguous():
                raise ValueError(
                    "offline balanced label pools must be contiguous rank-1 tensors: "
                    f"label={label!r} shape={tuple(indices.shape)}"
                )
            if indices.numel() == 0:
                raise ValueError(f"offline balanced sampler missing labels: [{label!r}]")
            expected = tuple(
                index
                for index, source_label in enumerate(self.source_labels)
                if source_label == label
            )
            if tuple(int(index) for index in indices) != expected:
                raise ValueError(
                    f"offline balanced label pool does not match source labels: label={label!r}"
                )
        if self.payload_bytes > 8 * len(self.source_labels):
            raise ValueError("offline balanced label pool exceeds the 8N payload bound")

    @property
    def payload_bytes(self) -> int:
        return sum(indices.numel() * indices.element_size() for indices in self.row_indices)


def _indexed_batch(dataset: DistillationTensorDataset, indices: torch.Tensor) -> DistillationBatch:
    indices = indices.to(device=dataset.student_obs.device)
    role_labels = None
    if dataset.role_labels is not None:
        role_labels = tuple(dataset.role_labels[int(index)] for index in indices.detach().cpu())
    return DistillationBatch(
        student_obs=dataset.student_obs.index_select(0, indices),
        teacher_obs=dataset.teacher_obs.index_select(0, indices),
        role_labels=role_labels,
        teacher_actions=(
            None
            if dataset.teacher_actions is None
            else dataset.teacher_actions.index_select(0, indices)
        ),
        commands=None if dataset.commands is None else dataset.commands.index_select(0, indices),
        target_height=(
            None
            if dataset.target_height is None
            else dataset.target_height.index_select(0, indices)
        ),
        command_intents=(
            None
            if dataset.command_intents is None
            else tuple(dataset.command_intents[int(index)] for index in indices.detach().cpu())
        ),
        scenario_labels=(
            None
            if dataset.scenario_labels is None
            else tuple(dataset.scenario_labels[int(index)] for index in indices.detach().cpu())
        ),
        transition_ages=(
            None
            if dataset.transition_ages is None
            else dataset.transition_ages.index_select(0, indices)
        ),
        command_before=(
            None
            if dataset.command_before is None
            else dataset.command_before.index_select(0, indices)
        ),
        command_after=(
            None
            if dataset.command_after is None
            else dataset.command_after.index_select(0, indices)
        ),
    )


def _labels_for_balance_key(
    dataset: DistillationTensorDataset,
    balance_key: str,
) -> tuple[str, ...] | None:
    if balance_key == "none":
        return None
    if balance_key == "role":
        if dataset.role_labels is None:
            raise ValueError("offline balance_key='role' requires dataset.role_labels")
        return dataset.role_labels
    if balance_key == "command_intent":
        if dataset.command_intents is None:
            raise ValueError(
                "offline balance_key='command_intent' requires dataset.command_intents"
            )
        return dataset.command_intents
    if balance_key == "scenario":
        if dataset.scenario_labels is None:
            raise ValueError("offline balance_key='scenario' requires dataset.scenario_labels")
        return dataset.scenario_labels
    raise ValueError(
        "offline balance_key must be one of 'none', 'role', 'command_intent', or 'scenario', "
        f"got {balance_key!r}"
    )


def _balanced_batch_indices(
    labels: tuple[str, ...],
    *,
    batch_size: int,
    balanced_labels: Sequence[str] | None,
    balance_quotas: Mapping[str, float] | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, int]]:
    selected_labels = _resolve_balanced_labels(
        labels, batch_size=batch_size, balanced_labels=balanced_labels
    )
    label_to_indices = _build_balanced_label_pools(
        labels, selected_labels, balance_key="unspecified"
    )
    return _sample_balanced_batch_indices_from_pools(
        label_to_indices,
        batch_size=batch_size,
        balance_quotas=balance_quotas,
        generator=generator,
    )


def _resolve_balanced_labels(
    labels: tuple[str, ...],
    *,
    batch_size: int,
    balanced_labels: Sequence[str] | None,
) -> tuple[str, ...]:
    selected_labels = (
        tuple(str(label) for label in balanced_labels)
        if balanced_labels
        else tuple(sorted(set(labels)))
    )
    if not selected_labels:
        raise ValueError("offline balanced sampler requires at least one label")
    if len(set(selected_labels)) != len(selected_labels):
        raise ValueError(f"offline balanced labels must be unique: {selected_labels}")
    if int(batch_size) < len(selected_labels):
        raise ValueError(
            "offline balanced sampler requires batch_size >= number of labels: "
            f"batch_size={int(batch_size)} labels={len(selected_labels)}"
        )
    return selected_labels


def _build_balanced_label_pools(
    labels: tuple[str, ...],
    selected_labels: Sequence[str],
    *,
    balance_key: str = "unspecified",
) -> BalancedLabelIndexPools:
    """Build immutable CPU row-index pools for the selected balance labels."""
    resolved_labels = tuple(str(label) for label in selected_labels)
    row_indices = tuple(
        torch.as_tensor(
            [idx for idx, value in enumerate(labels) if value == label],
            dtype=torch.long,
        )
        for label in resolved_labels
    )
    return BalancedLabelIndexPools(
        source_labels=labels,
        balance_key=str(balance_key),
        selected_labels=resolved_labels,
        row_indices=row_indices,
    )


def _sample_balanced_batch_indices_from_pools(
    label_to_indices: BalancedLabelIndexPools,
    *,
    batch_size: int,
    balance_quotas: Mapping[str, float] | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Sample one balanced CPU index batch from prevalidated label pools."""
    selected_labels = label_to_indices.selected_labels
    pools_by_label = dict(zip(selected_labels, label_to_indices.row_indices, strict=True))

    if balance_quotas:
        quota_values = {str(label): float(value) for label, value in balance_quotas.items()}
        unknown = sorted(set(quota_values) - set(selected_labels))
        missing = sorted(set(selected_labels) - set(quota_values))
        if unknown or missing:
            raise ValueError(
                "offline balance_quotas labels must match balanced_labels: "
                f"unknown={unknown} missing={missing}"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in quota_values.values()):
            raise ValueError("offline balance_quotas must contain finite positive weights")
        total_weight = sum(quota_values[label] for label in selected_labels)
        exact_quotas = [
            int(batch_size) * quota_values[label] / total_weight for label in selected_labels
        ]
        counts_list = [int(quota) for quota in exact_quotas]
        if any(count < 1 for count in counts_list):
            raise ValueError(
                "offline balance_quotas must allocate at least one sample per label: "
                f"batch_size={int(batch_size)} quotas={quota_values}"
            )
        remainder = int(batch_size) - sum(counts_list)
        order = sorted(
            range(len(selected_labels)),
            key=lambda index: exact_quotas[index] - counts_list[index],
            reverse=True,
        )
        for index in order[:remainder]:
            counts_list[index] += 1
    else:
        base_quota = int(batch_size) // len(selected_labels)
        remainder = int(batch_size) % len(selected_labels)
        counts_list = [
            base_quota + (1 if index < remainder else 0) for index in range(len(selected_labels))
        ]
    chunks: list[torch.Tensor] = []
    counts: dict[str, int] = {}
    for label_idx, label in enumerate(selected_labels):
        quota = counts_list[label_idx]
        source = pools_by_label[label]
        picks = torch.randint(
            int(source.numel()),
            (quota,),
            generator=generator,
            dtype=torch.long,
        )
        chunks.append(source.index_select(0, picks))
        counts[label] = int(quota)
    indices = torch.cat(chunks, dim=0)
    order = torch.randperm(int(indices.numel()), generator=generator)
    return indices.index_select(0, order), counts


def _required_balanced_replay_updates(
    labels: tuple[str, ...],
    *,
    batch_size: int,
    balanced_labels: Sequence[str] | None,
    balance_quotas: Mapping[str, float] | None,
    replay_labels: Sequence[str],
    replay_passes: int,
) -> int:
    """Return updates needed for expected replay passes of selected labels."""

    if int(replay_passes) <= 0 or not replay_labels:
        return 0
    selected_labels = (
        tuple(str(label) for label in balanced_labels)
        if balanced_labels
        else tuple(sorted(set(labels)))
    )
    if not selected_labels:
        raise ValueError("offline replay requires at least one balanced label")
    replay_label_set = {str(label) for label in replay_labels}
    unknown = sorted(replay_label_set - set(selected_labels))
    if unknown:
        raise ValueError(
            f"offline replay labels must be present in balanced_labels: unknown={unknown}"
        )
    if not balance_quotas:
        base = int(batch_size) // len(selected_labels)
        remainder = int(batch_size) % len(selected_labels)
        batch_counts = {
            label: base + int(index < remainder) for index, label in enumerate(selected_labels)
        }
    else:
        quota_values = {str(label): float(value) for label, value in balance_quotas.items()}
        unknown = sorted(set(quota_values) - set(selected_labels))
        missing = sorted(set(selected_labels) - set(quota_values))
        if unknown or missing:
            raise ValueError(
                "offline balance_quotas labels must match balanced_labels: "
                f"unknown={unknown} missing={missing}"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in quota_values.values()):
            raise ValueError("offline balance_quotas must contain finite positive weights")
        total_weight = sum(quota_values[label] for label in selected_labels)
        exact = {
            label: int(batch_size * quota_values[label] / total_weight) for label in selected_labels
        }
        remainder = int(batch_size) - sum(exact.values())
        order = sorted(
            selected_labels,
            key=lambda label: batch_size * quota_values[label] / total_weight - exact[label],
            reverse=True,
        )
        for label in order[:remainder]:
            exact[label] += 1
        batch_counts = exact

    required = 0
    for label in sorted(replay_label_set):
        dataset_count = sum(value == label for value in labels)
        samples_per_update = int(batch_counts[label])
        if dataset_count <= 0 or samples_per_update <= 0:
            raise ValueError(
                "offline replay label has no usable samples: "
                f"label={label!r} dataset_count={dataset_count} "
                f"batch_count={samples_per_update}"
            )
        required = max(
            required,
            int(math.ceil(dataset_count * int(replay_passes) / samples_per_update)),
        )
    return required


def required_balanced_replay_updates(
    dataset: DistillationTensorDataset,
    *,
    balance_key: str,
    batch_size: int,
    balanced_labels: Sequence[str] | None,
    balance_quotas: Mapping[str, float] | None,
    replay_labels: Sequence[str],
    replay_passes: int,
) -> int:
    """Compute the minimum update budget for a balanced replay contract."""

    labels = _labels_for_balance_key(dataset, str(balance_key))
    return _required_balanced_replay_updates(
        labels or (),
        batch_size=int(batch_size),
        balanced_labels=balanced_labels,
        balance_quotas=balance_quotas,
        replay_labels=replay_labels,
        replay_passes=int(replay_passes),
    )


def required_balanced_replay_updates_for_labels(
    labels: Sequence[str],
    *,
    batch_size: int,
    balanced_labels: Sequence[str] | None,
    balance_quotas: Mapping[str, float] | None,
    replay_labels: Sequence[str],
    replay_passes: int,
) -> int:
    """Compute the replay budget from an explicit label sequence."""

    return _required_balanced_replay_updates(
        tuple(str(label) for label in labels),
        batch_size=int(batch_size),
        balanced_labels=balanced_labels,
        balance_quotas=balance_quotas,
        replay_labels=replay_labels,
        replay_passes=int(replay_passes),
    )


@dataclass
class _OfflineBatchSampler:
    """Own deterministic sequential, shuffled, or label-balanced batch selection."""

    dataset: DistillationTensorDataset
    batch_size: int
    repeat_dataset: bool
    shuffle: bool
    balance_quotas: Mapping[str, float] | None
    generator: torch.Generator
    balanced_pools: BalancedLabelIndexPools | None
    order: torch.Tensor
    cursor: int = 0

    @classmethod
    def create(
        cls,
        dataset: DistillationTensorDataset,
        *,
        batch_size: int,
        repeat_dataset: bool,
        shuffle: bool,
        seed: int | None,
        balance_key: str,
        balance_labels: tuple[str, ...] | None,
        balanced_labels: Sequence[str] | None,
        balance_quotas: Mapping[str, float] | None,
    ) -> _OfflineBatchSampler:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
        pools = None
        if balance_labels is not None:
            selected = _resolve_balanced_labels(
                balance_labels, batch_size=batch_size, balanced_labels=balanced_labels
            )
            pools = _build_balanced_label_pools(
                balance_labels, selected, balance_key=balance_key
            )
        order = (
            torch.randperm(dataset.num_samples, generator=generator)
            if shuffle
            else torch.arange(dataset.num_samples)
        )
        return cls(
            dataset=dataset,
            batch_size=batch_size,
            repeat_dataset=repeat_dataset,
            shuffle=shuffle,
            balance_quotas=balance_quotas,
            generator=generator,
            balanced_pools=pools,
            order=order,
        )

    def next_batch(self, update_index: int) -> tuple[DistillationBatch, dict[str, int]] | None:
        if self.balanced_pools is not None:
            indices, counts = _sample_balanced_batch_indices_from_pools(
                self.balanced_pools,
                batch_size=self.batch_size,
                balance_quotas=self.balance_quotas,
                generator=self.generator,
            )
            return _indexed_batch(self.dataset, indices), counts
        if self.repeat_dataset or self.shuffle:
            if self.cursor >= self.dataset.num_samples:
                if not self.repeat_dataset:
                    return None
                self.order = (
                    torch.randperm(self.dataset.num_samples, generator=self.generator)
                    if self.shuffle
                    else torch.arange(self.dataset.num_samples)
                )
                self.cursor = 0
            end = min(self.dataset.num_samples, self.cursor + self.batch_size)
            batch = _indexed_batch(self.dataset, self.order[self.cursor : end])
            self.cursor = end
            return batch, {}
        start = update_index * self.batch_size
        if start >= self.dataset.num_samples:
            return None
        return self.dataset.as_batch(start=start, batch_size=self.batch_size), {}


@dataclass(frozen=True)
class _OfflineUpdateTransaction:
    """Own validation, update diagnostics, persistence, and result publication."""

    trainer: BehaviorDistillationTrainer
    dataset: DistillationTensorDataset
    options: Mapping[str, Any]

    def run(self) -> OfflineDistillationRunResult:
        return _execute_offline_distillation_updates(self, **self.options)

    def run_update(
        self,
        batch: DistillationBatch,
        *,
        update_number: int,
        max_updates: int,
        performance: DistillationStageObservationAccumulator | None,
        balance_key: str,
        label_counts: Mapping[str, int],
        recent_updates: deque[dict[str, Any]],
    ) -> Any:
        snapshot = _offline_batch_runtime_snapshot(batch=batch, update_number=update_number)
        recent_updates.append(snapshot)
        trace_update = (
            update_number == 1
            or update_number % _DISTILL_OFFLINE_TRACE_INTERVAL == 0
            or update_number == max_updates
        )
        if trace_update:
            _emit_offline_runtime(
                "offline/before_trainer_update",
                **snapshot,
                max_updates=max_updates,
                trainer_update_count=self.trainer.update_count,
                balance_key=balance_key,
                sampled_balance_label_counts=dict(label_counts),
            )
        try:
            stats = self.trainer.update(batch, performance=performance)
        except Exception as error:
            _emit_offline_runtime(
                "offline/trainer_update_failure",
                **snapshot,
                max_updates=max_updates,
                trainer_update_count=self.trainer.update_count,
                balance_key=balance_key,
                sampled_balance_label_counts=dict(label_counts),
                error_type=type(error).__name__,
                error_repr=repr(error),
                recent_updates=list(recent_updates),
            )
            raise
        if trace_update:
            _emit_offline_runtime(
                "offline/after_trainer_update",
                **snapshot,
                max_updates=max_updates,
                trainer_update_count=self.trainer.update_count,
                stats_update_count=stats.update_count,
                loss=stats.loss,
                grad_norm=stats.student_grad_norm,
            )
        return stats

    def finalize(
        self,
        state: _OfflineUpdateState,
        *,
        checkpoint_path: str | Path | None,
        teacher_metadata: Mapping[str, Any] | None,
        distill_runtime_cfg: Mapping[str, Any] | None,
        save_optimizer_state: bool,
        performance: DistillationStageObservationAccumulator | None,
        balance_key: str,
    ) -> OfflineDistillationRunResult:
        if not state.losses:
            raise ValueError("offline distillation loop did not execute any update")
        resolved_path = Path(checkpoint_path) if checkpoint_path is not None else None
        if resolved_path is not None:
            span = nullcontext() if performance is None else performance.measure("checkpoint_save")
            with span:
                save_distillation_checkpoint(
                    resolved_path,
                    student=self.trainer.student,
                    optimizer=self.trainer.optimizer if save_optimizer_state else None,
                    agent_steps=state.samples_seen,
                    teacher_metadata=teacher_metadata,
                    distill_runtime_cfg=distill_runtime_cfg,
                )
        observations = (
            ()
            if performance is None
            else tuple(
                performance.observation(
                    stage=stage,
                    row_count=state.samples_seen,
                    env_step_count=0,
                )
                for stage in (
                    "learner_batch_staging",
                    "learner_forward",
                    "learner_backward",
                    "optimizer_step",
                    "checkpoint_save",
                )
            )
        )
        return state.result(
            update_count=self.trainer.update_count,
            checkpoint_path=resolved_path,
            balance_key=balance_key,
            performance_observations=observations,
        )


def _execute_offline_distillation_updates(
    transaction: _OfflineUpdateTransaction,
    *,
    batch_size: int,
    max_updates: int,
    checkpoint_path: str | Path | None = None,
    teacher_metadata: Mapping[str, Any] | None = None,
    distill_runtime_cfg: Mapping[str, Any] | None = None,
    repeat_dataset: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
    balance_key: str = "none",
    balanced_labels: Sequence[str] | None = None,
    balance_quotas: Mapping[str, float] | None = None,
    min_balanced_replay_passes: int = 0,
    min_balanced_replay_labels: Sequence[str] | None = None,
    progress_interval: int = 0,
    progress_callback: Callable[[int, int, Any], None] | None = None,
    performance_clock: Callable[[], float] | None = None,
    save_optimizer_state: bool = True,
) -> OfflineDistillationRunResult:
    """Run a bounded sequential offline distillation loop over a validated dataset."""

    dataset = transaction.dataset

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_updates <= 0:
        raise ValueError(f"max_updates must be positive, got {max_updates}")
    if dataset.num_samples <= 0:
        raise ValueError("offline distillation dataset must contain at least one sample")

    state = _OfflineUpdateState()
    resolved_balance_key = str(balance_key)
    balance_labels = _labels_for_balance_key(dataset, resolved_balance_key)
    replay_labels = tuple(str(label) for label in (min_balanced_replay_labels or ()))
    required_updates = _required_balanced_replay_updates(
        balance_labels or (),
        batch_size=int(batch_size),
        balanced_labels=balanced_labels,
        balance_quotas=balance_quotas,
        replay_labels=replay_labels,
        replay_passes=int(min_balanced_replay_passes),
    )
    if int(max_updates) < required_updates:
        raise ValueError(
            "offline balanced replay budget is too small: "
            f"max_updates={int(max_updates)} required_updates={required_updates} "
            f"replay_labels={list(replay_labels)} "
            f"replay_passes={int(min_balanced_replay_passes)} "
            f"batch_size={int(batch_size)}"
        )
    sampler = _OfflineBatchSampler.create(
        dataset,
        batch_size=int(batch_size),
        repeat_dataset=repeat_dataset,
        shuffle=shuffle,
        seed=seed,
        balance_key=resolved_balance_key,
        balance_labels=balance_labels,
        balanced_labels=balanced_labels,
        balance_quotas=balance_quotas,
    )
    progress_interval = int(progress_interval)
    if progress_interval < 0:
        raise ValueError(f"progress_interval must be non-negative, got {progress_interval}")

    performance = (
        None
        if performance_clock is None
        else DistillationStageObservationAccumulator(clock=performance_clock)
    )
    recent_updates: deque[dict[str, Any]] = deque(maxlen=32)

    for update_idx in range(int(max_updates)):
        label_counts: dict[str, int] = {}
        staging_span = (
            nullcontext() if performance is None else performance.measure("learner_batch_staging")
        )
        with staging_span:
            sampled = sampler.next_batch(update_idx)
            if sampled is None:
                break
            batch, label_counts = sampled
        completed_updates = update_idx + 1
        stats = transaction.run_update(
            batch,
            update_number=completed_updates,
            max_updates=int(max_updates),
            performance=performance,
            balance_key=resolved_balance_key,
            label_counts=label_counts,
            recent_updates=recent_updates,
        )

        if progress_interval > 0 and (
            completed_updates % progress_interval == 0 or completed_updates == int(max_updates)
        ):
            if progress_callback is not None:
                progress_callback(completed_updates, int(max_updates), stats)
            else:
                print(
                    "[distill-progress] "
                    f"updates={completed_updates}/{int(max_updates)} "
                    f"loss={stats.loss:.6f}",
                    flush=True,
                )

        state.record(batch, stats, label_counts)

    return transaction.finalize(
        state,
        checkpoint_path=checkpoint_path,
        teacher_metadata=teacher_metadata,
        distill_runtime_cfg=distill_runtime_cfg,
        save_optimizer_state=save_optimizer_state,
        performance=performance,
        balance_key=resolved_balance_key,
    )


def run_offline_distillation_updates(
    trainer: BehaviorDistillationTrainer,
    dataset: DistillationTensorDataset,
    *,
    batch_size: int,
    max_updates: int,
    checkpoint_path: str | Path | None = None,
    teacher_metadata: Mapping[str, Any] | None = None,
    distill_runtime_cfg: Mapping[str, Any] | None = None,
    repeat_dataset: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
    balance_key: str = "none",
    balanced_labels: Sequence[str] | None = None,
    balance_quotas: Mapping[str, float] | None = None,
    min_balanced_replay_passes: int = 0,
    min_balanced_replay_labels: Sequence[str] | None = None,
    progress_interval: int = 0,
    progress_callback: Callable[[int, int, Any], None] | None = None,
    performance_clock: Callable[[], float] | None = None,
    save_optimizer_state: bool = True,
) -> OfflineDistillationRunResult:
    """Run a bounded sequential offline distillation transaction."""

    return _OfflineUpdateTransaction(
        trainer=trainer,
        dataset=dataset,
        options={
            "batch_size": batch_size,
            "max_updates": max_updates,
            "checkpoint_path": checkpoint_path,
            "teacher_metadata": teacher_metadata,
            "distill_runtime_cfg": distill_runtime_cfg,
            "repeat_dataset": repeat_dataset,
            "shuffle": shuffle,
            "seed": seed,
            "balance_key": balance_key,
            "balanced_labels": balanced_labels,
            "balance_quotas": balance_quotas,
            "min_balanced_replay_passes": min_balanced_replay_passes,
            "min_balanced_replay_labels": min_balanced_replay_labels,
            "progress_interval": progress_interval,
            "progress_callback": progress_callback,
            "performance_clock": performance_clock,
            "save_optimizer_state": save_optimizer_state,
        },
    ).run()
