from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from unilab.algos.torch.distill.contracts.dataset import (
    _command_intents_from_commands,
    _command_intents_from_role_labels,
)
from unilab.algos.torch.distill.datasets.dataset import (
    DistillationTensorDataset,
    annotate_distillation_dataset_scenario,
    build_distillation_dataset,
)
from unilab.algos.torch.distill.datasets.diagnostics import (
    _ORIGINAL_STR,
    _ORIGINAL_TYPE,
    _command_intent_contract_debug_snapshot,
    _command_intent_debug_snapshot,
    _emit_data_runtime,
    _expected_command_intent_for_scenario,
    _label_counts,
    _metadata_workflow_scenario,
    _multitask_source_debug_snapshot,
    _safe_runtime_repr,
    _scenario_label_debug_snapshot,
)
from unilab.algos.torch.distill.datasets.io import load_distillation_dataset


def _source_value(source: Mapping[str, Any], key: str) -> Any:
    value = source.get(key)
    if value in (None, ""):
        raise ValueError(f"multitask source must define non-empty {key!r}")
    return value


@dataclass(frozen=True)
class _MergedMultitaskRows:
    student_obs: torch.Tensor
    teacher_obs: torch.Tensor
    teacher_actions: torch.Tensor
    commands: torch.Tensor | None
    target_height: torch.Tensor | None
    command_intents: tuple[str, ...] | None
    scenario_labels: tuple[str, ...] | None
    transition_ages: torch.Tensor | None
    command_before: torch.Tensor | None
    command_after: torch.Tensor | None
    role_labels: tuple[str, ...]
    scenario_source_ranges: tuple[dict[str, Any], ...]


def _concatenate_multitask_sources(
    *,
    datasets: Sequence[DistillationTensorDataset],
    source_paths: Sequence[str],
    source_roles: Sequence[str],
    source_scenarios: Sequence[str | None],
    source_preserve_role_labels: Sequence[bool],
    source_has_commands: bool,
    source_has_target_height: bool,
    source_has_command_intents: bool,
    transition_presence: Mapping[str, bool],
) -> _MergedMultitaskRows:
    student_obs = torch.cat([dataset.student_obs for dataset in datasets], dim=0)
    teacher_obs = torch.cat([dataset.teacher_obs for dataset in datasets], dim=0)
    teacher_actions = torch.cat(
        [dataset.teacher_actions for dataset in datasets if dataset.teacher_actions is not None],
        dim=0,
    )
    commands = (
        torch.cat([dataset.commands for dataset in datasets if dataset.commands is not None], dim=0)
        if source_has_commands
        else None
    )
    target_height = (
        torch.cat(
            [dataset.target_height for dataset in datasets if dataset.target_height is not None],
            dim=0,
        )
        if source_has_target_height
        else None
    )
    command_intents = (
        tuple(
            intent
            for dataset in datasets
            if dataset.command_intents is not None
            for intent in dataset.command_intents
        )
        if source_has_command_intents
        else None
    )
    scenario_source_ranges: list[dict[str, Any]] = []
    if transition_presence["scenario_labels"]:
        global_start = 0
        for source_index, (source_path, role, scenario, dataset) in enumerate(
            zip(source_paths, source_roles, source_scenarios, datasets, strict=True)
        ):
            assert dataset.scenario_labels is not None
            global_stop = global_start + len(dataset.scenario_labels)
            source_range = {
                "source_index": source_index,
                "path": source_path,
                "role": role,
                "scenario": scenario,
                "global_start": global_start,
                "global_stop": global_stop,
            }
            scenario_source_ranges.append(source_range)
            _emit_data_runtime(
                "multitask/scenario_source_ready",
                **source_range,
                num_samples=dataset.num_samples,
                scenario_labels=_scenario_label_debug_snapshot(dataset.scenario_labels),
            )
            global_start = global_stop
    scenario_labels = (
        tuple(
            label
            for dataset in datasets
            if dataset.scenario_labels is not None
            for label in dataset.scenario_labels
        )
        if transition_presence["scenario_labels"]
        else None
    )
    if scenario_labels is not None:
        for source_range, dataset in zip(scenario_source_ranges, datasets, strict=True):
            assert dataset.scenario_labels is not None
            global_start = cast(int, source_range["global_start"])
            global_stop = cast(int, source_range["global_stop"])
            aggregate_slice = scenario_labels[global_start:global_stop]
            _emit_data_runtime(
                "multitask/scenario_concat_chunk",
                **source_range,
                observation_timing="post_flatten_slice_check",
                source_scenario_labels=_scenario_label_debug_snapshot(dataset.scenario_labels),
                aggregate_slice=_scenario_label_debug_snapshot(aggregate_slice),
                source_matches_aggregate_slice=(dataset.scenario_labels == aggregate_slice),
            )
        _emit_data_runtime(
            "multitask/scenario_concat_complete",
            source_count=len(datasets),
            scenario_labels=_scenario_label_debug_snapshot(
                scenario_labels,
                source_ranges=scenario_source_ranges,
            ),
        )
    transition_ages = (
        torch.cat(
            [
                dataset.transition_ages
                for dataset in datasets
                if dataset.transition_ages is not None
            ],
            dim=0,
        )
        if transition_presence["transition_ages"]
        else None
    )
    command_before = (
        torch.cat(
            [dataset.command_before for dataset in datasets if dataset.command_before is not None],
            dim=0,
        )
        if transition_presence["command_before"]
        else None
    )
    command_after = (
        torch.cat(
            [dataset.command_after for dataset in datasets if dataset.command_after is not None],
            dim=0,
        )
        if transition_presence["command_after"]
        else None
    )
    role_label_chunks: list[tuple[str, ...]] = []
    for role, dataset, preserve in zip(
        source_roles, datasets, source_preserve_role_labels, strict=True
    ):
        if preserve:
            assert dataset.role_labels is not None
            role_label_chunks.append(dataset.role_labels)
        else:
            role_label_chunks.append((role,) * dataset.num_samples)
    role_labels = tuple(label for labels in role_label_chunks for label in labels)
    return _MergedMultitaskRows(
        student_obs=student_obs,
        teacher_obs=teacher_obs,
        teacher_actions=teacher_actions,
        commands=commands,
        target_height=target_height,
        command_intents=command_intents,
        scenario_labels=scenario_labels,
        transition_ages=transition_ages,
        command_before=command_before,
        command_after=command_after,
        role_labels=role_labels,
        scenario_source_ranges=tuple(scenario_source_ranges),
    )


@dataclass(frozen=True)
class _LoadedMultitaskSources:
    datasets: tuple[DistillationTensorDataset, ...]
    roles: tuple[str, ...]
    paths: tuple[str, ...]
    sample_counts: tuple[int, ...]
    metadata: tuple[dict[str, Any], ...]
    preserve_role_labels: tuple[bool, ...]
    scenarios: tuple[str | None, ...]
    has_commands: bool
    has_target_height: bool
    has_command_intents: bool
    transition_presence: dict[str, bool]


@dataclass
class _MultitaskSourceContract:
    """Accumulate cross-source optional-field and dimension invariants."""

    student_obs_dim: int | None = None
    teacher_obs_dim: int | None = None
    teacher_action_dim: int | None = None
    has_commands: bool | None = None
    has_target_height: bool | None = None
    has_command_intents: bool | None = None
    transition_presence: dict[str, bool] | None = None

    def accept(self, dataset: DistillationTensorDataset, *, path: Path, role: str) -> None:
        self.has_commands = self._accept_presence(
            "commands", self.has_commands, dataset.commands is not None
        )
        self.has_target_height = self._accept_presence(
            "target_height", self.has_target_height, dataset.target_height is not None
        )
        self.has_command_intents = self._accept_presence(
            "command_intents", self.has_command_intents, dataset.command_intents is not None
        )
        transition_presence = {
            "scenario_labels": dataset.scenario_labels is not None,
            "transition_ages": dataset.transition_ages is not None,
            "command_before": dataset.command_before is not None,
            "command_after": dataset.command_after is not None,
        }
        if self.transition_presence is None:
            self.transition_presence = transition_presence
        elif transition_presence != self.transition_presence:
            mismatched = next(
                name
                for name, present in transition_presence.items()
                if present != self.transition_presence[name]
            )
            raise ValueError(f"multitask sources must either all include {mismatched} or none")
        self.student_obs_dim = self._accept_dim(
            "student_obs", self.student_obs_dim, dataset.student_obs_dim, path, role
        )
        self.teacher_obs_dim = self._accept_dim(
            "teacher_obs", self.teacher_obs_dim, dataset.teacher_obs_dim, path, role
        )
        self.teacher_action_dim = self._accept_dim(
            "teacher_actions", self.teacher_action_dim, dataset.teacher_action_dim, path, role
        )

    @staticmethod
    def _accept_presence(name: str, expected: bool | None, actual: bool) -> bool:
        if expected is not None and actual != expected:
            raise ValueError(f"multitask sources must either all include {name} or none")
        return actual

    @staticmethod
    def _accept_dim(
        name: str,
        expected: int | None,
        actual: int,
        path: Path,
        role: str,
    ) -> int:
        if expected is not None and actual != expected:
            raise ValueError(
                f"multitask source {path} role={role!r} {name} dim mismatch: "
                f"expected {expected}, got {actual}"
            )
        return actual


def _merged_multitask_metadata(
    loaded: _LoadedMultitaskSources,
    command_intents: tuple[str, ...] | None,
) -> dict[str, Any]:
    metadata = {
        "source": "multitask_adapter",
        "source_count": len(loaded.datasets),
        "source_paths": list(loaded.paths),
        "source_roles": list(loaded.roles),
        "source_sample_counts": list(loaded.sample_counts),
        "source_metadata": list(loaded.metadata),
        "source_scenarios": list(loaded.scenarios),
    }
    if command_intents is not None:
        metadata["command_intent_counts"] = _label_counts(command_intents)
    return metadata


def _raise_final_validation_diagnostic(
    *,
    error: ValueError,
    loaded: _LoadedMultitaskSources,
    merged: _MergedMultitaskRows,
    command_intents_before: dict[str, Any] | None,
    scenario_labels_before: dict[str, Any] | None,
) -> None:
    datasets = loaded.datasets
    error_text = _ORIGINAL_STR(error)
    if merged.scenario_labels is not None and "scenario_labels" in error_text:
        after_failure = _scenario_label_debug_snapshot(
            merged.scenario_labels,
            source_ranges=merged.scenario_source_ranges,
            force=True,
        )
        _emit_data_runtime(
            "multitask/final_validation_failure",
            source_count=len(datasets),
            error_type=_ORIGINAL_TYPE(error).__name__,
            error_repr=_safe_runtime_repr(error),
            scenario_labels_before=scenario_labels_before,
            scenario_labels_after=after_failure,
        )
        sources = []
        for source_range, dataset in zip(
            merged.scenario_source_ranges, datasets, strict=True
        ):
            assert dataset.scenario_labels is not None
            sources.append(
                {
                    **source_range,
                    "num_samples": dataset.num_samples,
                    "scenario_labels": _scenario_label_debug_snapshot(
                        dataset.scenario_labels, force=True
                    ),
                }
            )
        snapshot = {
            "stage": "multitask/final_validation_failure",
            "pid": os.getpid(),
            "error_type": _ORIGINAL_TYPE(error).__name__,
            "error": error_text,
            "source_count": len(datasets),
            "sources": sources,
            "aggregate": after_failure,
            "before_final_validation": scenario_labels_before,
        }
        print(
            "[distill-scenario-label-sentinel] " + json.dumps(snapshot, sort_keys=True),
            flush=True,
        )
        raise error
    if merged.command_intents is None or "command_intents" not in error_text:
        raise error
    after_failure = _command_intent_debug_snapshot(merged.command_intents)
    _emit_data_runtime(
        "multitask/final_validation_failure",
        source_count=len(datasets),
        error_type=_ORIGINAL_TYPE(error).__name__,
        error_repr=_safe_runtime_repr(error),
        command_intents_before=command_intents_before,
        command_intents_after=after_failure,
    )
    sources = []
    for path, role, scenario, dataset in zip(
        loaded.paths, loaded.roles, loaded.scenarios, datasets, strict=True
    ):
        assert dataset.command_intents is not None
        source_intents = _command_intent_debug_snapshot(dataset.command_intents)
        sources.append(
            {
                "path": path,
                "role": role,
                "scenario": scenario,
                "num_samples": dataset.num_samples,
                "command_intent_counts": source_intents["command_intent_counts"],
                "invalid_head": source_intents["invalid_head"],
            }
        )
    snapshot = {
        "stage": "multitask/final_validation_failure",
        "pid": os.getpid(),
        "error_type": _ORIGINAL_TYPE(error).__name__,
        "error": error_text,
        "source_count": len(datasets),
        "sources": sources,
        "before_final_validation": command_intents_before,
        "after_final_validation_failure": after_failure,
    }
    print("[distill-command-intent-sentinel] " + json.dumps(snapshot, sort_keys=True))
    raise error


@dataclass(frozen=True)
class _LoadedMultitaskSource:
    dataset: DistillationTensorDataset
    role: str
    path: str
    metadata: dict[str, Any]
    preserve_role_labels: bool
    scenario: str | None


def _load_multitask_source(
    source: Mapping[str, Any],
    *,
    loop_source_index: int,
    expected_student_obs_dim: int | None,
    expected_teacher_obs_dim: int | None,
    expected_teacher_action_dim: int | None,
    device: str | torch.device,
    preserve_source_role_labels: bool,
) -> _LoadedMultitaskSource:
    source_index = int(source.get("source_index", loop_source_index))
    path = Path(_source_value(source, "path"))
    role = str(_source_value(source, "role"))
    scenario = source.get("scenario")
    requested_scenario = None if scenario in (None, "") else str(scenario)
    _emit_data_runtime(
        "multitask/before_source_load",
        source_index=source_index,
        path=str(path),
        role=role,
        scenario=requested_scenario,
    )
    dataset = load_distillation_dataset(
        path,
        expected_student_obs_dim=expected_student_obs_dim,
        expected_teacher_obs_dim=expected_teacher_obs_dim,
        expected_teacher_action_dim=expected_teacher_action_dim,
        device=device,
    )
    metadata_scenario = _metadata_workflow_scenario(dataset)
    if requested_scenario is None:
        requested_scenario = metadata_scenario
    elif metadata_scenario is not None and metadata_scenario != requested_scenario:
        snapshot = {
            "stage": "multitask/source_scenario_contract_mismatch",
            "pid": os.getpid(),
            **_multitask_source_debug_snapshot(
                source_index=source_index,
                path=path,
                role=role,
                scenario=requested_scenario,
                dataset=dataset,
            ),
            "metadata_workflow_scenario": metadata_scenario,
        }
        print(
            "[distill-source-contract-sentinel] " + json.dumps(snapshot, sort_keys=True),
            flush=True,
        )
        raise ValueError(
            "multitask source scenario contract mismatch: "
            + json.dumps(snapshot, sort_keys=True)
        )
    if requested_scenario is not None:
        try:
            dataset = annotate_distillation_dataset_scenario(dataset, requested_scenario)
        except ValueError as error:
            snapshot = {
                "stage": "multitask/source_annotation_failure",
                "pid": os.getpid(),
                **_multitask_source_debug_snapshot(
                    source_index=source_index,
                    path=path,
                    role=role,
                    scenario=requested_scenario,
                    dataset=dataset,
                    error=error,
                ),
            }
            _emit_data_runtime(
                "multitask/source_annotation_failure",
                **{key: value for key, value in snapshot.items() if key != "stage"},
            )
            print(
                "[distill-source-annotation-sentinel] "
                + json.dumps(snapshot, sort_keys=True),
                flush=True,
            )
            raise ValueError(
                "multitask source scenario annotation failed: "
                + json.dumps(snapshot, sort_keys=True)
            ) from error
    _emit_data_runtime(
        "multitask/after_source_annotation",
        source_index=source_index,
        path=str(path),
        role=role,
        scenario=requested_scenario,
        num_samples=dataset.num_samples,
        student_obs_shape=tuple(dataset.student_obs.shape),
        teacher_obs_shape=tuple(dataset.teacher_obs.shape),
        teacher_actions_shape=(
            None if dataset.teacher_actions is None else tuple(dataset.teacher_actions.shape)
        ),
        target_height_shape=(
            None if dataset.target_height is None else tuple(dataset.target_height.shape)
        ),
        command_intents=(
            None
            if dataset.command_intents is None
            else _command_intent_debug_snapshot(dataset.command_intents)
        ),
        scenario_labels=(
            None
            if dataset.scenario_labels is None
            else _scenario_label_debug_snapshot(dataset.scenario_labels)
        ),
    )
    preserve_row_labels = bool(
        source.get("preserve_row_role_labels", preserve_source_role_labels)
    )
    if preserve_row_labels and dataset.role_labels is None:
        raise ValueError(
            f"multitask source {path} requires row role_labels when preservation is enabled"
        )
    if dataset.teacher_actions is None:
        raise ValueError(f"multitask source {path} must contain cached teacher_actions")
    return _LoadedMultitaskSource(
        dataset=dataset,
        role=role,
        path=str(path),
        metadata=dict(dataset.metadata),
        preserve_role_labels=preserve_row_labels,
        scenario=requested_scenario,
    )

def _load_multitask_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    expected_student_obs_dim: int | None,
    expected_teacher_obs_dim: int | None,
    expected_teacher_action_dim: int | None,
    device: str | torch.device,
    preserve_source_role_labels: bool,
) -> _LoadedMultitaskSources:
    if not sources:
        raise ValueError("multitask distillation dataset requires at least one source")

    _emit_data_runtime(
        "multitask/entry",
        source_count=len(sources),
        expected_student_obs_dim=expected_student_obs_dim,
        expected_teacher_obs_dim=expected_teacher_obs_dim,
        expected_teacher_action_dim=expected_teacher_action_dim,
        device=str(device),
    )

    datasets: list[DistillationTensorDataset] = []
    source_roles: list[str] = []
    source_paths: list[str] = []
    source_sample_counts: list[int] = []
    source_metadata: list[dict[str, Any]] = []
    source_preserve_role_labels: list[bool] = []
    source_scenarios: list[str | None] = []
    contract = _MultitaskSourceContract()
    for loop_source_index, source in enumerate(sources):
        loaded_source = _load_multitask_source(
            source,
            loop_source_index=loop_source_index,
            expected_student_obs_dim=expected_student_obs_dim,
            expected_teacher_obs_dim=expected_teacher_obs_dim,
            expected_teacher_action_dim=expected_teacher_action_dim,
            device=device,
            preserve_source_role_labels=preserve_source_role_labels,
        )
        contract.accept(
            loaded_source.dataset,
            path=Path(loaded_source.path),
            role=loaded_source.role,
        )
        datasets.append(loaded_source.dataset)
        source_roles.append(loaded_source.role)
        source_paths.append(loaded_source.path)
        source_sample_counts.append(loaded_source.dataset.num_samples)
        source_metadata.append(loaded_source.metadata)
        source_preserve_role_labels.append(loaded_source.preserve_role_labels)
        source_scenarios.append(loaded_source.scenario)

    if contract.transition_presence is None:
        raise RuntimeError("multitask source transition presence was not initialized")

    return _LoadedMultitaskSources(
        datasets=tuple(datasets),
        roles=tuple(source_roles),
        paths=tuple(source_paths),
        sample_counts=tuple(source_sample_counts),
        metadata=tuple(source_metadata),
        preserve_role_labels=tuple(source_preserve_role_labels),
        scenarios=tuple(source_scenarios),
        has_commands=bool(contract.has_commands),
        has_target_height=bool(contract.has_target_height),
        has_command_intents=bool(contract.has_command_intents),
        transition_presence=contract.transition_presence,
    )


def _build_merged_multitask_dataset(
    loaded: _LoadedMultitaskSources,
    *,
    expected_student_obs_dim: int | None,
    expected_teacher_obs_dim: int | None,
    expected_teacher_action_dim: int | None,
) -> DistillationTensorDataset:
    datasets = loaded.datasets
    source_roles = list(loaded.roles)
    source_paths = list(loaded.paths)
    source_preserve_role_labels = list(loaded.preserve_role_labels)
    source_scenarios = list(loaded.scenarios)
    source_has_commands = loaded.has_commands
    source_has_target_height = loaded.has_target_height
    source_has_command_intents = loaded.has_command_intents
    validated_transition_presence = loaded.transition_presence
    merged = _concatenate_multitask_sources(
        datasets=datasets,
        source_paths=source_paths,
        source_roles=source_roles,
        source_scenarios=source_scenarios,
        source_preserve_role_labels=source_preserve_role_labels,
        source_has_commands=bool(source_has_commands),
        source_has_target_height=bool(source_has_target_height),
        source_has_command_intents=bool(source_has_command_intents),
        transition_presence=validated_transition_presence,
    )
    student_obs = merged.student_obs
    teacher_obs = merged.teacher_obs
    teacher_actions = merged.teacher_actions
    commands = merged.commands
    target_height = merged.target_height
    command_intents = merged.command_intents
    scenario_labels = merged.scenario_labels
    transition_ages = merged.transition_ages
    command_before = merged.command_before
    command_after = merged.command_after
    role_labels = merged.role_labels
    scenario_source_ranges = merged.scenario_source_ranges
    _emit_data_runtime(
        "multitask/after_concat",
        source_count=len(datasets),
        student_obs_shape=tuple(student_obs.shape),
        teacher_obs_shape=tuple(teacher_obs.shape),
        teacher_actions_shape=tuple(teacher_actions.shape),
        commands_shape=None if commands is None else tuple(commands.shape),
        target_height_shape=(None if target_height is None else tuple(target_height.shape)),
        command_intents=(
            None if command_intents is None else _command_intent_debug_snapshot(command_intents)
        ),
        scenario_labels=(
            None
            if scenario_labels is None
            else _scenario_label_debug_snapshot(
                scenario_labels,
                source_ranges=scenario_source_ranges,
            )
        ),
        role_labels_length=len(role_labels),
    )
    metadata = _merged_multitask_metadata(loaded, command_intents)
    before_final_validation = (
        None if command_intents is None else _command_intent_debug_snapshot(command_intents)
    )
    before_final_scenario_validation = (
        None
        if scenario_labels is None
        else _scenario_label_debug_snapshot(
            scenario_labels,
            source_ranges=scenario_source_ranges,
        )
    )
    _emit_data_runtime(
        "multitask/before_final_validation",
        source_count=len(datasets),
        command_intents=before_final_validation,
        scenario_labels=before_final_scenario_validation,
        student_obs_shape=tuple(student_obs.shape),
        teacher_obs_shape=tuple(teacher_obs.shape),
        role_labels_length=len(role_labels),
    )
    try:
        result = build_distillation_dataset(
            student_obs,
            teacher_obs,
            expected_student_obs_dim=expected_student_obs_dim,
            expected_teacher_obs_dim=expected_teacher_obs_dim,
            expected_teacher_action_dim=expected_teacher_action_dim,
            metadata=metadata,
            role_labels=role_labels,
            teacher_actions=teacher_actions,
            commands=commands,
            target_height=target_height,
            command_intents=command_intents,
            scenario_labels=scenario_labels,
            transition_ages=transition_ages,
            command_before=command_before,
            command_after=command_after,
        )
        _emit_data_runtime(
            "multitask/after_final_validation",
            source_count=len(datasets),
            num_samples=result.num_samples,
            command_intents=(
                None
                if result.command_intents is None
                else _command_intent_debug_snapshot(result.command_intents)
            ),
            scenario_labels=(
                None
                if result.scenario_labels is None
                else _scenario_label_debug_snapshot(
                    result.scenario_labels,
                    source_ranges=scenario_source_ranges,
                )
            ),
        )
        return result
    except ValueError as error:
        _raise_final_validation_diagnostic(
            error=error,
            loaded=loaded,
            merged=merged,
            command_intents_before=before_final_validation,
            scenario_labels_before=before_final_scenario_validation,
        )


def build_multitask_distillation_dataset(
    sources: Sequence[Mapping[str, Any]],
    *,
    expected_student_obs_dim: int | None = None,
    expected_teacher_obs_dim: int | None = None,
    expected_teacher_action_dim: int | None = None,
    device: str | torch.device = "cpu",
    preserve_source_role_labels: bool = False,
) -> DistillationTensorDataset:
    """Merge saved role-specific datasets into one cached-target dataset."""

    loaded = _load_multitask_sources(
        sources,
        expected_student_obs_dim=expected_student_obs_dim,
        expected_teacher_obs_dim=expected_teacher_obs_dim,
        expected_teacher_action_dim=expected_teacher_action_dim,
        device=device,
        preserve_source_role_labels=preserve_source_role_labels,
    )
    return _build_merged_multitask_dataset(
        loaded,
        expected_student_obs_dim=expected_student_obs_dim,
        expected_teacher_obs_dim=expected_teacher_obs_dim,
        expected_teacher_action_dim=expected_teacher_action_dim,
    )
