from __future__ import annotations

import torch


def test_deep_collection_and_iteration_owners_are_typed_boundaries() -> None:
    from dataclasses import fields, is_dataclass

    from unilab.algos.torch.distill.collection_transition import (
        _PreparedTransitionCollection,
        _TransitionCollectionOutcome,
    )
    from unilab.algos.torch.distill.entry_collection import _CollectionEntryContext
    from unilab.algos.torch.distill.fada_collection_transaction import (
        FADACollectionTransaction,
    )
    from unilab.algos.torch.distill.offline import _OfflineUpdateTransaction
    from unilab.algos.torch.distill.workflow_dagger_iteration import (
        DaggerIterationContext,
        DaggerIterationResult,
    )

    assert all(
        is_dataclass(owner)
        for owner in (
            _PreparedTransitionCollection,
            _TransitionCollectionOutcome,
            FADACollectionTransaction,
            DaggerIterationContext,
            DaggerIterationResult,
            _CollectionEntryContext,
            _OfflineUpdateTransaction,
        )
    )
    assert {field.name for field in fields(DaggerIterationResult)} == {
        "checkpoint",
        "cumulative_num_samples",
    }
    assert callable(FADACollectionTransaction.run)
    assert callable(DaggerIterationContext.run)
    assert callable(_CollectionEntryContext.collect)
    assert callable(_OfflineUpdateTransaction.run)


def test_multitask_merge_has_single_source_contract_owner() -> None:
    from dataclasses import is_dataclass

    from unilab.algos.torch.distill.dataset_merge import _MultitaskSourceContract

    assert is_dataclass(_MultitaskSourceContract)
    assert callable(_MultitaskSourceContract.accept)


def test_remaining_distill_hotspots_have_phase_owners() -> None:
    """Catch regression to five monolithic orchestration functions."""

    from dataclasses import is_dataclass

    from unilab.algos.torch.distill.collection_transition import (
        _TransitionCollectionState,
    )
    from unilab.algos.torch.distill.dataset_merge import (
        _load_multitask_source,
        _LoadedMultitaskSource,
    )
    from unilab.algos.torch.distill.entry_workflow import WorkflowRuntimeSession
    from unilab.algos.torch.distill.offline import _OfflineUpdateTransaction
    from unilab.algos.torch.distill.trainer import _TrainerForwardPass

    assert all(
        is_dataclass(owner)
        for owner in (
            WorkflowRuntimeSession,
            _TrainerForwardPass,
            _TransitionCollectionState,
            _LoadedMultitaskSource,
        )
    )
    assert callable(WorkflowRuntimeSession.run)
    assert callable(_OfflineUpdateTransaction.run_update)
    assert callable(_load_multitask_source)


def test_trainer_routing_resolves_labels_in_row_order() -> None:
    from unilab.algos.torch.distill.trainer_routing import resolve_label_target_indices

    indices = resolve_label_target_indices(
        labels=("walk", "stand", "walk"),
        targets={"stand": 0, "walk": 2},
        batch_size=3,
        num_experts=3,
        label_name="role",
        required=True,
    )

    assert indices == (2, 0, 2)


def test_trainer_routing_returns_none_for_optional_unmapped_label() -> None:
    from unilab.algos.torch.distill.trainer_routing import resolve_label_target_indices

    assert (
        resolve_label_target_indices(
            labels=("unknown",),
            targets={"walk": 1},
            batch_size=1,
            num_experts=2,
            label_name="role",
            required=False,
        )
        is None
    )


def test_trainer_routing_materializes_long_tensor_on_requested_device() -> None:
    from unilab.algos.torch.distill.trainer_routing import materialize_target_indices

    result = materialize_target_indices((1, 0, 1), device=torch.device("cpu"))

    assert result.dtype is torch.long
    assert result.device.type == "cpu"
    assert result.tolist() == [1, 0, 1]


def test_transition_buffer_preserves_row_append_order() -> None:
    from unilab.algos.torch.distill.collection_transition import TransitionRowBuffer

    buffer = TransitionRowBuffer()
    buffer.append(
        student_obs=torch.tensor([[1.0, 2.0]]),
        teacher_obs=torch.tensor([[3.0]]),
        teacher_actions=torch.tensor([[4.0]]),
        role_labels=("walk",),
        command_intents=("forward",),
        scenario_labels=("steady",),
        transition_ages=torch.tensor([[5.0]]),
        command_before=torch.tensor([[0.1, 0.0, 0.0]]),
        command_after=torch.tensor([[0.2, 0.0, 0.0]]),
    )
    buffer.append(
        student_obs=torch.tensor([[6.0, 7.0]]),
        teacher_obs=torch.tensor([[8.0]]),
        teacher_actions=torch.tensor([[9.0]]),
        role_labels=("stand",),
        command_intents=("stationary",),
        scenario_labels=("transition",),
        transition_ages=torch.tensor([[10.0]]),
        command_before=torch.tensor([[0.0, 0.0, 0.0]]),
        command_after=torch.tensor([[0.0, 0.0, 0.0]]),
    )

    assert torch.cat(buffer.student_obs, dim=0).tolist() == [[1.0, 2.0], [6.0, 7.0]]
    assert buffer.role_labels == ["walk", "stand"]
    assert buffer.command_intents == ["forward", "stationary"]
    assert buffer.scenario_labels == ["steady", "transition"]


def test_fada_window_accumulator_compacts_without_losing_window_count() -> None:
    from types import SimpleNamespace

    from unilab.algos.torch.distill.fada_collection_state import FADAWindowAccumulator

    merged: list[tuple[object, ...]] = []

    def merge(batches, _config):
        merged.append(tuple(batches))
        return SimpleNamespace(observation_history=torch.zeros((len(batches), 1, 1)))

    accumulator = FADAWindowAccumulator(config=object(), compact_size=2, merge=merge)
    accumulator.append(SimpleNamespace(observation_history=torch.zeros((1, 1, 1))))
    accumulator.append(SimpleNamespace(observation_history=torch.zeros((1, 1, 1))))

    assert accumulator.window_count == 2
    assert len(merged) == 1
    assert accumulator.pending_batches == []


def test_deep_distill_owners_do_not_import_compatibility_facades() -> None:
    from pathlib import Path

    owner_paths = (
        Path("src/unilab/algos/torch/distill/trainer_routing.py"),
        Path("src/unilab/algos/torch/distill/trainer_diagnostics.py"),
    )
    forbidden = (
        "from unilab.algos.torch.distill.data import",
        "from unilab.algos.torch.distill.collector import",
        "from unilab.algos.torch.distill.workflow import",
        "from scripts",
    )

    for path in owner_paths:
        source = path.read_text()
        assert not any(token in source for token in forbidden)
