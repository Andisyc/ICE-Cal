from __future__ import annotations

import torch

from unilab.algos.torch.distill import data, offline, trainer


def test_distill_runtime_debug_prints_are_opt_in(monkeypatch, capsys) -> None:
    monkeypatch.delenv("UNILAB_DISTILL_RUNTIME_DEBUG", raising=False)

    assert trainer._runtime_trace_update(1) is False
    assert trainer._runtime_trace_update(100) is False
    trainer._emit_trainer_runtime("probe/trainer")
    data._emit_data_runtime("probe/data")
    offline._emit_offline_runtime("probe/offline")

    assert capsys.readouterr().out == ""

    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")

    assert trainer._runtime_trace_update(1) is True
    assert trainer._runtime_trace_update(2) is False
    assert trainer._runtime_trace_update(100) is True
    trainer._emit_trainer_runtime("probe/trainer")
    data._emit_data_runtime("probe/data")
    offline._emit_offline_runtime("probe/offline")

    out = capsys.readouterr().out
    assert "[distill-trainer-runtime]" in out
    assert "[distill-data-runtime]" in out
    assert "[distill-offline-runtime]" in out


def test_scenario_label_snapshot_is_cold_when_runtime_debug_is_disabled(monkeypatch) -> None:
    monkeypatch.delenv("UNILAB_DISTILL_RUNTIME_DEBUG", raising=False)

    def fail_if_called(value):
        raise AssertionError(f"unexpected eager diagnostic repr: {value!r}")

    monkeypatch.setattr(data, "_safe_runtime_repr", fail_if_called)

    dataset = data.build_distillation_dataset(
        torch.zeros(4, 3),
        torch.zeros(4, 3),
        scenario_labels=("walk_to_stop",) * 4,
        transition_ages=torch.arange(4),
        command_before=torch.ones(4, 3),
        command_after=torch.zeros(4, 3),
    )

    assert dataset.scenario_labels == ("walk_to_stop",) * 4


def test_scenario_label_snapshot_avoids_closure_cell_iterator(monkeypatch) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    labels = ("walk_to_stop", "invalid")
    source_ranges = (
        {
            "source_index": 0,
            "global_start": 0,
            "global_stop": 2,
            "path": "source.pt",
            "role": "walk_flat",
            "scenario": "walk_to_stop",
        },
    )

    snapshot = data._scenario_label_debug_snapshot(labels, source_ranges=source_ranges)

    assert snapshot["invalid_head"] == [
        {
            "global_index": 1,
            "raw_type": "str",
            "raw_repr": "'invalid'",
            "normalized": "invalid",
            "source_index": 0,
            "source_row_index": 1,
            "path": "source.pt",
            "role": "walk_flat",
            "scenario": "walk_to_stop",
        }
    ]
