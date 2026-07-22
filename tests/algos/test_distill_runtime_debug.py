from __future__ import annotations

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
