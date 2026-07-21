from __future__ import annotations

from dataclasses import replace

import pytest
from scripts.deploy.check_distill_native_numeric_differential import (
    PythonCanary,
    PythonObjectIntegrityError,
    WorkerConfig,
    assert_python_canary,
    run_numeric_worker,
    stage_matrix,
)


def _tiny_config(**overrides) -> WorkerConfig:
    config = WorkerConfig(
        stage="cpu_tiny",
        device="cpu",
        worker_index=0,
        seconds=1.0,
        iterations=2,
        matrix_size=16,
        batch_size=8,
        allocation_mib=1,
        canary_size=64,
        seed=7,
        serialization_interval=1,
    )
    return replace(config, **overrides)


def test_cpu_tiny_worker_checks_numeric_and_python_canaries() -> None:
    result = run_numeric_worker(_tiny_config())

    assert result["status"] == "completed"
    assert result["device"] == "cpu"
    assert result["iterations_completed"] == 2
    assert result["last_loss"] >= 0.0


def test_python_canary_rejects_an_impossible_object_without_normalizing_it() -> None:
    token = "walk_to_stop::test"

    def callback(value: int) -> int:
        return value + 1

    canary = PythonCanary(
        token=token,
        labels=(token, object()),  # type: ignore[arg-type]
        callback=callback,
        callbacks=(callback,),
        builtin_identities=(str, int, list, tuple, type, isinstance, callable),
    )

    with pytest.raises(PythonObjectIntegrityError, match=r"labels\[1\] changed"):
        assert_python_canary(canary, checkpoint="test")


def test_numeric_worker_propagates_injected_failure_at_exact_iteration() -> None:
    with pytest.raises(RuntimeError, match="synthetic numeric failure at iteration 2"):
        run_numeric_worker(_tiny_config(inject_failure_iteration=2))


def test_stage_matrix_changes_only_device_then_process_count() -> None:
    assert stage_matrix("cuda:0") == (
        ("cpu_single", "cpu", 1),
        ("gpu_single", "cuda:0", 1),
        ("gpu_dual", "cuda:0", 2),
    )
