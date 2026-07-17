from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_analyzer():
    path = Path("scripts/deploy/analyze_hp7c3_bounded_run.py")
    spec = importlib.util.spec_from_file_location("analyze_hp7c3_bounded_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metrics_time_and_gpu_semantics(tmp_path: Path) -> None:
    module = _load_analyzer()
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "identity": {
                            "scenario": "walk_flat",
                            "worker_pid": 7,
                            "weight_version": 3,
                            "outer_iteration": 1,
                            "execution_mode": "persistent_async",
                        },
                        "stage": "learner_batch_staging",
                        "duration_seconds": 2.0,
                        "success": True,
                    },
                    {
                        "identity": {
                            "scenario": "__workflow__",
                            "worker_pid": 8,
                            "weight_version": 3,
                            "outer_iteration": 1,
                            "execution_mode": "persistent_async",
                        },
                        "stage": "optimizer_step",
                        "duration_seconds": 3.0,
                        "success": True,
                    },
                ]
            }
        )
    )
    timing = tmp_path / "time.txt"
    timing.write_text(
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n"
        "Maximum resident set size (kbytes): 4096\nExit status: 0\n"
    )
    gpu = tmp_path / "gpu.csv"
    gpu.write_text("t0, 7, GPU-a, 100 MiB\nt1, 7, GPU-a, 300 MiB\n")

    summary = module.analyze_metrics(metrics)
    assert summary["stage_seconds"] == {
        "optimizer_step": 3.0,
        "learner_batch_staging": 2.0,
    }
    assert summary["failure_count"] == 0
    assert module.parse_time_v(timing)["elapsed_seconds"] == 62.5
    gpu_summary = module.parse_gpu_csv(gpu)
    assert gpu_summary["peak_used_gpu_memory_mib"] == 300
    assert gpu_summary["mean_used_gpu_memory_mib"] == 200
