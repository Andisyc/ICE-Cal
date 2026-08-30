"""Read-only serialization probes and DAgger logger callback state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch

from unilab.logging import OffPolicyLogger


def _probe_torch_serialization_runtime(stage: str) -> dict[str, Any]:
    """Fail fast when the parent process torch serialization identity is corrupted."""

    is_storage = torch.is_storage
    snapshot = {
        "stage": str(stage),
        "pid": os.getpid(),
        "is_storage_type": type(is_storage).__name__,
        "is_storage_callable": callable(is_storage),
        "is_storage_module": getattr(is_storage, "__module__", None),
    }
    print(
        f"[distill-runtime-sentinel] {json.dumps(snapshot, sort_keys=True)}",
        flush=True,
    )
    if not snapshot["is_storage_callable"]:
        raise RuntimeError(
            "torch serialization runtime identity corrupted: "
            f"stage={snapshot['stage']} pid={snapshot['pid']} "
            f"type={snapshot['is_storage_type']} "
            f"callable={snapshot['is_storage_callable']}"
        )
    return snapshot

@dataclass
class WorkflowLoggerCallbacks:
    logger: OffPolicyLogger
    runtime_sentinel: Callable[[str], Any] | None
    current_iteration: int = 0

    def on_iteration(self, iteration: int, total: int) -> None:
        self.current_iteration = int(iteration)
        if self.runtime_sentinel is not None:
            self.runtime_sentinel(f"workflow/iteration_{iteration}/logger_iteration_entry")
        self.logger.log_step(self.current_iteration)
        self.logger.log_status(
            f"Iteration {iteration}/{total}: collecting scenarios",
            force=True,
        )
        if self.runtime_sentinel is not None:
            self.runtime_sentinel(f"workflow/iteration_{iteration}/logger_iteration_exit")

    def on_status(self, status: str) -> None:
        if self.runtime_sentinel is not None:
            self.runtime_sentinel(
                f"workflow/iteration_{self.current_iteration}/status_callback_entry"
            )
        self.logger.log_status(status, force=True)
        if self.runtime_sentinel is not None:
            self.runtime_sentinel(
                f"workflow/iteration_{self.current_iteration}/status_callback_exit"
            )

    def on_update_progress(self, update: int, total: int, stats: Any) -> None:
        metrics = {
            "loss/total": float(stats.loss),
            "loss/behavior": float(stats.behavior_loss),
            "loss/aux": float(stats.aux_loss),
            "loss/role": float(stats.role_loss),
            "loss/command_intent": float(stats.command_intent_loss),
            "train/grad_norm": float(stats.student_grad_norm),
            "train/update": float(update),
        }
        if stats.route_entropy is not None:
            metrics["router/route_entropy"] = float(stats.route_entropy)
        self.logger.log_step(self.current_iteration, metrics=metrics)
        self.logger.log_status(
            f"Iteration {self.current_iteration}: update {update:,}/{total:,}",
            force=True,
        )
