"""Benchmark DAgger learner batch staging without running learner updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch

from unilab.algos.torch.distill.data import DistillationTensorDataset, load_distillation_dataset
from unilab.algos.torch.distill.offline import (
    _build_balanced_label_pools,
    _sample_balanced_batch_indices_from_pools,
)

T = TypeVar("T")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dataset_path(
    *, dataset_path: Path | None, run_dir: Path | None, outer_iteration: int | None
) -> Path:
    if dataset_path is not None:
        if run_dir is not None or outer_iteration is not None:
            raise ValueError("--dataset cannot be combined with --run-dir/--outer-iteration")
        resolved = dataset_path.resolve()
    else:
        if run_dir is None or outer_iteration is None:
            raise ValueError("provide --dataset or both --run-dir and --outer-iteration")
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        matches = [
            item
            for item in manifest.get("dagger_iterations", [])
            if int(item.get("iteration", -1)) == int(outer_iteration)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one manifest record for outer iteration {outer_iteration}, "
                f"found {len(matches)}"
            )
        resolved = Path(str(matches[0]["aggregate_dataset_path"])).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"distillation dataset does not exist: {resolved}")
    return resolved


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, fn: Callable[[], T]) -> tuple[T, float]:
    _sync(device)
    start = time.perf_counter()
    value = fn()
    _sync(device)
    return value, time.perf_counter() - start


def _tensor_fields(dataset: DistillationTensorDataset) -> tuple[torch.Tensor, ...]:
    fields = [dataset.student_obs, dataset.teacher_obs]
    for value in (
        dataset.teacher_actions,
        dataset.commands,
        dataset.transition_ages,
        dataset.command_before,
        dataset.command_after,
    ):
        if value is not None:
            fields.append(value)
    return tuple(fields)


def _select_tensors(
    tensors: Sequence[torch.Tensor], device_indices: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.index_select(0, device_indices) for tensor in tensors)


def _label_fields(dataset: DistillationTensorDataset) -> tuple[tuple[str, ...], ...]:
    return tuple(
        value
        for value in (dataset.role_labels, dataset.command_intents, dataset.scenario_labels)
        if value is not None
    )


def _recover_labels(
    label_fields: Sequence[tuple[str, ...]], cpu_indices: torch.Tensor
) -> tuple[tuple[str, ...], ...]:
    positions = tuple(int(index) for index in cpu_indices)
    return tuple(tuple(labels[index] for index in positions) for labels in label_fields)


def _recover_labels_like_current(
    label_fields: Sequence[tuple[str, ...]], device_indices: torch.Tensor
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(labels[int(index)] for index in device_indices.detach().cpu())
        for labels in label_fields
    )


def _add(totals: dict[str, float], stage: str, duration: float) -> None:
    totals[stage] = totals.get(stage, 0.0) + float(duration)


def _parse_quotas(values: Sequence[str]) -> dict[str, float] | None:
    if not values:
        return None
    quotas: dict[str, float] = {}
    for value in values:
        label, separator, weight = value.partition("=")
        if not separator or not label:
            raise ValueError(f"quota must use LABEL=WEIGHT, got {value!r}")
        quotas[label] = float(weight)
    return quotas


def run_staging_benchmark(
    *,
    dataset_path: Path,
    device: torch.device,
    batch_size: int,
    updates: int,
    warmup_updates: int,
    seed: int,
    balance_key: str,
    balanced_labels: Sequence[str],
    balance_quotas: Mapping[str, float] | None,
) -> dict[str, Any]:
    if batch_size <= 0 or updates <= 0 or warmup_updates < 0:
        raise ValueError("batch_size/updates must be positive and warmup_updates non-negative")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("HP-7a requested CUDA, but torch.cuda.is_available() is false")
        device_index = torch.cuda.current_device() if device.index is None else device.index
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(
                "HP-7a CUDA device index is not visible: "
                f"requested={device_index} visible_device_count={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(device_index)
    dataset = load_distillation_dataset(dataset_path, device=device)
    if device.type == "cuda":
        # The dataset load initializes the CUDA context. Older supported PyTorch
        # builds may reject an uninitialized torch.device in this API, so reset
        # with the validated integer index only after the first CUDA allocation.
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    labels = {
        "role": dataset.role_labels,
        "command_intent": dataset.command_intents,
        "scenario": dataset.scenario_labels,
    }[balance_key]
    if labels is None:
        raise ValueError(f"dataset has no labels for balance_key={balance_key!r}")
    selected_labels = tuple(balanced_labels) if balanced_labels else tuple(sorted(set(labels)))
    if len(set(selected_labels)) != len(selected_labels):
        raise ValueError(f"balanced labels must be unique: {selected_labels}")

    tensors = _tensor_fields(dataset)
    string_labels = _label_fields(dataset)
    cached_pools, cached_pool_seconds = _timed(
        torch.device("cpu"), lambda: _build_balanced_label_pools(labels, selected_labels)
    )
    current_generator = torch.Generator(device="cpu").manual_seed(seed)
    cached_generator = torch.Generator(device="cpu").manual_seed(seed)
    current_totals: dict[str, float] = {}
    cached_totals: dict[str, float] = {"label_pool_build_once": cached_pool_seconds}
    all_indices_equal = True
    all_counts_equal = True
    all_labels_equal = True
    all_tensors_equal = True

    total_iterations = warmup_updates + updates
    for iteration in range(total_iterations):
        current_pools, pool_seconds = _timed(
            torch.device("cpu"), lambda: _build_balanced_label_pools(labels, selected_labels)
        )
        (current_indices, current_counts), sampling_seconds = _timed(
            torch.device("cpu"),
            lambda: _sample_balanced_batch_indices_from_pools(
                current_pools,
                batch_size=batch_size,
                balance_quotas=balance_quotas,
                generator=current_generator,
            ),
        )
        current_device_indices, h2d_seconds = _timed(
            device, lambda: current_indices.to(device=device)
        )
        current_tensors, tensor_seconds = _timed(
            device, lambda: _select_tensors(tensors, current_device_indices)
        )
        current_labels, label_seconds = _timed(
            device, lambda: _recover_labels_like_current(string_labels, current_device_indices)
        )

        (cached_indices, cached_counts), cached_sampling_seconds = _timed(
            torch.device("cpu"),
            lambda: _sample_balanced_batch_indices_from_pools(
                cached_pools,
                batch_size=batch_size,
                balance_quotas=balance_quotas,
                generator=cached_generator,
            ),
        )
        cached_labels, cached_label_seconds = _timed(
            torch.device("cpu"), lambda: _recover_labels(string_labels, cached_indices)
        )
        cached_device_indices, cached_h2d_seconds = _timed(
            device, lambda: cached_indices.to(device=device)
        )
        cached_tensors, cached_tensor_seconds = _timed(
            device, lambda: _select_tensors(tensors, cached_device_indices)
        )

        all_indices_equal &= torch.equal(current_indices, cached_indices)
        all_counts_equal &= current_counts == cached_counts
        all_labels_equal &= current_labels == cached_labels
        all_tensors_equal &= all(
            torch.equal(current, cached)
            for current, cached in zip(current_tensors, cached_tensors, strict=True)
        )
        if iteration < warmup_updates:
            continue
        for stage, duration in (
            ("label_pool_build", pool_seconds),
            ("balanced_sampling", sampling_seconds),
            ("index_h2d", h2d_seconds),
            ("tensor_index_select", tensor_seconds),
            ("python_label_recovery", label_seconds),
        ):
            _add(current_totals, stage, duration)
        for stage, duration in (
            ("balanced_sampling", cached_sampling_seconds),
            ("python_label_recovery", cached_label_seconds),
            ("index_h2d", cached_h2d_seconds),
            ("tensor_index_select", cached_tensor_seconds),
        ):
            _add(cached_totals, stage, duration)

    current_total = sum(current_totals.values())
    cached_repeated_total = sum(
        value for stage, value in cached_totals.items() if stage != "label_pool_build_once"
    )
    cached_total = cached_repeated_total + cached_pool_seconds
    return {
        "probe": "hp7a_distill_learner_staging",
        "training_executed": False,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": _sha256(dataset_path),
            "rows": dataset.num_samples,
        },
        "workload": {
            "device": str(device),
            "batch_size": batch_size,
            "measured_updates": updates,
            "warmup_updates": warmup_updates,
            "seed": seed,
            "balance_key": balance_key,
            "balanced_labels": list(selected_labels),
            "balance_quotas": None if balance_quotas is None else dict(balance_quotas),
        },
        "current": {
            "stages_seconds": current_totals,
            "total_seconds": current_total,
            "seconds_per_update": current_total / updates,
        },
        "cached_candidate": {
            "stages_seconds": cached_totals,
            "repeated_total_seconds": cached_repeated_total,
            "total_seconds": cached_total,
            "seconds_per_update": cached_total / updates,
            "current_over_candidate_ratio": (
                None if cached_total == 0.0 else current_total / cached_total
            ),
        },
        "semantic_differential": {
            "sampled_indices_equal": all_indices_equal,
            "label_counts_equal": all_counts_equal,
            "string_labels_equal": all_labels_equal,
            "tensor_batches_equal": all_tensors_equal,
            "pass": all((all_indices_equal, all_counts_equal, all_labels_equal, all_tensors_equal)),
        },
        "memory": {
            "cuda_peak_allocated_bytes": (
                0 if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
            )
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--outer-iteration", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--warmup-updates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--balance-key", choices=("role", "command_intent", "scenario"), required=True
    )
    parser.add_argument("--balanced-label", action="append", default=[])
    parser.add_argument("--balance-quota", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    dataset_path = _resolve_dataset_path(
        dataset_path=args.dataset,
        run_dir=args.run_dir,
        outer_iteration=args.outer_iteration,
    )
    device = torch.device(args.device)
    report = run_staging_benchmark(
        dataset_path=dataset_path,
        device=device,
        batch_size=args.batch_size,
        updates=args.updates,
        warmup_updates=args.warmup_updates,
        seed=args.seed,
        balance_key=args.balance_key,
        balanced_labels=args.balanced_label,
        balance_quotas=_parse_quotas(args.balance_quota),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    if not report["semantic_differential"]["pass"]:
        raise SystemExit("HP-7a semantic differential failed")


if __name__ == "__main__":
    main()
