"""Preflight and explicitly run paper Figure 3(d) FADA target adaptation."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.distill import (  # noqa: E402
    FADAAdaptationTrainer,
    FADAAdaptedPolicy,
    FADALoRAConfig,
    FADATargetBatch,
    FADATargetSplit,
    assert_fada_active_route_contract,
    assert_fada_adaptation_source_checkpoint,
    fada_adaptation_loss,
    fada_adapter_parameters,
    file_sha256,
    inject_fada_idm_lora,
    load_fada_policy_checkpoint,
    load_fada_target_artifact,
    resolve_fada_fault,
    save_fada_adapted_checkpoint,
    select_fada_target_rows,
    split_fada_target_batch,
)
from unilab.training import (  # noqa: E402
    assert_offpolicy_task_choice_matches_algo,
    get_hydra_runtime_choice,
)

_PAPER_LORA = {"rank": 8, "alpha": 16.0, "dropout": 0.05}


@dataclass(frozen=True)
class FADAAdaptationPreflight:
    source_checkpoint_path: Path
    target_artifact_path: Path
    output_checkpoint_path: Path
    source_checkpoint_sha256: str
    target_artifact_sha256: str
    train_rows: int
    validation_rows: int
    trainable_parameter_count: int
    total_parameter_count: int
    confirm_train: bool
    adapted: FADAAdaptedPolicy
    target_batch: Any
    split: FADATargetSplit
    optimizer: torch.optim.Optimizer


def _root_relative(path_value: Any, *, root_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    return path.resolve() if path.is_absolute() else (root_dir / path).resolve()


def _positive_int(cfg: DictConfig, path: str) -> int:
    value = OmegaConf.select(cfg, path)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{path} must be a positive integer, got {value!r}")
    return int(value)


def _nonnegative_int(cfg: DictConfig, path: str) -> int:
    value = OmegaConf.select(cfg, path)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer, got {value!r}")
    return int(value)


def _assert_identity(cfg: DictConfig) -> None:
    fault = resolve_fada_fault(cfg)
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    if get_hydra_runtime_choice(cfg, "task") != fault.task:
        raise ValueError(f"FADA adaptation requires task={fault.task}")
    if str(OmegaConf.select(cfg, "training.task_name")) != fault.task_name:
        raise ValueError(f"FADA adaptation requires task_name={fault.task_name}")
    if str(OmegaConf.select(cfg, "training.sim_backend")) != fault.backend:
        raise ValueError("FADA adaptation identity requires the MuJoCo target owner")
    for name, expected in _PAPER_LORA.items():
        observed = OmegaConf.select(cfg, f"adaptation.{name}")
        if isinstance(expected, int):
            if isinstance(observed, bool) or observed != expected:
                raise ValueError(f"FADA paper LoRA {name} must be {expected}, got {observed!r}")
        elif float(observed) != expected:
            raise ValueError(f"FADA paper LoRA {name} must be {expected}, got {observed!r}")
    confirm_train = OmegaConf.select(cfg, "adaptation.confirm_train")
    if not isinstance(confirm_train, bool):
        raise ValueError("adaptation.confirm_train must be boolean")
    _positive_int(cfg, "adaptation.batch_size")
    _positive_int(cfg, "adaptation.max_updates")
    _nonnegative_int(cfg, "adaptation.seed")
    learning_rate = float(OmegaConf.select(cfg, "adaptation.learning_rate"))
    weight_decay = float(OmegaConf.select(cfg, "adaptation.weight_decay"))
    validation_fraction = float(OmegaConf.select(cfg, "adaptation.validation_fraction"))
    max_grad_norm = float(OmegaConf.select(cfg, "adaptation.max_grad_norm"))
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("adaptation.learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("adaptation.weight_decay must be finite and non-negative")
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("adaptation.validation_fraction must be in (0, 1)")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
        raise ValueError("adaptation.max_grad_norm must be finite and positive")
    if str(OmegaConf.select(cfg, "adaptation.optimizer")).lower() != "adamw":
        raise ValueError("FADA adaptation optimizer must be adamw")
    if str(OmegaConf.select(cfg, "adaptation.observation_contract")) != ("g1_fada_state_v2"):
        raise ValueError("active FADA adaptation requires observation_contract=g1_fada_state_v2")


def preflight_fada_adaptation(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> FADAAdaptationPreflight:
    """Close identities and construct the LoRA-only optimizer without taking a step."""

    root = Path(root_dir).resolve()
    _assert_identity(cfg)
    source_path = _root_relative(
        OmegaConf.select(cfg, "adaptation.source_checkpoint_path"), root_dir=root
    )
    target_path = _root_relative(
        OmegaConf.select(cfg, "adaptation.target_artifact_path"), root_dir=root
    )
    output_path = _root_relative(
        OmegaConf.select(cfg, "adaptation.output_checkpoint_path"), root_dir=root
    )
    if len({source_path, target_path, output_path}) != 3:
        raise ValueError("FADA adaptation input and output paths must be distinct")
    if not source_path.is_file():
        raise FileNotFoundError(f"FADA source checkpoint not found: {source_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"FADA target artifact not found: {target_path}")
    if output_path.exists():
        raise FileExistsError(f"FADA adapted output already exists: {output_path}")
    if output_path.suffix != ".pt":
        raise ValueError("FADA adapted output must use a .pt suffix")

    source_sha = file_sha256(source_path)
    expected_source = OmegaConf.select(cfg, "adaptation.expected_source_checkpoint_sha256")
    if expected_source is not None and source_sha != str(expected_source):
        raise ValueError(
            f"FADA source checkpoint SHA-256 mismatch: expected={expected_source} observed={source_sha}"
        )
    target_sha = file_sha256(target_path)
    expected_target = OmegaConf.select(cfg, "adaptation.expected_target_artifact_sha256")
    if expected_target is not None and target_sha != str(expected_target):
        raise ValueError(
            f"FADA target artifact SHA-256 mismatch: expected={expected_target} observed={target_sha}"
        )

    loaded_source = assert_fada_adaptation_source_checkpoint(
        load_fada_policy_checkpoint(
            source_path, device=str(OmegaConf.select(cfg, "adaptation.device"))
        )
    )
    assert_fada_active_route_contract(
        observation_contract=loaded_source.policy.config.observation_contract,
        projection=str(OmegaConf.select(cfg, "adaptation.observation_contract")),
    )
    loaded_target = load_fada_target_artifact(target_path, config=loaded_source.policy.config)
    if loaded_target.metadata["policy_checkpoint_sha256"] != source_sha:
        raise ValueError(
            "FADA target artifact policy checkpoint identity does not match the source"
        )
    fault = resolve_fada_fault(cfg)
    if loaded_target.metadata["task"] != fault.task_name:
        raise ValueError("FADA target artifact task identity does not match adaptation")
    if loaded_target.metadata["fault_profile"] != fault.fault_profile:
        raise ValueError("FADA target artifact fault profile does not match adaptation")

    adapted = inject_fada_idm_lora(
        loaded_source.policy,
        FADALoRAConfig(
            rank=int(OmegaConf.select(cfg, "adaptation.rank")),
            alpha=float(OmegaConf.select(cfg, "adaptation.alpha")),
            dropout=float(OmegaConf.select(cfg, "adaptation.dropout")),
        ),
    )
    split = split_fada_target_batch(
        loaded_target.batch,
        validation_fraction=float(OmegaConf.select(cfg, "adaptation.validation_fraction")),
        seed=_nonnegative_int(cfg, "adaptation.seed"),
    )
    optimizer = torch.optim.AdamW(
        fada_adapter_parameters(adapted.policy),
        lr=float(OmegaConf.select(cfg, "adaptation.learning_rate")),
        weight_decay=float(OmegaConf.select(cfg, "adaptation.weight_decay")),
    )
    trainable = sum(
        parameter.numel() for parameter in adapted.policy.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in adapted.policy.parameters())
    return FADAAdaptationPreflight(
        source_checkpoint_path=source_path,
        target_artifact_path=target_path,
        output_checkpoint_path=output_path,
        source_checkpoint_sha256=source_sha,
        target_artifact_sha256=target_sha,
        train_rows=int(split.train_indices.numel()),
        validation_rows=int(split.validation_indices.numel()),
        trainable_parameter_count=int(trainable),
        total_parameter_count=int(total),
        confirm_train=bool(OmegaConf.select(cfg, "adaptation.confirm_train")),
        adapted=adapted,
        target_batch=loaded_target.batch,
        split=split,
        optimizer=optimizer,
    )


def _runtime_config(cfg: DictConfig) -> Mapping[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.select(cfg, "adaptation"), resolve=True)
    if not isinstance(payload, Mapping):
        raise TypeError("resolved adaptation config must be a mapping")
    return cast(Mapping[str, Any], payload)


def train_fada_adaptation(
    preflight: FADAAdaptationPreflight,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Execute the explicitly confirmed real target optimizer loop."""

    trainer = FADAAdaptationTrainer(
        preflight.adapted.policy,
        preflight.optimizer,
        lora_config=preflight.adapted.lora_config,
        max_grad_norm=float(OmegaConf.select(cfg, "adaptation.max_grad_norm")),
    )
    batch_size = _positive_int(cfg, "adaptation.batch_size")
    max_updates = _positive_int(cfg, "adaptation.max_updates")
    generator = torch.Generator(device="cpu").manual_seed(_nonnegative_int(cfg, "adaptation.seed"))
    last_stats = None
    for _ in range(max_updates):
        draws = torch.randint(
            int(preflight.split.train_indices.numel()),
            (batch_size,),
            generator=generator,
        )
        indices = preflight.split.train_indices.index_select(0, draws)
        # Sampling with replacement is legal for optimization; selection itself must remain unique.
        batch = FADATargetBatch(
            **{
                field: getattr(preflight.target_batch, field).index_select(
                    0, indices.to(getattr(preflight.target_batch, field).device)
                )
                for field in FADATargetBatch.__dataclass_fields__
            }
        )
        last_stats = trainer.update(batch)
    validation = select_fada_target_rows(preflight.target_batch, preflight.split.validation_indices)
    preflight.adapted.policy.eval()
    with torch.no_grad():
        validation_loss = float(fada_adaptation_loss(preflight.adapted.policy, validation).detach())
    save_fada_adapted_checkpoint(
        preflight.output_checkpoint_path,
        preflight.adapted.policy,
        preflight.optimizer,
        lora_config=preflight.adapted.lora_config,
        source_checkpoint_sha256=preflight.source_checkpoint_sha256,
        target_artifact_sha256=preflight.target_artifact_sha256,
        completed_steps=max_updates,
        samples_seen=max_updates * batch_size,
        runtime_config=_runtime_config(cfg),
    )
    assert last_stats is not None
    return {
        "status": "completed",
        "output_checkpoint_path": str(preflight.output_checkpoint_path),
        "completed_steps": max_updates,
        "samples_seen": max_updates * batch_size,
        "train_loss": last_stats.loss,
        "validation_loss": validation_loss,
    }


def run_fada_adaptation(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
    train_fn: Callable[[FADAAdaptationPreflight, DictConfig], dict[str, Any]] = (
        train_fada_adaptation
    ),
) -> dict[str, Any]:
    preflight = preflight_fada_adaptation(cfg, root_dir=root_dir)
    if not preflight.confirm_train:
        return {
            "status": "D_TRAIN_READY",
            "source_checkpoint_sha256": preflight.source_checkpoint_sha256,
            "target_artifact_sha256": preflight.target_artifact_sha256,
            "train_rows": preflight.train_rows,
            "validation_rows": preflight.validation_rows,
            "trainable_parameter_count": preflight.trainable_parameter_count,
            "total_parameter_count": preflight.total_parameter_count,
            "output_checkpoint_path": str(preflight.output_checkpoint_path),
            "confirm_train": False,
        }
    return train_fn(preflight, cfg)


@hydra.main(version_base="1.3", config_path="../conf/offpolicy", config_name="fada_adapt")
def main(cfg: DictConfig) -> None:
    print(json.dumps(run_fada_adaptation(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
