"""Owner for preflight and execution of FADA target-only LoRA adaptation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.adaptation import (
    FADAAdaptationTrainer,
    FADAAdaptedPolicy,
    FADALoRAConfig,
    FADATargetSplit,
    fada_adaptation_loss,
    fada_adapter_parameters,
    inject_fada_idm_lora,
    select_fada_target_rows,
    split_fada_target_batch,
)
from unilab.algos.torch.distill.fada.adaptation_checkpoint import (
    assert_fada_adaptation_source_checkpoint,
    save_fada_adapted_checkpoint,
)
from unilab.algos.torch.distill.fada.checkpoint import load_fada_policy_checkpoint
from unilab.algos.torch.distill.fada.observation import assert_fada_active_route_contract
from unilab.algos.torch.distill.fada.target_data import (
    FADATargetBatch,
    load_fada_target_artifact,
)
from unilab.algos.torch.distill.fada.target_domain import (
    assert_nominal_slope_environment,
    resolve_fada_target_domain,
)
from unilab.algos.torch.distill.workflow import file_sha256
from unilab.training import (
    assert_offpolicy_task_choice_matches_algo,
    get_hydra_runtime_choice,
)

ROOT_DIR = Path(__file__).resolve().parents[6]
_PAPER_LORA = {"rank": 8, "alpha": 16.0, "dropout": 0.05}


@dataclass(frozen=True)
class FADAAdaptationPreflight:
    source_checkpoint_path: Path
    target_artifact_path: Path
    output_checkpoint_path: Path
    source_checkpoint_sha256: str
    target_artifact_sha256: str
    target_artifact_schema_version: str
    train_rows: int
    validation_rows: int
    trainable_parameter_count: int
    total_parameter_count: int
    confirm_train: bool
    adapted: FADAAdaptedPolicy
    target_batch: FADATargetBatch
    split: FADATargetSplit
    optimizer: torch.optim.Optimizer


def _root_relative(value: Any, *, root_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root_dir / path).resolve()


def _integer(cfg: DictConfig, path: str, *, positive: bool) -> int:
    value = OmegaConf.select(cfg, path)
    invalid = isinstance(value, bool) or not isinstance(value, Integral)
    invalid = invalid or (value <= 0 if positive else value < 0)
    if invalid:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{path} must be a {qualifier} integer, got {value!r}")
    return int(value)


def _assert_identity(cfg: DictConfig) -> Any:
    domain = resolve_fada_target_domain(cfg)
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    task_choice = get_hydra_runtime_choice(cfg, "task")
    if domain.kind == "slope":
        assert_nominal_slope_environment(cfg, domain, task_choice=task_choice)
    else:
        if task_choice != domain.task:
            raise ValueError(f"FADA adaptation requires task={domain.task}")
        if str(OmegaConf.select(cfg, "training.task_name")) != domain.task_name:
            raise ValueError(f"FADA adaptation requires task_name={domain.task_name}")
        if str(OmegaConf.select(cfg, "training.sim_backend")) != domain.backend:
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
    _integer(cfg, "adaptation.batch_size", positive=True)
    _integer(cfg, "adaptation.max_updates", positive=True)
    _integer(cfg, "adaptation.seed", positive=False)
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
    if str(OmegaConf.select(cfg, "adaptation.observation_contract")) != "g1_fada_state_v2":
        raise ValueError("active FADA adaptation requires observation_contract=g1_fada_state_v2")
    return domain


def preflight_fada_adaptation(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> FADAAdaptationPreflight:
    """Close identities and construct a LoRA-only optimizer without stepping."""

    root = Path(root_dir).resolve()
    domain = _assert_identity(cfg)
    source_path = _root_relative(cfg.adaptation.source_checkpoint_path, root_dir=root)
    target_path = _root_relative(cfg.adaptation.target_artifact_path, root_dir=root)
    output_path = _root_relative(cfg.adaptation.output_checkpoint_path, root_dir=root)
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
    target_sha = file_sha256(target_path)
    for label, observed, selected in (
        ("source checkpoint", source_sha, "adaptation.expected_source_checkpoint_sha256"),
        ("target artifact", target_sha, "adaptation.expected_target_artifact_sha256"),
    ):
        expected = OmegaConf.select(cfg, selected)
        if expected is not None and observed != str(expected):
            raise ValueError(
                f"FADA {label} SHA-256 mismatch: expected={expected} observed={observed}"
            )
    loaded_source = assert_fada_adaptation_source_checkpoint(
        load_fada_policy_checkpoint(source_path, device=str(cfg.adaptation.device))
    )
    assert_fada_active_route_contract(
        observation_contract=loaded_source.policy.config.observation_contract,
        projection=str(cfg.adaptation.observation_contract),
    )
    loaded_target = load_fada_target_artifact(target_path, config=loaded_source.policy.config)
    metadata = loaded_target.metadata
    if metadata["policy_checkpoint_sha256"] != source_sha:
        raise ValueError(
            "FADA target artifact policy checkpoint identity does not match the source"
        )
    if metadata["task"] != domain.task_name:
        raise ValueError("FADA target artifact task identity does not match adaptation")
    if domain.kind == "slope":
        if metadata.get("target_domain_id") != domain.target_domain_id:
            raise ValueError(
                "FADA target artifact target-domain identity does not match adaptation"
            )
    elif metadata.get("fault_profile") != domain.legacy_fault_profile:
        raise ValueError("FADA target artifact fault profile does not match adaptation")

    adapted = inject_fada_idm_lora(
        loaded_source.policy,
        FADALoRAConfig(
            rank=int(cfg.adaptation.rank),
            alpha=float(cfg.adaptation.alpha),
            dropout=float(cfg.adaptation.dropout),
        ),
    )
    split = split_fada_target_batch(
        loaded_target.batch,
        validation_fraction=float(cfg.adaptation.validation_fraction),
        seed=_integer(cfg, "adaptation.seed", positive=False),
    )
    optimizer = torch.optim.AdamW(
        fada_adapter_parameters(adapted.policy),
        lr=float(cfg.adaptation.learning_rate),
        weight_decay=float(cfg.adaptation.weight_decay),
    )
    trainable = sum(p.numel() for p in adapted.policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in adapted.policy.parameters())
    return FADAAdaptationPreflight(
        source_path,
        target_path,
        output_path,
        source_sha,
        target_sha,
        loaded_target.source_schema_version,
        int(split.train_indices.numel()),
        int(split.validation_indices.numel()),
        int(trainable),
        int(total),
        bool(cfg.adaptation.confirm_train),
        adapted,
        loaded_target.batch,
        split,
        optimizer,
    )


def _runtime_config(cfg: DictConfig) -> Mapping[str, Any]:
    adaptation = OmegaConf.to_container(cfg.adaptation, resolve=True)
    if not isinstance(adaptation, Mapping):
        raise TypeError("resolved adaptation config must be a mapping")
    target_cfg = OmegaConf.select(cfg, "target_domain")
    if target_cfg is None:
        return cast(Mapping[str, Any], adaptation)
    target = OmegaConf.to_container(target_cfg, resolve=True)
    return {**cast(Mapping[str, Any], adaptation), "target_domain": target}


def train_fada_adaptation(
    preflight: FADAAdaptationPreflight,
    cfg: DictConfig,
) -> dict[str, Any]:
    trainer = FADAAdaptationTrainer(
        preflight.adapted.policy,
        preflight.optimizer,
        lora_config=preflight.adapted.lora_config,
        max_grad_norm=float(cfg.adaptation.max_grad_norm),
    )
    batch_size = _integer(cfg, "adaptation.batch_size", positive=True)
    max_updates = _integer(cfg, "adaptation.max_updates", positive=True)
    generator = torch.Generator(device="cpu").manual_seed(
        _integer(cfg, "adaptation.seed", positive=False)
    )
    last_stats = None
    for _ in range(max_updates):
        draws = torch.randint(
            int(preflight.split.train_indices.numel()),
            (batch_size,),
            generator=generator,
        )
        indices = preflight.split.train_indices.index_select(0, draws)
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
        target_artifact_schema_version=preflight.target_artifact_schema_version,
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
    train_fn: Callable[
        [FADAAdaptationPreflight, DictConfig], dict[str, Any]
    ] = train_fada_adaptation,
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
