from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.support_query import (
    FADA_CONTEXT_METHOD_CONTRACT_ID,
    FADASupportContextEncoder,
    FrozenIDMSupportQueryPolicy,
    SupportQueryBatch,
    SupportQueryContextConfig,
    context_first_action_loss,
)
from unilab.algos.torch.fada_context.support_query_data import (
    load_support_query_dataset,
    split_support_query_by_rollout,
    support_query_split_identity_sha256,
)

CONTEXT_SUPPORT_QUERY_CHECKPOINT_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class PreparedSupportQueryTraining:
    policy: FrozenIDMSupportQueryPolicy
    optimizer: torch.optim.Optimizer


@dataclass(frozen=True)
class PreparedContextSupportQueryArtifact:
    policy: FrozenIDMSupportQueryPolicy
    dataset: SupportQueryBatch
    metadata: Mapping[str, Any]
    train: SupportQueryBatch
    validation: SupportQueryBatch
    method_contract_id: str
    checkpoint_schema: int
    checkpoint_step: int
    source_checkpoint_sha256: str
    dataset_sha256: str
    train_split_sha256: str
    validation_split_sha256: str
    split_contract: str
    checkpoint_identity_binding: str


@dataclass(frozen=True)
class SupportQueryTrainingLoopConfig:
    steps: int
    batch_size: int
    log_interval: int
    checkpoint_interval: int
    gradient_clip_norm: float
    minimum_zero_context_mse: float

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("training steps must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("training batch_size must be positive")
        if self.log_interval <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("training intervals must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.minimum_zero_context_mse < 0.0:
            raise ValueError("minimum_zero_context_mse must be non-negative")


@dataclass(frozen=True)
class SupportQueryTrainingResult:
    final_checkpoint: Path
    baseline_train_mse: float
    baseline_validation_mse: float
    final_train_mse: float
    final_validation_mse: float
    best_validation_mse: float
    best_step: int


@dataclass(frozen=True)
class SupportQueryPreflightResult:
    method_contract_id: str
    delta_z_shape: tuple[int, int, int]
    zero_context_first_action_mse: float
    minimum_required_mse: float
    context_grad_norm: float
    optimizer_steps: int
    planner_frozen: bool
    idm_frozen: bool


@dataclass(frozen=True)
class ResumedSupportQueryTraining:
    setup: PreparedSupportQueryTraining
    method_contract_id: str
    checkpoint_schema: int
    checkpoint_step: int


@dataclass(frozen=True)
class SupportQueryArtifactAdmissionResult:
    method_contract_id: str
    checkpoint_schema: int
    checkpoint_step: int
    pair_ids: tuple[int, ...]
    support_rollout_ids: tuple[int, ...]
    query_rollout_ids: tuple[int, ...]
    window_count: int
    delta_z_shape: tuple[int, int, int]


TrainingEventEmitter = Callable[..., None]


def require_fresh_support_query_run_paths(
    dataset_artifact: str | Path,
    output_dir: str | Path,
) -> None:
    artifact = Path(dataset_artifact)
    target_dir = Path(output_dir)
    if artifact.exists():
        raise FileExistsError(f"Support-Query dataset artifact already exists: {artifact}")
    if target_dir.exists():
        raise FileExistsError(f"Context training output directory already exists: {target_dir}")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


BorrowedOwnerLifecycleSnapshot = tuple[
    tuple[tuple[torch.nn.Module, bool], ...],
    tuple[tuple[torch.nn.Parameter, bool], ...],
]


def _snapshot_borrowed_owner_lifecycle(
    healthy_policy: FADAPlannerIDMPolicy,
) -> BorrowedOwnerLifecycleSnapshot:
    owners = (healthy_policy.planner, healthy_policy.idm)
    modules = tuple((module, module.training) for owner in owners for module in owner.modules())
    parameters = tuple(
        (parameter, parameter.requires_grad) for owner in owners for parameter in owner.parameters()
    )
    return modules, parameters


def _restore_borrowed_owner_lifecycle(
    snapshot: BorrowedOwnerLifecycleSnapshot,
) -> None:
    modules, parameters = snapshot
    for module, training in modules:
        module.training = training
    for parameter, requires_grad in parameters:
        parameter.requires_grad_(requires_grad)


def prepare_context_support_query_policy(
    healthy_policy: FADAPlannerIDMPolicy,
    context_config: SupportQueryContextConfig,
) -> FrozenIDMSupportQueryPolicy:
    device = next(healthy_policy.parameters()).device
    context = FADASupportContextEncoder(healthy_policy.config, context_config).to(device)
    return FrozenIDMSupportQueryPolicy(
        healthy_policy.planner,
        healthy_policy.idm,
        context,
    )


def prepare_support_query_training(
    healthy_policy: FADAPlannerIDMPolicy,
    context_config: SupportQueryContextConfig,
    *,
    learning_rate: float,
) -> PreparedSupportQueryTraining:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    policy = prepare_context_support_query_policy(healthy_policy, context_config)
    optimizer = torch.optim.Adam(policy.context_encoder.parameters(), lr=learning_rate)
    context_ids = {id(parameter) for parameter in policy.context_encoder.parameters()}
    if _optimizer_parameter_ids(optimizer) != context_ids:
        raise RuntimeError("Context optimizer must own exactly Context Encoder parameters")
    if any(parameter.requires_grad for parameter in policy.planner.parameters()):
        raise RuntimeError("Planner must be frozen")
    if any(parameter.requires_grad for parameter in policy.idm.parameters()):
        raise RuntimeError("IDM must be frozen")
    return PreparedSupportQueryTraining(policy=policy, optimizer=optimizer)


@torch.no_grad()
def evaluate_context_action_mse(
    policy: FrozenIDMSupportQueryPolicy,
    batch: SupportQueryBatch,
) -> float:
    return float(context_first_action_loss(policy, batch).detach())


def _sample_support_query_batch(
    batch: SupportQueryBatch,
    batch_size: int,
) -> SupportQueryBatch:
    indices = torch.randint(
        batch.batch_size,
        (min(batch_size, batch.batch_size),),
        device=batch.pair_id.device,
    )
    return batch.index_select(indices)


def _parameter_snapshot(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().cpu().clone() for parameter in module.parameters())


def _parameters_equal(
    module: torch.nn.Module,
    snapshot: tuple[torch.Tensor, ...],
) -> bool:
    return all(
        torch.equal(parameter.detach().cpu(), original)
        for parameter, original in zip(module.parameters(), snapshot, strict=True)
    )


def _restore_parameters(
    module: torch.nn.Module,
    snapshot: tuple[torch.Tensor, ...],
) -> None:
    with torch.no_grad():
        for parameter, original in zip(module.parameters(), snapshot, strict=True):
            parameter.copy_(original.to(device=parameter.device, dtype=parameter.dtype))


def _require_frozen_owners_unchanged(
    policy: FrozenIDMSupportQueryPolicy,
    planner_snapshot: tuple[torch.Tensor, ...],
    idm_snapshot: tuple[torch.Tensor, ...],
) -> None:
    changed = []
    if not _parameters_equal(policy.planner, planner_snapshot):
        changed.append("Planner")
    if not _parameters_equal(policy.idm, idm_snapshot):
        changed.append("IDM")
    if changed:
        _restore_parameters(policy.planner, planner_snapshot)
        _restore_parameters(policy.idm, idm_snapshot)
        raise RuntimeError(f"{'/'.join(changed)} changed during Context training")


def run_support_query_preflight(
    healthy_policy: FADAPlannerIDMPolicy,
    batch: SupportQueryBatch,
    context_config: SupportQueryContextConfig,
    *,
    learning_rate: float,
    minimum_zero_context_mse: float,
) -> SupportQueryPreflightResult:
    if minimum_zero_context_mse < 0.0:
        raise ValueError("minimum_zero_context_mse must be non-negative")
    setup = prepare_support_query_training(
        healthy_policy,
        context_config,
        learning_rate=learning_rate,
    )
    batch.validate(healthy_policy.config, support_length=context_config.support_length)
    planner_before = _parameter_snapshot(setup.policy.planner)
    idm_before = _parameter_snapshot(setup.policy.idm)
    with torch.inference_mode():
        reconstructed = setup.policy.reconstruct_query(batch)
    expected_delta_shape = (
        batch.batch_size,
        batch.query.window_count,
        healthy_policy.config.hidden_dim,
    )
    if tuple(reconstructed.delta_z.shape) != expected_delta_shape:
        raise ValueError(
            "query-conditioned Context delta shape mismatch: "
            f"expected={expected_delta_shape} observed={tuple(reconstructed.delta_z.shape)}"
        )
    setup.optimizer.zero_grad(set_to_none=True)
    loss = context_first_action_loss(setup.policy, batch)
    zero_context_mse = float(loss.detach())
    if not torch.isfinite(loss):
        raise ValueError("zero-Context Query action loss must be finite")
    if zero_context_mse <= minimum_zero_context_mse:
        raise ValueError(
            "zero-Context Query action loss is too small to establish supervision signal: "
            f"observed={zero_context_mse} threshold={minimum_zero_context_mse}"
        )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in setup.policy.context_encoder.parameters()
        if parameter.grad is not None
    ]
    context_grad_norm = (
        float(torch.sqrt(sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)))
        if gradients
        else 0.0
    )
    if not torch.isfinite(torch.tensor(context_grad_norm)) or context_grad_norm <= 0.0:
        raise ValueError("Context gradient norm must be finite and positive")
    _require_frozen_owners_unchanged(setup.policy, planner_before, idm_before)
    if any(parameter.grad is not None for parameter in setup.policy.planner.parameters()):
        raise RuntimeError("Planner received gradients during Context backward")
    if any(parameter.grad is not None for parameter in setup.policy.idm.parameters()):
        raise RuntimeError("IDM received gradients during Context backward")
    return SupportQueryPreflightResult(
        method_contract_id=FADA_CONTEXT_METHOD_CONTRACT_ID,
        delta_z_shape=expected_delta_shape,
        zero_context_first_action_mse=zero_context_mse,
        minimum_required_mse=float(minimum_zero_context_mse),
        context_grad_norm=context_grad_norm,
        optimizer_steps=0,
        planner_frozen=True,
        idm_frozen=True,
    )


def save_context_support_query_checkpoint(
    path: str | Path,
    setup: PreparedSupportQueryTraining,
    *,
    source_checkpoint_sha256: str,
    dataset_sha256: str,
    train_split_sha256: str,
    validation_split_sha256: str,
    step: int,
    split_seed: int,
    metrics: Mapping[str, float],
    resolved_config: Mapping[str, Any],
) -> Path:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    identities = {
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "train_split_sha256": train_split_sha256,
        "validation_split_sha256": validation_split_sha256,
    }
    if any(not value for value in identities.values()):
        raise ValueError("checkpoint identity digests must be non-empty")
    if any(not torch.isfinite(torch.tensor(float(value))) for value in metrics.values()):
        raise ValueError("checkpoint metrics must be finite")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    torch.save(
        {
            "schema_version": CONTEXT_SUPPORT_QUERY_CHECKPOINT_SCHEMA_VERSION,
            "method_contract_id": FADA_CONTEXT_METHOD_CONTRACT_ID,
            "fada_architecture": asdict(setup.policy.config),
            "context_config": asdict(setup.policy.context_encoder.context_config),
            "history_length": setup.policy.config.history_length,
            "prediction_horizon": setup.policy.config.prediction_horizon,
            "support_length": setup.policy.context_encoder.context_config.support_length,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "dataset_sha256": dataset_sha256,
            "train_split_sha256": train_split_sha256,
            "validation_split_sha256": validation_split_sha256,
            "step": int(step),
            "split_seed": int(split_seed),
            "metrics": {name: float(value) for name, value in metrics.items()},
            "context_state_dict": setup.policy.context_encoder.state_dict(),
            "optimizer_state_dict": setup.optimizer.state_dict(),
            "resolved_config": dict(resolved_config),
        },
        temporary,
    )
    temporary.replace(target)
    return target


def run_support_query_training(
    setup: PreparedSupportQueryTraining,
    train: SupportQueryBatch,
    validation: SupportQueryBatch,
    *,
    output_dir: str | Path,
    source_checkpoint_sha256: str,
    dataset_sha256: str,
    train_split_sha256: str,
    validation_split_sha256: str,
    split_seed: int,
    resolved_config: Mapping[str, Any],
    config: SupportQueryTrainingLoopConfig,
    emit: TrainingEventEmitter | None = None,
) -> SupportQueryTrainingResult:
    target_dir = Path(output_dir).expanduser().resolve()
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Context training output directory does not exist: {target_dir}")
    emit_event = emit if emit is not None else lambda _event, **_payload: None
    planner_before = _parameter_snapshot(setup.policy.planner)
    idm_before = _parameter_snapshot(setup.policy.idm)

    def require_frozen() -> None:
        _require_frozen_owners_unchanged(setup.policy, planner_before, idm_before)

    def save_verified_checkpoint(path: Path, **kwargs: Any) -> Path:
        require_frozen()
        return save_context_support_query_checkpoint(path, setup, **kwargs)

    baseline_train = evaluate_context_action_mse(setup.policy, train)
    require_frozen()
    baseline_validation = evaluate_context_action_mse(setup.policy, validation)
    require_frozen()
    if baseline_train <= config.minimum_zero_context_mse:
        raise ValueError(
            "zero-Context training MSE is too small to establish supervision signal: "
            f"observed={baseline_train} threshold={config.minimum_zero_context_mse}"
        )
    emit_event(
        "training_started",
        train_pairs=train.batch_size,
        validation_pairs=validation.batch_size,
        baseline_train_mse=baseline_train,
        baseline_validation_mse=baseline_validation,
    )
    best_validation = baseline_validation
    best_step = 0
    save_verified_checkpoint(
        target_dir / "best.pt",
        source_checkpoint_sha256=source_checkpoint_sha256,
        dataset_sha256=dataset_sha256,
        train_split_sha256=train_split_sha256,
        validation_split_sha256=validation_split_sha256,
        step=0,
        split_seed=split_seed,
        metrics={
            "baseline_train_mse": baseline_train,
            "baseline_validation_mse": baseline_validation,
            "full_train_mse": baseline_train,
            "validation_mse": baseline_validation,
        },
        resolved_config=resolved_config,
    )

    for step in range(1, config.steps + 1):
        setup.optimizer.zero_grad(set_to_none=True)
        loss = context_first_action_loss(
            setup.policy,
            _sample_support_query_batch(train, config.batch_size),
        )
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite Context loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            setup.policy.context_encoder.parameters(),
            config.gradient_clip_norm,
        )
        if not torch.isfinite(grad_norm):
            raise ValueError(f"non-finite Context gradient at step {step}")
        if float(grad_norm) <= 0.0:
            raise ValueError(f"zero Context gradient at step {step}")
        require_frozen()
        setup.optimizer.step()
        require_frozen()
        latest_train = float(loss.detach())
        if step == 1 or step % config.log_interval == 0:
            latest_validation = evaluate_context_action_mse(setup.policy, validation)
            require_frozen()
            emit_event(
                "training_step",
                step=step,
                train_first_action_mse=latest_train,
                validation_first_action_mse=latest_validation,
                context_grad_norm=float(grad_norm),
            )
            if latest_validation < best_validation:
                best_validation = latest_validation
                best_step = step
                save_verified_checkpoint(
                    target_dir / "best.pt",
                    source_checkpoint_sha256=source_checkpoint_sha256,
                    dataset_sha256=dataset_sha256,
                    train_split_sha256=train_split_sha256,
                    validation_split_sha256=validation_split_sha256,
                    step=step,
                    split_seed=split_seed,
                    metrics={
                        "baseline_train_mse": baseline_train,
                        "baseline_validation_mse": baseline_validation,
                        "minibatch_train_mse": latest_train,
                        "validation_mse": latest_validation,
                    },
                    resolved_config=resolved_config,
                )
        if step % config.checkpoint_interval == 0:
            checkpoint_train = evaluate_context_action_mse(setup.policy, train)
            require_frozen()
            checkpoint_validation = evaluate_context_action_mse(setup.policy, validation)
            require_frozen()
            save_verified_checkpoint(
                target_dir / f"context_{step}.pt",
                source_checkpoint_sha256=source_checkpoint_sha256,
                dataset_sha256=dataset_sha256,
                train_split_sha256=train_split_sha256,
                validation_split_sha256=validation_split_sha256,
                step=step,
                split_seed=split_seed,
                metrics={
                    "baseline_train_mse": baseline_train,
                    "baseline_validation_mse": baseline_validation,
                    "full_train_mse": checkpoint_train,
                    "validation_mse": checkpoint_validation,
                },
                resolved_config=resolved_config,
            )

    require_frozen()
    final_train = evaluate_context_action_mse(setup.policy, train)
    require_frozen()
    final_validation = evaluate_context_action_mse(setup.policy, validation)
    require_frozen()
    final_path = save_verified_checkpoint(
        target_dir / "final.pt",
        source_checkpoint_sha256=source_checkpoint_sha256,
        dataset_sha256=dataset_sha256,
        train_split_sha256=train_split_sha256,
        validation_split_sha256=validation_split_sha256,
        step=config.steps,
        split_seed=split_seed,
        metrics={
            "baseline_train_mse": baseline_train,
            "baseline_validation_mse": baseline_validation,
            "full_train_mse": final_train,
            "validation_mse": final_validation,
            "best_validation_mse": best_validation,
            "best_step": float(best_step),
        },
        resolved_config=resolved_config,
    )
    emit_event("training_completed", checkpoint=str(final_path))
    return SupportQueryTrainingResult(
        final_checkpoint=final_path,
        baseline_train_mse=baseline_train,
        baseline_validation_mse=baseline_validation,
        final_train_mse=final_train,
        final_validation_mse=final_validation,
        best_validation_mse=best_validation,
        best_step=best_step,
    )


def _read_context_support_query_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Context Support-Query checkpoint must be a mapping")
    return payload


def _validate_checkpoint_state_tree(
    value: Any,
    *,
    path: str,
    allow_scalar_leaves: bool,
) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta":
            raise ValueError(f"{path} tensor must be materialized before admission")
        try:
            finite = torch.isfinite(value.detach())
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"{path} contains an unsupported tensor type") from exc
        if not bool(finite.all().detach().to(device="cpu")):
            raise ValueError(f"{path} contains a non-finite tensor")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise ValueError(f"{path} contains an unsupported mapping key")
            _validate_checkpoint_state_tree(
                nested,
                path=f"{path}[{key!r}]",
                allow_scalar_leaves=allow_scalar_leaves,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_checkpoint_state_tree(
                nested,
                path=f"{path}[{index}]",
                allow_scalar_leaves=allow_scalar_leaves,
            )
        return
    if allow_scalar_leaves and (value is None or isinstance(value, (str, bool, int, float))):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite scalar")
        return
    raise ValueError(f"{path} must be a tensor leaf")


def _validate_context_support_query_checkpoint(
    payload: Mapping[str, Any],
    *,
    fada_architecture: Mapping[str, Any],
    context_config: Mapping[str, Any],
    expected_source_checkpoint_sha256: str | None = None,
    expected_dataset_sha256: str | None = None,
    expected_train_split_sha256: str | None = None,
    expected_validation_split_sha256: str | None = None,
    expected_split_seed: int | None = None,
) -> Mapping[str, Any]:
    schema = int(payload.get("schema_version", -1))
    if schema in (1, 2, 3):
        raise ValueError(
            "historical fixed-residual checkpoint schema is incompatible with "
            f"{FADA_CONTEXT_METHOD_CONTRACT_ID}: schema={schema}"
        )
    if schema != CONTEXT_SUPPORT_QUERY_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported Context Support-Query checkpoint schema")
    if payload.get("method_contract_id") != FADA_CONTEXT_METHOD_CONTRACT_ID:
        raise ValueError(
            "Context checkpoint method Contract mismatch: "
            f"expected={FADA_CONTEXT_METHOD_CONTRACT_ID} "
            f"observed={payload.get('method_contract_id')!r}"
        )
    checkpoint_step = payload.get("step")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise ValueError("Context checkpoint step must be a non-negative integer")
    if payload.get("fada_architecture") != dict(fada_architecture):
        raise ValueError("Context checkpoint FADA architecture mismatch")
    if payload.get("context_config") != dict(context_config):
        raise ValueError("Context checkpoint architecture mismatch")
    explicit_dimensions = {
        "history_length": fada_architecture["history_length"],
        "prediction_horizon": fada_architecture["prediction_horizon"],
        "support_length": context_config["support_length"],
    }
    for name, expected in explicit_dimensions.items():
        if payload.get(name) != expected:
            raise ValueError(f"Context checkpoint {name} mismatch")
    expected_identities = {
        "source_checkpoint_sha256": expected_source_checkpoint_sha256,
        "dataset_sha256": expected_dataset_sha256,
        "train_split_sha256": expected_train_split_sha256,
        "validation_split_sha256": expected_validation_split_sha256,
    }
    for name, expected in expected_identities.items():
        if expected is not None and payload.get(name) != expected:
            if name == "source_checkpoint_sha256":
                raise ValueError("Context checkpoint healthy source identity mismatch")
            raise ValueError(f"Context checkpoint {name} mismatch")
    if expected_split_seed is not None and payload.get("split_seed") != int(expected_split_seed):
        raise ValueError("Context checkpoint split seed mismatch")
    context_state = payload.get("context_state_dict")
    if not isinstance(context_state, Mapping):
        raise ValueError("Context checkpoint is missing context_state_dict")
    if any(not isinstance(key, str) for key in context_state):
        raise ValueError("Context checkpoint context_state_dict keys must be strings")
    _validate_checkpoint_state_tree(
        context_state,
        path="Context checkpoint context_state_dict",
        allow_scalar_leaves=False,
    )
    return payload


def resume_context_support_query_training(
    healthy_policy: FADAPlannerIDMPolicy,
    context_config: SupportQueryContextConfig,
    checkpoint_path: str | Path,
    *,
    learning_rate: float,
    expected_source_checkpoint_sha256: str,
    expected_dataset_sha256: str,
    expected_train_split_sha256: str,
    expected_validation_split_sha256: str,
    expected_split_seed: int | None = None,
    map_location: str | torch.device = "cpu",
) -> ResumedSupportQueryTraining:
    payload = _read_context_support_query_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    _validate_context_support_query_checkpoint(
        payload,
        fada_architecture=asdict(healthy_policy.config),
        context_config=asdict(context_config),
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        expected_dataset_sha256=expected_dataset_sha256,
        expected_train_split_sha256=expected_train_split_sha256,
        expected_validation_split_sha256=expected_validation_split_sha256,
        expected_split_seed=expected_split_seed,
    )
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("Context checkpoint is missing optimizer_state_dict")
    _validate_checkpoint_state_tree(
        optimizer_state,
        path="Context checkpoint optimizer_state_dict",
        allow_scalar_leaves=True,
    )
    lifecycle = _snapshot_borrowed_owner_lifecycle(healthy_policy)
    try:
        setup = prepare_support_query_training(
            healthy_policy,
            context_config,
            learning_rate=learning_rate,
        )
        setup.policy.context_encoder.load_state_dict(payload["context_state_dict"], strict=True)
        setup.optimizer.load_state_dict(dict(optimizer_state))
    except Exception:
        _restore_borrowed_owner_lifecycle(lifecycle)
        raise
    return ResumedSupportQueryTraining(
        setup=setup,
        method_contract_id=FADA_CONTEXT_METHOD_CONTRACT_ID,
        checkpoint_schema=CONTEXT_SUPPORT_QUERY_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_step=int(payload["step"]),
    )


def prepare_context_support_query_artifact(
    healthy_policy: FADAPlannerIDMPolicy,
    context_config: SupportQueryContextConfig,
    *,
    source_checkpoint_path: str | Path,
    dataset_path: str | Path,
    context_checkpoint_path: str | Path,
    support_length: int,
    query_length: int,
    validation_fraction: float,
    split_seed: int,
    map_location: str | torch.device = "cpu",
) -> PreparedContextSupportQueryArtifact:
    checkpoint_payload = _read_context_support_query_checkpoint(
        context_checkpoint_path,
        map_location=map_location,
    )
    _validate_context_support_query_checkpoint(
        checkpoint_payload,
        fada_architecture=asdict(healthy_policy.config),
        context_config=asdict(context_config),
    )
    source_sha256 = _sha256_file(source_checkpoint_path)
    dataset_sha256 = _sha256_file(dataset_path)
    dataset, metadata = load_support_query_dataset(
        dataset_path,
        healthy_policy.config,
        support_length=support_length,
        query_length=query_length,
        map_location=map_location,
    )
    if metadata.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError("Support dataset healthy checkpoint identity mismatch")
    train, validation = split_support_query_by_rollout(
        dataset,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )

    train_split_sha256 = support_query_split_identity_sha256(train)
    validation_split_sha256 = support_query_split_identity_sha256(validation)
    _validate_context_support_query_checkpoint(
        checkpoint_payload,
        fada_architecture=asdict(healthy_policy.config),
        context_config=asdict(context_config),
        expected_source_checkpoint_sha256=source_sha256,
        expected_dataset_sha256=dataset_sha256,
        expected_train_split_sha256=train_split_sha256,
        expected_validation_split_sha256=validation_split_sha256,
        expected_split_seed=split_seed,
    )
    lifecycle = _snapshot_borrowed_owner_lifecycle(healthy_policy)
    try:
        policy = prepare_context_support_query_policy(healthy_policy, context_config)
        policy.context_encoder.load_state_dict(
            checkpoint_payload["context_state_dict"], strict=True
        )
        policy.eval()
    except Exception:
        _restore_borrowed_owner_lifecycle(lifecycle)
        raise
    return PreparedContextSupportQueryArtifact(
        policy=policy,
        dataset=dataset,
        metadata=metadata,
        train=train,
        validation=validation,
        method_contract_id=FADA_CONTEXT_METHOD_CONTRACT_ID,
        checkpoint_schema=CONTEXT_SUPPORT_QUERY_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_step=int(checkpoint_payload["step"]),
        source_checkpoint_sha256=source_sha256,
        dataset_sha256=dataset_sha256,
        train_split_sha256=train_split_sha256,
        validation_split_sha256=validation_split_sha256,
        split_contract="rollout_group_split",
        checkpoint_identity_binding="v006_healthy_dataset_and_splits",
    )


def preflight_context_support_query_artifact(
    healthy_policy: FADAPlannerIDMPolicy,
    context_config: SupportQueryContextConfig,
    *,
    source_checkpoint_path: str | Path,
    dataset_path: str | Path,
    context_checkpoint_path: str | Path,
    support_length: int,
    query_length: int,
    validation_fraction: float,
    split_seed: int,
    map_location: str | torch.device = "cpu",
) -> SupportQueryArtifactAdmissionResult:
    prepared = prepare_context_support_query_artifact(
        healthy_policy,
        context_config,
        source_checkpoint_path=source_checkpoint_path,
        dataset_path=dataset_path,
        context_checkpoint_path=context_checkpoint_path,
        support_length=support_length,
        query_length=query_length,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        map_location=map_location,
    )
    with torch.inference_mode():
        reconstructed = prepared.policy.reconstruct_query(prepared.dataset)
    expected_delta_shape = (
        prepared.dataset.batch_size,
        prepared.dataset.query.window_count,
        prepared.policy.config.hidden_dim,
    )
    if tuple(reconstructed.delta_z.shape) != expected_delta_shape:
        raise ValueError(
            "admitted query-conditioned Context delta shape mismatch: "
            f"expected={expected_delta_shape} observed={tuple(reconstructed.delta_z.shape)}"
        )
    if not bool(torch.isfinite(reconstructed.delta_z).all()):
        raise ValueError("artifact preflight reconstructed delta_z contains non-finite values")
    return SupportQueryArtifactAdmissionResult(
        method_contract_id=prepared.method_contract_id,
        checkpoint_schema=prepared.checkpoint_schema,
        checkpoint_step=prepared.checkpoint_step,
        pair_ids=tuple(int(value) for value in prepared.dataset.pair_id.tolist()),
        support_rollout_ids=tuple(
            int(value) for value in prepared.dataset.support_rollout_id.tolist()
        ),
        query_rollout_ids=tuple(int(value) for value in prepared.dataset.query_rollout_id.tolist()),
        window_count=prepared.dataset.query.window_count,
        delta_z_shape=expected_delta_shape,
    )
