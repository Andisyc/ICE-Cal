"""Generic behavior distillation entrypoint assembly.

This module keeps live environment sampling in distill-owned helpers and only
assembles the configured entrypoint routes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill import (
    BehaviorDistillationTrainer,
    DistillationTeacherSpec,
    MLPStudentPolicy,
    MoEStudentPolicy,
    build_multitask_distillation_dataset,
    collect_distillation_dataset_from_env,
    load_distillation_dataset,
    load_distillation_student_policy,
    load_sac_teacher_policy,
    make_fake_distillation_dataset,
    run_offline_distillation_updates,
    save_distillation_dataset,
    validate_sac_teacher_checkpoint_contract,
)
from unilab.training import BackendAdapter, ExperimentTracker, create_env, ensure_registries
from unilab.training.run import resolve_task_checkpoint_path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in values)


def _student_model_type(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "student.model_type", default="mlp"))


def build_teacher_spec(cfg: DictConfig) -> DistillationTeacherSpec:
    """Build the frozen teacher load contract from owner config."""

    return DistillationTeacherSpec(
        obs_dim=int(cfg.teacher.obs_dim),
        action_dim=int(cfg.teacher.action_dim),
        algo_type=str(cfg.teacher.algo_type),
        actor_hidden_dim=int(cfg.teacher.actor_hidden_dim),
        use_layer_norm=bool(cfg.teacher.use_layer_norm),
        obs_normalization=bool(cfg.teacher.obs_normalization),
    )


def build_student_policy(
    cfg: DictConfig,
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build the student actor from owner config."""

    model_type = _student_model_type(cfg)
    if model_type == "mlp":
        student = MLPStudentPolicy(
            obs_dim=int(cfg.student.obs_dim),
            action_dim=int(cfg.student.action_dim),
            hidden_dims=_int_tuple(cfg.student.hidden_dims),
            activation=str(cfg.student.activation),
            squash_action=bool(cfg.student.squash_action),
        )
    elif model_type == "moe":
        student = MoEStudentPolicy(
            obs_dim=int(cfg.student.obs_dim),
            action_dim=int(cfg.student.action_dim),
            num_experts=int(cfg.student.num_experts),
            expert_hidden_dims=_int_tuple(cfg.student.expert_hidden_dims),
            router_hidden_dims=_int_tuple(cfg.student.router_hidden_dims),
            activation=str(cfg.student.activation),
            squash_action=bool(cfg.student.squash_action),
            routing_mode=str(cfg.student.routing_mode),
            router_temperature=float(cfg.student.router_temperature),
        )
    else:
        raise ValueError(f"Unsupported distillation student.model_type: {model_type!r}")
    return cast(torch.nn.Module, student.to(device))


def _student_runtime_cfg(cfg: DictConfig) -> dict[str, Any]:
    model_type = _student_model_type(cfg)
    payload: dict[str, Any] = {
        "student_model_type": model_type,
        "student_obs_dim": int(cfg.student.obs_dim),
        "student_action_dim": int(cfg.student.action_dim),
        "student_activation": str(cfg.student.activation),
        "student_squash_action": bool(cfg.student.squash_action),
    }
    if model_type == "mlp":
        payload["student_hidden_dims"] = [int(dim) for dim in cfg.student.hidden_dims]
    elif model_type == "moe":
        payload.update(
            {
                "student_num_experts": int(cfg.student.num_experts),
                "student_expert_hidden_dims": [
                    int(dim) for dim in cfg.student.expert_hidden_dims
                ],
                "student_router_hidden_dims": [
                    int(dim) for dim in cfg.student.router_hidden_dims
                ],
                "student_routing_mode": str(cfg.student.routing_mode),
                "student_router_temperature": float(cfg.student.router_temperature),
            }
        )
    else:
        raise ValueError(f"Unsupported distillation student.model_type: {model_type!r}")
    return payload


def _teacher_metadata(cfg: DictConfig, teacher_checkpoint: str | Path) -> dict[str, Any]:
    metadata = {
        "algo_family": str(cfg.teacher.algo_family),
        "algo_type": str(cfg.teacher.algo_type),
        "task": str(cfg.teacher.task),
        "task_name": str(cfg.teacher.task_name),
        "checkpoint_path": str(teacher_checkpoint),
    }
    info = validate_sac_teacher_checkpoint_contract(
        teacher_checkpoint,
        build_teacher_spec(cfg),
        device="cpu",
    )
    metadata.update(
        {
            "checkpoint_actor_input_dim": info.actor_input_dim,
            "checkpoint_first_weight_key": info.first_weight_key,
        }
    )
    return metadata


def _distill_runtime_cfg(
    cfg: DictConfig,
    *,
    distill_source: str,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "distill_source": str(distill_source),
        "loss_type": str(cfg.algo.loss_type),
        "learning_rate": float(cfg.algo.learning_rate),
        "aux_loss_coef": float(OmegaConf.select(cfg, "algo.aux_loss_coef", default=0.0)),
        "role_loss_coef": float(OmegaConf.select(cfg, "algo.role_loss_coef", default=0.0)),
        "role_expert_targets": dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "algo.role_expert_targets", default={}),
                resolve=True,
            )
        ),
        "offline_repeat_dataset": bool(
            OmegaConf.select(cfg, "training.offline_repeat_dataset", default=False)
        ),
        "offline_shuffle": bool(OmegaConf.select(cfg, "training.offline_shuffle", default=False)),
        **_student_runtime_cfg(cfg),
        "teacher_obs_dim": int(cfg.teacher.obs_dim),
    }
    if dataset_path is not None:
        payload["dataset_path"] = str(dataset_path)
    return payload


def _probe_result(
    cfg: DictConfig,
    *,
    dataset: Any,
    result: Any,
    distill_source: str,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    probe = {
        "distill_source": str(distill_source),
        "student_model_type": _student_model_type(cfg),
        "student_obs_shape": result.student_obs_shape,
        "teacher_obs_shape": result.teacher_obs_shape,
        "dataset_num_samples": dataset.num_samples,
        "dataset_student_obs_dim": dataset.student_obs_dim,
        "dataset_teacher_obs_dim": dataset.teacher_obs_dim,
        "dataset_metadata": dict(getattr(dataset, "metadata", {})),
        "student_action_shape": result.student_action_shape,
        "teacher_action_shape": result.teacher_action_shape,
        "teacher_action_requires_grad": result.teacher_action_requires_grad,
        "teacher_action_source": result.last_teacher_action_source,
        "student_grad_norm": result.last_student_grad_norm,
        "loss": result.last_loss,
        "behavior_loss": result.last_behavior_loss,
        "aux_loss": result.last_aux_loss,
        "role_loss": result.last_role_loss,
        "role_target_count": result.last_role_target_count,
        "expert_usage": result.last_expert_usage,
        "route_entropy": result.last_route_entropy,
        "update_count": result.update_count,
        "samples_seen": result.samples_seen,
        "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
    }
    if dataset_path is not None:
        probe["dataset_path"] = str(dataset_path)
    return probe


def _normalize_checkpoint_selector(selector: Any) -> str | None:
    if selector in (None, "", -1, "-1"):
        return None
    return str(selector)


def resolve_teacher_checkpoint(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> tuple[Path | None, Path | None]:
    """Resolve the teacher checkpoint through shared training path semantics."""

    explicit_checkpoint_path = OmegaConf.select(cfg, "teacher.checkpoint_path")
    if explicit_checkpoint_path not in (None, ""):
        path = Path(str(explicit_checkpoint_path))
        resolved_path = path if path.is_absolute() else Path(root_dir) / path
        if not resolved_path.is_file():
            raise FileNotFoundError(f"teacher.checkpoint_path does not exist: {resolved_path}")
        return resolved_path, resolved_path.parent

    log_root = OmegaConf.select(
        cfg,
        "teacher.log_root",
        default=OmegaConf.select(cfg, "training.log_root"),
    )
    return resolve_task_checkpoint_path(
        root_dir,
        task_name=str(cfg.teacher.task_name),
        load_run=str(cfg.teacher.load_run),
        algo_log_name=str(cfg.teacher.algo_log_name),
        checkpoint=_normalize_checkpoint_selector(cfg.teacher.checkpoint),
        suffix=".pt",
        log_root=log_root,
    )


def build_distillation_trainer(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> BehaviorDistillationTrainer:
    """Load teacher, build student, and assemble one behavior distillation trainer."""

    validate_sac_teacher_checkpoint_contract(
        teacher_checkpoint,
        build_teacher_spec(cfg),
        device=device,
    )
    teacher = load_sac_teacher_policy(
        teacher_checkpoint,
        build_teacher_spec(cfg),
        device=device,
    )
    student = build_student_policy(cfg, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(cfg.algo.learning_rate))
    return BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        loss_type=str(cfg.algo.loss_type),
        max_grad_norm=float(cfg.algo.max_grad_norm),
        aux_loss_coef=float(OmegaConf.select(cfg, "algo.aux_loss_coef", default=0.0)),
        role_loss_coef=float(OmegaConf.select(cfg, "algo.role_loss_coef", default=0.0)),
        role_expert_targets=dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "algo.role_expert_targets", default={}),
                resolve=True,
            )
        ),
    )


def run_fake_batch_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    batch_size: int = 8,
    max_updates: int = 1,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run a bounded shape-valid offline distillation probe for the entrypoint."""

    torch.manual_seed(int(cfg.algo.seed))
    trainer = build_distillation_trainer(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        device=device,
    )
    dataset = make_fake_distillation_dataset(
        num_samples=int(batch_size) * int(max_updates),
        student_obs_dim=int(cfg.student.obs_dim),
        teacher_obs_dim=int(cfg.teacher.obs_dim),
        seed=int(cfg.algo.seed),
        device=device,
    )
    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=int(batch_size),
        max_updates=int(max_updates),
        checkpoint_path=checkpoint_path,
        teacher_metadata=_teacher_metadata(cfg, teacher_checkpoint),
        distill_runtime_cfg=_distill_runtime_cfg(cfg, distill_source="fake_probe"),
    )

    return _probe_result(cfg, dataset=dataset, result=result, distill_source="fake_probe")


def run_offline_dataset_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    dataset_path: str | Path,
    batch_size: int | None = None,
    max_updates: int | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run bounded offline updates from a saved distillation tensor dataset."""

    torch.manual_seed(int(cfg.algo.seed))
    resolved_batch_size = int(
        batch_size
        if batch_size is not None
        else OmegaConf.select(cfg, "training.offline_batch_size", default=256)
    )
    resolved_max_updates = int(
        max_updates
        if max_updates is not None
        else OmegaConf.select(cfg, "training.offline_max_updates", default=1)
    )
    trainer = build_distillation_trainer(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        device=device,
    )
    dataset = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=int(cfg.student.obs_dim),
        expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
        expected_teacher_action_dim=int(cfg.teacher.action_dim),
        device=device,
    )
    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=resolved_batch_size,
        max_updates=resolved_max_updates,
        checkpoint_path=checkpoint_path,
        teacher_metadata=_teacher_metadata(cfg, teacher_checkpoint),
        distill_runtime_cfg=_distill_runtime_cfg(
            cfg,
            distill_source="offline_dataset",
            dataset_path=dataset_path,
        ),
        repeat_dataset=bool(OmegaConf.select(cfg, "training.offline_repeat_dataset", default=False)),
        shuffle=bool(OmegaConf.select(cfg, "training.offline_shuffle", default=False)),
        seed=int(cfg.algo.seed),
    )
    return _probe_result(
        cfg,
        dataset=dataset,
        result=result,
        distill_source="offline_dataset",
        dataset_path=dataset_path,
    )


def _multitask_sources(cfg: DictConfig) -> list[dict[str, Any]]:
    sources = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.multitask_sources", default=[]),
        resolve=True,
    )
    if not isinstance(sources, list):
        raise ValueError("training.multitask_sources must be a list")
    return [dict(cast(dict[str, Any], source)) for source in sources]


def _optional_int_cfg(cfg: DictConfig, path: str) -> int | None:
    value = OmegaConf.select(cfg, path)
    if value in (None, ""):
        return None
    return int(value)


def run_multitask_dataset_assembly(
    cfg: DictConfig,
    *,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge saved role-specific datasets into one cached-target dataset."""

    resolved_dataset_path = dataset_path or OmegaConf.select(
        cfg,
        "training.multitask_dataset_path",
    )
    if resolved_dataset_path in (None, ""):
        raise ValueError("training.multitask_dataset_path must be set")
    dataset = build_multitask_distillation_dataset(
        _multitask_sources(cfg),
        expected_student_obs_dim=_optional_int_cfg(
            cfg,
            "training.multitask_expected_student_obs_dim",
        ),
        expected_teacher_obs_dim=_optional_int_cfg(
            cfg,
            "training.multitask_expected_teacher_obs_dim",
        ),
        expected_teacher_action_dim=_optional_int_cfg(
            cfg,
            "training.multitask_expected_teacher_action_dim",
        ),
        device=_distill_device(cfg),
    )
    save_distillation_dataset(resolved_dataset_path, dataset)
    return {
        "distill_source": "multitask_adapter",
        "dataset_path": str(resolved_dataset_path),
        "dataset_num_samples": dataset.num_samples,
        "dataset_student_obs_dim": dataset.student_obs_dim,
        "dataset_teacher_obs_dim": dataset.teacher_obs_dim,
        "dataset_teacher_action_dim": dataset.teacher_action_dim,
        "dataset_metadata": dict(dataset.metadata),
        "student_obs_shape": tuple(dataset.student_obs.shape),
        "teacher_obs_shape": tuple(dataset.teacher_obs.shape),
        "teacher_actions_shape": (
            None if dataset.teacher_actions is None else tuple(dataset.teacher_actions.shape)
        ),
        "source_roles": list(dataset.metadata["source_roles"]),
        "source_sample_counts": list(dataset.metadata["source_sample_counts"]),
    }


def _resolve_formal_run_dir(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> Path:
    explicit = OmegaConf.select(cfg, "training.formal_run_dir")
    if explicit not in (None, ""):
        path = Path(str(explicit))
        return path if path.is_absolute() else Path(root_dir) / path

    log_dir = OmegaConf.select(cfg, "training.log_dir")
    if log_dir not in (None, ""):
        path = Path(str(log_dir))
        return path if path.is_absolute() else Path(root_dir) / path

    log_root = OmegaConf.select(cfg, "training.log_root")
    root = Path(str(log_root)) if log_root not in (None, "") else Path(root_dir) / "logs"
    if not root.is_absolute():
        root = Path(root_dir) / root
    run_name = OmegaConf.select(cfg, "training.formal_run_name")
    if run_name in (None, ""):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{OmegaConf.select(cfg, 'training.sim_backend', default='mujoco')}"
    return root / str(cfg.algo.algo_log_name) / str(cfg.training.task_name) / str(run_name)


def _expected_samples_seen_for_offline_run(
    cfg: DictConfig,
    *,
    dataset_path: str | Path,
    batch_size: int,
    max_updates: int,
    device: str | torch.device = "cpu",
) -> int:
    dataset = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=int(cfg.student.obs_dim),
        expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
        expected_teacher_action_dim=int(cfg.teacher.action_dim),
        device=device,
    )
    if bool(OmegaConf.select(cfg, "training.offline_repeat_dataset", default=False)):
        samples_seen = 0
        cursor = 0
        for _ in range(int(max_updates)):
            if cursor >= dataset.num_samples:
                cursor = 0
            end = min(dataset.num_samples, cursor + int(batch_size))
            samples_seen += end - cursor
            cursor = end
        return samples_seen
    return min(int(dataset.num_samples), int(batch_size) * int(max_updates))


def run_formal_offline_dataset_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    dataset_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    batch_size: int | None = None,
    max_updates: int | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run the saved-dataset distillation path with run metadata and checkpoint layout."""

    resolved_dataset_path = dataset_path or OmegaConf.select(cfg, "training.offline_dataset_path")
    if resolved_dataset_path in (None, ""):
        raise ValueError("training.offline_dataset_path must be set for formal distill runs")
    resolved_batch_size = int(
        batch_size
        if batch_size is not None
        else OmegaConf.select(cfg, "training.offline_batch_size", default=256)
    )
    resolved_max_updates = int(
        max_updates
        if max_updates is not None
        else OmegaConf.select(cfg, "training.offline_max_updates", default=1)
    )
    resolved_run_dir = (
        Path(run_dir) if run_dir is not None else _resolve_formal_run_dir(cfg, root_dir=ROOT_DIR)
    )
    samples_seen = _expected_samples_seen_for_offline_run(
        cfg,
        dataset_path=resolved_dataset_path,
        batch_size=resolved_batch_size,
        max_updates=resolved_max_updates,
        device=device,
    )
    checkpoint_path = resolved_run_dir / f"model_{samples_seen}.pt"

    tracker = ExperimentTracker(
        root_dir=ROOT_DIR,
        log_dir=resolved_run_dir,
        algo_name=str(cfg.algo.algo_log_name),
        task_name=str(cfg.training.task_name),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        training_cfg=cfg.training,
        full_cfg=cfg,
        device=str(device),
    )
    tracker.start()
    try:
        probe = run_offline_dataset_update(
            cfg,
            teacher_checkpoint=teacher_checkpoint,
            dataset_path=resolved_dataset_path,
            batch_size=resolved_batch_size,
            max_updates=resolved_max_updates,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        tracker.update_summary(
            {
                "status": "completed",
                "distill_source": "formal_offline_dataset",
                "dataset_path": str(resolved_dataset_path),
                "checkpoint_path": str(checkpoint_path),
                "update_count": int(probe["update_count"]),
                "samples_seen": int(probe["samples_seen"]),
                "loss": float(probe["loss"]),
                "behavior_loss": float(probe["behavior_loss"]),
                "aux_loss": float(probe["aux_loss"]),
                "student_grad_norm": float(probe["student_grad_norm"]),
            }
        )
    finally:
        tracker.finish()

    probe.update(
        {
            "distill_source": "formal_offline_dataset",
            "run_dir": str(resolved_run_dir),
            "run_config_path": str(resolved_run_dir / "run_config.json"),
            "run_summary_path": str(resolved_run_dir / "run_summary.json"),
            "checkpoint_path": str(checkpoint_path),
        }
    )
    return probe


def _collect_action_mode(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "training.collect_action_mode", default="zero"))


def _distill_device(cfg: DictConfig) -> str:
    device = OmegaConf.select(cfg, "training.device", default="cpu")
    return "cpu" if device in (None, "") else str(device)


def _require_teacher_policy_collection_route(cfg: DictConfig) -> None:
    """Keep teacher-target collection scoped to explicit 98-D flat/standing routes."""

    task_name = str(OmegaConf.select(cfg, "training.task_name"))
    teacher_task_name = str(OmegaConf.select(cfg, "teacher.task_name"))
    allowed_tasks = {"G1WalkFlat", "G1StandStill"}
    if task_name not in allowed_tasks:
        raise ValueError("teacher target collection only supports 98-D G1WalkFlat/G1StandStill")
    if teacher_task_name != task_name:
        raise ValueError("teacher target collection requires teacher.task_name to match training.task_name")
    if int(cfg.teacher.obs_dim) != 98 or int(cfg.student.obs_dim) != 98:
        raise ValueError("teacher target collection requires 98-D teacher and student obs")
    if str(OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")) != "obs":
        raise ValueError("teacher target collection requires training.collect_teacher_obs_key=obs")
    if str(OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity")) != "identity":
        raise ValueError("teacher target collection requires identity teacher projection")
    if str(OmegaConf.select(cfg, "training.collect_student_projection", default="identity")) != "identity":
        raise ValueError("teacher target collection requires identity student projection")
    if OmegaConf.select(cfg, "training.collect_student_drop_index") is not None:
        raise ValueError("teacher target collection does not support collect_student_drop_index")
    if OmegaConf.select(cfg, "training.collect_action_seed") is not None:
        raise ValueError("teacher target collection does not use training.collect_action_seed")
    if bool(OmegaConf.select(cfg, "env.commands.observe_height_command", default=False)):
        raise ValueError("teacher target collection must not use height-command observations")


def _resolve_collect_rollout_checkpoint(cfg: DictConfig) -> Path:
    checkpoint_path = OmegaConf.select(cfg, "training.collect_rollout_checkpoint_path")
    if checkpoint_path in (None, ""):
        raise ValueError(
            "training.collect_rollout_checkpoint_path must be set when "
            "training.collect_action_mode=student_policy"
        )
    path = Path(str(checkpoint_path))
    resolved_path = path if path.is_absolute() else ROOT_DIR / path
    if not resolved_path.is_file():
        raise FileNotFoundError(f"training.collect_rollout_checkpoint_path does not exist: {resolved_path}")
    return resolved_path


def run_collect_dataset(
    cfg: DictConfig,
    *,
    dataset_path: str | Path | None = None,
    create_env_fn: Any | None = None,
    env_cfg_override_fn: Any | None = None,
) -> dict[str, Any]:
    """Collect and save a small live-env distillation observation dataset."""

    resolved_dataset_path = dataset_path or OmegaConf.select(cfg, "training.collect_dataset_path")
    if resolved_dataset_path in (None, ""):
        raise ValueError("training.collect_dataset_path must be set for live dataset collection")

    action_mode = _collect_action_mode(cfg)
    teacher_policy = None
    rollout_policy = None
    teacher_policy_checkpoint_path: Path | None = None
    rollout_policy_checkpoint_path: Path | None = None
    if action_mode in {"teacher_policy", "student_policy"}:
        _require_teacher_policy_collection_route(cfg)
        teacher_policy_checkpoint_path, _run_dir = resolve_teacher_checkpoint(cfg, root_dir=ROOT_DIR)
        if teacher_policy_checkpoint_path is None:
            raise FileNotFoundError(
                "No SAC teacher checkpoint resolved for teacher target collection. "
                "Set teacher.load_run/teacher.checkpoint or training.log_root."
            )
        teacher_policy = load_sac_teacher_policy(
            teacher_policy_checkpoint_path,
            build_teacher_spec(cfg),
            device=_distill_device(cfg),
        )
    if action_mode == "student_policy":
        rollout_policy_checkpoint_path = _resolve_collect_rollout_checkpoint(cfg)
        loaded_rollout_policy = load_distillation_student_policy(
            rollout_policy_checkpoint_path,
            device=_distill_device(cfg),
        )
        if int(loaded_rollout_policy.obs_dim) != int(cfg.student.obs_dim):
            raise ValueError(
                "student_policy rollout obs dim mismatch: "
                f"checkpoint={loaded_rollout_policy.obs_dim} cfg.student.obs_dim={int(cfg.student.obs_dim)}"
            )
        if int(loaded_rollout_policy.action_dim) != int(cfg.student.action_dim):
            raise ValueError(
                "student_policy rollout action dim mismatch: "
                f"checkpoint={loaded_rollout_policy.action_dim} "
                f"cfg.student.action_dim={int(cfg.student.action_dim)}"
            )
        rollout_policy = loaded_rollout_policy.policy

    if create_env_fn is None:
        ensure_registries()
        create_env_fn = create_env
    if env_cfg_override_fn is None:
        env_cfg_override_fn = lambda cfg: BackendAdapter(  # noqa: E731
            cfg,
            root_dir=ROOT_DIR,
            algo_name="distill",
        ).build_task_env_cfg_override()

    env = create_env_fn(
        cfg,
        num_envs=int(OmegaConf.select(cfg, "training.collect_num_envs", default=1)),
        env_cfg_override=env_cfg_override_fn(cfg),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        task_name=str(OmegaConf.select(cfg, "training.task_name")),
    )
    try:
        drop_index = OmegaConf.select(cfg, "training.collect_student_drop_index")
        collect_max_env_steps = OmegaConf.select(cfg, "training.collect_max_env_steps")
        metadata = {
            "task_name": str(OmegaConf.select(cfg, "training.task_name")),
            "sim_backend": str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        }
        if teacher_policy_checkpoint_path is not None:
            metadata["teacher_policy_checkpoint_path"] = str(teacher_policy_checkpoint_path)
        if rollout_policy_checkpoint_path is not None:
            metadata["rollout_policy_checkpoint_path"] = str(rollout_policy_checkpoint_path)
        dataset = collect_distillation_dataset_from_env(
            env,
            num_samples=int(OmegaConf.select(cfg, "training.collect_num_samples", default=1024)),
            expected_student_obs_dim=int(cfg.student.obs_dim),
            expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
            teacher_obs_key=str(OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")),
            teacher_projection=str(
                OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity")
            ),
            student_projection=str(
                OmegaConf.select(cfg, "training.collect_student_projection", default="identity")
            ),
            student_drop_index=None if drop_index is None else int(drop_index),
            action_mode=action_mode,
            action_seed=OmegaConf.select(cfg, "training.collect_action_seed"),
            teacher_policy=teacher_policy,
            rollout_policy=rollout_policy,
            command_sample_filter=str(
                OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
            ),
            command_info_key=str(
                OmegaConf.select(cfg, "training.collect_command_info_key", default="commands")
            ),
            command_xy_threshold=float(
                OmegaConf.select(cfg, "training.collect_command_xy_threshold", default=0.05)
            ),
            command_yaw_threshold=float(
                OmegaConf.select(cfg, "training.collect_command_yaw_threshold", default=0.05)
            ),
            max_env_steps=None if collect_max_env_steps is None else int(collect_max_env_steps),
            metadata=metadata,
        )
        save_distillation_dataset(resolved_dataset_path, dataset)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return {
        "distill_source": "live_env_rollout",
        "dataset_path": str(resolved_dataset_path),
        "dataset_num_samples": dataset.num_samples,
        "dataset_student_obs_dim": dataset.student_obs_dim,
        "dataset_teacher_obs_dim": dataset.teacher_obs_dim,
        "dataset_metadata": dict(dataset.metadata),
        "student_obs_shape": tuple(dataset.student_obs.shape),
        "teacher_obs_shape": tuple(dataset.teacher_obs.shape),
        "collect_num_envs": int(OmegaConf.select(cfg, "training.collect_num_envs", default=1)),
        "collect_action_mode": action_mode,
        "collect_action_seed": OmegaConf.select(cfg, "training.collect_action_seed"),
        "collect_action_abs_max": float(dataset.metadata.get("action_abs_max", 0.0)),
        "teacher_policy_checkpoint_path": (
            str(teacher_policy_checkpoint_path) if teacher_policy_checkpoint_path is not None else None
        ),
        "rollout_policy_checkpoint_path": (
            str(rollout_policy_checkpoint_path)
            if rollout_policy_checkpoint_path is not None
            else None
        ),
        "teacher_obs_key": str(OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")),
        "teacher_projection": str(
            OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity")
        ),
        "student_projection": str(
            OmegaConf.select(cfg, "training.collect_student_projection", default="identity")
        ),
        "student_drop_index": None if drop_index is None else int(drop_index),
        "collect_command_sample_filter": str(
            dataset.metadata.get("command_sample_filter", "none")
        ),
        "collect_command_seen_samples": dataset.metadata.get("command_seen_samples"),
        "collect_command_selected_samples": dataset.metadata.get("command_selected_samples"),
    }


@hydra.main(config_path="../conf/distill", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Entrypoint guard for the still-offline behavior distillation path."""

    multitask_dataset_path = OmegaConf.select(cfg, "training.multitask_dataset_path")
    if multitask_dataset_path not in (None, ""):
        print(run_multitask_dataset_assembly(cfg, dataset_path=multitask_dataset_path))
        return

    collect_dataset_path = OmegaConf.select(cfg, "training.collect_dataset_path")
    if collect_dataset_path not in (None, ""):
        print(run_collect_dataset(cfg, dataset_path=collect_dataset_path))
        return

    checkpoint_path, _run_dir = resolve_teacher_checkpoint(cfg, root_dir=ROOT_DIR)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No SAC teacher checkpoint resolved for distillation. "
            "Set teacher.load_run/teacher.checkpoint or training.log_root."
        )

    if bool(OmegaConf.select(cfg, "training.dry_run", default=False)):
        print(
            run_fake_batch_update(
                cfg,
                teacher_checkpoint=checkpoint_path,
                batch_size=int(OmegaConf.select(cfg, "training.dry_run_batch_size", default=8)),
                max_updates=int(OmegaConf.select(cfg, "training.dry_run_updates", default=1)),
                checkpoint_path=OmegaConf.select(cfg, "training.dry_run_checkpoint"),
            )
        )
        return

    offline_dataset_path = OmegaConf.select(cfg, "training.offline_dataset_path")
    if offline_dataset_path not in (None, ""):
        if bool(OmegaConf.select(cfg, "training.formal_run", default=False)):
            print(
                run_formal_offline_dataset_update(
                    cfg,
                    teacher_checkpoint=checkpoint_path,
                    dataset_path=offline_dataset_path,
                    batch_size=int(OmegaConf.select(cfg, "training.offline_batch_size", default=256)),
                    max_updates=int(OmegaConf.select(cfg, "training.offline_max_updates", default=1)),
                    device=_distill_device(cfg),
                )
            )
            return
        print(
            run_offline_dataset_update(
                cfg,
                teacher_checkpoint=checkpoint_path,
                dataset_path=offline_dataset_path,
                batch_size=int(OmegaConf.select(cfg, "training.offline_batch_size", default=256)),
                max_updates=int(OmegaConf.select(cfg, "training.offline_max_updates", default=1)),
                checkpoint_path=OmegaConf.select(cfg, "training.offline_checkpoint"),
            )
        )
        return

    raise NotImplementedError(
        "Live behavior distillation sampling/training loop is not wired in this offline phase. "
        "Use training.collect_dataset_path for live dataset collection, training.dry_run=true "
        "for the fake-batch probe, or set training.offline_dataset_path for saved-dataset "
        "offline updates."
    )


if __name__ == "__main__":
    main()
