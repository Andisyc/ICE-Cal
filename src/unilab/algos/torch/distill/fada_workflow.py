"""FADA Planner-IDM training use-case and artifact contracts.

This owner module contains long-lived FADA validation, replay, learner-loop, and
checkpoint lifecycle rules. CLI scripts provide dependencies and dispatch only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, cast

import torch
from omegaconf import DictConfig, OmegaConf

from .async_runtime import DaggerCollectRequest
from .fada import (
    FADA_SCENARIO_IDS,
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
    FADASourceBatch,
)
from .fada_async_runtime import (
    FADA_ASYNC_SCENARIO,
    allocate_fada_command_scenarios,
)
from .fada_collector import FADACollectionSpec, collect_fada_source_windows
from .fada_training import (
    FADAPaperSourcePlan,
    FADAReplayBuffer,
    FADATrainer,
    build_fada_paper_source_plan,
    evaluate_fada_source_batch,
    load_fada_checkpoint,
    load_fada_policy_checkpoint,
    load_fada_source_batch,
    save_fada_checkpoint,
)
from .teacher import load_sac_teacher_policy as _default_load_sac_teacher_policy

ROOT_DIR = Path(__file__).resolve().parents[5]


@dataclass(frozen=True)
class FADAWorkflowDependencies:
    """Composition-root dependencies used by the FADA training owner."""

    require_teacher_policy_collection_route: Callable[[DictConfig], None]
    apply_collect_command_distribution_overrides: Callable[[DictConfig], Mapping[str, Any]]
    resolve_teacher_checkpoint: Callable[..., tuple[Path | None, Path | None]]
    build_teacher_spec: Callable[[DictConfig], Any]
    build_persistent_fada_runtime: Callable[..., Any]
    ensure_registries: Callable[[], None]
    create_env: Callable[..., Any]
    backend_adapter_cls: Callable[..., Any]
    load_sac_teacher_policy: Callable[..., torch.nn.Module] = _default_load_sac_teacher_policy


def _distill_device(cfg: DictConfig) -> str:
    device = OmegaConf.select(cfg, "training.device", default="cpu")
    return "cpu" if device in (None, "") else str(device)


def build_fada_architecture_config(cfg: DictConfig) -> FADAArchitectureConfig:
    """Translate the single Hydra FADA owner into the paper architecture contract."""

    fada = cfg.training.fada
    return FADAArchitectureConfig(
        obs_dim=int(cfg.student.obs_dim),
        action_dim=int(cfg.student.action_dim),
        command_dim=int(fada.command_dim),
        history_length=int(fada.history_length),
        prediction_horizon=int(fada.prediction_horizon),
        hidden_dim=int(fada.hidden_dim),
        num_heads=int(fada.num_heads),
        planner_layers=int(fada.planner_layers),
        idm_encoder_layers=int(fada.idm_encoder_layers),
        idm_decoder_layers=int(fada.idm_decoder_layers),
        feedforward_dim=int(fada.feedforward_dim),
        dropout=float(fada.dropout),
    )


def _fada_execution_mode(cfg: DictConfig) -> str:
    fada_cfg = cfg.training.fada
    execution_mode = str(OmegaConf.select(fada_cfg, "execution_mode", default="legacy"))
    if execution_mode not in {"legacy", "persistent_async"}:
        raise ValueError(
            "training.fada.execution_mode must be 'legacy' or 'persistent_async', "
            f"got {execution_mode!r}"
        )
    curriculum_enabled = bool(
        OmegaConf.select(
            fada_cfg,
            "stand_transition_curriculum.enabled",
            default=False,
        )
    )
    if curriculum_enabled and execution_mode != "persistent_async":
        raise ValueError(
            "training.fada.stand_transition_curriculum requires "
            "training.fada.execution_mode=persistent_async"
        )
    v005_enabled = bool(OmegaConf.select(fada_cfg, "v005_replay.enabled", default=False))
    if v005_enabled and not curriculum_enabled:
        raise ValueError("training.fada.v005_replay requires stand_transition_curriculum.enabled")
    if v005_enabled and execution_mode != "persistent_async":
        raise ValueError(
            "training.fada.v005_replay requires training.fada.execution_mode=persistent_async"
        )
    return execution_mode


def _fada_path(value: Any, *, field_name: str, required: bool) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"{field_name} must be set when training.fada.enabled=true")
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT_DIR / path


def _paper_source_plan(cfg: DictConfig) -> FADAPaperSourcePlan:
    """Resolve Hydra paths and delegate Appendix B.2 rules to the FADA owner."""

    fada = cfg.training.fada
    enabled = bool(OmegaConf.select(fada, "paper_source_enabled", default=False))
    if not enabled:
        return FADAPaperSourcePlan(enabled=False, source_allocations=())
    # B1: composition root resolves repository-relative paths into one explicit source namespace.
    raw_value = OmegaConf.select(fada, "intermediate_oracle_checkpoint_paths", default=[])
    raw = (
        OmegaConf.to_container(raw_value, resolve=True)
        if OmegaConf.is_config(raw_value)
        else raw_value
    )
    if not isinstance(raw, list):
        raise ValueError("intermediate_oracle_checkpoint_paths must be a list")
    paths = tuple(
        path
        for value in raw
        if (path := _fada_path(value, field_name="intermediate oracle checkpoint", required=True))
        is not None
    )
    # B2: FADA owner validates paper constants and returns the sealed allocation.
    return build_fada_paper_source_plan(
        enabled=enabled,
        oracle_shadow_enabled=bool(OmegaConf.select(fada, "oracle_shadow_enabled", default=False)),
        checkpoint_paths=paths,
        configured_checkpoint_count=int(
            OmegaConf.select(fada, "intermediate_oracle_count", default=20)
        ),
        suboptimal_data_ratio=float(OmegaConf.select(fada, "suboptimal_data_ratio", default=0.0)),
        optimal_windows=int(OmegaConf.select(fada, "windows_per_iteration")),
        resume_path=OmegaConf.select(fada, "resume_path"),
    )


def _slice_fada_batch(batch: FADASourceBatch, limit: int) -> FADASourceBatch:
    size = min(int(limit), int(batch.command.shape[0]))
    if size <= 0:
        raise ValueError(f"quality_eval_max_windows must be positive, got {limit}")
    return FADASourceBatch(
        **{field: getattr(batch, field)[:size] for field in FADASourceBatch.__dataclass_fields__}
    )


def _fada_v005_replay_settings(
    fada_cfg: DictConfig,
    *,
    batch_size: int,
) -> tuple[bool, dict[str, float], float]:
    enabled = bool(OmegaConf.select(fada_cfg, "v005_replay.enabled", default=False))
    ratios_cfg = OmegaConf.select(fada_cfg, "v005_replay.planner_scenario_ratios")
    ratios_value = (
        OmegaConf.to_container(ratios_cfg, resolve=True)
        if OmegaConf.is_config(ratios_cfg)
        else ratios_cfg
    )
    if ratios_value is not None and not isinstance(ratios_value, Mapping):
        raise ValueError("v005 planner_scenario_ratios must be a mapping")
    ratio_mapping = cast(
        Mapping[str, Any],
        ratios_value or {"walk": 0.5, "static_stand": 0.25, "walk_to_stand": 0.25},
    )
    ratios = {str(name): float(value) for name, value in ratio_mapping.items()}
    cold_ratio = float(
        OmegaConf.select(fada_cfg, "v005_replay.static_cold_start_ratio", default=0.5)
    )
    if not enabled:
        return False, ratios, cold_ratio
    expected_ratios = {"walk": 0.5, "static_stand": 0.25, "walk_to_stand": 0.25}
    if set(ratios) != set(expected_ratios) or any(
        not math.isclose(ratios[name], expected, rel_tol=0.0, abs_tol=1.0e-12)
        for name, expected in expected_ratios.items()
    ):
        raise ValueError(
            "v005 Planner scenario ratios are fixed at "
            "walk/static_stand/walk_to_stand=0.5/0.25/0.25"
        )
    if not math.isclose(cold_ratio, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("v005 static_cold_start_ratio is fixed at 0.5")
    allocations = dict(allocate_fada_command_scenarios(int(batch_size), ratios))
    static_count = int(allocations.get("static_stand", 0))
    if static_count < 2:
        raise ValueError("v005 Planner batch must allocate static cold-start and steady rows")
    return True, ratios, cold_ratio


def _fada_quality_batch(
    batch: FADASourceBatch,
    *,
    config: FADAArchitectureConfig,
    limit: int,
    scenario_ratios: Mapping[str, float],
    static_cold_start_ratio: float,
) -> FADASourceBatch:
    size = min(int(limit), int(batch.command.shape[0]))
    if size <= 0:
        raise ValueError(f"quality_eval_max_windows must be positive, got {limit}")
    replay = FADAReplayBuffer(config, capacity=int(batch.command.shape[0]))
    replay.add(batch)
    return replay.sample_planner(
        size,
        scenario_ratios=scenario_ratios,
        static_cold_start_ratio=static_cold_start_ratio,
        generator=torch.Generator().manual_seed(0),
    )


def _require_fada_curriculum_artifact(
    cfg: DictConfig,
    metadata: Mapping[str, Any],
    batch: FADASourceBatch | None = None,
) -> None:
    """在 replay mutation 前验证 scenario 配额与 Oracle role artifact contract."""

    # B1: 由当前 owner config 重算 expected allocations, 产出独立校验基准.
    curriculum = OmegaConf.select(cfg, "training.fada.stand_transition_curriculum")
    if curriculum is None or not bool(OmegaConf.select(curriculum, "enabled", default=False)):
        return
    expected = dict(
        allocate_fada_command_scenarios(
            int(OmegaConf.select(cfg, "training.fada.windows_per_iteration")),
            {
                "walk": float(OmegaConf.select(curriculum, "walk_ratio")),
                "static_stand": float(OmegaConf.select(curriculum, "static_stand_ratio")),
                "walk_to_stand": float(OmegaConf.select(curriculum, "walk_to_stand_ratio")),
            },
        )
    )
    if metadata.get("stand_transition_curriculum_enabled") is not True:
        raise ValueError("FADA async artifact omitted enabled standing curriculum identity")
    if dict(metadata.get("scenario_allocations") or {}) != expected:
        raise ValueError(
            "FADA async artifact scenario allocation mismatch: "
            f"expected={expected} observed={metadata.get('scenario_allocations')}"
        )
    # B2: 聚合 main-source summaries, 产出 observed scenario counts 与 role mapping.
    summaries = metadata.get("collections")
    if not isinstance(summaries, list):
        raise ValueError("FADA async artifact collections must be a list")
    main = [item for item in summaries if item.get("source") == "optimal_or_current_policy"]
    observed = {
        scenario: sum(
            int(item.get("windows", 0)) for item in main if item.get("command_scenario") == scenario
        )
        for scenario in expected
    }
    if observed != expected:
        raise ValueError(
            f"FADA async artifact scenario summary mismatch: expected={expected} observed={observed}"
        )
    for item in main:
        scenario = str(item.get("command_scenario"))
        expected_role = "walking" if scenario == "walk" else "standing"
        if item.get("oracle_role") != expected_role:
            raise ValueError(
                "FADA async artifact Oracle role mismatch: "
                f"scenario={scenario!r} expected={expected_role!r} "
                f"observed={item.get('oracle_role')!r}"
            )
    # B3: 拒绝 standing authority 漂移和 intermediate Oracle 越权, 再允许 replay consumer.
    if any(
        item.get("command_scenario") != "walk" or item.get("oracle_role") != "walking"
        for item in summaries
        if item.get("source") == "intermediate_oracle"
    ):
        raise ValueError("intermediate Oracle artifacts must remain walking-source only")
    v005_enabled, _planner_ratios, cold_ratio = _fada_v005_replay_settings(
        cfg.training.fada,
        batch_size=int(OmegaConf.select(cfg, "training.fada.batch_size", default=512)),
    )
    if not v005_enabled:
        return
    if metadata.get("v005_replay_enabled") is not True:
        raise ValueError("FADA async artifact omitted enabled v005 replay identity")
    if batch is None:
        raise ValueError("v005 FADA artifact validation requires row-level source identity")
    batch.validate(build_fada_architecture_config(cfg))
    main_windows = int(metadata.get("main_windows", 0))
    if main_windows <= 0 or main_windows > int(batch.command.shape[0]):
        raise ValueError(f"invalid v005 FADA main_windows={main_windows}")
    main_mask = torch.arange(batch.command.shape[0]) < main_windows
    if not bool(batch.planner_eligible[main_mask].all()):
        raise ValueError("v005 main-source rows must remain Planner eligible")
    if bool(batch.planner_eligible[~main_mask].any()):
        raise ValueError("v005 intermediate-Oracle rows must be excluded from Planner replay")
    observed_rows = {
        scenario: int((batch.command_scenario[main_mask] == scenario_id).sum())
        for scenario, scenario_id in FADA_SCENARIO_IDS.items()
        if scenario in expected
    }
    if observed_rows != expected:
        raise ValueError(
            f"v005 row scenario counts mismatch: expected={expected} observed={observed_rows}"
        )
    static_mask = main_mask & (batch.command_scenario == FADA_SCENARIO_IDS["static_stand"])
    expected_cold = int(math.floor(expected["static_stand"] * cold_ratio + 0.5))
    observed_cold = int((static_mask & batch.cold_start).sum())
    if observed_cold != expected_cold:
        raise ValueError(
            "v005 static cold-start count mismatch: "
            f"expected={expected_cold} observed={observed_cold}"
        )


def _run_fada_persistent_async(
    cfg: DictConfig,
    *,
    config: FADAArchitectureConfig,
    paper_source_plan: FADAPaperSourcePlan,
    resolved_teacher: Path,
    checkpoint_path: Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer,
    replay: FADAReplayBuffer,
    start_iteration: int,
    samples_seen: int,
    dependencies: FADAWorkflowDependencies,
) -> dict[str, Any]:
    """Run the learner in the parent and all FADA rollout work in one resident child."""

    fada_cfg = cfg.training.fada
    iterations = int(fada_cfg.iterations)
    v005_enabled, planner_ratios, cold_start_ratio = _fada_v005_replay_settings(
        fada_cfg,
        batch_size=int(fada_cfg.batch_size),
    )
    runtime_config = cast(dict[str, Any], OmegaConf.to_container(fada_cfg, resolve=True))
    artifact_dir_value = OmegaConf.select(
        fada_cfg, "async_artifact_dir", default="logs/fada/source_batches"
    )
    artifact_dir = _fada_path(
        artifact_dir_value,
        field_name="training.fada.async_artifact_dir",
        required=True,
    )
    if artifact_dir is None:
        raise RuntimeError("FADA async artifact directory was not materialized")

    # The first publication is a complete paired checkpoint even before update 0.
    save_fada_checkpoint(
        checkpoint_path,
        policy,
        trainer,
        completed_iterations=start_iteration,
        samples_seen=samples_seen,
        runtime_config=runtime_config,
    )
    runtime = dependencies.build_persistent_fada_runtime(
        cfg=cfg,
        architecture=config,
        paper_source_plan=paper_source_plan,
        final_teacher_checkpoint=resolved_teacher,
        request_timeout_seconds=float(
            OmegaConf.select(fada_cfg, "async_request_timeout_seconds", default=3600.0)
        ),
    )
    last_stats = None
    last_quality_metrics: dict[str, float] = {}
    collection_summaries: list[dict[str, Any]] = []
    try:
        weight_version = runtime.activate_checkpoint(checkpoint_path)
        for iteration in range(start_iteration, iterations):
            artifact_path = (artifact_dir / f"iteration_{iteration:04d}.pt").resolve()
            request = DaggerCollectRequest(
                request_id=f"fada-{iteration:04d}-v{weight_version}",
                scenario=FADA_ASYNC_SCENARIO,
                iteration=iteration,
                checkpoint_path=str(checkpoint_path.resolve()),
                output_path=str(artifact_path),
                expected_weight_version=weight_version,
            )
            result = runtime.collect(request)
            loaded = load_fada_source_batch(artifact_path, config=config)
            if int(loaded.batch.command.shape[0]) != result.num_samples:
                raise ValueError(
                    "FADA async artifact/result sample mismatch: "
                    f"artifact={loaded.batch.command.shape[0]} result={result.num_samples}"
                )
            main_windows = int(loaded.metadata.get("main_windows", 0))
            if main_windows <= 0 or main_windows > result.num_samples:
                raise ValueError(f"invalid FADA async main_windows={main_windows}")
            summaries = loaded.metadata.get("collections")
            if not isinstance(summaries, list):
                raise ValueError("FADA async artifact collections must be a list")
            _require_fada_curriculum_artifact(cfg, loaded.metadata, loaded.batch)
            collection_summaries.extend(cast(list[dict[str, Any]], summaries))
            replay.add(loaded.batch)
            samples_seen += result.num_samples

            last_stats = trainer.update_from_replay(
                replay,
                batch_size=int(fada_cfg.batch_size),
                idm_updates=int(fada_cfg.idm_updates),
                planner_updates=int(fada_cfg.planner_updates),
                device=_distill_device(cfg),
                planner_scenario_ratios=planner_ratios if v005_enabled else None,
                planner_static_cold_start_ratio=cold_start_ratio,
            )
            if bool(fada_cfg.oracle_shadow_enabled):
                main_batch = _slice_fada_batch(loaded.batch, main_windows)
                quality_limit = int(
                    OmegaConf.select(fada_cfg, "quality_eval_max_windows", default=4096)
                )
                quality_batch = (
                    _fada_quality_batch(
                        main_batch,
                        config=config,
                        limit=quality_limit,
                        scenario_ratios=planner_ratios,
                        static_cold_start_ratio=cold_start_ratio,
                    )
                    if v005_enabled
                    else _slice_fada_batch(main_batch, quality_limit)
                )
                last_quality_metrics = evaluate_fada_source_batch(
                    policy,
                    quality_batch,
                    require_scenario_metrics=v005_enabled,
                )
                last_quality_metrics.update(
                    {
                        "rollout_rejected_done_transitions": float(
                            sum(int(item["rejected_done_transitions"]) for item in summaries)
                        ),
                        "rollout_rejected_command_windows": float(
                            sum(int(item["rejected_command_windows"]) for item in summaries)
                        ),
                    }
                )
            save_fada_checkpoint(
                checkpoint_path,
                policy,
                trainer,
                completed_iterations=iteration + 1,
                samples_seen=samples_seen,
                runtime_config=runtime_config,
                quality_metrics=last_quality_metrics,
            )
            if iteration + 1 < iterations:
                weight_version = runtime.activate_checkpoint(checkpoint_path)
    finally:
        runtime.close()

    return {
        "mode": "fada_planner_idm_training",
        "execution_mode": "persistent_async",
        "checkpoint_path": str(checkpoint_path),
        "completed_iterations": iterations,
        "samples_seen": samples_seen,
        "replay_size": len(replay),
        "last_idm_loss": None if last_stats is None else last_stats.idm_loss,
        "last_planner_loss": None if last_stats is None else last_stats.planner_loss,
        "quality_metrics": last_quality_metrics,
        "collections": collection_summaries,
    }


def run_fada_training_owner(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path | None = None,
    create_env_fn: Any | None = None,
    env_cfg_override_fn: Any | None = None,
    dependencies: FADAWorkflowDependencies,
) -> dict[str, Any]:
    """Assemble the UniLab Oracle/Planner-IDM DAgger training route."""

    # B1: 由唯一 FADA flag 配置族构造 model, Oracle, optimizer 与 resume cursor.
    if not bool(OmegaConf.select(cfg, "training.fada.enabled", default=False)):
        raise ValueError("run_fada_training requires training.fada.enabled=true")
    dependencies.require_teacher_policy_collection_route(cfg)
    dependencies.apply_collect_command_distribution_overrides(cfg)
    paper_source_plan = _paper_source_plan(cfg)
    device = _distill_device(cfg)
    config = build_fada_architecture_config(cfg)
    fada_cfg = cfg.training.fada
    execution_mode = _fada_execution_mode(cfg)
    paper_source_enabled = paper_source_plan.enabled
    collect_oracle_shadow = bool(OmegaConf.select(fada_cfg, "oracle_shadow_enabled", default=False))
    iterations = int(fada_cfg.iterations)
    windows_per_iteration = int(fada_cfg.windows_per_iteration)
    batch_size = int(fada_cfg.batch_size)
    _fada_v005_replay_settings(fada_cfg, batch_size=batch_size)
    if min(iterations, windows_per_iteration, batch_size) <= 0:
        raise ValueError("FADA iterations, windows_per_iteration, and batch_size must be positive")
    resolved_teacher = (
        Path(teacher_checkpoint)
        if teacher_checkpoint is not None
        else dependencies.resolve_teacher_checkpoint(cfg, root_dir=ROOT_DIR)[0]
    )
    if resolved_teacher is None:
        raise FileNotFoundError(
            "No SAC Oracle checkpoint resolved for FADA training. Set teacher.checkpoint_path "
            "or teacher.load_run/teacher.checkpoint."
        )
    teacher_spec = dependencies.build_teacher_spec(cfg)
    # B2: cold-path strict-load every intermediate Oracle before env/replay mutation.
    if paper_source_plan.enabled:
        for intermediate_path in paper_source_plan.checkpoint_paths:
            dependencies.load_sac_teacher_policy(intermediate_path, teacher_spec, device="cpu")
    policy = FADAPlannerIDMPolicy(config).to(device)
    initial_weights_path = _fada_path(
        OmegaConf.select(fada_cfg, "initial_weights_path"),
        field_name="training.fada.initial_weights_path",
        required=False,
    )
    resume_value = OmegaConf.select(fada_cfg, "resume_path")
    if initial_weights_path is not None and resume_value not in (None, ""):
        raise ValueError("FADA initial_weights_path and resume_path are mutually exclusive")
    if initial_weights_path is not None:
        if not initial_weights_path.is_file():
            raise FileNotFoundError(
                f"FADA initialization checkpoint does not exist: {initial_weights_path}"
            )
        initialized = load_fada_policy_checkpoint(initial_weights_path, device=device)
        if initialized.policy.config != config:
            raise ValueError(
                "FADA initialization architecture mismatch: "
                f"expected={config} observed={initialized.policy.config}"
            )
        policy.load_state_dict(initialized.policy.state_dict(), strict=True)
    idm_optimizer = torch.optim.Adam(policy.idm.parameters(), lr=float(fada_cfg.idm_learning_rate))
    planner_optimizer = torch.optim.Adam(
        policy.planner.parameters(), lr=float(fada_cfg.planner_learning_rate)
    )
    trainer = FADATrainer(
        policy,
        idm_optimizer=idm_optimizer,
        planner_optimizer=planner_optimizer,
        max_grad_norm=float(fada_cfg.max_grad_norm),
    )
    start_iteration = 0
    samples_seen = 0
    resume_path = _fada_path(
        fada_cfg.resume_path,
        field_name="training.fada.resume_path",
        required=False,
    )
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"FADA resume checkpoint does not exist: {resume_path}")
        restored = load_fada_checkpoint(resume_path, policy, trainer, map_location=device)
        start_iteration = int(restored["completed_iterations"])
        samples_seen = int(restored["samples_seen"])

    checkpoint_path = _fada_path(
        fada_cfg.checkpoint_path,
        field_name="training.fada.checkpoint_path",
        required=True,
    )
    if checkpoint_path is None:
        raise RuntimeError("FADA checkpoint path contract was not materialized")
    replay = FADAReplayBuffer(config, capacity=int(fada_cfg.replay_capacity))

    if execution_mode == "persistent_async":
        return _run_fada_persistent_async(
            cfg,
            config=config,
            paper_source_plan=paper_source_plan,
            resolved_teacher=resolved_teacher,
            checkpoint_path=checkpoint_path,
            policy=policy,
            trainer=trainer,
            replay=replay,
            start_iteration=start_iteration,
            samples_seen=samples_seen,
            dependencies=dependencies,
        )

    teacher_policy = dependencies.load_sac_teacher_policy(
        resolved_teacher,
        teacher_spec,
        device=device,
    )

    # B3: 通过 public UniLab env factory 组装 collector, 不向脚本泄漏 backend 状态.
    if create_env_fn is None:
        dependencies.ensure_registries()
        create_env_fn = dependencies.create_env
    if env_cfg_override_fn is None:
        env_cfg_override_fn = lambda owner_cfg: dependencies.backend_adapter_cls(  # noqa: E731
            owner_cfg,
            root_dir=ROOT_DIR,
            algo_name="distill",
        ).build_task_env_cfg_override()
    env = create_env_fn(
        cfg,
        num_envs=int(fada_cfg.num_envs),
        env_cfg_override=env_cfg_override_fn(cfg),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        task_name=str(OmegaConf.select(cfg, "training.task_name")),
    )
    last_stats = None
    last_quality_metrics: dict[str, float] = {}
    collection_summaries: list[dict[str, Any]] = []
    try:
        command_keys = OmegaConf.to_container(fada_cfg.command_info_keys, resolve=True)
        if not isinstance(command_keys, list) or not command_keys:
            raise ValueError("training.fada.command_info_keys must be a non-empty list")
        collection_spec = FADACollectionSpec(
            observation_key=str(fada_cfg.observation_key),
            teacher_projection=str(fada_cfg.teacher_projection),
            student_projection=str(fada_cfg.student_projection),
            student_drop_index=OmegaConf.select(cfg, "training.fada.student_drop_index"),
            command_info_keys=tuple(str(key) for key in command_keys),
            max_env_steps=OmegaConf.select(cfg, "training.fada.max_env_steps"),
            collect_oracle_shadow=collect_oracle_shadow,
        )
        for iteration in range(start_iteration, iterations):
            # B4: 收集 current-policy optimal source, 并为每个 visited state 生成 final-Oracle shadow.
            collection = collect_fada_source_windows(
                env,
                teacher_policy=teacher_policy,
                config=config,
                num_windows=windows_per_iteration,
                rollout_policy=None if iteration == 0 else policy,
                spec=collection_spec,
            )
            replay.add(collection.batch)
            samples_seen += int(collection.batch.command.shape[0])
            collection_summaries.append(
                {
                    "iteration": iteration,
                    "source": "optimal_or_current_policy",
                    "rollout_mode": collection.rollout_mode,
                    "windows": int(collection.batch.command.shape[0]),
                    "env_steps": collection.env_steps,
                    "rejected_done_transitions": collection.rejected_done_transitions,
                    "rejected_command_windows": collection.rejected_command_windows,
                }
            )

            # B5: Appendix B.2 以 2:1 总预算轮转 20 个 intermediate Oracle rollout.
            if paper_source_enabled:
                for intermediate_path, source_windows in paper_source_plan.source_allocations:
                    intermediate_policy = dependencies.load_sac_teacher_policy(
                        intermediate_path,
                        teacher_spec,
                        device=device,
                    )
                    suboptimal = collect_fada_source_windows(
                        env,
                        teacher_policy=teacher_policy,
                        config=config,
                        num_windows=source_windows,
                        rollout_teacher_policy=intermediate_policy,
                        spec=replace(collection_spec, collect_oracle_shadow=True),
                    )
                    replay.add(suboptimal.batch)
                    samples_seen += int(suboptimal.batch.command.shape[0])
                    collection_summaries.append(
                        {
                            "iteration": iteration,
                            "source": "intermediate_oracle",
                            "source_checkpoint": str(intermediate_path),
                            "rollout_mode": suboptimal.rollout_mode,
                            "windows": int(suboptimal.batch.command.shape[0]),
                            "env_steps": suboptimal.env_steps,
                            "rejected_done_transitions": suboptimal.rejected_done_transitions,
                            "rejected_command_windows": suboptimal.rejected_command_windows,
                        }
                    )
            last_stats = trainer.update_from_replay(
                replay,
                batch_size=batch_size,
                idm_updates=int(fada_cfg.idm_updates),
                planner_updates=int(fada_cfg.planner_updates),
                device=device,
            )
            if collect_oracle_shadow:
                last_quality_metrics = evaluate_fada_source_batch(
                    policy,
                    _slice_fada_batch(
                        collection.batch,
                        int(OmegaConf.select(fada_cfg, "quality_eval_max_windows", default=4096)),
                    ),
                )
                iteration_collections = [
                    summary
                    for summary in collection_summaries
                    if int(summary["iteration"]) == iteration
                ]
                last_quality_metrics.update(
                    {
                        "rollout_rejected_done_transitions": float(
                            sum(
                                int(summary["rejected_done_transitions"])
                                for summary in iteration_collections
                            )
                        ),
                        "rollout_rejected_command_windows": float(
                            sum(
                                int(summary["rejected_command_windows"])
                                for summary in iteration_collections
                            )
                        ),
                    }
                )
            runtime_config = cast(dict[str, Any], OmegaConf.to_container(fada_cfg, resolve=True))
            save_fada_checkpoint(
                checkpoint_path,
                policy,
                trainer,
                completed_iterations=iteration + 1,
                samples_seen=samples_seen,
                runtime_config=runtime_config,
                quality_metrics=last_quality_metrics,
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return {
        "mode": "fada_planner_idm_training",
        "checkpoint_path": str(checkpoint_path),
        "completed_iterations": iterations,
        "samples_seen": samples_seen,
        "replay_size": len(replay),
        "last_idm_loss": None if last_stats is None else last_stats.idm_loss,
        "last_planner_loss": None if last_stats is None else last_stats.planner_loss,
        "quality_metrics": last_quality_metrics,
        "collections": collection_summaries,
    }
