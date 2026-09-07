"""Runner construction for supported off-policy algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parents[4]

from unilab.base.nan_guard import NanGuardCfg
from unilab.training import (
    BackendAdapter,
    assert_offpolicy_task_choice_matches_algo,
    create_env,
)


def build_offpolicy_env_cfg_override(algo_name: str, cfg: DictConfig) -> dict[str, Any] | None:
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)
    return cast(
        dict[str, Any] | None,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name=algo_name).build_task_env_cfg_override(),
    )


def apply_configured_actor_warm_start(
    algo_name: str,
    cfg: DictConfig,
    runner: Any,
) -> dict[str, Any] | None:
    """Apply the explicit G1 actor-only warm start before runner.learn()."""

    checkpoint = OmegaConf.select(cfg, "algo.actor_warm_start_checkpoint")
    if checkpoint in (None, ""):
        return None
    adapter_id = OmegaConf.select(cfg, "algo.actor_warm_start_adapter")
    from unilab.algos.torch.offpolicy.checkpoint_adapter import (
        G1_HEIGHT_ACTOR_ADAPTER_ID,
        G1_HEIGHT_ACTOR_CONTINUATION_ADAPTER_ID,
        load_g1_height_actor_continuation_warm_start,
        load_g1_height_actor_warm_start,
    )

    if algo_name != "sac":
        raise ValueError("actor-only G1 height warm start currently supports only SAC")
    if str(cfg.training.task_name) != "G1StandHeight":
        raise ValueError("G1 height actor adapter may only initialize G1StandHeight")
    loaders = {
        G1_HEIGHT_ACTOR_ADAPTER_ID: load_g1_height_actor_warm_start,
        G1_HEIGHT_ACTOR_CONTINUATION_ADAPTER_ID: (load_g1_height_actor_continuation_warm_start),
    }
    if adapter_id not in loaders:
        raise ValueError(
            f"algo.actor_warm_start_adapter must be one of {sorted(loaders)!r}, got {adapter_id!r}"
        )
    metadata = loaders[adapter_id](runner.learner, str(checkpoint))
    print(
        "[ActorWarmStart] "
        f"adapter={metadata['adapter_id']} "
        f"parent_sha256={metadata['parent_checkpoint_sha256']}"
    )
    return metadata


def build_runner(algo_name: str, cfg: DictConfig):
    """Build algorithm runner from unified Hydra config."""
    env_cfg_override = build_offpolicy_env_cfg_override(algo_name, cfg)

    nan_guard_cfg = getattr(cfg.training, "nan_guard", None)
    _nan_guard_cfg: NanGuardCfg | None = None
    if nan_guard_cfg is not None and getattr(nan_guard_cfg, "enabled", False):
        _nan_guard_cfg = NanGuardCfg(
            enabled=True,
            buffer_size=int(getattr(nan_guard_cfg, "buffer_size", 100)),
            max_envs_to_dump=int(getattr(nan_guard_cfg, "max_envs_to_dump", 5)),
            output_dir=getattr(nan_guard_cfg, "output_dir", None),
        )

    replay_prefetch_mode = getattr(cfg.training, "replay_prefetch_mode", "one_tick")
    if replay_prefetch_mode != "one_tick":
        raise ValueError(
            f"Unsupported training.replay_prefetch_mode={replay_prefetch_mode!r}; "
            "expected 'one_tick'"
        )
    verbose_metrics = bool(getattr(cfg.training, "verbose_metrics", False))
    if cfg.training.num_gpus > 1:
        if algo_name == "flashsac":
            raise ValueError("FlashSAC does not support training.num_gpus > 1")
        raise ValueError("cpu_pinned_double_buffer is currently single-GPU only")

    if cfg.training.no_sync_collection:
        raise ValueError("cpu_pinned_double_buffer requires synchronized collection")

    if algo_name == "sac":
        from unilab.algos.torch.fast_sac.learner import FastSACLearner
        from unilab.algos.torch.offpolicy.double_buffer_runner import (
            DoubleBufferOffPolicyRunner,
        )
        from unilab.algos.torch.offpolicy.runtime import resolve_custom_offpolicy_runtime
        from unilab.base.registry import ensure_registries as _ensure
        from unilab.utils.device import get_default_device

        _ensure()
        _device = cfg.training.device or get_default_device()
        _rl_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
        _custom_runtime = resolve_custom_offpolicy_runtime(_rl_cfg)
        if _custom_runtime is not None:
            _custom_runtime.validate_training_config(cfg)
        _training_model_kwargs: dict[str, Any] = {}
        _env = cast(Any, create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override))
        try:
            assert _env.action_space.shape
            from unilab.base.observations import get_obs_dims as _get_obs_dims

            _obs_dim, _critic_dim = _get_obs_dims(_env.obs_groups_spec)
            _action_dim = _env.action_space.shape[0]
            if _custom_runtime is not None:
                _training_model_kwargs = cast(
                    dict[str, Any],
                    _custom_runtime.build_training_model_kwargs(
                        cfg=cfg,
                        env=_env,
                        obs_dim=int(_obs_dim),
                        critic_obs_dim=int(_critic_dim),
                        action_dim=int(_action_dim),
                    ),
                )
            _symmetry_aug = None
            if (
                _custom_runtime is not None
                and cfg.algo.use_symmetry
                and not _custom_runtime.supports_symmetry
            ):
                raise ValueError("Selected SAC off-policy runtime does not support symmetry.")
            if cfg.algo.use_symmetry:
                _symmetry_aug = _env.build_symmetry_augmentation(device=_device)
                if _symmetry_aug is None:
                    raise ValueError(
                        f"{cfg.training.task_name} does not provide symmetry augmentation"
                    )
        finally:
            _env.close()

        _batch_size = cfg.algo.batch_size
        if _symmetry_aug is not None:
            if _batch_size % _symmetry_aug.batch_multiplier != 0:
                raise ValueError(
                    "Symmetry augmentation requires batch_size divisible by "
                    f"{_symmetry_aug.batch_multiplier}, got {_batch_size}"
                )
            _batch_size = _batch_size // _symmetry_aug.batch_multiplier

        _learner_cls = FastSACLearner
        _algo_type = "sac"
        _actor_kwargs: dict[str, Any] = {}
        _learner_extra_kwargs: dict[str, Any] = {}
        if _custom_runtime is not None:
            _learner_extra_kwargs = dict(_training_model_kwargs)
            if _custom_runtime.learner_cls is not None:
                _learner_cls = _custom_runtime.learner_cls
            if _custom_runtime.algo_type is not None:
                _algo_type = str(_custom_runtime.algo_type)
            _actor_kwargs = dict(_learner_extra_kwargs)
            _actor_kwargs.pop("checkpoint_contract", None)

        _learner = _learner_cls(
            obs_dim=_obs_dim,
            action_dim=_action_dim,
            device=_device,
            gamma=cfg.algo.gamma,
            tau=cfg.algo.tau,
            actor_lr=cfg.algo.actor_lr,
            critic_lr=cfg.algo.critic_lr,
            alpha_lr=cfg.algo.algo_params.alpha_lr,
            alpha_init=cfg.algo.algo_params.alpha_init,
            target_entropy_ratio=cfg.algo.algo_params.target_entropy_ratio,
            actor_hidden_dim=cfg.algo.actor_hidden_dim,
            critic_hidden_dim=cfg.algo.critic_hidden_dim,
            num_atoms=cfg.algo.num_atoms,
            use_layer_norm=cfg.algo.use_layer_norm,
            max_grad_norm=cfg.algo.algo_params.max_grad_norm,
            use_amp=cfg.training.use_amp,
            amp_dtype=cfg.algo.algo_params.amp_dtype,
            use_compile=cfg.algo.algo_params.use_compile,
            use_symmetry=cfg.algo.use_symmetry,
            symmetry_augmentation=_symmetry_aug,
            critic_obs_dim=_critic_dim,
            **_learner_extra_kwargs,
        )
        _checkpoint_saver = (
            None if _custom_runtime is None else _custom_runtime.build_checkpoint_saver(_learner)
        )

        return DoubleBufferOffPolicyRunner(
            learner=_learner,
            env_name=cfg.training.task_name,
            algo_type=_algo_type,
            num_envs=cfg.algo.num_envs,
            replay_buffer_n=cfg.algo.replay_buffer_n,
            batch_size=_batch_size,
            learning_starts=cfg.algo.learning_starts,
            updates_per_step=cfg.algo.updates_per_step,
            policy_frequency=cfg.algo.policy_frequency,
            sync_collection=True,
            env_steps_per_sync=cfg.training.env_steps_per_sync,
            device=_device,
            actor_hidden_dim=cfg.algo.actor_hidden_dim,
            use_layer_norm=cfg.algo.use_layer_norm,
            obs_normalization=cfg.algo.obs_normalization,
            sim_backend=cfg.training.sim_backend,
            env_cfg_override=env_cfg_override,
            actor_kwargs=_actor_kwargs,
            checkpoint_saver=_checkpoint_saver,
            trace_enabled=cfg.training.trace_enabled,
            trace_output_dir=cfg.training.trace_output_dir,
            trace_thread_time=cfg.training.trace_thread_time,
            trace_cuda_events=cfg.training.trace_cuda_events,
            replay_prefetch_mode=replay_prefetch_mode,
            verbose_metrics=verbose_metrics,
            seed=cfg.algo.seed,
            nan_guard_cfg=_nan_guard_cfg,
        )

    if algo_name == "td3":
        from unilab.algos.torch.common.device import get_env_dims
        from unilab.algos.torch.fast_td3.learner import FastTD3Learner
        from unilab.algos.torch.offpolicy.double_buffer_runner import (
            DoubleBufferOffPolicyRunner,
        )
        from unilab.utils.device import get_default_device

        _device = cfg.training.device or get_default_device()
        _obs_dim, _action_dim, _critic_dim = get_env_dims(
            cfg.training.task_name,
            cfg.training.sim_backend,
            env_cfg_override=env_cfg_override,
        )
        _learner = FastTD3Learner(
            obs_dim=_obs_dim,
            action_dim=_action_dim,
            critic_obs_dim=_critic_dim,
            num_envs=cfg.algo.num_envs,
            device=_device,
            gamma=cfg.algo.gamma,
            tau=cfg.algo.tau,
            actor_lr=cfg.algo.actor_lr,
            critic_lr=cfg.algo.critic_lr,
            actor_hidden_dim=cfg.algo.actor_hidden_dim,
            critic_hidden_dim=cfg.algo.critic_hidden_dim,
            num_atoms=cfg.algo.num_atoms,
            v_min=cfg.algo.algo_params.v_min,
            v_max=cfg.algo.algo_params.v_max,
            init_scale=cfg.algo.algo_params.init_scale,
            log_std_min=cfg.algo.algo_params.log_std_min,
            log_std_max=cfg.algo.algo_params.log_std_max,
            weight_decay=cfg.algo.algo_params.weight_decay,
            use_cdq=cfg.algo.algo_params.use_cdq,
            policy_noise=cfg.algo.algo_params.policy_noise,
            noise_clip=cfg.algo.algo_params.noise_clip,
            policy_frequency=cfg.algo.policy_frequency,
            obs_normalization=cfg.algo.obs_normalization,
        )

        _actor_kwargs = {
            "init_scale": cfg.algo.algo_params.init_scale,
            "log_std_min": cfg.algo.algo_params.log_std_min,
            "log_std_max": cfg.algo.algo_params.log_std_max,
        }

        return DoubleBufferOffPolicyRunner(
            learner=_learner,
            env_name=cfg.training.task_name,
            algo_type="td3",
            env_cfg_override=env_cfg_override,
            device=_device,
            num_envs=cfg.algo.num_envs,
            replay_buffer_n=cfg.algo.replay_buffer_n,
            batch_size=cfg.algo.batch_size,
            learning_starts=cfg.algo.learning_starts,
            updates_per_step=cfg.algo.updates_per_step,
            policy_frequency=cfg.algo.policy_frequency,
            sync_collection=True,
            env_steps_per_sync=cfg.training.env_steps_per_sync,
            actor_hidden_dim=cfg.algo.actor_hidden_dim,
            use_layer_norm=False,
            obs_normalization=cfg.algo.obs_normalization,
            sim_backend=cfg.training.sim_backend,
            seed=cfg.algo.seed,
            trace_enabled=cfg.training.trace_enabled,
            trace_output_dir=cfg.training.trace_output_dir,
            trace_thread_time=cfg.training.trace_thread_time,
            trace_cuda_events=cfg.training.trace_cuda_events,
            replay_prefetch_mode=replay_prefetch_mode,
            verbose_metrics=verbose_metrics,
            actor_kwargs=_actor_kwargs,
            nan_guard_cfg=_nan_guard_cfg,
        )

    if algo_name == "flashsac":
        from unilab.algos.torch.flash_sac.double_buffer import (
            build_flashsac_double_buffer_runner,
        )

        return build_flashsac_double_buffer_runner(
            cfg,
            env_cfg_override=env_cfg_override,
            replay_prefetch_mode=replay_prefetch_mode,
            verbose_metrics=verbose_metrics,
            nan_guard_cfg=_nan_guard_cfg,
        )

    raise ValueError(f"Unsupported algo: {algo_name}")
