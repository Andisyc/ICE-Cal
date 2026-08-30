"""Public subprocess entrypoint for SAC/TD3 off-policy collection."""

from __future__ import annotations

import sys

from unilab.algos.torch.offpolicy.collector_session import (
    OffPolicyCollectorDependencies,
    OffPolicyCollectorSession,
    OffPolicyCollectorSpec,
)
from unilab.algos.torch.offpolicy.collector_support import (
    COLLECTOR_TIMING_KEYS,
    dashboard_components,
    offpolicy_actor_requires_priv_info,
    resolve_collector_actor_dims,
    resolve_offpolicy_actor_priv_info,
    sample_offpolicy_actions,
)
from unilab.algos.torch.offpolicy.collector_support import (
    collector_pack_shared_batch as _collector_pack_shared_batch,
)
from unilab.algos.torch.offpolicy.collector_support import (
    record_phase_ms as _record_phase_ms,
)
from unilab.algos.torch.offpolicy.collector_support import (
    record_timing_ms as _record_timing_ms,
)
from unilab.algos.torch.offpolicy.collector_support import (
    service_collector_pack_requests as _service_collector_pack_requests,
)


def off_policy_collector_fn(
    stop_event,
    env_name: str,
    num_envs: int,
    replay_buffer,
    weight_sync_name: str,
    weight_param_shapes: dict,
    algo_type: str = "sac",
    actor_hidden_dim: int = 512,
    use_layer_norm: bool = True,
    learning_starts: int = 0,
    metrics_queue=None,
    weight_sync_lock=None,
    sync_collection: bool = False,
    collection_ready_queue=None,
    trainer_done_queue=None,
    env_steps_per_sync: int = 1,
    obs_normalization: bool = False,
    shared_obs_normalizer_stats=None,
    sim_backend: str = "mujoco",
    env_cfg_override: dict | None = None,
    obs_dim: int | None = None,
    action_dim: int | None = None,
    actor_kwargs: dict | None = None,
    seed: int | None = None,
    trace_enabled: bool = False,
    trace_thread_time: bool = False,
    collector_pack_request_queue=None,
    collector_pack_ready_queue=None,
    collector_pack_shared_slots=None,
    nan_guard_cfg=None,
    **kwargs,
):
    """Run one collector child; AsyncRunner owns outer error propagation."""

    del kwargs
    print("[Collector] Entry point called", file=sys.stderr, flush=True)
    _run_collector(
        OffPolicyCollectorSpec(
            stop_event=stop_event,
            env_name=env_name,
            num_envs=num_envs,
            replay_buffer=replay_buffer,
            weight_sync_name=weight_sync_name,
            weight_param_shapes=weight_param_shapes,
            algo_type=algo_type,
            actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            learning_starts=learning_starts,
            metrics_queue=metrics_queue,
            weight_sync_lock=weight_sync_lock,
            sync_collection=sync_collection,
            collection_ready_queue=collection_ready_queue,
            trainer_done_queue=trainer_done_queue,
            env_steps_per_sync=env_steps_per_sync,
            obs_normalization=obs_normalization,
            shared_obs_normalizer_stats=shared_obs_normalizer_stats,
            sim_backend=sim_backend,
            env_cfg_override=env_cfg_override,
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_kwargs=actor_kwargs,
            seed=seed,
            trace_enabled=trace_enabled,
            trace_thread_time=trace_thread_time,
            collector_pack_request_queue=collector_pack_request_queue,
            collector_pack_ready_queue=collector_pack_ready_queue,
            collector_pack_shared_slots=collector_pack_shared_slots,
            nan_guard_cfg=nan_guard_cfg,
        )
    )


def _run_collector(
    spec: OffPolicyCollectorSpec,
    dependencies: OffPolicyCollectorDependencies | None = None,
) -> None:
    OffPolicyCollectorSession(spec, dependencies).run()


__all__ = [
    "COLLECTOR_TIMING_KEYS",
    "OffPolicyCollectorDependencies",
    "OffPolicyCollectorSession",
    "OffPolicyCollectorSpec",
    "_collector_pack_shared_batch",
    "_record_phase_ms",
    "_record_timing_ms",
    "_run_collector",
    "_service_collector_pack_requests",
    "dashboard_components",
    "off_policy_collector_fn",
    "offpolicy_actor_requires_priv_info",
    "resolve_collector_actor_dims",
    "resolve_offpolicy_actor_priv_info",
    "sample_offpolicy_actions",
]
