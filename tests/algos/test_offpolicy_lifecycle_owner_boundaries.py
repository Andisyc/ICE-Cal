from __future__ import annotations

import inspect


def test_collector_entrypoint_preserves_signature_and_delegates_lifecycle() -> None:
    from unilab.algos.torch.offpolicy.collector_session import (
        OffPolicyCollectorSession,
        OffPolicyCollectorSpec,
    )
    from unilab.algos.torch.offpolicy.worker import (
        _run_collector,
        off_policy_collector_fn,
    )

    expected_parameters = (
        "stop_event",
        "env_name",
        "num_envs",
        "replay_buffer",
        "weight_sync_name",
        "weight_param_shapes",
        "algo_type",
        "actor_hidden_dim",
        "use_layer_norm",
        "learning_starts",
        "metrics_queue",
        "weight_sync_lock",
        "sync_collection",
        "collection_ready_queue",
        "trainer_done_queue",
        "env_steps_per_sync",
        "obs_normalization",
        "shared_obs_normalizer_stats",
        "sim_backend",
        "env_cfg_override",
        "obs_dim",
        "action_dim",
        "actor_kwargs",
        "seed",
        "trace_enabled",
        "trace_thread_time",
        "collector_pack_request_queue",
        "collector_pack_ready_queue",
        "collector_pack_shared_slots",
        "nan_guard_cfg",
        "kwargs",
    )
    assert tuple(inspect.signature(off_policy_collector_fn).parameters) == expected_parameters
    assert OffPolicyCollectorSpec.__dataclass_params__.frozen
    assert len(inspect.getsourcelines(_run_collector)[0]) <= 25
    for phase in (
        "initialize",
        "_select_actions",
        "_step_environment",
        "_coordinate_collection",
        "_publish_metrics",
        "close",
        "run",
    ):
        assert callable(getattr(OffPolicyCollectorSession, phase))


def test_double_buffer_runner_delegates_one_run_to_explicit_session() -> None:
    from unilab.algos.torch.offpolicy.double_buffer_runner import (
        DoubleBufferOffPolicyRunner,
    )
    from unilab.algos.torch.offpolicy.double_buffer_session import (
        DoubleBufferTrainingSession,
    )

    assert len(inspect.getsourcelines(DoubleBufferOffPolicyRunner.learn)[0]) <= 30
    for phase in (
        "prepare",
        "_start_collector",
        "_wait_for_data",
        "_run_iteration",
        "_finalize_success",
        "_handle_collector_died",
        "run",
    ):
        assert callable(getattr(DoubleBufferTrainingSession, phase))
