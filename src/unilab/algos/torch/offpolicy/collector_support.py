"""Pure decisions and replay-pack helpers shared by the off-policy collector."""

from __future__ import annotations

import queue
import time
from typing import cast

import numpy as np
import torch

from unilab.base.observations import get_obs_dims

COLLECTOR_TIMING_KEYS = (
    "weight_sync_ms",
    "action_select_ms",
    "env_step_ms",
    "replay_ms",
    "sync_coordination_ms",
)


def dashboard_components(log_info: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in log_info.items() if key.startswith(("reward/", "curriculum/"))
    }


def offpolicy_actor_requires_priv_info(algo_type: str) -> bool:
    return algo_type in {
        "hora_sac",
        "privileged_locomotion_sac",
        "privileged_residual_sac",
        "privileged_full_action_sac",
    }


def resolve_collector_actor_dims(
    env,
    obs_dim: int | None = None,
    action_dim: int | None = None,
) -> tuple[int, int]:
    if obs_dim is None:
        obs_dim, _ = get_obs_dims(env.obs_groups_spec)
    if action_dim is None:
        assert env.action_space.shape is not None
        action_dim = env.action_space.shape[0]
    assert obs_dim is not None
    assert action_dim is not None
    return obs_dim, action_dim


def sample_offpolicy_actions(
    actor,
    algo_type: str,
    obs_torch: torch.Tensor,
    prev_dones_torch: torch.Tensor,
    priv_info_torch: torch.Tensor | None = None,
) -> torch.Tensor:
    if algo_type in ("sac", "td3", "flashsac"):
        return cast(
            torch.Tensor,
            actor.explore(obs_torch, dones=prev_dones_torch, deterministic=False),
        )
    if offpolicy_actor_requires_priv_info(algo_type):
        if priv_info_torch is None:
            raise ValueError(f"{algo_type} collector action sampling requires priv_info_torch.")
        return cast(
            torch.Tensor,
            actor.explore(obs_torch, priv_info_torch, deterministic=False),
        )
    raise ValueError(f"Unsupported off-policy algo_type for collector action sampling: {algo_type}")


def resolve_offpolicy_actor_priv_info(
    *,
    algo_type: str,
    obs_np: np.ndarray,
    critic_np: np.ndarray,
    info: dict | None,
) -> np.ndarray | None:
    if not offpolicy_actor_requires_priv_info(algo_type):
        return None
    if algo_type in {"hora_sac", "privileged_locomotion_sac"}:
        from unilab.algos.torch.hora.observations import split_hora_obs_with_priv_info

        _, _, priv_info_np = split_hora_obs_with_priv_info(
            {"obs": obs_np, "critic": critic_np},
            info,
        )
        if priv_info_np is None:
            raise ValueError(
                "HORA-SAC collector requires privileged info from info['critic_info'] "
                "or the critic observation tail."
            )
        return np.asarray(priv_info_np, dtype=np.float32)

    from unilab.algos.torch.fada_context.privileged_full_action_sac import (
        PRIVILEGED_ACTUATOR_STRENGTH_DIM,
    )

    explicit = None if info is None else info.get("privileged_actuator_strength")
    if explicit is not None:
        strength = np.asarray(explicit, dtype=np.float32)
        expected_shape = (obs_np.shape[0], PRIVILEGED_ACTUATOR_STRENGTH_DIM)
        if strength.shape != expected_shape:
            raise ValueError(
                f"{algo_type} requires explicit 29D actuator strength with "
                f"shape {expected_shape}, got {strength.shape}."
            )
        return strength
    minimum_critic_dim = int(obs_np.shape[-1]) + PRIVILEGED_ACTUATOR_STRENGTH_DIM
    if critic_np.ndim != 2 or int(critic_np.shape[-1]) < minimum_critic_dim:
        raise ValueError(
            f"{algo_type} requires 29D actuator strength from "
            "info['privileged_actuator_strength'] or the critic observation tail."
        )
    return np.asarray(critic_np[:, -PRIVILEGED_ACTUATOR_STRENGTH_DIM:], dtype=np.float32)


def record_timing_ms(timing_accum_ms, timing_counts, key: str, value: float) -> None:
    timing_accum_ms[key] += float(value)
    timing_counts[key] += 1


def record_phase_ms(cycle_timing_ms: dict[str, float], key: str, start_ns: int) -> int:
    end_ns = time.perf_counter_ns()
    cycle_timing_ms[key] += (end_ns - start_ns) / 1e6
    return end_ns


def collector_pack_shared_batch(replay_buffer, request: dict, shared_slots) -> dict:
    tick_id = int(request["tick_id"])
    snapshot_ptr = int(replay_buffer.ptr[0])
    snapshot_size = int(replay_buffer.size[0])
    sample_seed = int(request["sample_seed"])
    sample_count = int(request["sample_count"])
    shared_slot = int(request["shared_slot"])
    target_gpu_slot = int(request["target_gpu_slot"])
    learner_hot_gpu_slot = int(request["learner_hot_gpu_slot"])
    if target_gpu_slot == learner_hot_gpu_slot:
        raise RuntimeError(
            "collector_thread pack target_gpu_slot must differ from learner_hot_gpu_slot"
        )
    pack_begin_ns = time.perf_counter_ns()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(sample_seed)
    indices = torch.randint(0, snapshot_size, (sample_count,), generator=gen)
    dst = shared_slots[shared_slot]
    torch.index_select(replay_buffer._storage, 0, indices, out=dst)
    pack_end_ns = time.perf_counter_ns()
    return {
        "tick_id": tick_id,
        "snapshot_ptr": snapshot_ptr,
        "snapshot_size": snapshot_size,
        "sample_seed": sample_seed,
        "sample_count": sample_count,
        "shared_slot": shared_slot,
        "target_gpu_slot": target_gpu_slot,
        "learner_hot_gpu_slot": learner_hot_gpu_slot,
        "pack_layout": "packed",
        "pack_executor": "collector_thread",
        "pack_begin_ns": pack_begin_ns,
        "pack_end_ns": pack_end_ns,
    }


def service_collector_pack_requests(
    replay_buffer,
    request_queue,
    ready_queue,
    shared_slots,
    trace_recorder=None,
    *,
    block_timeout: float = 0.0,
    pending_request: dict | None = None,
) -> tuple[bool, dict | None]:
    if request_queue is None or ready_queue is None or shared_slots is None:
        return False, pending_request
    request = pending_request
    if request is None:
        try:
            request = (
                request_queue.get(timeout=block_timeout)
                if block_timeout > 0
                else request_queue.get_nowait()
            )
        except queue.Empty:
            return False, None
    if request is None:
        return False, None
    if int(replay_buffer.ptr[0]) < int(request.get("min_snapshot_ptr", 0)):
        return False, request
    ready = collector_pack_shared_batch(replay_buffer, request, shared_slots)
    if trace_recorder:
        trace_recorder.add_slice(
            "collector/cpu_pack_sample_batch",
            category="collector",
            start_ns=int(ready["pack_begin_ns"]),
            end_ns=int(ready["pack_end_ns"]),
            args={
                "tick_id": int(ready["tick_id"]),
                "sample_count": int(ready["sample_count"]),
                "shared_slot": int(ready["shared_slot"]),
                "target_gpu_slot": int(ready["target_gpu_slot"]),
                "learner_hot_gpu_slot": int(ready["learner_hot_gpu_slot"]),
                "pack_layout": "packed",
                "pack_executor": "collector_thread",
                "shared_memory": True,
                "pinned_memory": False,
            },
        )
    ready_queue.put(ready)
    return True, None
