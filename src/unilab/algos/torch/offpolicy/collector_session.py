"""Lifecycle owner for one off-policy collector subprocess."""

from __future__ import annotations

import queue
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.common.actor_factory import build_actor
from unilab.algos.torch.offpolicy.collector_support import (
    COLLECTOR_TIMING_KEYS,
    dashboard_components,
    record_phase_ms,
    record_timing_ms,
    resolve_collector_actor_dims,
    resolve_offpolicy_actor_priv_info,
    sample_offpolicy_actions,
    service_collector_pack_requests,
)
from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.base.observations import split_obs_dict
from unilab.base.registry import ensure_registries
from unilab.training.seed import apply_training_seed


@dataclass(frozen=True)
class OffPolicyCollectorSpec:
    stop_event: Any
    env_name: str
    num_envs: int
    replay_buffer: Any
    weight_sync_name: str
    weight_param_shapes: dict
    algo_type: str = "sac"
    actor_hidden_dim: int = 512
    use_layer_norm: bool = True
    learning_starts: int = 0
    metrics_queue: Any = None
    weight_sync_lock: Any = None
    sync_collection: bool = False
    collection_ready_queue: Any = None
    trainer_done_queue: Any = None
    env_steps_per_sync: int = 1
    obs_normalization: bool = False
    shared_obs_normalizer_stats: Any = None
    sim_backend: str = "mujoco"
    env_cfg_override: dict | None = None
    obs_dim: int | None = None
    action_dim: int | None = None
    actor_kwargs: dict | None = None
    seed: int | None = None
    trace_enabled: bool = False
    trace_thread_time: bool = False
    collector_pack_request_queue: Any = None
    collector_pack_ready_queue: Any = None
    collector_pack_shared_slots: Any = None
    nan_guard_cfg: Any = None


@dataclass(frozen=True)
class OffPolicyCollectorDependencies:
    ensure_registries_fn: Callable[..., Any]
    apply_training_seed_fn: Callable[..., Any]
    build_actor_fn: Callable[..., Any]
    split_obs_dict_fn: Callable[..., Any]
    terminal_contract_fn: Callable[..., Any]

    @classmethod
    def defaults(cls) -> OffPolicyCollectorDependencies:
        return cls(
            ensure_registries_fn=ensure_registries,
            apply_training_seed_fn=apply_training_seed,
            build_actor_fn=build_actor,
            split_obs_dict_fn=split_obs_dict,
            terminal_contract_fn=resolve_terminal_observation_contract,
        )


class OffPolicyCollectorSession:
    """Own mutable state and cleanup for one collector child process."""

    def __init__(
        self,
        spec: OffPolicyCollectorSpec,
        dependencies: OffPolicyCollectorDependencies | None = None,
    ) -> None:
        self.spec = spec
        self.dependencies = dependencies or OffPolicyCollectorDependencies.defaults()
        self.trace_recorder = None
        self.pending_collector_pack_request = None
        self.total_steps = 0
        self.env_steps_since_sync = 0
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []
        self.ep_reward_components: defaultdict[str, list[Any]] = defaultdict(list)
        self.timing_accum_ms: defaultdict[str, float] = defaultdict(float)
        self.timing_counts: defaultdict[str, int] = defaultdict(int)
        self.done_count_window = 0
        self.timeout_count_window = 0
        self.terminated_count_window = 0

    def initialize(self) -> None:
        from unilab.base import registry
        from unilab.ipc import SharedWeightSync

        spec = self.spec
        self.dependencies.ensure_registries_fn()
        self.dependencies.apply_training_seed_fn(spec.seed, torch_runtime=True, cuda=True)
        if spec.trace_enabled:
            from unilab.logging.trace_event import TraceRecorder

            self.trace_recorder = TraceRecorder("offpolicy_collector")
        self.env: Any = registry.make(
            spec.env_name,
            num_envs=spec.num_envs,
            sim_backend=spec.sim_backend,
            env_cfg_override=spec.env_cfg_override,
        )
        if spec.nan_guard_cfg is not None and spec.nan_guard_cfg.enabled:
            from unilab.base.nan_guard import NanGuard

            self.env.set_nan_guard(
                NanGuard(
                    spec.nan_guard_cfg,
                    num_envs=self.env.num_envs,
                    supports_state_playback=self.env.play_capabilities.supports_physics_state_playback,
                )
            )
        if self.env.state is None:
            self.env.init_state()
        self.weight_sync = SharedWeightSync(
            spec.weight_param_shapes,
            create=False,
            shm_name=spec.weight_sync_name,
            lock=spec.weight_sync_lock,
        )
        self.weight_sync.trace_recorder = self.trace_recorder
        self.weight_sync.trace_thread_time = spec.trace_thread_time
        self.obs_dim, self.action_dim = resolve_collector_actor_dims(
            self.env,
            obs_dim=spec.obs_dim,
            action_dim=spec.action_dim,
        )
        self.actor = self.dependencies.build_actor_fn(
            spec.algo_type,
            self.obs_dim,
            self.action_dim,
            spec.actor_hidden_dim,
            spec.use_layer_norm,
            "cpu",
            spec.num_envs,
            **(spec.actor_kwargs or {}),
        )
        self.actor.eval()
        spec.replay_buffer.trace_recorder = self.trace_recorder
        spec.replay_buffer.trace_thread_time = spec.trace_thread_time
        state_dict = dict(self.actor.state_dict())
        self.weight_sync.read_weights_into(state_dict)
        self.actor.load_state_dict(state_dict)
        self.local_weight_version = self.weight_sync.version
        self.current_ep_rewards = np.zeros(spec.num_envs, dtype=np.float32)
        self.current_ep_lengths = np.zeros(spec.num_envs, dtype=np.int32)
        actions = np.zeros((spec.num_envs, self.action_dim), dtype=np.float32)
        self.state = self.env.step(actions)
        self.obs_np, self.critic_np = self._split_observations(self.state.obs)
        self.info_dict = self.state.info
        self.prev_dones_np = np.zeros(spec.num_envs, dtype=np.float32)
        self.last_log_time = time.time()

    def _split_observations(self, obs: Any) -> tuple[np.ndarray, np.ndarray]:
        actor_obs, critic_obs = self.dependencies.split_obs_dict_fn(obs)
        return np.asarray(actor_obs, dtype=np.float32), np.asarray(critic_obs, dtype=np.float32)

    def _refresh_weights(self) -> None:
        spec = self.spec
        if self.weight_sync.version <= self.local_weight_version:
            return
        start_ns = time.perf_counter_ns()
        state_dict = dict(self.actor.state_dict())
        self.local_weight_version = self.weight_sync.read_weights_into(state_dict)
        self.actor.load_state_dict(state_dict)
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "collector/check_weight_update",
                category="collector",
                start_ns=start_ns,
                end_ns=time.perf_counter_ns(),
            )
        if spec.obs_normalization and spec.shared_obs_normalizer_stats is not None:
            spec.shared_obs_normalizer_stats.get()

    def _select_actions(self) -> np.ndarray:
        spec = self.spec
        obs_input = self.obs_np
        if spec.obs_normalization and spec.shared_obs_normalizer_stats is not None:
            stats = spec.shared_obs_normalizer_stats.get()
            if stats is not None:
                mean, std = stats
                obs_input = (self.obs_np - mean) / (std + 1e-8)
        with torch.no_grad():
            start_ns = time.perf_counter_ns()
            priv_info_np = resolve_offpolicy_actor_priv_info(
                algo_type=spec.algo_type,
                obs_np=self.obs_np,
                critic_np=self.critic_np,
                info=self.info_dict,
            )
            actions = sample_offpolicy_actions(
                actor=self.actor,
                algo_type=spec.algo_type,
                obs_torch=torch.from_numpy(obs_input),
                prev_dones_torch=torch.from_numpy(self.prev_dones_np),
                priv_info_torch=(
                    torch.from_numpy(priv_info_np) if priv_info_np is not None else None
                ),
            ).numpy()
            if self.trace_recorder:
                self.trace_recorder.add_slice(
                    "collector/actor_infer_cpu",
                    category="collector",
                    start_ns=start_ns,
                    end_ns=time.perf_counter_ns(),
                )
        return actions

    def _step_environment(self, actions_np: np.ndarray) -> Any:
        start_ns = time.perf_counter_ns()
        state = self.env.step(actions_np)
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "collector/env_step",
                category="collector",
                start_ns=start_ns,
                end_ns=time.perf_counter_ns(),
                args={"num_envs": self.spec.num_envs},
            )
        return state

    def _store_transition(self, state: Any, actions_np: np.ndarray) -> None:
        spec = self.spec
        next_obs_np, next_critic_np = self._split_observations(state.obs)
        rewards_np = np.asarray(state.reward, dtype=np.float32).ravel()
        terminated_np = state.terminated.astype(np.float32, copy=False).ravel()
        truncated_np = state.truncated.astype(np.float32, copy=False).ravel()
        combined_dones = (state.terminated | state.truncated).astype(np.float32, copy=False).ravel()
        done_mask_np = combined_dones > 0.5
        timeout_mask_np = truncated_np > 0.5
        terminated_mask_np = np.logical_and(terminated_np > 0.5, ~timeout_mask_np)
        self.done_count_window += int(np.count_nonzero(done_mask_np))
        self.timeout_count_window += int(np.count_nonzero(timeout_mask_np))
        self.terminated_count_window += int(np.count_nonzero(terminated_mask_np))
        terminal_contract = self.dependencies.terminal_contract_fn(
            next_obs_batch_size=next_obs_np.shape[0],
            final_observation=state.final_observation,
            done=done_mask_np,
            info=state.info,
            truncated=truncated_np,
        )
        replay_start_ns = time.perf_counter_ns()
        spec.replay_buffer.add(
            torch.from_numpy(self.obs_np),
            torch.from_numpy(actions_np),
            torch.from_numpy(rewards_np),
            torch.from_numpy(next_obs_np),
            torch.from_numpy(combined_dones),
            torch.from_numpy(truncated_np),
            terminal_mask=torch.from_numpy(terminal_contract.terminal_mask),
            terminal_next_obs=(
                torch.from_numpy(terminal_contract.terminal_obs)
                if terminal_contract.terminal_obs is not None
                else None
            ),
            critic=torch.from_numpy(self.critic_np),
            next_critic=torch.from_numpy(next_critic_np),
            terminal_next_critic=(
                torch.from_numpy(terminal_contract.terminal_critic)
                if terminal_contract.terminal_critic is not None
                else None
            ),
        )
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "collector/replay_add",
                category="collector",
                start_ns=replay_start_ns,
                end_ns=time.perf_counter_ns(),
            )
        self._service_pack_requests()
        self._record_episode_rows(rewards_np, combined_dones)
        self.state = state
        self.obs_np = next_obs_np
        self.critic_np = next_critic_np
        self.info_dict = state.info
        self.prev_dones_np = combined_dones
        self.total_steps += spec.num_envs
        self.env_steps_since_sync += 1

    def _record_episode_rows(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        self.current_ep_rewards += rewards
        self.current_ep_lengths += 1
        reset_indices = np.where(dones > 0.5)[0]
        if len(reset_indices) == 0:
            return
        self.ep_rewards.extend(self.current_ep_rewards[reset_indices].tolist())
        self.ep_lengths.extend(self.current_ep_lengths[reset_indices].tolist())
        self.current_ep_rewards[reset_indices] = 0.0
        self.current_ep_lengths[reset_indices] = 0

    def _service_pack_requests(self) -> None:
        spec = self.spec
        _, self.pending_collector_pack_request = service_collector_pack_requests(
            spec.replay_buffer,
            spec.collector_pack_request_queue,
            spec.collector_pack_ready_queue,
            spec.collector_pack_shared_slots,
            self.trace_recorder,
            block_timeout=0.0,
            pending_request=self.pending_collector_pack_request,
        )

    def _coordinate_collection(self) -> None:
        spec = self.spec
        if not (
            spec.sync_collection
            and spec.collection_ready_queue is not None
            and spec.trainer_done_queue is not None
        ):
            if self.env_steps_since_sync >= spec.env_steps_per_sync:
                self.env_steps_since_sync = 0
            return
        if self.env_steps_since_sync < spec.env_steps_per_sync:
            return
        signal_ns = time.perf_counter_ns()
        spec.collection_ready_queue.put(1)
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "collector/signal_ready",
                category="collector",
                start_ns=signal_ns,
                end_ns=time.perf_counter_ns(),
            )
        wait_ns = time.perf_counter_ns()
        while not spec.stop_event.is_set():
            self._service_pack_requests()
            try:
                spec.trainer_done_queue.get(timeout=0.001)
                self._service_pack_requests()
                break
            except queue.Empty:
                continue
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "collector/wait_trainer_done",
                category="collector",
                start_ns=wait_ns,
                end_ns=time.perf_counter_ns(),
            )
            if spec.metrics_queue is not None:
                try:
                    spec.metrics_queue.put_nowait(
                        {"trace_events": self.trace_recorder.drain_events()}
                    )
                except Exception:
                    pass
        self.env_steps_since_sync = 0

    def _publish_metrics(self) -> None:
        spec = self.spec
        if time.time() - self.last_log_time > 2.0:
            self.last_log_time = time.time()
        log_info = self.state.info.get("log", {})
        if log_info:
            for key, value in dashboard_components(log_info).items():
                self.ep_reward_components[key].append(value)
        if spec.metrics_queue is None or self.total_steps % (spec.num_envs * 10) != 0:
            return
        try:
            message: dict[str, Any] = {
                "total_steps": self.total_steps,
                "buffer_size": int(spec.replay_buffer.size[0]),
            }
            if self.ep_rewards:
                message["mean_ep_reward"] = statistics.mean(self.ep_rewards[-100:])
                message["mean_ep_length"] = (
                    statistics.mean(self.ep_lengths[-100:]) if self.ep_lengths else 0.0
                )
            if self.ep_reward_components:
                message["reward_components"] = {
                    key: statistics.mean(values)
                    for key, values in self.ep_reward_components.items()
                    if values
                }
                self.ep_reward_components.clear()
            if self.timing_counts:
                message["collector_timing_ms"] = {
                    key: value / self.timing_counts[key]
                    for key, value in self.timing_accum_ms.items()
                    if self.timing_counts[key] > 0
                }
            if self.done_count_window > 0:
                message["timeout_rate"] = self.timeout_count_window / self.done_count_window
                message["terminated_rate"] = self.terminated_count_window / self.done_count_window
                self.done_count_window = 0
                self.timeout_count_window = 0
                self.terminated_count_window = 0
            if self.trace_recorder:
                message["trace_events"] = self.trace_recorder.drain_events()
            spec.metrics_queue.put_nowait(message)
            if "collector_timing_ms" in message:
                self.timing_accum_ms.clear()
                self.timing_counts.clear()
        except Exception as error:
            print(f"[OffPolicyWorker] metrics enqueue error: {error}", file=sys.stderr)

    def _run_cycle(self) -> None:
        cycle_timing_ms: dict[str, float] = dict.fromkeys(COLLECTOR_TIMING_KEYS, 0.0)
        phase_start_ns = time.perf_counter_ns()
        self._refresh_weights()
        phase_start_ns = record_phase_ms(cycle_timing_ms, "weight_sync_ms", phase_start_ns)
        actions = self._select_actions()
        phase_start_ns = record_phase_ms(cycle_timing_ms, "action_select_ms", phase_start_ns)
        state = self._step_environment(actions)
        phase_start_ns = record_phase_ms(cycle_timing_ms, "env_step_ms", phase_start_ns)
        self._store_transition(state, actions)
        phase_start_ns = record_phase_ms(cycle_timing_ms, "replay_ms", phase_start_ns)
        self._coordinate_collection()
        phase_start_ns = record_phase_ms(
            cycle_timing_ms,
            "sync_coordination_ms",
            phase_start_ns,
        )
        self._publish_metrics()
        record_phase_ms(cycle_timing_ms, "sync_coordination_ms", phase_start_ns)
        for key in COLLECTOR_TIMING_KEYS:
            record_timing_ms(self.timing_accum_ms, self.timing_counts, key, cycle_timing_ms[key])

    def close(self) -> None:
        if self.spec.metrics_queue is not None and self.trace_recorder:
            try:
                self.spec.metrics_queue.put_nowait(
                    {"trace_events": self.trace_recorder.drain_events()}
                )
            except Exception:
                pass
        self.weight_sync.close()

    def run(self) -> None:
        self.initialize()
        while not self.spec.stop_event.is_set():
            self._run_cycle()
        self.close()
