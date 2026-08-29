from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch

from .collector import project_student_obs, project_teacher_obs, set_transition_input_rows
from .fada import FADAArchitectureConfig, FADAPlannerIDMPolicy, FADASourceBatch
from .fada_collection_contract import (
    FADACollectionResult,
    FADACollectionSpec,
    FADACollectionTransition,
)
from .fada_collection_io import (
    _command_array,
    _done_mask,
    _fada_actions,
    _next_after_done,
    _obs_array,
    _oracle_actions,
    _oracle_shadow_pair,
    _policy_actions,
)
from .fada_collection_windows import (
    _cold_start_window_from_records,
    _concat_batches,
    _terminal_planner_window,
    _walking_recovery_window,
    _window_from_records,
)
from .fada_observation import assert_fada_projection_matches_contract
from .fada_training_diagnostics import FADACollectionProgressReporter

_BATCH_COMPACTION_SIZE = 256


def _default_collection_step_limit(
    *,
    num_windows: int,
    num_envs: int,
    record_count: int,
) -> int:
    """Return a conservative cap without assuming >=10% row acceptance."""

    ideal_parallel_steps = int(np.ceil(int(num_windows) / max(int(num_envs), 1)))
    legacy_headroom = int(record_count) + ideal_parallel_steps * 10
    one_window_per_vector_step = int(record_count) + int(num_windows)
    return max(legacy_headroom, one_window_per_vector_step, 1)


def collect_fada_source_windows(
    env: Any,
    *,
    teacher_policy: torch.nn.Module,
    config: FADAArchitectureConfig,
    num_windows: int,
    rollout_policy: FADAPlannerIDMPolicy | None = None,
    rollout_teacher_policy: torch.nn.Module | None = None,
    spec: FADACollectionSpec | None = None,
) -> FADACollectionResult:
    """产出 command-scenario 对齐的 causal windows 与 same-state Oracle labels.

    函数名说明:
        该 collector owner 负责 rollout/history/window provenance, 不拥有 loss 或 optimizer.

    主链路:
        上游: persistent worker 选择的 walk/static_stand/walk_to_stand scenario 与 Oracle role.
        下游: FADASourceBatch 进入 replay, IDM pass 和 fixed-IDM Planner pass.

    语义:
        future K-step command 必须恒定; walk_to_stand 额外要求 history 含 active command 且 future 为零.
    """

    # B1: 解析一个 collection spec 并建立 rollout history, 禁止 caller 参数半开和 state 分叉.
    spec = FADACollectionSpec() if spec is None else spec
    observation_key = spec.observation_key
    teacher_projection = spec.teacher_projection
    student_projection = spec.student_projection
    student_drop_index = spec.student_drop_index
    command_info_keys = spec.command_info_keys
    max_env_steps = spec.max_env_steps
    collect_oracle_shadow = spec.collect_oracle_shadow
    command_scenario = spec.command_scenario
    transition_walk_command = spec.transition_walk_command
    transition_pre_switch_steps = spec.transition_pre_switch_steps
    transition_post_switch_steps = spec.transition_post_switch_steps
    planner_eligible = spec.planner_eligible
    cold_start_windows = spec.cold_start_windows
    idm_source_role = 1 if rollout_policy is None and rollout_teacher_policy is None else 0
    assert_fada_projection_matches_contract(
        observation_contract=config.observation_contract,
        projection=student_projection,
    )
    if int(num_windows) <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")
    if rollout_policy is not None and rollout_teacher_policy is not None:
        raise ValueError("set only one of rollout_policy and rollout_teacher_policy")
    if command_scenario not in {"walk", "static_stand", "walk_to_stand"}:
        raise ValueError(f"unsupported FADA command_scenario: {command_scenario!r}")
    if command_scenario != "walk" and rollout_teacher_policy is not None:
        raise ValueError("intermediate Oracle rollouts support only the walk scenario")
    if cold_start_windows and command_scenario == "walk_to_stand":
        raise ValueError("cold_start_windows does not support command_scenario='walk_to_stand'")
    walking_recovery = bool(cold_start_windows and command_scenario == "walk")
    if walking_recovery and not collect_oracle_shadow:
        raise ValueError("walking cold-start collection requires Oracle shadow supervision")
    num_envs = int(env.num_envs)
    reset_all = getattr(env, "reset_all", None)
    if callable(reset_all):
        initial_state = reset_all()
        initial_obs = cast(Mapping[str, Any], getattr(initial_state, "obs"))
        initial_info = cast(Mapping[str, Any], getattr(initial_state, "info"))
        obs = {key: np.asarray(value).copy() for key, value in initial_obs.items()}
        info = dict(initial_info)
    else:
        if getattr(env, "state", None) is None and callable(getattr(env, "init_state", None)):
            env.init_state()
        obs, info = env.reset(np.arange(num_envs, dtype=np.int32))
        obs = dict(obs)
        info = dict(info)
    forced_command_rows: np.ndarray | None = None
    transition_phase_step = 0
    transition_pre_steps = (
        config.history_length
        if transition_pre_switch_steps is None
        else int(transition_pre_switch_steps)
    )
    transition_post_steps = (
        config.history_length + config.prediction_horizon
        if transition_post_switch_steps is None
        else int(transition_post_switch_steps)
    )
    if command_scenario != "walk":
        if config.command_dim != 3 or list(command_info_keys) != ["commands"]:
            raise ValueError(
                "standing curriculum requires command_dim=3 and command_info_keys=['commands']"
            )
        if transition_pre_steps < config.history_length:
            raise ValueError(
                "transition_pre_switch_steps must be at least history_length, "
                f"got {transition_pre_steps} < {config.history_length}"
            )
        if transition_post_steps < config.prediction_horizon:
            raise ValueError(
                "transition_post_switch_steps must be at least prediction_horizon, "
                f"got {transition_post_steps} < {config.prediction_horizon}"
            )
        walk_command = np.asarray(transition_walk_command, dtype=np.float32)
        if walk_command.shape != (3,) or not bool(np.all(np.isfinite(walk_command))):
            raise ValueError("transition_walk_command must be a finite 3-D command")
        if not bool(np.any(np.abs(walk_command) > 1.0e-6)):
            raise ValueError("transition_walk_command must be active")
        forced_command_rows = np.zeros((num_envs, 3), dtype=np.float32)
        if command_scenario == "walk_to_stand":
            forced_command_rows[:] = walk_command
        obs, info = set_transition_input_rows(
            env,
            command_info_key="commands",
            command_rows=forced_command_rows,
        )

    source = _obs_array(obs, observation_key)
    student_obs = project_student_obs(
        source,
        projection=student_projection,
        expected_student_obs_dim=config.obs_dim,
        student_drop_index=student_drop_index,
    )
    observation_history = np.repeat(student_obs[:, None, :], config.history_length, axis=1)
    action_history = np.zeros(
        (num_envs, config.history_length, config.action_dim), dtype=np.float32
    )
    record_count = (
        config.prediction_horizon
        if cold_start_windows
        else config.history_length + config.prediction_horizon - 1
    )
    records: list[deque[FADACollectionTransition]] = [
        deque(maxlen=record_count) for _ in range(num_envs)
    ]
    episode_ids = np.zeros((num_envs,), dtype=np.int64)
    episode_timesteps = np.zeros((num_envs,), dtype=np.int64)
    pending_batches: list[FADASourceBatch] = []
    compacted_batches: list[FADASourceBatch] = []
    window_count = 0

    def append_window(batch: FADASourceBatch) -> None:
        nonlocal window_count
        pending_batches.append(batch)
        window_count += int(batch.observation_history.shape[0])
        if len(pending_batches) >= _BATCH_COMPACTION_SIZE:
            compacted_batches.append(_concat_batches(tuple(pending_batches), config))
            pending_batches.clear()

    rejected_done = 0
    rejected_command = 0
    rejected_scenario = 0
    env_steps = 0
    rollout_mode = (
        "intermediate_oracle"
        if rollout_teacher_policy is not None
        else ("oracle" if rollout_policy is None else "planner_idm")
    )
    step_limit = (
        int(max_env_steps)
        if max_env_steps is not None
        else _default_collection_step_limit(
            num_windows=int(num_windows),
            num_envs=num_envs,
            record_count=record_count,
        )
    )
    reporter = FADACollectionProgressReporter(
        scenario=command_scenario,
        window_profile=("cold_start" if cold_start_windows else "steady_state"),
        rollout_mode=rollout_mode,
        target_windows=int(num_windows),
        num_envs=num_envs,
    )
    reporter.report(
        windows=0,
        env_steps=0,
        rejected_done=0,
        rejected_command=0,
        rejected_scenario=0,
    )
    # B2: 每步先查询同状态 Oracle label, 再执行 Oracle bootstrap 或当前 Planner-IDM 动作.
    while window_count < int(num_windows):
        if env_steps >= step_limit:
            reporter.report(
                windows=window_count,
                env_steps=env_steps,
                rejected_done=rejected_done,
                rejected_command=rejected_command,
                rejected_scenario=rejected_scenario,
                force=True,
            )
            attempted_rows = env_steps * num_envs
            acceptance = (
                0.0 if attempted_rows <= 0 else 100.0 * window_count / attempted_rows
            )
            raise RuntimeError(
                f"FADA collector produced {window_count}/{num_windows} windows after "
                f"{env_steps} env steps (acceptance={acceptance:.2f}%, "
                f"rejected_done={rejected_done}, rejected_command={rejected_command}, "
                f"rejected_scenario={rejected_scenario}); increase max_env_steps or "
                "inspect command resets"
            )
        source = _obs_array(obs, observation_key)
        teacher_obs, _ = project_teacher_obs(
            source,
            projection=teacher_projection,
            expected_teacher_obs_dim=int(getattr(teacher_policy, "obs_dim", source.shape[1])),
        )
        student_obs = project_student_obs(
            source,
            projection=student_projection,
            expected_student_obs_dim=config.obs_dim,
            student_drop_index=student_drop_index,
        )
        command = _command_array(
            info,
            command_info_keys,
            expected_rows=num_envs,
            expected_dim=config.command_dim,
        )
        authoritative_teacher_obs, _ = project_teacher_obs(
            source,
            projection=teacher_projection,
            expected_teacher_obs_dim=int(getattr(teacher_policy, "obs_dim", source.shape[1])),
        )
        oracle_actions = _oracle_actions(
            teacher_policy,
            obs,
            info,
            authoritative_teacher_obs,
            action_dim=config.action_dim,
        )
        if collect_oracle_shadow:
            oracle_future, oracle_action_chunk, oracle_shadow_valid = _oracle_shadow_pair(
                env,
                teacher_policy=teacher_policy,
                initial_oracle_actions=oracle_actions,
                initial_command=command,
                config=config,
                observation_key=observation_key,
                teacher_projection=teacher_projection,
                student_projection=student_projection,
                student_drop_index=student_drop_index,
                command_info_keys=command_info_keys,
            )
        else:
            oracle_future = np.zeros(
                (num_envs, config.prediction_horizon, config.obs_dim), dtype=np.float32
            )
            oracle_action_chunk = np.zeros(
                (num_envs, config.prediction_horizon, config.action_dim), dtype=np.float32
            )
            oracle_shadow_valid = np.zeros((num_envs,), dtype=np.bool_)
        if walking_recovery:
            for index in range(num_envs):
                if window_count >= int(num_windows):
                    break
                if int(episode_timesteps[index]) < config.history_length and bool(
                    oracle_shadow_valid[index]
                ):
                    append_window(
                        _walking_recovery_window(
                            index=index,
                            observation_history=observation_history,
                            action_history=action_history,
                            command=command,
                            oracle_future=oracle_future,
                            oracle_action_chunk=oracle_action_chunk,
                            oracle_first_action=oracle_actions,
                            config=config,
                            planner_eligible=planner_eligible,
                        )
                    )
            if window_count >= int(num_windows):
                reporter.report(
                    windows=window_count,
                    env_steps=env_steps,
                    rejected_done=rejected_done,
                    rejected_command=rejected_command,
                    rejected_scenario=rejected_scenario,
                )
                break
        if rollout_teacher_policy is not None:
            actions = _oracle_actions(
                rollout_teacher_policy,
                obs,
                info,
                teacher_obs,
                action_dim=config.action_dim,
            )
        elif rollout_policy is not None:
            actions = _fada_actions(rollout_policy, observation_history, action_history, command)
        else:
            actions = oracle_actions
        previous_actions = action_history[:, -1].copy()
        state = env.step(actions)
        env_steps += 1
        done = _done_mask(state, num_envs=num_envs)
        next_obs, next_info = _next_after_done(env, state, done)
        next_student_obs = project_student_obs(
            _obs_array(next_obs, observation_key),
            projection=student_projection,
            expected_student_obs_dim=config.obs_dim,
            student_drop_index=student_drop_index,
        )

        # B3: 原子更新下一步 command observation, 产出可验证的 static/transition phase.
        if command_scenario == "static_stand":
            next_obs, next_info = set_transition_input_rows(
                env,
                command_info_key="commands",
                command_rows=np.zeros((num_envs, 3), dtype=np.float32),
            )
            next_student_obs = project_student_obs(
                _obs_array(next_obs, observation_key),
                projection=student_projection,
                expected_student_obs_dim=config.obs_dim,
                student_drop_index=student_drop_index,
            )
        elif command_scenario == "walk_to_stand":
            cycle_steps = transition_pre_steps + transition_post_steps
            transition_phase_step = (transition_phase_step + 1) % cycle_steps
            next_command = np.zeros((num_envs, 3), dtype=np.float32)
            if transition_phase_step < transition_pre_steps:
                next_command[:] = np.asarray(transition_walk_command, dtype=np.float32)
            next_obs, next_info = set_transition_input_rows(
                env,
                command_info_key="commands",
                command_rows=next_command,
            )
            next_student_obs = project_student_obs(
                _obs_array(next_obs, observation_key),
                projection=student_projection,
                expected_student_obs_dim=config.obs_dim,
                student_drop_index=student_drop_index,
            )

        # B4: 只从未跨 episode 且满足 scenario command 约束的 transition 产出 causal source batch.
        for index in range(num_envs):
            if bool(done[index]):
                if (
                    window_count < int(num_windows)
                    and planner_eligible
                    and not cold_start_windows
                    and int(episode_timesteps[index]) >= config.history_length - 1
                ):
                    append_window(
                        _terminal_planner_window(
                            observation_history=observation_history[index : index + 1],
                            action_history=action_history[index : index + 1],
                            command=command[index : index + 1],
                            oracle_future=oracle_future[index : index + 1],
                            oracle_action_chunk=oracle_action_chunk[index : index + 1],
                            oracle_first_action=oracle_actions[index : index + 1],
                            config=config,
                            command_scenario=command_scenario,
                            planner_eligible=planner_eligible,
                        )
                    )
                records[index].clear()
                episode_ids[index] += 1
                episode_timesteps[index] = 0
                rejected_done += 1
                continue
            if walking_recovery:
                episode_timesteps[index] += 1
                continue
            records[index].append(
                FADACollectionTransition(
                    observation=student_obs[index].copy(),
                    previous_action=previous_actions[index].copy(),
                    command=command[index].copy(),
                    executed_action=actions[index].copy(),
                    oracle_action=oracle_actions[index].copy(),
                    next_observation=next_student_obs[index].copy(),
                    episode_id=int(episode_ids[index]),
                    timestep=int(episode_timesteps[index]),
                    oracle_future=oracle_future[index].copy(),
                    oracle_action_chunk=oracle_action_chunk[index].copy(),
                    oracle_shadow_valid=bool(oracle_shadow_valid[index]),
                )
            )
            episode_timesteps[index] += 1
            if len(records[index]) == record_count:
                window = (
                    _cold_start_window_from_records(
                        tuple(records[index]),
                        config,
                        planner_eligible=planner_eligible,
                        idm_source_role=idm_source_role,
                    )
                    if cold_start_windows
                    else _window_from_records(
                        tuple(records[index]),
                        config,
                        command_scenario=command_scenario,
                        planner_eligible=planner_eligible,
                        idm_source_role=idm_source_role,
                    )
                )
                if window is None:
                    future_commands = (
                        tuple(records[index])
                        if cold_start_windows
                        else tuple(records[index])[
                            config.history_length - 1 : config.history_length
                            - 1
                            + config.prediction_horizon
                        ]
                    )
                    if any(
                        not np.array_equal(record.command, future_commands[0].command)
                        for record in future_commands[1:]
                    ):
                        rejected_command += 1
                    else:
                        rejected_scenario += 1
                else:
                    append_window(window)
                    if window_count >= int(num_windows):
                        break

        reporter.report(
            windows=window_count,
            env_steps=env_steps,
            rejected_done=rejected_done,
            rejected_command=rejected_command,
            rejected_scenario=rejected_scenario,
        )

        observation_history = np.roll(observation_history, shift=-1, axis=1)
        observation_history[:, -1] = next_student_obs
        action_history = np.roll(action_history, shift=-1, axis=1)
        action_history[:, -1] = actions
        if bool(np.any(done)):
            observation_history[done] = np.repeat(
                next_student_obs[done, None, :], config.history_length, axis=1
            )
            action_history[done] = 0.0
        obs, info = next_obs, next_info

    return FADACollectionResult(
        batch=_concat_batches([*compacted_batches, *pending_batches], config),
        env_steps=env_steps,
        rejected_done_transitions=rejected_done,
        rejected_command_windows=rejected_command,
        rollout_mode=rollout_mode,
        command_scenario=command_scenario,
        oracle_role="unified",
        rejected_scenario_windows=rejected_scenario,
        window_profile=("cold_start" if cold_start_windows else "steady_state"),
    )
