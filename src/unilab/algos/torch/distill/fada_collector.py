from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ContextManager, Literal, cast

import numpy as np
import torch

from .collector import project_student_obs, project_teacher_obs, set_transition_input_rows
from .fada import (
    FADA_SCENARIO_IDS,
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
    FADASourceBatch,
)


@dataclass(frozen=True)
class FADACollectionResult:
    batch: FADASourceBatch
    env_steps: int
    rejected_done_transitions: int
    rejected_command_windows: int
    rollout_mode: str
    command_scenario: str = "walk"
    oracle_role: str = "walking"
    rejected_scenario_windows: int = 0
    window_profile: str = "steady_state"


@dataclass(frozen=True)
class FADACollectionSpec:
    """Validated collection semantics that travel together across FADA callers."""

    observation_key: str = "obs"
    teacher_projection: str = "identity"
    student_projection: str = "identity"
    student_drop_index: int | None = None
    command_info_keys: tuple[str, ...] = ("commands",)
    max_env_steps: int | None = None
    collect_oracle_shadow: bool = False
    command_scenario: Literal["walk", "static_stand", "walk_to_stand"] = "walk"
    transition_walk_command: tuple[float, ...] = (0.4, 0.0, 0.0)
    transition_pre_switch_steps: int | None = None
    transition_post_switch_steps: int | None = None
    planner_eligible: bool = True
    cold_start_windows: bool = False


@dataclass(frozen=True)
class _Transition:
    observation: np.ndarray
    previous_action: np.ndarray
    command: np.ndarray
    executed_action: np.ndarray
    oracle_action: np.ndarray
    next_observation: np.ndarray
    oracle_future: np.ndarray
    oracle_action_chunk: np.ndarray
    oracle_shadow_valid: bool


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _obs_array(obs: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"observation key {key!r} not found; available={sorted(obs)}")
    value = np.asarray(obs[key], dtype=np.float32)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"observation {key!r} must be finite rank-2, got {value.shape}")
    return value


def _command_array(
    info: Mapping[str, Any],
    keys: Sequence[str],
    *,
    expected_rows: int,
    expected_dim: int,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for key in keys:
        if key not in info:
            raise KeyError(f"command info key {key!r} not found; available={sorted(info)}")
        value = np.asarray(info[key], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[0] != int(expected_rows):
            raise ValueError(
                f"command info key {key!r} must have {expected_rows} rows, got {value.shape}"
            )
        columns.append(value)
    command = np.concatenate(columns, axis=1)
    if command.shape != (int(expected_rows), int(expected_dim)):
        raise ValueError(
            f"complete command shape mismatch: expected {(expected_rows, expected_dim)}, "
            f"got {command.shape} from keys={list(keys)}"
        )
    if not np.all(np.isfinite(command)):
        raise ValueError("complete command must contain only finite values")
    return command


def _policy_actions(policy: torch.nn.Module, obs: np.ndarray, *, action_dim: int) -> np.ndarray:
    tensor = torch.as_tensor(obs, dtype=torch.float32, device=_module_device(policy))
    with torch.inference_mode():
        output = policy(tensor)
    if isinstance(output, tuple):
        output = output[0]
    action = torch.as_tensor(output).detach()
    if action.shape != (obs.shape[0], int(action_dim)):
        raise ValueError(
            f"Oracle action shape mismatch: expected {(obs.shape[0], action_dim)}, "
            f"got {tuple(action.shape)}"
        )
    result = action.cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("Oracle produced non-finite actions")
    return result


def _fada_actions(
    policy: FADAPlannerIDMPolicy,
    observation_history: np.ndarray,
    action_history: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    device = _module_device(policy)
    with torch.inference_mode():
        output = policy(
            torch.as_tensor(observation_history, dtype=torch.float32, device=device),
            torch.as_tensor(action_history, dtype=torch.float32, device=device),
            torch.as_tensor(command, dtype=torch.float32, device=device),
        )
    result = output.action.detach().cpu().numpy().astype(np.float32)
    if result.shape != (observation_history.shape[0], policy.config.action_dim):
        raise ValueError(f"FADA rollout action shape mismatch: got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("FADA rollout produced non-finite actions")
    return result


def _done_mask(state: Any, *, num_envs: int) -> np.ndarray:
    done = np.zeros((int(num_envs),), dtype=np.bool_)
    for value in (getattr(state, "terminated", None), getattr(state, "truncated", None)):
        if value is None:
            continue
        current = np.asarray(value, dtype=np.bool_).reshape(-1)
        if current.shape != done.shape:
            raise ValueError(
                f"done mask shape mismatch: expected {done.shape}, got {current.shape}"
            )
        done |= current
    return done


def _next_after_done(
    env: Any,
    state: Any,
    done: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    obs = {key: np.asarray(value).copy() for key, value in state.obs.items()}
    info = dict(getattr(state, "info", {}))
    if not bool(np.any(done)):
        return obs, info
    final_observation = getattr(state, "final_observation", None)
    if isinstance(final_observation, Mapping) or isinstance(info.get("final_observation"), Mapping):
        return obs, info
    indices = np.flatnonzero(done).astype(np.int32)
    reset_obs, reset_info = env.reset(indices)
    for key, value in reset_obs.items():
        obs[key][indices] = value
    for key, value in reset_info.items():
        if isinstance(value, np.ndarray):
            if key not in info:
                info[key] = np.zeros((len(done),) + value.shape[1:], dtype=value.dtype)
            info[key][indices] = value
        else:
            info[key] = value
    return obs, info


def _oracle_shadow_pair(
    env: Any,
    *,
    teacher_policy: torch.nn.Module,
    initial_oracle_actions: np.ndarray,
    initial_command: np.ndarray,
    config: FADAArchitectureConfig,
    observation_key: str,
    teacher_projection: str,
    student_projection: str,
    student_drop_index: int | None,
    command_info_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll final Oracle for K steps and restore the exact visited-state snapshot."""

    preserve = getattr(env, "preserve_rollout_state", None)
    if not callable(preserve):
        raise TypeError(
            "Oracle-shadow collection requires env.preserve_rollout_state() exact snapshot support"
        )
    futures: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    valid = np.ones((int(env.num_envs),), dtype=np.bool_)
    oracle_actions = initial_oracle_actions

    # B1: shadow branch 从正式 visited state 出发, success/exception 均由 env transaction 恢复.
    preserve_context = cast(Callable[[], ContextManager[None]], preserve)
    with preserve_context():
        for offset in range(config.prediction_horizon):
            actions.append(oracle_actions.copy())
            shadow_state = env.step(oracle_actions)
            valid &= ~_done_mask(shadow_state, num_envs=int(env.num_envs))
            shadow_obs = {key: np.asarray(value) for key, value in shadow_state.obs.items()}
            future = project_student_obs(
                _obs_array(shadow_obs, observation_key),
                projection=student_projection,
                expected_student_obs_dim=config.obs_dim,
                student_drop_index=student_drop_index,
            )
            futures.append(future.copy())
            shadow_info = dict(shadow_state.info)
            shadow_command = _command_array(
                shadow_info,
                command_info_keys,
                expected_rows=int(env.num_envs),
                expected_dim=config.command_dim,
            )
            valid &= np.all(shadow_command == initial_command, axis=1)
            if offset + 1 < config.prediction_horizon:
                teacher_obs, _ = project_teacher_obs(
                    _obs_array(shadow_obs, observation_key),
                    projection=teacher_projection,
                    expected_teacher_obs_dim=int(
                        getattr(teacher_policy, "obs_dim", future.shape[1])
                    ),
                )
                oracle_actions = _policy_actions(
                    teacher_policy,
                    teacher_obs,
                    action_dim=config.action_dim,
                )

    # B2: 产出 row-aligned causal pair; invalid shadow rows 不进入 IDM loss.
    return np.stack(futures, axis=1), np.stack(actions, axis=1), valid


def _window_from_records(
    records: Sequence[_Transition],
    config: FADAArchitectureConfig,
    *,
    command_scenario: str,
    planner_eligible: bool,
) -> FADASourceBatch | None:
    expected = config.history_length + config.prediction_horizon - 1
    if len(records) != expected:
        raise ValueError(f"window record count mismatch: expected {expected}, got {len(records)}")
    anchor = config.history_length - 1
    future_records = records[anchor : anchor + config.prediction_horizon]
    command = future_records[0].command
    if any(not np.array_equal(record.command, command) for record in future_records[1:]):
        return None
    history_records = records[: config.history_length]
    if command_scenario == "static_stand":
        if bool(np.any(np.abs(command) > 1.0e-6)):
            return None
    elif command_scenario == "walk_to_stand":
        future_is_standing = not bool(np.any(np.abs(command) > 1.0e-6))
        history_has_walking = any(
            bool(np.any(np.abs(record.command) > 1.0e-6)) for record in history_records[:-1]
        )
        if not future_is_standing or not history_has_walking:
            return None
    return FADASourceBatch(
        observation_history=torch.from_numpy(
            np.stack([record.observation for record in history_records])[None]
        ),
        action_history=torch.from_numpy(
            np.stack([record.previous_action for record in history_records])[None]
        ),
        command=torch.from_numpy(command[None]),
        realized_future=torch.from_numpy(
            np.stack([record.next_observation for record in future_records])[None]
        ),
        executed_action_chunk=torch.from_numpy(
            np.stack([record.executed_action for record in future_records])[None]
        ),
        oracle_future=torch.from_numpy(future_records[0].oracle_future[None]),
        oracle_action_chunk=torch.from_numpy(future_records[0].oracle_action_chunk[None]),
        oracle_shadow_valid=torch.tensor([future_records[0].oracle_shadow_valid], dtype=torch.bool),
        oracle_first_action=torch.from_numpy(future_records[0].oracle_action[None]),
        command_scenario=torch.tensor([FADA_SCENARIO_IDS[command_scenario]], dtype=torch.int64),
        planner_eligible=torch.tensor([planner_eligible], dtype=torch.bool),
        cold_start=torch.zeros((1,), dtype=torch.bool),
    ).validate(config)


def _cold_start_window_from_records(
    records: Sequence[_Transition],
    config: FADAArchitectureConfig,
    *,
    planner_eligible: bool,
) -> FADASourceBatch | None:
    if len(records) != config.prediction_horizon:
        raise ValueError(
            "cold-start record count mismatch: "
            f"expected {config.prediction_horizon}, got {len(records)}"
        )
    command = records[0].command
    if any(not np.array_equal(record.command, command) for record in records[1:]):
        return None
    if bool(np.any(np.abs(command) > 1.0e-6)):
        return None
    reset_observation = torch.from_numpy(records[0].observation)
    return FADASourceBatch(
        observation_history=reset_observation[None, None, :].repeat(1, config.history_length, 1),
        action_history=torch.zeros(
            (1, config.history_length, config.action_dim), dtype=torch.float32
        ),
        command=torch.from_numpy(command[None]),
        realized_future=torch.from_numpy(
            np.stack([record.next_observation for record in records])[None]
        ),
        executed_action_chunk=torch.from_numpy(
            np.stack([record.executed_action for record in records])[None]
        ),
        oracle_future=torch.from_numpy(records[0].oracle_future[None]),
        oracle_action_chunk=torch.from_numpy(records[0].oracle_action_chunk[None]),
        oracle_shadow_valid=torch.tensor([records[0].oracle_shadow_valid], dtype=torch.bool),
        oracle_first_action=torch.from_numpy(records[0].oracle_action[None]),
        command_scenario=torch.tensor([FADA_SCENARIO_IDS["static_stand"]], dtype=torch.int64),
        planner_eligible=torch.tensor([planner_eligible], dtype=torch.bool),
        cold_start=torch.ones((1,), dtype=torch.bool),
    ).validate(config)


def _concat_batches(
    batches: Sequence[FADASourceBatch], config: FADAArchitectureConfig
) -> FADASourceBatch:
    return FADASourceBatch(
        **{
            field: torch.cat([getattr(batch, field) for batch in batches], dim=0)
            for field in FADASourceBatch.__dataclass_fields__
        }
    ).validate(config)


def collect_fada_source_windows(
    env: Any,
    *,
    teacher_policy: torch.nn.Module,
    config: FADAArchitectureConfig,
    num_windows: int,
    rollout_policy: FADAPlannerIDMPolicy | None = None,
    rollout_teacher_policy: torch.nn.Module | None = None,
    standing_teacher_policy: torch.nn.Module | None = None,
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
    if int(num_windows) <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")
    if rollout_policy is not None and rollout_teacher_policy is not None:
        raise ValueError("set only one of rollout_policy and rollout_teacher_policy")
    if command_scenario not in {"walk", "static_stand", "walk_to_stand"}:
        raise ValueError(f"unsupported FADA command_scenario: {command_scenario!r}")
    if command_scenario != "walk" and rollout_teacher_policy is not None:
        raise ValueError("intermediate Oracle rollouts support only the walk scenario")
    if command_scenario != "walk" and standing_teacher_policy is None:
        raise ValueError(f"{command_scenario} requires standing_teacher_policy")
    if cold_start_windows and command_scenario != "static_stand":
        raise ValueError("cold_start_windows requires command_scenario='static_stand'")
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
    records: list[deque[_Transition]] = [deque(maxlen=record_count) for _ in range(num_envs)]
    batches: list[FADASourceBatch] = []
    rejected_done = 0
    rejected_command = 0
    rejected_scenario = 0
    env_steps = 0
    step_limit = (
        int(max_env_steps)
        if max_env_steps is not None
        else max(record_count + int(np.ceil(num_windows / max(num_envs, 1))) * 10, 1)
    )

    # B2: 每步先查询同状态 Oracle label, 再执行 Oracle bootstrap 或当前 Planner-IDM 动作.
    while len(batches) < int(num_windows):
        if env_steps >= step_limit:
            raise RuntimeError(
                f"FADA collector produced {len(batches)}/{num_windows} windows after "
                f"{env_steps} env steps; increase max_env_steps or inspect command resets"
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
        standing_phase = command_scenario == "static_stand" or (
            command_scenario == "walk_to_stand" and transition_phase_step >= transition_pre_steps
        )
        authoritative_teacher = standing_teacher_policy if standing_phase else teacher_policy
        if authoritative_teacher is None:
            raise RuntimeError("standing curriculum teacher authority was not materialized")
        authoritative_teacher_obs, _ = project_teacher_obs(
            source,
            projection=teacher_projection,
            expected_teacher_obs_dim=int(
                getattr(authoritative_teacher, "obs_dim", source.shape[1])
            ),
        )
        oracle_actions = _policy_actions(
            authoritative_teacher,
            authoritative_teacher_obs,
            action_dim=config.action_dim,
        )
        if collect_oracle_shadow:
            oracle_future, oracle_action_chunk, oracle_shadow_valid = _oracle_shadow_pair(
                env,
                teacher_policy=authoritative_teacher,
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
        if rollout_teacher_policy is not None:
            actions = _policy_actions(
                rollout_teacher_policy,
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
                records[index].clear()
                rejected_done += 1
                continue
            records[index].append(
                _Transition(
                    observation=student_obs[index].copy(),
                    previous_action=previous_actions[index].copy(),
                    command=command[index].copy(),
                    executed_action=actions[index].copy(),
                    oracle_action=oracle_actions[index].copy(),
                    next_observation=next_student_obs[index].copy(),
                    oracle_future=oracle_future[index].copy(),
                    oracle_action_chunk=oracle_action_chunk[index].copy(),
                    oracle_shadow_valid=bool(oracle_shadow_valid[index]),
                )
            )
            if len(records[index]) == record_count:
                window = (
                    _cold_start_window_from_records(
                        tuple(records[index]),
                        config,
                        planner_eligible=planner_eligible,
                    )
                    if cold_start_windows
                    else _window_from_records(
                        tuple(records[index]),
                        config,
                        command_scenario=command_scenario,
                        planner_eligible=planner_eligible,
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
                    batches.append(window)
                    if len(batches) >= int(num_windows):
                        break

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
        batch=_concat_batches(batches[: int(num_windows)], config),
        env_steps=env_steps,
        rejected_done_transitions=rejected_done,
        rejected_command_windows=rejected_command,
        rollout_mode=(
            "intermediate_oracle"
            if rollout_teacher_policy is not None
            else ("oracle" if rollout_policy is None else "planner_idm")
        ),
        command_scenario=command_scenario,
        oracle_role=("standing" if command_scenario != "walk" else "walking"),
        rejected_scenario_windows=rejected_scenario,
        window_profile=("cold_start" if cold_start_windows else "steady_state"),
    )
