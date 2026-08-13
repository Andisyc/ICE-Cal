from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.support_query import (
    ContextQueryBatch,
    SupportContextBatch,
    SupportQueryBatch,
)


@dataclass(frozen=True)
class SupportQueryCollectionConfig:
    num_pairs: int
    support_length: int
    max_reset_pairs: int = 64
    observation_key: str = "obs"
    command_key: str = "commands"

    def __post_init__(self) -> None:
        if self.num_pairs <= 0 or self.support_length <= 0 or self.max_reset_pairs <= 0:
            raise ValueError("collection sizes must be positive")


@dataclass(frozen=True)
class SupportQueryCollectionResult:
    batch: SupportQueryBatch
    accepted_pairs: int
    rejected_pairs: int
    reset_pairs: int


@dataclass(frozen=True)
class _Transition:
    observation: np.ndarray
    previous_action: np.ndarray
    command: np.ndarray
    planner_intent: np.ndarray
    executed_action: np.ndarray
    next_observation: np.ndarray


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _state_field(state: Any, field: str, key: str) -> np.ndarray:
    carrier = getattr(state, field, None)
    if not isinstance(carrier, Mapping) or key not in carrier:
        available = sorted(carrier) if isinstance(carrier, Mapping) else []
        raise KeyError(f"state.{field}[{key!r}] missing; available={available}")
    value = np.asarray(carrier[key], dtype=np.float32)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"state.{field}[{key!r}] must be finite rank-2")
    return value


def _done(state: Any, rows: int) -> np.ndarray:
    result = np.zeros((rows,), dtype=np.bool_)
    for name in ("terminated", "truncated"):
        value = np.asarray(getattr(state, name), dtype=np.bool_).reshape(-1)
        if value.shape != result.shape:
            raise ValueError(f"{name} shape mismatch: expected={result.shape} observed={value.shape}")
        result |= value
    return result


def _policy_output(
    policy: FADAPlannerIDMPolicy,
    observation_history: np.ndarray,
    action_history: np.ndarray,
    command: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device = _module_device(policy)
    with torch.inference_mode():
        output = policy(
            torch.as_tensor(observation_history, device=device),
            torch.as_tensor(action_history, device=device),
            torch.as_tensor(command, device=device),
        )
    intent = output.predicted_future.detach().cpu().numpy().astype(np.float32)
    action = output.action.detach().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(intent)) or not np.all(np.isfinite(action)):
        raise ValueError("frozen Planner-IDM produced non-finite output")
    return intent, action


def _rollout(
    env: Any,
    policy: FADAPlannerIDMPolicy,
    initial_state: Any,
    *,
    steps: int,
    observation_key: str,
    command_key: str,
) -> tuple[list[_Transition], np.ndarray]:
    rows = int(env.num_envs)
    config = policy.config
    observation = _state_field(initial_state, "obs", observation_key)
    command = _state_field(initial_state, "info", command_key)
    if observation.shape != (rows, config.obs_dim):
        raise ValueError(
            "environment observation dimension does not match FADA checkpoint: "
            f"expected={(rows, config.obs_dim)} observed={observation.shape}"
        )
    if command.shape != (rows, config.command_dim):
        raise ValueError(
            "environment command dimension does not match FADA checkpoint: "
            f"expected={(rows, config.command_dim)} observed={command.shape}"
        )
    observation_history = np.repeat(observation[:, None, :], config.history_length, axis=1)
    action_history = np.zeros((rows, config.history_length, config.action_dim), dtype=np.float32)
    invalid = np.zeros((rows,), dtype=np.bool_)
    transitions: list[_Transition] = []
    for _ in range(steps):
        planner_intent, action = _policy_output(
            policy, observation_history, action_history, command
        )
        state = env.step(action)
        next_observation = _state_field(state, "obs", observation_key)
        next_command = _state_field(state, "info", command_key)
        invalid |= _done(state, rows)
        invalid |= np.any(next_command != command, axis=1)
        transitions.append(
            _Transition(
                observation=observation.copy(),
                previous_action=action_history[:, -1].copy(),
                command=command.copy(),
                planner_intent=planner_intent.copy(),
                executed_action=action.copy(),
                next_observation=next_observation.copy(),
            )
        )
        observation = next_observation
        observation_history = np.concatenate(
            (observation_history[:, 1:], observation[:, None, :]), axis=1
        )
        action_history = np.concatenate((action_history[:, 1:], action[:, None, :]), axis=1)
    return transitions, invalid


def _query_row(
    records: Sequence[_Transition],
    row: int,
    policy: FADAPlannerIDMPolicy,
) -> ContextQueryBatch:
    config = policy.config
    expected = config.history_length + config.prediction_horizon - 1
    if len(records) != expected:
        raise ValueError(f"Query record count mismatch: expected={expected} observed={len(records)}")
    anchor = config.history_length - 1
    history = records[: config.history_length]
    future = records[anchor : anchor + config.prediction_horizon]
    return ContextQueryBatch(
        observation_history=torch.from_numpy(
            np.stack([record.observation[row] for record in history])[None]
        ),
        action_history=torch.from_numpy(
            np.stack([record.previous_action[row] for record in history])[None]
        ),
        command=torch.from_numpy(future[0].command[row][None]),
        planner_intent=torch.from_numpy(future[0].planner_intent[row][None]),
        realized_future=torch.from_numpy(
            np.stack([record.next_observation[row] for record in future])[None]
        ),
        executed_action_chunk=torch.from_numpy(
            np.stack([record.executed_action[row] for record in future])[None]
        ),
    )


def _support_row(records: Sequence[_Transition], row: int) -> SupportContextBatch:
    return SupportContextBatch(
        target_future=torch.from_numpy(
            np.stack([record.planner_intent[row] for record in records])[None]
        ),
        realized_state=torch.from_numpy(
            np.stack([record.next_observation[row] for record in records])[None]
        ),
        executed_action=torch.from_numpy(
            np.stack([record.executed_action[row] for record in records])[None]
        ),
    )


def _concat_support(rows: Sequence[SupportContextBatch]) -> SupportContextBatch:
    return SupportContextBatch(
        target_future=torch.cat([row.target_future for row in rows], dim=0),
        realized_state=torch.cat([row.realized_state for row in rows], dim=0),
        executed_action=torch.cat([row.executed_action for row in rows], dim=0),
    )


def _concat_query(rows: Sequence[ContextQueryBatch]) -> ContextQueryBatch:
    names = (
        "observation_history",
        "action_history",
        "command",
        "planner_intent",
        "realized_future",
        "executed_action_chunk",
    )
    return ContextQueryBatch(
        **{
            name: torch.cat([getattr(row, name) for row in rows], dim=0)
            for name in names
        }
    )


def collect_support_query_pairs(
    fault_env: Any,
    policy: FADAPlannerIDMPolicy,
    spec: SupportQueryCollectionConfig,
) -> SupportQueryCollectionResult:
    """Collect independent no-Context Support and Query rollouts in one fault environment."""

    set_autoreset = getattr(fault_env, "set_autoreset", None)
    if not callable(set_autoreset):
        raise TypeError("Support-Query collection requires set_autoreset")
    set_autoreset(False)
    config = policy.config
    query_steps = config.history_length + config.prediction_horizon - 1
    supports: list[SupportContextBatch] = []
    queries: list[ContextQueryBatch] = []
    support_commands: list[torch.Tensor] = []
    pair_ids: list[int] = []
    support_rollout_ids: list[int] = []
    query_rollout_ids: list[int] = []
    rejected = 0
    reset_pairs = 0
    while len(pair_ids) < spec.num_pairs and reset_pairs < spec.max_reset_pairs:
        support_initial = fault_env.reset_all()
        support_records, support_invalid = _rollout(
            fault_env,
            policy,
            support_initial,
            steps=spec.support_length,
            observation_key=spec.observation_key,
            command_key=spec.command_key,
        )
        query_initial = fault_env.reset_all()
        query_records, query_invalid = _rollout(
            fault_env,
            policy,
            query_initial,
            steps=query_steps,
            observation_key=spec.observation_key,
            command_key=spec.command_key,
        )
        rows = int(fault_env.num_envs)
        support_command = support_records[0].command
        query_command = query_records[0].command
        valid = ~(support_invalid | query_invalid)
        valid &= np.all(support_command == query_command, axis=1)
        indices = np.flatnonzero(valid)
        rejected += int(rows - indices.size)
        support_rollout_id = 2 * reset_pairs
        query_rollout_id = support_rollout_id + 1
        for row in indices:
            if len(pair_ids) >= spec.num_pairs:
                break
            supports.append(_support_row(support_records, int(row)))
            queries.append(_query_row(query_records, int(row), policy))
            support_commands.append(torch.from_numpy(support_command[row][None]))
            pair_ids.append(len(pair_ids))
            support_rollout_ids.append(support_rollout_id)
            query_rollout_ids.append(query_rollout_id)
        reset_pairs += 1
    if len(pair_ids) != spec.num_pairs:
        raise RuntimeError(
            "Support-Query collection exhausted reset budget: "
            f"accepted={len(pair_ids)} requested={spec.num_pairs} rejected={rejected}"
        )
    batch = SupportQueryBatch(
        support=_concat_support(supports),
        query=_concat_query(queries),
        support_command=torch.cat(support_commands, dim=0),
        pair_id=torch.tensor(pair_ids, dtype=torch.int64),
        support_rollout_id=torch.tensor(support_rollout_ids, dtype=torch.int64),
        query_rollout_id=torch.tensor(query_rollout_ids, dtype=torch.int64),
    ).validate(config, support_length=spec.support_length)
    return SupportQueryCollectionResult(
        batch=batch,
        accepted_pairs=len(pair_ids),
        rejected_pairs=rejected,
        reset_pairs=reset_pairs,
    )
