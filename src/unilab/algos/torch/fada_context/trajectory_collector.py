from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.trajectory_data import ContextTrajectoryDataset


@dataclass(frozen=True)
class PairedTrajectoryCollectionConfig:
    num_samples: int
    reference_horizon: int
    max_reset_batches: int = 32
    observation_key: str = "obs"
    command_key: str = "commands"
    initial_alignment_atol: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.num_samples <= 0 or self.reference_horizon <= 0 or self.max_reset_batches <= 0:
            raise ValueError("collection sizes must be positive")
        if self.initial_alignment_atol < 0.0:
            raise ValueError("initial_alignment_atol must be non-negative")


@dataclass(frozen=True)
class PairedTrajectoryCollectionResult:
    dataset: ContextTrajectoryDataset
    accepted_samples: int
    rejected_done_samples: int
    reset_batches: int


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
            raise ValueError(f"{name} shape mismatch")
        result |= value
    return result


def _policy_action(
    policy: FADAPlannerIDMPolicy,
    observation_history: np.ndarray,
    action_history: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    device = _module_device(policy)
    with torch.inference_mode():
        action = policy(
            torch.as_tensor(observation_history, device=device),
            torch.as_tensor(action_history, device=device),
            torch.as_tensor(command, device=device),
        ).action
    result = action.detach().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("FADA policy produced non-finite actions")
    return result


def _rollout(
    env: Any,
    policy: FADAPlannerIDMPolicy,
    *,
    initial_state: Any,
    total_steps: int,
    observation_key: str,
    command_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = int(env.num_envs)
    config = policy.config
    observation = _state_field(initial_state, "obs", observation_key)
    command = _state_field(initial_state, "info", command_key)
    if observation.shape != (rows, config.obs_dim):
        raise ValueError("environment observation dimension does not match FADA checkpoint")
    if command.shape != (rows, config.command_dim):
        raise ValueError("environment command dimension does not match FADA checkpoint")
    observation_history = np.repeat(observation[:, None, :], config.history_length, axis=1)
    action_history = np.zeros(
        (rows, config.history_length, config.action_dim), dtype=np.float32
    )
    states = [observation.copy()]
    actions: list[np.ndarray] = []
    invalid = np.zeros((rows,), dtype=np.bool_)
    for _ in range(total_steps):
        action = _policy_action(policy, observation_history, action_history, command)
        state = env.step(action)
        next_observation = _state_field(state, "obs", observation_key)
        next_command = _state_field(state, "info", command_key)
        invalid |= _done(state, rows)
        invalid |= np.any(next_command != command, axis=1)
        states.append(next_observation.copy())
        actions.append(action.copy())
        observation_history = np.concatenate(
            (observation_history[:, 1:], next_observation[:, None, :]), axis=1
        )
        action_history = np.concatenate((action_history[:, 1:], action[:, None, :]), axis=1)
    return (
        np.stack(states, axis=1),
        np.stack(actions, axis=1),
        observation_history,
        action_history,
        invalid,
    )


def collect_paired_context_trajectories(
    nominal_env: Any,
    fault_env: Any,
    policy: FADAPlannerIDMPolicy,
    spec: PairedTrajectoryCollectionConfig,
) -> PairedTrajectoryCollectionResult:
    """Collect same-start healthy references and zero-repair fault probes."""

    if int(nominal_env.num_envs) != int(fault_env.num_envs):
        raise ValueError("nominal and fault environments must have the same row count")
    for env in (nominal_env, fault_env):
        if not callable(getattr(env, "capture_rollout_snapshot", None)) or not callable(
            getattr(env, "restore_rollout_snapshot", None)
        ):
            raise TypeError("paired collection requires exact rollout snapshot support")
        set_autoreset = getattr(env, "set_autoreset", None)
        if not callable(set_autoreset):
            raise TypeError("paired collection requires set_autoreset")
        set_autoreset(False)

    config = policy.config
    total_steps = config.history_length + spec.reference_horizon
    fields: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "observation_history",
            "action_history",
            "command",
            "healthy_reference",
            "fault_state",
            "fault_action",
            "pair_id",
        )
    }
    accepted = 0
    rejected = 0
    reset_batches = 0
    while accepted < spec.num_samples and reset_batches < spec.max_reset_batches:
        nominal_initial = nominal_env.reset_all()
        snapshot = nominal_env.capture_rollout_snapshot()
        fault_env.restore_rollout_snapshot(snapshot)
        fault_initial = fault_env.state
        if fault_initial is None:
            raise RuntimeError("fault environment restore did not initialize state")
        nominal_obs = _state_field(nominal_initial, "obs", spec.observation_key)
        fault_obs = _state_field(fault_initial, "obs", spec.observation_key)
        nominal_command = _state_field(nominal_initial, "info", spec.command_key)
        fault_command = _state_field(fault_initial, "info", spec.command_key)
        if not np.allclose(
            nominal_obs, fault_obs, rtol=0.0, atol=spec.initial_alignment_atol
        ) or not np.array_equal(nominal_command, fault_command):
            raise ValueError("paired environments do not share the same initial observation/command")

        nominal_env.restore_rollout_snapshot(snapshot)
        healthy = _rollout(
            nominal_env,
            policy,
            initial_state=nominal_env.state,
            total_steps=total_steps,
            observation_key=spec.observation_key,
            command_key=spec.command_key,
        )
        fault_env.restore_rollout_snapshot(snapshot)
        fault = _rollout(
            fault_env,
            policy,
            initial_state=fault_env.state,
            total_steps=total_steps,
            observation_key=spec.observation_key,
            command_key=spec.command_key,
        )
        valid = ~(healthy[4] | fault[4])
        indices = np.flatnonzero(valid)
        rejected += int(valid.size - indices.size)
        remaining = spec.num_samples - accepted
        indices = indices[:remaining]
        if indices.size:
            anchor = config.history_length
            fault_states, fault_actions = fault[0], fault[1]
            fields["observation_history"].append(
                torch.from_numpy(fault_states[indices, 1 : anchor + 1])
            )
            fields["action_history"].append(torch.from_numpy(fault_actions[indices, :anchor]))
            fields["command"].append(torch.from_numpy(fault_command[indices]))
            fields["healthy_reference"].append(
                torch.from_numpy(healthy[0][indices, anchor + 1 :])
            )
            fields["fault_state"].append(torch.from_numpy(fault_states[indices]))
            fields["fault_action"].append(torch.from_numpy(fault_actions[indices]))
            fields["pair_id"].append(
                torch.arange(accepted, accepted + indices.size, dtype=torch.int64)
            )
            accepted += int(indices.size)
        reset_batches += 1
    if accepted != spec.num_samples:
        raise RuntimeError(
            f"paired collection exhausted reset budget: accepted={accepted} "
            f"requested={spec.num_samples} rejected_done={rejected}"
        )
    dataset = ContextTrajectoryDataset(
        **{name: torch.cat(values, dim=0) for name, values in fields.items()}
    )
    dataset.validate(config)
    return PairedTrajectoryCollectionResult(dataset, accepted, rejected, reset_batches)
