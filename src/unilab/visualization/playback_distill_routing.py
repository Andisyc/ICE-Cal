"""Pure command-intent routing decisions for distillation playback."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def _distill_student_obs_tensor(obs: Any, *, device: str | torch.device) -> torch.Tensor:
    if isinstance(obs, Mapping):
        if "obs" in obs:
            obs = obs["obs"]
        elif "actor" in obs:
            obs = obs["actor"]
    if isinstance(obs, torch.Tensor):
        return obs.to(device=device, dtype=torch.float32)
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


def distill_command_intents_from_commands(
    commands: Any,
    *,
    xy_threshold: float = 0.05,
    yaw_threshold: float = 0.05,
) -> tuple[str, ...]:
    command_array = np.asarray(commands, dtype=np.float32)
    if command_array.ndim == 1:
        command_array = command_array.reshape(1, -1)
    if command_array.ndim != 2 or command_array.shape[1] < 3:
        raise ValueError(
            "distill command intent requires commands with shape (N, >=3), "
            f"got {command_array.shape}"
        )
    if not np.isfinite(command_array[:, :3]).all():
        raise ValueError("distill command intent requires finite command values")
    xy_norm = np.linalg.norm(command_array[:, :2], axis=1)
    active = (xy_norm > float(xy_threshold)) | (np.abs(command_array[:, 2]) > float(yaw_threshold))
    return tuple("active" if bool(value) else "inactive" for value in active)


def _cfg_select(cfg: Any, dotted_path: str, default: Any = None) -> Any:
    current = cfg
    for key in dotted_path.split("."):
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current


def _distill_commands_from_env(env: Any, *, batch_size: int) -> np.ndarray | None:
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    if not isinstance(info, Mapping) or "commands" not in info:
        return None
    commands = np.asarray(info["commands"], dtype=np.float32)
    if commands.ndim == 1:
        commands = commands.reshape(1, -1)
    if commands.ndim != 2 or commands.shape[1] < 3:
        raise ValueError(
            "distill command routing requires env.state.info['commands'] with "
            f"shape (N, >=3), got {commands.shape}"
        )
    if commands.shape[0] == 1 and int(batch_size) > 1:
        commands = np.repeat(commands, int(batch_size), axis=0)
    if commands.shape[0] != int(batch_size):
        raise ValueError(
            "distill command routing command batch mismatch: "
            f"commands={commands.shape[0]} obs_batch={int(batch_size)}"
        )
    return commands[:, :3]


def _distill_command_intent_targets(
    cfg: Any,
    runtime_cfg: Mapping[str, Any],
) -> dict[str, int]:
    targets = runtime_cfg.get("command_intent_expert_targets")
    if not isinstance(targets, Mapping):
        targets = _cfg_select(cfg, "algo.command_intent_expert_targets", None)
    if not isinstance(targets, Mapping):
        targets = {"active": 0, "inactive": 1}
    resolved = {str(key): int(value) for key, value in targets.items()}
    missing = {"active", "inactive"} - set(resolved)
    if missing:
        raise ValueError(
            f"distill command routing requires command_intent_expert_targets for {sorted(missing)}"
        )
    return resolved


def _distill_effective_command_routing_mode(
    cfg: Any,
    runtime_cfg: Mapping[str, Any],
    *,
    is_moe: bool,
) -> tuple[str, str]:
    configured = str(_cfg_select(cfg, "interactive.distill_command_routing", "auto")).lower()
    if configured not in {"none", "auto", "hard", "bias"}:
        raise ValueError(
            "interactive.distill_command_routing must be one of "
            f"none, auto, hard, bias; got {configured!r}"
        )
    if not is_moe:
        return configured, "none"
    if configured == "auto":
        coef = float(runtime_cfg.get("command_intent_loss_coef") or 0.0)
        behavior_source = str(runtime_cfg.get("expert_behavior_loss_source") or "none")
        command_intent_trained = coef > 0.0 or behavior_source == "command_intent"
        return configured, "hard" if command_intent_trained else "none"
    return configured, configured


def _distill_expected_expert_tensor(
    intents: tuple[str, ...],
    targets: Mapping[str, int],
    *,
    num_experts: int,
    device: torch.device | str,
) -> torch.Tensor:
    indices = [int(targets[intent]) for intent in intents]
    if not indices:
        return torch.empty((0,), dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
    if int(target_tensor.min().item()) < 0 or int(target_tensor.max().item()) >= int(num_experts):
        raise ValueError(
            "distill command routing expert target out of range: "
            f"targets={sorted(set(indices))} num_experts={int(num_experts)}"
        )
    return target_tensor
