"""Read-only diagnostics for interactive distillation playback."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .playback_sessions import KeyboardCommander


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return max(parsed, 1)


def _trace_obs_tensor(obs: Any, *, device: str | torch.device) -> torch.Tensor:
    if isinstance(obs, Mapping):
        if "obs" in obs:
            obs = obs["obs"]
        elif "actor" in obs:
            obs = obs["actor"]
    if isinstance(obs, torch.Tensor):
        return obs.to(device=device, dtype=torch.float32)
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


def _first_env_row(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        return arr.reshape(1)
    return np.asarray(arr[0]).reshape(-1)


def _format_trace_vector(value: np.ndarray | None, *, max_items: int = 6) -> str:
    if value is None:
        return "None"
    head = np.asarray(value, dtype=np.float64).reshape(-1)[:max_items]
    return "[" + ",".join(f"{float(item):+.3f}" for item in head) + "]"


def _trace_abs_max(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().abs().max().cpu().item())
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return float(np.max(np.abs(arr)))


def _format_trace_shape(shape: Sequence[int]) -> str:
    return "x".join(str(int(dim)) for dim in shape)


def _trace_tensor_stats(prefix: str, value: torch.Tensor) -> list[str]:
    tensor = value.detach().float()
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().cpu().item())
    total_count = int(tensor.numel())
    parts = [
        f"{prefix}_shape={_format_trace_shape(tuple(tensor.shape))}",
        f"{prefix}_finite={finite_count}/{total_count}",
    ]
    if finite_count == 0:
        parts.extend(
            [
                f"{prefix}_mean=None",
                f"{prefix}_std=None",
                f"{prefix}_min=None",
                f"{prefix}_max=None",
                f"{prefix}_head=None",
            ]
        )
        return parts

    finite_values = tensor[finite]
    head = tensor.reshape(-1)[:6].detach().cpu().numpy()
    parts.extend(
        [
            f"{prefix}_mean={float(finite_values.mean().cpu().item()):.6g}",
            f"{prefix}_std={float(finite_values.std(unbiased=False).cpu().item()):.6g}",
            f"{prefix}_min={float(finite_values.min().cpu().item()):.6g}",
            f"{prefix}_max={float(finite_values.max().cpu().item()):.6g}",
            f"{prefix}_head={_format_trace_vector(head, max_items=6)}",
        ]
    )
    return parts


def _trace_checkpoint_diagnostics(policy_fn: Any) -> list[str]:
    if policy_fn is None:
        return []
    normalizer_present = getattr(policy_fn, "_unilab_distill_obs_normalizer_present", None)
    normalizer_status = (
        "unknown"
        if normalizer_present is None
        else ("present" if bool(normalizer_present) else "absent")
    )
    parts = [f"checkpoint_obs_normalizer={normalizer_status}"]
    agent_steps = getattr(policy_fn, "_unilab_distill_agent_steps", None)
    if agent_steps is not None:
        parts.append(f"checkpoint_agent_steps={int(agent_steps)}")
    runtime_cfg = getattr(policy_fn, "_unilab_distill_runtime_cfg", None)
    if isinstance(runtime_cfg, Mapping) and runtime_cfg.get("student_obs_dim") is not None:
        parts.append(f"checkpoint_student_obs_dim={int(runtime_cfg['student_obs_dim'])}")
    return parts


def _trace_first_done(info: Mapping[str, Any]) -> bool | None:
    for key in ("done", "dones", "terminated", "terminations", "truncated", "timeouts"):
        if key not in info:
            continue
        row = _first_env_row(info.get(key))
        if row is not None:
            return bool(np.asarray(row).reshape(-1)[0])
    return None


def _trace_first_command(env: Any, commander: KeyboardCommander | None) -> np.ndarray | None:
    if commander is not None:
        return np.asarray(commander.command, dtype=np.float64).reshape(3)
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    if isinstance(info, Mapping):
        return _first_env_row(info.get("commands"))
    return None


def _load_trace_standing_teacher(
    cfg: DictConfig | None,
    *,
    device: str,
    log: Callable[[str], None],
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    checkpoint = os.environ.get("UNILAB_G1_STANDING_TEACHER_CHECKPOINT")
    if checkpoint is None:
        checkpoint = os.environ.get("UNILAB_G1_DISTILL_STANDING_TEACHER_CHECKPOINT")
    if checkpoint is None or checkpoint.strip() == "":
        return None
    if cfg is None:
        log("standing teacher trace disabled: no Hydra cfg available")
        return None
    from unilab.algos.torch.distill import DistillationTeacherSpec, load_sac_teacher_policy

    spec = DistillationTeacherSpec(
        algo_type=str(OmegaConf.select(cfg, "teacher.algo_type", default="sac")),
        obs_dim=int(OmegaConf.select(cfg, "teacher.obs_dim")),
        action_dim=int(OmegaConf.select(cfg, "teacher.action_dim")),
        actor_hidden_dim=int(OmegaConf.select(cfg, "teacher.actor_hidden_dim", default=512)),
        use_layer_norm=bool(OmegaConf.select(cfg, "teacher.use_layer_norm", default=True)),
        obs_normalization=bool(OmegaConf.select(cfg, "teacher.obs_normalization", default=True)),
    )
    teacher = load_sac_teacher_policy(checkpoint, spec, device=device)
    log(f"Standing teacher trace enabled: {checkpoint}")
    return teacher


def _print_distill_action_trace(
    playback_session: Any,
    *,
    env: Any,
    commander: KeyboardCommander | None,
    base_height: float,
    standing_teacher: Callable[[torch.Tensor], torch.Tensor] | None,
) -> None:
    action_obs = getattr(playback_session, "action_obs", None)
    actions = getattr(playback_session, "actions", None)
    policy_fn = getattr(playback_session, "policy", None)
    student_policy = getattr(policy_fn, "_unilab_distill_student_policy", None)
    device = str(
        getattr(policy_fn, "_unilab_distill_device", getattr(playback_session, "device", "cpu"))
    )
    info = playback_session.info
    command = _trace_first_command(env, commander)

    parts = [
        f"step={int(getattr(playback_session, 'step_count', -1))}",
        f"cmd={_format_trace_vector(command, max_items=3)}",
        f"base_z={base_height:.3f}",
        f"done={_trace_first_done(info)}",
        f"student_action_abs_max={_trace_abs_max(actions)}",
    ]
    parts.extend(_trace_checkpoint_diagnostics(policy_fn))
    routing_mode = getattr(policy_fn, "_unilab_distill_command_routing_mode", None)
    if routing_mode is not None:
        parts.append(f"routing_mode={routing_mode}")
        routing_applied = getattr(policy_fn, "_unilab_distill_command_routing_applied", None)
        if routing_applied is not None:
            parts.append(f"routing_applied={bool(routing_applied)}")
        command_intents = tuple(
            getattr(policy_fn, "_unilab_distill_last_command_intents", ()) or ()
        )
        expected_experts = tuple(
            getattr(policy_fn, "_unilab_distill_last_expected_experts", ()) or ()
        )
        selected_experts = tuple(
            getattr(policy_fn, "_unilab_distill_last_selected_experts", ()) or ()
        )
        raw_selected_experts = tuple(
            getattr(policy_fn, "_unilab_distill_last_raw_selected_experts", ()) or ()
        )
        if command_intents:
            parts.append(f"expected_intent={command_intents[0]}")
        if expected_experts:
            parts.append(f"expected_expert={int(expected_experts[0])}")
        if selected_experts:
            parts.append(f"selected_expert={int(selected_experts[0])}")
        if raw_selected_experts:
            parts.append(f"raw_selected_expert={int(raw_selected_experts[0])}")
    if student_policy is None or action_obs is None:
        print("[play_interactive][distill-trace] " + " ".join(parts))
        return

    from unilab.algos.torch.distill import MoEStudentOutput, MoEStudentPolicy

    obs_tensor = _trace_obs_tensor(action_obs, device=device)
    parts.extend(_trace_tensor_stats("student_obs", obs_tensor))
    student_action = _trace_obs_tensor(actions, device=device) if actions is not None else None
    with torch.no_grad():
        if isinstance(student_policy, MoEStudentPolicy):
            output = student_policy(obs_tensor, return_diagnostics=True)
        else:
            output = student_policy(obs_tensor)
    if isinstance(output, MoEStudentOutput):
        routed_route_probs = getattr(policy_fn, "_unilab_distill_last_route_probs", None)
        if routed_route_probs is not None:
            route_probs_array = np.asarray(routed_route_probs)
            route_probs = route_probs_array[0] if route_probs_array.ndim > 1 else route_probs_array
        else:
            route_probs = output.route_probs[0].detach().cpu().numpy()
        if student_action is None:
            student_action = output.action
        parts.extend(
            [
                f"route_probs={_format_trace_vector(route_probs, max_items=student_policy.num_experts)}",
            ]
        )
    else:
        if student_action is None:
            student_action = output

    if standing_teacher is not None:
        with torch.no_grad():
            teacher_action = standing_teacher(obs_tensor)
        target = teacher_action.to(device=student_action.device, dtype=student_action.dtype)
        mse = float((student_action - target).square().mean().detach().cpu().item())
        parts.extend(
            [
                f"standing_teacher_action_abs_max={_trace_abs_max(target)}",
                f"student_vs_standing_teacher_mse={mse:.6f}",
            ]
        )
    print("[play_interactive][distill-trace] " + " ".join(parts))
