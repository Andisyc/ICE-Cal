"""Checkpoint-backed distillation policy owner for interactive playback."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from .playback_distill_routing import (
    _cfg_select,
    _distill_command_intent_targets,
    _distill_commands_from_env,
    _distill_effective_command_routing_mode,
    _distill_expected_expert_tensor,
    _distill_student_obs_tensor,
    distill_command_intents_from_commands,
)


def _initialize_policy_diagnostics(
    policy: Callable[[Any], Any],
    *,
    student: Any,
    checkpoint: Path,
    device: str,
    runtime_cfg: Mapping[str, Any],
    obs_normalizer_keys: tuple[str, ...],
    routing_config_mode: str,
    routing_mode: str,
    routing_targets: Mapping[str, int],
) -> None:
    values = {
        "_unilab_distill_student_policy": student.policy,
        "_unilab_distill_device": device,
        "_unilab_distill_checkpoint_path": str(checkpoint),
        "_unilab_distill_agent_steps": int(student.agent_steps),
        "_unilab_distill_runtime_cfg": dict(runtime_cfg),
        "_unilab_distill_obs_normalizer_present": bool(obs_normalizer_keys),
        "_unilab_distill_obs_normalizer_keys": obs_normalizer_keys,
        "_unilab_distill_command_routing_mode": routing_mode,
        "_unilab_distill_command_routing_config_mode": routing_config_mode,
        "_unilab_distill_command_routing_targets": dict(routing_targets),
        "_unilab_distill_command_routing_applied": False,
        "_unilab_distill_last_command_intents": (),
        "_unilab_distill_last_expected_experts": (),
        "_unilab_distill_last_selected_experts": (),
        "_unilab_distill_last_route_probs": None,
        "_unilab_distill_last_raw_route_probs": None,
    }
    for name, value in values.items():
        setattr(policy, name, value)


def load_distill_playback_policy(
    *,
    checkpoint: str | Path,
    cfg: Any,
    env: Any,
    device: str,
    load_student_policy: Callable[..., Any],
    log: Callable[[str], None],
) -> Callable[[Any], Any]:
    """Load one student checkpoint and return its diagnostic playback callable."""

    checkpoint_path = Path(checkpoint)
    student = load_student_policy(checkpoint_path, device=device)
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalizer_state = (
        raw_checkpoint.get("obs_normalizer") if isinstance(raw_checkpoint, Mapping) else None
    )
    normalizer_keys = (
        tuple(str(key) for key in normalizer_state)
        if isinstance(normalizer_state, Mapping)
        else ()
    )
    runtime_cfg = dict(student.distill_runtime_cfg)
    model_type = str(runtime_cfg.get("student_model_type", "mlp"))
    is_moe = model_type == "moe" and hasattr(student.policy, "experts")
    routing_config_mode, routing_mode = _distill_effective_command_routing_mode(
        cfg, runtime_cfg, is_moe=is_moe
    )
    routing_targets = _distill_command_intent_targets(cfg, runtime_cfg)
    xy_threshold = float(_cfg_select(cfg, "interactive.distill_command_xy_threshold", 0.05))
    yaw_threshold = float(_cfg_select(cfg, "interactive.distill_command_yaw_threshold", 0.05))
    routing_bias = float(_cfg_select(cfg, "interactive.distill_command_routing_bias", 10.0))

    def policy(obs: Any) -> Any:
        obs_tensor = _distill_student_obs_tensor(obs, device=device)
        with torch.no_grad():
            if not is_moe:
                action = student.policy(obs_tensor).detach()
                setattr(policy, "_unilab_distill_command_routing_applied", False)
                setattr(policy, "_unilab_distill_last_command_intents", ())
                setattr(policy, "_unilab_distill_last_expected_experts", ())
                setattr(policy, "_unilab_distill_last_selected_experts", ())
                setattr(policy, "_unilab_distill_last_route_probs", None)
                setattr(policy, "_unilab_distill_last_raw_route_probs", None)
                return action

            output = student.policy(obs_tensor, return_diagnostics=True)
            raw_route_probs = output.route_probs
            route_probs = raw_route_probs
            raw_selected = torch.argmax(raw_route_probs, dim=-1)
            selected = raw_selected
            action = output.action
            intents: tuple[str, ...] = ()
            expected: torch.Tensor | None = None
            routing_applied = False
            if routing_mode in {"hard", "bias"}:
                commands = _distill_commands_from_env(env, batch_size=int(obs_tensor.shape[0]))
                if commands is None:
                    raise ValueError(
                        "distill command routing requires env.state.info['commands'] during playback"
                    )
                intents = distill_command_intents_from_commands(
                    commands, xy_threshold=xy_threshold, yaw_threshold=yaw_threshold
                )
                expected = _distill_expected_expert_tensor(
                    intents,
                    routing_targets,
                    num_experts=int(student.policy.num_experts),
                    device=obs_tensor.device,
                )
                rows = torch.arange(int(obs_tensor.shape[0]), device=obs_tensor.device)
                if routing_mode == "hard":
                    action = output.expert_actions[rows, expected]
                    selected = expected
                    route_probs = torch.nn.functional.one_hot(
                        expected, num_classes=int(student.policy.num_experts)
                    ).to(dtype=output.route_probs.dtype)
                else:
                    biased_logits = output.router_logits.clone()
                    biased_logits[rows, expected] += routing_bias
                    temperature = max(
                        float(getattr(student.policy, "router_temperature", 1.0)), 1e-8
                    )
                    route_probs = torch.softmax(biased_logits / temperature, dim=-1)
                    selected = torch.argmax(route_probs, dim=-1)
                    action = torch.sum(output.expert_actions * route_probs.unsqueeze(-1), dim=1)
                routing_applied = True

            expected_tuple = (
                tuple(int(value) for value in expected.detach().cpu().tolist())
                if expected is not None
                else ()
            )
            setattr(policy, "_unilab_distill_command_routing_applied", routing_applied)
            setattr(policy, "_unilab_distill_last_command_intents", intents)
            setattr(policy, "_unilab_distill_last_expected_experts", expected_tuple)
            setattr(
                policy,
                "_unilab_distill_last_selected_experts",
                tuple(int(value) for value in selected.detach().cpu().tolist()),
            )
            setattr(policy, "_unilab_distill_last_route_probs", route_probs.detach().cpu())
            setattr(policy, "_unilab_distill_last_raw_route_probs", raw_route_probs.detach().cpu())
            setattr(
                policy,
                "_unilab_distill_last_raw_selected_experts",
                tuple(int(value) for value in raw_selected.detach().cpu().tolist()),
            )
            return action.detach()

    _initialize_policy_diagnostics(
        policy,
        student=student,
        checkpoint=checkpoint_path,
        device=device,
        runtime_cfg=runtime_cfg,
        obs_normalizer_keys=normalizer_keys,
        routing_config_mode=routing_config_mode,
        routing_mode=routing_mode,
        routing_targets=routing_targets,
    )
    log(
        "Distill checkpoint diagnostics: "
        f"student_obs_dim={student.obs_dim}, student_action_dim={student.action_dim}, "
        f"agent_steps={int(student.agent_steps)}, "
        f"obs_normalizer={'present' if normalizer_keys else 'absent'}"
    )
    if routing_mode != "none":
        log(
            "Distill command routing: "
            f"configured={routing_config_mode}, effective={routing_mode}, "
            f"targets={dict(routing_targets)}"
        )
    return policy
