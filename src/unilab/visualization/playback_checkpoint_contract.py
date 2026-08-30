"""Checkpoint identity and environment-contract replay for interactive playback."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from unilab.training import resolve_task_checkpoint_path
from unilab.visualization.playback_cli import PlayInteractiveArgs
from unilab.visualization.playback_policy_sessions import (
    _actor_input_dim_from_state_dict,
    _load_playback_checkpoint,
)

ROOT_DIR = Path(__file__).parents[3]
_OFFPOLICY_INTERACTIVE_ALGOS = {"sac", "flashsac"}
_G1_STANDING_CONTRACT_STAND_TERMS = {
    "stand_still", "stand_action_l2", "stand_dof_vel_l2", "stand_lin_vel_xy_l2", "stand_yaw_vel_l2",
}

def _infer_checkpoint_actor_input_dim(ckpt_path: str) -> int | None:
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = loaded.get("actor_state_dict")
    if not isinstance(state_dict, dict):
        return None

    # Common rsl-rl naming: "mlp.0.weight" or nested prefixes ending with ".0.weight".
    for key in ("mlp.0.weight", "actor.mlp.0.weight"):
        w = state_dict.get(key)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[1])

    for key, w in state_dict.items():
        if key.endswith(".0.weight") and isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[1])
    return None


def _nested_get(data: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = data
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _g1_standing_contract_issues(run_config: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    rel_standing = _nested_get(run_config, "config.env.commands.rel_standing_envs")
    if rel_standing is None or float(rel_standing) <= 0.0:
        issues.append("missing config.env.commands.rel_standing_envs > 0")

    mode_enabled = _nested_get(run_config, "config.reward.mode.enabled")
    if mode_enabled is not True:
        issues.append("missing config.reward.mode.enabled=true")

    stand_terms_raw = _nested_get(run_config, "config.reward.mode.stand_terms", [])
    stand_terms = set(stand_terms_raw if isinstance(stand_terms_raw, list) else [])
    missing_terms = sorted(_G1_STANDING_CONTRACT_STAND_TERMS - stand_terms)
    if missing_terms:
        issues.append("missing stand reward terms: " + ", ".join(missing_terms))
    if "tracking_lin_vel" in stand_terms:
        issues.append("stand_terms must not include tracking_lin_vel")

    freeze_stand_phase = _nested_get(
        run_config, "config.reward.gait_constraint.freeze_phase_in_stand_mode"
    )
    if freeze_stand_phase is not True:
        issues.append("missing reward.gait_constraint.freeze_phase_in_stand_mode=true")

    stand_action_authority = _nested_get(run_config, "config.env.stand_action_authority")
    if stand_action_authority is True:
        issues.append(
            "env.stand_action_authority=true hard-gates standing actions; "
            "retrain with false to test learned residual standing"
        )

    feet_phase_scale = _nested_get(run_config, "config.reward.scales.feet_phase", 0.0)
    if float(feet_phase_scale) > 0.0:
        issues.append(
            f"positive old gait reward config.reward.scales.feet_phase={feet_phase_scale}"
        )

    return issues


def _resolve_play_checkpoint_path(args: PlayInteractiveArgs) -> Path | None:
    checkpoint_path, _checkpoint_dir = resolve_task_checkpoint_path(
        ROOT_DIR,
        task_name=args.task,
        load_run=args.load_run,
        algo_log_name=args.algo_log_name,
        checkpoint=args.checkpoint,
        log_root=args.log_root,
    )
    return checkpoint_path


def _load_checkpoint_run_config(args: PlayInteractiveArgs) -> Mapping[str, Any] | None:
    checkpoint_path = _resolve_play_checkpoint_path(args)
    if checkpoint_path is None:
        return None
    run_config_path = checkpoint_path.parent / "run_config.json"
    if not run_config_path.is_file():
        return None
    with run_config_path.open("r", encoding="utf-8") as f:
        run_config = json.load(f)
    return run_config if isinstance(run_config, Mapping) else None


def _checkpoint_actor_input_dim(args: PlayInteractiveArgs) -> int | None:
    checkpoint_path = _resolve_play_checkpoint_path(args)
    if checkpoint_path is None or not checkpoint_path.is_file():
        return None
    checkpoint = _load_playback_checkpoint(
        str(checkpoint_path),
        device_name="cpu",
        log=lambda message: None,
    )
    actor = checkpoint.get("actor") if isinstance(checkpoint, Mapping) else None
    if not isinstance(actor, Mapping):
        return None
    return _actor_input_dim_from_state_dict(actor)


def _apply_missing_g1_height_command_contract(
    merged: dict[str, Any],
    args: PlayInteractiveArgs,
) -> dict[str, Any]:
    if getattr(args, "task", None) not in {"g1_walk_flat", "G1WalkFlat"}:
        return merged
    if getattr(args, "algo", None) not in _OFFPOLICY_INTERACTIVE_ALGOS:
        return merged
    commands = merged.get("commands")
    if not isinstance(commands, dict):
        return merged
    if commands.get("observe_height_command") is True:
        return merged
    if bool(merged.get("mode_observation", False)) is not True:
        return merged
    if _checkpoint_actor_input_dim(args) != 100:
        return merged

    updated_commands = dict(commands)
    updated_commands["observe_height_command"] = True
    merged["commands"] = updated_commands
    return merged


def _apply_missing_g1_mode_observation_contract(
    merged: dict[str, Any],
    args: PlayInteractiveArgs,
) -> dict[str, Any]:
    if getattr(args, "task", None) not in {"g1_walk_flat", "G1WalkFlat"}:
        return merged
    if getattr(args, "algo", None) not in _OFFPOLICY_INTERACTIVE_ALGOS:
        return merged
    if _checkpoint_actor_input_dim(args) != 99:
        return merged

    merged["mode_observation"] = True
    return merged


def _apply_checkpoint_env_contract(
    env_cfg_override: dict[str, Any] | None,
    args: PlayInteractiveArgs,
) -> dict[str, Any] | None:
    """Replay env/reward owner contract from the selected checkpoint run_config."""
    merged = dict(env_cfg_override or {})
    run_config = _load_checkpoint_run_config(args)
    if run_config is None:
        merged = _apply_missing_g1_mode_observation_contract(merged, args)
        return _apply_missing_g1_height_command_contract(merged, args)
    run_cfg = run_config.get("config")
    if not isinstance(run_cfg, Mapping):
        merged = _apply_missing_g1_mode_observation_contract(merged, args)
        return _apply_missing_g1_height_command_contract(merged, args)

    run_env = run_cfg.get("env")
    if isinstance(run_env, Mapping):
        merged.update(dict(run_env))
        if (
            args.task in {"g1_walk_flat", "G1WalkFlat"}
            and "mode_observation" not in run_env
            and "mode_observation" in merged
        ):
            merged["mode_observation"] = False
        elif args.task in {"g1_walk_flat", "G1WalkFlat"} and "mode_observation" not in run_env:
            merged = _apply_missing_g1_mode_observation_contract(merged, args)
    run_reward = run_cfg.get("reward")
    if isinstance(run_reward, Mapping):
        merged["reward_config"] = dict(run_reward)
    return merged


apply_checkpoint_env_contract = _apply_checkpoint_env_contract


def _warn_if_g1_sac_checkpoint_lacks_standing_contract(
    *,
    algo: str,
    task_name: str,
    checkpoint_path: str | None,
    log: Callable[[str], None] = print,
) -> list[str]:
    if algo not in _OFFPOLICY_INTERACTIVE_ALGOS:
        return []
    if task_name not in {"g1_walk_flat", "G1WalkFlat"}:
        return []
    if checkpoint_path is None:
        return []

    run_config_path = Path(checkpoint_path).parent / "run_config.json"
    if not run_config_path.is_file():
        issues = [f"missing run_config.json beside checkpoint: {run_config_path}"]
    else:
        with run_config_path.open("r", encoding="utf-8") as f:
            run_config = json.load(f)
        issues = _g1_standing_contract_issues(run_config)

    if issues:
        log(
            "WARNING: selected G1 SAC checkpoint does not satisfy the "
            "standing/walking reward-mode contract."
        )
        log(f"  checkpoint: {checkpoint_path}")
        log(f"  run_config: {run_config_path}")
        for issue in issues:
            log(f"  - {issue}")
        log("  Retrain or pass algo.load_run=<new standing-mode run> before judging standing.")
    return issues


def resolve_checkpoint(
    task: str,
    load_run: str,
    checkpoint: str | None = None,
    algo_log_name: str = "rsl_rl_ppo",
    log_root: str | None = None,
    *,
    root_dir: Path = ROOT_DIR,
    resolver=resolve_task_checkpoint_path,
) -> str | None:
    checkpoint_path, checkpoint_dir = resolver(
        root_dir,
        task_name=task,
        load_run=load_run,
        algo_log_name=algo_log_name,
        checkpoint=checkpoint,
        log_root=log_root,
    )
    if checkpoint_path is None:
        if checkpoint is not None and checkpoint_dir is not None:
            checkpoint_name = f"model_{checkpoint}.pt" if str(checkpoint).isdigit() else str(checkpoint)
            print(f"[play_interactive] Checkpoint not found: {checkpoint_dir / checkpoint_name}")
        elif checkpoint_dir is not None:
            print(f"[play_interactive] No model_*.pt files in {checkpoint_dir}")
        else:
            print(f"[play_interactive] Run not found for load_run={load_run}")
        return None
    print(f"[play_interactive] Loading checkpoint: {checkpoint_path}")
    return str(checkpoint_path)
