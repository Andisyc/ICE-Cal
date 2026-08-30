"""CLI and Hydra composition for interactive playback."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from unilab.structured_configs import PPOConfig as _StructuredPPOConfig

ROOT_DIR = Path(__file__).parents[3]
PPOConfig = _StructuredPPOConfig
SUPPORTED_INTERACTIVE_ALGOS = ("ppo", "appo", "sac", "flashsac", "hora_distill", "distill", "fada")
_CONFIG_ROOT_BY_ALGO = {
    "ppo": "ppo", "appo": "appo", "sac": "offpolicy", "flashsac": "offpolicy",
    "hora_distill": "hora_distill", "distill": "distill", "fada": "distill",
}
_OFFPOLICY_INTERACTIVE_ALGOS = {"sac", "flashsac"}

@dataclass
class PlayInteractiveArgs:
    task: str
    load_run: str
    checkpoint: str | None
    checkpoint_path: str | None
    action_mode: str
    policy_obs_mode: str
    algo_log_name: str
    log_root: str | None
    show_target_bodies: bool
    show_reward_debug: bool
    target_show_axes: bool
    target_body_names: str
    target_max_bodies: int
    target_marker_radius: float
    target_axis_length: float
    target_marker_alpha: float
    reward_debug_show_velocity: bool
    reward_debug_lin_vel_scale: float
    reward_debug_ang_vel_scale: float
    reward_debug_show_connectors: bool
    reward_debug_show_global_anchor: bool
    camera_follow_body: bool
    camera_focus_body_name: str
    camera_height_offset: float
    camera_distance: float | None
    camera_elevation: float | None
    camera_azimuth: float | None
    use_env_visual_model: bool
    speed: float
    start_paused: bool
    keyboard: bool = False
    keyboard_step_lin: float = 0.1
    keyboard_step_ang: float = 0.2
    keyboard_step_height: float = 0.01
    require_keyboard_command_obs: bool = True
    algo: str = "ppo"


def _algo_config_dict(cfg: DictConfig | None) -> dict[str, Any]:
    """Return the composed PPO algo config as a plain dict.

    Args:
        cfg: Hydra config for the current playback run, or ``None`` when the
            script is driven through its legacy non-Hydra path.

    Returns:
        The resolved ``cfg.algo`` subtree as a mutable dict for rsl_rl.
    """
    if cfg is None:
        return cast(dict[str, Any], PPOConfig().to_dict())
    train_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(train_cfg_raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    return cast(dict[str, Any], train_cfg_raw)


@dataclass(frozen=True)
class InteractiveCliArgs:
    algo: str
    task: str
    sim: str
    overrides: list[str]


def _override_key(override: str) -> str:
    key = override.split("=", 1)[0].strip()
    return key.lstrip("+~")


def _parse_interactive_cli(argv: Sequence[str]) -> InteractiveCliArgs:
    parser = argparse.ArgumentParser(
        prog="play_interactive.py",
        description="Open a MuJoCo viewer for an interactive policy playback config.",
    )
    parser.add_argument("--algo", choices=SUPPORTED_INTERACTIVE_ALGOS, default="ppo")
    parser.add_argument("--task", required=True, help="Task name, for example go2_joystick_flat.")
    parser.add_argument("--sim", required=True, help="Owner backend config name to read.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides.")
    namespace = parser.parse_args(list(argv))

    task = str(namespace.task)
    sim = str(namespace.sim)
    if "/" in task:
        parser.error("--task must be a task name without '/'; pass backend via --sim.")
    if "/" in sim:
        parser.error("--sim must be a backend/config name without '/'.")

    extra_overrides = [str(item) for item in namespace.overrides]
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]
    overrides = _interactive_overrides_from_cli(task, sim, extra_overrides)
    return InteractiveCliArgs(
        algo=str(namespace.algo),
        task=task,
        sim=sim,
        overrides=overrides,
    )


def _interactive_overrides_from_cli(
    task: str, sim: str, extra_overrides: Sequence[str]
) -> list[str]:
    normalized = [f"task={task}/{sim}"]
    for override in extra_overrides:
        key = _override_key(str(override))
        if key in {"task", "training.sim_backend"}:
            raise SystemExit(
                f"{key} is controlled by --task/--sim; use explicit CLI flags instead."
            )
        normalized.append(str(override))
    return normalized


def _normalize_interactive_overrides(algo: str, overrides: list[str]) -> list[str]:
    normalized: list[str] = []
    has_algo_group = False
    has_action_mode = False
    has_play_only = False

    for override in overrides:
        key = _override_key(override)
        has_action_mode = has_action_mode or key == "interactive.action_mode"
        has_play_only = has_play_only or key == "training.play_only"
        if algo in _OFFPOLICY_INTERACTIVE_ALGOS and key == "algo":
            value = override.split("=", 1)[1] if "=" in override else ""
            if value != algo:
                raise SystemExit(
                    f"--algo {algo} cannot be combined with a non-{algo} Hydra algo group."
                )
            has_algo_group = True
        if algo in _OFFPOLICY_INTERACTIVE_ALGOS and key == "task" and "=" in override:
            value = override.split("=", 1)[1]
            if not value.startswith(f"{algo}/"):
                override = f"task={algo}/{value}"
        normalized.append(override)

    if algo in _OFFPOLICY_INTERACTIVE_ALGOS and not has_algo_group:
        normalized.insert(0, f"algo={algo}")
    if not has_play_only:
        normalized.append("training.play_only=true")
    if algo == "fada" and not has_action_mode:
        normalized.append("interactive.action_mode=policy")
    return normalized


def _compose_interactive_config(algo: str, overrides: list[str]) -> DictConfig:
    config_group = _CONFIG_ROOT_BY_ALGO[algo]
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / config_group),
        version_base="1.3",
    ):
        return compose(
            config_name="config",
            overrides=_normalize_interactive_overrides(algo, overrides),
        )


def _normalize_checkpoint_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in {"-1", "None", "null"} else text


def _build_play_args(cfg: DictConfig, *, algo: str = "ppo") -> PlayInteractiveArgs:
    return PlayInteractiveArgs(
        task=str(cfg.training.task_name),
        load_run=str(cfg.algo.load_run),
        checkpoint=_normalize_checkpoint_value(OmegaConf.select(cfg, "algo.checkpoint")),
        checkpoint_path=_normalize_checkpoint_value(
            OmegaConf.select(cfg, "training.play_checkpoint_path")
        ),
        action_mode=str(cfg.interactive.action_mode),
        policy_obs_mode=str(cfg.interactive.policy_obs_mode),
        algo_log_name=str(cfg.algo.algo_log_name),
        log_root=(
            str(cfg.training.log_root)
            if OmegaConf.select(cfg, "training.log_root") is not None
            else None
        ),
        show_target_bodies=bool(cfg.interactive.show_target_bodies),
        show_reward_debug=bool(cfg.interactive.show_reward_debug),
        target_show_axes=bool(cfg.interactive.target_show_axes),
        target_body_names=str(cfg.interactive.target_body_names),
        target_max_bodies=int(cfg.interactive.target_max_bodies),
        target_marker_radius=float(cfg.interactive.target_marker_radius),
        target_axis_length=float(cfg.interactive.target_axis_length),
        target_marker_alpha=float(cfg.interactive.target_marker_alpha),
        reward_debug_show_velocity=bool(cfg.interactive.reward_debug_show_velocity),
        reward_debug_lin_vel_scale=float(cfg.interactive.reward_debug_lin_vel_scale),
        reward_debug_ang_vel_scale=float(cfg.interactive.reward_debug_ang_vel_scale),
        reward_debug_show_connectors=bool(cfg.interactive.reward_debug_show_connectors),
        reward_debug_show_global_anchor=bool(cfg.interactive.reward_debug_show_global_anchor),
        camera_follow_body=bool(cfg.interactive.camera_follow_body),
        camera_focus_body_name=str(cfg.interactive.camera_focus_body_name),
        camera_height_offset=float(cfg.interactive.camera_height_offset),
        camera_distance=(
            float(cfg.interactive.camera_distance)
            if OmegaConf.select(cfg, "interactive.camera_distance") is not None
            else None
        ),
        camera_elevation=(
            float(cfg.interactive.camera_elevation)
            if OmegaConf.select(cfg, "interactive.camera_elevation") is not None
            else None
        ),
        camera_azimuth=(
            float(cfg.interactive.camera_azimuth)
            if OmegaConf.select(cfg, "interactive.camera_azimuth") is not None
            else None
        ),
        use_env_visual_model=bool(cfg.interactive.use_env_visual_model),
        speed=float(OmegaConf.select(cfg, "interactive.speed", default=1.0)),
        start_paused=bool(OmegaConf.select(cfg, "interactive.start_paused", default=False)),
        keyboard=bool(OmegaConf.select(cfg, "interactive.keyboard", default=False)),
        keyboard_step_lin=float(
            OmegaConf.select(cfg, "interactive.keyboard_step_lin", default=0.1)
        ),
        keyboard_step_ang=float(
            OmegaConf.select(cfg, "interactive.keyboard_step_ang", default=0.2)
        ),
        keyboard_step_height=float(
            OmegaConf.select(cfg, "interactive.keyboard_step_height", default=0.01)
        ),
        require_keyboard_command_obs=bool(
            OmegaConf.select(cfg, "interactive.require_keyboard_command_obs", default=True)
        ),
        algo=algo,
    )
