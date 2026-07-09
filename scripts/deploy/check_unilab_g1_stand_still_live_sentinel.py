#!/usr/bin/env python3
"""Live MuJoCo sentinel for the standalone G1 stand-still expert."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from unilab.training import (  # noqa: E402
    BackendAdapter,
    assert_offpolicy_task_choice_matches_algo,
    create_env,
    ensure_registries,
)


def _compose_cfg() -> Any:
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        return compose(config_name="config", overrides=["task=sac/g1_stand_still/mujoco"])


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
    }


def _action_scale_vector(env: Any) -> np.ndarray:
    scale = np.asarray(env.cfg.control_config.action_scale, dtype=np.float32)
    if scale.ndim == 0:
        scale = np.full((env.action_space.shape[0],), float(scale), dtype=np.float32)
    return scale


def _top_abs(values: np.ndarray, *, limit: int = 8) -> list[dict[str, float | int]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=0)
    if arr.size == 0:
        return []
    indices = np.argsort(np.abs(arr))[-limit:][::-1]
    return [{"index": int(idx), "value": float(arr[idx])} for idx in indices]


def _role_stats(values: np.ndarray, roles: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    arr = np.asarray(values)
    for role in sorted(set(roles.tolist())):
        mask = roles == role
        out[str(role)] = _stats(arr[mask])
    return out


def _role_counts(values: np.ndarray, roles: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    arr = np.asarray(values)
    for role in sorted(set(roles.tolist())):
        mask = roles == role
        out[str(role)] = int(np.count_nonzero(arr[mask]))
    return out


def _base_qvel_stats(env: Any) -> dict[str, dict[str, float]]:
    backend = env._backend
    state = np.asarray(backend.get_physics_state(), dtype=np.float32)
    idx_qvel = int(getattr(backend, "_idx_qvel"))
    base_qvel = state[:, idx_qvel : idx_qvel + 6]
    return {
        "linear": _stats(base_qvel[:, 0:3]),
        "angular": _stats(base_qvel[:, 3:6]),
    }


def _sensor(env: Any, name: str) -> np.ndarray:
    return np.asarray(env._backend.get_sensor_data(name), dtype=np.float32)


def _height(env: Any) -> np.ndarray:
    return np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)


def _tilt_deg(env: Any) -> np.ndarray:
    gravity = _sensor(env, env.cfg.sensor.upvector)
    return np.rad2deg(np.arccos(np.clip(gravity[:, 2], -1.0, 1.0))).astype(np.float32)


def _foot_metrics(env: Any) -> dict[str, np.ndarray]:
    left = _sensor(env, "left_foot_pos")
    right = _sensor(env, "right_foot_pos")
    delta = env._feet_delta_in_base_yaw_frame(left, right)
    base_delta = env._base_delta_from_feet_center_in_base_yaw_frame()
    return {
        "foot_width": np.abs(delta[:, 1]),
        "foot_sagittal_abs": np.abs(delta[:, 0]),
        "base_over_feet_x": base_delta[:, 0],
        "base_over_feet_y": base_delta[:, 1],
    }


def _contact_sensor_count(env: Any, prefix: str) -> np.ndarray:
    values = [_sensor(env, f"{prefix}_foot_contact_{i}") for i in range(4)]
    return np.sum(np.stack(values, axis=1) > 0.5, axis=1).astype(np.float32).reshape(-1)


def _support_metrics(env: Any) -> dict[str, np.ndarray]:
    left_count = _contact_sensor_count(env, "left")
    right_count = _contact_sensor_count(env, "right")
    total = left_count + right_count
    left_vel = _sensor(env, "left_foot_linvel")
    right_vel = _sensor(env, "right_foot_linvel")
    left_contact = (left_count > 0.0).astype(np.float32)
    right_contact = (right_count > 0.0).astype(np.float32)
    return {
        "left_contact_count": left_count,
        "right_contact_count": right_count,
        "both_feet_contact": ((left_count > 0.0) & (right_count > 0.0)).astype(np.float32),
        "contact_balance": np.where(
            total > 1.0e-6,
            np.abs(left_count - right_count) / np.maximum(total, 1.0e-6),
            1.0,
        ),
        "contact_feet_slide_xy": (
            np.sum(np.square(left_vel[:, :2]), axis=1) * left_contact
            + np.sum(np.square(right_vel[:, :2]), axis=1) * right_contact
        ),
    }


def _termination_reasons(env: Any, terminated: np.ndarray) -> dict[str, Any]:
    height = _height(env)
    tilt = _tilt_deg(env)
    low = height < float(env.cfg.reward_config.min_base_height)
    tilted = tilt > float(env.cfg.reward_config.max_tilt_deg)
    reasons = {
        "low_height": int(np.count_nonzero(terminated & low & ~tilted)),
        "large_tilt": int(np.count_nonzero(terminated & tilted & ~low)),
        "low_height_and_large_tilt": int(np.count_nonzero(terminated & low & tilted)),
        "terminated_total": int(np.count_nonzero(terminated)),
    }
    if np.any(terminated):
        first = int(np.flatnonzero(terminated)[0])
        if low[first] and tilted[first]:
            first_reason = "low_height_and_large_tilt"
        elif low[first]:
            first_reason = "low_height"
        elif tilted[first]:
            first_reason = "large_tilt"
        else:
            first_reason = "unknown"
    else:
        first = None
        first_reason = None
    return {"counts": reasons, "first_env": first, "first_reason": first_reason}


def _reward_context(env: Any, state: Any) -> Any:
    linvel = env.get_local_linvel()
    gyro = env.get_gyro()
    gravity = env._backend.get_sensor_data(env.cfg.sensor.upvector)
    dof_pos = env.get_dof_pos()
    dof_vel = env.get_dof_vel()
    return env._build_reward_context(state.info, linvel, gyro, gravity, dof_pos, dof_vel)


def _term_contributions(env: Any, state: Any) -> dict[str, float]:
    ctx = _reward_context(env, state)
    terms: dict[str, float] = {}
    for name, scale in env.cfg.reward_config.scales.items():
        if float(scale) == 0.0 or name not in env._reward_fns:
            continue
        values = np.asarray(env._reward_fns[name](ctx), dtype=np.float32)
        terms[name] = float(np.mean(values * float(scale) * float(env.cfg.ctrl_dt)))
    return terms


def _term_contributions_by_role(env: Any, state: Any, roles: np.ndarray) -> dict[str, dict[str, float]]:
    ctx = _reward_context(env, state)
    terms: dict[str, dict[str, float]] = {}
    for name, scale in env.cfg.reward_config.scales.items():
        if float(scale) == 0.0 or name not in env._reward_fns:
            continue
        values = np.asarray(env._reward_fns[name](ctx), dtype=np.float32)
        weighted = values * float(scale) * float(env.cfg.ctrl_dt)
        terms[name] = {
            role: stats["mean"] for role, stats in _role_stats(weighted, roles).items()
        }
    return terms


def _snapshot(env: Any, state: Any, *, include_termination: bool = True) -> dict[str, Any]:
    foot = _foot_metrics(env)
    support = _support_metrics(env)
    commands = np.asarray(state.info.get("commands"), dtype=np.float32)
    gait_enabled = np.asarray(state.info.get("gait_enabled", np.zeros((env.num_envs,))), dtype=np.float32)
    height = _height(env)
    height_target = float(env.cfg.reward_config.base_height_target)
    out = {
        "base_height": _stats(height),
        "base_height_target": height_target,
        "base_height_error": _stats(height - height_target),
        "base_height_deficit": _stats(np.maximum(height_target - height, 0.0)),
        "tilt_deg": _stats(_tilt_deg(env)),
        "foot_width": _stats(foot["foot_width"]),
        "foot_sagittal_abs": _stats(foot["foot_sagittal_abs"]),
        "base_over_feet_x": _stats(foot["base_over_feet_x"]),
        "base_over_feet_y": _stats(foot["base_over_feet_y"]),
        "left_contact_count": _stats(support["left_contact_count"]),
        "right_contact_count": _stats(support["right_contact_count"]),
        "both_feet_contact": _stats(support["both_feet_contact"]),
        "contact_balance": _stats(support["contact_balance"]),
        "contact_feet_slide_xy": _stats(support["contact_feet_slide_xy"]),
        "commands_max_abs": float(np.max(np.abs(commands))) if commands.size else 0.0,
        "gait_enabled_mean": float(np.mean(gait_enabled)) if gait_enabled.size else 0.0,
        "reward_mean": float(np.mean(state.reward)),
    }
    if include_termination:
        out["terminated_total"] = int(np.count_nonzero(state.terminated))
        out["termination"] = _termination_reasons(env, np.asarray(state.terminated, dtype=bool))
    return out


def _snapshot_by_role(env: Any, state: Any, roles: np.ndarray) -> dict[str, Any]:
    foot = _foot_metrics(env)
    support = _support_metrics(env)
    height = _height(env)
    height_target = float(env.cfg.reward_config.base_height_target)
    return {
        "base_height": _role_stats(height, roles),
        "base_height_error": _role_stats(height - height_target, roles),
        "base_height_deficit": _role_stats(np.maximum(height_target - height, 0.0), roles),
        "tilt_deg": _role_stats(_tilt_deg(env), roles),
        "foot_width": _role_stats(foot["foot_width"], roles),
        "foot_sagittal_abs": _role_stats(foot["foot_sagittal_abs"], roles),
        "base_over_feet_x": _role_stats(foot["base_over_feet_x"], roles),
        "base_over_feet_y": _role_stats(foot["base_over_feet_y"], roles),
        "left_contact_count": _role_stats(support["left_contact_count"], roles),
        "right_contact_count": _role_stats(support["right_contact_count"], roles),
        "both_feet_contact": _role_stats(support["both_feet_contact"], roles),
        "contact_balance": _role_stats(support["contact_balance"], roles),
        "contact_feet_slide_xy": _role_stats(support["contact_feet_slide_xy"], roles),
        "terminated_total": _role_counts(state.terminated, roles),
        "reward_mean": _role_stats(np.asarray(state.reward), roles),
    }


def _create_env(cfg: Any, num_envs: int) -> Any:
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="sac")
    env_override = adapter.build_task_env_cfg_override()
    ensure_registries()
    env = create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=env_override,
        sim_backend="mujoco",
        task_name=str(cfg.training.task_name),
    )
    return env, env_override


def _sync_envs_to_first_state(env: Any, state: Any) -> Any:
    """Copy env0 physics and info to all envs for role-comparable probes."""

    if env.num_envs <= 1:
        return state
    backend = env._backend
    physics = np.asarray(backend.get_physics_state(), dtype=np.float32)
    idx_qpos = int(getattr(backend, "_idx_qpos"))
    idx_qvel = int(getattr(backend, "_idx_qvel"))
    nq = int(getattr(backend, "nq"))
    nv = int(getattr(backend, "nv"))
    env_ids = np.arange(env.num_envs, dtype=np.int32)
    qpos = np.broadcast_to(physics[0:1, idx_qpos : idx_qpos + nq], (env.num_envs, nq)).copy()
    qvel = np.broadcast_to(physics[0:1, idx_qvel : idx_qvel + nv], (env.num_envs, nv)).copy()
    backend.set_state(env_ids, qpos, qvel)
    for key, value in list(state.info.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (env.num_envs,):
            value[...] = np.broadcast_to(value[0:1], value.shape)
    refreshed_state = env.update_state(state)
    return refreshed_state if refreshed_state is not None else state


def _load_policy_actor(cfg: Any, env: Any, *, device_name: str = "cpu") -> dict[str, Any]:
    """Load the same deterministic SAC actor used by interactive playback."""

    from train_offpolicy import (
        extract_play_obs,
        resolve_checkpoint_path,
        resolve_play_actor_spec,
        resolve_play_obs_dims,
    )

    from unilab.algos.torch.common.actor_factory import build_actor
    from unilab.visualization.interactive_playback import _load_playback_checkpoint

    obs_dim, critic_obs_dim = resolve_play_obs_dims(env.obs_groups_spec)
    action_dim = int(env.action_space.shape[0])
    actor_algo_type, actor_kwargs = resolve_play_actor_spec(
        "sac",
        cfg,
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
    )
    actor = build_actor(
        actor_algo_type,
        obs_dim,
        action_dim,
        cfg.algo.actor_hidden_dim,
        cfg.algo.use_layer_norm,
        device_name,
        **actor_kwargs,
    )
    checkpoint_path, _checkpoint_dir = resolve_checkpoint_path(
        ROOT_DIR,
        cfg.algo.algo_log_name,
        cfg.training.task_name,
        cfg.algo.load_run,
    )
    if checkpoint_path is None:
        raise RuntimeError("No SAC checkpoint found for current G1StandStill probe config.")
    checkpoint = _load_playback_checkpoint(
        checkpoint_path,
        device_name=device_name,
        log=lambda _message: None,
    )
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    normalizer = None
    if bool(getattr(cfg.algo, "obs_normalization", False)) and checkpoint.get("obs_normalizer"):
        from unilab.algos.torch.common.normalization import EmpiricalNormalization

        normalizer = EmpiricalNormalization(shape=obs_dim, device=device_name)
        normalizer.load_state_dict(checkpoint["obs_normalizer"])
        normalizer.eval()
    return {
        "actor": actor,
        "actor_algo_type": actor_algo_type,
        "checkpoint_path": str(checkpoint_path),
        "device_name": device_name,
        "extract_play_obs": extract_play_obs,
        "normalizer": normalizer,
    }


def _policy_actions(policy: dict[str, Any], state: Any) -> np.ndarray:
    obs = np.asarray(policy["extract_play_obs"](state.obs), dtype=np.float32)
    obs_torch = torch.from_numpy(obs).to(policy["device_name"])
    normalizer = policy.get("normalizer")
    if normalizer is not None:
        obs_torch = normalizer(obs_torch, update=False)
    with torch.inference_mode():
        actions = policy["actor"].explore(obs_torch, deterministic=True)
    return actions.detach().cpu().numpy().astype(np.float32)


def _stability_score(
    env: Any,
    state: Any,
    *,
    actions: np.ndarray,
    max_ok_tilt_deg: float,
    max_ok_base_over_feet_x_abs: float,
    min_ok_base_height: float,
    max_abs_action: float,
) -> np.ndarray:
    tilt = np.asarray(_tilt_deg(env), dtype=np.float32).reshape(-1)
    height = np.asarray(_height(env), dtype=np.float32).reshape(-1)
    foot = _foot_metrics(env)
    support = _support_metrics(env)
    height_target = float(env.cfg.reward_config.base_height_target)
    base_x = np.abs(np.asarray(foot["base_over_feet_x"], dtype=np.float32).reshape(-1))
    action_l2 = np.linalg.norm(actions, axis=1) / max(np.sqrt(actions.shape[1]), 1.0)
    height_deficit_cost = np.maximum(0.0, height_target - height) / max(height_target, 1.0e-6)
    height_floor_cost = np.maximum(0.0, min_ok_base_height - height) / max(
        min_ok_base_height, 1.0e-6
    )
    height_overshoot_cost = np.maximum(0.0, height - height_target) / max(height_target, 1.0e-6)
    tilt_cost = np.square(tilt / max(max_ok_tilt_deg, 1.0))
    base_x_cost = np.maximum(0.0, base_x - max_ok_base_over_feet_x_abs) / max(
        max_ok_base_over_feet_x_abs, 1.0e-6
    )
    both_contact = np.asarray(support["both_feet_contact"], dtype=np.float32).reshape(-1)
    contact_balance = np.asarray(support["contact_balance"], dtype=np.float32).reshape(-1)
    feet_slide = np.asarray(support["contact_feet_slide_xy"], dtype=np.float32).reshape(-1)
    both_contact_cost = 1.0 - both_contact
    contact_balance_cost = contact_balance
    slide_cost = np.minimum(feet_slide / 0.01, 10.0)
    action_cost = action_l2 / max(max_abs_action, 1.0e-6)
    termination_cost = np.asarray(state.terminated, dtype=np.float32) * 10.0
    return -(
        10.0 * height_deficit_cost
        + 6.0 * height_floor_cost
        + 2.0 * height_overshoot_cost
        + 2.0 * tilt_cost
        + 3.0 * base_x_cost
        + 2.0 * both_contact_cost
        + 1.0 * contact_balance_cost
        + 0.5 * slide_cost
        + 0.25 * action_cost
        + termination_cost
    )


def _evaluate_constant_actions(
    cfg: Any,
    actions: np.ndarray,
    *,
    steps: int,
    max_ok_tilt_deg: float,
    max_ok_base_over_feet_x_abs: float,
    min_ok_base_height: float,
    max_abs_action: float,
) -> dict[str, Any]:
    env, _ = _create_env(cfg, int(actions.shape[0]))
    final_state = None
    try:
        env.set_autoreset(False)
        state = env.init_state()
        state = _sync_envs_to_first_state(env, state)
        final_state = state
        for _ in range(steps):
            state = env.step(actions.astype(np.float32, copy=False))
            final_state = state
        assert final_state is not None
        score = _stability_score(
            env,
            final_state,
            actions=actions,
            max_ok_tilt_deg=max_ok_tilt_deg,
            max_ok_base_over_feet_x_abs=max_ok_base_over_feet_x_abs,
            min_ok_base_height=min_ok_base_height,
            max_abs_action=max_abs_action,
        )
        foot = _foot_metrics(env)
        support = _support_metrics(env)
        return {
            "score": score,
            "tilt_deg": _tilt_deg(env),
            "base_height": _height(env),
            "base_over_feet_x": foot["base_over_feet_x"],
            "both_feet_contact": support["both_feet_contact"],
            "contact_balance": support["contact_balance"],
            "contact_feet_slide_xy": support["contact_feet_slide_xy"],
            "action_l2": np.linalg.norm(actions, axis=1),
            "terminated": np.asarray(final_state.terminated, dtype=bool),
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _static_anchor_search(
    cfg: Any,
    *,
    action_dim: int,
    steps: int,
    seed: int,
    candidates: int,
    iterations: int,
    elite_count: int,
    max_abs: float,
    std_init: float,
    max_ok_tilt_deg: float,
    max_ok_base_over_feet_x_abs: float,
    min_ok_base_height: float,
    search_dim_override: int | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    search_dim = action_dim if search_dim_override is None else min(search_dim_override, action_dim)
    dims = np.arange(search_dim)
    mean = np.zeros(action_dim, dtype=np.float32)
    std = np.zeros(action_dim, dtype=np.float32)
    std[dims] = std_init
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    elite_count = min(max(1, elite_count), candidates)

    zero_actions = np.zeros((1, action_dim), dtype=np.float32)
    zero_eval = _evaluate_constant_actions(
        cfg,
        zero_actions,
        steps=steps,
        max_ok_tilt_deg=max_ok_tilt_deg,
        max_ok_base_over_feet_x_abs=max_ok_base_over_feet_x_abs,
        min_ok_base_height=min_ok_base_height,
        max_abs_action=max_abs,
    )

    for iteration in range(iterations):
        samples = np.zeros((candidates, action_dim), dtype=np.float32)
        samples[:, dims] = rng.normal(mean[dims], std[dims], size=(candidates, search_dim))
        samples = np.clip(samples, -max_abs, max_abs).astype(np.float32)
        result = _evaluate_constant_actions(
            cfg,
            samples,
            steps=steps,
            max_ok_tilt_deg=max_ok_tilt_deg,
            max_ok_base_over_feet_x_abs=max_ok_base_over_feet_x_abs,
            min_ok_base_height=min_ok_base_height,
            max_abs_action=max_abs,
        )
        scores = np.asarray(result["score"], dtype=np.float32)
        elite_idx = np.argsort(scores)[-elite_count:]
        top_idx = int(elite_idx[-1])
        elite = samples[elite_idx]
        mean[dims] = np.mean(elite[:, dims], axis=0)
        std[dims] = np.maximum(np.std(elite[:, dims], axis=0), 0.02)
        current = {
            "iteration": iteration + 1,
            "best_score": float(scores[top_idx]),
            "best_index": top_idx,
            "best_tilt_deg": float(result["tilt_deg"][top_idx]),
            "best_base_height": float(result["base_height"][top_idx]),
            "best_base_over_feet_x": float(result["base_over_feet_x"][top_idx]),
            "best_both_feet_contact": float(result["both_feet_contact"][top_idx]),
            "best_contact_balance": float(result["contact_balance"][top_idx]),
            "best_contact_feet_slide_xy": float(result["contact_feet_slide_xy"][top_idx]),
            "best_terminated": bool(result["terminated"][top_idx]),
            "best_action_l2": float(result["action_l2"][top_idx]),
            "best_action_top_abs": _top_abs(samples[top_idx]),
        }
        history.append(current)
        if best is None or current["best_score"] > best["best_score"]:
            best = {**current, "action": samples[top_idx].tolist()}

    assert best is not None
    zero_score = float(zero_eval["score"][0])
    return {
        "searched_joint_indices": dims.tolist(),
        "zero_action": {
            "score": zero_score,
            "tilt_deg": float(zero_eval["tilt_deg"][0]),
            "base_height": float(zero_eval["base_height"][0]),
            "base_over_feet_x": float(zero_eval["base_over_feet_x"][0]),
            "both_feet_contact": float(zero_eval["both_feet_contact"][0]),
            "contact_balance": float(zero_eval["contact_balance"][0]),
            "contact_feet_slide_xy": float(zero_eval["contact_feet_slide_xy"][0]),
            "action_l2": float(zero_eval["action_l2"][0]),
            "terminated": bool(zero_eval["terminated"][0]),
        },
        "best_constant_action": best,
        "history": history,
        "interpretation": (
            "nonzero static action anchor exists; reward/action anchor should not force zero action"
            if best["best_score"] > zero_score + 0.25
            else "no simple lower-body constant action anchor improved stability; reset/PD physics anchor remains suspect"
        ),
    }


def _role_boundary_decision(role_final: dict[str, Any]) -> dict[str, Any]:
    zero_tilt = role_final["tilt_deg"]["zero_action"]["max"]
    zero_height = role_final["base_height"]["zero_action"]["min"]
    zero_deficit = role_final["base_height_deficit"]["zero_action"]["max"]
    zero_x = max(
        abs(role_final["base_over_feet_x"]["zero_action"]["min"]),
        abs(role_final["base_over_feet_x"]["zero_action"]["max"]),
    )
    out: dict[str, Any] = {
        "zero_action": {
            "tilt_max": zero_tilt,
            "height_min": zero_height,
            "height_deficit_max": zero_deficit,
            "base_over_feet_x_abs_max": zero_x,
        }
    }

    for role in sorted(role_final["base_height"]):
        if role == "zero_action":
            continue
        role_tilt = role_final["tilt_deg"][role]["max"]
        role_height = role_final["base_height"][role]["min"]
        role_deficit = role_final["base_height_deficit"][role]["max"]
        role_x = max(
            abs(role_final["base_over_feet_x"][role]["min"]),
            abs(role_final["base_over_feet_x"][role]["max"]),
        )
        out[role] = {
            "tilt_max": role_tilt,
            "height_min": role_height,
            "height_deficit_max": role_deficit,
            "base_over_feet_x_abs_max": role_x,
            "delta_vs_zero": {
                "tilt_max": role_tilt - zero_tilt,
                "height_min": role_height - zero_height,
                "height_deficit_max": role_deficit - zero_deficit,
                "base_over_feet_x_abs_max": role_x - zero_x,
            },
        }

    searched = out.get("searched_support_action")
    if searched is not None:
        delta = searched["delta_vs_zero"]
        improves_height = delta["height_min"] > 0.01
        reduces_tilt = delta["tilt_max"] < -1.0
        out["interpretation"] = (
            "searched support action improves loaded standing equilibrium"
            if improves_height or reduces_tilt
            else "searched support action did not clearly beat zero action in this rollout"
        )
    if "current_trained_policy_action" in out:
        policy = out["current_trained_policy_action"]
        policy_delta = policy["delta_vs_zero"]
        out["policy_interpretation"] = (
            "trained policy improves loaded equilibrium over zero action"
            if policy_delta["height_min"] > 0.01 or policy_delta["tilt_max"] < -1.0
            else "trained policy does not clearly improve loaded equilibrium over zero action"
        )
    elif "hold_reset_pose" in out:
        hold = out["hold_reset_pose"]
        out["interpretation"] = (
            "G1LOC-ACT-001 default_angles/static action anchor mismatch"
            if hold["tilt_max"] + 10.0 < zero_tilt or hold["base_over_feet_x_abs_max"] + 0.1 < zero_x
            else "G1LOC-ENV-001 reset/keyframe or physics anchor remains suspect"
        )
    else:
        out["interpretation"] = "zero-action-only rollout; no action alternative was evaluated"
    return out


def run_check(
    *,
    num_envs: int,
    steps: int,
    seed: int,
    max_ok_tilt_deg: float,
    max_ok_base_over_feet_x_abs: float,
    min_ok_base_height: float | None,
    probe_mode: str,
    anchor_search_candidates: int,
    anchor_search_iterations: int,
    anchor_search_elites: int,
    anchor_search_max_abs: float,
    anchor_search_std: float,
    anchor_search_dims: int,
) -> tuple[list[str], dict[str, Any]]:
    np.random.seed(seed)
    cfg = _compose_cfg()
    if probe_mode in {"anchor-diff", "support-diff"}:
        num_envs = max(num_envs, 2)
    if probe_mode == "support-policy-diff":
        num_envs = max(num_envs, 3)
    env, env_override = _create_env(cfg, num_envs)
    failures: list[str] = []
    per_term_sum: dict[str, float] = {}
    per_term_role_sum: dict[str, dict[str, float]] = {}
    per_term_min: dict[str, float] = {}
    first_termination_step: int | None = None
    final_state = None
    try:
        env.set_autoreset(False)
        state = env.init_state()
        state = _sync_envs_to_first_state(env, state)
        final_state = state
        action_dim = int(env.action_space.shape[0])
        roles = np.full((num_envs,), "zero_action", dtype=object)
        actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        policy: dict[str, Any] | None = None
        reset_dof_pos = np.asarray(env.get_dof_pos(), dtype=np.float32)
        default_angles = np.asarray(env.default_angles, dtype=np.float32)
        action_scale = _action_scale_vector(env)
        hold_reset_pose_action = (reset_dof_pos - default_angles[None, :]) / action_scale[None, :]
        support_anchor_search: dict[str, Any] | None = None
        if probe_mode in {"support-diff", "support-policy-diff"}:
            height_floor = (
                min_ok_base_height
                if min_ok_base_height is not None
                else float(env.cfg.reward_config.base_height_target) - 0.12
            )
            support_anchor_search = _static_anchor_search(
                cfg,
                action_dim=action_dim,
                steps=steps,
                seed=seed + 1000,
                candidates=anchor_search_candidates,
                iterations=anchor_search_iterations,
                elite_count=anchor_search_elites,
                max_abs=anchor_search_max_abs,
                std_init=anchor_search_std,
                max_ok_tilt_deg=max_ok_tilt_deg,
                max_ok_base_over_feet_x_abs=max_ok_base_over_feet_x_abs,
                min_ok_base_height=height_floor,
                search_dim_override=None if anchor_search_dims <= 0 else anchor_search_dims,
            )
            support_action = np.asarray(
                support_anchor_search["best_constant_action"]["action"], dtype=np.float32
            )
            if probe_mode == "support-policy-diff":
                role_cycle = np.asarray(
                    [
                        "zero_action",
                        "searched_support_action",
                        "current_trained_policy_action",
                    ],
                    dtype=object,
                )
                roles = role_cycle[np.arange(num_envs) % role_cycle.size]
                policy = _load_policy_actor(cfg, env)
            else:
                roles[1::2] = "searched_support_action"
            actions[roles == "searched_support_action"] = support_action
        elif probe_mode == "anchor-diff":
            roles[1::2] = "hold_reset_pose"
            actions[roles == "hold_reset_pose"] = hold_reset_pose_action[roles == "hold_reset_pose"]
        exec_actions0 = env._actions_for_execution(actions, state.info)
        ctrl0 = exec_actions0 * action_scale[None, :] + default_angles[None, :]
        reset_dof_delta = reset_dof_pos - default_angles[None, :]
        ctrl_minus_reset_dof = ctrl0 - reset_dof_pos
        details: dict[str, Any] = {
            "task": str(cfg.training.task_name),
            "num_envs": num_envs,
            "steps_requested": steps,
            "seed": seed,
            "probe_mode": probe_mode,
            "stability_thresholds": {
                "max_ok_tilt_deg": max_ok_tilt_deg,
                "max_ok_base_over_feet_x_abs": max_ok_base_over_feet_x_abs,
                "min_ok_base_height": min_ok_base_height
                if min_ok_base_height is not None
                else float(env.cfg.reward_config.base_height_target) - 0.12,
            },
            "obs_groups_spec": dict(env.obs_groups_spec),
            "reward_keys": list(env.cfg.reward_config.scales.keys()),
            "forbidden_reward_keys_present": sorted(
                set(env.cfg.reward_config.scales)
                & {
                    "tracking_lin_vel",
                    "tracking_ang_vel",
                    "feet_phase",
                    "feet_phase_contrast",
                    "feet_phase_contact",
                    "track_base_height_exp_smooth",
                }
            ),
            "env_override": {
                "mode_observation": env_override.get("mode_observation"),
                "stand_action_authority": env_override.get("stand_action_authority"),
                "reset_base_qvel_limit": env_override.get("reset_base_qvel_limit"),
                "rel_standing_envs": env_override.get("commands", {}).get("rel_standing_envs"),
                "rel_transition_envs": env_override.get("commands", {}).get("rel_transition_envs"),
                "observe_height_command": env_override.get("commands", {}).get("observe_height_command"),
            },
            "initial": _snapshot(env, state, include_termination=False),
            "module_probe": {
                "config_G1LOC_C_004": {
                    "task": str(cfg.training.task_name),
                    "mode_observation": env_override.get("mode_observation"),
                    "stand_action_authority": env_override.get("stand_action_authority"),
                    "commands_max_abs": _snapshot(env, state, include_termination=False)[
                        "commands_max_abs"
                    ],
                    "gait_enabled_mean": _snapshot(env, state, include_termination=False)[
                        "gait_enabled_mean"
                    ],
                    "forbidden_reward_keys_present": sorted(
                        set(env.cfg.reward_config.scales)
                        & {
                            "tracking_lin_vel",
                            "tracking_ang_vel",
                            "feet_phase",
                            "feet_phase_contrast",
                            "feet_phase_contact",
                            "track_base_height_exp_smooth",
                        }
                    ),
                },
                "reset_G1LOC_ENV_001": {
                    "base_qvel": _base_qvel_stats(env),
                    "reset_dof_minus_default_angles": {
                        "stats": _stats(reset_dof_delta),
                        "top_abs_joint_indices": _top_abs(reset_dof_delta),
                    },
                    "initial_by_role": _snapshot_by_role(env, state, roles),
                },
                "action_G1LOC_ACT_001": {
                    "action_scale": _stats(action_scale),
                    "hold_reset_pose_action": {
                        "stats": _stats(hold_reset_pose_action),
                        "top_abs_joint_indices": _top_abs(hold_reset_pose_action),
                    },
                    "first_step_ctrl_minus_reset_dof": {
                        "by_role": _role_stats(ctrl_minus_reset_dof, roles),
                        "top_abs_zero_action_joint_indices": _top_abs(
                            ctrl_minus_reset_dof[roles == "zero_action"]
                        ),
                        "top_abs_hold_reset_pose_joint_indices": _top_abs(
                            ctrl_minus_reset_dof[roles == "hold_reset_pose"]
                        )
                        if np.any(roles == "hold_reset_pose")
                        else [],
                    },
                },
            },
            "first_step_action": {
                "raw_action": _stats(actions),
                "raw_action_by_role": _role_stats(actions, roles),
                "executed_action_by_role": _role_stats(exec_actions0, roles),
                "executed_ctrl": _stats(ctrl0),
                "executed_ctrl_by_role": _role_stats(ctrl0, roles),
                "action_dim": action_dim,
            },
        }
        if policy is not None:
            policy_actions0 = _policy_actions(policy, state)
            actions[roles == "current_trained_policy_action"] = policy_actions0[
                roles == "current_trained_policy_action"
            ]
            exec_actions0 = env._actions_for_execution(actions, state.info)
            ctrl0 = exec_actions0 * action_scale[None, :] + default_angles[None, :]
            ctrl_minus_reset_dof = ctrl0 - reset_dof_pos
            details["module_probe"]["action_G1LOC_ACT_001"][
                "first_step_ctrl_minus_reset_dof"
            ] = {
                "by_role": _role_stats(ctrl_minus_reset_dof, roles),
                "top_abs_zero_action_joint_indices": _top_abs(
                    ctrl_minus_reset_dof[roles == "zero_action"]
                ),
                "top_abs_searched_support_joint_indices": _top_abs(
                    ctrl_minus_reset_dof[roles == "searched_support_action"]
                ),
                "top_abs_current_policy_joint_indices": _top_abs(
                    ctrl_minus_reset_dof[roles == "current_trained_policy_action"]
                ),
            }
            details["policy_actor"] = {
                "checkpoint_path": policy["checkpoint_path"],
                "actor_algo_type": policy["actor_algo_type"],
            }
            details["first_step_action"] = {
                "raw_action": _stats(actions),
                "raw_action_by_role": _role_stats(actions, roles),
                "executed_action_by_role": _role_stats(exec_actions0, roles),
                "executed_ctrl": _stats(ctrl0),
                "executed_ctrl_by_role": _role_stats(ctrl0, roles),
                "policy_action_top_abs_joint_indices": _top_abs(
                    actions[roles == "current_trained_policy_action"]
                ),
                "action_dim": action_dim,
            }
        if support_anchor_search is not None:
            details["module_probe"]["static_anchor_search_G1LOC_ACT_001"] = support_anchor_search
            support_actions = actions[roles == "searched_support_action"]
            support_ctrl = ctrl0[roles == "searched_support_action"]
            details["module_probe"]["support_action_candidate_G1LOC_ACT_001"] = {
                "selected_from_search_seed": seed + 1000,
                "action": {
                    "stats": _stats(support_actions),
                    "top_abs_joint_indices": _top_abs(support_actions),
                },
                "ctrl_minus_default": {
                    "stats": _stats(support_ctrl - default_angles[None, :]),
                    "top_abs_joint_indices": _top_abs(support_ctrl - default_angles[None, :]),
                },
                "searched_joint_indices": support_anchor_search["searched_joint_indices"],
            }
        if details["forbidden_reward_keys_present"]:
            failures.append("stand-still reward contains walking/gait/height tracking keys")
        if details["initial"]["commands_max_abs"] > 1.0e-7:
            failures.append("stand-still reset did not produce zero commands")

        completed_steps = 0
        for step in range(1, steps + 1):
            if policy is not None:
                policy_actions_step = _policy_actions(policy, state)
                actions[roles == "current_trained_policy_action"] = policy_actions_step[
                    roles == "current_trained_policy_action"
                ]
            state = env.step(actions)
            final_state = state
            completed_steps = step
            terms = _term_contributions(env, state)
            for name, value in terms.items():
                per_term_sum[name] = per_term_sum.get(name, 0.0) + value
                per_term_min[name] = min(per_term_min.get(name, value), value)
            role_terms = _term_contributions_by_role(env, state, roles)
            for name, role_values in role_terms.items():
                per_role = per_term_role_sum.setdefault(name, {})
                for role, value in role_values.items():
                    per_role[role] = per_role.get(role, 0.0) + value
            if first_termination_step is None and np.any(state.terminated):
                first_termination_step = step
                break

        assert final_state is not None
        final_snapshot = _snapshot(env, final_state)
        details["rollout"] = {
            "completed_steps": completed_steps,
            "first_termination_step": first_termination_step,
            "final": final_snapshot,
            "final_by_role": _snapshot_by_role(env, final_state, roles),
            "per_term_reward_mean_per_step": {
                name: value / max(completed_steps, 1) for name, value in sorted(per_term_sum.items())
            },
            "per_term_reward_mean_per_step_by_role": {
                name: {
                    role: value / max(completed_steps, 1)
                    for role, value in sorted(role_values.items())
                }
                for name, role_values in sorted(per_term_role_sum.items())
            },
            "per_term_reward_min_step_mean": dict(sorted(per_term_min.items())),
        }
        if probe_mode == "anchor-diff":
            details["module_probe"]["boundary_decision"] = _role_boundary_decision(
                details["rollout"]["final_by_role"]
            )
        if probe_mode in {"support-diff", "support-policy-diff"}:
            details["module_probe"]["boundary_decision"] = _role_boundary_decision(
                details["rollout"]["final_by_role"]
            )
        if probe_mode == "anchor-search":
            height_floor = (
                min_ok_base_height
                if min_ok_base_height is not None
                else float(env.cfg.reward_config.base_height_target) - 0.12
            )
            details["module_probe"]["static_anchor_search_G1LOC_ACT_001"] = _static_anchor_search(
                cfg,
                action_dim=action_dim,
                steps=steps,
                seed=seed + 1000,
                candidates=anchor_search_candidates,
                iterations=anchor_search_iterations,
                elite_count=anchor_search_elites,
                max_abs=anchor_search_max_abs,
                std_init=anchor_search_std,
                max_ok_tilt_deg=max_ok_tilt_deg,
                max_ok_base_over_feet_x_abs=max_ok_base_over_feet_x_abs,
                min_ok_base_height=height_floor,
                search_dim_override=None if anchor_search_dims <= 0 else anchor_search_dims,
            )
        if first_termination_step is not None:
            reason = details["rollout"]["final"]["termination"]["first_reason"]
            failures.append(f"stand-still zero-action rollout terminated at step {first_termination_step}: {reason}")
        if final_snapshot["tilt_deg"]["max"] > max_ok_tilt_deg:
            failures.append(
                "stand-still zero-action rollout exceeded stable tilt: "
                f"{final_snapshot['tilt_deg']['max']:.2f} > {max_ok_tilt_deg:.2f} deg"
            )
        if abs(final_snapshot["base_over_feet_x"]["max"]) > max_ok_base_over_feet_x_abs or (
            abs(final_snapshot["base_over_feet_x"]["min"]) > max_ok_base_over_feet_x_abs
        ):
            failures.append(
                "stand-still zero-action rollout exceeded base-over-feet x bound: "
                f"range=[{final_snapshot['base_over_feet_x']['min']:.4f}, "
                f"{final_snapshot['base_over_feet_x']['max']:.4f}], "
                f"bound={max_ok_base_over_feet_x_abs:.4f}"
            )
        height_floor = (
            min_ok_base_height
            if min_ok_base_height is not None
            else float(env.cfg.reward_config.base_height_target) - 0.12
        )
        if final_snapshot["base_height"]["min"] < height_floor:
            failures.append(
                "stand-still zero-action rollout dropped below stable height floor: "
                f"{final_snapshot['base_height']['min']:.4f} < {height_floor:.4f}"
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return failures, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-ok-tilt-deg", type=float, default=20.0)
    parser.add_argument("--max-ok-base-over-feet-x-abs", type=float, default=0.2)
    parser.add_argument("--min-ok-base-height", type=float, default=None)
    parser.add_argument(
        "--probe-mode",
        choices=("zero", "anchor-diff", "anchor-search", "support-diff", "support-policy-diff"),
        default="anchor-diff",
    )
    parser.add_argument("--anchor-search-candidates", type=int, default=48)
    parser.add_argument("--anchor-search-iterations", type=int, default=4)
    parser.add_argument("--anchor-search-elites", type=int, default=8)
    parser.add_argument("--anchor-search-max-abs", type=float, default=0.45)
    parser.add_argument("--anchor-search-std", type=float, default=0.16)
    parser.add_argument(
        "--anchor-search-dims",
        type=int,
        default=0,
        help="Number of leading action dims to search; <=0 means all action dims.",
    )
    args = parser.parse_args()

    failures, details = run_check(
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        max_ok_tilt_deg=args.max_ok_tilt_deg,
        max_ok_base_over_feet_x_abs=args.max_ok_base_over_feet_x_abs,
        min_ok_base_height=args.min_ok_base_height,
        probe_mode=args.probe_mode,
        anchor_search_candidates=args.anchor_search_candidates,
        anchor_search_iterations=args.anchor_search_iterations,
        anchor_search_elites=args.anchor_search_elites,
        anchor_search_max_abs=args.anchor_search_max_abs,
        anchor_search_std=args.anchor_search_std,
        anchor_search_dims=args.anchor_search_dims,
    )
    print("G1 stand-still live sentinel")
    print(json.dumps(details, indent=2, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[PASS] G1StandStill real MuJoCo zero-action sentinel completed without termination")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
