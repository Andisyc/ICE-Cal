#!/usr/bin/env python3
"""Live-path sentinel for the G1 height tracking task."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.training import (  # noqa: E402
    BackendAdapter,
    assert_offpolicy_task_choice_matches_algo,
    create_env,
    ensure_registries,
)


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level, name, detail))


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _compose_cfg():
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        return compose(config_name="config", overrides=["task=sac/g1_walk_height/mujoco"])


def _stats(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return f"min={np.min(arr):.6f}, max={np.max(arr):.6f}, mean={np.mean(arr):.6f}"


def run_check(*, num_envs: int, steps: int, seed: int) -> tuple[list[Check], dict[str, Any]]:
    np.random.seed(seed)
    cfg = _compose_cfg()
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="sac")
    env_override = adapter.build_task_env_cfg_override()

    checks: list[Check] = []
    details: dict[str, Any] = {
        "height_tracking/config_task": str(cfg.training.task_name),
        "height_tracking/num_envs": num_envs,
        "height_tracking/steps": steps,
        "height_tracking/seed": seed,
        "height_tracking/observe_height_command": env_override.get("commands", {}).get(
            "observe_height_command"
        ),
        "height_tracking/random_height_during_walking": env_override.get("commands", {}).get(
            "random_height_during_walking"
        ),
    }

    env = None
    try:
        ensure_registries()
        env = create_env(
            cfg,
            num_envs=num_envs,
            env_cfg_override=env_override,
            sim_backend="mujoco",
        )
        state = env.init_state()
        if "steps" in state.info:
            state.info["steps"].fill(0)

        actions = np.zeros((num_envs, env.action_space.shape[0]), dtype=np.float32)
        for _ in range(steps):
            state = env.step(actions)

        obs = np.asarray(state.obs["obs"], dtype=np.float32)
        critic = np.asarray(state.obs["critic"], dtype=np.float32)
        info = state.info
        commands = np.asarray(info.get("commands"), dtype=np.float32)
        height_commands = np.asarray(info.get("height_commands"), dtype=np.float32)
        measured_height = np.asarray(env._terrain_relative_base_height(), dtype=np.float32)
        reward = np.asarray(state.reward, dtype=np.float32)
        log = info.get("log", {})
        command_start = 3 + 3 + env.action_space.shape[0] * 3
        command_obs = obs[:, command_start : command_start + 4]
        expected_command_obs = np.concatenate([commands, height_commands], axis=1)

        details.update(
            {
                "height_tracking/commands_shape": list(commands.shape),
                "height_tracking/height_commands_shape": list(height_commands.shape),
                "height_tracking/target_height_min_max_mean": _stats(height_commands[:, 0]),
                "height_tracking/measured_height_min_max_mean": _stats(measured_height),
                "height_tracking/reward_mean": float(np.mean(reward)),
                "height_tracking/obs_dim": int(obs.shape[1]),
                "height_tracking/critic_dim": int(critic.shape[1]),
                "height_tracking/finite": bool(
                    np.all(np.isfinite(obs))
                    and np.all(np.isfinite(critic))
                    and np.all(np.isfinite(reward))
                    and np.all(np.isfinite(height_commands))
                ),
                "height_tracking/log_reward": float(
                    log.get("reward/track_base_height_exp_smooth", np.nan)
                ),
            }
        )

        if (
            env.obs_groups_spec == {"obs": 100, "critic": 103}
            and obs.shape[1] == 100
            and critic.shape[1] == 103
        ):
            _add(checks, "PASS", "height_tracking/obs_contract", str(env.obs_groups_spec))
        else:
            _add(
                checks,
                "FAIL",
                "height_tracking/obs_contract",
                f"spec={env.obs_groups_spec}, obs={obs.shape}, critic={critic.shape}",
            )

        if commands.shape == (num_envs, 3) and height_commands.shape == (num_envs, 1):
            _add(
                checks,
                "PASS",
                "height_tracking/commands_shape",
                f"commands={commands.shape}, height={height_commands.shape}",
            )
        else:
            _add(
                checks,
                "FAIL",
                "height_tracking/commands_shape",
                f"commands={commands.shape}, height={height_commands.shape}",
            )

        if np.allclose(command_obs, expected_command_obs):
            _add(
                checks,
                "PASS",
                "height_tracking/obs_command_block",
                "matches [vx, vy, yaw, target_height]",
            )
        else:
            _add(
                checks,
                "FAIL",
                "height_tracking/obs_command_block",
                f"head={command_obs[:2].tolist()}",
            )

        if details["height_tracking/finite"]:
            _add(
                checks,
                "PASS",
                "height_tracking/finite",
                "obs, critic, reward, height commands finite",
            )
        else:
            _add(checks, "FAIL", "height_tracking/finite", "non-finite value found")

        if np.isfinite(details["height_tracking/log_reward"]):
            _add(
                checks,
                "PASS",
                "height_tracking/reward_log",
                f"{details['height_tracking/log_reward']:.6f}",
            )
        else:
            _add(
                checks,
                "FAIL",
                "height_tracking/reward_log",
                "missing reward/track_base_height_exp_smooth",
            )
    finally:
        if env is not None:
            _close_env(env)

    return checks, details


def print_report(checks: list[Check], details: dict[str, Any]) -> None:
    print("UniLab G1 height tracking live-path sentinel")
    for key, value in details.items():
        print(f"{key}: {value}")
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks, details = run_check(num_envs=args.num_envs, steps=args.steps, seed=args.seed)
    print_report(checks, details)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
