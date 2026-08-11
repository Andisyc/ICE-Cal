#!/usr/bin/env python3
"""Live-path sentinel for the G1 walk-height and stand-height tasks.

Status: active bounded runtime probe. This validates route identity and compact
runtime facts; it does not validate policy quality or start training.
"""

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


@dataclass(frozen=True)
class TaskContract:
    owner_choice: str
    registry_task_name: str
    actor_obs_dim: int
    critic_obs_dim: int
    require_zero_velocity: bool


TASK_CONTRACTS = {
    "g1_walk_height": TaskContract(
        owner_choice="sac/g1_walk_height/mujoco",
        registry_task_name="G1WalkHeight",
        actor_obs_dim=99,
        critic_obs_dim=102,
        require_zero_velocity=False,
    ),
    "g1_stand_height": TaskContract(
        owner_choice="sac/g1_stand_height/mujoco",
        registry_task_name="G1StandHeight",
        actor_obs_dim=99,
        critic_obs_dim=102,
        require_zero_velocity=True,
    ),
}


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level, name, detail))


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _task_contract(task: str) -> TaskContract:
    try:
        return TASK_CONTRACTS[task]
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_CONTRACTS))
        raise ValueError(
            f"unsupported height-tracking task {task!r}; choose one of: {choices}"
        ) from exc


def _compose_cfg(task: str):
    contract = _task_contract(task)
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"task={contract.owner_choice}"])


def _stats(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return f"min={np.min(arr):.6f}, max={np.max(arr):.6f}, mean={np.mean(arr):.6f}"


def _stand_runtime_snapshot(env: Any, state: Any) -> dict[str, Any]:
    num_envs = int(env.num_envs)
    upvector = np.asarray(
        env._backend.get_sensor_data(env.cfg.sensor.upvector), dtype=np.float32
    ).reshape(num_envs, -1)
    tilt_deg = np.rad2deg(np.arccos(np.clip(upvector[:, 2], -1.0, 1.0))).astype(np.float32)

    def _contact_count(prefix: str) -> np.ndarray:
        active = []
        for index in range(4):
            value = np.asarray(
                env._backend.get_sensor_data(f"{prefix}_foot_contact_{index}"),
                dtype=np.float32,
            ).reshape(num_envs, -1)
            active.append(np.any(value > 0.5, axis=1))
        return np.sum(np.stack(active, axis=1), axis=1)

    left_contact = _contact_count("left")
    right_contact = _contact_count("right")
    both_feet_contact = (left_contact > 0) & (right_contact > 0)
    terminated = np.asarray(state.terminated, dtype=bool).reshape(num_envs)
    return {
        "height_tracking/tilt_deg_min_max_mean": _stats(tilt_deg),
        "height_tracking/double_support_fraction": float(np.mean(both_feet_contact)),
        "height_tracking/terminated_total": int(np.count_nonzero(terminated)),
        "height_tracking/runtime_snapshot_finite": bool(
            np.all(np.isfinite(tilt_deg))
            and np.all(np.isfinite(left_contact))
            and np.all(np.isfinite(right_contact))
        ),
    }


def run_check(
    *,
    num_envs: int,
    steps: int,
    seed: int,
    task: str = "g1_walk_height",
    create_env_fn: Any = create_env,
) -> tuple[list[Check], dict[str, Any]]:
    np.random.seed(seed)
    contract = _task_contract(task)
    cfg = _compose_cfg(task)
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="sac")
    env_override = adapter.build_task_env_cfg_override()

    checks: list[Check] = []
    details: dict[str, Any] = {
        "height_tracking/task_owner": task,
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
        env = create_env_fn(
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
        target_obs = obs[:, 96]
        target_height = height_commands[:, 0]
        height_range = np.asarray(cfg.env.commands.height_range, dtype=np.float32)
        height_error = measured_height - target_height

        details.update(
            {
                "height_tracking/commands_shape": list(commands.shape),
                "height_tracking/height_commands_shape": list(height_commands.shape),
                "height_tracking/target_height_min_max_mean": _stats(height_commands[:, 0]),
                "height_tracking/measured_height_min_max_mean": _stats(measured_height),
                "height_tracking/height_error_min_max_mean": _stats(height_error),
                "height_tracking/commands_max_abs": float(np.max(np.abs(commands))),
                "height_tracking/reward_mean": float(np.mean(reward)),
                "height_tracking/obs_dim": int(obs.shape[1]),
                "height_tracking/critic_dim": int(critic.shape[1]),
                "height_tracking/finite": bool(
                    np.all(np.isfinite(obs))
                    and np.all(np.isfinite(critic))
                    and np.all(np.isfinite(reward))
                    and np.all(np.isfinite(height_commands))
                    and np.all(np.isfinite(measured_height))
                    and np.all(np.isfinite(target_obs))
                ),
                "height_tracking/log_reward": float(
                    log.get("reward/track_base_height_exp_smooth", np.nan)
                ),
            }
        )
        if contract.require_zero_velocity:
            details.update(_stand_runtime_snapshot(env, state))

        if str(cfg.training.task_name) == contract.registry_task_name:
            _add(
                checks,
                "PASS",
                "height_tracking/task_identity",
                f"{task} -> {contract.registry_task_name}",
            )
        else:
            _add(
                checks,
                "FAIL",
                "height_tracking/task_identity",
                f"expected={contract.registry_task_name}, actual={cfg.training.task_name}",
            )

        if (
            env.obs_groups_spec
            == {"obs": contract.actor_obs_dim, "critic": contract.critic_obs_dim}
            and obs.shape[1] == contract.actor_obs_dim
            and critic.shape[1] == contract.critic_obs_dim
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

        if np.allclose(target_obs, target_height):
            _add(
                checks,
                "PASS",
                "height_tracking/target_obs_index_96",
                "actor observation matches height_commands[:, 0]",
            )
        else:
            _add(
                checks,
                "FAIL",
                "height_tracking/target_obs_index_96",
                f"obs={target_obs[:2].tolist()}, target={target_height[:2].tolist()}",
            )

        target_in_range = bool(
            np.all(target_height >= height_range[0]) and np.all(target_height <= height_range[1])
        )
        _add(
            checks,
            "PASS" if target_in_range else "FAIL",
            "height_tracking/target_range",
            f"configured={height_range.tolist()}, observed={_stats(target_height)}",
        )

        if contract.require_zero_velocity:
            commands_are_zero = bool(np.allclose(commands, 0.0))
            _add(
                checks,
                "PASS" if commands_are_zero else "FAIL",
                "height_tracking/zero_velocity_command",
                f"max_abs={details['height_tracking/commands_max_abs']:.6f}",
            )
            runtime_finite = bool(details["height_tracking/runtime_snapshot_finite"])
            _add(
                checks,
                "PASS" if runtime_finite else "FAIL",
                "height_tracking/runtime_snapshot_finite",
                (
                    f"tilt={details['height_tracking/tilt_deg_min_max_mean']}, "
                    f"double_support={details['height_tracking/double_support_fraction']:.6f}"
                ),
            )
            terminated_total = int(details["height_tracking/terminated_total"])
            _add(
                checks,
                "PASS" if terminated_total == 0 else "FAIL",
                "height_tracking/one_step_termination",
                f"terminated_total={terminated_total}",
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
    parser.add_argument(
        "--task",
        choices=sorted(TASK_CONTRACTS),
        default="g1_walk_height",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks, details = run_check(
        task=args.task,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
    )
    print_report(checks, details)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
