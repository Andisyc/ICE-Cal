#!/usr/bin/env python3
"""Audit the generic G1 distillation teacher observation contract."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level, name, detail))


def _compose_cfg(task: str) -> Any:
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "distill"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"task={task}"])


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _first_actor_input_dim(checkpoint_path: str | Path) -> tuple[int | None, str | None]:
    from unilab.algos.torch.distill import inspect_sac_teacher_checkpoint

    try:
        info = inspect_sac_teacher_checkpoint(checkpoint_path)
    except ValueError:
        return None, None
    return info.actor_input_dim, info.first_weight_key


def run_check(
    *,
    task: str = "g1_walk_height/mujoco",
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    cfg: Any | None = None,
    create_env_fn: Any | None = None,
    env_cfg_override_fn: Any | None = None,
) -> tuple[list[Check], dict[str, Any]]:
    cfg = cfg if cfg is not None else _compose_cfg(task)
    if device is not None:
        cfg.training.device = str(device)
    teacher_dim = int(cfg.teacher.obs_dim)
    student_dim = int(cfg.student.obs_dim)
    teacher_projection = str(
        OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity")
    )

    if create_env_fn is None:
        ensure_registries()
        create_env_fn = create_env
    if env_cfg_override_fn is None:
        env_cfg_override_fn = lambda cfg: BackendAdapter(  # noqa: E731
            cfg,
            root_dir=ROOT_DIR,
            algo_name="distill",
        ).build_task_env_cfg_override()

    checks: list[Check] = []
    details: dict[str, Any] = {
        "distill_teacher_obs/task_owner": str(task),
        "distill_teacher_obs/task": str(OmegaConf.select(cfg, "training.task_name")),
        "distill_teacher_obs/device": None if device is None else str(device),
        "distill_teacher_obs/teacher_obs_dim": teacher_dim,
        "distill_teacher_obs/student_obs_dim": student_dim,
        "distill_teacher_obs/teacher_projection": teacher_projection,
        "distill_teacher_obs/checkpoint_path": None
        if checkpoint_path is None
        else str(checkpoint_path),
    }

    env = create_env_fn(
        cfg,
        num_envs=1,
        env_cfg_override=env_cfg_override_fn(cfg),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        task_name=str(OmegaConf.select(cfg, "training.task_name")),
    )
    try:
        spec = dict(getattr(env, "obs_groups_spec"))
        obs_dim = int(spec["obs"])
        critic_dim = int(spec["critic"])
        obs, _info = env.reset(np.arange(1, dtype=np.int32))
        obs_shape = tuple(np.asarray(obs["obs"]).shape)
        critic_shape = tuple(np.asarray(obs["critic"]).shape)
    finally:
        _close_env(env)

    details.update(
        {
            "distill_teacher_obs/env_obs_groups_spec": spec,
            "distill_teacher_obs/live_obs_shape": obs_shape,
            "distill_teacher_obs/live_critic_shape": critic_shape,
        }
    )

    if obs_dim == student_dim:
        _add(checks, "PASS", "distill_teacher_obs/student_live_dim", str(obs_dim))
    else:
        _add(
            checks,
            "FAIL",
            "distill_teacher_obs/student_live_dim",
            f"student={student_dim} live_obs={obs_dim}",
        )

    if obs_shape == (1, obs_dim) and critic_shape == (1, critic_dim):
        _add(
            checks,
            "PASS",
            "distill_teacher_obs/live_reset_shapes",
            f"obs={obs_shape} critic={critic_shape}",
        )
    else:
        _add(
            checks,
            "FAIL",
            "distill_teacher_obs/live_reset_shapes",
            f"obs={obs_shape} critic={critic_shape} spec={spec}",
        )

    if teacher_dim == obs_dim and teacher_projection == "identity":
        _add(checks, "PASS", "distill_teacher_obs/teacher_live_dim", str(teacher_dim))
    elif teacher_dim > obs_dim and teacher_projection == "pad_zeros":
        _add(
            checks,
            "WARN",
            "distill_teacher_obs/projection_bridge",
            f"live_obs={obs_dim} -> teacher={teacher_dim}; synthetic_tail={teacher_dim - obs_dim}",
        )
    else:
        _add(
            checks,
            "FAIL",
            "distill_teacher_obs/teacher_live_dim",
            f"teacher={teacher_dim} live_obs={obs_dim} projection={teacher_projection}",
        )

    if checkpoint_path is None:
        _add(checks, "WARN", "distill_teacher_obs/checkpoint_input_dim", "not inspected")
    else:
        first_dim, first_key = _first_actor_input_dim(checkpoint_path)
        details["distill_teacher_obs/checkpoint_first_weight"] = first_key
        details["distill_teacher_obs/checkpoint_input_dim"] = first_dim
        if first_dim == teacher_dim:
            _add(
                checks,
                "PASS",
                "distill_teacher_obs/checkpoint_input_dim",
                f"{first_key}={first_dim}",
            )
        else:
            _add(
                checks,
                "FAIL",
                "distill_teacher_obs/checkpoint_input_dim",
                f"expected={teacher_dim} got={first_dim} key={first_key}",
            )

    return checks, details


def print_report(checks: list[Check], details: dict[str, Any]) -> None:
    print("UniLab G1 generic distill teacher obs contract audit")
    for key, value in details.items():
        print(f"{key}: {value}")
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="g1_walk_height/mujoco")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks, details = run_check(
        task=args.task,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    print_report(checks, details)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
