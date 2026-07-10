#!/usr/bin/env python3
"""Live MuJoCo sentinel for generic G1 distillation playback."""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.visualization.interactive_playback import (  # noqa: E402
    RslRlPlaybackConfig,
    create_distill_playback_session,
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


def _compose_cfg(task: str = "g1_walk_height/mujoco"):
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "distill"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"task={task}"])


def _finite_array(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return False
    return bool(arr.size > 0 and np.all(np.isfinite(arr)))


def _shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return list(np.asarray(value).shape)


def _array_on_cpu(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
    try:
        return np.asarray(value, dtype=np.float64)
    except Exception:
        return None


def _make_temp_student_checkpoint(
    cfg: Any,
    run_dir: Path,
    *,
    student_model_type: str = "mlp",
) -> Path:
    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        MoEStudentPolicy,
        save_distillation_checkpoint,
    )

    if student_model_type == "mlp":
        student = MLPStudentPolicy(
            obs_dim=int(cfg.student.obs_dim),
            action_dim=int(cfg.student.action_dim),
            hidden_dims=tuple(int(dim) for dim in cfg.student.hidden_dims),
            activation=str(cfg.student.activation),
            squash_action=bool(cfg.student.squash_action),
        )
    elif student_model_type == "moe":
        student = MoEStudentPolicy(
            obs_dim=int(cfg.student.obs_dim),
            action_dim=int(cfg.student.action_dim),
            num_experts=int(cfg.student.num_experts),
            expert_hidden_dims=tuple(int(dim) for dim in cfg.student.expert_hidden_dims),
            router_hidden_dims=tuple(int(dim) for dim in cfg.student.router_hidden_dims),
            activation=str(cfg.student.activation),
            squash_action=bool(cfg.student.squash_action),
            routing_mode=str(cfg.student.routing_mode),
            router_temperature=float(cfg.student.router_temperature),
        )
    else:
        raise ValueError(f"Unsupported temp student model type: {student_model_type!r}")
    for param in student.parameters():
        param.data.zero_()
    if student_model_type == "moe":
        for expert in student.experts:
            linear_layers = [
                module for module in expert.modules() if isinstance(module, nn.Linear)
            ]
            if linear_layers and linear_layers[-1].bias is not None:
                linear_layers[-1].bias.data.fill_(0.05)
    else:
        linear_layers = [module for module in student.modules() if isinstance(module, nn.Linear)]
        if linear_layers and linear_layers[-1].bias is not None:
            linear_layers[-1].bias.data.fill_(0.05)

    checkpoint_path = run_dir / "model_1.pt"
    distill_runtime_cfg = {
        "student_model_type": student_model_type,
        "student_obs_dim": int(cfg.student.obs_dim),
        "student_action_dim": int(cfg.student.action_dim),
        "student_activation": str(cfg.student.activation),
        "student_squash_action": bool(cfg.student.squash_action),
    }
    if student_model_type == "mlp":
        distill_runtime_cfg["student_hidden_dims"] = [
            int(dim) for dim in cfg.student.hidden_dims
        ]
    else:
        distill_runtime_cfg.update(
            {
                "student_num_experts": int(cfg.student.num_experts),
                "student_expert_hidden_dims": [
                    int(dim) for dim in cfg.student.expert_hidden_dims
                ],
                "student_router_hidden_dims": [
                    int(dim) for dim in cfg.student.router_hidden_dims
                ],
                "student_routing_mode": str(cfg.student.routing_mode),
                "student_router_temperature": float(cfg.student.router_temperature),
            }
        )
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=1,
        teacher_metadata={
            "sentinel": "temp_policy_checkpoint",
            "student_model_type": student_model_type,
        },
        distill_runtime_cfg=distill_runtime_cfg,
    )
    return checkpoint_path


def run_check(
    *,
    steps: int,
    action_mode: str,
    load_run: str,
    checkpoint: str | None,
    task: str = "g1_walk_height/mujoco",
    device: str | None = "cpu",
    make_temp_policy_checkpoint: bool = False,
    temp_student_model_type: str = "mlp",
    create_session_fn=create_distill_playback_session,
) -> tuple[list[Check], dict[str, Any]]:
    cfg = _compose_cfg(task)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    temp_checkpoint_path: Path | None = None
    if make_temp_policy_checkpoint:
        if action_mode != "policy":
            raise ValueError("--make-temp-policy-checkpoint requires --action-mode policy")
        temp_dir = tempfile.TemporaryDirectory(prefix="unilab-distill-policy-")
        temp_run_dir = Path(temp_dir.name)
        temp_checkpoint_path = _make_temp_student_checkpoint(
            cfg,
            temp_run_dir,
            student_model_type=temp_student_model_type,
        )
        load_run = str(temp_run_dir)
        checkpoint = None

    playback_cfg = RslRlPlaybackConfig(
        task=str(cfg.training.task_name),
        load_run=str(load_run),
        checkpoint=checkpoint,
        action_mode=str(action_mode),
        policy_obs_mode=str(cfg.interactive.policy_obs_mode),
        algo_log_name=str(cfg.algo.algo_log_name),
        log_root=str(cfg.training.log_root) if cfg.training.log_root is not None else None,
        num_envs=1,
        speed=1.0,
        start_paused=True,
    )
    messages: list[str] = []
    checks: list[Check] = []
    details: dict[str, Any] = {
        "distill_playback/task": str(cfg.training.task_name),
        "distill_playback/action_mode": str(action_mode),
        "distill_playback/task_owner": str(task),
        "distill_playback/load_run": str(load_run),
        "distill_playback/checkpoint": checkpoint,
        "distill_playback/steps": int(steps),
        "distill_playback/device": device,
        "distill_playback/cfg_student_obs_dim": int(cfg.student.obs_dim),
        "distill_playback/cfg_teacher_obs_dim": int(cfg.teacher.obs_dim),
        "distill_playback/temp_policy_checkpoint": (
            None if temp_checkpoint_path is None else str(temp_checkpoint_path)
        ),
        "distill_playback/temp_student_model_type": str(temp_student_model_type),
    }

    session = None
    try:
        playback_device = (
            device
            if device is not None
            else (str(cfg.training.device) if cfg.training.device is not None else None)
        )
        session, policy_obs_mode, checkpoint_path = create_session_fn(
            playback_cfg=playback_cfg,
            cfg=cfg,
            root_dir=ROOT_DIR,
            device=playback_device,
            log=messages.append,
        )
        env = session.env
        action_dim = int(env.action_space.shape[0])
        session.reset()
        for _ in range(steps):
            session.step_once()
        physics = session.physics_state()
        info = getattr(session, "info", {})
        actions = getattr(session, "actions", None)
        actions_arr = _array_on_cpu(actions)
        actions_abs_max = None if actions_arr is None else float(np.max(np.abs(actions_arr)))

        details.update(
            {
                "distill_playback/policy_obs_mode": policy_obs_mode,
                "distill_playback/checkpoint_path": checkpoint_path,
                "distill_playback/action_dim": action_dim,
                "distill_playback/physics_shape": _shape(physics),
                "distill_playback/actions_shape": _shape(actions),
                "distill_playback/actions_abs_max": actions_abs_max,
                "distill_playback/info_keys": sorted(info.keys())[:12] if isinstance(info, dict) else [],
                "distill_playback/messages": messages,
            }
        )

        if policy_obs_mode == "actor":
            _add(checks, "PASS", "distill_playback/policy_obs_mode", "actor")
        else:
            _add(checks, "FAIL", "distill_playback/policy_obs_mode", str(policy_obs_mode))

        if action_dim == 29:
            _add(checks, "PASS", "distill_playback/action_dim", "29")
        else:
            _add(checks, "FAIL", "distill_playback/action_dim", str(action_dim))

        if _finite_array(physics) and np.asarray(physics).ndim == 2:
            _add(checks, "PASS", "distill_playback/physics_state", str(np.asarray(physics).shape))
        else:
            _add(checks, "FAIL", "distill_playback/physics_state", str(np.asarray(physics).shape))

        if actions is not None and _finite_array(actions) and tuple(_shape(actions) or ()) == (1, action_dim):
            _add(checks, "PASS", "distill_playback/actions", str(tuple(_shape(actions) or ())))
        else:
            _add(
                checks,
                "FAIL",
                "distill_playback/actions",
                "missing or wrong action shape",
            )

        if action_mode == "policy":
            if checkpoint_path is not None:
                _add(checks, "PASS", "distill_playback/policy_checkpoint", str(checkpoint_path))
            else:
                _add(checks, "FAIL", "distill_playback/policy_checkpoint", "missing")
            if actions_abs_max is not None and actions_abs_max > 1.0e-6:
                _add(checks, "PASS", "distill_playback/policy_action_nonzero", f"{actions_abs_max:.6f}")
            else:
                _add(checks, "FAIL", "distill_playback/policy_action_nonzero", str(actions_abs_max))
    finally:
        if session is not None:
            _close_env(session.env)
        if temp_dir is not None:
            temp_dir.cleanup()

    return checks, details


def print_report(checks: list[Check], details: dict[str, Any]) -> None:
    print("UniLab G1 generic distill playback live sentinel")
    for key, value in details.items():
        print(f"{key}: {value}")
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--task", default="g1_walk_height/mujoco")
    parser.add_argument("--action-mode", choices=("zero", "policy"), default="zero")
    parser.add_argument("--load-run", default="-1")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--make-temp-policy-checkpoint", action="store_true")
    parser.add_argument("--temp-student-model-type", choices=("mlp", "moe"), default="mlp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks, details = run_check(
        steps=args.steps,
        task=args.task,
        action_mode=args.action_mode,
        load_run=args.load_run,
            checkpoint=args.checkpoint,
            device=args.device,
            make_temp_policy_checkpoint=bool(args.make_temp_policy_checkpoint),
            temp_student_model_type=str(args.temp_student_model_type),
        )
    print_report(checks, details)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
