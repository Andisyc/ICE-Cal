#!/usr/bin/env python3
"""Same-snapshot paired evaluation for the Phase-1 privileged residual teacher."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.common.actor_factory import build_actor  # noqa: E402
from unilab.algos.torch.fada_context.formal_protocol import (  # noqa: E402
    FORMAL_AGGREGATION,
    FORMAL_EVALUATION_NUM_ENVS,
    FORMAL_EVALUATION_SEEDS,
    FORMAL_EVALUATION_STEPS,
    FORMAL_TASK_CONFIG,
    assess_phase1_teacher_quality,
    formal_quality_thresholds,
    validate_phase1_formal_evaluation_contract,
)
from unilab.algos.torch.fada_context.paired_evaluation import (  # noqa: E402
    aggregate_paired_reports,
    evaluate_paired_rollouts,
)
from unilab.algos.torch.fada_context.privileged_residual_sac import (  # noqa: E402
    PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE,
    PrivilegedResidualSACActor,
    load_privileged_residual_actor_checkpoint,
)
from unilab.algos.torch.offpolicy.runtime import (  # noqa: E402
    resolve_custom_offpolicy_runtime,
)
from unilab.base.observations import get_obs_dims  # noqa: E402
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402
from unilab.training.seed import apply_training_seed  # noqa: E402

DEFAULT_TASK_CONFIG = FORMAL_TASK_CONFIG


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--num-envs", type=int, default=FORMAL_EVALUATION_NUM_ENVS)
    parser.add_argument("--steps", type=int, default=FORMAL_EVALUATION_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_EVALUATION_SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError(f"--num-envs must be positive, got {args.num_envs}")
    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain non-negative integers")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique")


def _compose_cfg(task_config: str) -> Any:
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / "offpolicy"),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=[f"task={task_config}"])
    if str(cfg.algo.runtime_impl) != PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE:
        raise ValueError("Phase-1 evaluation requires algo.runtime_impl=privileged_residual_sac")
    return cfg


def _env_override(cfg: Any) -> dict[str, Any] | None:
    return cast(
        dict[str, Any] | None,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="sac").build_task_env_cfg_override(),
    )


def _create_seed_env(cfg: Any, *, num_envs: int, seed: int, env_override: Any) -> Any:
    apply_training_seed(seed, torch_runtime=True, cuda=False)
    env = create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=env_override,
        sim_backend="mujoco",
    )
    env.init_state()
    return env


def _build_teacher_actor(
    cfg: Any,
    env: Any,
    *,
    checkpoint_path: Path,
    device: str,
) -> tuple[PrivilegedResidualSACActor, dict[str, Any]]:
    obs_dim, critic_obs_dim = get_obs_dims(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("Phase-1 evaluation environment action_space.shape is missing")
    action_dim = int(action_shape[0])
    rl_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
    runtime = resolve_custom_offpolicy_runtime(rl_cfg)
    if runtime is None or runtime.algo_type != PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE:
        raise ValueError("Phase-1 evaluation could not resolve privileged residual SAC runtime")
    actor_kwargs = runtime.build_model_kwargs(
        obs_dim=int(obs_dim),
        critic_obs_dim=int(critic_obs_dim),
    )
    actor = build_actor(
        PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE,
        int(obs_dim),
        action_dim,
        int(cfg.algo.actor_hidden_dim),
        bool(cfg.algo.use_layer_norm),
        device,
        **actor_kwargs,
    )
    if not isinstance(actor, PrivilegedResidualSACActor):
        raise TypeError("Phase-1 runtime did not build PrivilegedResidualSACActor")
    checkpoint_identity = load_privileged_residual_actor_checkpoint(
        actor,
        checkpoint_path,
        device=device,
    )
    return actor, checkpoint_identity


def _evaluation_contract(cfg: Any, args: argparse.Namespace) -> dict[str, Any]:
    strength = cfg.env.domain_rand.actuator_strength
    return {
        "task_config": str(args.task_config),
        "task_name": str(cfg.training.task_name),
        "sim_backend": "mujoco",
        "num_envs_per_seed": int(args.num_envs),
        "steps": int(args.steps),
        "seeds": [int(seed) for seed in args.seeds],
        "aggregation": FORMAL_AGGREGATION,
        "device": str(args.device),
        "command": list(cfg.env.commands.vel_limit[0]),
        "actuator_strength": {
            "sampling_mode": str(strength.sampling_mode),
            "candidate_actuator_indices": list(strength.candidate_actuator_indices),
            "multiplier_range": list(strength.multiplier_range),
            "nominal_probability": float(strength.nominal_probability),
        },
        "quality_thresholds_defined": True,
        "quality_thresholds": formal_quality_thresholds(),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    checkpoint_path = args.teacher_checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint_path}")
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.with_name(f"{checkpoint_path.stem}_paired_evaluation.json")
    )

    ensure_registries()
    cfg = _compose_cfg(str(args.task_config))
    env_override = _env_override(cfg)
    actor: PrivilegedResidualSACActor | None = None
    checkpoint_identity: dict[str, Any] | None = None
    seed_reports: list[dict[str, Any]] = []
    for seed in args.seeds:
        env = _create_seed_env(
            cfg,
            num_envs=int(args.num_envs),
            seed=int(seed),
            env_override=env_override,
        )
        try:
            if actor is None:
                actor, checkpoint_identity = _build_teacher_actor(
                    cfg,
                    env,
                    checkpoint_path=checkpoint_path,
                    device=str(args.device),
                )
            report = evaluate_paired_rollouts(
                env,
                actor,
                steps=int(args.steps),
                device=str(args.device),
            )
            report["seed"] = int(seed)
            seed_reports.append(report)
        finally:
            env.close()

    assert checkpoint_identity is not None
    aggregate = aggregate_paired_reports(seed_reports)
    evaluation_contract = _evaluation_contract(cfg, args)
    protocol_validation = validate_phase1_formal_evaluation_contract(evaluation_contract)
    quality_assessment = assess_phase1_teacher_quality(aggregate, evaluation_contract)
    evaluation_contract["quality_status"] = quality_assessment["quality_status"]
    payload = {
        "schema": "unilab_context_teacher_phase1_evaluation_run_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_checkpoint": checkpoint_identity,
        "contract": evaluation_contract,
        "protocol_validation": protocol_validation,
        "quality_assessment": quality_assessment,
        "per_seed": seed_reports,
        "aggregate": aggregate,
    }
    _write_json_atomic(output_path, payload)

    overall = payload["aggregate"]
    nominal = overall["nominal"]["overall"]
    teacher = overall["teacher"]["overall"]
    print(f"Paired evaluation written to: {output_path}")
    print(f"Seeds: {overall['seeds']} | scenario_counts={overall['scenario_counts']}")
    print(
        "Forward velocity MAE: "
        f"nominal={nominal['forward_velocity_mae_mps']:.6f}, "
        f"teacher={teacher['forward_velocity_mae_mps']:.6f}"
    )
    print(
        "Max lateral displacement: "
        f"nominal={nominal['max_lateral_abs_m']:.6f}, "
        f"teacher={teacher['max_lateral_abs_m']:.6f}"
    )
    print(f"Fall rate: nominal={nominal['fall_rate']:.6f}, teacher={teacher['fall_rate']:.6f}")
    print(
        "Teacher residual/clipping: "
        f"l2_mean={teacher['residual_l2_mean']:.6f}, "
        f"element_clip_rate={teacher['clipping_element_rate']:.6f}, "
        f"step_clip_rate={teacher['clipping_step_rate']:.6f}"
    )
    quality_status = str(quality_assessment["quality_status"])
    print(f"Quality status: {quality_status}")
    if quality_status == "failed":
        print(f"Failed checks: {quality_assessment['failed_checks']}")
        return 2
    if quality_status == "unassessed":
        print(f"Protocol mismatches: {quality_assessment['protocol_mismatches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
