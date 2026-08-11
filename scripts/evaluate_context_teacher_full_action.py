#!/usr/bin/env python3
"""Evaluate a formal full-action teacher against the original policy at fixed 0.9."""

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
from unilab.algos.torch.fada_context.full_action_formal_protocol import (  # noqa: E402
    FORMAL_AGGREGATION,
    FORMAL_EVALUATION_NUM_ENVS,
    FORMAL_EVALUATION_SEEDS,
    FORMAL_EVALUATION_STEPS,
    FORMAL_STRENGTH,
    FORMAL_TASK_CONFIG,
    assess_full_action_teacher_quality,
    formal_quality_thresholds,
    validate_full_action_formal_evaluation_contract,
)
from unilab.algos.torch.fada_context.full_action_paired_evaluation import (  # noqa: E402
    aggregate_full_action_paired_reports,
    evaluate_full_action_paired_rollouts,
)
from unilab.algos.torch.fada_context.privileged_full_action_sac import (  # noqa: E402
    PRIVILEGED_FULL_ACTION_SAC_ALGO_TYPE,
    PrivilegedFullActionSACActor,
    load_privileged_full_action_actor_checkpoint,
)
from unilab.algos.torch.fast_sac.learner import SACActor  # noqa: E402
from unilab.algos.torch.offpolicy.runtime import resolve_custom_offpolicy_runtime  # noqa: E402
from unilab.base.observations import get_obs_dims  # noqa: E402
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402
from unilab.training.seed import apply_training_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=FORMAL_EVALUATION_NUM_ENVS)
    parser.add_argument("--steps", type=int, default=FORMAL_EVALUATION_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_EVALUATION_SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _compose_cfg() -> Any:
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / "offpolicy"),
        version_base="1.3",
    ):
        return compose(config_name="config", overrides=[f"task={FORMAL_TASK_CONFIG}"])


def _build_actors(
    cfg: Any, env: Any, checkpoint: Path, device: str
) -> tuple[Any, Any, dict[str, Any]]:
    obs_dim, critic_obs_dim = get_obs_dims(env.obs_groups_spec)
    action_dim = int(env.action_space.shape[0])
    baseline = SACActor(
        obs_dim=int(obs_dim),
        action_dim=action_dim,
        hidden_dim=int(cfg.algo.actor_hidden_dim),
        use_layer_norm=bool(cfg.algo.use_layer_norm),
        device=device,
    )
    nominal_path = Path(str(cfg.algo.actor.nominal_initialization_checkpoint)).resolve()
    nominal_state = torch.load(nominal_path, map_location=device, weights_only=True)
    baseline.load_state_dict(nominal_state["actor"], strict=True)
    baseline.eval()

    rl_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
    runtime = resolve_custom_offpolicy_runtime(rl_cfg)
    if runtime is None or runtime.algo_type != PRIVILEGED_FULL_ACTION_SAC_ALGO_TYPE:
        raise ValueError("Could not resolve privileged full-action SAC runtime")
    actor_kwargs = runtime.build_model_kwargs(
        obs_dim=int(obs_dim), critic_obs_dim=int(critic_obs_dim)
    )
    teacher = build_actor(
        PRIVILEGED_FULL_ACTION_SAC_ALGO_TYPE,
        int(obs_dim),
        action_dim,
        int(cfg.algo.actor_hidden_dim),
        bool(cfg.algo.use_layer_norm),
        device,
        **actor_kwargs,
    )
    if not isinstance(teacher, PrivilegedFullActionSACActor):
        raise TypeError("Runtime did not build PrivilegedFullActionSACActor")
    identity = load_privileged_full_action_actor_checkpoint(teacher, checkpoint, device=device)
    return baseline, teacher, identity


def main() -> int:
    args = _parse_args()
    cfg = _compose_cfg()
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="sac"
    ).build_task_env_cfg_override()
    ensure_registries()
    reports: list[dict[str, Any]] = []
    identity: dict[str, Any] | None = None
    baseline = teacher = None
    for seed in args.seeds:
        apply_training_seed(int(seed), torch_runtime=True, cuda=False)
        env = create_env(
            cfg,
            num_envs=int(args.num_envs),
            env_cfg_override=env_override,
            sim_backend="mujoco",
        )
        try:
            env.init_state()
            if baseline is None or teacher is None:
                baseline, teacher, identity = _build_actors(
                    cfg, env, args.teacher_checkpoint.expanduser().resolve(), str(args.device)
                )
            report = evaluate_full_action_paired_rollouts(
                env, baseline, teacher, steps=int(args.steps), device=str(args.device)
            )
            report["seed"] = int(seed)
            reports.append(report)
        finally:
            env.close()

    aggregate = aggregate_full_action_paired_reports(reports)
    contract = {
        "task_config": FORMAL_TASK_CONFIG,
        "num_envs_per_seed": int(args.num_envs),
        "steps": int(args.steps),
        "seeds": [int(seed) for seed in args.seeds],
        "command": list(cfg.env.commands.vel_limit[0]),
        "aggregation": FORMAL_AGGREGATION,
        "actuator_strength": list(FORMAL_STRENGTH),
        "quality_thresholds": formal_quality_thresholds(),
    }
    protocol = validate_full_action_formal_evaluation_contract(contract)
    quality = assess_full_action_teacher_quality(aggregate, contract)
    payload = {
        "schema": "unilab_context_full_action_evaluation_run_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_checkpoint": identity,
        "contract": contract,
        "protocol_validation": protocol,
        "quality_assessment": quality,
        "per_seed": reports,
        "aggregate": aggregate,
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.teacher_checkpoint.expanduser()
        .resolve()
        .with_name(f"{args.teacher_checkpoint.stem}_full_action_paired.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    baseline_metrics = aggregate["baseline"]["overall"]
    teacher_metrics = aggregate["teacher"]["overall"]
    print(f"Paired evaluation written to: {output}")
    print(
        "Max lateral: "
        f"baseline={baseline_metrics['max_lateral_abs_m']:.6f}, "
        f"teacher={teacher_metrics['max_lateral_abs_m']:.6f}"
    )
    print(
        "Max yaw: "
        f"baseline={baseline_metrics['max_yaw_abs_rad']:.6f}, "
        f"teacher={teacher_metrics['max_yaw_abs_rad']:.6f}"
    )
    print(f"Quality status: {quality['quality_status']}")
    return 0 if quality["quality_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
