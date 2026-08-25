"""Collect the isolated, Oracle-free FADA Stage-C target artifact."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.distill import (
    FADATargetCollectionSpec,
    assert_fada_active_route_contract,
    assert_fada_adaptation_source_checkpoint,
    collect_fada_target_windows,
    config_fingerprint,
    file_sha256,
    load_fada_policy_checkpoint,
    save_fada_target_artifact,
)
from unilab.training import (
    BackendAdapter,
    assert_offpolicy_task_choice_matches_algo,
    create_env,
    ensure_registries,
    get_hydra_runtime_choice,
)

_TARGET_TASK_CHOICE = "sac/g1_walk_flat/mujoco_left_knee_090"
_TARGET_TASK_NAME = "G1WalkFlat"
_TARGET_BACKEND = "mujoco"
_TARGET_FAULT_PROFILE = "left_knee_strength_0.9"
_TARGET_COMMAND_LIMIT = [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]
_TARGET_ACTUATOR_INDEX = 3
_TARGET_ACTUATOR_STRENGTH = 0.9
_ACTUATOR_COUNT = 29
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FADATargetPreflight:
    checkpoint_path: Path
    output_path: Path
    checkpoint_sha256: str


def _root_relative(path_value: Any, *, root_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    return path.resolve() if path.is_absolute() else (root_dir / path).resolve()


def _require_positive_int(cfg: DictConfig, path: str) -> int:
    value = OmegaConf.select(cfg, path)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{path} must be a positive integer, got {value!r}")
    return int(value)


def _assert_exact_target_identity(cfg: DictConfig) -> None:
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    task_choice = get_hydra_runtime_choice(cfg, "task")
    if task_choice != _TARGET_TASK_CHOICE:
        raise ValueError(
            f"FADA target collection requires task={_TARGET_TASK_CHOICE}, got {task_choice!r}"
        )
    if str(OmegaConf.select(cfg, "training.task_name")) != _TARGET_TASK_NAME:
        raise ValueError(f"FADA target task_name must be {_TARGET_TASK_NAME}")
    if str(OmegaConf.select(cfg, "training.sim_backend")) != _TARGET_BACKEND:
        raise ValueError("FADA target collection requires the MuJoCo backend")
    if str(OmegaConf.select(cfg, "collection.fault_profile")) != _TARGET_FAULT_PROFILE:
        raise ValueError(f"FADA target fault_profile must be {_TARGET_FAULT_PROFILE}")

    command_limit = OmegaConf.to_container(
        OmegaConf.select(cfg, "env.commands.vel_limit"), resolve=True
    )
    if command_limit != _TARGET_COMMAND_LIMIT:
        raise ValueError(
            f"FADA target command limit must be {_TARGET_COMMAND_LIMIT}, got {command_limit}"
        )
    if (
        float(OmegaConf.select(cfg, "env.commands.rel_standing_envs")) != 0.0
        or float(OmegaConf.select(cfg, "env.commands.rel_transition_envs")) != 0.0
    ):
        raise ValueError("FADA target command profile must contain only walking environments")
    if bool(OmegaConf.select(cfg, "env.curriculum.enabled")):
        raise ValueError("FADA target curriculum must be disabled")
    if float(OmegaConf.select(cfg, "env.reset_base_qvel_limit")) != 0.0:
        raise ValueError("FADA target reset_base_qvel_limit must be zero")

    if not bool(OmegaConf.select(cfg, "env.domain_rand.actuator_strength.enabled")):
        raise ValueError("FADA target actuator fault must be enabled")
    if bool(OmegaConf.select(cfg, "env.domain_rand.randomize_kp")) or bool(
        OmegaConf.select(cfg, "env.domain_rand.randomize_kd")
    ):
        raise ValueError("FADA target actuator gains must not be randomized")
    multiplier_values = OmegaConf.to_container(
        OmegaConf.select(cfg, "env.domain_rand.actuator_strength.multipliers"),
        resolve=True,
    )
    if not isinstance(multiplier_values, list):
        raise ValueError("FADA target actuator multipliers must be a list")
    multipliers = list(multiplier_values)
    expected = [1.0] * _ACTUATOR_COUNT
    expected[_TARGET_ACTUATOR_INDEX] = _TARGET_ACTUATOR_STRENGTH
    if multipliers != expected:
        raise ValueError("FADA target actuator multipliers must encode only left-knee strength 0.9")


def preflight_fada_target_collection(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> FADATargetPreflight:
    """Fail closed on task, fault, checkpoint, and output identity before env creation."""

    root = Path(root_dir).resolve()
    _assert_exact_target_identity(cfg)
    _require_positive_int(cfg, "collection.num_envs")
    _require_positive_int(cfg, "collection.num_windows")
    _require_positive_int(cfg, "collection.max_env_steps")

    checkpoint_path = _root_relative(
        OmegaConf.select(cfg, "collection.policy_checkpoint_path"), root_dir=root
    )
    output_path = _root_relative(OmegaConf.select(cfg, "collection.output_path"), root_dir=root)
    if checkpoint_path == output_path:
        raise ValueError("FADA target checkpoint and output paths must differ")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"FADA target checkpoint not found: {checkpoint_path}")
    if output_path.exists() and not bool(OmegaConf.select(cfg, "collection.overwrite")):
        raise FileExistsError(
            f"FADA target output already exists: {output_path}; set collection.overwrite=true "
            "only after confirming replacement authority"
        )
    if output_path.suffix != ".pt":
        raise ValueError(f"FADA target output must use a .pt suffix, got {output_path}")

    expected_sha256 = str(OmegaConf.select(cfg, "collection.expected_checkpoint_sha256"))
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("collection.expected_checkpoint_sha256 must be lowercase SHA-256 hex")
    observed_sha256 = file_sha256(checkpoint_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "FADA target checkpoint SHA-256 mismatch: "
            f"expected={expected_sha256} observed={observed_sha256}"
        )
    return FADATargetPreflight(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        checkpoint_sha256=observed_sha256,
    )


def _fingerprint_payload(cfg: DictConfig) -> Mapping[str, Any]:
    bounded = OmegaConf.masked_copy(cfg, ["algo", "training", "env", "collection"])
    payload = OmegaConf.to_container(bounded, resolve=True)
    if not isinstance(payload, Mapping):
        raise TypeError("resolved FADA target config must be a mapping")
    return cast(Mapping[str, Any], payload)


def run_fada_target_collection(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
    load_policy_fn: Callable[..., Any] = load_fada_policy_checkpoint,
    ensure_registries_fn: Callable[[], None] = ensure_registries,
    create_env_fn: Callable[..., Any] = create_env,
    collect_fn: Callable[..., Any] = collect_fada_target_windows,
    save_fn: Callable[..., Path] = save_fada_target_artifact,
) -> dict[str, Any]:
    """Run one bounded Stage-C collection and return a machine-readable summary."""

    root = Path(root_dir).resolve()
    preflight = preflight_fada_target_collection(cfg, root_dir=root)
    device = str(OmegaConf.select(cfg, "collection.device"))
    loaded = assert_fada_adaptation_source_checkpoint(
        load_policy_fn(preflight.checkpoint_path, device=device)
    )
    policy = loaded.policy
    assert_fada_active_route_contract(
        observation_contract=policy.config.observation_contract,
        projection=str(OmegaConf.select(cfg, "collection.student_projection")),
    )

    ensure_registries_fn()
    env_override = BackendAdapter(cfg, root_dir=root, algo_name="sac").build_task_env_cfg_override()
    env = create_env_fn(
        cfg,
        num_envs=_require_positive_int(cfg, "collection.num_envs"),
        env_cfg_override=env_override,
        sim_backend=_TARGET_BACKEND,
    )
    result = None
    try:
        spec = FADATargetCollectionSpec(
            observation_key=str(OmegaConf.select(cfg, "collection.observation_key")),
            student_projection=str(OmegaConf.select(cfg, "collection.student_projection")),
            student_drop_index=OmegaConf.select(cfg, "collection.student_drop_index"),
            command_info_keys=tuple(OmegaConf.select(cfg, "collection.command_info_keys")),
            max_env_steps=_require_positive_int(cfg, "collection.max_env_steps"),
        )
        result = collect_fn(
            env,
            policy,
            policy.config,
            _require_positive_int(cfg, "collection.num_windows"),
            spec,
        )
        artifact_path = save_fn(
            preflight.output_path,
            result.batch,
            config=policy.config,
            metadata={
                "policy_checkpoint_sha256": preflight.checkpoint_sha256,
                "config_fingerprint": config_fingerprint(_fingerprint_payload(cfg)),
                "task": _TARGET_TASK_NAME,
                "fault_profile": _TARGET_FAULT_PROFILE,
                "num_envs": int(env.num_envs),
                "num_windows": int(result.batch.observation_history.shape[0]),
            },
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    assert result is not None
    return {
        "status": "completed",
        "artifact_path": str(artifact_path),
        "checkpoint_sha256": preflight.checkpoint_sha256,
        "config_fingerprint": config_fingerprint(_fingerprint_payload(cfg)),
        "num_envs": int(OmegaConf.select(cfg, "collection.num_envs")),
        "num_windows": int(result.batch.observation_history.shape[0]),
        "env_steps": int(result.env_steps),
        "rejected_done_transitions": int(result.rejected_done_transitions),
        "rejected_command_windows": int(result.rejected_command_windows),
    }


@hydra.main(version_base="1.3", config_path="../conf/offpolicy", config_name="fada_target")
def main(cfg: DictConfig) -> None:
    print(json.dumps(run_fada_target_collection(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
