"""Atomic paired Stage-C collection workflow."""

from __future__ import annotations

import json
import random
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.adaptation_checkpoint import (
    assert_fada_adaptation_source_checkpoint,
)
from unilab.algos.torch.distill.fada.checkpoint import load_fada_policy_checkpoint
from unilab.algos.torch.distill.fada.fault import resolve_fada_fault
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig
from unilab.algos.torch.distill.fada.observation import assert_fada_active_route_contract
from unilab.algos.torch.distill.fada.path_capture import (
    FADAStageCPathCapture,
    FADAStageCPathTrace,
)
from unilab.algos.torch.distill.fada.path_deviation import (
    build_straight_line_deviation_report,
)
from unilab.algos.torch.distill.fada.target_collector import (
    FADATargetCollectionSpec,
    collect_fada_target_windows,
)
from unilab.algos.torch.distill.fada.target_data import (
    FADATargetBatch,
    save_fada_target_artifact,
)
from unilab.algos.torch.distill.workflow import config_fingerprint, file_sha256
from unilab.training import (
    BackendAdapter,
    assert_offpolicy_task_choice_matches_algo,
    create_env,
    ensure_registries,
    get_hydra_runtime_choice,
)

ROOT_DIR = Path(__file__).resolve().parents[6]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FADATargetPreflight:
    checkpoint_path: Path
    output_dir: Path
    checkpoint_sha256: str


def _root_relative(value: Any, *, root_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root_dir / path).resolve()


def _positive_int(cfg: DictConfig, path: str) -> int:
    value = OmegaConf.select(cfg, path)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{path} must be a positive integer, got {value!r}")
    return int(value)


def _nonnegative_int(cfg: DictConfig, path: str) -> int:
    value = OmegaConf.select(cfg, path)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer, got {value!r}")
    return int(value)


def _command3(cfg: DictConfig, path: str) -> tuple[float, float, float]:
    values = OmegaConf.to_container(OmegaConf.select(cfg, path), resolve=True)
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{path} must contain exactly three values")
    command = tuple(float(value) for value in values)
    if not all(np.isfinite(command)):
        raise ValueError(f"{path} must contain only finite values")
    return command[0], command[1], command[2]


def _assert_identity(cfg: DictConfig) -> None:
    fault = resolve_fada_fault(cfg)
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    if get_hydra_runtime_choice(cfg, "task") != fault.task:
        raise ValueError(f"FADA target collection requires task={fault.task}")
    if str(OmegaConf.select(cfg, "training.task_name")) != fault.task_name:
        raise ValueError(f"FADA target task_name must be {fault.task_name}")
    if str(OmegaConf.select(cfg, "training.sim_backend")) != fault.backend:
        raise ValueError("FADA target collection requires the MuJoCo backend")
    if _positive_int(cfg, "collection.num_envs") != 1:
        raise ValueError("FADA paired single-trajectory collection requires num_envs=1")
    if not bool(OmegaConf.select(cfg, "collection.single_trajectory")):
        raise ValueError("FADA target collection requires single_trajectory=true")
    if not bool(OmegaConf.select(cfg, "collection.record_video")):
        raise ValueError("FADA target collection requires record_video=true")
    if (
        OmegaConf.to_container(OmegaConf.select(cfg, "env.commands.vel_limit"), resolve=True)
        != fault.command_limit
    ):
        raise ValueError("FADA target command identity does not match the selected fault")
    if fault.actuator_count <= 0:
        raise ValueError("FADA target actuator_count must be positive")
    if not 0 <= fault.actuator_index < fault.actuator_count:
        raise ValueError("FADA target actuator_index is outside actuator_count")
    if not np.isfinite(fault.actuator_strength) or fault.actuator_strength < 0.0:
        raise ValueError("FADA target actuator_strength must be finite and non-negative")
    if bool(OmegaConf.select(cfg, "env.curriculum.enabled")):
        raise ValueError("FADA target curriculum must be disabled")


def preflight_fada_target_collection(
    cfg: DictConfig, *, root_dir: str | Path = ROOT_DIR
) -> FADATargetPreflight:
    root = Path(root_dir).resolve()
    _assert_identity(cfg)
    _positive_int(cfg, "collection.num_windows")
    _positive_int(cfg, "collection.max_env_steps")
    _nonnegative_int(cfg, "collection.ramp_steps")
    _nonnegative_int(cfg, "collection.settle_steps")
    _nonnegative_int(cfg, "collection.seed")
    _command3(cfg, "collection.command_start")
    _command3(cfg, "collection.command_target")
    checkpoint = _root_relative(
        OmegaConf.select(cfg, "collection.policy_checkpoint_path"), root_dir=root
    )
    output_dir = _root_relative(OmegaConf.select(cfg, "collection.output_dir"), root_dir=root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FADA target checkpoint not found: {checkpoint}")
    if output_dir.exists():
        raise FileExistsError(f"FADA target bundle already exists: {output_dir}")
    expected = OmegaConf.select(cfg, "collection.expected_checkpoint_sha256")
    observed = file_sha256(checkpoint)
    if expected is not None:
        expected = str(expected)
        if _SHA256.fullmatch(expected) is None or observed != expected:
            raise ValueError(
                f"FADA target checkpoint SHA-256 mismatch: expected={expected} observed={observed}"
            )
    return FADATargetPreflight(checkpoint, output_dir, observed)


def _fingerprint(cfg: DictConfig) -> str:
    payload = OmegaConf.to_container(
        OmegaConf.masked_copy(cfg, ["algo", "training", "env", "collection", "fault"]), resolve=True
    )
    if not isinstance(payload, Mapping):
        raise TypeError("resolved FADA target config must be a mapping")
    return config_fingerprint(cast(Mapping[str, Any], payload))


def _spec(
    cfg: DictConfig,
    frame_sink: Callable[[], None],
    initial_frame_sink: Callable[[], None],
) -> FADATargetCollectionSpec:
    return FADATargetCollectionSpec(
        observation_key=str(cfg.collection.observation_key),
        student_projection=str(cfg.collection.student_projection),
        student_drop_index=cfg.collection.student_drop_index,
        command_info_keys=tuple(cfg.collection.command_info_keys),
        max_env_steps=_positive_int(cfg, "collection.max_env_steps"),
        command_start=_command3(cfg, "collection.command_start"),
        command_target=_command3(cfg, "collection.command_target"),
        ramp_steps=_nonnegative_int(cfg, "collection.ramp_steps"),
        settle_steps=_nonnegative_int(cfg, "collection.settle_steps"),
        single_trajectory=True,
        capture_initial_frame=initial_frame_sink,
        capture_frame=frame_sink,
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _env_override(cfg: DictConfig, *, nominal: bool, root: Path) -> dict[str, Any]:
    override = BackendAdapter(cfg, root_dir=root, algo_name="sac").build_task_env_cfg_override()
    override.setdefault("noise_config", {})["level"] = 0.0
    domain_rand = override.setdefault("domain_rand", {})
    domain_rand.update(
        {
            "randomize_reset_pose": False,
            "randomize_kp": False,
            "randomize_kd": False,
            "randomize_ground_friction": False,
            "randomize_base_mass": False,
            "randomize_body_mass": False,
            "random_com": False,
            "randomize_gravity": False,
            "randomize_dof_armature": False,
            "randomize_dof_position_bias": False,
            "randomize_control_delay": False,
            "torque_rfi_fraction": 0.0,
            "push_robots": False,
        }
    )
    fault = resolve_fada_fault(cfg)
    multipliers = [1.0] * fault.actuator_count
    if not nominal:
        multipliers[fault.actuator_index] = fault.actuator_strength
    domain_rand.setdefault("actuator_strength", {}).update(
        {"enabled": True, "multipliers": multipliers}
    )
    return override


@dataclass(frozen=True)
class _CollectedBranch:
    result: Any
    path: FADAStageCPathTrace


def _collect_branch(
    cfg: DictConfig,
    policy: Any,
    *,
    root: Path,
    nominal: bool,
    video: Path,
    create_env_fn: Callable[..., Any],
    collect_fn: Callable[..., Any],
) -> _CollectedBranch:
    seed = _nonnegative_int(cfg, "collection.seed")
    _seed_all(seed)
    env = create_env_fn(
        cfg,
        num_envs=1,
        env_cfg_override=_env_override(cfg, nominal=nominal, root=root),
        sim_backend=resolve_fada_fault(cfg).backend,
    )
    sink: FADAStageCPathCapture | None = None
    try:
        if not env.play_capabilities.supports_physics_state_playback:
            raise NotImplementedError(
                f"{env.__class__.__name__} does not support physics-state video playback"
            )
        sink = FADAStageCPathCapture(env, video)
        result = collect_fn(
            env,
            policy,
            policy.config,
            _positive_int(cfg, "collection.num_windows"),
            _spec(cfg, sink.capture_step, sink.capture_initial),
        )
        sink.discard_terminal_frames(int(result.rejected_done_transitions))
        sink.write_video()
        path = sink.path_trace()
        sink = None
        return _CollectedBranch(result, path)
    finally:
        env.close()


def _slice_target_batch(
    batch: FADATargetBatch,
    rows: int,
    config: FADAArchitectureConfig,
) -> FADATargetBatch:
    return FADATargetBatch(
        **{
            field: getattr(batch, field)[:rows]
            for field in FADATargetBatch.__dataclass_fields__
        }
    ).validate(config)


def _align_paired_batches(
    nominal: FADATargetBatch,
    faulty: FADATargetBatch,
    config: FADAArchitectureConfig,
) -> tuple[FADATargetBatch, FADATargetBatch]:
    rows = min(
        int(nominal.observation_history.shape[0]),
        int(faulty.observation_history.shape[0]),
    )
    if rows <= 0:
        raise ValueError("FADA paired target collection has no common trajectory window")
    aligned_nominal = _slice_target_batch(nominal, rows, config)
    aligned_faulty = _slice_target_batch(faulty, rows, config)
    for name in ("episode_id", "start_timestep", "command"):
        if not torch.equal(getattr(aligned_nominal, name), getattr(aligned_faulty, name)):
            raise ValueError(f"FADA paired target row identity mismatch: {name}")
    return aligned_nominal, aligned_faulty


def _save_delta(
    path: Path,
    nominal: FADATargetBatch,
    faulty: FADATargetBatch,
    metadata: Mapping[str, Any],
) -> None:
    for name in ("episode_id", "start_timestep", "command"):
        if not torch.equal(getattr(nominal, name), getattr(faulty, name)):
            raise ValueError(f"FADA paired target row identity mismatch: {name}")
    torch.save(
        {
            "schema_version": "fada-target-paired-delta/v1",
            "metadata": dict(metadata),
            "delta": {
                name: getattr(faulty, name) - getattr(nominal, name)
                for name in (
                    "observation_history",
                    "action_history",
                    "realized_future",
                    "executed_action_chunk",
                )
            },
        },
        path,
    )


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
    root = Path(root_dir).resolve()
    preflight = preflight_fada_target_collection(cfg, root_dir=root)
    loaded = assert_fada_adaptation_source_checkpoint(
        load_policy_fn(preflight.checkpoint_path, device=str(cfg.collection.device))
    )
    policy = loaded.policy
    assert_fada_active_route_contract(
        observation_contract=policy.config.observation_contract,
        projection=str(cfg.collection.student_projection),
    )
    ensure_registries_fn()
    fault, fingerprint = resolve_fada_fault(cfg), _fingerprint(cfg)
    preflight.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{preflight.output_dir.name}-", dir=preflight.output_dir.parent
    ) as tmp:
        stage = Path(tmp) / preflight.output_dir.name
        stage.mkdir()
        nominal = _collect_branch(
            cfg,
            policy,
            root=root,
            nominal=True,
            video=stage / "nominal.mp4",
            create_env_fn=create_env_fn,
            collect_fn=collect_fn,
        )
        faulty = _collect_branch(
            cfg,
            policy,
            root=root,
            nominal=False,
            video=stage / "faulty.mp4",
            create_env_fn=create_env_fn,
            collect_fn=collect_fn,
        )
        nominal_batch, faulty_batch = _align_paired_batches(
            nominal.result.batch,
            faulty.result.batch,
            policy.config,
        )
        if not np.allclose(
            nominal.path.origin_xy_m, faulty.path.origin_xy_m, rtol=0.0, atol=1e-6
        ):
            raise ValueError("FADA paired target branches do not share the same start position")
        heading_delta = np.arctan2(
            np.sin(nominal.path.heading_rad - faulty.path.heading_rad),
            np.cos(nominal.path.heading_rad - faulty.path.heading_rad),
        )
        if abs(float(heading_delta)) > 1e-6:
            raise ValueError("FADA paired target branches do not share the same start heading")
        measurement_start_step = _nonnegative_int(cfg, "collection.ramp_steps") + _nonnegative_int(
            cfg, "collection.settle_steps"
        )
        path_deviation = build_straight_line_deviation_report(
            nominal_xy_m=nominal.path.position_xy_m[measurement_start_step:],
            faulty_xy_m=faulty.path.position_xy_m[measurement_start_step:],
            nominal_yaw_rad=nominal.path.yaw_rad[measurement_start_step:],
            faulty_yaw_rad=faulty.path.yaw_rad[measurement_start_step:],
            origin_xy_m=nominal.path.origin_xy_m,
            heading_rad=nominal.path.heading_rad,
            measurement_start_step=measurement_start_step,
        )
        (stage / "path_deviation.json").write_text(
            json.dumps(path_deviation, indent=2), encoding="utf-8"
        )
        common = {
            "policy_checkpoint_sha256": preflight.checkpoint_sha256,
            "config_fingerprint": fingerprint,
            "task": fault.task_name,
            "num_envs": 1,
            "num_windows": int(faulty_batch.observation_history.shape[0]),
        }
        save_fn(
            stage / "nominal.pt",
            nominal_batch,
            config=policy.config,
            metadata={**common, "fault_profile": "nominal"},
        )
        save_fn(
            stage / "faulty.pt",
            faulty_batch,
            config=policy.config,
            metadata={**common, "fault_profile": fault.fault_profile},
        )
        _save_delta(
            stage / "delta.pt",
            nominal_batch,
            faulty_batch,
            {
                "fault_profile": fault.fault_profile,
                "checkpoint_sha256": preflight.checkpoint_sha256,
            },
        )
        files = [
            "nominal.pt",
            "faulty.pt",
            "delta.pt",
            "nominal.mp4",
            "faulty.mp4",
            "path_deviation.json",
        ]
        (stage / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "fada-target-bundle/v1",
                    "files": files,
                    "fault_profile": fault.fault_profile,
                    "checkpoint_sha256": preflight.checkpoint_sha256,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        stage.replace(preflight.output_dir)
    return {
        "status": "completed",
        "bundle_dir": str(preflight.output_dir),
        "artifact_path": str(preflight.output_dir / "faulty.pt"),
        "nominal_artifact_path": str(preflight.output_dir / "nominal.pt"),
        "paired_delta_path": str(preflight.output_dir / "delta.pt"),
        "fault_video_path": str(preflight.output_dir / "faulty.mp4"),
        "nominal_video_path": str(preflight.output_dir / "nominal.mp4"),
        "path_deviation_path": str(preflight.output_dir / "path_deviation.json"),
        "path_deviation": {
            branch: {
                key: value
                for key, value in cast(Mapping[str, Any], path_deviation[branch]).items()
                if key not in {"position_xy_m", "lateral_m", "yaw_rad", "yaw_drift_rad"}
            }
            for branch in ("nominal", "faulty", "excess")
        },
        "num_windows": int(faulty.result.batch.observation_history.shape[0]),
    }
