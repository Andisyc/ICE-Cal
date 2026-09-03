"""Transactional target-only Stage-C workflow for the G1 slope domain."""

from __future__ import annotations

import json
import random
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.adaptation_checkpoint import (
    assert_fada_target_collection_checkpoint,
    load_fada_deployable_policy_checkpoint,
)
from unilab.algos.torch.distill.fada.observation import assert_fada_active_route_contract
from unilab.algos.torch.distill.fada.target_collector import (
    FADASlopeEpisodePolicy,
    FADATargetCollectionSpec,
    collect_fada_slope_windows,
)
from unilab.algos.torch.distill.fada.target_data import save_fada_target_artifact
from unilab.algos.torch.distill.fada.target_domain import (
    FADATargetDomainSpec,
    assert_nominal_slope_environment,
    resolve_fada_target_domain,
)
from unilab.algos.torch.distill.fada.target_rollout import target_tracking_camera_kwargs
from unilab.algos.torch.distill.workflow import config_fingerprint, file_sha256
from unilab.base.backend.mujoco.playback import render_mujoco_states_video
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
class FADASlopeTargetPreflight:
    checkpoint_path: Path
    output_dir: Path
    checkpoint_sha256: str
    domain: FADATargetDomainSpec
    control_steps: int
    max_env_steps: int


def _root_relative(value: Any, *, root_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root_dir / path).resolve()


def _integer(cfg: DictConfig, path: str, *, positive: bool) -> int:
    value = OmegaConf.select(cfg, path)
    invalid = isinstance(value, bool) or not isinstance(value, Integral)
    invalid = invalid or (value <= 0 if positive else value < 0)
    if invalid:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{path} must be a {qualifier} integer, got {value!r}")
    return int(value)


def _command3(cfg: DictConfig, path: str) -> tuple[float, float, float]:
    raw = OmegaConf.to_container(OmegaConf.select(cfg, path), resolve=True)
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{path} must contain exactly three values")
    command = tuple(float(value) for value in raw)
    if not all(np.isfinite(command)):
        raise ValueError(f"{path} must contain only finite values")
    return cast(tuple[float, float, float], command)


def preflight_fada_slope_collection(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> FADASlopeTargetPreflight:
    root = Path(root_dir).resolve()
    domain = resolve_fada_target_domain(cfg)
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    assert_nominal_slope_environment(
        cfg,
        domain,
        task_choice=get_hydra_runtime_choice(cfg, "task"),
    )
    if _integer(cfg, "collection.num_envs", positive=True) != 1:
        raise ValueError("FADA slope collection requires collection.num_envs=1")
    if not bool(OmegaConf.select(cfg, "collection.record_video")):
        raise ValueError("FADA slope collection requires collection.record_video=true")
    control_steps = _integer(cfg, "collection.control_steps", positive=True)
    max_env_steps = _integer(cfg, "collection.max_env_steps", positive=True)
    _integer(cfg, "collection.ramp_steps", positive=False)
    _integer(cfg, "collection.settle_steps", positive=False)
    _integer(cfg, "collection.seed", positive=False)
    _command3(cfg, "collection.command_start")
    checkpoint = _root_relative(
        OmegaConf.select(cfg, "collection.policy_checkpoint_path"), root_dir=root
    )
    output_dir = _root_relative(OmegaConf.select(cfg, "collection.output_dir"), root_dir=root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FADA slope checkpoint not found: {checkpoint}")
    if output_dir.exists():
        raise FileExistsError(f"FADA slope bundle already exists: {output_dir}")
    observed = file_sha256(checkpoint)
    expected = OmegaConf.select(cfg, "collection.expected_checkpoint_sha256")
    if expected is not None:
        expected = str(expected)
        if _SHA256.fullmatch(expected) is None or expected != observed:
            raise ValueError(
                f"FADA slope checkpoint SHA-256 mismatch: expected={expected} observed={observed}"
            )
    return FADASlopeTargetPreflight(
        checkpoint,
        output_dir,
        observed,
        domain,
        control_steps,
        max_env_steps,
    )


def _fingerprint(cfg: DictConfig) -> str:
    selected = OmegaConf.masked_copy(
        cfg,
        ["algo", "training", "env", "collection", "target_domain"],
    )
    payload = OmegaConf.to_container(selected, resolve=True)
    if not isinstance(payload, Mapping):
        raise TypeError("resolved FADA slope config must be a mapping")
    return config_fingerprint(cast(Mapping[str, Any], payload))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _collection_spec(cfg: DictConfig, *, max_env_steps: int) -> FADATargetCollectionSpec:
    return FADATargetCollectionSpec(
        observation_key=str(cfg.collection.observation_key),
        student_projection=str(cfg.collection.student_projection),
        student_drop_index=cfg.collection.student_drop_index,
        command_info_keys=tuple(cfg.collection.command_info_keys),
        max_env_steps=max_env_steps,
        command_start=_command3(cfg, "collection.command_start"),
        ramp_steps=_integer(cfg, "collection.ramp_steps", positive=False),
        settle_steps=_integer(cfg, "collection.settle_steps", positive=False),
    )


def run_fada_slope_collection(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
    load_policy_fn: Callable[..., Any] = load_fada_deployable_policy_checkpoint,
    ensure_registries_fn: Callable[[], None] = ensure_registries,
    create_env_fn: Callable[..., Any] = create_env,
    collect_fn: Callable[..., Any] = collect_fada_slope_windows,
    save_fn: Callable[..., Path] = save_fada_target_artifact,
    render_fn: Callable[..., str] = render_mujoco_states_video,
) -> dict[str, Any]:
    root = Path(root_dir).resolve()
    preflight = preflight_fada_slope_collection(cfg, root_dir=root)
    loaded = assert_fada_target_collection_checkpoint(
        load_policy_fn(preflight.checkpoint_path, device=str(cfg.collection.device))
    )
    policy = loaded.policy
    assert_fada_active_route_contract(
        observation_contract=policy.config.observation_contract,
        projection=str(cfg.collection.student_projection),
    )
    spec = _collection_spec(cfg, max_env_steps=preflight.max_env_steps)
    ensure_registries_fn()
    _seed_all(_integer(cfg, "collection.seed", positive=False))
    env = create_env_fn(
        cfg,
        num_envs=1,
        env_cfg_override=BackendAdapter(
            cfg,
            root_dir=root,
            algo_name="sac",
        ).build_task_env_cfg_override(),
        sim_backend=preflight.domain.backend,
    )
    preflight.output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not env.play_capabilities.supports_physics_state_playback:
            raise NotImplementedError(
                f"{env.__class__.__name__} does not support physics-state video playback"
            )
        episode_policy = FADASlopeEpisodePolicy(
            cast(Any, preflight.domain.slope),
            preflight.domain.command_sequence,
        )
        result = collect_fn(
            env,
            policy,
            policy.config,
            preflight.control_steps,
            episode_policy,
            spec,
        )
        with tempfile.TemporaryDirectory(
            prefix=f".{preflight.output_dir.name}-",
            dir=preflight.output_dir.parent,
        ) as tmp:
            stage = Path(tmp) / preflight.output_dir.name
            stage.mkdir()
            metadata = {
                "policy_checkpoint_sha256": preflight.checkpoint_sha256,
                "config_fingerprint": _fingerprint(cfg),
                "task": preflight.domain.task_name,
                "target_domain_id": preflight.domain.target_domain_id,
                "target_domain_kind": preflight.domain.kind,
                "command_sequence": [
                    list(command) for command in preflight.domain.command_sequence
                ],
                "slope_geometry": asdict(cast(Any, preflight.domain.slope)),
                "num_envs": 1,
                "num_windows": int(result.batch.observation_history.shape[0]),
                "observation_contract": policy.config.observation_contract,
                "accepted_steps": int(result.accepted_steps),
                "rejected_pre_entry_steps": int(result.rejected_pre_entry_steps),
                "rejected_command_windows": int(result.rejected_command_windows),
                "episode_count": int(result.episode_count),
                "termination_counts": dict(result.termination_counts or {}),
                "randomization_disabled": True,
            }
            save_fn(stage / "target.pt", result.batch, config=policy.config, metadata=metadata)
            render_fn(
                env=env,
                state_list=result.representative_physics_states,
                output_video=stage / "collection.mp4",
                camera_kwargs=target_tracking_camera_kwargs(),
            )
            summary = {
                "num_windows": int(result.batch.observation_history.shape[0]),
                "env_steps": int(result.env_steps),
                "accepted_steps": int(result.accepted_steps),
                "episode_count": int(result.episode_count),
                "rejected_pre_entry_steps": int(result.rejected_pre_entry_steps),
                "rejected_command_windows": int(result.rejected_command_windows),
                "termination_counts": dict(result.termination_counts or {}),
            }
            (stage / "collection_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            files = ["target.pt", "collection.mp4", "collection_summary.json"]
            (stage / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "fada-target-bundle/v2",
                        "target_domain_id": preflight.domain.target_domain_id,
                        "checkpoint_sha256": preflight.checkpoint_sha256,
                        "files": files,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            stage.replace(preflight.output_dir)
    finally:
        env.close()
    return {
        "status": "completed",
        "bundle_dir": str(preflight.output_dir),
        "artifact_path": str(preflight.output_dir / "target.pt"),
        "video_path": str(preflight.output_dir / "collection.mp4"),
        "summary_path": str(preflight.output_dir / "collection_summary.json"),
        "num_windows": int(result.batch.observation_history.shape[0]),
    }
