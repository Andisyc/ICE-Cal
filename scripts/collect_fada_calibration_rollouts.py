#!/usr/bin/env python3
"""Collect the approved gain-only FADA calibration smoke rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.distill import load_fada_policy_checkpoint  # noqa: E402
from unilab.algos.torch.fada_context.calibration import (  # noqa: E402
    CALIBRATION_AXIS_CATALOG_VERSION,
)
from unilab.algos.torch.fada_context.calibration_collection import (  # noqa: E402
    GainCalibrationCollectionProtocol,
    GainCalibrationRawIdentity,
    canonicalize_resolved_task_backend_payload,
    collect_gain_calibration_rollouts,
    load_gain_calibration_protocol,
    save_gain_calibration_raw_rollouts,
    sha256_file,
)
from unilab.training import (  # noqa: E402
    BackendAdapter,
    apply_training_seed,
    create_env,
    ensure_registries,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT_DIR
        / "conf"
        / "fada_context"
        / "calibration_collection"
        / "gain_smoke_v1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _compose_task(protocol: GainCalibrationCollectionProtocol) -> DictConfig:
    protocol.validate_approved()
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / "distill"),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=[f"task={protocol.task_config}"])
    if (
        str(cfg.training.task_name) != protocol.task_name
        or str(cfg.training.sim_backend) != protocol.sim_backend
    ):
        raise ValueError("resolved distill task/backend does not match the smoke protocol")
    return cfg


def _base_env_override(
    cfg: DictConfig,
    protocol: GainCalibrationCollectionProtocol,
) -> dict[str, Any]:
    override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    commands = dict(override.get("commands", {}))
    fixed = [float(value) for value in protocol.fixed_command]
    commands.update(
        {
            "vel_limit": [fixed.copy(), fixed.copy()],
            "resampling_time": 0.0,
            "heading_command": False,
            "rel_standing_envs": 0.0,
            "rel_transition_envs": 0.0,
            "small_xy_threshold": 0.0,
        }
    )
    override["commands"] = commands
    override.pop("action_execution_fault", None)
    return override


def _faulted_env_override(base: Mapping[str, Any], *, gain: float) -> dict[str, Any]:
    override = copy.deepcopy(dict(base))
    override["action_execution_fault"] = {"mode": "gain", "gain": float(gain)}
    return override


def main() -> int:
    args = _parse_args()
    protocol, protocol_bytes, protocol_sha256 = load_gain_calibration_protocol(args.protocol)
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    observed_sha256 = sha256_file(source_checkpoint)
    if observed_sha256 != args.expected_source_sha256:
        raise ValueError(
            "source checkpoint SHA256 mismatch: "
            f"expected={args.expected_source_sha256} observed={observed_sha256}"
        )
    cfg = _compose_task(protocol)
    base_override = _base_env_override(cfg, protocol)
    task_backend_payload, task_backend_sha256 = canonicalize_resolved_task_backend_payload(
        cfg, base_override
    )
    loaded = load_fada_policy_checkpoint(source_checkpoint, device=args.device)
    ensure_registries()

    def environment_factory(point, split):
        apply_training_seed(
            split.seed,
            torch_runtime=True,
            cuda=str(args.device).startswith("cuda"),
        )
        return create_env(
            cfg,
            num_envs=1,
            env_cfg_override=_faulted_env_override(base_override, gain=point.gain),
            sim_backend=protocol.sim_backend,
            task_name=protocol.task_name,
        )

    artifact = collect_gain_calibration_rollouts(
        loaded.policy,
        protocol,
        environment_factory,
        identity=GainCalibrationRawIdentity(
            source_checkpoint_sha256=observed_sha256,
            source_checkpoint_path=str(source_checkpoint),
            protocol_sha256=protocol_sha256,
            resolved_task_backend_sha256=task_backend_sha256,
            axis_catalog_version=CALIBRATION_AXIS_CATALOG_VERSION,
        ),
        protocol_bytes=protocol_bytes,
        resolved_task_backend_payload=task_backend_payload,
    )
    output = save_gain_calibration_raw_rollouts(args.output, artifact)
    print(
        json.dumps(
            {
                "schema": artifact["schema_version"],
                "output": str(output),
                "rows": int(artifact["observation_history"].shape[0]),
                "source_checkpoint_sha256": observed_sha256,
                "protocol_sha256": protocol_sha256,
                "resolved_task_backend_sha256": task_backend_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
