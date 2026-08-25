from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.algos.torch.fada_context.calibration_types import CalibrationAxisSpec

GAIN_CALIBRATION_RAW_SCHEMA = "unilab_fada_gain_calibration_raw_rollouts_v2"
_LEGACY_GAIN_CALIBRATION_RAW_SCHEMA = "unilab_fada_gain_calibration_raw_rollouts_v1"
_LEGACY_METHOD_CONTRACT_ID = "FADA-CONTEXT-METHOD-v007"
_LEGACY_TRAINING_CONTRACT_ID = "FADA-CONTEXT-TRAIN-v006"
_LEGACY_AXIS_CATALOG_VERSION = "gain-delay-offset-v1"
_LEGACY_AXIS_NAMES = ("gain", "delay", "offset")
_APPROVED_POINTS = tuple(
    (round(-1.0 + 2.0 * index / 31.0, 9), round(0.8 + 0.4 * index / 31.0, 9))
    for index in range(32)
)
_APPROVED_SPLITS = (("train", 0, 101), ("validation", 1, 201))
_HEX_DIGITS = frozenset("0123456789abcdef")
_RESERVED_AXIS_METADATA_KEYS = frozenset(
    {
        "active_axes",
        "axis_catalog_version",
        "axis_count",
        "axis_names",
        "axis_spec",
        "catalog_version",
    }
)


@dataclass(frozen=True)
class GainCalibrationPoint:
    c_true: float
    gain: float

    def validate(self) -> GainCalibrationPoint:
        values = np.asarray((self.c_true, self.gain), dtype=np.float64)
        if not bool(np.isfinite(values).all()):
            raise ValueError("gain calibration point values must be finite")
        if self.gain <= 0.0:
            raise ValueError("gain calibration physical gain must be positive")
        return self


@dataclass(frozen=True)
class GainCalibrationSplit:
    name: str
    split_id: int
    seed: int

    def validate(self) -> GainCalibrationSplit:
        if not self.name or self.split_id < 0 or self.seed < 0:
            raise ValueError("gain calibration split name, id, and seed must be valid")
        return self


@dataclass(frozen=True)
class GainCalibrationScenarioSpec:
    point: GainCalibrationPoint
    split: GainCalibrationSplit
    fixed_command: tuple[float, ...]
    accepted_rows: int
    max_environment_steps: int
    observation_key: str = "obs"
    command_key: str = "commands"

    def validate(self) -> GainCalibrationScenarioSpec:
        self.point.validate()
        self.split.validate()
        command = np.asarray(self.fixed_command, dtype=np.float32)
        if command.ndim != 1 or command.size == 0 or not bool(np.isfinite(command).all()):
            raise ValueError("gain calibration fixed command must be a finite vector")
        if self.accepted_rows <= 0 or self.max_environment_steps <= 0:
            raise ValueError("gain calibration row quota and environment limit must be positive")
        if not self.observation_key or not self.command_key:
            raise ValueError("gain calibration observation and command keys are required")
        return self


@dataclass(frozen=True)
class GainCalibrationCollectionProtocol:
    version: str
    task_config: str
    task_name: str
    sim_backend: str
    observation_key: str
    command_key: str
    fixed_command: tuple[float, ...]
    points: tuple[GainCalibrationPoint, ...]
    splits: tuple[GainCalibrationSplit, ...]
    accepted_rows_per_scenario: int
    max_environment_steps_per_scenario: int

    def validate_approved(self) -> GainCalibrationCollectionProtocol:
        observed_points = tuple(
            (float(point.c_true), float(point.gain))
            for point in self.points
            if isinstance(point, GainCalibrationPoint)
        )
        if observed_points != _APPROVED_POINTS or len(observed_points) != len(self.points):
            raise ValueError(
                f"protocol does not match the exact approved gain grid {_APPROVED_POINTS}"
            )
        observed_splits = tuple(
            (split.name, int(split.split_id), int(split.seed))
            for split in self.splits
            if isinstance(split, GainCalibrationSplit)
        )
        if observed_splits != _APPROVED_SPLITS or len(observed_splits) != len(self.splits):
            raise ValueError("protocol does not match the approved train/validation splits")
        if (
            self.version != "gain-smoke-v2"
            or self.task_config != "g1_walk_flat/mujoco"
            or self.task_name != "G1WalkFlat"
            or self.sim_backend != "mujoco"
            or self.observation_key != "obs"
            or self.command_key != "commands"
            or tuple(float(value) for value in self.fixed_command) != (0.4, 0.0, 0.0)
            or int(self.accepted_rows_per_scenario) != 32
            or int(self.max_environment_steps_per_scenario) != 512
        ):
            raise ValueError("protocol does not match the approved gain smoke identity")
        return self


@dataclass(frozen=True)
class GainCalibrationRawIdentity:
    source_checkpoint_sha256: str
    source_checkpoint_path: str
    protocol_sha256: str
    resolved_task_backend_sha256: str
    axis_catalog_version: str

    def validate(self, axis_spec: CalibrationAxisSpec) -> GainCalibrationRawIdentity:
        for name in (
            "source_checkpoint_sha256",
            "protocol_sha256",
            "resolved_task_backend_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
                raise ValueError(f"raw rollout {name} must be a lowercase SHA256")
        if not self.source_checkpoint_path:
            raise ValueError("raw rollout source checkpoint path is required")
        if self.axis_catalog_version != axis_spec.catalog_version:
            raise ValueError("raw rollout axis catalog version mismatch")
        return self


@dataclass(frozen=True)
class GainCalibrationScenarioResult:
    rows: Mapping[str, Any]
    environment_steps: int
    rejected_transactions: int
    next_rollout_id: int
