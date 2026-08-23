from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from unilab.algos.torch.fada_context.calibration import CalibrationAxisSpec

_IDENTITY_FIELDS = (
    "source_tracker_sha256",
    "dataset_sha256",
    "split_sha256",
)
_DIRECTION_STAGE: Literal["direction_frozen"] = "direction_frozen"
_COEFFICIENT_STAGE: Literal["coefficient_frozen"] = "coefficient_frozen"
_COMPENSATION_RATIO_LIMIT = 0.1
_COEFFICIENT_ERROR_LIMIT = 0.05


@dataclass(frozen=True)
class CalibrationStageIdentity:
    source_tracker_sha256: str
    dataset_sha256: str
    split_sha256: str
    axis_spec: CalibrationAxisSpec

    def validate(self) -> CalibrationStageIdentity:
        for name in _IDENTITY_FIELDS:
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"calibration stage identity {name} must be a "
                    "64-character lowercase hexadecimal digest"
                )
        if not isinstance(self.axis_spec, CalibrationAxisSpec):
            raise ValueError("calibration stage identity axis spec is incomplete")
        return self


@dataclass(frozen=True)
class DirectionStageConfig:
    steps_per_axis: int = 100
    learning_rate: float = 3.0e-4
    compensation_ratio_threshold: float = 0.1
    training_split_id: int = 0
    validation_split_id: int = 1

    def __post_init__(self) -> None:
        if self.steps_per_axis <= 0:
            raise ValueError("Stage 1 steps_per_axis must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Stage 1 learning_rate must be finite and positive")
        if not 0 < self.compensation_ratio_threshold <= _COMPENSATION_RATIO_LIMIT:
            raise ValueError("Stage 1 compensation_ratio_threshold must be in (0,0.1]")
        if self.training_split_id == self.validation_split_id:
            raise ValueError("Stage 1 training and validation split IDs must differ")


@dataclass(frozen=True)
class DirectionDiagnosticConfig:
    checkpoint_steps: tuple[int, ...] = (0, 25, 100, 300, 1000)
    learning_rate: float = 3.0e-4
    training_split_id: int = 0
    validation_split_id: int = 1

    def __post_init__(self) -> None:
        if (
            not self.checkpoint_steps
            or any(
                not isinstance(step, int) or isinstance(step, bool) or step < 0
                for step in self.checkpoint_steps
            )
            or tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
        ):
            raise ValueError(
                "Stage 1 diagnostic checkpoint_steps must be non-negative, unique, and ordered"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Stage 1 diagnostic learning_rate must be finite and positive")
        if self.training_split_id == self.validation_split_id:
            raise ValueError("Stage 1 diagnostic training and validation split IDs must differ")


@dataclass(frozen=True)
class DirectionDiagnosticPoint:
    axis_index: int
    step: int
    training_loss: float
    training_compensation_ratio: float
    validation_compensation_ratio: float
    direction_norm: float


@dataclass(frozen=True)
class DirectionGeometryConfig:
    training_split_id: int = 0
    validation_split_id: int = 1
    minimum_abs_coefficient: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.training_split_id == self.validation_split_id:
            raise ValueError("direction geometry training and validation split IDs must differ")
        if (
            not math.isfinite(self.minimum_abs_coefficient)
            or self.minimum_abs_coefficient <= 0
        ):
            raise ValueError(
                "direction geometry minimum_abs_coefficient must be finite and positive"
            )


@dataclass(frozen=True)
class DirectionGeometrySplitReport:
    split_id: int
    sample_count: int
    excluded_zero_coefficient_count: int
    excluded_zero_target_error_count: int
    zero_direction_count: int
    individual_ratio_median: float
    individual_ratio_p90: float
    individual_ratio_max: float
    individual_gate_fraction: float
    top1_energy_fraction: float
    cosine_to_consensus_mean: float
    cosine_to_consensus_p10: float
    opposing_direction_fraction: float
    direction_norm_p10: float
    direction_norm_median: float
    direction_norm_p90: float
    direction_norm_p90_p10_ratio: float


@dataclass(frozen=True)
class DirectionGeometryAxisReport:
    axis_index: int
    supervision_scope: Literal["executed_first_action"]
    solver: Literal["linear_decoder_minimum_norm"]
    shared_training_ratio: float
    shared_validation_ratio: float
    training: DirectionGeometrySplitReport
    validation: DirectionGeometrySplitReport


@dataclass(frozen=True)
class CoefficientStageConfig:
    steps: int = 1000
    learning_rate: float = 3.0e-4
    coefficient_error_threshold: float = 0.05
    training_split_id: int = 0
    validation_split_id: int = 1

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("Stage 2 steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Stage 2 learning_rate must be finite and positive")
        if not 0 < self.coefficient_error_threshold <= _COEFFICIENT_ERROR_LIMIT:
            raise ValueError("Stage 2 coefficient_error_threshold must be in (0,0.05]")
        if self.training_split_id == self.validation_split_id:
            raise ValueError("Stage 2 training and validation split IDs must differ")


@dataclass(frozen=True)
class DirectionStageResult:
    stage: Literal["direction_frozen"]
    artifact_path: Path
    artifact_sha256: str
    compensation_ratios: tuple[float, ...]


@dataclass(frozen=True)
class CoefficientStageResult:
    stage: Literal["coefficient_frozen"]
    artifact_path: Path
    artifact_sha256: str
    parent_stage_sha256: str
    coefficient_error: float


@dataclass(frozen=True)
class ScaleStageResult:
    stage: Literal["complete"]
    artifact_path: Path
    artifact_sha256: str
    parent_stage_sha256: str
    scale_evidence_sha256: str


@dataclass(frozen=True)
class SerialCalibrationConfig:
    stage1_steps_per_axis: int = 100
    stage2_steps: int = 1000
    learning_rate: float = 3.0e-4
    compensation_ratio_threshold: float = 0.1
    coefficient_error_threshold: float = 0.05
    training_split_id: int = 0
    validation_split_id: int = 1

    def __post_init__(self) -> None:
        if self.stage1_steps_per_axis <= 0 or self.stage2_steps <= 0:
            raise ValueError("serial calibration steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("serial calibration learning_rate must be finite and positive")
        if not 0 < self.compensation_ratio_threshold <= _COMPENSATION_RATIO_LIMIT:
            raise ValueError("compensation_ratio_threshold must be in (0,0.1]")
        if not 0 < self.coefficient_error_threshold <= _COEFFICIENT_ERROR_LIMIT:
            raise ValueError("coefficient_error_threshold must be in (0,0.05]")
        if self.training_split_id == self.validation_split_id:
            raise ValueError("training and validation split IDs must differ")

    def direction_stage(self) -> DirectionStageConfig:
        return DirectionStageConfig(
            steps_per_axis=self.stage1_steps_per_axis,
            learning_rate=self.learning_rate,
            compensation_ratio_threshold=self.compensation_ratio_threshold,
            training_split_id=self.training_split_id,
            validation_split_id=self.validation_split_id,
        )

    def coefficient_stage(self) -> CoefficientStageConfig:
        return CoefficientStageConfig(
            steps=self.stage2_steps,
            learning_rate=self.learning_rate,
            coefficient_error_threshold=self.coefficient_error_threshold,
            training_split_id=self.training_split_id,
            validation_split_id=self.validation_split_id,
        )
