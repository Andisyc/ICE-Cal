from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from unilab.algos.torch.distill.fada import FADAArchitectureConfig, _validate_finite

CALIBRATION_METHOD_CONTRACT_ID = "FADA-CONTEXT-METHOD-v008"
CALIBRATION_TRAINING_CONTRACT_ID = "FADA-CONTEXT-TRAIN-v007"
CALIBRATION_ARTIFACT_SCHEMA = "unilab_fada_calibration_artifact_v2"
CALIBRATION_AXIS_NAMES = ("gain", "delay", "offset")
CALIBRATION_AXIS_CATALOG_VERSION = "gain-delay-offset-v1"


@dataclass(frozen=True)
class FaultAxis:
    name: str
    normalized_range: tuple[float, float]
    units: str
    injection: str


class FaultAxisCatalog:
    def __init__(
        self,
        axes: tuple[FaultAxis, ...],
        *,
        version: str = CALIBRATION_AXIS_CATALOG_VERSION,
    ) -> None:
        if not axes or len({axis.name for axis in axes}) != len(axes):
            raise ValueError("axis catalog must contain unique axes")
        if not version:
            raise ValueError("axis catalog version must be non-empty")
        self.axes = axes
        self.version = version

    @classmethod
    def default(cls) -> FaultAxisCatalog:
        return cls(
            (
                FaultAxis("gain", (-1.0, 1.0), "dimensionless", "action execution gain"),
                FaultAxis("delay", (-1.0, 1.0), "control steps", "action execution delay"),
                FaultAxis("offset", (-1.0, 1.0), "action units", "action execution offset"),
            ),
            version=CALIBRATION_AXIS_CATALOG_VERSION,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise ValueError(f"unregistered fault axis: {name}") from exc

    def analytic_target(
        self, name: str, nominal_action: torch.Tensor, strength: float
    ) -> torch.Tensor:
        if nominal_action.ndim != 3:
            raise ValueError("nominal_action must be [batch, horizon, action_dim]")
        _validate_finite("nominal_action", nominal_action)
        if not torch.isfinite(torch.tensor(float(strength))):
            raise ValueError("fault strength must be finite")
        if name == "gain":
            if float(strength) == 0.0:
                raise ValueError("gain strength cannot be zero")
            return nominal_action / float(strength)
        if name == "delay":
            shift = int(round(float(strength)))
            if abs(float(strength) - shift) > 1e-6 or shift < 0:
                raise ValueError("delay strength must be a non-negative integer")
            if shift >= nominal_action.shape[1]:
                raise ValueError("delay strength exceeds prediction horizon")
            if shift == 0:
                return nominal_action.clone()
            tail = nominal_action[:, -1:].expand(-1, shift, -1)
            return torch.cat((nominal_action[:, shift:], tail), dim=1)
        if name == "offset":
            return nominal_action - float(strength)
        raise ValueError(f"unregistered fault axis: {name}")


@dataclass(frozen=True)
class CalibrationAxisSpec:
    catalog_version: str
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.catalog_version, str) or not self.catalog_version:
            raise ValueError("axis spec catalog_version must be non-empty")
        if not self.names:
            raise ValueError("axis spec names must be non-empty")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("axis spec names must be non-empty strings")
        if len(set(self.names)) != len(self.names):
            raise ValueError("axis spec names must be unique")

    @classmethod
    def from_catalog(
        cls,
        catalog: FaultAxisCatalog,
        names: tuple[str, ...] | list[str] | None = None,
    ) -> CalibrationAxisSpec:
        selected = catalog.names if names is None else tuple(names)
        for name in selected:
            catalog.index(name)
        return cls(catalog_version=catalog.version, names=selected)

    @classmethod
    def from_payload(
        cls,
        payload: object,
        catalog: FaultAxisCatalog,
    ) -> CalibrationAxisSpec:
        if not isinstance(payload, Mapping):
            raise ValueError("axis spec payload must be a mapping")
        if set(payload) != {"catalog_version", "names"}:
            raise ValueError("axis spec payload fields are invalid")
        raw_names = payload.get("names")
        if not isinstance(raw_names, (list, tuple)):
            raise ValueError("axis spec payload names must be a sequence")
        catalog_version = payload.get("catalog_version")
        if not isinstance(catalog_version, str):
            raise ValueError("axis spec payload catalog_version must be a string")
        spec = cls(
            catalog_version=catalog_version,
            names=tuple(raw_names),
        )
        if spec.catalog_version != catalog.version:
            raise ValueError("axis spec catalog version mismatch")
        for name in spec.names:
            catalog.index(name)
        return spec

    @property
    def axis_count(self) -> int:
        return len(self.names)

    def catalog_indices(self, catalog: FaultAxisCatalog) -> tuple[int, ...]:
        if catalog.version != self.catalog_version:
            raise ValueError("axis spec catalog version mismatch")
        return tuple(catalog.index(name) for name in self.names)

    def to_payload(self) -> dict[str, object]:
        return {"catalog_version": self.catalog_version, "names": list(self.names)}


@dataclass(frozen=True)
class CalibrationRolloutBatch:
    observation_history: torch.Tensor
    action_history: torch.Tensor
    command: torch.Tensor
    nominal_action_chunk: torch.Tensor
    target_action_chunk: torch.Tensor
    c_true: torch.Tensor
    axis_id: torch.Tensor
    is_held_out_combination: torch.Tensor
    injected_strength: torch.Tensor
    planner_intent: torch.Tensor
    rollout_id: torch.Tensor
    seed: torch.Tensor
    split_id: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> CalibrationRolloutBatch:
        return CalibrationRolloutBatch(
            **{
                name: value.index_select(0, indices.to(value.device))
                for name, value in self.__dict__.items()
            }
        )

    def validate(
        self, config: FADAArchitectureConfig, *, axis_count: int
    ) -> CalibrationRolloutBatch:
        expected_sequences = (
            (
                "observation_history",
                self.observation_history,
                config.history_length,
                config.obs_dim,
            ),
            ("action_history", self.action_history, config.history_length, config.action_dim),
            (
                "nominal_action_chunk",
                self.nominal_action_chunk,
                config.prediction_horizon,
                config.action_dim,
            ),
            (
                "target_action_chunk",
                self.target_action_chunk,
                config.prediction_horizon,
                config.action_dim,
            ),
        )
        for name, value, length, width in expected_sequences:
            if value.ndim != 3 or tuple(value.shape[1:]) != (length, width):
                raise ValueError(f"{name} shape mismatch")
            _validate_finite(name, value)
        if self.command.ndim != 2 or self.command.shape[-1] != config.command_dim:
            raise ValueError("command shape mismatch")
        if self.c_true.ndim != 2 or self.c_true.shape[-1] != axis_count:
            raise ValueError(f"c_true axis count mismatch: expected {axis_count}")
        if bool((self.c_true.abs() > 1.0).any()):
            raise ValueError("c_true must stay inside the normalized [-1,1] range")
        identity_fields = (
            ("axis_id", self.axis_id),
            ("rollout_id", self.rollout_id),
            ("seed", self.seed),
            ("split_id", self.split_id),
        )
        for name, value in identity_fields:
            if value.ndim != 1 or value.dtype != torch.int64:
                raise ValueError(f"{name} must be rank-1 int64")
        if (
            self.is_held_out_combination.ndim != 1
            or self.is_held_out_combination.dtype != torch.bool
        ):
            raise ValueError("is_held_out_combination must be rank-1 bool")
        single_axis = ~self.is_held_out_combination
        if bool(
            ((self.axis_id[single_axis] < 0) | (self.axis_id[single_axis] >= axis_count)).any()
        ):
            raise ValueError("single-axis row axis_id is outside the declared axis count")
        if bool((self.axis_id[self.is_held_out_combination] != -1).any()):
            raise ValueError("held-out combination rows must use axis_id=-1")
        if bool(single_axis.any()):
            active = torch.nn.functional.one_hot(
                self.axis_id[single_axis],
                num_classes=axis_count,
            ).bool()
            if bool((self.c_true[single_axis].masked_select(~active) != 0).any()):
                raise ValueError("single-axis rows may contain only their declared coefficient")
        if bool(self.is_held_out_combination.any()):
            nonzero_axes = torch.count_nonzero(
                self.c_true[self.is_held_out_combination],
                dim=1,
            )
            if bool((nonzero_axes < 2).any()):
                raise ValueError("held-out combination rows require at least two active axes")
        if self.injected_strength.ndim != 1 or self.planner_intent.ndim != 3:
            raise ValueError("injected_strength or planner_intent shape mismatch")
        if tuple(self.planner_intent.shape[1:]) != (config.prediction_horizon, config.obs_dim):
            raise ValueError("planner_intent shape mismatch")
        _validate_finite("command", self.command)
        _validate_finite("c_true", self.c_true)
        _validate_finite("injected_strength", self.injected_strength)
        _validate_finite("planner_intent", self.planner_intent)
        if bool((self.split_id < 0).any()):
            raise ValueError("split_id must be non-negative")
        batch_sizes = {
            int(self.observation_history.shape[0]),
            int(self.action_history.shape[0]),
            int(self.command.shape[0]),
            int(self.nominal_action_chunk.shape[0]),
            int(self.target_action_chunk.shape[0]),
            int(self.c_true.shape[0]),
            int(self.axis_id.shape[0]),
            int(self.is_held_out_combination.shape[0]),
            int(self.injected_strength.shape[0]),
            int(self.planner_intent.shape[0]),
            int(self.rollout_id.shape[0]),
            int(self.seed.shape[0]),
            int(self.split_id.shape[0]),
        }
        if len(batch_sizes) != 1:
            raise ValueError("calibration rollout batch sizes must match")
        return self
