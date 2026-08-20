from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from unilab.algos.torch.distill.fada import (
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlanner,
    PlannerIDMOutput,
    _validate_finite,
)

CALIBRATION_METHOD_CONTRACT_ID = "FADA-CONTEXT-METHOD-v007"
CALIBRATION_TRAINING_CONTRACT_ID = "FADA-CONTEXT-TRAIN-v006"
CALIBRATION_ARTIFACT_SCHEMA = "unilab_fada_calibration_artifact_v1"
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


class DirectionBank(nn.Module):
    def __init__(self, *, axis_count: int, prediction_horizon: int, latent_dim: int) -> None:
        super().__init__()
        if min(axis_count, prediction_horizon, latent_dim) <= 0:
            raise ValueError("Direction Bank dimensions must be positive")
        self.axis_count = int(axis_count)
        self.prediction_horizon = int(prediction_horizon)
        self.latent_dim = int(latent_dim)
        self.directions = nn.Parameter(torch.zeros(axis_count, prediction_horizon, latent_dim))
        self.register_buffer("normalization_scale", torch.ones(axis_count))

    def normalize_(self) -> DirectionBank:
        for axis_index in range(self.axis_count):
            self.normalize_axis_(axis_index)
        return self

    def normalize_axis_(self, axis_index: int) -> DirectionBank:
        if axis_index < 0 or axis_index >= self.axis_count:
            raise ValueError("axis_index is outside Direction Bank")
        with torch.no_grad():
            norm = self.directions[axis_index].norm()
            if bool(norm <= 0) or not bool(torch.isfinite(norm)):
                raise ValueError("Direction Bank cannot publish zero or non-finite directions")
            self.directions[axis_index].div_(norm)
            self.normalization_scale[axis_index].mul_(norm)
        return self

    def compose(
        self,
        latent: torch.Tensor,
        coefficients: torch.Tensor,
        scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tuple(self.directions.shape) != (
            self.axis_count,
            self.prediction_horizon,
            self.latent_dim,
        ):
            raise ValueError("Direction Bank directions must be [axis, horizon, latent]")
        if latent.ndim != 3 or tuple(latent.shape[1:]) != (
            self.prediction_horizon,
            self.latent_dim,
        ):
            raise ValueError("latent shape must be [batch, prediction_horizon, latent_dim]")
        if coefficients.ndim != 2 or coefficients.shape[-1] != self.axis_count:
            raise ValueError("coefficients shape must be [batch, axis_count]")
        _validate_finite("latent", latent)
        _validate_finite("coefficients", coefficients)
        if scales is None:
            scales = coefficients * self.normalization_scale.to(coefficients)[None]
        if scales.shape != coefficients.shape:
            raise ValueError("scales and coefficients must have the same shape")
        _validate_finite("scales", scales)
        return latent + torch.einsum("bm,mkd->bkd", scales, self.directions.to(latent))


class CoefficientEncoder(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        axis_count: int,
        hidden_dim: int = 128,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or layers <= 0:
            raise ValueError("Coefficient Encoder dimensions must be positive")
        self.history_length = 30
        self.axis_count = int(axis_count)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, self.history_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4 if hidden_dim % 4 == 0 else 1,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.readout = nn.Linear(hidden_dim, axis_count)

    def forward(self, state_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        if state_history.ndim != 3 or action_history.ndim != 3:
            raise ValueError("histories must be rank-3")
        if (
            state_history.shape[1] != self.history_length
            or action_history.shape[1] != self.history_length
        ):
            raise ValueError("history length must be 30")
        if state_history.shape[0] != action_history.shape[0]:
            raise ValueError("state/action history batch sizes must match")
        _validate_finite("state_history", state_history)
        _validate_finite("action_history", action_history)
        tokens = self.state_embedding(state_history) + self.action_embedding(action_history)
        tokens = self.encoder(tokens + self.position.to(dtype=tokens.dtype, device=tokens.device))
        output = self.readout(tokens.mean(dim=1))
        _validate_finite("coefficient_readout", output)
        return output


@dataclass(frozen=True)
class MonotoneScaleCurve:
    x: torch.Tensor
    y: torch.Tensor
    slopes: torch.Tensor
    kind: str = "pchip"

    @classmethod
    def fit(cls, x: torch.Tensor, y: torch.Tensor) -> MonotoneScaleCurve:
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.numel() < 2:
            raise ValueError("scale curve samples must be matching rank-1 arrays")
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError("scale curve samples must be finite")
        if bool((x[1:] <= x[:-1]).any()):
            raise ValueError("scale curve x values must be strictly increasing")
        dy = y[1:] - y[:-1]
        if not bool((dy >= 0).all()) and not bool((dy <= 0).all()):
            raise ValueError("scale curve samples must be monotone")
        slopes = _pchip_slopes(x, y)
        return cls(x.detach().clone(), y.detach().clone(), slopes.detach().clone())

    def map(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_finite("scale_reading", values)
        flat = values.flatten()
        x = self.x.to(flat)
        y = self.y.to(flat)
        slopes = self.slopes.to(flat)
        events = (flat < x[0]) | (flat > x[-1])
        clipped = flat.clamp(min=x[0], max=x[-1])
        idx = torch.bucketize(clipped, x, right=True).clamp(1, x.numel() - 1) - 1
        x0, x1 = x[idx], x[idx + 1]
        y0, y1 = y[idx], y[idx + 1]
        h = x1 - x0
        t = (clipped - x0) / h
        h00 = (2 * t**3) - (3 * t**2) + 1
        h10 = t**3 - (2 * t**2) + t
        h01 = (-2 * t**3) + (3 * t**2)
        h11 = t**3 - t**2
        mapped = h00 * y0 + h10 * h * slopes[idx] + h01 * y1 + h11 * h * slopes[idx + 1]
        mapped = mapped.clamp(min=y.min(), max=y.max())
        return mapped.reshape(values.shape), events.reshape(values.shape)


@dataclass(frozen=True)
class CalibrationReadout:
    raw_coefficients: torch.Tensor
    scales: torch.Tensor
    range_events: torch.Tensor
    jump_events: torch.Tensor
    cold_start: torch.Tensor


@dataclass(frozen=True)
class CalibratedPolicyOutput:
    predicted_future: torch.Tensor
    action_chunk: torch.Tensor
    action: torch.Tensor
    readout: CalibrationReadout


class CalibrationReadoutState:
    def __init__(self, *, axis_count: int, jump_threshold: torch.Tensor) -> None:
        threshold = torch.as_tensor(jump_threshold, dtype=torch.float32)
        if threshold.shape != (axis_count,) or not bool(torch.isfinite(threshold).all()):
            raise ValueError("jump_threshold must be finite [axis_count]")
        if bool((threshold <= 0).any()):
            raise ValueError("jump_threshold must be positive")
        self.axis_count = int(axis_count)
        self.jump_threshold = threshold
        self.previous_coefficients: torch.Tensor | None = None
        self.previous_scales: torch.Tensor | None = None
        self.previous_valid: torch.Tensor | None = None

    def reset(self, rows: torch.Tensor | None = None) -> None:
        if rows is None or self.previous_valid is None:
            self.previous_coefficients = None
            self.previous_scales = None
            self.previous_valid = None
            return
        mask = torch.as_tensor(rows, dtype=torch.bool, device=self.previous_valid.device)
        if mask.shape != self.previous_valid.shape:
            raise ValueError("readout reset mask must match the current batch")
        self.previous_valid[mask] = False

    def apply(
        self,
        coefficients: torch.Tensor,
        curves: tuple[MonotoneScaleCurve, ...],
        *,
        ready: torch.Tensor,
    ) -> CalibrationReadout:
        if coefficients.ndim != 2 or coefficients.shape[-1] != self.axis_count:
            raise ValueError("readout coefficients must be [batch, axis_count]")
        if len(curves) != self.axis_count:
            raise ValueError("readout curve count must match axis count")
        ready = torch.as_tensor(ready, dtype=torch.bool, device=coefficients.device)
        if ready.shape != (coefficients.shape[0],):
            raise ValueError("readout ready mask must be [batch]")
        _validate_finite("readout coefficients", coefficients)
        mapped_columns, range_columns = [], []
        for index, curve in enumerate(curves):
            mapped, events = curve.map(coefficients[:, index].detach())
            mapped_columns.append(mapped)
            range_columns.append(events)
        scales = torch.stack(mapped_columns, dim=1)
        range_events = torch.stack(range_columns, dim=1) & ready[:, None]
        jump_events = torch.zeros_like(range_events)
        if self.previous_coefficients is not None:
            if self.previous_coefficients.shape != coefficients.shape:
                raise ValueError("readout batch changed without reset")
            threshold = self.jump_threshold.to(coefficients)[None]
            assert self.previous_valid is not None
            jump_events = (
                coefficients - self.previous_coefficients.to(coefficients)
            ).abs() > threshold
            jump_events &= ready[:, None] & self.previous_valid.to(coefficients.device)[:, None]
            assert self.previous_scales is not None
            scales = torch.where(jump_events, self.previous_scales.to(scales), scales)
        scales = torch.where(ready[:, None], scales, torch.zeros_like(scales))
        update = ready[:, None] & ~jump_events
        if self.previous_coefficients is None:
            self.previous_coefficients = torch.zeros_like(coefficients)
            self.previous_scales = scales.detach().clone()
            self.previous_valid = ready.detach().clone()
            self.previous_coefficients[ready] = coefficients.detach()[ready]
        else:
            assert self.previous_valid is not None
            assert self.previous_scales is not None
            previous_coefficients = self.previous_coefficients.to(coefficients)
            previous_scales = self.previous_scales.to(scales)
            self.previous_coefficients = torch.where(
                update, coefficients.detach(), previous_coefficients
            )
            self.previous_scales = torch.where(update, scales.detach(), previous_scales)
            self.previous_valid = self.previous_valid.to(ready.device) | ready
        return CalibrationReadout(
            raw_coefficients=coefficients,
            scales=scales,
            range_events=range_events,
            jump_events=jump_events,
            cold_start=~ready,
        )


def _pchip_slopes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    h = x[1:] - x[:-1]
    delta = (y[1:] - y[:-1]) / h
    slopes = torch.zeros_like(y)
    if y.numel() == 2:
        slopes[:] = delta[0]
        return slopes
    same_sign = delta[:-1] * delta[1:] > 0
    w1 = (2 * h[1:]) + h[:-1]
    w2 = h[1:] + (2 * h[:-1])
    harmonic = (w1 + w2) / ((w1 / delta[:-1]) + (w2 / delta[1:]))
    slopes[1:-1] = torch.where(same_sign, harmonic, torch.zeros_like(harmonic))
    slopes[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    slopes[-1] = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if slopes[0] * delta[0] <= 0:
        slopes[0] = 0
    elif delta[0] * delta[1] < 0 and abs(slopes[0]) > abs(3 * delta[0]):
        slopes[0] = 3 * delta[0]
    if slopes[-1] * delta[-1] <= 0:
        slopes[-1] = 0
    elif delta[-1] * delta[-2] < 0 and abs(slopes[-1]) > abs(3 * delta[-1]):
        slopes[-1] = 3 * delta[-1]
    return slopes


def fit_scale_curve_bank(
    readings: torch.Tensor, scales: torch.Tensor
) -> tuple[MonotoneScaleCurve, ...]:
    if readings.ndim != 2 or scales.shape != readings.shape:
        raise ValueError("scale curve bank samples must be [axis, grid]")
    return tuple(MonotoneScaleCurve.fit(readings[i], scales[i]) for i in range(readings.shape[0]))


def _validate_calibration_artifact_metadata(metadata: object) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("calibration artifact metadata is missing")
    digest_fields = (
        "source_tracker_sha256",
        "dataset_sha256",
        "split_sha256",
        "parent_stage_sha256",
        "scale_evidence_sha256",
    )
    for name in digest_fields:
        value = metadata.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"calibration artifact lineage {name} must be a "
                "64-character lowercase hexadecimal digest"
            )
    if not isinstance(metadata.get("axis_catalog_version"), str):
        raise ValueError("calibration artifact metadata axis catalog is missing")
    if metadata["axis_catalog_version"] != CALIBRATION_AXIS_CATALOG_VERSION:
        raise ValueError("calibration artifact axis catalog mismatch")
    if metadata.get("stage") != "complete":
        raise ValueError("calibration artifact lineage stage must be complete")
    return metadata


def save_calibration_artifact(
    path: str | Path,
    *,
    config: FADAArchitectureConfig,
    direction_bank: DirectionBank,
    scale_curves: tuple[MonotoneScaleCurve, ...],
    coefficient_encoder: CoefficientEncoder,
    metadata: Mapping[str, Any],
) -> Path:
    if len(scale_curves) != direction_bank.axis_count:
        raise ValueError("one scale curve is required per direction axis")
    if direction_bank.axis_count != len(CALIBRATION_AXIS_NAMES):
        raise ValueError("calibration artifact axis count does not match the active catalog")
    if tuple(direction_bank.directions.shape[1:]) != (
        config.prediction_horizon,
        config.hidden_dim,
    ):
        raise ValueError("calibration artifact Direction Bank architecture mismatch")
    direction_norms = direction_bank.directions.detach().flatten(1).norm(dim=1)
    if not torch.allclose(direction_norms, torch.ones_like(direction_norms), rtol=1e-5, atol=1e-6):
        raise ValueError("calibration artifact requires normalized Direction Bank fields")
    if bool((direction_bank.normalization_scale <= 0).any()) or not bool(
        torch.isfinite(direction_bank.normalization_scale).all()
    ):
        raise ValueError("calibration artifact normalization scale must be finite and positive")
    if coefficient_encoder.history_length != config.history_length:
        raise ValueError("calibration artifact Encoder history mismatch")
    expected_encoder = (
        config.obs_dim,
        config.action_dim,
        len(CALIBRATION_AXIS_NAMES),
        128,
        2,
    )
    observed_encoder = (
        coefficient_encoder.state_embedding.in_features,
        coefficient_encoder.action_embedding.in_features,
        coefficient_encoder.axis_count,
        coefficient_encoder.state_embedding.out_features,
        len(coefficient_encoder.encoder.layers),
    )
    if observed_encoder != expected_encoder:
        raise ValueError("calibration artifact Coefficient Encoder architecture mismatch")
    _validate_calibration_artifact_metadata(metadata)
    for curve in scale_curves:
        if curve.x.numel() != 21:
            raise ValueError("calibration artifact requires a 21-point scale grid")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CALIBRATION_ARTIFACT_SCHEMA,
        "method_contract_id": CALIBRATION_METHOD_CONTRACT_ID,
        "training_contract_id": CALIBRATION_TRAINING_CONTRACT_ID,
        "architecture": asdict(config),
        "axis_names": CALIBRATION_AXIS_NAMES,
        "direction_bank": direction_bank.state_dict(),
        "coefficient_encoder": coefficient_encoder.state_dict(),
        "coefficient_encoder_config": {
            "state_dim": coefficient_encoder.state_embedding.in_features,
            "action_dim": coefficient_encoder.action_embedding.in_features,
            "axis_count": coefficient_encoder.axis_count,
            "hidden_dim": coefficient_encoder.state_embedding.out_features,
            "layers": len(coefficient_encoder.encoder.layers),
        },
        "scale_curves": [
            {"x": curve.x, "y": curve.y, "slopes": curve.slopes, "kind": curve.kind}
            for curve in scale_curves
        ],
        "metadata": dict(metadata),
    }
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_calibration_artifact(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CALIBRATION_ARTIFACT_SCHEMA
    ):
        raise ValueError("unsupported calibration artifact schema")
    if payload.get("method_contract_id") != CALIBRATION_METHOD_CONTRACT_ID:
        raise ValueError("calibration artifact method Contract mismatch")
    if payload.get("training_contract_id") != CALIBRATION_TRAINING_CONTRACT_ID:
        raise ValueError("calibration artifact training Contract mismatch")
    if not isinstance(payload.get("architecture"), Mapping):
        raise ValueError("calibration artifact architecture is missing")
    if tuple(payload.get("axis_names", ())) != CALIBRATION_AXIS_NAMES:
        raise ValueError("calibration artifact axis catalog mismatch")
    _validate_calibration_artifact_metadata(payload.get("metadata"))
    if (
        not isinstance(payload.get("direction_bank"), Mapping)
        or not isinstance(payload.get("coefficient_encoder"), Mapping)
        or not isinstance(payload.get("coefficient_encoder_config"), Mapping)
        or not isinstance(payload.get("scale_curves"), list)
    ):
        raise ValueError("calibration artifact is missing typed owners")
    _validate_finite_state_tree("calibration artifact", payload)
    direction_state = payload["direction_bank"]
    directions = direction_state.get("directions")
    normalization_scale = direction_state.get("normalization_scale")
    if (
        not isinstance(directions, torch.Tensor)
        or directions.ndim != 3
        or not isinstance(normalization_scale, torch.Tensor)
        or normalization_scale.shape != (directions.shape[0],)
    ):
        raise ValueError("calibration artifact Direction Bank state is malformed")
    norms = directions.flatten(1).norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-6):
        raise ValueError("calibration artifact Direction Bank is not normalized")
    if bool((normalization_scale <= 0).any()):
        raise ValueError("calibration artifact normalization scale must be positive")
    if len(payload["scale_curves"]) != len(CALIBRATION_AXIS_NAMES):
        raise ValueError("calibration artifact scale curve count mismatch")
    for curve in payload["scale_curves"]:
        _validate_scale_curve_payload(curve)
    return payload


def _validate_finite_state_tree(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta" or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} tensors must be finite materialized values")
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_finite_state_tree(name, child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_finite_state_tree(name, child)


def _validate_scale_curve_payload(value: Any) -> MonotoneScaleCurve:
    if (
        not isinstance(value, Mapping)
        or not all(isinstance(value.get(name), torch.Tensor) for name in ("x", "y", "slopes"))
        or value.get("kind") != "pchip"
    ):
        raise ValueError("calibration scale curve is malformed")
    if value["x"].numel() != 21:
        raise ValueError("calibration scale curve requires a 21-point grid")
    validated = MonotoneScaleCurve.fit(value["x"], value["y"])
    if value["slopes"].shape != validated.slopes.shape or not torch.allclose(
        value["slopes"],
        validated.slopes,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("calibration PCHIP slopes are inconsistent")
    return validated


class CalibratedFADAPolicy(nn.Module):
    def __init__(
        self,
        config: FADAArchitectureConfig,
        *,
        direction_bank: DirectionBank,
        coefficient_encoder: CoefficientEncoder,
        scale_curves: tuple[MonotoneScaleCurve, ...],
        planner: FADAPlanner | None = None,
        idm: FADAInverseDynamicsModel | None = None,
    ) -> None:
        super().__init__()
        if len(scale_curves) != direction_bank.axis_count:
            raise ValueError("scale curve count must match direction axis count")
        self.config = config
        self.planner = planner if planner is not None else FADAPlanner(config)
        self.idm = idm if idm is not None else FADAInverseDynamicsModel(config)
        self.direction_bank = direction_bank
        self.coefficient_encoder = coefficient_encoder
        direction_device = direction_bank.directions.device
        direction_dtype = direction_bank.directions.dtype
        self.scale_curves = tuple(
            MonotoneScaleCurve(
                x=curve.x.to(device=direction_device, dtype=direction_dtype),
                y=curve.y.to(device=direction_device, dtype=direction_dtype),
                slopes=curve.slopes.to(device=direction_device, dtype=direction_dtype),
                kind=curve.kind,
            )
            for curve in scale_curves
        )
        for module in (self.planner, self.idm, self.direction_bank, self.coefficient_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()

    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> PlannerIDMOutput:
        state = CalibrationReadoutState(
            axis_count=self.direction_bank.axis_count,
            jump_threshold=torch.full(
                (self.direction_bank.axis_count,),
                torch.finfo(torch.float32).max,
            ),
        )
        output = self.forward_with_readout(
            observation_history,
            action_history,
            command,
            ready=torch.ones(
                observation_history.shape[0], dtype=torch.bool, device=observation_history.device
            ),
            readout_state=state,
        )
        return PlannerIDMOutput(
            predicted_future=output.predicted_future,
            action_chunk=output.action_chunk,
            action=output.action,
        )

    def forward_with_readout(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
        *,
        ready: torch.Tensor,
        readout_state: CalibrationReadoutState,
    ) -> CalibratedPolicyOutput:
        predicted_future = self.planner(observation_history, command)
        latent = self.idm.encode_latent(observation_history, action_history, predicted_future)
        ready = torch.as_tensor(ready, dtype=torch.bool, device=latent.device)
        if ready.shape != (latent.shape[0],):
            raise ValueError("calibration readiness must be [batch]")
        coefficients = torch.zeros(
            latent.shape[0],
            self.direction_bank.axis_count,
            device=latent.device,
            dtype=latent.dtype,
        )
        if bool(ready.any()):
            coefficients[ready] = self.coefficient_encoder(
                observation_history[ready, -30:],
                action_history[ready, -30:],
            )
        readout = readout_state.apply(coefficients, self.scale_curves, ready=ready)
        calibrated = self.direction_bank.compose(latent, coefficients, scales=readout.scales)
        action_chunk = self.idm.decode_latent(calibrated)
        return CalibratedPolicyOutput(
            predicted_future=predicted_future,
            action_chunk=action_chunk,
            action=action_chunk[:, 0],
            readout=readout,
        )

    def reconstruct_with_coefficients(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> PlannerIDMOutput:
        predicted_future = self.planner(observation_history, command)
        latent = self.idm.encode_latent(observation_history, action_history, predicted_future)
        if coefficients.shape != (latent.shape[0], self.direction_bank.axis_count):
            raise ValueError("calibration coefficients must be [batch, axis_count]")
        state = CalibrationReadoutState(
            axis_count=self.direction_bank.axis_count,
            jump_threshold=torch.full(
                (self.direction_bank.axis_count,),
                torch.finfo(torch.float32).max,
            ),
        )
        readout = state.apply(
            coefficients.to(latent),
            self.scale_curves,
            ready=torch.ones(latent.shape[0], dtype=torch.bool, device=latent.device),
        )
        calibrated = self.direction_bank.compose(
            latent,
            coefficients.to(latent),
            scales=readout.scales,
        )
        action_chunk = self.idm.decode_latent(calibrated)
        return PlannerIDMOutput(
            predicted_future=predicted_future,
            action_chunk=action_chunk,
            action=action_chunk[:, 0],
        )
