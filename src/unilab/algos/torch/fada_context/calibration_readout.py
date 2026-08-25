from __future__ import annotations

from dataclasses import dataclass

import torch

from unilab.algos.torch.distill.fada import _validate_finite


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
