from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from unilab.algos.torch.distill.fada import (
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlanner,
    PlannerIDMOutput,
)
from unilab.algos.torch.fada_context.calibration_models import (
    CoefficientEncoder,
    DirectionBank,
)
from unilab.algos.torch.fada_context.calibration_readout import (
    CalibrationReadout,
    CalibrationReadoutState,
    MonotoneScaleCurve,
)
from unilab.algos.torch.fada_context.calibration_types import CalibrationAxisSpec


@dataclass(frozen=True)
class CalibratedPolicyOutput:
    predicted_future: torch.Tensor
    action_chunk: torch.Tensor
    action: torch.Tensor
    readout: CalibrationReadout

class CalibratedFADAPolicy(nn.Module):
    def __init__(
        self,
        config: FADAArchitectureConfig,
        *,
        direction_bank: DirectionBank,
        coefficient_encoder: CoefficientEncoder,
        scale_curves: tuple[MonotoneScaleCurve, ...],
        axis_spec: CalibrationAxisSpec,
        planner: FADAPlanner | None = None,
        idm: FADAInverseDynamicsModel | None = None,
    ) -> None:
        super().__init__()
        if len(scale_curves) != direction_bank.axis_count:
            raise ValueError("scale curve count must match direction axis count")
        if direction_bank.axis_count != axis_spec.axis_count:
            raise ValueError("calibrated policy owner widths must match the axis spec")
        self.config = config
        self.axis_spec = axis_spec
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
