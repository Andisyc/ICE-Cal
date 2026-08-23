from __future__ import annotations

from pathlib import Path

import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import (
    CalibrationAxisSpec,
    CoefficientEncoder,
    DirectionBank,
    MonotoneScaleCurve,
    fit_scale_curve_bank,
)
from unilab.algos.torch.fada_context.calibration_training.io import (
    _atomic_save_deployment_artifact,
    _calibration_scale_evidence_from_payload,
    _identity_payload,
    _load_coefficient_stage_artifact,
    _load_exact_torch_payload,
    _sha256_file,
    _validate_scale_evidence_tensors,
)
from unilab.algos.torch.fada_context.calibration_training.types import (
    CalibrationStageIdentity,
    ScaleStageResult,
)


def fit_scale_stage(
    readings: torch.Tensor,
    candidate_scales: torch.Tensor,
    action_errors: torch.Tensor,
    axis_spec: CalibrationAxisSpec,
) -> tuple[MonotoneScaleCurve, ...]:
    coefficient_scan_grid = torch.linspace(
        -1.0,
        1.0,
        21,
        dtype=readings.dtype,
        device=readings.device,
    ).repeat(axis_spec.axis_count, 1)
    _validate_scale_evidence_tensors(
        coefficient_scan_grid,
        readings,
        candidate_scales,
        action_errors,
        axis_spec,
    )
    optimal_indices = action_errors.argmin(dim=-1)
    optimal_scales = candidate_scales.to(action_errors)[optimal_indices]
    curves = fit_scale_curve_bank(readings.mean(dim=2), optimal_scales.mean(dim=2))
    for axis_index, curve in enumerate(curves):
        predicted, _ = curve.map(readings[axis_index].reshape(-1))
        expected = optimal_scales[axis_index].reshape(-1)
        residual = torch.sum((expected - predicted) ** 2)
        centered = torch.sum((expected - expected.mean()) ** 2)
        if bool(centered <= 0):
            raise ValueError("Stage 3 R^2 requires non-constant scale evidence")
        r_squared = 1.0 - residual / centered
        if not bool(torch.isfinite(r_squared)) or bool(r_squared < 0.95):
            raise ValueError(f"Stage 3 R^2 {float(r_squared):.6f} is below 0.95")
    return curves


def run_scale_stage_fitting(
    policy: FADAPlannerIDMPolicy,
    coefficient_artifact_path: str | Path,
    scale_evidence_path: str | Path,
    output_path: str | Path,
    identity: CalibrationStageIdentity,
) -> ScaleStageResult:
    identity.validate()
    direction_bank, encoder, parent_digest, _ = _load_coefficient_stage_artifact(
        coefficient_artifact_path,
        policy=policy,
        identity=identity,
    )
    scale_payload, scale_digest = _load_exact_torch_payload(scale_evidence_path)
    evidence = _calibration_scale_evidence_from_payload(
        scale_payload,
        identity,
    )
    curves = fit_scale_stage(
        evidence.readings,
        evidence.candidate_scales,
        evidence.action_errors,
        evidence.axis_spec,
    )
    artifact_path = _atomic_save_deployment_artifact(
        output_path,
        policy=policy,
        direction_bank=direction_bank,
        coefficient_encoder=encoder,
        scale_curves=curves,
        axis_spec=identity.axis_spec,
        metadata={
            **_identity_payload(identity, include_axis_spec=False),
            "stage": "complete",
            "parent_stage_sha256": parent_digest,
            "scale_evidence_sha256": scale_digest,
        },
    )
    return ScaleStageResult(
        stage="complete",
        artifact_path=artifact_path,
        artifact_sha256=_sha256_file(artifact_path),
        parent_stage_sha256=parent_digest,
        scale_evidence_sha256=scale_digest,
    )
