from __future__ import annotations

from pathlib import Path

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import (
    CalibrationAxisSpec,
    CalibrationRolloutBatch,
)
from unilab.algos.torch.fada_context.calibration_training.io import (
    CalibrationScaleEvidence,
    save_calibration_scale_evidence,
)
from unilab.algos.torch.fada_context.calibration_training.stage1 import (
    run_direction_stage_training,
)
from unilab.algos.torch.fada_context.calibration_training.stage2 import (
    run_coefficient_stage_training,
)
from unilab.algos.torch.fada_context.calibration_training.stage3 import (
    run_scale_stage_fitting,
)
from unilab.algos.torch.fada_context.calibration_training.types import (
    CalibrationStageIdentity,
    SerialCalibrationConfig,
)


def run_serial_calibration_training(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    *,
    output_dir: str | Path,
    source_tracker_sha256: str,
    dataset_sha256: str,
    split_sha256: str,
    axis_spec: CalibrationAxisSpec,
    scale_evidence: CalibrationScaleEvidence | None = None,
    scale_evidence_path: str | Path | None = None,
    config: SerialCalibrationConfig = SerialCalibrationConfig(),
) -> dict[str, object]:
    """Compose S1/S2/S3 through the same persisted boundaries as independent runs."""

    identity = CalibrationStageIdentity(
        source_tracker_sha256=source_tracker_sha256,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        axis_spec=axis_spec,
    ).validate()
    if (scale_evidence is None) == (scale_evidence_path is None):
        raise ValueError("serial calibration requires exactly one typed scale evidence source")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    direction_result = run_direction_stage_training(
        policy,
        batch,
        output / "stage1_direction_frozen.pt",
        identity,
        config.direction_stage(),
    )
    coefficient_result = run_coefficient_stage_training(
        policy,
        batch,
        direction_result.artifact_path,
        output / "stage2_coefficient_frozen.pt",
        identity,
        config.coefficient_stage(),
    )
    if scale_evidence is not None:
        scale_path = save_calibration_scale_evidence(
            output / "scale_evidence.pt",
            scale_evidence,
        )
    else:
        assert scale_evidence_path is not None
        scale_path = Path(scale_evidence_path).expanduser().resolve()
    scale_result = run_scale_stage_fitting(
        policy,
        coefficient_result.artifact_path,
        scale_path,
        output / "calibration_artifact.pt",
        identity,
    )
    return {
        "stage": "complete",
        "artifact_path": str(scale_result.artifact_path),
        "coefficient_error": coefficient_result.coefficient_error,
        "axis_count": axis_spec.axis_count,
        "axis_spec": axis_spec.to_payload(),
        "direction_stage": direction_result,
        "coefficient_stage": coefficient_result,
        "scale_stage": scale_result,
    }
