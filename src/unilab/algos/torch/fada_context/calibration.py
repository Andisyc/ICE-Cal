"""Compatibility facade for ICE-Cal calibration owners."""

import torch

from unilab.algos.torch.fada_context.calibration_artifact import (
    load_calibration_artifact,
    save_calibration_artifact,
)
from unilab.algos.torch.fada_context.calibration_models import (
    CoefficientEncoder,
    DirectionBank,
)
from unilab.algos.torch.fada_context.calibration_policy import (
    CalibratedFADAPolicy,
    CalibratedPolicyOutput,
)
from unilab.algos.torch.fada_context.calibration_readout import (
    CalibrationReadout,
    CalibrationReadoutState,
    MonotoneScaleCurve,
    fit_scale_curve_bank,
)
from unilab.algos.torch.fada_context.calibration_types import (
    CALIBRATION_ARTIFACT_SCHEMA,
    CALIBRATION_AXIS_CATALOG_VERSION,
    CALIBRATION_AXIS_NAMES,
    CALIBRATION_METHOD_CONTRACT_ID,
    CALIBRATION_TRAINING_CONTRACT_ID,
    CalibrationAxisSpec,
    CalibrationRolloutBatch,
    FaultAxis,
    FaultAxisCatalog,
)

__all__ = [
    "CALIBRATION_ARTIFACT_SCHEMA",
    "CALIBRATION_AXIS_CATALOG_VERSION",
    "CALIBRATION_AXIS_NAMES",
    "CALIBRATION_METHOD_CONTRACT_ID",
    "CALIBRATION_TRAINING_CONTRACT_ID",
    "CalibratedFADAPolicy",
    "CalibratedPolicyOutput",
    "CalibrationAxisSpec",
    "CalibrationReadout",
    "CalibrationReadoutState",
    "CalibrationRolloutBatch",
    "CoefficientEncoder",
    "DirectionBank",
    "FaultAxis",
    "FaultAxisCatalog",
    "MonotoneScaleCurve",
    "fit_scale_curve_bank",
    "load_calibration_artifact",
    "save_calibration_artifact",
]
