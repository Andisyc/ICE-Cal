"""Compatibility facade for gain calibration collection owners."""

import torch

from unilab.algos.torch.fada_context.gain_collection_artifact import (
    build_gain_calibration_raw_artifact,
    load_gain_calibration_raw_rollouts,
    save_gain_calibration_raw_rollouts,
    validate_gain_calibration_raw_artifact,
)
from unilab.algos.torch.fada_context.gain_collection_provenance import (
    canonicalize_resolved_task_backend_payload,
    load_gain_calibration_protocol,
    sha256_canonical_mapping,
    sha256_file,
)
from unilab.algos.torch.fada_context.gain_collection_runtime import (
    collect_gain_calibration_rollouts,
    collect_gain_calibration_scenario,
)
from unilab.algos.torch.fada_context.gain_collection_types import (
    _APPROVED_POINTS,
    GAIN_CALIBRATION_RAW_SCHEMA,
    GainCalibrationCollectionProtocol,
    GainCalibrationPoint,
    GainCalibrationRawIdentity,
    GainCalibrationScenarioResult,
    GainCalibrationScenarioSpec,
    GainCalibrationSplit,
)

__all__ = [
    "GAIN_CALIBRATION_RAW_SCHEMA",
    "GainCalibrationCollectionProtocol",
    "GainCalibrationPoint",
    "GainCalibrationRawIdentity",
    "GainCalibrationScenarioResult",
    "GainCalibrationScenarioSpec",
    "GainCalibrationSplit",
    "build_gain_calibration_raw_artifact",
    "canonicalize_resolved_task_backend_payload",
    "collect_gain_calibration_rollouts",
    "collect_gain_calibration_scenario",
    "load_gain_calibration_protocol",
    "load_gain_calibration_raw_rollouts",
    "save_gain_calibration_raw_rollouts",
    "sha256_canonical_mapping",
    "sha256_file",
    "validate_gain_calibration_raw_artifact",
]
