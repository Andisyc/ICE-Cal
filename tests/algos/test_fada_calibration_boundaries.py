from __future__ import annotations

import ast
import importlib
from pathlib import Path

ROOT = Path(__file__).parents[2]
PACKAGE = ROOT / "src" / "unilab" / "algos" / "torch" / "fada_context"

CORE_OWNERS = (
    "calibration_types",
    "calibration_models",
    "calibration_readout",
    "calibration_artifact",
    "calibration_policy",
)
GAIN_OWNERS = (
    "gain_collection_types",
    "gain_collection_provenance",
    "gain_collection_artifact",
    "gain_collection_runtime",
)


def _top_level_definitions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def test_calibration_hotspots_are_facades_over_named_owner_modules() -> None:
    for module_name in (*CORE_OWNERS, *GAIN_OWNERS):
        assert (PACKAGE / f"{module_name}.py").is_file(), module_name
    assert _top_level_definitions(PACKAGE / "calibration.py") == []
    assert _top_level_definitions(PACKAGE / "calibration_collection.py") == []


def test_calibration_facades_preserve_public_identity_and_owner_dependencies() -> None:
    calibration = importlib.import_module("unilab.algos.torch.fada_context.calibration")
    core_expectations = {
        "CalibrationAxisSpec": "calibration_types",
        "CalibrationRolloutBatch": "calibration_types",
        "DirectionBank": "calibration_models",
        "CoefficientEncoder": "calibration_models",
        "MonotoneScaleCurve": "calibration_readout",
        "CalibrationReadoutState": "calibration_readout",
        "save_calibration_artifact": "calibration_artifact",
        "load_calibration_artifact": "calibration_artifact",
        "CalibratedFADAPolicy": "calibration_policy",
    }
    for symbol, owner in core_expectations.items():
        implementation = importlib.import_module(f"unilab.algos.torch.fada_context.{owner}")
        assert getattr(calibration, symbol) is getattr(implementation, symbol)

    collection = importlib.import_module(
        "unilab.algos.torch.fada_context.calibration_collection"
    )
    gain_expectations = {
        "GainCalibrationCollectionProtocol": "gain_collection_types",
        "GainCalibrationRawIdentity": "gain_collection_types",
        "canonicalize_resolved_task_backend_payload": "gain_collection_provenance",
        "load_gain_calibration_protocol": "gain_collection_provenance",
        "build_gain_calibration_raw_artifact": "gain_collection_artifact",
        "load_gain_calibration_raw_rollouts": "gain_collection_artifact",
        "collect_gain_calibration_scenario": "gain_collection_runtime",
        "collect_gain_calibration_rollouts": "gain_collection_runtime",
    }
    for symbol, owner in gain_expectations.items():
        implementation = importlib.import_module(f"unilab.algos.torch.fada_context.{owner}")
        assert getattr(collection, symbol) is getattr(implementation, symbol)

    forbidden_facades = {
        "unilab.algos.torch.fada_context.calibration",
        "unilab.algos.torch.fada_context.calibration_collection",
    }
    for module_name in (*CORE_OWNERS, *GAIN_OWNERS):
        tree = ast.parse((PACKAGE / f"{module_name}.py").read_text())
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert imported.isdisjoint(forbidden_facades), module_name
