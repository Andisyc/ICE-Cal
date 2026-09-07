"""Target adaptation boundary."""

from ..target_adaptation_workflow import (
    FADAAdaptationPreflight,
    preflight_fada_adaptation,
    run_fada_adaptation,
    train_fada_adaptation,
)

run_fada_target_adaptation = run_fada_adaptation

__all__ = [
    "FADAAdaptationPreflight",
    "preflight_fada_adaptation",
    "run_fada_adaptation",
    "run_fada_target_adaptation",
    "train_fada_adaptation",
]
