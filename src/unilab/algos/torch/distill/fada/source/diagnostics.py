"""Source artifact diagnostic boundary."""

from ..async_config import teacher_spec
from ..collection_contract import FADACollectionSpec
from ..oracle import load_fada_oracle_policy
from ..source_diagnostics import run_fada_coverage_diagnostic

__all__ = [
    "FADACollectionSpec",
    "load_fada_oracle_policy",
    "run_fada_coverage_diagnostic",
    "teacher_spec",
]
