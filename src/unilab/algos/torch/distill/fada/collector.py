from __future__ import annotations

from unilab.algos.torch.distill.fada.collection_contract import (
    FADACollectionResult,
    FADACollectionSpec,
    FADACollectionTransition,
)
from unilab.algos.torch.distill.fada.collection_io import (
    _command_array,
    _done_mask,
    _fada_actions,
    _module_device,
    _next_after_done,
    _obs_array,
    _oracle_shadow_pair,
    _policy_actions,
)
from unilab.algos.torch.distill.fada.collection_transaction import collect_fada_source_windows
from unilab.algos.torch.distill.fada.collection_windows import (
    _cold_start_window_from_records,
    _concat_batches,
    _terminal_planner_window,
    _walking_recovery_window,
    _window_from_records,
)

# Compatibility alias for callers that imported the former module-private type.
_Transition = FADACollectionTransition

__all__ = [
    "FADACollectionResult",
    "FADACollectionSpec",
    "collect_fada_source_windows",
]
