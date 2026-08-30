"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.collector`."""

from .fada.collection_contract import (  # noqa: F401
    FADACollectionResult,
    FADACollectionSpec,
    FADACollectionTransition,
)
from .fada.collection_transaction import collect_fada_source_windows  # noqa: F401
from .fada.collector import *  # noqa: F401,F403
from .fada.collector import (
    _cold_start_window_from_records as _cold_start_window_from_records,  # noqa: F401
)
from .fada.collector import _command_array as _command_array  # noqa: F401
from .fada.collector import _concat_batches as _concat_batches  # noqa: F401
from .fada.collector import _done_mask as _done_mask  # noqa: F401
from .fada.collector import _fada_actions as _fada_actions  # noqa: F401
from .fada.collector import _module_device as _module_device  # noqa: F401
from .fada.collector import _next_after_done as _next_after_done  # noqa: F401
from .fada.collector import _obs_array as _obs_array  # noqa: F401
from .fada.collector import _oracle_shadow_pair as _oracle_shadow_pair  # noqa: F401
from .fada.collector import _policy_actions as _policy_actions  # noqa: F401
from .fada.collector import _terminal_planner_window as _terminal_planner_window  # noqa: F401
from .fada.collector import _Transition as _Transition  # noqa: F401
from .fada.collector import _walking_recovery_window as _walking_recovery_window  # noqa: F401
from .fada.collector import _window_from_records as _window_from_records  # noqa: F401
