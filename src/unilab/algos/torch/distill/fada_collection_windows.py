"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.collection_windows`."""

from .fada.collection_windows import *  # noqa: F401,F403
from .fada.collection_windows import (
    _cold_start_window_from_records as _cold_start_window_from_records,  # noqa: F401
)
from .fada.collection_windows import _concat_batches as _concat_batches  # noqa: F401
from .fada.collection_windows import (
    _terminal_planner_window as _terminal_planner_window,  # noqa: F401
)
from .fada.collection_windows import (
    _walking_recovery_window as _walking_recovery_window,  # noqa: F401
)
from .fada.collection_windows import _window_from_records as _window_from_records  # noqa: F401
