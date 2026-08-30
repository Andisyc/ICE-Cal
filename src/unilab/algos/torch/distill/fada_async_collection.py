"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.async_collection`."""

from .fada.async_collection import *  # noqa: F401,F403
from .fada.async_collection import (
    _collect_cold_start_windows as _collect_cold_start_windows,  # noqa: F401
)
from .fada.async_collection import _concat_source_batches as _concat_source_batches  # noqa: F401
from .fada.async_collection import _summary as _summary  # noqa: F401
