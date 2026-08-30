"""Compatibility facade for :mod:`unilab.algos.torch.distill.datasets.io`."""

from .datasets.io import *  # noqa: F401,F403
from .datasets.io import _abort_for_native_capture as _abort_for_native_capture  # noqa: F401
from .datasets.io import _emit_data_runtime as _emit_data_runtime  # noqa: F401
from .datasets.io import (
    _native_abort_for_impossible_callable_error_requested as _native_abort_for_impossible_callable_error_requested,  # noqa: F401
)
