"""Compatibility facade for :mod:`unilab.algos.torch.distill.learning.diagnostics`."""

from .learning.diagnostics import *  # noqa: F401,F403
from .learning.diagnostics import (
    _DISTILL_RUNTIME_TRACE_INTERVAL as _DISTILL_RUNTIME_TRACE_INTERVAL,  # noqa: F401
)
from .learning.diagnostics import _ORIGINAL_INT as _ORIGINAL_INT  # noqa: F401
from .learning.diagnostics import _ORIGINAL_REPR as _ORIGINAL_REPR  # noqa: F401
from .learning.diagnostics import _ORIGINAL_TORCH_TENSOR as _ORIGINAL_TORCH_TENSOR  # noqa: F401
from .learning.diagnostics import _ORIGINAL_TYPE as _ORIGINAL_TYPE  # noqa: F401
from .learning.diagnostics import (
    _distill_runtime_debug_enabled as _distill_runtime_debug_enabled,  # noqa: F401
)
from .learning.diagnostics import _emit_trainer_runtime as _emit_trainer_runtime  # noqa: F401
from .learning.diagnostics import _label_counts as _label_counts  # noqa: F401
from .learning.diagnostics import (
    _runtime_identity_snapshot as _runtime_identity_snapshot,  # noqa: F401
)
from .learning.diagnostics import _runtime_trace_update as _runtime_trace_update  # noqa: F401
from .learning.diagnostics import _safe_runtime_repr as _safe_runtime_repr  # noqa: F401
from .learning.diagnostics import (
    _target_index_list_runtime_snapshot as _target_index_list_runtime_snapshot,  # noqa: F401
)
from .learning.diagnostics import _tensor_runtime_snapshot as _tensor_runtime_snapshot  # noqa: F401
from .observability.debug import (
    _DISTILL_RUNTIME_DEBUG_ENV as _DISTILL_RUNTIME_DEBUG_ENV,  # noqa: F401
)
from .observability.debug import (
    _DISTILL_RUNTIME_DEBUG_FALSE_VALUES as _DISTILL_RUNTIME_DEBUG_FALSE_VALUES,  # noqa: F401
)
