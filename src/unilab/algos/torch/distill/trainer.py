"""Compatibility facade for :mod:`unilab.algos.torch.distill.learning.trainer`."""

from .learning.trainer import *  # noqa: F401,F403
from .learning.trainer import _ORIGINAL_TORCH_TENSOR as _ORIGINAL_TORCH_TENSOR  # noqa: F401
from .learning.trainer import _ORIGINAL_TYPE as _ORIGINAL_TYPE  # noqa: F401
from .learning.trainer import (
    _distill_runtime_debug_enabled as _distill_runtime_debug_enabled,  # noqa: F401
)
from .learning.trainer import _emit_trainer_runtime as _emit_trainer_runtime  # noqa: F401
from .learning.trainer import _label_counts as _label_counts  # noqa: F401
from .learning.trainer import _runtime_trace_update as _runtime_trace_update  # noqa: F401
from .learning.trainer import _safe_runtime_repr as _safe_runtime_repr  # noqa: F401
from .learning.trainer import (
    _target_index_list_runtime_snapshot as _target_index_list_runtime_snapshot,  # noqa: F401
)
from .learning.trainer import _tensor_runtime_snapshot as _tensor_runtime_snapshot  # noqa: F401
from .learning.trainer import _TrainerForwardPass as _TrainerForwardPass  # noqa: F401
