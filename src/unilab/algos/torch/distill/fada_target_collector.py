"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.target_collector`."""

from .fada.target_collector import *  # noqa: F401,F403
from .fada.target_collector import _command_array as _command_array  # noqa: F401
from .fada.target_collector import _done_mask as _done_mask  # noqa: F401
from .fada.target_collector import _module_device as _module_device  # noqa: F401
from .fada.target_collector import _observation_array as _observation_array  # noqa: F401
from .fada.target_collector import (
    concat_fada_target_batches as _concat_target_batches,  # noqa: F401
)
from .fada.target_collector import (
    fada_target_batch_from_window as _target_batch_from_window,  # noqa: F401
)
