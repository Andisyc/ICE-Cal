"""Compatibility facade for :mod:`unilab.algos.torch.distill.collection.common`."""

from .collection.common import *  # noqa: F401,F403
from .collection.common import (
    _attach_collector_performance as _attach_collector_performance,  # noqa: F401
)
from .collection.common import _command_sample_mask as _command_sample_mask  # noqa: F401
from .collection.common import _info_array as _info_array  # noqa: F401
from .collection.common import _module_device as _module_device  # noqa: F401
from .collection.common import _obs_array as _obs_array  # noqa: F401
from .collection.common import _performance_span as _performance_span  # noqa: F401
from .collection.common import _policy_actions as _policy_actions  # noqa: F401
from .collection.common import (
    _reset_done_rows_after_step as _reset_done_rows_after_step,  # noqa: F401
)
from .collection.common import _resolve_collection_reset as _resolve_collection_reset  # noqa: F401
from .collection.common import _state_done_mask as _state_done_mask  # noqa: F401
from .collection.common import (
    _state_has_autoreset_final_observation as _state_has_autoreset_final_observation,  # noqa: F401
)
from .collection.common import _target_height_array as _target_height_array  # noqa: F401
