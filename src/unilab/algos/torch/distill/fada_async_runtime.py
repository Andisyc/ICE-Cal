"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.async_runtime`."""

from .fada.async_collection import collect_fada_iteration  # noqa: F401
from .fada.async_runtime import *  # noqa: F401,F403
from .fada.async_runtime import (
    _build_persistent_fada_worker as _build_persistent_fada_worker,  # noqa: F401
)
from .fada.async_runtime import (
    _curriculum_and_allocations as _curriculum_and_allocations,  # noqa: F401
)
from .fada.async_runtime import _fada_runtime_device as _fada_runtime_device  # noqa: F401
from .fada.async_runtime import (
    _stand_transition_curriculum_cfg as _stand_transition_curriculum_cfg,  # noqa: F401
)
from .fada.async_runtime import _standing_owner_cfg as _standing_owner_cfg  # noqa: F401
from .fada.async_runtime import _teacher_spec as _teacher_spec  # noqa: F401
from .fada.async_runtime import _v005_replay_cfg as _v005_replay_cfg  # noqa: F401
