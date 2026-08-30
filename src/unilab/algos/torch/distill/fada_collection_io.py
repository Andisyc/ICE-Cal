"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.collection_io`."""

from .fada.collection_io import *  # noqa: F401,F403
from .fada.collection_io import _command_array as _command_array  # noqa: F401
from .fada.collection_io import _done_mask as _done_mask  # noqa: F401
from .fada.collection_io import _fada_actions as _fada_actions  # noqa: F401
from .fada.collection_io import _module_device as _module_device  # noqa: F401
from .fada.collection_io import _next_after_done as _next_after_done  # noqa: F401
from .fada.collection_io import _obs_array as _obs_array  # noqa: F401
from .fada.collection_io import _oracle_actions as _oracle_actions  # noqa: F401
from .fada.collection_io import _oracle_shadow_pair as _oracle_shadow_pair  # noqa: F401
from .fada.collection_io import _policy_actions as _policy_actions  # noqa: F401
