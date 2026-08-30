"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.checkpoint`."""

from .fada.checkpoint import *  # noqa: F401,F403
from .fada.checkpoint import (
    _canonical_state_dict_sha256 as _canonical_state_dict_sha256,  # noqa: F401
)
from .fada.checkpoint import (
    _validate_schema5_training_state as _validate_schema5_training_state,  # noqa: F401
)
from .fada.checkpoint import _validated_idm_state as _validated_idm_state  # noqa: F401
