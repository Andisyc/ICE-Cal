"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.source_artifact`."""

from .fada.source_artifact import *  # noqa: F401,F403
from .fada.source_artifact import _batch_to_device as _batch_to_device  # noqa: F401
from .fada.source_artifact import (
    _load_architecture_config as _load_architecture_config,  # noqa: F401
)
