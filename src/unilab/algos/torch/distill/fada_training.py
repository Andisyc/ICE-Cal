"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.training`."""

from .fada.training import *  # noqa: F401,F403
from .fada.training import _batch_to_device as _batch_to_device  # noqa: F401
from .fada.training import _load_architecture_config as _load_architecture_config  # noqa: F401
