"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.trainer`."""

from .fada.trainer import *  # noqa: F401,F403
from .fada.trainer import _grad_norm as _grad_norm  # noqa: F401
