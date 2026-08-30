"""Compatibility facade for :mod:`unilab.algos.torch.distill.runtime.async_runtime`."""

from .runtime.async_runtime import *  # noqa: F401,F403
from .runtime.async_runtime import _SPAWN_CTX as _SPAWN_CTX  # noqa: F401
from .runtime.async_runtime import (
    _persistent_dagger_collector_entry as _persistent_dagger_collector_entry,  # noqa: F401
)
