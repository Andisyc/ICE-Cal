"""Compatibility facade for :mod:`unilab.algos.torch.distill.workflows.diagnostics`."""

from .workflows.diagnostics import *  # noqa: F401,F403
from .workflows.diagnostics import (
    _probe_torch_serialization_runtime as _probe_torch_serialization_runtime,  # noqa: F401
)
