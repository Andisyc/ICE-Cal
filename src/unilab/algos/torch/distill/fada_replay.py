"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.replay`."""

from .fada.replay import *  # noqa: F401,F403
from .fada.replay import _allocate_ratio_counts as _allocate_ratio_counts  # noqa: F401
from .fada.replay import _sample_mask_indices as _sample_mask_indices  # noqa: F401
