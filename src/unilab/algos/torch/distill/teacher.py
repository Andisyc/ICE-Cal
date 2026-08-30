"""Compatibility facade for :mod:`unilab.algos.torch.distill.learning.teacher`."""

from .learning.teacher import *  # noqa: F401,F403
from .learning.teacher import (
    _load_optional_obs_normalizer as _load_optional_obs_normalizer,  # noqa: F401
)
