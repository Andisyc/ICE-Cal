"""Compatibility facade for :mod:`unilab.algos.torch.distill.learning.dagger`."""

from .learning.dagger import *  # noqa: F401,F403
from .learning.dagger import _aggregate_dagger_datasets as _aggregate_dagger_datasets  # noqa: F401
from .learning.dagger import _attach_role_label as _attach_role_label  # noqa: F401
from .learning.dagger import _FixedExpertRolloutPolicy as _FixedExpertRolloutPolicy  # noqa: F401
from .learning.dagger import _module_device as _module_device  # noqa: F401
from .learning.dagger import (
    _resolve_dagger_rollout_policy as _resolve_dagger_rollout_policy,  # noqa: F401
)
