"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.adaptation`."""

from .fada.adaptation import *  # noqa: F401,F403
from .fada.adaptation import _batch_to_device as _batch_to_device  # noqa: F401
from .fada.adaptation import _discover_lora_targets as _discover_lora_targets  # noqa: F401
from .fada.adaptation import (
    _inject_fada_idm_legacy_linear_lora as _inject_fada_idm_legacy_linear_lora,  # noqa: F401
)
from .fada.adaptation import _optimizer_parameter_ids as _optimizer_parameter_ids  # noqa: F401
from .fada.adaptation import _replace_submodule as _replace_submodule  # noqa: F401
