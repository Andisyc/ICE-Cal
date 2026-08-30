"""Compatibility facade for :mod:`unilab.algos.torch.distill.workflows.formal_identity`."""

from .workflows.formal_identity import *  # noqa: F401,F403
from .workflows.formal_identity import (
    _FORMAL_RUN_NAME_PATTERN as _FORMAL_RUN_NAME_PATTERN,  # noqa: F401
)
from .workflows.formal_identity import _file_identity as _file_identity  # noqa: F401
from .workflows.formal_identity import _identity_paths as _identity_paths  # noqa: F401
from .workflows.formal_identity import _reject_sentinel_path as _reject_sentinel_path  # noqa: F401
from .workflows.formal_identity import _validate_spec as _validate_spec  # noqa: F401
