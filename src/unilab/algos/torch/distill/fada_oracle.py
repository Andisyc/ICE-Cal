"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.oracle`."""

from .fada.oracle import *  # noqa: F401,F403
from .fada.oracle import _FADA_ORACLE_COMMAND_LIMITS as _FADA_ORACLE_COMMAND_LIMITS  # noqa: F401
from .fada.oracle import (
    _is_distillation_student_checkpoint as _is_distillation_student_checkpoint,  # noqa: F401
)
from .fada.oracle import (
    _load_privileged_oracle_policy as _load_privileged_oracle_policy,  # noqa: F401
)
from .fada.oracle import (
    _privileged_checkpoint_metadata as _privileged_checkpoint_metadata,  # noqa: F401
)
