"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.workflow_setup`."""

from .fada.workflow_setup import *  # noqa: F401,F403
from .fada.workflow_setup import (
    _default_load_fada_oracle_policy as _default_load_fada_oracle_policy,  # noqa: F401
)
from .fada.workflow_setup import _distill_device as _distill_device  # noqa: F401
from .fada.workflow_setup import _fada_execution_mode as _fada_execution_mode  # noqa: F401
from .fada.workflow_setup import _fada_path as _fada_path  # noqa: F401
from .fada.workflow_setup import (
    _fada_v005_replay_settings as _fada_v005_replay_settings,  # noqa: F401
)
from .fada.workflow_setup import _paper_source_plan as _paper_source_plan  # noqa: F401
