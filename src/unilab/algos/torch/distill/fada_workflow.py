"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.workflow`."""

from .fada.workflow import *  # noqa: F401,F403
from .fada.workflow import _distill_device as _distill_device  # noqa: F401
from .fada.workflow import _fada_execution_mode as _fada_execution_mode  # noqa: F401
from .fada.workflow import _fada_path as _fada_path  # noqa: F401
from .fada.workflow import _fada_quality_batch as _fada_quality_batch  # noqa: F401
from .fada.workflow import _fada_v005_replay_settings as _fada_v005_replay_settings  # noqa: F401
from .fada.workflow import _paper_source_plan as _paper_source_plan  # noqa: F401
from .fada.workflow import (
    _require_fada_curriculum_artifact as _require_fada_curriculum_artifact,  # noqa: F401
)
from .fada.workflow import _run_fada_legacy as _run_fada_legacy  # noqa: F401
from .fada.workflow import _run_fada_persistent_async as _run_fada_persistent_async  # noqa: F401
from .fada.workflow import _slice_fada_batch as _slice_fada_batch  # noqa: F401
