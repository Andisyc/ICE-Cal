"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.artifact_admission`."""

from .fada.artifact_admission import *  # noqa: F401,F403
from .fada.artifact_admission import _fada_quality_batch as _fada_quality_batch  # noqa: F401
from .fada.artifact_admission import (
    _fada_v005_replay_settings as _fada_v005_replay_settings,  # noqa: F401
)
from .fada.artifact_admission import (
    _require_fada_curriculum_artifact as _require_fada_curriculum_artifact,  # noqa: F401
)
from .fada.artifact_admission import _slice_fada_batch as _slice_fada_batch  # noqa: F401
