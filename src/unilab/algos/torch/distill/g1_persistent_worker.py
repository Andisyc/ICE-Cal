"""Compatibility facade for :mod:`unilab.algos.torch.distill.runtime.g1_worker`."""

from .runtime.g1_worker import *  # noqa: F401,F403
from .runtime.g1_worker import (
    _build_persistent_g1_worker as _build_persistent_g1_worker,  # noqa: F401
)
from .runtime.g1_worker import _teacher_spec as _teacher_spec  # noqa: F401
from .runtime.g1_worker import _teacher_spec_fingerprint as _teacher_spec_fingerprint  # noqa: F401
from .runtime.g1_worker import _TeacherResource as _TeacherResource  # noqa: F401
