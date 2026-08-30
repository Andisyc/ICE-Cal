"""Compatibility facade for :mod:`unilab.algos.torch.distill.collection.transition_state`."""

from .collection.transition_state import *  # noqa: F401,F403
from .collection.transition_state import (
    _build_transition_case_assignment as _build_transition_case_assignment,  # noqa: F401
)
from .collection.transition_state import (
    _build_transition_case_metadata as _build_transition_case_metadata,  # noqa: F401
)
from .collection.transition_state import (
    _TransitionCaseAssignment as _TransitionCaseAssignment,  # noqa: F401
)
from .collection.transition_state import (
    _validate_transition_coverage as _validate_transition_coverage,  # noqa: F401
)
