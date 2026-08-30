"""Compatibility facade for :mod:`unilab.algos.torch.distill.collection.transition`."""

from .collection.transition import *  # noqa: F401,F403
from .collection.transition import (
    _advance_transition_step as _advance_transition_step,  # noqa: F401
)
from .collection.transition import (
    _attach_collector_performance as _attach_collector_performance,  # noqa: F401
)
from .collection.transition import (
    _build_transition_case_assignment as _build_transition_case_assignment,  # noqa: F401
)
from .collection.transition import (
    _build_transition_case_metadata as _build_transition_case_metadata,  # noqa: F401
)
from .collection.transition import (
    _collect_transition_rows as _collect_transition_rows,  # noqa: F401
)
from .collection.transition import (
    _finalize_transition_collection as _finalize_transition_collection,  # noqa: F401
)
from .collection.transition import _info_array as _info_array  # noqa: F401
from .collection.transition import _label_transition_step as _label_transition_step  # noqa: F401
from .collection.transition import _module_device as _module_device  # noqa: F401
from .collection.transition import _obs_array as _obs_array  # noqa: F401
from .collection.transition import _performance_span as _performance_span  # noqa: F401
from .collection.transition import _policy_actions as _policy_actions  # noqa: F401
from .collection.transition import (
    _prepare_transition_collection as _prepare_transition_collection,  # noqa: F401
)
from .collection.transition import (
    _PreparedTransitionCollection as _PreparedTransitionCollection,  # noqa: F401
)
from .collection.transition import (
    _reset_done_rows_after_step as _reset_done_rows_after_step,  # noqa: F401
)
from .collection.transition import (
    _resolve_collection_reset as _resolve_collection_reset,  # noqa: F401
)
from .collection.transition import _state_done_mask as _state_done_mask  # noqa: F401
from .collection.transition import _target_height_array as _target_height_array  # noqa: F401
from .collection.transition import (
    _TransitionCollectionOutcome as _TransitionCollectionOutcome,  # noqa: F401
)
from .collection.transition import (
    _TransitionCollectionState as _TransitionCollectionState,  # noqa: F401
)
from .collection.transition import _TransitionStepLabels as _TransitionStepLabels  # noqa: F401
from .collection.transition import (
    _validate_transition_coverage as _validate_transition_coverage,  # noqa: F401
)
