"""Compatibility facade for :mod:`unilab.algos.torch.distill.contracts.dataset`."""

from .contracts.dataset import *  # noqa: F401,F403
from .contracts.dataset import _ORIGINAL_TYPE as _ORIGINAL_TYPE  # noqa: F401
from .contracts.dataset import _TRANSITION_SCENARIOS as _TRANSITION_SCENARIOS  # noqa: F401
from .contracts.dataset import _abort_for_native_capture as _abort_for_native_capture  # noqa: F401
from .contracts.dataset import (
    _command_intents_from_commands as _command_intents_from_commands,  # noqa: F401
)
from .contracts.dataset import (
    _command_intents_from_role_labels as _command_intents_from_role_labels,  # noqa: F401
)
from .contracts.dataset import _emit_data_runtime as _emit_data_runtime  # noqa: F401
from .contracts.dataset import _safe_runtime_repr as _safe_runtime_repr  # noqa: F401
from .contracts.dataset import (
    _scenario_label_debug_snapshot as _scenario_label_debug_snapshot,  # noqa: F401
)
from .contracts.dataset import _validate_action_tensor as _validate_action_tensor  # noqa: F401
from .contracts.dataset import _validate_command_intents as _validate_command_intents  # noqa: F401
from .contracts.dataset import _validate_command_tensor as _validate_command_tensor  # noqa: F401
from .contracts.dataset import _validate_commands as _validate_commands  # noqa: F401
from .contracts.dataset import _validate_obs_tensor as _validate_obs_tensor  # noqa: F401
from .contracts.dataset import _validate_role_labels as _validate_role_labels  # noqa: F401
from .contracts.dataset import _validate_scenario_labels as _validate_scenario_labels  # noqa: F401
from .contracts.dataset import _validate_target_height as _validate_target_height  # noqa: F401
from .contracts.dataset import _validate_transition_ages as _validate_transition_ages  # noqa: F401
from .contracts.dataset import (
    _validate_transition_fields as _validate_transition_fields,  # noqa: F401
)
