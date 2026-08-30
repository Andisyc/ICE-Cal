"""Compatibility facade for :mod:`unilab.algos.torch.distill.workflows.entry_workflow`."""

from .workflows.entry_workflow import *  # noqa: F401,F403
from .workflows.entry_workflow import _distill_device as _distill_device  # noqa: F401
from .workflows.entry_workflow import (
    _probe_torch_serialization_runtime as _probe_torch_serialization_runtime,  # noqa: F401
)
from .workflows.entry_workflow import _workflow_entry_result as _workflow_entry_result  # noqa: F401
from .workflows.entry_workflow import _workflow_path as _workflow_path  # noqa: F401
from .workflows.entry_workflow import _workflow_role_cfg as _workflow_role_cfg  # noqa: F401
from .workflows.entry_workflow import _workflow_role_entries as _workflow_role_entries  # noqa: F401
from .workflows.entry_workflow import (
    _workflow_scenario_specs as _workflow_scenario_specs,  # noqa: F401
)
