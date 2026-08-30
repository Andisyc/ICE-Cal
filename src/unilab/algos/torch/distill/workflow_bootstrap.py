"""Compatibility facade for :mod:`unilab.algos.torch.distill.workflows.bootstrap`."""

from .workflows.bootstrap import *  # noqa: F401,F403
from .workflows.bootstrap import (
    _load_role_artifact_manifest as _load_role_artifact_manifest,  # noqa: F401
)
from .workflows.bootstrap import (
    _validate_workflow_scenarios as _validate_workflow_scenarios,  # noqa: F401
)
from .workflows.bootstrap import _write_json_atomic as _write_json_atomic  # noqa: F401
