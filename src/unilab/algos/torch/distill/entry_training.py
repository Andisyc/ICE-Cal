"""Compatibility facade for :mod:`unilab.algos.torch.distill.workflows.entry_training`."""

from .workflows.entry_training import *  # noqa: F401,F403
from .workflows.entry_training import (
    _ROLE_DATA_ASSEMBLY_DEVICE as _ROLE_DATA_ASSEMBLY_DEVICE,  # noqa: F401
)
from .workflows.entry_training import _distill_runtime_cfg as _distill_runtime_cfg  # noqa: F401
from .workflows.entry_training import (
    _expected_samples_seen_for_offline_run as _expected_samples_seen_for_offline_run,  # noqa: F401
)
from .workflows.entry_training import _int_tuple as _int_tuple  # noqa: F401
from .workflows.entry_training import (
    _load_student_init_checkpoint as _load_student_init_checkpoint,  # noqa: F401
)
from .workflows.entry_training import _multitask_sources as _multitask_sources  # noqa: F401
from .workflows.entry_training import (
    _normalize_checkpoint_selector as _normalize_checkpoint_selector,  # noqa: F401
)
from .workflows.entry_training import _optional_int_cfg as _optional_int_cfg  # noqa: F401
from .workflows.entry_training import _probe_result as _probe_result  # noqa: F401
from .workflows.entry_training import (
    _resolve_formal_run_dir as _resolve_formal_run_dir,  # noqa: F401
)
from .workflows.entry_training import (
    _resolve_optional_checkpoint_path as _resolve_optional_checkpoint_path,  # noqa: F401
)
from .workflows.entry_training import (
    _runtime_cfg_subset_for_student as _runtime_cfg_subset_for_student,  # noqa: F401
)
from .workflows.entry_training import _student_model_type as _student_model_type  # noqa: F401
from .workflows.entry_training import _student_runtime_cfg as _student_runtime_cfg  # noqa: F401
from .workflows.entry_training import _teacher_metadata as _teacher_metadata  # noqa: F401
from .workflows.entry_training import (
    _validate_student_init_runtime_cfg as _validate_student_init_runtime_cfg,  # noqa: F401
)
