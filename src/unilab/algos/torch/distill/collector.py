"""Compatibility surface for distillation collection owners."""

from unilab.algos.torch.distill.collection.common import (
    command_active_mask,
    project_student_obs,
    project_teacher_obs,
)
from unilab.algos.torch.distill.collection.standard import collect_distillation_dataset_from_env
from unilab.algos.torch.distill.collection.transition import (
    collect_transition_distillation_dataset_from_env,
    set_transition_input_rows,
)

__all__ = [
    "collect_distillation_dataset_from_env",
    "collect_transition_distillation_dataset_from_env",
    "command_active_mask",
    "project_student_obs",
    "project_teacher_obs",
    "set_transition_input_rows",
]
