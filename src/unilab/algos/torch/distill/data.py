from __future__ import annotations

import os

import torch

from unilab.algos.torch.distill.contracts.dataset import _validate_command_intents
from unilab.algos.torch.distill.datasets.dataset import (
    DistillationTensorDataset,
    annotate_distillation_dataset_scenario,
    build_distillation_dataset,
    make_fake_distillation_dataset,
)
from unilab.algos.torch.distill.datasets.diagnostics import _abort_for_native_capture
from unilab.algos.torch.distill.datasets.io import (
    load_distillation_dataset,
    save_distillation_dataset,
)
from unilab.algos.torch.distill.datasets.merge import build_multitask_distillation_dataset
from unilab.algos.torch.distill.learning.trainer import DistillationBatch

__all__ = [
    "DistillationBatch",
    "DistillationTensorDataset",
    "annotate_distillation_dataset_scenario",
    "build_distillation_dataset",
    "build_multitask_distillation_dataset",
    "load_distillation_dataset",
    "make_fake_distillation_dataset",
    "save_distillation_dataset",
]
