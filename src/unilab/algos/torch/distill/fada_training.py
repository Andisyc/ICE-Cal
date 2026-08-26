"""Compatibility facade for FADA training owner modules."""

from __future__ import annotations

import torch

from .fada import (
    FADA_COMMAND_SCENARIOS,
    FADA_SCENARIO_IDS,
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlanner,
    FADAPlannerIDMPolicy,
    FADASourceBatch,
    idm_source_loss,
    planner_source_loss,
)
from .fada_checkpoint import (
    FADA_CHECKPOINT_SCHEMA_VERSION,
    FADA_TRAINING_SCHEDULE,
    FADA_V005_REQUIRED_QUALITY_METRICS,
    LoadedFADAPlannerIDMPolicy,
    load_fada_checkpoint,
    load_fada_policy_checkpoint,
    save_fada_checkpoint,
)
from .fada_oracle import load_fada_oracle_policy
from .fada_replay import FADAReplayBuffer, FADAReplayRoleCounts
from .fada_source_artifact import (
    FADA_SOURCE_BATCH_SCHEMA_VERSION,
    LoadedFADASourceBatch,
    batch_to_device,
    load_architecture_config,
    load_fada_source_batch,
    save_fada_source_batch,
)
from .fada_source_evaluation import evaluate_fada_source_batch
from .fada_source_plan import (
    FADA_INTERMEDIATE_ORACLE_COUNT,
    FADAPaperSourcePlan,
    build_fada_paper_source_plan,
)
from .fada_trainer import FADATrainer, FADATrainingStats

_batch_to_device = batch_to_device
_load_architecture_config = load_architecture_config

__all__ = [
    "FADA_CHECKPOINT_SCHEMA_VERSION",
    "FADA_TRAINING_SCHEDULE",
    "FADA_COMMAND_SCENARIOS",
    "FADA_INTERMEDIATE_ORACLE_COUNT",
    "FADA_SCENARIO_IDS",
    "FADA_SOURCE_BATCH_SCHEMA_VERSION",
    "FADA_V005_REQUIRED_QUALITY_METRICS",
    "FADAArchitectureConfig",
    "FADAInverseDynamicsModel",
    "FADAPaperSourcePlan",
    "FADAPlanner",
    "FADAPlannerIDMPolicy",
    "FADAReplayBuffer",
    "FADAReplayRoleCounts",
    "FADASourceBatch",
    "FADATrainer",
    "FADATrainingStats",
    "LoadedFADAPlannerIDMPolicy",
    "LoadedFADASourceBatch",
    "build_fada_paper_source_plan",
    "evaluate_fada_source_batch",
    "idm_source_loss",
    "load_fada_checkpoint",
    "load_fada_oracle_policy",
    "load_fada_policy_checkpoint",
    "load_fada_source_batch",
    "planner_source_loss",
    "save_fada_checkpoint",
    "save_fada_source_batch",
]
