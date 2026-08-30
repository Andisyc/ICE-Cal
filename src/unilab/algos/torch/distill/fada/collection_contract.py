from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from unilab.algos.torch.distill.fada.model import FADASourceBatch
from unilab.algos.torch.distill.fada.windows import FADACausalTransition


@dataclass(frozen=True)
class FADACollectionResult:
    batch: FADASourceBatch
    env_steps: int
    rejected_done_transitions: int
    rejected_command_windows: int
    rollout_mode: str
    command_scenario: str = "walk"
    oracle_role: str = "walking"
    rejected_scenario_windows: int = 0
    window_profile: str = "steady_state"


@dataclass(frozen=True)
class FADACollectionSpec:
    """Validated collection semantics that travel together across FADA callers."""

    observation_key: str = "obs"
    teacher_projection: str = "identity"
    student_projection: str = "identity"
    student_drop_index: int | None = None
    command_info_keys: tuple[str, ...] = ("commands",)
    max_env_steps: int | None = None
    collect_oracle_shadow: bool = False
    command_scenario: Literal["walk", "static_stand", "walk_to_stand"] = "walk"
    transition_walk_command: tuple[float, ...] = (0.4, 0.0, 0.0)
    transition_pre_switch_steps: int | None = None
    transition_post_switch_steps: int | None = None
    planner_eligible: bool = True
    cold_start_windows: bool = False


@dataclass(frozen=True)
class FADACollectionTransition(FADACausalTransition):
    oracle_action: np.ndarray
    oracle_future: np.ndarray
    oracle_action_chunk: np.ndarray
    oracle_shadow_valid: bool
