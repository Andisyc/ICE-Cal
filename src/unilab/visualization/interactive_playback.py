"""Stable compatibility surface for interactive playback owners."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .playback_distill_routing import distill_command_intents_from_commands
from .playback_distill_sessions import (
    _apply_distill_playback_reset_contract,
    _apply_keyboard_playback_reset_contract,
    _default_distill_playback_deps,
    _default_fada_playback_deps,
    _default_hora_distill_playback_deps,
    create_distill_playback_session,
    create_fada_playback_session,
    create_hora_distill_playback_session,
)
from .playback_overlay import prepare_motion_overlay_selection
from .playback_policy_sessions import (
    _LEGACY_TAR_WEIGHTS_ONLY_ERROR,
    _PRIVILEGED_CHECKPOINT_SCHEMAS,
    _actor_input_dim_from_state_dict,
    _build_appo_actor,
    _cfg_checkpoint_value,
    _ensure_scripts_dir,
    _load_playback_checkpoint,
    _normalize_checkpoint_value,
    _offpolicy_checkpoint_actor_input_dim,
    _resolve_appo_checkpoint_from_cfg,
    _resolve_task_checkpoint_from_playback_cfg,
    create_appo_playback_session,
    create_rsl_rl_playback_session,
    create_sac_playback_session,
    select_torch_device,
)
from .playback_sessions import (
    _HORA_DISTILL_CHECKPOINT_UNAVAILABLE,
    FADAPlaybackSession,
    HeightCommander,
    KeyboardCommander,
    MotionOverlaySelection,
    OffPolicyPlaybackSession,
    PlaybackControls,
    PlaybackSession,
    RslRlPlaybackConfig,
    RslRlPlaybackSession,
    _external_velocity_command_rows,
)

LogFn = Callable[[str], None]

__all__ = [
    "FADAPlaybackSession",
    "HeightCommander",
    "KeyboardCommander",
    "MotionOverlaySelection",
    "OffPolicyPlaybackSession",
    "PlaybackControls",
    "PlaybackSession",
    "RslRlPlaybackConfig",
    "RslRlPlaybackSession",
    "create_appo_playback_session",
    "create_distill_playback_session",
    "create_fada_playback_session",
    "create_hora_distill_playback_session",
    "create_rsl_rl_playback_session",
    "create_sac_playback_session",
    "distill_command_intents_from_commands",
    "prepare_motion_overlay_selection",
    "select_torch_device",
]
