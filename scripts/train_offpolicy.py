"""Hydra composition root for off-policy training and playback."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unilab.training.offpolicy import (
    apply_configured_actor_warm_start,
    build_failure_summary,
    build_offpolicy_env_cfg_override,
    build_runner,
    default_device,
    enable_faulthandler,
    extract_play_obs,
    extract_reset_obs,
    play_offpolicy,
    resolve_checkpoint_path,
    resolve_play_actor_spec,
    resolve_play_obs_dim,
    resolve_play_obs_dims,
    run_offpolicy,
)

__all__ = [
    "apply_configured_actor_warm_start",
    "build_failure_summary",
    "build_offpolicy_env_cfg_override",
    "build_runner",
    "default_device",
    "enable_faulthandler",
    "extract_play_obs",
    "extract_reset_obs",
    "play_offpolicy",
    "resolve_checkpoint_path",
    "resolve_play_actor_spec",
    "resolve_play_obs_dim",
    "resolve_play_obs_dims",
]


@hydra.main(version_base="1.3", config_path="../conf/offpolicy", config_name="config")
def main(cfg: DictConfig) -> None:
    run_offpolicy(cfg)


if __name__ == "__main__":
    main()
