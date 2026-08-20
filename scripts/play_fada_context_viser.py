"""Launch healthy, fault-zero, or fault-Context FADA playback in Viser."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from play_interactive import _build_play_args, play_interactive  # noqa: E402

from unilab.algos.torch.distill import load_fada_policy_checkpoint  # noqa: E402
from unilab.algos.torch.distill.fada_playback import FADAPlaybackController  # noqa: E402
from unilab.algos.torch.fada_context.support_query import (  # noqa: E402
    SupportBoundContextPolicy,
    SupportQueryContextConfig,
)
from unilab.algos.torch.fada_context.support_query_training import (  # noqa: E402
    prepare_context_support_query_artifact,
)
from unilab.visualization.interactive_playback import (  # noqa: E402
    create_fada_playback_session,
)


def _context_controller(cfg: DictConfig, *, device: str) -> FADAPlaybackController:
    healthy_path = (ROOT_DIR / str(cfg.context_playback.healthy_checkpoint)).resolve()
    context_path = (ROOT_DIR / str(cfg.context_playback.context_checkpoint)).resolve()
    dataset_path = (ROOT_DIR / str(cfg.context_playback.dataset)).resolve()
    loaded = load_fada_policy_checkpoint(healthy_path, device=device)
    support_length = int(cfg.context_playback.support_length)
    prepared = prepare_context_support_query_artifact(
        loaded.policy,
        SupportQueryContextConfig(
            support_length=support_length,
            context_hidden_dim=int(cfg.context_playback.hidden_dim),
            context_layers=int(cfg.context_playback.num_layers),
            delta_scale=float(cfg.context_playback.delta_scale),
        ),
        source_checkpoint_path=healthy_path,
        dataset_path=dataset_path,
        context_checkpoint_path=context_path,
        support_length=support_length,
        query_length=int(
            OmegaConf.select(cfg, "context_playback.query_length", default=support_length)
        ),
        validation_fraction=float(cfg.context_playback.validation_fraction),
        split_seed=int(cfg.context_playback.split_seed),
    )
    validation = prepared.validation
    support_index = int(cfg.context_playback.support_index)
    if not 0 <= support_index < validation.batch_size:
        raise ValueError(
            f"support_index must be in [0, {validation.batch_size}), got {support_index}"
        )
    policy = prepared.policy
    index = torch.tensor([support_index], dtype=torch.int64)
    support = validation.support.index_select(index).to(device)
    support_command = validation.support_command.index_select(0, index).to(device)
    bound_policy = SupportBoundContextPolicy(
        policy,
        support,
        support_command,
    ).eval()

    print(
        "[play_fada_context_viser] Context enabled: "
        f"validation pair_id={int(validation.pair_id[support_index])}, "
        "delta_z=recomputed_per_control_cycle"
    )
    return FADAPlaybackController(bound_policy, device=device)


def _session_factory(**kwargs: Any) -> Any:
    cfg = kwargs["cfg"]
    context_controller = None
    if bool(OmegaConf.select(cfg, "context_playback.enabled", default=False)):
        context_controller = _context_controller(cfg, device=str(kwargs["device"]))
    session, policy_obs_mode, checkpoint_path = create_fada_playback_session(**kwargs)
    if context_controller is not None:
        session.bind_controller(context_controller)
    return session, policy_obs_mode, checkpoint_path


@hydra.main(version_base="1.3", config_path="../conf/distill", config_name="config")
def main(cfg: DictConfig) -> None:
    if str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("Context playback only supports MuJoCo")
    play_interactive(
        _build_play_args(cfg, algo="fada"),
        cfg,
        algo="fada",
        fada_session_factory=_session_factory,
    )


if __name__ == "__main__":
    main()
