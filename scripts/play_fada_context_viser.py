"""Launch healthy, fault-zero, or fault-Context FADA playback in Viser."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

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
    SupportQueryBatch,
    SupportQueryContextConfig,
)
from unilab.algos.torch.fada_context.support_query_data import (  # noqa: E402
    load_support_query_dataset,
)
from unilab.algos.torch.fada_context.support_query_training import (  # noqa: E402
    prepare_support_query_training,
)
from unilab.visualization.interactive_playback import (  # noqa: E402
    create_fada_playback_session,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_validation_split(
    batch: SupportQueryBatch,
    *,
    validation_fraction: float,
    seed: int,
) -> SupportQueryBatch:
    validation_count = max(1, int(round(batch.batch_size * validation_fraction)))
    order = torch.randperm(batch.batch_size, generator=torch.Generator().manual_seed(seed))
    return batch.index_select(order[:validation_count])


def _context_controller(cfg: DictConfig, *, device: str) -> FADAPlaybackController:
    healthy_path = (ROOT_DIR / str(cfg.context_playback.healthy_checkpoint)).resolve()
    context_path = (ROOT_DIR / str(cfg.context_playback.context_checkpoint)).resolve()
    dataset_path = (ROOT_DIR / str(cfg.context_playback.dataset)).resolve()
    healthy_sha = _sha256(healthy_path)
    loaded = load_fada_policy_checkpoint(healthy_path, device=device)
    dataset, metadata = load_support_query_dataset(
        dataset_path,
        loaded.policy.config,
        support_length=int(cfg.context_playback.support_length),
        query_length=int(cfg.context_playback.support_length),
        allow_legacy_single_anchor=True,
    )
    if metadata.get("source_checkpoint_sha256") != healthy_sha:
        raise ValueError("Context playback dataset healthy checkpoint identity mismatch")
    validation = _legacy_validation_split(
        dataset,
        validation_fraction=float(cfg.context_playback.validation_fraction),
        seed=int(cfg.context_playback.split_seed),
    )
    support_index = int(cfg.context_playback.support_index)
    if not 0 <= support_index < validation.batch_size:
        raise ValueError(
            f"support_index must be in [0, {validation.batch_size}), got {support_index}"
        )
    setup = prepare_support_query_training(
        loaded.policy,
        SupportQueryContextConfig(
            support_length=int(cfg.context_playback.support_length),
            context_hidden_dim=int(cfg.context_playback.hidden_dim),
            context_layers=int(cfg.context_playback.num_layers),
            delta_scale=float(cfg.context_playback.delta_scale),
        ),
        learning_rate=3.0e-4,
    )
    payload = torch.load(context_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Context playback checkpoint must be a mapping")
    if payload.get("fada_architecture") != asdict(setup.policy.config):
        raise ValueError("Context playback FADA architecture mismatch")
    if payload.get("context_config") != asdict(setup.policy.context_encoder.context_config):
        raise ValueError("Context playback encoder architecture mismatch")
    if payload.get("source_checkpoint_sha256") != healthy_sha:
        raise ValueError("Context playback healthy checkpoint identity mismatch")
    state = payload.get("context_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Context playback checkpoint is missing context_state_dict")
    setup.policy.context_encoder.load_state_dict(state, strict=True)
    setup.policy.eval()
    index = torch.tensor([support_index], dtype=torch.int64)
    support = validation.support.index_select(index).to(device)
    with torch.inference_mode():
        delta_z = setup.policy.context_encoder(support)

    class _FixedContextPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = setup.policy.config

        def forward(
            self,
            observation_history: torch.Tensor,
            action_history: torch.Tensor,
            command: torch.Tensor,
        ) -> Any:
            rows = observation_history.shape[0]
            return setup.policy.act_with_context(
                observation_history,
                action_history,
                command,
                delta_z.expand(rows, -1),
            )

    print(
        "[play_fada_context_viser] Context enabled: "
        f"validation pair_id={int(validation.pair_id[support_index])}, "
        f"delta_z_l2={float(torch.linalg.vector_norm(delta_z)):.6f}"
    )
    return FADAPlaybackController(_FixedContextPolicy().to(device), device=device)  # type: ignore[arg-type]


def _session_factory(**kwargs: Any) -> Any:
    cfg = kwargs["cfg"]
    context_controller = None
    if bool(OmegaConf.select(cfg, "context_playback.enabled", default=False)):
        context_controller = _context_controller(cfg, device=str(kwargs["device"]))
    session, policy_obs_mode, checkpoint_path = create_fada_playback_session(**kwargs)
    if context_controller is not None:
        session.controller = context_controller
        session.policy = session._fada_policy
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
