"""Launch frozen v008 calibrated FADA playback through the existing viewer composition root."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR / "scripts", ROOT_DIR / "src", ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from play_interactive import _build_play_args, play_interactive  # noqa: E402

from unilab.algos.torch.distill import load_fada_policy_checkpoint  # noqa: E402
from unilab.algos.torch.fada_context.calibration_data import (  # noqa: E402
    load_fault_axis_catalog,
)
from unilab.algos.torch.fada_context.calibration_runtime import (  # noqa: E402
    CalibratedFADAPlaybackController,
    load_calibrated_policy,
)
from unilab.visualization.interactive_playback import create_fada_playback_session  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_factory(**kwargs: Any) -> Any:
    cfg = kwargs["cfg"]
    source = (ROOT_DIR / str(cfg.calibration_playback.source_checkpoint)).resolve()
    artifact = (ROOT_DIR / str(cfg.calibration_playback.artifact)).resolve()
    axis_catalog_path = (ROOT_DIR / str(cfg.calibration_playback.axis_catalog)).resolve()
    catalog = load_fault_axis_catalog(axis_catalog_path)
    healthy = load_fada_policy_checkpoint(source, device=str(kwargs["device"])).policy
    policy = load_calibrated_policy(
        healthy,
        artifact,
        device=str(kwargs["device"]),
        expected_metadata={"source_tracker_sha256": _sha256(source)},
        catalog=catalog,
    )
    session, policy_obs_mode, checkpoint_path = create_fada_playback_session(**kwargs)
    threshold = {
        str(name): float(value) for name, value in cfg.calibration_playback.jump_threshold.items()
    }
    session.bind_controller(
        CalibratedFADAPlaybackController(
            policy,
            device=str(kwargs["device"]),
            jump_threshold=threshold,
        )
    )
    return session, policy_obs_mode, checkpoint_path


@hydra.main(version_base="1.3", config_path="../conf/distill", config_name="config")
def main(cfg: DictConfig) -> None:
    play_interactive(
        _build_play_args(cfg, algo="fada"),
        cfg,
        algo="fada",
        fada_session_factory=_session_factory,
    )


if __name__ == "__main__":
    main()
