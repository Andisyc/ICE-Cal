"""Launch browser-based MuJoCo playback for a trained FADA Planner-IDM checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from play_interactive import _build_play_args
from play_viser import play_viser


@hydra.main(version_base="1.3", config_path="../conf/distill", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compose the distill task owner and hand one FADA session to the Viser viewer."""

    # B1: FADA browser playback is MuJoCo-only; action mode remains config-owned.
    if str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("play_fada_viser.py only supports MuJoCo backend; use task=<task>/mujoco.")
    # B2: 复用 Viser scene lifecycle, 仅替换为 stateful FADA playback session.
    play_viser(_build_play_args(cfg, algo="fada"), cfg, algo="fada")


if __name__ == "__main__":
    main()
