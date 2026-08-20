from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scripts import play_fada_calibration_viser as playback_cli

ROOT = Path(__file__).resolve().parents[2]


def test_calibration_cli_entrypoints_expose_real_parsers() -> None:
    for script in (
        "prepare_fada_calibration_dataset.py",
        "prepare_fada_calibration_scale_evidence.py",
        "train_fada_calibration.py",
        "train_fada_calibration_stage1.py",
        "train_fada_calibration_stage2.py",
        "train_fada_calibration_stage3.py",
        "evaluate_fada_calibration.py",
        "play_fada_calibration_viser.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        if script == "train_fada_calibration.py":
            assert "--scale-evidence" in result.stdout
            assert "--scale-readings" not in result.stdout
        if script == "train_fada_calibration_stage1.py":
            assert "--stage1-steps" in result.stdout
            assert "--stage1-artifact" not in result.stdout
            assert "--stage2-artifact" not in result.stdout
            assert "--scale-evidence" not in result.stdout
        if script == "train_fada_calibration_stage2.py":
            assert "--stage1-artifact" in result.stdout
            assert "--stage2-steps" in result.stdout
            assert "--scale-evidence" not in result.stdout
        if script == "train_fada_calibration_stage3.py":
            assert "--stage2-artifact" in result.stdout
            assert "--scale-evidence" in result.stdout
            assert "--learning-rate" not in result.stdout
            assert "--stage1-steps" not in result.stdout
            assert "--stage2-steps" not in result.stdout


def test_calibration_playback_preset_enables_policy_consumption() -> None:
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(ROOT / "conf" / "distill"),
    ):
        cfg = compose(
            config_name="config",
            overrides=["calibration_playback=gain_delay_offset_v1"],
        )
    assert cfg.interactive.action_mode == "policy"
    assert list(cfg.calibration_playback.jump_threshold) == [0.25, 0.25, 0.25]


def test_playback_factory_binds_calibrated_controller(monkeypatch) -> None:
    bound = []

    class _Session:
        def bind_controller(self, controller) -> None:
            bound.append(controller)

    policy = object()
    healthy = SimpleNamespace(policy=object())
    monkeypatch.setattr(
        playback_cli,
        "load_fada_policy_checkpoint",
        lambda *args, **kwargs: healthy,
    )
    monkeypatch.setattr(
        playback_cli,
        "load_calibrated_policy",
        lambda *args, **kwargs: policy,
    )
    monkeypatch.setattr(playback_cli, "_sha256", lambda path: "source")
    monkeypatch.setattr(
        playback_cli,
        "create_fada_playback_session",
        lambda **kwargs: (_Session(), "actor", Path("source.pt")),
    )
    monkeypatch.setattr(
        playback_cli,
        "CalibratedFADAPlaybackController",
        lambda observed, **kwargs: (observed, kwargs),
    )
    cfg = OmegaConf.create(
        {
            "calibration_playback": {
                "source_checkpoint": "source.pt",
                "artifact": "artifact.pt",
                "jump_threshold": [0.1, 0.2, 0.3],
            }
        }
    )
    playback_cli._session_factory(cfg=cfg, device="cpu")
    assert bound == [(policy, {"device": "cpu", "jump_threshold": [0.1, 0.2, 0.3]})]
