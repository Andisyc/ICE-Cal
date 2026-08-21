from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scripts import play_fada_calibration_viser as playback_cli

from unilab.algos.torch.fada_context.calibration_collection import (
    canonicalize_resolved_task_backend_payload,
    load_gain_calibration_protocol,
)

ROOT = Path(__file__).resolve().parents[2]


def _collection_cli():
    try:
        return importlib.import_module("scripts.collect_fada_calibration_rollouts")
    except ModuleNotFoundError:
        pytest.fail("official gain calibration collection entrypoint is missing")


def test_calibration_cli_entrypoints_expose_real_parsers() -> None:
    for script in (
        "collect_fada_calibration_rollouts.py",
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
        if script == "collect_fada_calibration_rollouts.py":
            for flag in (
                "--source-checkpoint",
                "--expected-source-sha256",
                "--protocol",
                "--output",
                "--device",
            ):
                assert flag in result.stdout
            assert "--gain" not in result.stdout
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


def test_gain_collection_protocol_and_base_override_are_config_owned() -> None:
    collection_cli = _collection_cli()
    protocol, protocol_bytes, digest = load_gain_calibration_protocol(
        ROOT / "conf/fada_context/calibration_collection/gain_smoke_v1.yaml"
    )
    assert [(point.c_true, point.gain) for point in protocol.points] == [
        (-1.0, 0.8),
        (0.0, 1.0),
        (1.0, 1.2),
    ]
    assert (
        protocol_bytes
        == (ROOT / "conf/fada_context/calibration_collection/gain_smoke_v1.yaml").read_bytes()
    )
    assert len(digest) == 64
    cfg = collection_cli._compose_task(protocol)
    override = collection_cli._base_env_override(cfg, protocol)
    assert override["commands"]["vel_limit"] == [
        [0.4, 0.0, 0.0],
        [0.4, 0.0, 0.0],
    ]
    assert "action_execution_fault" not in override
    payload, payload_digest = canonicalize_resolved_task_backend_payload(cfg, override)
    assert payload["resolved_distill_config"]["training"]["task_name"] == "G1WalkFlat"
    assert len(payload_digest) == 64
    faulted = collection_cli._faulted_env_override(override, gain=0.8)
    assert faulted["action_execution_fault"] == {"mode": "gain", "gain": 0.8}


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
