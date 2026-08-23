from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scripts import evaluate_fada_calibration as evaluation_cli
from scripts import play_fada_calibration_viser as playback_cli
from scripts import prepare_fada_calibration_dataset as prepare_cli

from unilab.algos.torch.fada_context.calibration import (
    CalibrationAxisSpec,
    FaultAxisCatalog,
)
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
                "--axis-catalog",
                "--output",
                "--device",
            ):
                assert flag in result.stdout
            assert "--gain" not in result.stdout
        if script == "prepare_fada_calibration_dataset.py":
            assert "--active-axis" in result.stdout
        elif script.startswith("train_fada_calibration"):
            assert "--active-axis" not in result.stdout
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
    assert dict(cfg.calibration_playback.jump_threshold) == {
        "gain": 0.25,
        "delay": 0.25,
        "offset": 0.25,
    }


def test_single_axis_evaluation_rejects_before_artifact_or_upper_bound_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = FaultAxisCatalog.default()
    healthy = SimpleNamespace(config=object())
    dataset = SimpleNamespace(
        batch=object(),
        metadata={},
        axis_spec=CalibrationAxisSpec.from_catalog(catalog, ("gain",)),
    )
    monkeypatch.setattr(
        evaluation_cli,
        "load_fada_policy_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(policy=healthy),
    )
    monkeypatch.setattr(evaluation_cli, "load_fault_axis_catalog", lambda path: catalog)
    monkeypatch.setattr(
        evaluation_cli,
        "load_calibration_dataset",
        lambda *args, **kwargs: dataset,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("single-axis preflight must stop before downstream artifacts")

    monkeypatch.setattr(evaluation_cli, "load_calibrated_policy", forbidden)
    monkeypatch.setattr(evaluation_cli, "load_calibration_full_finetune_upper_bound", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_fada_calibration.py",
            "--source-checkpoint",
            "source.pt",
            "--calibration-artifact",
            "calibration.pt",
            "--dataset",
            "dataset.pt",
            "--full-finetune-action-chunks",
            "upper.pt",
        ],
    )
    with pytest.raises(ValueError, match="not applicable"):
        evaluation_cli.main()


def test_prepare_dataset_preserves_repeated_active_axis_cli_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = FaultAxisCatalog.default()
    policy = SimpleNamespace(config=object())
    raw = {
        "metadata": {
            "protocol_sha256": "protocol",
            "resolved_task_backend_sha256": "backend",
        }
    }
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        prepare_cli,
        "load_fada_policy_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(policy=policy),
    )
    monkeypatch.setattr(prepare_cli, "_sha256", lambda path: "source")
    monkeypatch.setattr(prepare_cli, "load_fault_axis_catalog", lambda path: catalog)
    monkeypatch.setattr(
        prepare_cli,
        "load_gain_calibration_raw_rollouts",
        lambda *args, **kwargs: raw,
    )

    def prepare(raw_value, config, catalog_value, axis_spec):
        observed["axis_spec"] = axis_spec
        return object()

    monkeypatch.setattr(prepare_cli, "prepare_calibration_rollout_batch", prepare)
    monkeypatch.setattr(
        prepare_cli,
        "calibration_split_identity_sha256",
        lambda batch: "split",
    )
    monkeypatch.setattr(
        prepare_cli,
        "save_calibration_dataset",
        lambda *args, **kwargs: observed.setdefault("saved_axis_spec", kwargs["axis_spec"]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_fada_calibration_dataset.py",
            "--source-checkpoint",
            "source.pt",
            "--raw-rollouts",
            "raw.pt",
            "--output",
            "dataset.pt",
            "--active-axis",
            "offset",
            "--active-axis",
            "gain",
        ],
    )
    assert prepare_cli.main() == 0
    axis_spec = observed["axis_spec"]
    assert isinstance(axis_spec, CalibrationAxisSpec)
    assert axis_spec.names == ("offset", "gain")
    assert observed["saved_axis_spec"] == axis_spec


def test_evaluation_passes_dataset_axis_spec_into_runtime_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = FaultAxisCatalog.default()
    axis_spec = CalibrationAxisSpec.from_catalog(catalog, ("offset", "gain"))
    healthy = SimpleNamespace(config=object())
    dataset = SimpleNamespace(
        batch="batch",
        metadata={
            "source_tracker_sha256": "source",
            "split_identity_sha256": "split",
        },
        axis_spec=axis_spec,
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        evaluation_cli,
        "load_fada_policy_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(policy=healthy),
    )
    monkeypatch.setattr(evaluation_cli, "load_fault_axis_catalog", lambda path: catalog)
    monkeypatch.setattr(
        evaluation_cli,
        "load_calibration_dataset",
        lambda *args, **kwargs: dataset,
    )
    monkeypatch.setattr(
        evaluation_cli,
        "_sha256",
        lambda path: "source" if Path(path).name == "source.pt" else "dataset",
    )

    def load_calibrated(*args, **kwargs):
        observed["expected_axis_spec"] = kwargs["expected_axis_spec"]
        return "calibrated"

    monkeypatch.setattr(evaluation_cli, "load_calibrated_policy", load_calibrated)
    monkeypatch.setattr(
        evaluation_cli,
        "load_calibration_full_finetune_upper_bound",
        lambda *args, **kwargs: "upper",
    )
    monkeypatch.setattr(
        evaluation_cli,
        "evaluate_held_out_calibration",
        lambda *args, **kwargs: {"status": "offline"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_fada_calibration.py",
            "--source-checkpoint",
            "source.pt",
            "--calibration-artifact",
            "calibration.pt",
            "--dataset",
            "dataset.pt",
            "--full-finetune-action-chunks",
            "upper.pt",
        ],
    )
    assert evaluation_cli.main() == 0
    assert observed["expected_axis_spec"] == axis_spec


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
    monkeypatch.setattr(playback_cli, "load_fault_axis_catalog", lambda path: object())
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
                "axis_catalog": "axes.yaml",
                "jump_threshold": {"offset": 0.3, "gain": 0.1},
            }
        }
    )
    playback_cli._session_factory(cfg=cfg, device="cpu")
    assert bound == [(policy, {"device": "cpu", "jump_threshold": {"offset": 0.3, "gain": 0.1}})]
