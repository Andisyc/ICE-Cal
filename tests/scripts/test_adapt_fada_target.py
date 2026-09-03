from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill import (
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
    FADATrainer,
    load_fada_adapted_checkpoint,
    load_fada_policy_checkpoint,
    save_fada_checkpoint,
)
from unilab.algos.torch.distill.fada_target_data import (
    FADATargetBatch,
    save_fada_target_artifact,
)
from unilab.algos.torch.distill.workflow import file_sha256

ROOT_DIR = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT_DIR / "conf" / "offpolicy"
SCRIPT_PATH = ROOT_DIR / "scripts" / "adapt_fada_target.py"


def _load_script() -> Any:
    if not SCRIPT_PATH.is_file():
        pytest.fail(f"Stage-D CLI owner is missing: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("adapt_fada_target", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _compose(*overrides: str) -> DictConfig:
    if not (CONF_DIR / "fada_adapt.yaml").is_file():
        pytest.fail("Stage-D Hydra composition root is missing: conf/offpolicy/fada_adapt.yaml")
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            config_name="fada_adapt",
            overrides=list(overrides),
            return_hydra_config=True,
        )


def _compose_slope(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            config_name="fada_slope_adapt",
            overrides=list(overrides),
            return_hydra_config=True,
        )


def test_slope_adaptation_reads_target_only_v3_artifact() -> None:
    cfg = _compose_slope()

    assert cfg.hydra.runtime.choices.task == "sac/g1_walk_flat/mujoco_fada_slope_15"
    assert cfg.adaptation.target_artifact_path.endswith("g1_slope_15_mujoco/target.pt")
    assert cfg.adaptation.output_checkpoint_path.endswith("g1_slope_15_mujoco_v3.pt")
    assert cfg.adaptation.rank == 8


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=66,
        action_dim=29,
        command_dim=3,
        observation_contract="g1_fada_state_v2",
        history_length=3,
        prediction_horizon=2,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


def _artifacts(tmp_path: Path) -> tuple[Path, Path, str, str]:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    source = tmp_path / "source.pt"
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters()),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters()),
        max_grad_norm=1.0,
    )
    save_fada_checkpoint(
        source,
        policy,
        trainer,
        completed_iterations=5,
        samples_seen=100,
        runtime_config={"training_schedule": "alternating_idm_then_planner"},
    )
    source_sha = file_sha256(source)
    rows = 6
    batch = FADATargetBatch(
        observation_history=torch.arange(
            rows * config.history_length * config.obs_dim, dtype=torch.float32
        ).reshape(rows, config.history_length, config.obs_dim),
        action_history=torch.arange(
            rows * config.history_length * config.action_dim, dtype=torch.float32
        ).reshape(rows, config.history_length, config.action_dim),
        command=torch.tensor([[0.4, 0.0, 0.0]] * rows, dtype=torch.float32),
        realized_future=torch.arange(
            rows * config.prediction_horizon * config.obs_dim, dtype=torch.float32
        ).reshape(rows, config.prediction_horizon, config.obs_dim),
        executed_action_chunk=torch.arange(
            rows * config.prediction_horizon * config.action_dim, dtype=torch.float32
        ).reshape(rows, config.prediction_horizon, config.action_dim),
        episode_id=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
        start_timestep=torch.tensor([3, 4, 13, 14, 23, 24], dtype=torch.int64),
    )
    target = tmp_path / "target.pt"
    save_fada_target_artifact(
        target,
        batch,
        config=config,
        metadata={
            "policy_checkpoint_sha256": source_sha,
            "config_fingerprint": "1" * 64,
            "task": "G1WalkFlat",
            "fault_profile": "left_knee_strength_0.9",
            "num_envs": 1,
            "num_windows": rows,
        },
    )
    return source, target, source_sha, file_sha256(target)


def test_adaptation_config_reuses_target_owner_and_paper_lora_defaults() -> None:
    cfg = _compose()

    assert cfg.hydra.runtime.choices.task == "sac/g1_walk_flat/mujoco_fada_target"
    assert cfg.hydra.runtime.choices.adaptation == "fada_lora"
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.adaptation.rank == 8
    assert cfg.adaptation.alpha == 16.0
    assert cfg.adaptation.dropout == 0.05
    assert cfg.adaptation.confirm_train is False
    assert cfg.adaptation.batch_size == 512
    assert cfg.adaptation.max_updates == 400
    assert cfg.adaptation.observation_contract == "g1_fada_state_v2"
    assert cfg.adaptation.source_checkpoint_path == "planner_idm_v022_cpu_limited.pt"
    assert cfg.adaptation.expected_source_checkpoint_sha256 is None
    assert cfg.adaptation.expected_target_artifact_sha256 is None
    assert cfg.adaptation.target_artifact_path.endswith("left_knee_090/faulty.pt")


def test_right_knee_adaptation_uses_the_matching_stage_c_bundle() -> None:
    cfg = _compose("fault=right_knee_090")

    assert cfg.fault.actuator_index == 9
    assert cfg.adaptation.target_artifact_path.endswith("right_knee_090/faulty.pt")
    assert cfg.adaptation.output_checkpoint_path.endswith("right_knee_090_v3.pt")


def test_preflight_builds_frozen_adapter_and_writes_nothing(tmp_path: Path) -> None:
    module = _load_script()
    source, target, source_sha, target_sha = _artifacts(tmp_path)
    output = tmp_path / "adapted.pt"
    cfg = _compose(
        f"adaptation.source_checkpoint_path={source}",
        f"adaptation.expected_source_checkpoint_sha256={source_sha}",
        f"adaptation.target_artifact_path={target}",
        f"adaptation.expected_target_artifact_sha256={target_sha}",
        f"adaptation.output_checkpoint_path={output}",
        "adaptation.batch_size=2",
    )

    result = module.preflight_fada_adaptation(cfg, root_dir=ROOT_DIR)

    assert result.source_checkpoint_sha256 == source_sha
    assert result.target_artifact_sha256 == target_sha
    assert result.train_rows + result.validation_rows == 6
    assert result.trainable_parameter_count > 0
    assert result.total_parameter_count > result.trainable_parameter_count
    assert result.confirm_train is False
    assert not output.exists()


def test_preflight_accepts_null_hashes_and_records_observed_identity(tmp_path: Path) -> None:
    module = _load_script()
    source, target, source_sha, target_sha = _artifacts(tmp_path)
    output = tmp_path / "adapted.pt"
    cfg = _compose(
        f"adaptation.source_checkpoint_path={source}",
        "adaptation.expected_source_checkpoint_sha256=null",
        f"adaptation.target_artifact_path={target}",
        "adaptation.expected_target_artifact_sha256=null",
        f"adaptation.output_checkpoint_path={output}",
        "adaptation.batch_size=2",
    )

    result = module.preflight_fada_adaptation(cfg, root_dir=ROOT_DIR)

    assert result.source_checkpoint_sha256 == source_sha
    assert result.target_artifact_sha256 == target_sha
    assert not output.exists()


def test_default_run_stops_ready_without_calling_training_or_writing(tmp_path: Path) -> None:
    module = _load_script()
    source, target, source_sha, target_sha = _artifacts(tmp_path)
    output = tmp_path / "adapted.pt"
    cfg = _compose(
        f"adaptation.source_checkpoint_path={source}",
        f"adaptation.expected_source_checkpoint_sha256={source_sha}",
        f"adaptation.target_artifact_path={target}",
        f"adaptation.expected_target_artifact_sha256={target_sha}",
        f"adaptation.output_checkpoint_path={output}",
        "adaptation.batch_size=2",
    )
    called = False

    def forbidden_train(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("preflight-only default must not train")

    summary = module.run_fada_adaptation(
        cfg,
        root_dir=ROOT_DIR,
        train_fn=forbidden_train,
    )

    assert summary["status"] == "D_TRAIN_READY"
    assert called is False
    assert not output.exists()


def test_preflight_rejects_legacy_source_before_target_load_or_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    owner = importlib.import_module(
        "unilab.algos.torch.distill.fada.target_adaptation_workflow"
    )
    source = tmp_path / "legacy-source.pt"
    target = tmp_path / "target.pt"
    source.write_bytes(b"legacy")
    target.write_bytes(b"target")
    output = tmp_path / "adapted.pt"
    cfg = _compose(
        f"adaptation.source_checkpoint_path={source}",
        f"adaptation.expected_source_checkpoint_sha256={file_sha256(source)}",
        f"adaptation.target_artifact_path={target}",
        f"adaptation.expected_target_artifact_sha256={file_sha256(target)}",
        f"adaptation.output_checkpoint_path={output}",
    )
    legacy_policy = FADAPlannerIDMPolicy(
        FADAArchitectureConfig(
            obs_dim=4,
            action_dim=2,
            command_dim=3,
            history_length=3,
            prediction_horizon=2,
            hidden_dim=8,
            num_heads=2,
            planner_layers=1,
            idm_encoder_layers=1,
            idm_decoder_layers=1,
            feedforward_dim=16,
        )
    )
    monkeypatch.setattr(
        owner,
        "load_fada_policy_checkpoint",
        lambda *_args, **_kwargs: type(
            "Loaded", (), {"policy": legacy_policy, "checkpoint": {"schema_version": 2}}
        )(),
    )
    target_loads = 0

    def forbidden_target_load(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal target_loads
        target_loads += 1
        raise AssertionError("legacy source must fail before target loading")

    monkeypatch.setattr(owner, "load_fada_target_artifact", forbidden_target_load)

    with pytest.raises(ValueError, match="requires current schema-5"):
        module.preflight_fada_adaptation(cfg, root_dir=ROOT_DIR)

    assert target_loads == 0
    assert not output.exists()


def test_confirmed_one_update_runs_official_transaction_and_persists_resume_state(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source, target, source_sha, target_sha = _artifacts(tmp_path)
    output = tmp_path / "adapted.pt"
    cfg = _compose(
        f"adaptation.source_checkpoint_path={source}",
        f"adaptation.expected_source_checkpoint_sha256={source_sha}",
        f"adaptation.target_artifact_path={target}",
        f"adaptation.expected_target_artifact_sha256={target_sha}",
        f"adaptation.output_checkpoint_path={output}",
        "adaptation.confirm_train=true",
        "adaptation.batch_size=2",
        "adaptation.max_updates=1",
    )
    source_before = file_sha256(source)
    target_before = file_sha256(target)
    source_state = load_fada_policy_checkpoint(source, device="cpu").policy.state_dict()
    torch.manual_seed(17)

    summary = module.run_fada_adaptation(cfg, root_dir=ROOT_DIR)

    assert summary["status"] == "completed"
    assert summary["completed_steps"] == 1
    assert summary["samples_seen"] == 2
    assert torch.isfinite(torch.tensor(summary["train_loss"]))
    assert torch.isfinite(torch.tensor(summary["validation_loss"]))
    assert output.is_file()
    assert file_sha256(source) == source_before == source_sha
    assert file_sha256(target) == target_before == target_sha

    loaded = load_fada_adapted_checkpoint(output, device="cpu")
    checkpoint = loaded.checkpoint
    assert checkpoint["completed_steps"] == 1
    assert checkpoint["samples_seen"] == 2
    assert checkpoint["source_checkpoint_sha256"] == source_sha
    assert checkpoint["target_artifact_sha256"] == target_sha
    optimizer_state = checkpoint["optimizer_state_dict"]["state"]
    assert optimizer_state
    for parameter_state in optimizer_state.values():
        assert float(parameter_state["step"]) == 1.0
        for value in parameter_state.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()

    adapted_state = loaded.policy.state_dict()
    targets = set(checkpoint["lora_config"]["target_modules"])
    for name, expected in source_state.items():
        adapted_name = name
        if name.startswith("idm."):
            module_name, separator, parameter_name = name.removeprefix("idm.").rpartition(".")
            if separator:
                for target in targets:
                    if module_name == target or module_name.startswith(f"{target}."):
                        suffix = module_name.removeprefix(target)
                        adapted_name = f"idm.{target}.base{suffix}.{parameter_name}"
                        break
        torch.testing.assert_close(adapted_state[adapted_name], expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("selector", "value", "match"),
    [
        ("adaptation.expected_source_checkpoint_sha256", "0" * 64, "source checkpoint"),
        ("adaptation.expected_target_artifact_sha256", "0" * 64, "target artifact"),
        ("adaptation.rank", 4, "rank"),
        ("adaptation.dropout", 0.0, "dropout"),
        ("adaptation.seed", 1.5, "seed"),
        ("adaptation.max_grad_norm", float("nan"), "max_grad_norm"),
    ],
)
def test_preflight_rejects_identity_or_paper_parameter_drift(
    tmp_path: Path, selector: str, value: Any, match: str
) -> None:
    module = _load_script()
    source, target, source_sha, target_sha = _artifacts(tmp_path)
    cfg = _compose(
        f"adaptation.source_checkpoint_path={source}",
        f"adaptation.expected_source_checkpoint_sha256={source_sha}",
        f"adaptation.target_artifact_path={target}",
        f"adaptation.expected_target_artifact_sha256={target_sha}",
        f"adaptation.output_checkpoint_path={tmp_path / 'adapted.pt'}",
        "adaptation.batch_size=2",
    )
    OmegaConf.update(cfg, selector, value, merge=False)

    with pytest.raises(ValueError, match=match):
        module.preflight_fada_adaptation(cfg, root_dir=ROOT_DIR)


def test_preflight_rejects_existing_output_and_target_source_identity_mismatch(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source, target, source_sha, target_sha = _artifacts(tmp_path)
    output = tmp_path / "adapted.pt"
    output.write_bytes(b"owned")
    existing = _compose(
        f"adaptation.source_checkpoint_path={source}",
        f"adaptation.expected_source_checkpoint_sha256={source_sha}",
        f"adaptation.target_artifact_path={target}",
        f"adaptation.expected_target_artifact_sha256={target_sha}",
        f"adaptation.output_checkpoint_path={output}",
        "adaptation.batch_size=2",
    )
    with pytest.raises(FileExistsError, match="already exists"):
        module.preflight_fada_adaptation(existing, root_dir=ROOT_DIR)

    payload = torch.load(target, map_location="cpu", weights_only=True)
    payload["metadata"]["policy_checkpoint_sha256"] = "9" * 64
    torch.save(payload, target)
    mismatched = _compose(
        f"adaptation.source_checkpoint_path={source}",
        f"adaptation.expected_source_checkpoint_sha256={source_sha}",
        f"adaptation.target_artifact_path={target}",
        f"adaptation.expected_target_artifact_sha256={file_sha256(target)}",
        f"adaptation.output_checkpoint_path={tmp_path / 'new.pt'}",
        "adaptation.batch_size=2",
    )
    with pytest.raises(ValueError, match="policy checkpoint"):
        module.preflight_fada_adaptation(mismatched, root_dir=ROOT_DIR)
