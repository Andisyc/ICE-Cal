from __future__ import annotations

import importlib
from dataclasses import fields, replace

import pytest
import torch

from unilab.algos.torch.distill import (
    FADAArchitectureConfig,
    load_fada_source_batch,
    save_fada_source_batch,
)
from unilab.algos.torch.distill.fada import FADASourceBatch


def _target_module():
    return importlib.import_module("unilab.algos.torch.distill.fada_target_data")


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=3,
        action_dim=2,
        command_dim=2,
        history_length=2,
        prediction_horizon=2,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


def _target_batch():
    module = _target_module()
    return module.FADATargetBatch(
        observation_history=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32),
        action_history=torch.tensor([[[7.0, 8.0], [9.0, 10.0]]], dtype=torch.float32),
        command=torch.tensor([[0.4, -0.1]], dtype=torch.float32),
        realized_future=torch.tensor(
            [[[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]], dtype=torch.float32
        ),
        executed_action_chunk=torch.tensor([[[17.0, 18.0], [19.0, 20.0]]], dtype=torch.float32),
        episode_id=torch.tensor([3], dtype=torch.int64),
        start_timestep=torch.tensor([4], dtype=torch.int64),
    )


def _metadata() -> dict[str, object]:
    return {
        "policy_checkpoint_sha256": "a" * 64,
        "config_fingerprint": "b" * 64,
        "task": "sac/g1_walk_flat/mujoco_left_knee_090",
        "fault_profile": "left_knee_strength_0.9",
        "num_envs": 1,
        "num_windows": 1,
    }


def _source_batch(config: FADAArchitectureConfig) -> FADASourceBatch:
    size = 1
    return FADASourceBatch(
        observation_history=torch.zeros(size, config.history_length, config.obs_dim),
        action_history=torch.zeros(size, config.history_length, config.action_dim),
        command=torch.zeros(size, config.command_dim),
        realized_future=torch.zeros(size, config.prediction_horizon, config.obs_dim),
        executed_action_chunk=torch.zeros(size, config.prediction_horizon, config.action_dim),
        oracle_future=torch.zeros(size, config.prediction_horizon, config.obs_dim),
        oracle_action_chunk=torch.zeros(size, config.prediction_horizon, config.action_dim),
        oracle_shadow_valid=torch.ones(size, dtype=torch.bool),
        idm_source_role=torch.zeros(size, dtype=torch.int64),
        oracle_first_action=torch.zeros(size, config.action_dim),
        command_scenario=torch.zeros(size, dtype=torch.int64),
        planner_eligible=torch.ones(size, dtype=torch.bool),
        cold_start=torch.zeros(size, dtype=torch.bool),
    )


def test_target_batch_schema_is_oracle_free_and_validates_exact_axes() -> None:
    module = _target_module()
    batch = _target_batch().validate(_config())

    assert batch is not None
    names = {field.name for field in fields(module.FADATargetBatch)}
    assert names == {
        "observation_history",
        "action_history",
        "command",
        "realized_future",
        "executed_action_chunk",
        "episode_id",
        "start_timestep",
    }
    assert all("oracle" not in name and "privileged" not in name for name in names)


def test_target_batch_rejects_nonfinite_or_invalid_lifecycle_identity() -> None:
    batch = _target_batch()
    bad_future = batch.realized_future.clone()
    bad_future[0, 0, 1] = float("nan")
    with pytest.raises(ValueError, match="realized_future.*finite"):
        replace(batch, realized_future=bad_future).validate(_config())
    with pytest.raises(ValueError, match="non-negative"):
        replace(batch, episode_id=torch.tensor([-1], dtype=torch.int64)).validate(_config())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_target_batch_requires_exact_float32_dtype(dtype: torch.dtype) -> None:
    batch = _target_batch()

    with pytest.raises(ValueError, match="observation_history must be torch.float32"):
        replace(batch, observation_history=batch.observation_history.to(dtype)).validate(_config())


def test_target_artifact_round_trip_preserves_tensors_and_identity(tmp_path) -> None:
    module = _target_module()
    path = tmp_path / "target.pt"

    module.save_fada_target_artifact(
        path,
        _target_batch(),
        config=_config(),
        metadata=_metadata(),
    )
    loaded = module.load_fada_target_artifact(path, config=_config())

    for field in fields(module.FADATargetBatch):
        torch.testing.assert_close(
            getattr(loaded.batch, field.name),
            getattr(_target_batch(), field.name),
        )
    assert dict(loaded.metadata) == _metadata()
    assert not path.with_suffix(".pt.tmp").exists()


def test_v2_target_artifact_owns_contract_and_legacy_v1_remains_readable(tmp_path) -> None:
    module = _target_module()
    path = tmp_path / "target.pt"
    module.save_fada_target_artifact(path, _target_batch(), config=_config(), metadata=_metadata())
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert module.FADA_TARGET_ARTIFACT_SCHEMA_VERSION == "fada-target-batch/v2"
    assert payload["architecture"]["observation_contract"] == "legacy_actor_obs_v1"

    payload["architecture"].pop("observation_contract")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="observation_contract"):
        module.load_fada_target_artifact(path, config=_config())

    payload["schema_version"] = "fada-target-batch/v1"
    torch.save(payload, path)
    loaded = module.load_fada_target_artifact(path, config=_config())
    torch.testing.assert_close(loaded.batch.command, _target_batch().command)


def test_target_artifact_rejects_architecture_or_metadata_drift(tmp_path) -> None:
    module = _target_module()
    path = tmp_path / "target.pt"
    module.save_fada_target_artifact(
        path,
        _target_batch(),
        config=_config(),
        metadata=_metadata(),
    )
    incompatible = replace(_config(), history_length=3)
    with pytest.raises(ValueError, match="architecture mismatch"):
        module.load_fada_target_artifact(path, config=incompatible)

    bad_metadata = _metadata()
    bad_metadata.pop("fault_profile")
    with pytest.raises(ValueError, match="metadata missing"):
        module.save_fada_target_artifact(
            tmp_path / "bad.pt",
            _target_batch(),
            config=_config(),
            metadata=bad_metadata,
        )


def test_target_artifact_load_rejects_persisted_non_float32_tensor(tmp_path) -> None:
    module = _target_module()
    path = tmp_path / "target.pt"
    module.save_fada_target_artifact(
        path,
        _target_batch(),
        config=_config(),
        metadata=_metadata(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["batch"]["realized_future"] = payload["batch"]["realized_future"].to(torch.float64)
    torch.save(payload, path)

    with pytest.raises(ValueError, match="realized_future must be torch.float32"):
        module.load_fada_target_artifact(path, config=_config())


@pytest.mark.parametrize(("field", "value"), [("num_envs", 1.5), ("num_windows", 2.75)])
def test_target_artifact_rejects_fractional_integer_metadata(
    tmp_path, field: str, value: float
) -> None:
    module = _target_module()
    metadata = _metadata()
    metadata[field] = value

    with pytest.raises(ValueError, match=rf"{field} must be a positive integer"):
        module.save_fada_target_artifact(
            tmp_path / "target.pt",
            _target_batch(),
            config=_config(),
            metadata=metadata,
        )


def test_target_artifact_save_rejects_num_windows_row_mismatch(tmp_path) -> None:
    module = _target_module()
    metadata = _metadata()
    metadata["num_windows"] = 2

    with pytest.raises(ValueError, match="num_windows must equal target batch row count"):
        module.save_fada_target_artifact(
            tmp_path / "target.pt",
            _target_batch(),
            config=_config(),
            metadata=metadata,
        )


def test_target_artifact_load_rejects_num_windows_row_mismatch(tmp_path) -> None:
    module = _target_module()
    path = tmp_path / "target.pt"
    module.save_fada_target_artifact(
        path,
        _target_batch(),
        config=_config(),
        metadata=_metadata(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"]["num_windows"] = 2
    torch.save(payload, path)

    with pytest.raises(ValueError, match="num_windows must equal target batch row count"):
        module.load_fada_target_artifact(path, config=_config())


def test_target_and_source_artifact_readers_fail_closed_on_cross_load(tmp_path) -> None:
    module = _target_module()
    target_path = tmp_path / "target.pt"
    source_path = tmp_path / "source.pt"
    module.save_fada_target_artifact(
        target_path,
        _target_batch(),
        config=_config(),
        metadata=_metadata(),
    )
    save_fada_source_batch(
        source_path,
        _source_batch(_config()),
        config=_config(),
        metadata={"iteration": 0},
    )

    with pytest.raises(ValueError, match="target artifact schema"):
        module.load_fada_target_artifact(source_path, config=_config())
    with pytest.raises(ValueError, match="source batch schema"):
        load_fada_source_batch(target_path, config=_config())
