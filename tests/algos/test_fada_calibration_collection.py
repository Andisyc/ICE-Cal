from __future__ import annotations

import copy
import hashlib
import importlib
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scripts import prepare_fada_calibration_dataset as prepare_cli
from torch import nn

import unilab.algos.torch.fada_context as fada_context
from unilab.algos.torch.distill.fada import FADAArchitectureConfig, PlannerIDMOutput
from unilab.algos.torch.fada_context.calibration import (
    CalibrationAxisSpec,
    DirectionBank,
    FaultAxisCatalog,
)
from unilab.algos.torch.fada_context.calibration_data import prepare_calibration_rollout_batch
from unilab.algos.torch.fada_context.calibration_training import direction_stage_loss
from unilab.base.np_env import NpEnvState


def _collection_module():
    try:
        return importlib.import_module("unilab.algos.torch.fada_context.calibration_collection")
    except ModuleNotFoundError:
        pytest.fail("public calibration rollout collection owner is missing")


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=2,
        action_dim=2,
        command_dim=3,
        history_length=30,
        prediction_horizon=6,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


def test_collection_owner_is_exported_from_the_fada_context_boundary() -> None:
    assert {
        "GainCalibrationCollectionProtocol",
        "collect_gain_calibration_rollouts",
        "load_gain_calibration_raw_rollouts",
    } <= set(fada_context.__all__)


class _FrozenPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _config()
        self.sentinel = nn.Parameter(torch.tensor([7.0]), requires_grad=False)
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.eval()

    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> PlannerIDMOutput:
        self.calls.append((observation_history.clone(), action_history.clone(), command.clone()))
        batch = observation_history.shape[0]
        time = torch.arange(6, dtype=observation_history.dtype).reshape(1, 6, 1)
        intent = observation_history[:, -1:, :].expand(-1, 6, -1) + time
        first = observation_history[:, -1, :]
        chunk = first[:, None, :].expand(-1, 6, -1) + time
        assert chunk.shape == (batch, 6, 2)
        return PlannerIDMOutput(
            predicted_future=intent,
            action_chunk=chunk,
            action=chunk[:, 0],
        )


class _PseudoEnv:
    num_envs = 1

    def __init__(self, *, gain: float, drift_first_rollout: bool = False) -> None:
        self.gain = gain
        self.drift_first_rollout = drift_first_rollout
        self.reset_count = 0
        self.local_step = 0
        self.actions: list[np.ndarray] = []
        self.closed = False
        self.autoreset = True

    def set_autoreset(self, enabled: bool) -> None:
        self.autoreset = enabled

    def reset_all(self) -> NpEnvState:
        self.reset_count += 1
        self.local_step = 0
        return self._state(np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32))

    def _state(self, command: np.ndarray) -> NpEnvState:
        value = float(self.reset_count * 100 + self.local_step)
        return NpEnvState(
            obs={"obs": np.asarray([[value, value + 0.25]], dtype=np.float32)},
            reward=np.zeros((1,), dtype=np.float32),
            terminated=np.zeros((1,), dtype=np.bool_),
            truncated=np.zeros((1,), dtype=np.bool_),
            info={"commands": command},
        )

    def step(self, action: np.ndarray) -> NpEnvState:
        nominal = np.asarray(action, dtype=np.float32).copy()
        self.actions.append(nominal)
        self.local_step += 1
        command = np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32)
        if self.drift_first_rollout and self.reset_count == 1 and self.local_step == 32:
            command[0, 0] = 0.5
        state = self._state(command)
        state.info["current_actions"] = nominal.copy()
        state.info["authority_actions"] = nominal.copy()
        state.info["executed_actions"] = nominal * self.gain
        return state

    def close(self) -> None:
        self.closed = True


def _scenario_spec(*, rows: int = 2, max_steps: int = 128):
    module = _collection_module()
    return module.GainCalibrationScenarioSpec(
        point=module.GainCalibrationPoint(c_true=-1.0, gain=0.8),
        split=module.GainCalibrationSplit(name="train", split_id=0, seed=101),
        fixed_command=(0.4, 0.0, 0.0),
        accepted_rows=rows,
        max_environment_steps=max_steps,
        observation_key="obs",
        command_key="commands",
    )


def test_scenario_uses_real_warmup_and_rolls_back_the_whole_drifted_episode() -> None:
    module = _collection_module()
    policy = _FrozenPolicy()
    env = _PseudoEnv(gain=0.8, drift_first_rollout=True)
    before = policy.sentinel.detach().clone()

    result = module.collect_gain_calibration_scenario(
        env,
        policy,
        _scenario_spec(),
        rollout_id_start=10,
        axis_spec=CalibrationAxisSpec.from_catalog(FaultAxisCatalog.default()),
    )

    assert result.environment_steps == 64
    assert result.rejected_transactions == 1
    assert result.next_rollout_id == 12
    assert result.rows["rollout_id"].tolist() == [11, 11]
    assert result.rows["observation_history"].shape == (2, 30, 2)
    assert result.rows["action_history"].shape == (2, 30, 2)
    assert not torch.any(result.rows["action_history"] == 0.0)
    assert len(env.actions) == 64
    for passed, policy_call in zip(env.actions, policy.calls, strict=True):
        np.testing.assert_array_equal(passed, policy_call[0][:, -1].numpy())
    torch.testing.assert_close(
        result.rows["executed_action"],
        result.rows["nominal_action_chunk"][:, 0] * 0.8,
    )
    torch.testing.assert_close(policy.sentinel, before, rtol=0.0, atol=0.0)
    assert env.autoreset is False


def _approved_protocol(module):
    return module.GainCalibrationCollectionProtocol(
        version="gain-smoke-v1",
        task_config="g1_walk_flat/mujoco",
        task_name="G1WalkFlat",
        sim_backend="mujoco",
        observation_key="obs",
        command_key="commands",
        fixed_command=(0.4, 0.0, 0.0),
        points=(
            module.GainCalibrationPoint(-1.0, 0.8),
            module.GainCalibrationPoint(0.0, 1.0),
            module.GainCalibrationPoint(1.0, 1.2),
        ),
        splits=(
            module.GainCalibrationSplit("train", 0, 101),
            module.GainCalibrationSplit("validation", 1, 201),
        ),
        accepted_rows_per_scenario=32,
        max_environment_steps_per_scenario=512,
    )


def test_protocol_admits_only_the_exact_approved_grid_before_env_creation() -> None:
    module = _collection_module()
    protocol = _approved_protocol(module)
    protocol.validate_approved()
    reordered = module.GainCalibrationCollectionProtocol(
        **{**asdict(protocol), "points": tuple(reversed(protocol.points))}
    )
    with pytest.raises(ValueError, match="approved gain grid"):
        reordered.validate_approved()
    extra = module.GainCalibrationCollectionProtocol(
        **{
            **asdict(protocol),
            "points": (*protocol.points, module.GainCalibrationPoint(0.5, 1.1)),
        }
    )
    with pytest.raises(ValueError, match="approved gain grid"):
        extra.validate_approved()


def _identity(module):
    return module.GainCalibrationRawIdentity(
        source_checkpoint_sha256="a" * 64,
        source_checkpoint_path="/server/planner_idm.pt",
        protocol_sha256=hashlib.sha256(_protocol_bytes()).hexdigest(),
        resolved_task_backend_sha256=module.sha256_canonical_mapping(
            _resolved_task_backend_payload()
        ),
        axis_catalog_version="gain-delay-offset-v1",
    )


def _protocol_bytes() -> bytes:
    return (
        Path(__file__).resolve().parents[2]
        / "conf/fada_context/calibration_collection/gain_smoke_v1.yaml"
    ).read_bytes()


def _resolved_task_backend_payload() -> dict[str, object]:
    return {
        "resolved_distill_config": {
            "training": {"task_name": "G1WalkFlat", "sim_backend": "mujoco"}
        },
        "base_env_override": {
            "commands": {
                "vel_limit": [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]],
                "heading_command": False,
            }
        },
    }


@pytest.fixture(scope="module")
def calibration_raw_artifacts():
    module = _collection_module()
    policy = _FrozenPolicy()
    current = module.collect_gain_calibration_rollouts(
        policy,
        _approved_protocol(module),
        lambda point, split: _PseudoEnv(gain=point.gain),
        catalog=FaultAxisCatalog.default(),
        identity=_identity(module),
        protocol_bytes=_protocol_bytes(),
        resolved_task_backend_payload=_resolved_task_backend_payload(),
    )
    legacy = copy.deepcopy(current)
    legacy["schema_version"] = "unilab_fada_gain_calibration_raw_rollouts_v1"
    legacy["method_contract_id"] = "FADA-CONTEXT-METHOD-v007"
    legacy["training_contract_id"] = "FADA-CONTEXT-TRAIN-v006"
    legacy.pop("axis_spec")
    legacy["axis_count"] = 3
    legacy["axis_names"] = ("gain", "delay", "offset")
    legacy["metadata"] = {
        **legacy["metadata"],
        "axis_catalog_version": "gain-delay-offset-v1",
    }
    return current, legacy


def test_full_collection_binds_identity_and_keeps_diagnostic_actions_out_of_training() -> None:
    module = _collection_module()
    policy = _FrozenPolicy()
    protocol = _approved_protocol(module)
    environments: list[_PseudoEnv] = []

    def factory(point, split):
        del split
        env = _PseudoEnv(gain=point.gain)
        environments.append(env)
        return env

    artifact = module.collect_gain_calibration_rollouts(
        policy,
        protocol,
        factory,
        catalog=FaultAxisCatalog.default(),
        identity=_identity(module),
        protocol_bytes=_protocol_bytes(),
        resolved_task_backend_payload=_resolved_task_backend_payload(),
    )

    assert artifact["observation_history"].shape == (192, 30, 2)
    assert artifact["nominal_action_chunk"].shape == (192, 6, 2)
    assert artifact["planner_intent"].shape == (192, 6, 2)
    assert artifact["c_true"].shape == (192, 3)
    assert artifact["executed_action"].shape == (192, 2)
    assert set(artifact["axis_name"]) == {"gain"}
    train_ids = set(artifact["rollout_id"][artifact["split_id"] == 0].tolist())
    validation_ids = set(artifact["rollout_id"][artifact["split_id"] == 1].tolist())
    assert train_ids.isdisjoint(validation_ids)
    assert all(env.closed for env in environments)
    batch = prepare_calibration_rollout_batch(artifact, policy.config, FaultAxisCatalog.default())
    assert "executed_action" not in batch.__dict__
    torch.testing.assert_close(
        batch.target_action_chunk,
        batch.nominal_action_chunk / batch.injected_strength[:, None, None],
    )
    with pytest.raises(ValueError, match="no rows for axis 1"):
        direction_stage_loss(
            policy,
            DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8),
            batch,
            axis_index=1,
        )


def test_raw_round_trip_rejects_corrupt_identity_and_atomic_failure_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _collection_module()
    policy = _FrozenPolicy()
    artifact = module.collect_gain_calibration_rollouts(
        policy,
        _approved_protocol(module),
        lambda point, split: _PseudoEnv(gain=point.gain),
        catalog=FaultAxisCatalog.default(),
        identity=_identity(module),
        protocol_bytes=_protocol_bytes(),
        resolved_task_backend_payload=_resolved_task_backend_payload(),
    )
    target = module.save_gain_calibration_raw_rollouts(
        tmp_path / "raw.pt",
        artifact,
        catalog=FaultAxisCatalog.default(),
    )
    loaded = module.load_gain_calibration_raw_rollouts(
        target,
        catalog=FaultAxisCatalog.default(),
        expected_source_sha256="a" * 64,
        expected_architecture=policy.config,
    )
    assert loaded["protocol_bytes"] == _protocol_bytes()
    assert loaded["resolved_task_backend_payload"] == _resolved_task_backend_payload()
    assert loaded["metadata"]["protocol_sha256"] == hashlib.sha256(_protocol_bytes()).hexdigest()
    payload = torch.load(target, map_location="cpu", weights_only=True)
    payload["architecture"]["hidden_dim"] = 16
    torch.save(payload, target)
    with pytest.raises(ValueError, match="architecture identity"):
        module.load_gain_calibration_raw_rollouts(
            target,
            catalog=FaultAxisCatalog.default(),
            expected_source_sha256="a" * 64,
            expected_architecture=policy.config,
        )
    module.save_gain_calibration_raw_rollouts(
        target,
        artifact,
        catalog=FaultAxisCatalog.default(),
    )
    payload = torch.load(target, map_location="cpu", weights_only=True)
    payload["method_contract_id"] = "wrong"
    torch.save(payload, target)
    with pytest.raises(ValueError, match="method Contract"):
        module.load_gain_calibration_raw_rollouts(
            target,
            catalog=FaultAxisCatalog.default(),
            expected_source_sha256="a" * 64,
        )

    target.write_bytes(b"preserve-me")

    def fail_save(payload, path):
        Path(path).write_bytes(b"partial")
        raise OSError("synthetic write failure")

    monkeypatch.setattr(module.torch, "save", fail_save)
    with pytest.raises(OSError, match="synthetic"):
        module.save_gain_calibration_raw_rollouts(
            target,
            artifact,
            catalog=FaultAxisCatalog.default(),
        )
    assert target.read_bytes() == b"preserve-me"
    assert list(tmp_path.glob(".raw.pt.*.tmp")) == []


def test_exact_legacy_gain_raw_v1_is_admitted_for_dataset_resealing(
    tmp_path: Path,
    calibration_raw_artifacts,
) -> None:
    module = _collection_module()
    policy = _FrozenPolicy()
    _, legacy = calibration_raw_artifacts
    path = tmp_path / "legacy-raw.pt"
    torch.save(legacy, path)
    loaded = module.load_gain_calibration_raw_rollouts(
        path,
        catalog=FaultAxisCatalog.default(),
        expected_source_sha256="a" * 64,
        expected_architecture=policy.config,
    )
    assert loaded["axis_name"] == ["gain"] * 192
    forbidden_target = tmp_path / "forbidden-legacy-write.pt"
    with pytest.raises(ValueError, match="unsupported.*schema"):
        module.save_gain_calibration_raw_rollouts(
            forbidden_target,
            legacy,
            catalog=FaultAxisCatalog.default(),
        )
    assert not forbidden_target.exists()
    assert list(tmp_path.glob(".forbidden-legacy-write.pt.*")) == []
    tampered = dict(legacy)
    tampered["training_contract_id"] = "FADA-CONTEXT-TRAIN-v007"
    torch.save(tampered, path)
    with pytest.raises(ValueError, match="training Contract"):
        module.load_gain_calibration_raw_rollouts(
            path,
            catalog=FaultAxisCatalog.default(),
        )


@pytest.mark.parametrize(
    "duplicate_key",
    ["active_axes", "axis_catalog_version", "axis_count", "axis_names", "catalog_version"],
)
def test_active_raw_writer_rejects_duplicate_top_level_axis_identity(
    tmp_path: Path,
    calibration_raw_artifacts,
    duplicate_key: str,
) -> None:
    module = _collection_module()
    current, _ = calibration_raw_artifacts
    artifact = copy.deepcopy(current)
    artifact[duplicate_key] = ["gain", "delay", "offset"]
    target = tmp_path / f"duplicate-{duplicate_key}.pt"
    with pytest.raises(ValueError, match="duplicate axis identity"):
        module.save_gain_calibration_raw_rollouts(
            target,
            artifact,
            catalog=FaultAxisCatalog.default(),
        )
    assert not target.exists()
    assert list(tmp_path.glob(f".duplicate-{duplicate_key}.pt.*")) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "method_contract",
        "catalog_version",
        "axis_order",
        "non_gain_label",
        "nonzero_omitted_coordinate",
        "source",
        "protocol",
        "backend",
        "architecture",
    ],
)
def test_legacy_gain_raw_gateway_rejects_identity_and_semantic_mutations(
    tmp_path: Path,
    calibration_raw_artifacts,
    mutation: str,
) -> None:
    module = _collection_module()
    _, source = calibration_raw_artifacts
    artifact = copy.deepcopy(source)
    expected_source = "a" * 64
    if mutation == "schema":
        artifact["schema_version"] = "unilab_fada_gain_calibration_raw_rollouts_v0"
    elif mutation == "method_contract":
        artifact["method_contract_id"] = "FADA-CONTEXT-METHOD-v008"
    elif mutation == "catalog_version":
        artifact["metadata"]["axis_catalog_version"] = "wrong-catalog"
    elif mutation == "axis_order":
        artifact["axis_names"] = ("offset", "delay", "gain")
    elif mutation == "non_gain_label":
        artifact["axis_name"][0] = "delay"
    elif mutation == "nonzero_omitted_coordinate":
        artifact["c_true"][0, 1] = 0.2
    elif mutation == "source":
        expected_source = "b" * 64
    elif mutation == "protocol":
        artifact["metadata"]["protocol_sha256"] = "d" * 64
    elif mutation == "backend":
        artifact["metadata"]["resolved_task_backend_sha256"] = "d" * 64
    else:
        artifact["architecture"]["hidden_dim"] = 16
    path = tmp_path / f"legacy-{mutation}.pt"
    torch.save(artifact, path)
    with pytest.raises(ValueError):
        module.load_gain_calibration_raw_rollouts(
            path,
            catalog=FaultAxisCatalog.default(),
            expected_source_sha256=expected_source,
            expected_architecture=_config(),
        )


@pytest.mark.parametrize(
    "field",
    ["protocol_sha256", "resolved_task_backend_sha256"],
)
def test_raw_loader_recomputes_provenance_digests_from_embedded_material(
    tmp_path: Path,
    field: str,
) -> None:
    module = _collection_module()
    policy = _FrozenPolicy()
    artifact = module.collect_gain_calibration_rollouts(
        policy,
        _approved_protocol(module),
        lambda point, split: _PseudoEnv(gain=point.gain),
        catalog=FaultAxisCatalog.default(),
        identity=_identity(module),
        protocol_bytes=_protocol_bytes(),
        resolved_task_backend_payload=_resolved_task_backend_payload(),
    )
    artifact["metadata"][field] = "d" * 64
    target = tmp_path / "tampered.pt"
    torch.save(artifact, target)

    with pytest.raises(ValueError, match="provenance digest"):
        module.load_gain_calibration_raw_rollouts(
            target,
            catalog=FaultAxisCatalog.default(),
            expected_source_sha256="a" * 64,
            expected_architecture=policy.config,
        )


def test_prepare_cli_rejects_tampered_provenance_before_dataset_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _collection_module()
    policy = _FrozenPolicy()
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"source-checkpoint")
    source_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    identity = module.GainCalibrationRawIdentity(
        source_checkpoint_sha256=source_sha256,
        source_checkpoint_path=str(checkpoint),
        protocol_sha256=hashlib.sha256(_protocol_bytes()).hexdigest(),
        resolved_task_backend_sha256=module.sha256_canonical_mapping(
            _resolved_task_backend_payload()
        ),
        axis_catalog_version="gain-delay-offset-v1",
    )
    artifact = module.collect_gain_calibration_rollouts(
        policy,
        _approved_protocol(module),
        lambda point, split: _PseudoEnv(gain=point.gain),
        catalog=FaultAxisCatalog.default(),
        identity=identity,
        protocol_bytes=_protocol_bytes(),
        resolved_task_backend_payload=_resolved_task_backend_payload(),
    )
    artifact["metadata"]["protocol_sha256"] = "d" * 64
    raw = tmp_path / "raw.pt"
    torch.save(artifact, raw)
    saved: list[Path] = []
    monkeypatch.setattr(
        prepare_cli,
        "load_fada_policy_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(policy=policy),
    )
    monkeypatch.setattr(
        prepare_cli,
        "save_calibration_dataset",
        lambda path, *args, **kwargs: saved.append(Path(path)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_fada_calibration_dataset.py",
            "--source-checkpoint",
            str(checkpoint),
            "--raw-rollouts",
            str(raw),
            "--output",
            str(tmp_path / "dataset.pt"),
        ],
    )

    with pytest.raises(ValueError, match="provenance digest"):
        prepare_cli.main()
    assert saved == []
