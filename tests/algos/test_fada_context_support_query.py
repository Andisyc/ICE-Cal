from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.train_fada_context_support_query import _require_fresh_run_paths

from unilab.algos.torch.distill.fada import (
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlanner,
    FADAPlannerIDMPolicy,
)
from unilab.algos.torch.fada_context.support_query import (
    ContextQueryBatch,
    SupportContextBatch,
    SupportQueryBatch,
    SupportQueryContextConfig,
    context_first_action_loss,
)
from unilab.algos.torch.fada_context.support_query_collector import (
    SupportQueryCollectionConfig,
    collect_support_query_pairs,
)
from unilab.algos.torch.fada_context.support_query_data import (
    load_support_query_dataset,
    save_support_query_dataset,
    split_support_query_by_rollout,
    support_query_split_identity_sha256,
)
from unilab.algos.torch.fada_context.support_query_training import (
    load_context_support_query_checkpoint,
    prepare_support_query_training,
    save_context_support_query_checkpoint,
)

CHECKPOINT_IDENTITIES = {
    "dataset_sha256": "dataset-sha",
    "train_split_sha256": "train-split-sha",
    "validation_split_sha256": "validation-split-sha",
}

EXPECTED_CHECKPOINT_IDENTITIES = {
    "expected_dataset_sha256": "dataset-sha",
    "expected_train_split_sha256": "train-split-sha",
    "expected_validation_split_sha256": "validation-split-sha",
}


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=7,
        action_dim=3,
        command_dim=2,
        history_length=5,
        prediction_horizon=3,
        hidden_dim=16,
        num_heads=4,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=32,
    )


def _batch(config: FADAArchitectureConfig, *, batch_size: int = 4, support_length: int = 8) -> SupportQueryBatch:
    command = torch.full((batch_size, config.command_dim), 0.4)
    return SupportQueryBatch(
        support=SupportContextBatch(
            target_future=torch.randn(
                batch_size, support_length, config.prediction_horizon, config.obs_dim
            ),
            realized_state=torch.randn(batch_size, support_length, config.obs_dim),
            executed_action=torch.randn(batch_size, support_length, config.action_dim),
        ),
        query=ContextQueryBatch(
            observation_history=torch.randn(
                batch_size, config.history_length, config.obs_dim
            ),
            action_history=torch.randn(
                batch_size, config.history_length, config.action_dim
            ),
            command=command.clone(),
            planner_intent=torch.randn(
                batch_size, config.prediction_horizon, config.obs_dim
            ),
            realized_future=torch.randn(
                batch_size, config.prediction_horizon, config.obs_dim
            ),
            executed_action_chunk=torch.randn(
                batch_size, config.prediction_horizon, config.action_dim
            ),
        ),
        support_command=command,
        pair_id=torch.arange(batch_size, dtype=torch.int64),
        support_rollout_id=torch.arange(batch_size, dtype=torch.int64) * 2,
        query_rollout_id=torch.arange(batch_size, dtype=torch.int64) * 2 + 1,
    ).validate(config, support_length=support_length)


def test_idm_latent_split_is_exactly_original_forward() -> None:
    torch.manual_seed(2)
    config = _config()
    idm = FADAInverseDynamicsModel(config).eval()
    observation = torch.randn(3, config.history_length, config.obs_dim)
    action = torch.randn(3, config.history_length, config.action_dim)
    future = torch.randn(3, config.prediction_horizon, config.obs_dim)

    original = idm(observation, action, future)
    split = idm.decode_latent(idm.encode_latent(observation, action, future))

    torch.testing.assert_close(split, original, rtol=0.0, atol=0.0)


def test_zero_context_matches_frozen_query_reconstruction() -> None:
    torch.manual_seed(3)
    config = _config()
    healthy = FADAPlannerIDMPolicy(config).eval()
    batch = _batch(config)
    setup = prepare_support_query_training(
        healthy,
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )

    output = setup.policy.reconstruct_query(batch)
    nominal = healthy.idm(
        batch.query.observation_history,
        batch.query.action_history,
        batch.query.realized_future,
    )

    torch.testing.assert_close(output.delta_z, torch.zeros_like(output.delta_z))
    torch.testing.assert_close(output.action_chunk, nominal, rtol=0.0, atol=0.0)


def test_context_loss_supervises_only_executed_first_action_and_updates_context() -> None:
    torch.manual_seed(5)
    config = _config()
    batch = _batch(config)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    changed_chunk = batch.query.executed_action_chunk.clone()
    changed_chunk[:, 1:] += 1000.0
    changed = replace(batch, query=replace(batch.query, executed_action_chunk=changed_chunk))

    loss = context_first_action_loss(setup.policy, batch)
    torch.testing.assert_close(context_first_action_loss(setup.policy, changed), loss)
    loss.backward()

    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in setup.policy.context_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in setup.policy.planner.parameters())
    assert all(parameter.grad is None for parameter in setup.policy.idm.parameters())


def test_support_query_rejects_command_or_rollout_identity_mismatch() -> None:
    config = _config()
    batch = _batch(config)
    with pytest.raises(ValueError, match="commands must match"):
        replace(batch, support_command=batch.support_command + 1.0).validate(config)
    with pytest.raises(ValueError, match="different rollout ids"):
        replace(batch, query_rollout_id=batch.support_rollout_id).validate(config)
    with pytest.raises(ValueError, match="share one dtype"):
        replace(batch, support_command=batch.support_command.double()).validate(config)


def test_support_query_dataset_round_trip(tmp_path: Path) -> None:
    config = _config()
    batch = _batch(config)
    path = save_support_query_dataset(
        tmp_path / "support_query.pt",
        batch,
        config,
        support_length=8,
        metadata={
            "source_checkpoint_sha256": "abc",
            "task_config": "fault-070",
            "fault_joint": "left_knee",
            "fault_strength": 0.7,
            "command": [0.4, 0.0],
            "seed": 7,
        },
    )

    loaded, metadata = load_support_query_dataset(path, config, support_length=8)

    torch.testing.assert_close(loaded.query.realized_future, batch.query.realized_future)
    torch.testing.assert_close(loaded.support.target_future, batch.support.target_future)
    assert metadata["fault_strength"] == 0.7


def test_support_query_split_keeps_rollout_groups_disjoint() -> None:
    config = _config()
    batch = _batch(config, batch_size=6)
    grouped = replace(
        batch,
        support_rollout_id=torch.tensor([0, 0, 2, 2, 4, 4]),
        query_rollout_id=torch.tensor([1, 1, 3, 3, 5, 5]),
    ).validate(config)

    train, validation = split_support_query_by_rollout(
        grouped, validation_fraction=0.34, seed=7
    )

    train_groups = set(zip(train.support_rollout_id.tolist(), train.query_rollout_id.tolist()))
    validation_groups = set(
        zip(validation.support_rollout_id.tolist(), validation.query_rollout_id.tolist())
    )
    assert train_groups.isdisjoint(validation_groups)
    assert train.batch_size + validation.batch_size == grouped.batch_size
    assert support_query_split_identity_sha256(train) != support_query_split_identity_sha256(
        validation
    )


def test_training_paths_must_be_fresh(tmp_path: Path) -> None:
    artifact = tmp_path / "dataset.pt"
    output_dir = tmp_path / "run"
    _require_fresh_run_paths(artifact, output_dir)
    artifact.touch()
    with pytest.raises(FileExistsError, match="dataset artifact already exists"):
        _require_fresh_run_paths(artifact, output_dir)
    artifact.unlink()
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="output directory already exists"):
        _require_fresh_run_paths(artifact, output_dir)


def test_context_checkpoint_strictly_binds_healthy_source(tmp_path: Path) -> None:
    config = _config()
    context_config = SupportQueryContextConfig(
        support_length=8, context_hidden_dim=12, context_layers=1
    )
    source = prepare_support_query_training(
        FADAPlannerIDMPolicy(config), context_config, learning_rate=3.0e-4
    )
    with torch.no_grad():
        source.policy.context_encoder.delta_head.bias.fill_(0.05)
    path = save_context_support_query_checkpoint(
        tmp_path / "context.pt",
        source,
        source_checkpoint_sha256="healthy-sha",
        **CHECKPOINT_IDENTITIES,
        step=3,
        split_seed=7,
        metrics={"validation_mse": 0.25},
        resolved_config={"task": "fault-070"},
    )
    restored = prepare_support_query_training(
        FADAPlannerIDMPolicy(config), context_config, learning_rate=3.0e-4
    )

    payload = load_context_support_query_checkpoint(
        path,
        restored,
        expected_source_checkpoint_sha256="healthy-sha",
        **EXPECTED_CHECKPOINT_IDENTITIES,
        load_optimizer=True,
    )

    torch.testing.assert_close(
        restored.policy.context_encoder.delta_head.bias,
        source.policy.context_encoder.delta_head.bias,
    )
    assert payload["step"] == 3
    with pytest.raises(ValueError, match="source identity mismatch"):
        load_context_support_query_checkpoint(
            path,
            restored,
            expected_source_checkpoint_sha256="wrong-sha",
            **EXPECTED_CHECKPOINT_IDENTITIES,
        )
    with pytest.raises(ValueError, match="dataset_sha256 mismatch"):
        load_context_support_query_checkpoint(
            path,
            restored,
            expected_source_checkpoint_sha256="healthy-sha",
            expected_dataset_sha256="wrong-dataset",
            expected_train_split_sha256="train-split-sha",
            expected_validation_split_sha256="validation-split-sha",
        )


def test_context_checkpoint_load_rolls_back_on_optimizer_failure(tmp_path: Path) -> None:
    config = _config()
    context_config = SupportQueryContextConfig(
        support_length=8, context_hidden_dim=12, context_layers=1
    )
    source = prepare_support_query_training(
        FADAPlannerIDMPolicy(config), context_config, learning_rate=3.0e-4
    )
    path = save_context_support_query_checkpoint(
        tmp_path / "context.pt",
        source,
        source_checkpoint_sha256="healthy-sha",
        **CHECKPOINT_IDENTITIES,
        step=1,
        split_seed=7,
        metrics={"validation_mse": 0.25},
        resolved_config={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["optimizer_state_dict"] = {"invalid": True}
    torch.save(payload, path)
    restored = prepare_support_query_training(
        FADAPlannerIDMPolicy(config), context_config, learning_rate=3.0e-4
    )
    context_before = {
        name: value.detach().clone()
        for name, value in restored.policy.context_encoder.state_dict().items()
    }
    optimizer_before = restored.optimizer.state_dict()

    with pytest.raises((KeyError, ValueError)):
        load_context_support_query_checkpoint(
            path,
            restored,
            expected_source_checkpoint_sha256="healthy-sha",
            **EXPECTED_CHECKPOINT_IDENTITIES,
            load_optimizer=True,
        )

    for name, value in restored.policy.context_encoder.state_dict().items():
        torch.testing.assert_close(value, context_before[name])
    assert restored.optimizer.state_dict() == optimizer_before


@dataclass
class _State:
    obs: dict[str, np.ndarray]
    info: dict[str, np.ndarray]
    terminated: np.ndarray
    truncated: np.ndarray


class _IndependentResetEnv:
    def __init__(self, config: FADAArchitectureConfig) -> None:
        self.config = config
        self.num_envs = 2
        self.reset_count = 0
        self.autoreset = True
        self.state: _State | None = None

    def set_autoreset(self, enabled: bool) -> None:
        self.autoreset = enabled

    def reset_all(self) -> _State:
        self.reset_count += 1
        obs = np.full(
            (self.num_envs, self.config.obs_dim), self.reset_count / 10.0, dtype=np.float32
        )
        command = np.tile(np.asarray([[0.4, 0.0]], dtype=np.float32), (self.num_envs, 1))
        self.state = _State(
            obs={"obs": obs},
            info={"commands": command},
            terminated=np.zeros(self.num_envs, dtype=np.bool_),
            truncated=np.zeros(self.num_envs, dtype=np.bool_),
        )
        return self.state

    def step(self, action: np.ndarray) -> _State:
        assert self.state is not None
        next_obs = self.state.obs["obs"].copy()
        next_obs[:, : self.config.action_dim] += action
        self.state = _State(
            obs={"obs": next_obs},
            info={"commands": self.state.info["commands"].copy()},
            terminated=np.zeros(self.num_envs, dtype=np.bool_),
            truncated=np.zeros(self.num_envs, dtype=np.bool_),
        )
        return self.state


def test_collector_uses_two_independent_fault_rollouts_and_actual_actions() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    torch.nn.init.zeros_(policy.idm.action_head.weight)
    torch.nn.init.constant_(policy.idm.action_head.bias, 0.25)
    env = _IndependentResetEnv(config)

    result = collect_support_query_pairs(
        env,
        policy,
        SupportQueryCollectionConfig(num_pairs=2, support_length=8, max_reset_pairs=1),
    )

    assert env.reset_count == 2
    assert env.autoreset is False
    assert result.accepted_pairs == 2
    assert torch.all(result.batch.support_rollout_id == 0)
    assert torch.all(result.batch.query_rollout_id == 1)
    torch.testing.assert_close(
        result.batch.query.executed_action_chunk,
        torch.full_like(result.batch.query.executed_action_chunk, 0.25),
    )
    assert not torch.equal(
        result.batch.support.realized_state[:, 0],
        result.batch.query.observation_history[:, 0],
    )
