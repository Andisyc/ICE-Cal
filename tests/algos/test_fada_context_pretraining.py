from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unilab.algos.torch.distill.fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.training_setup import (
    ContextTrainingSetupConfig,
    prepare_context_training,
)
from unilab.algos.torch.fada_context.trajectory_collector import (
    PairedTrajectoryCollectionConfig,
    collect_paired_context_trajectories,
)
from unilab.algos.torch.fada_context.trajectory_data import (
    ContextTrajectoryDataset,
    load_context_trajectory_dataset,
    save_context_trajectory_dataset,
)


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


def _dataset(config: FADAArchitectureConfig) -> ContextTrajectoryDataset:
    states = torch.randn(4, 9, config.obs_dim)
    return ContextTrajectoryDataset(
        observation_history=states[:, 1 : config.history_length + 1],
        action_history=torch.randn(4, config.history_length, config.action_dim),
        command=torch.randn(4, config.command_dim),
        healthy_reference=torch.randn(4, 3, config.obs_dim),
        fault_state=states,
        fault_action=torch.randn(4, 8, config.action_dim),
        pair_id=torch.arange(4),
    )


def test_context_trajectory_dataset_round_trip_and_contiguous_transition(tmp_path: Path) -> None:
    config = _config()
    dataset = _dataset(config)
    path = tmp_path / "context.pt"

    save_context_trajectory_dataset(path, dataset, config, metadata={"fault": "left_knee_0.9"})
    loaded, metadata = load_context_trajectory_dataset(path, config)

    loaded.validate(config)
    loaded.fault_transition_batch().validate(
        prepare_context_training(
            FADAPlannerIDMPolicy(config), ContextTrainingSetupConfig(dynamics_hidden_dims=(8,))
        ).dynamics.config
    )
    torch.testing.assert_close(loaded.fault_state, dataset.fault_state)
    assert metadata == {"fault": "left_knee_0.9"}


def test_context_trajectory_dataset_rejects_non_integral_pair_identity() -> None:
    config = _config()
    dataset = _dataset(config)
    invalid = ContextTrajectoryDataset(
        **{**dataset.__dict__, "pair_id": dataset.pair_id.float()}
    )

    with pytest.raises(ValueError, match="pair_id"):
        invalid.validate(config)


def test_training_setup_optimizer_ownership_is_disjoint_and_frozen() -> None:
    config = _config()
    prepared = prepare_context_training(
        FADAPlannerIDMPolicy(config),
        ContextTrainingSetupConfig(
            context_hidden_dim=12,
            context_num_layers=1,
            dynamics_hidden_dims=(8,),
            dynamics_ensemble_size=2,
        ),
    )

    context_ids = {id(parameter) for parameter in prepared.policy.context_encoder.parameters()}
    dynamics_ids = {id(parameter) for parameter in prepared.dynamics.parameters()}
    assert context_ids.isdisjoint(dynamics_ids)
    assert all(not parameter.requires_grad for parameter in prepared.policy.planner.parameters())
    assert all(not parameter.requires_grad for parameter in prepared.policy.idm.parameters())


class _PairedEnv:
    def __init__(self, *, strength: float) -> None:
        self.num_envs = 2
        self.strength = strength
        self._autoreset = True
        self._step = 0
        self.state = None

    def set_autoreset(self, enabled: bool) -> None:
        self._autoreset = enabled

    def reset_all(self):
        self._step = 0
        self.state = self._make_state(torch.zeros(2, 7).numpy())
        return self.state

    def capture_rollout_snapshot(self):
        return {"step": self._step, "state": self.state}

    def restore_rollout_snapshot(self, snapshot) -> None:
        self._step = snapshot["step"]
        self.state = snapshot["state"]

    def step(self, action):
        assert not self._autoreset
        observation = self.state.obs["obs"] + self.strength * action.mean(axis=1, keepdims=True)
        self._step += 1
        self.state = self._make_state(observation)
        return self.state

    def _make_state(self, observation):
        return SimpleNamespace(
            obs={"obs": observation.astype("float32")},
            info={"commands": torch.tensor([[0.4, 0.0], [0.4, 0.0]]).numpy()},
            terminated=torch.zeros(2, dtype=torch.bool).numpy(),
            truncated=torch.zeros(2, dtype=torch.bool).numpy(),
        )


def test_paired_collector_uses_fault_history_and_healthy_reference() -> None:
    torch.manual_seed(5)
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    result = collect_paired_context_trajectories(
        _PairedEnv(strength=1.0),
        _PairedEnv(strength=0.9),
        policy,
        PairedTrajectoryCollectionConfig(num_samples=2, reference_horizon=3),
    )

    dataset = result.dataset
    assert dataset.observation_history.shape == (2, 5, 7)
    assert dataset.healthy_reference.shape == (2, 3, 7)
    assert dataset.fault_state.shape == (2, 9, 7)
    assert dataset.fault_action.shape == (2, 8, 3)
    assert not torch.equal(dataset.healthy_reference, dataset.fault_state[:, 6:])
    dataset.fault_transition_batch().validate(
        prepare_context_training(
            FADAPlannerIDMPolicy(config),
            ContextTrainingSetupConfig(dynamics_hidden_dims=(8,), dynamics_ensemble_size=2),
        ).dynamics.config
    )
