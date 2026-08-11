from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import torch

from unilab.algos.torch.distill import (
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
    load_fada_policy_checkpoint,
)
from unilab.algos.torch.distill.fada_playback import FADAPlaybackController


class _RecordingPolicy:
    def __init__(self, config: FADAArchitectureConfig) -> None:
        self.config = config
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __call__(self, observation_history, action_history, command):
        self.calls.append((observation_history.clone(), action_history.clone(), command.clone()))
        value = float(len(self.calls))
        action = torch.full((observation_history.shape[0], self.config.action_dim), value)
        return SimpleNamespace(action=action)


def _small_config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
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


def test_fada_playback_controller_owns_history_and_done_reset() -> None:
    policy = _RecordingPolicy(_small_config())
    controller = FADAPlaybackController(policy, device="cpu")

    first_obs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    command = torch.tensor([[0.2, -0.1, 0.3]])
    torch.testing.assert_close(controller.act({"actor": first_obs}, command), torch.ones(1, 2))

    first_obs_history, first_action_history, first_command = policy.calls[0]
    torch.testing.assert_close(first_obs_history, first_obs.unsqueeze(1).repeat(1, 3, 1))
    torch.testing.assert_close(first_action_history, torch.zeros(1, 3, 2))
    torch.testing.assert_close(first_command, command)

    second_obs = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    controller.act({"actor": second_obs}, command)
    second_obs_history, second_action_history, _ = policy.calls[1]
    torch.testing.assert_close(second_obs_history[:, -1], second_obs)
    torch.testing.assert_close(second_action_history[:, -1], torch.ones(1, 2))

    controller.reset(torch.tensor([True]))
    reset_obs = torch.tensor([[9.0, 10.0, 11.0, 12.0]])
    controller.act({"actor": reset_obs}, command)
    reset_obs_history, reset_action_history, _ = policy.calls[2]
    torch.testing.assert_close(reset_obs_history, reset_obs.unsqueeze(1).repeat(1, 3, 1))
    torch.testing.assert_close(reset_action_history, torch.zeros(1, 3, 2))


def test_load_fada_policy_checkpoint_reconstructs_strict_inference_policy(tmp_path) -> None:
    config = _small_config()
    policy = FADAPlannerIDMPolicy(config)
    checkpoint = tmp_path / "planner_idm.pt"
    torch.save(
        {
            "schema_version": 1,
            "architecture": asdict(config),
            "planner_state_dict": policy.planner.state_dict(),
            "idm_state_dict": policy.idm.state_dict(),
            "planner_optimizer_state_dict": {},
            "idm_optimizer_state_dict": {},
            "completed_iterations": 8,
            "samples_seen": 100,
            "runtime_config": {},
        },
        checkpoint,
    )

    loaded = load_fada_policy_checkpoint(checkpoint, device="cpu")

    assert loaded.policy.training is False
    assert loaded.policy.config == config
    assert loaded.checkpoint["completed_iterations"] == 8
    output = loaded.policy(
        torch.zeros(1, config.history_length, config.obs_dim),
        torch.zeros(1, config.history_length, config.action_dim),
        torch.zeros(1, config.command_dim),
    )
    assert output.action.shape == (1, config.action_dim)
    assert torch.isfinite(output.action).all()
