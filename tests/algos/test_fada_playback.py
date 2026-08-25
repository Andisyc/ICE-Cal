from __future__ import annotations

import inspect
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from scripts import play_fada_context_viser
from torch import nn

from unilab.algos.torch.distill import (
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
    load_fada_policy_checkpoint,
)
from unilab.algos.torch.distill import fada_playback as fada_playback_module
from unilab.algos.torch.distill.fada_playback import FADAPlaybackController
from unilab.algos.torch.fada_context.support_query import (
    ContextActionOutput,
    FADASupportContextEncoder,
    FrozenIDMSupportQueryPolicy,
    SupportBoundContextPolicy,
    SupportContextBatch,
    SupportQueryContextConfig,
)
from unilab.visualization.interactive_playback import FADAPlaybackSession


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


def test_playback_controller_uses_public_structural_policy_protocol() -> None:
    protocol = getattr(fada_playback_module, "FADAPlaybackPolicy", None)

    assert protocol is not None
    assert get_type_hints(FADAPlaybackController.__init__)["policy"] is protocol
    assert "type: ignore[arg-type]" not in inspect.getsource(
        play_fada_context_viser._context_controller
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


def test_fada_playback_projects_raw_g1_actor_observation_for_v2_policy() -> None:
    config = FADAArchitectureConfig(
        obs_dim=66,
        action_dim=29,
        command_dim=3,
        observation_contract="g1_fada_state_v2",
        history_length=2,
        prediction_horizon=2,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )
    policy = _RecordingPolicy(config)
    controller = FADAPlaybackController(policy, device="cpu")
    raw = torch.arange(98, dtype=torch.float32).unsqueeze(0)

    action = controller.act(raw, torch.tensor([[0.4, 0.0, 0.0]]))

    expected = torch.cat((raw[:, :64], raw[:, 96:98]), dim=1)
    observed_history, observed_actions, observed_command = policy.calls[0]
    torch.testing.assert_close(observed_history, expected.unsqueeze(1).repeat(1, 2, 1))
    torch.testing.assert_close(observed_actions, torch.zeros(1, 2, 29))
    torch.testing.assert_close(observed_command, torch.tensor([[0.4, 0.0, 0.0]]))
    assert action.shape == (1, 29)


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


def test_support_bound_playback_rejects_command_before_first_context_action() -> None:
    config = _small_config()
    healthy = FADAPlannerIDMPolicy(config).eval()
    context = FADASupportContextEncoder(
        config,
        SupportQueryContextConfig(support_length=2, context_hidden_dim=5, context_layers=1),
    ).eval()
    policy = FrozenIDMSupportQueryPolicy(
        healthy.planner,
        healthy.idm,
        context,
    ).eval()
    support = SupportContextBatch(
        target_future=torch.zeros(
            1,
            2,
            config.prediction_horizon,
            config.obs_dim,
        ),
        realized_state=torch.zeros(1, 2, config.obs_dim),
        executed_action=torch.zeros(1, 2, config.action_dim),
    )
    support_command = torch.tensor([[0.2, -0.1, 0.3]])
    bound = SupportBoundContextPolicy(policy, support, support_command).eval()
    parameter_snapshot = {
        name: value.detach().clone() for name, value in policy.state_dict().items()
    }
    context_support_ids: list[int] = []
    handle = context.register_forward_pre_hook(
        lambda _module, inputs: context_support_ids.append(id(inputs[0]))
    )
    observation_history = torch.zeros(1, config.history_length, config.obs_dim)
    action_history = torch.zeros(1, config.history_length, config.action_dim)
    try:
        with pytest.raises(ValueError, match="does not match Support command provenance"):
            bound(
                observation_history,
                action_history,
                support_command + torch.tensor([[0.1, 0.0, 0.0]]),
            )
        assert context_support_ids == []
        first = bound(observation_history, action_history, support_command)
        second = bound(observation_history, action_history, support_command)
    finally:
        handle.remove()

    assert len(set(context_support_ids)) == 1
    assert len(context_support_ids) == 2
    assert torch.equal(first.action, first.action_chunk[:, 0, :])
    assert torch.equal(second.action, second.action_chunk[:, 0, :])
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, parameter_snapshot[name], rtol=0.0, atol=0.0)


class _RowwiseBoundPolicy(nn.Module):
    def __init__(self, config: FADAArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = SimpleNamespace(
            context_config=SupportQueryContextConfig(
                support_length=2,
                context_hidden_dim=5,
                context_layers=1,
            )
        )

    def act_with_context(
        self,
        support: SupportContextBatch,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> ContextActionOutput:
        delta_z = torch.zeros(
            support.batch_size,
            self.config.hidden_dim,
            dtype=support.realized_state.dtype,
            device=support.realized_state.device,
        )
        delta_z[:, :2] = (
            support.realized_state[:, 0, :2]
            + observation_history[:, -1, :2]
            + action_history[:, -1]
            + command[:, :2]
        )
        first_action = delta_z[:, :2]
        action_chunk = torch.stack(
            (first_action, first_action + torch.tensor([5.0, 7.0])),
            dim=1,
        )
        latent = torch.zeros(
            support.batch_size,
            self.config.prediction_horizon,
            self.config.hidden_dim,
        )
        return ContextActionOutput(
            delta_z=delta_z,
            query_latent=latent,
            repaired_latent=latent,
            action_chunk=action_chunk,
        )


def test_support_bound_context_policy_is_covariant_and_keeps_binding_immutable() -> None:
    config = _small_config()
    support = SupportContextBatch(
        target_future=torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 4),
        realized_state=torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
            ]
        ),
        executed_action=torch.tensor([[[0.5, 0.6], [0.7, 0.8]], [[5.0, 6.0], [7.0, 8.0]]]),
    )
    observation_history = torch.zeros(2, config.history_length, config.obs_dim)
    observation_history[0, -1, :2] = torch.tensor([0.1, 0.2])
    observation_history[1, -1, :2] = torch.tensor([1.0, 2.0])
    action_history = torch.zeros(2, config.history_length, config.action_dim)
    action_history[0, -1] = torch.tensor([0.3, 0.4])
    action_history[1, -1] = torch.tensor([3.0, 4.0])
    command = torch.tensor([[0.01, 0.02, 0.03], [0.1, 0.2, 0.3]])
    permutation = torch.tensor([1, 0], dtype=torch.int64)
    inverse = torch.argsort(permutation)
    permuted_support = support.index_select(permutation)
    permuted_command = command.index_select(0, permutation)

    bound = SupportBoundContextPolicy(  # type: ignore[arg-type]
        _RowwiseBoundPolicy(config), support, command
    ).eval()
    permuted_bound = SupportBoundContextPolicy(  # type: ignore[arg-type]
        _RowwiseBoundPolicy(config), permuted_support, permuted_command
    ).eval()
    bound_support_before = tuple(value.clone() for value in bound.support.tensors())
    bound_command_before = bound.support_command.clone()
    permuted_support_before = tuple(value.clone() for value in permuted_bound.support.tensors())
    permuted_command_before = permuted_bound.support_command.clone()

    # Caller-side mutation is the controlled counterexample for an aliased binding.
    for value in support.tensors():
        value.add_(1000.0)
    command.add_(1000.0)

    output = bound(
        observation_history,
        action_history,
        bound_command_before,
    )
    permuted_output = permuted_bound(
        observation_history.index_select(0, permutation),
        action_history.index_select(0, permutation),
        permuted_command_before,
    )

    expected_delta = torch.zeros(2, config.hidden_dim)
    expected_delta[:, :2] = torch.tensor([[1.41, 2.62], [14.1, 26.2]])
    expected_chunk = torch.stack(
        (expected_delta[:, :2], expected_delta[:, :2] + torch.tensor([5.0, 7.0])),
        dim=1,
    )
    torch.testing.assert_close(output.delta_z, expected_delta)
    torch.testing.assert_close(output.action_chunk, expected_chunk)
    torch.testing.assert_close(output.action, expected_chunk[:, 0])
    for name in ("delta_z", "action_chunk", "action"):
        torch.testing.assert_close(
            getattr(permuted_output, name).index_select(0, inverse),
            getattr(output, name),
        )
        assert not torch.equal(getattr(permuted_output, name), getattr(output, name))

    for observed, expected in zip(bound.support.tensors(), bound_support_before, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(bound.support_command, bound_command_before, rtol=0.0, atol=0.0)
    for observed, expected in zip(
        permuted_bound.support.tensors(), permuted_support_before, strict=True
    ):
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        permuted_bound.support_command,
        permuted_command_before,
        rtol=0.0,
        atol=0.0,
    )


class _PlaybackEnv:
    action_space = SimpleNamespace(
        shape=(2,),
        low=np.full(2, -1.0, dtype=np.float32),
        high=np.full(2, 1.0, dtype=np.float32),
    )

    def __init__(self) -> None:
        self.state = SimpleNamespace(
            info={"commands": np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32)}
        )


class _PlaybackWrapper:
    def __init__(self) -> None:
        self.actions: list[torch.Tensor] = []

    def reset(self):
        return {"actor": torch.zeros(1, 4)}, {}

    def step(self, actions):
        self.actions.append(actions.detach().clone())
        return (
            {"actor": torch.ones(1, 4)},
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            {},
        )


class _PlaybackController:
    def __init__(self, action_value: float) -> None:
        self.action_value = action_value
        self.act_calls = 0

    def reset(self, _done=None) -> None:
        return None

    def act(self, _observation, _commands) -> torch.Tensor:
        self.act_calls += 1
        return torch.full((1, 2), self.action_value)


def _playback_session(*, action_mode: str, controller=None):
    return FADAPlaybackSession(
        controller=controller,
        env=_PlaybackEnv(),
        wrapped_env=_PlaybackWrapper(),
        device="cpu",
        action_mode=action_mode,
        num_envs=1,
    )


class _RowwisePlaybackPolicy:
    def __init__(self) -> None:
        self.config = _small_config()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, observation_history, _action_history, command):
        current_observation = observation_history[:, -1]
        self.calls.append((current_observation.clone(), command.clone()))
        return SimpleNamespace(action=current_observation[:, :2] + 10.0 * command[:, :2])


class _BatchPlaybackEnv:
    action_space = _PlaybackEnv.action_space

    def __init__(self, commands: np.ndarray) -> None:
        self.state = SimpleNamespace(info={"commands": commands.copy()})


class _BatchPlaybackWrapper:
    def __init__(self, observation: torch.Tensor) -> None:
        self.observation = observation.clone()
        self.actions: list[torch.Tensor] = []

    def reset(self):
        return {"actor": self.observation.clone()}, {}

    def step(self, actions):
        self.actions.append(actions.detach().clone())
        rows = actions.shape[0]
        return (
            {"actor": self.observation.clone()},
            torch.zeros(rows),
            torch.zeros(rows, dtype=torch.bool),
            {},
        )


def test_fada_playback_session_is_covariant_to_two_row_permutation() -> None:
    observation = torch.tensor(
        [[1.0, 2.0, 30.0, 40.0], [5.0, 7.0, 80.0, 90.0]],
        dtype=torch.float32,
    )
    commands = np.asarray(
        [[0.1, 0.2, 0.0], [0.4, 0.3, 0.0]],
        dtype=np.float32,
    )
    permutation = torch.tensor([1, 0], dtype=torch.int64)

    def run(rows: torch.Tensor) -> tuple[torch.Tensor, _RowwisePlaybackPolicy]:
        policy = _RowwisePlaybackPolicy()
        wrapper = _BatchPlaybackWrapper(observation.index_select(0, rows))
        session = FADAPlaybackSession(
            controller=FADAPlaybackController(policy, device="cpu"),
            env=_BatchPlaybackEnv(commands[rows.numpy()]),
            wrapped_env=wrapper,
            device="cpu",
            action_mode="policy",
            num_envs=2,
        )
        session.reset()
        session.step_once()
        return wrapper.actions[0], policy

    base_actions, base_policy = run(torch.tensor([0, 1], dtype=torch.int64))
    permuted_actions, permuted_policy = run(permutation)
    hand_expected = torch.tensor([[2.0, 4.0], [9.0, 10.0]])

    torch.testing.assert_close(base_actions, hand_expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        permuted_actions,
        hand_expected.index_select(0, permutation),
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(permuted_actions, base_actions)
    torch.testing.assert_close(
        permuted_policy.calls[0][0],
        base_policy.calls[0][0].index_select(0, permutation),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        permuted_policy.calls[0][1],
        base_policy.calls[0][1].index_select(0, permutation),
        rtol=0.0,
        atol=0.0,
    )


def test_fada_playback_public_controller_binding_is_policy_only() -> None:
    controller = _PlaybackController(4.0)
    session = _playback_session(action_mode="policy")

    session.bind_controller(controller)
    session.reset()
    session.step_once()

    assert session.controller is controller
    assert controller.act_calls == 1
    torch.testing.assert_close(
        session.wrapped_env.actions[0],
        torch.full((1, 2), 4.0),
    )

    non_policy = _playback_session(action_mode="zero")
    with pytest.raises(ValueError, match="action_mode=policy"):
        non_policy.bind_controller(_PlaybackController(9.0))
    assert non_policy.controller is None
    assert non_policy.policy is None


def test_context_preset_factory_consumes_context_first_action(
    monkeypatch,
) -> None:
    config_dir = Path(play_fada_context_viser.ROOT_DIR) / "conf" / "distill"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["+context_playback=left_knee_070"],
        )
    assert cfg.interactive.action_mode == "policy"

    healthy_controller = _PlaybackController(1.0)
    context_controller = _PlaybackController(7.0)
    monkeypatch.setattr(
        play_fada_context_viser,
        "_context_controller",
        lambda *_args, **_kwargs: context_controller,
    )

    def create_session(**kwargs):
        return (
            _playback_session(
                action_mode=str(kwargs["cfg"].interactive.action_mode),
                controller=healthy_controller,
            ),
            "actor",
            "healthy.pt",
        )

    monkeypatch.setattr(
        play_fada_context_viser,
        "create_fada_playback_session",
        create_session,
    )

    session, policy_obs_mode, checkpoint_path = play_fada_context_viser._session_factory(
        cfg=cfg,
        device="cpu",
    )
    session.reset()
    session.step_once()

    assert policy_obs_mode == "actor"
    assert checkpoint_path == "healthy.pt"
    assert session.controller is context_controller
    assert healthy_controller.act_calls == 0
    assert context_controller.act_calls == 1
    torch.testing.assert_close(
        session.wrapped_env.actions[0],
        torch.full((1, 2), 7.0),
    )
