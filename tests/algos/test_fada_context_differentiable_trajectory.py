from __future__ import annotations

import pytest
import torch

from unilab.algos.torch.distill.fada import (
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlanner,
)
from unilab.algos.torch.fada_context.differentiable_rollout import (
    DifferentiableContextRollout,
    trajectory_context_loss,
)
from unilab.algos.torch.fada_context.fault_dynamics import (
    FaultDynamicsConfig,
    FaultDynamicsEnsemble,
    FaultTransitionBatch,
    fault_dynamics_loss,
)
from unilab.algos.torch.fada_context.trajectory_context import (
    ContextEncoderConfig,
    FADATrajectoryContextEncoder,
    FrozenPlannerIDMContextPolicy,
)


def _fada_config() -> FADAArchitectureConfig:
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


def _context_config(fada: FADAArchitectureConfig) -> ContextEncoderConfig:
    return ContextEncoderConfig.from_fada(fada, hidden_dim=12, num_layers=1, residual_scale=0.2)


def _histories(
    fada: FADAArchitectureConfig,
    *,
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch_size, fada.history_length, fada.obs_dim),
        torch.randn(batch_size, fada.history_length, fada.action_dim),
        torch.randn(batch_size, fada.command_dim),
    )


def test_context_encoder_emits_bounded_planner_future_residual() -> None:
    fada = _fada_config()
    encoder = FADATrajectoryContextEncoder(_context_config(fada))
    observation_history, action_history, command = _histories(fada)

    delta_z = encoder(observation_history, action_history, command)

    assert delta_z.shape == (4, fada.prediction_horizon, fada.obs_dim)
    assert torch.isfinite(delta_z).all()
    assert float(delta_z.abs().max()) <= 0.2


def test_context_encoder_rejects_privileged_or_misaligned_inputs() -> None:
    fada = _fada_config()
    encoder = FADATrajectoryContextEncoder(_context_config(fada))
    observation_history, action_history, command = _histories(fada)

    with pytest.raises(TypeError, match="three deployable tensors"):
        encoder(observation_history, action_history, command, torch.ones(4, 1))
    with pytest.raises(ValueError, match="observation_history shape mismatch"):
        encoder(observation_history[:, :-1], action_history, command)


def test_zero_context_policy_is_exactly_the_nominal_planner_idm_path() -> None:
    torch.manual_seed(3)
    fada = _fada_config()
    planner = FADAPlanner(fada).eval()
    idm = FADAInverseDynamicsModel(fada).eval()
    context = FADATrajectoryContextEncoder(_context_config(fada)).eval()
    policy = FrozenPlannerIDMContextPolicy(planner, idm, context)
    observation_history, action_history, command = _histories(fada)

    nominal_future = planner(observation_history, command)
    nominal_chunk = idm(observation_history, action_history, nominal_future)
    output = policy(
        observation_history,
        action_history,
        command,
        delta_z=torch.zeros_like(nominal_future),
    )

    torch.testing.assert_close(output.nominal_future, nominal_future)
    torch.testing.assert_close(output.repaired_future, nominal_future)
    torch.testing.assert_close(output.action_chunk, nominal_chunk)
    assert all(not parameter.requires_grad for parameter in planner.parameters())
    assert all(not parameter.requires_grad for parameter in idm.parameters())
    assert all(parameter.requires_grad for parameter in context.parameters())

    policy.train()
    assert policy.context_encoder.training is True
    assert policy.planner.training is False
    assert policy.idm.training is False


def test_fault_dynamics_loss_trains_one_step_and_short_rollout() -> None:
    config = FaultDynamicsConfig(state_dim=7, action_dim=3, hidden_dims=(16, 16))
    ensemble = FaultDynamicsEnsemble(config, ensemble_size=3)
    state = torch.randn(5, 5, 7)
    batch = FaultTransitionBatch(
        state=state[:, :-1],
        action=torch.randn(5, 4, 3),
        next_state=state[:, 1:],
    )

    loss = fault_dynamics_loss(ensemble, batch, rollout_horizon=3)

    assert loss.one_step.ndim == 0
    assert loss.multi_step.ndim == 0
    assert loss.total.ndim == 0
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in ensemble.parameters())


def test_fault_dynamics_loss_rejects_noncontiguous_rollout_rows() -> None:
    config = FaultDynamicsConfig(state_dim=7, action_dim=3)
    ensemble = FaultDynamicsEnsemble(config, ensemble_size=2)
    batch = FaultTransitionBatch(
        state=torch.randn(2, 3, 7),
        action=torch.randn(2, 3, 3),
        next_state=torch.randn(2, 3, 7),
    )

    with pytest.raises(ValueError, match="contiguous"):
        fault_dynamics_loss(ensemble, batch, rollout_horizon=3)


def test_context_trajectory_loss_updates_only_context_through_frozen_models() -> None:
    torch.manual_seed(11)
    fada = _fada_config()
    context = FADATrajectoryContextEncoder(_context_config(fada))
    policy = FrozenPlannerIDMContextPolicy(
        FADAPlanner(fada).eval(),
        FADAInverseDynamicsModel(fada).eval(),
        context,
    )
    dynamics = FaultDynamicsEnsemble(
        FaultDynamicsConfig(state_dim=fada.obs_dim, action_dim=fada.action_dim, hidden_dims=(16,)),
        ensemble_size=3,
    )
    rollout = DifferentiableContextRollout(policy, dynamics)
    observation_history, action_history, command = _histories(fada, batch_size=2)
    reference = torch.randn(2, 4, fada.obs_dim)

    result = rollout(observation_history, action_history, command, horizon=4)
    loss = trajectory_context_loss(
        result,
        reference,
        latent_weight=0.01,
        action_smoothness_weight=0.01,
        uncertainty_weight=0.01,
    )
    loss.total.backward()

    context_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in context.parameters()
        if parameter.grad is not None
    )
    assert context_grad > 0.0
    assert all(parameter.grad is None for parameter in policy.planner.parameters())
    assert all(parameter.grad is None for parameter in policy.idm.parameters())
    assert all(parameter.grad is None for parameter in dynamics.parameters())
    assert result.predicted_trajectory.shape == (2, 4, fada.obs_dim)
    assert result.actions.shape == (2, 4, fada.action_dim)
    assert result.model_disagreement.shape == (2, 4)


def test_context_rollout_rejects_reference_horizon_mismatch() -> None:
    fada = _fada_config()
    policy = FrozenPlannerIDMContextPolicy(
        FADAPlanner(fada),
        FADAInverseDynamicsModel(fada),
        FADATrajectoryContextEncoder(_context_config(fada)),
    )
    dynamics = FaultDynamicsEnsemble(
        FaultDynamicsConfig(state_dim=fada.obs_dim, action_dim=fada.action_dim),
        ensemble_size=2,
    )
    rollout = DifferentiableContextRollout(policy, dynamics)
    observation_history, action_history, command = _histories(fada, batch_size=2)
    result = rollout(observation_history, action_history, command, horizon=3)

    with pytest.raises(ValueError, match="reference_trajectory shape mismatch"):
        trajectory_context_loss(result, torch.randn(2, 2, fada.obs_dim))
