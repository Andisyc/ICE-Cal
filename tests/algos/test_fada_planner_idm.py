from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from unilab.algos.torch.distill.fada import (
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlanner,
    FADAPlannerIDMPolicy,
    FADASourceBatch,
    first_action_mse,
    idm_source_loss,
    planner_source_loss,
)


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=7,
        action_dim=3,
        command_dim=4,
        history_length=5,
        prediction_horizon=3,
        hidden_dim=16,
        num_heads=4,
        planner_layers=2,
        idm_encoder_layers=2,
        idm_decoder_layers=2,
        feedforward_dim=32,
    )


def _batch(config: FADAArchitectureConfig, *, batch_size: int = 4) -> FADASourceBatch:
    return FADASourceBatch(
        observation_history=torch.randn(batch_size, config.history_length, config.obs_dim),
        action_history=torch.randn(batch_size, config.history_length, config.action_dim),
        command=torch.randn(batch_size, config.command_dim),
        realized_future=torch.randn(batch_size, config.prediction_horizon, config.obs_dim),
        executed_action_chunk=torch.randn(
            batch_size,
            config.prediction_horizon,
            config.action_dim,
        ),
        oracle_future=torch.randn(batch_size, config.prediction_horizon, config.obs_dim),
        oracle_action_chunk=torch.randn(
            batch_size,
            config.prediction_horizon,
            config.action_dim,
        ),
        oracle_shadow_valid=torch.ones(batch_size, dtype=torch.bool),
        oracle_first_action=torch.randn(batch_size, config.action_dim),
        command_scenario=torch.zeros(batch_size, dtype=torch.int64),
        planner_eligible=torch.ones(batch_size, dtype=torch.bool),
        cold_start=torch.zeros(batch_size, dtype=torch.bool),
    )


def test_paper_defaults_and_policy_shapes() -> None:
    paper = FADAArchitectureConfig(obs_dim=48, action_dim=29, command_dim=3)
    assert paper.history_length == 30
    assert paper.prediction_horizon == 6
    assert paper.hidden_dim == 128
    assert paper.num_heads == 4
    assert paper.planner_layers == 3
    assert paper.idm_encoder_layers == 3
    assert paper.idm_decoder_layers == 2

    config = _config()
    batch = _batch(config)
    output = FADAPlannerIDMPolicy(config)(
        batch.observation_history,
        batch.action_history,
        batch.command,
    )
    assert output.predicted_future.shape == (4, 3, 7)
    assert output.action_chunk.shape == (4, 3, 3)
    assert output.action.shape == (4, 3)
    torch.testing.assert_close(output.action, output.action_chunk[:, 0])


def test_planner_head_is_residual_to_latest_observation() -> None:
    config = _config()
    planner = FADAPlanner(config)
    torch.nn.init.zeros_(planner.future_head.weight)
    torch.nn.init.zeros_(planner.future_head.bias)
    batch = _batch(config)

    future = planner(batch.observation_history, batch.command)

    expected = batch.observation_history[:, -1:].expand(-1, config.prediction_horizon, -1)
    torch.testing.assert_close(future, expected)


def test_idm_first_action_can_depend_on_later_future_tokens() -> None:
    torch.manual_seed(7)
    config = _config()
    idm = FADAInverseDynamicsModel(config)
    batch = _batch(config, batch_size=2)
    future = batch.realized_future.clone().requires_grad_(True)

    first_action = idm(batch.observation_history, batch.action_history, future)[:, 0].sum()
    first_action.backward()

    assert future.grad is not None
    assert float(future.grad[:, 1:].abs().sum()) > 0.0


def test_first_action_loss_ignores_nonexecuted_chunk_entries() -> None:
    predicted = torch.zeros(2, 3, 4)
    target = torch.zeros_like(predicted)
    target[:, 1:] = 100.0

    loss = first_action_mse(predicted, target)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_source_losses_route_causal_and_oracle_targets_separately() -> None:
    config = _config()
    planner = FADAPlanner(config)
    idm = FADAInverseDynamicsModel(config)
    batch = _batch(config)

    idm_loss = idm_source_loss(idm, batch)
    idm_loss.backward()
    assert any(parameter.grad is not None for parameter in idm.parameters())

    planner.zero_grad(set_to_none=True)
    idm.zero_grad(set_to_none=True)
    planner_loss = planner_source_loss(planner, idm, batch)
    planner_loss.backward()
    assert any(parameter.grad is not None for parameter in planner.parameters())
    assert all(parameter.grad is None for parameter in idm.parameters())
    assert all(parameter.requires_grad for parameter in idm.parameters())


def test_idm_source_loss_uses_only_valid_oracle_shadow_rows() -> None:
    config = _config()
    idm = FADAInverseDynamicsModel(config)
    batch = _batch(config)
    invalid = replace(
        batch,
        oracle_shadow_valid=torch.zeros_like(batch.oracle_shadow_valid),
        oracle_action_chunk=torch.full_like(batch.oracle_action_chunk, 1_000.0),
    )
    changed_but_invalid = replace(
        invalid,
        oracle_action_chunk=torch.full_like(batch.oracle_action_chunk, -1_000.0),
    )
    valid = replace(invalid, oracle_shadow_valid=torch.ones_like(batch.oracle_shadow_valid))

    invalid_loss = idm_source_loss(idm, invalid)
    torch.testing.assert_close(invalid_loss, idm_source_loss(idm, changed_but_invalid))
    assert float(idm_source_loss(idm, valid)) > float(invalid_loss)


def test_source_batch_rejects_noncausal_shape_mismatch() -> None:
    config = _config()
    batch = _batch(config)
    invalid = FADASourceBatch(
        observation_history=batch.observation_history,
        action_history=batch.action_history,
        command=batch.command,
        realized_future=batch.realized_future[:, :-1],
        executed_action_chunk=batch.executed_action_chunk,
        oracle_future=batch.oracle_future,
        oracle_action_chunk=batch.oracle_action_chunk,
        oracle_shadow_valid=batch.oracle_shadow_valid,
        oracle_first_action=batch.oracle_first_action,
        command_scenario=batch.command_scenario,
        planner_eligible=batch.planner_eligible,
        cold_start=batch.cold_start,
    )

    with pytest.raises(ValueError, match="realized_future shape mismatch"):
        invalid.validate(config)
