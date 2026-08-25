from __future__ import annotations

from collections.abc import Callable
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


def test_idm_source_loss_selects_one_causal_pair_per_row() -> None:
    config = _config()
    idm = FADAInverseDynamicsModel(config)
    batch = _batch(config, batch_size=4)
    roles = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    batch = replace(batch, idm_source_role=roles, oracle_shadow_valid=torch.tensor([True, True, True, False]))

    selected = replace(
        batch,
        realized_future=batch.realized_future.clone(),
        executed_action_chunk=batch.executed_action_chunk.clone(),
        oracle_future=batch.oracle_future.clone(),
        oracle_action_chunk=batch.oracle_action_chunk.clone(),
    )
    selected.realized_future[0] = 11.0
    selected.executed_action_chunk[0] = 12.0
    selected.oracle_future[1] = 21.0
    selected.oracle_action_chunk[1] = 22.0
    selected.realized_future[2] = 31.0
    selected.executed_action_chunk[2] = 32.0
    selected.oracle_future[3] = 41.0
    selected.oracle_action_chunk[3] = 42.0

    changed_unused = replace(
        selected,
        oracle_future=selected.oracle_future.clone(),
        oracle_action_chunk=selected.oracle_action_chunk.clone(),
    )
    changed_unused.oracle_future[0] = -101.0
    changed_unused.oracle_action_chunk[0] = -102.0
    changed_unused.oracle_future[2] = -103.0
    changed_unused.oracle_action_chunk[2] = -104.0
    changed_unused.oracle_future[3] = -105.0
    changed_unused.oracle_action_chunk[3] = -106.0

    torch.testing.assert_close(idm_source_loss(idm, selected), idm_source_loss(idm, changed_unused))


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


def _production_horizon_config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=7,
        action_dim=3,
        command_dim=4,
        history_length=30,
        prediction_horizon=6,
        hidden_dim=16,
        num_heads=4,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
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
        idm_source_role=torch.zeros(batch_size, dtype=torch.int64),
        oracle_first_action=torch.randn(batch_size, config.action_dim),
        command_scenario=torch.zeros(batch_size, dtype=torch.int64),
        planner_eligible=torch.ones(batch_size, dtype=torch.bool),
        cold_start=torch.zeros(batch_size, dtype=torch.bool),
    )


def _with_first_nan(value: torch.Tensor) -> torch.Tensor:
    changed = value.clone()
    changed[0, 0, 0] = float("nan")
    return changed


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


def test_planner_tracker_boundary_preserves_h30_k6_and_strict_source_state() -> None:
    torch.manual_seed(11)
    config = _production_horizon_config()
    policy = FADAPlannerIDMPolicy(config).eval()
    batch = _batch(config, batch_size=2)
    planner_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    tracker_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    planner_handle = policy.planner.register_forward_pre_hook(
        lambda _module, inputs: planner_inputs.append(tuple(value.detach() for value in inputs))
    )
    tracker_handle = policy.idm.register_forward_pre_hook(
        lambda _module, inputs: tracker_inputs.append(tuple(value.detach() for value in inputs))
    )
    try:
        with torch.inference_mode():
            output = policy(
                batch.observation_history,
                batch.action_history,
                batch.command,
            )
    finally:
        tracker_handle.remove()
        planner_handle.remove()

    assert len(planner_inputs) == len(tracker_inputs) == 1
    torch.testing.assert_close(planner_inputs[0][0], batch.observation_history)
    torch.testing.assert_close(planner_inputs[0][1], batch.command)
    torch.testing.assert_close(tracker_inputs[0][0], batch.observation_history)
    torch.testing.assert_close(tracker_inputs[0][1], batch.action_history)
    torch.testing.assert_close(tracker_inputs[0][2], output.predicted_future)
    assert output.predicted_future.shape == (2, 6, config.obs_dim)
    assert output.action_chunk.shape == (2, 6, config.action_dim)

    source_state = policy.state_dict()
    restored = FADAPlannerIDMPolicy(config)
    restored.load_state_dict(source_state, strict=True)
    assert restored.state_dict().keys() == source_state.keys()
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, source_state[name], rtol=0.0, atol=0.0)


def test_planner_head_is_residual_to_latest_observation() -> None:
    config = _config()
    planner = FADAPlanner(config)
    torch.nn.init.zeros_(planner.future_head.weight)
    torch.nn.init.zeros_(planner.future_head.bias)
    batch = _batch(config)

    future = planner(batch.observation_history, batch.command)

    expected = batch.observation_history[:, -1:].expand(-1, config.prediction_horizon, -1)
    torch.testing.assert_close(future, expected)


def test_planner_public_boundary_has_a_hand_oracle_and_is_batch_permutation_covariant() -> None:
    config = _production_horizon_config()
    planner = FADAPlanner(config).eval()
    with torch.no_grad():
        planner.future_head.weight.zero_()
        residual = torch.arange(
            config.prediction_horizon * config.obs_dim,
            dtype=planner.future_head.bias.dtype,
        ).reshape(config.prediction_horizon, config.obs_dim)
        planner.future_head.bias.copy_(residual.reshape(-1))

    history = torch.arange(
        2 * config.history_length * config.obs_dim,
        dtype=torch.float32,
    ).reshape(2, config.history_length, config.obs_dim)
    command = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]],
    )
    expected = history[:, -1:, :] + residual.unsqueeze(0)

    with torch.inference_mode():
        actual = planner(history, command)
        permuted = planner(history.flip(0), command.flip(0))

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(permuted.flip(0), actual, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("history_transform", "command_transform", "message"),
    [
        (lambda value: value[:, :-1], lambda value: value, "observation_history shape mismatch"),
        (lambda value: value, lambda value: value[:, :-1], "command shape mismatch"),
        (
            _with_first_nan,
            lambda value: value,
            "observation_history must contain only finite values",
        ),
    ],
)
def test_planner_public_boundary_rejects_malformed_or_nonfinite_inputs(
    history_transform: Callable[[torch.Tensor], torch.Tensor],
    command_transform: Callable[[torch.Tensor], torch.Tensor],
    message: str,
) -> None:
    config = _config()
    batch = _batch(config, batch_size=2)

    with pytest.raises(ValueError, match=message):
        FADAPlanner(config)(
            history_transform(batch.observation_history),
            command_transform(batch.command),
        )


def test_planner_command_discriminator_detects_an_ignored_command_path() -> None:
    torch.manual_seed(17)
    config = _config()
    planner = FADAPlanner(config).eval()
    history = torch.zeros(2, config.history_length, config.obs_dim)
    command = torch.zeros(2, config.command_dim)
    command[0, 0] = 1.0
    command[1, 1] = 1.0

    with torch.inference_mode():
        command_conditioned = planner(history, command)
    assert float((command_conditioned[0] - command_conditioned[1]).abs().max()) > 1.0e-4

    with torch.no_grad():
        planner.command_embedding.weight.zero_()
    with torch.inference_mode():
        ignored_command = planner(history, command)
    torch.testing.assert_close(
        ignored_command[0],
        ignored_command[1],
        rtol=0.0,
        atol=1.0e-6,
    )


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


def test_idm_decode_latent_has_an_exact_linear_oracle_and_permutation_covariance() -> None:
    config = _production_horizon_config()
    idm = FADAInverseDynamicsModel(config).eval()
    with torch.no_grad():
        idm.action_head.weight.zero_()
        idm.action_head.bias.copy_(torch.tensor([0.5, -1.0, 2.0]))
        idm.action_head.weight[0, 0] = 2.0
        idm.action_head.weight[1, 1] = -3.0
        idm.action_head.weight[2, 2] = 0.25

    latent = torch.arange(
        2 * config.prediction_horizon * config.hidden_dim,
        dtype=torch.float32,
    ).reshape(2, config.prediction_horizon, config.hidden_dim)
    latent.requires_grad_(True)
    expected = torch.stack(
        (
            2.0 * latent[..., 0] + 0.5,
            -3.0 * latent[..., 1] - 1.0,
            0.25 * latent[..., 2] + 2.0,
        ),
        dim=-1,
    )

    actual = idm.decode_latent(latent)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    row_order = torch.tensor([1, 0])
    token_order = torch.tensor([5, 3, 1, 4, 2, 0])
    permuted = idm.decode_latent(latent.detach()[row_order][:, token_order])
    torch.testing.assert_close(
        permuted,
        actual.detach()[row_order][:, token_order],
        rtol=0.0,
        atol=0.0,
    )

    actual.sum().backward()
    assert latent.grad is not None
    expected_gradient = torch.zeros_like(latent)
    expected_gradient[..., :3] = torch.tensor([2.0, -3.0, 0.25])
    torch.testing.assert_close(latent.grad, expected_gradient, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("boundary", "message"),
    [
        ("history_length", "observation_history shape mismatch"),
        ("future_horizon", "future shape mismatch"),
        ("latent_width", "latent shape mismatch"),
        ("latent_nonfinite", "latent must contain only finite values"),
    ],
)
def test_idm_public_boundaries_reject_bad_shapes_and_nonfinite_values(
    boundary: str,
    message: str,
) -> None:
    config = _config()
    idm = FADAInverseDynamicsModel(config)
    batch = _batch(config, batch_size=2)

    with pytest.raises(ValueError, match=message):
        if boundary == "history_length":
            idm.encode_latent(
                batch.observation_history[:, :-1],
                batch.action_history,
                batch.realized_future,
            )
        elif boundary == "future_horizon":
            idm.encode_latent(
                batch.observation_history,
                batch.action_history,
                batch.realized_future[:, :-1],
            )
        else:
            latent = torch.zeros(2, config.prediction_horizon, config.hidden_dim)
            if boundary == "latent_width":
                latent = latent[..., :-1]
            else:
                latent[0, 0, 0] = float("inf")
            idm.decode_latent(latent)


def test_idm_frozen_call_sentinel_detects_a_controlled_mutation() -> None:
    config = _config()
    batch = _batch(config, batch_size=2)

    def assert_call_preserves_parameters(idm: FADAInverseDynamicsModel) -> None:
        before = {name: value.detach().clone() for name, value in idm.named_parameters()}
        idm(batch.observation_history, batch.action_history, batch.realized_future)
        for name, value in idm.named_parameters():
            torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)

    assert_call_preserves_parameters(FADAInverseDynamicsModel(config).eval())

    mutated = FADAInverseDynamicsModel(config).eval()

    def mutate_once(module: torch.nn.Module, _inputs: object, output: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next(module.parameters()).add_(1.0)
        return output

    handle = mutated.register_forward_hook(mutate_once)
    try:
        with pytest.raises(AssertionError):
            assert_call_preserves_parameters(mutated)
    finally:
        handle.remove()


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


def test_idm_source_loss_ignores_oracle_shadow_fields_for_trajectory_rows() -> None:
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
    invalid_loss = idm_source_loss(idm, invalid)
    torch.testing.assert_close(invalid_loss, idm_source_loss(idm, changed_but_invalid))


def test_idm_source_loss_uses_only_valid_oracle_shadow_rows() -> None:
    config = _config()
    idm = FADAInverseDynamicsModel(config)
    batch = replace(
        _batch(config),
        idm_source_role=torch.ones(4, dtype=torch.int64),
        oracle_shadow_valid=torch.zeros(4, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="no valid role-selected causal rows"):
        idm_source_loss(idm, batch)
    valid = replace(batch, oracle_shadow_valid=torch.ones(4, dtype=torch.bool))
    assert float(idm_source_loss(idm, valid)) >= 0.0


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
        idm_source_role=batch.idm_source_role,
        oracle_first_action=batch.oracle_first_action,
        command_scenario=batch.command_scenario,
        planner_eligible=batch.planner_eligible,
        cold_start=batch.cold_start,
    )

    with pytest.raises(ValueError, match="realized_future shape mismatch"):
        invalid.validate(config)


def test_source_batch_rejects_unknown_idm_source_role() -> None:
    config = _config()
    batch = _batch(config)
    invalid = replace(batch, idm_source_role=torch.full((4,), 9, dtype=torch.int64))

    with pytest.raises(ValueError, match="idm_source_role contains an unknown role id"):
        invalid.validate(config)
