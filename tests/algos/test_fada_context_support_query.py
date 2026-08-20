from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill.fada import (
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlanner,
    FADAPlannerIDMPolicy,
)
from unilab.algos.torch.fada_context import support_query_training as training_module
from unilab.algos.torch.fada_context.support_query import (
    FADA_CONTEXT_METHOD_CONTRACT_ID,
    ContextActionOutput,
    ContextQueryBatch,
    FADASupportContextEncoder,
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
from unilab.algos.torch.fada_context.support_query_runtime import sha256_file
from unilab.algos.torch.fada_context.support_query_training import (
    SupportQueryTrainingLoopConfig,
    prepare_context_support_query_artifact,
    prepare_support_query_training,
    require_fresh_support_query_run_paths,
    resume_context_support_query_training,
    run_support_query_training,
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

STRUCTURAL_IDENTITY_MUTATIONS = [
    ("fada_architecture", "history_length"),
    ("fada_architecture", "prediction_horizon"),
    ("fada_architecture", "obs_dim"),
    ("fada_architecture", "action_dim"),
    ("fada_architecture", "hidden_dim"),
    ("context_config", "support_length"),
    ("context_config", "context_hidden_dim"),
    ("context_config", "context_layers"),
]


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


def _batch(
    config: FADAArchitectureConfig, *, batch_size: int = 4, support_length: int = 8
) -> SupportQueryBatch:
    command = torch.full((batch_size, config.command_dim), 0.4)
    windows = 4
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
                batch_size, windows, config.history_length, config.obs_dim
            ),
            action_history=torch.randn(
                batch_size, windows, config.history_length, config.action_dim
            ),
            command=command[:, None].expand(-1, windows, -1).clone(),
            planner_intent=torch.randn(
                batch_size, windows, config.prediction_horizon, config.obs_dim
            ),
            realized_future=torch.randn(
                batch_size, windows, config.prediction_horizon, config.obs_dim
            ),
            executed_action=torch.randn(batch_size, windows, config.action_dim),
            window_anchor=torch.arange(
                config.history_length - 1,
                config.history_length - 1 + windows,
                dtype=torch.int64,
            )[None]
            .expand(batch_size, -1)
            .clone(),
            valid_window_mask=torch.ones(batch_size, windows, dtype=torch.bool),
        ),
        support_command=command,
        pair_id=torch.arange(batch_size, dtype=torch.int64),
        support_rollout_id=torch.arange(batch_size, dtype=torch.int64) * 2,
        query_rollout_id=torch.arange(batch_size, dtype=torch.int64) * 2 + 1,
    ).validate(config, support_length=support_length)


def _tagged_tensor(shape: tuple[int, ...], tag: float) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    return torch.arange(count, dtype=torch.float32).reshape(shape) / 1000.0 + tag


def _semantic_batch(
    config: FADAArchitectureConfig,
    *,
    pairs: int = 2,
    windows: int = 3,
    support_length: int = 8,
) -> SupportQueryBatch:
    command = _tagged_tensor((pairs, config.command_dim), 7.0)
    valid = torch.zeros(pairs, windows, dtype=torch.bool)
    valid[torch.arange(pairs), torch.arange(pairs) % windows] = True
    anchors = (
        torch.arange(
            config.history_length - 1,
            config.history_length - 1 + windows,
            dtype=torch.int64,
        )[None]
        .expand(pairs, -1)
        .clone()
    )
    return SupportQueryBatch(
        support=SupportContextBatch(
            target_future=_tagged_tensor(
                (pairs, support_length, config.prediction_horizon, config.obs_dim),
                1.0,
            ),
            realized_state=_tagged_tensor(
                (pairs, support_length, config.obs_dim),
                2.0,
            ),
            executed_action=_tagged_tensor(
                (pairs, support_length, config.action_dim),
                3.0,
            ),
        ),
        query=ContextQueryBatch(
            observation_history=_tagged_tensor(
                (pairs, windows, config.history_length, config.obs_dim),
                4.0,
            ),
            action_history=_tagged_tensor(
                (pairs, windows, config.history_length, config.action_dim),
                5.0,
            ),
            command=command[:, None].expand(-1, windows, -1).clone(),
            planner_intent=_tagged_tensor(
                (pairs, windows, config.prediction_horizon, config.obs_dim),
                8.0,
            ),
            realized_future=_tagged_tensor(
                (pairs, windows, config.prediction_horizon, config.obs_dim),
                9.0,
            ),
            executed_action=_tagged_tensor(
                (pairs, windows, config.action_dim),
                10.0,
            ),
            window_anchor=anchors,
            valid_window_mask=valid,
        ),
        support_command=command,
        pair_id=torch.arange(11, 11 + pairs, dtype=torch.int64),
        support_rollout_id=torch.arange(101, 101 + 2 * pairs, 2, dtype=torch.int64),
        query_rollout_id=torch.arange(102, 102 + 2 * pairs, 2, dtype=torch.int64),
    )


def _named_batch_tensors(batch: SupportQueryBatch) -> dict[str, torch.Tensor]:
    return {
        "support_target_future": batch.support.target_future,
        "support_realized_state": batch.support.realized_state,
        "support_executed_action": batch.support.executed_action,
        "query_observation_history": batch.query.observation_history,
        "query_action_history": batch.query.action_history,
        "query_command": batch.query.command,
        "query_planner_intent": batch.query.planner_intent,
        "query_realized_future": batch.query.realized_future,
        "query_executed_action": batch.query.executed_action,
        "query_window_anchor": batch.query.window_anchor,
        "query_valid_window_mask": batch.query.valid_window_mask,
        "support_command": batch.support_command,
        "pair_id": batch.pair_id,
        "support_rollout_id": batch.support_rollout_id,
        "query_rollout_id": batch.query_rollout_id,
    }


def _install_deterministic_nonzero_query_path(
    encoder: FADASupportContextEncoder,
    *,
    query_feature_index: int,
) -> None:
    """Install one controlled query-history Jacobian without changing production init."""

    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
        query_projection = getattr(encoder, "query_frame_projection", None)
        query_encoder = getattr(encoder, "query_sequence_encoder", None)
        if query_projection is None or query_encoder is None:
            return
        hidden = encoder.context_config.context_hidden_dim
        query_projection.weight[0, query_feature_index] = 1.0
        query_encoder.weight_ih_l0[2 * hidden, 0] = 1.0
        encoder.delta_head.weight[0, hidden] = 1.0


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
        batch.query.observation_history.flatten(0, 1),
        batch.query.action_history.flatten(0, 1),
        batch.query.realized_future.flatten(0, 1),
    ).unflatten(0, (batch.batch_size, batch.query.window_count))

    torch.testing.assert_close(output.delta_z, torch.zeros_like(output.delta_z))
    torch.testing.assert_close(output.action_chunk, nominal, rtol=0.0, atol=0.0)


def test_fresh_context_encoder_emits_exact_zero_for_current_query() -> None:
    config = _config()
    batch = _batch(config, batch_size=2)
    encoder = FADASupportContextEncoder(
        config,
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
    ).eval()

    with torch.inference_mode():
        delta_z = encoder(
            batch.support,
            batch.query.observation_history[:, 0],
            batch.query.action_history[:, 0],
        )

    assert delta_z.shape == (2, config.hidden_dim)
    torch.testing.assert_close(delta_z, torch.zeros_like(delta_z), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("query_role", ["state", "action"])
def test_context_delta_depends_on_current_query_histories(query_role: str) -> None:
    config = _config()
    batch = _batch(config, batch_size=2)
    encoder = FADASupportContextEncoder(
        config,
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
    ).eval()
    feature_index = 0 if query_role == "state" else config.obs_dim
    _install_deterministic_nonzero_query_path(
        encoder,
        query_feature_index=feature_index,
    )
    state_a = torch.zeros(2, config.history_length, config.obs_dim)
    state_b = state_a.clone()
    action_a = torch.zeros(2, config.history_length, config.action_dim)
    action_b = action_a.clone()
    if query_role == "state":
        state_b[:, -1, 0] = 1.0
    else:
        action_b[:, -1, 0] = 1.0

    with torch.inference_mode():
        delta_a = encoder(batch.support, state_a, action_a)
        delta_b = encoder(batch.support, state_b, action_b)

    assert delta_a.shape == (2, config.hidden_dim)
    assert not torch.allclose(delta_a, delta_b)


def test_context_rejects_incomplete_support_and_query_history_mismatch() -> None:
    config = _config()
    batch = _batch(config, batch_size=2)
    encoder = FADASupportContextEncoder(
        config,
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
    )
    state = batch.query.observation_history[:, 0]
    action = batch.query.action_history[:, 0]

    with pytest.raises(ValueError, match="support length mismatch"):
        encoder(
            SupportContextBatch(*(value[:, :-1] for value in batch.support.tensors())),
            state,
            action,
        )
    with pytest.raises(ValueError, match="batch sizes must match"):
        encoder(batch.support, state[:1], action[:1])
    with pytest.raises(ValueError, match="action_history shape mismatch"):
        encoder(batch.support, state, action[:, :-1])
    with pytest.raises(ValueError, match="share one dtype"):
        encoder(batch.support, state.double(), action.double())
    with pytest.raises(ValueError, match="share one device"):
        encoder(batch.support, state.to("meta"), action.to("meta"))
    invalid_state = state.clone()
    invalid_state[0, -1, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        encoder(batch.support, invalid_state, action)


def test_context_encoder_pair_permutation_and_equal_width_role_discriminator() -> None:
    torch.manual_seed(101)
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    encoder = FADASupportContextEncoder(
        config,
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
    ).eval()
    with torch.no_grad():
        encoder.delta_head.weight.fill_(0.01)
    state = batch.query.observation_history[:, 0]
    action = batch.query.action_history[:, 0]
    order = torch.tensor([1, 0])

    original = encoder(batch.support, state, action)
    permuted = encoder(
        batch.support.index_select(order),
        state.index_select(0, order),
        action.index_select(0, order),
    )

    torch.testing.assert_close(
        permuted,
        original.index_select(0, order),
        rtol=0.0,
        atol=0.0,
    )

    equal_width = replace(config, obs_dim=config.action_dim)
    equal_batch = _semantic_batch(equal_width).validate(equal_width, support_length=8)
    role_encoder = FADASupportContextEncoder(
        equal_width,
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
    ).eval()
    _install_deterministic_nonzero_query_path(role_encoder, query_feature_index=0)
    state_role = torch.zeros(2, equal_width.history_length, equal_width.obs_dim)
    state_role[:, -1, 0] = 1.0
    action_role = torch.zeros(2, equal_width.history_length, equal_width.action_dim)

    state_tagged = role_encoder(equal_batch.support, state_role, action_role)
    action_tagged = role_encoder(equal_batch.support, action_role, state_role)

    assert not torch.equal(state_tagged, action_tagged)


def test_context_encoder_repeated_forward_is_bitwise_pure() -> None:
    torch.manual_seed(102)
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    encoder = FADASupportContextEncoder(
        config,
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
    ).eval()
    with torch.no_grad():
        encoder.delta_head.weight.fill_(0.01)
    support_before = tuple(value.clone() for value in batch.support.tensors())
    state = batch.query.observation_history[:, 0].clone()
    action = batch.query.action_history[:, 0].clone()
    state_before = state.clone()
    action_before = action.clone()
    parameters_before = {
        name: value.detach().clone() for name, value in encoder.state_dict().items()
    }

    first = encoder(batch.support, state, action)
    second = encoder(batch.support, state, action)

    assert torch.equal(first, second)
    for observed, expected in zip(batch.support.tensors(), support_before, strict=True):
        assert torch.equal(observed, expected)
    assert torch.equal(state, state_before)
    assert torch.equal(action, action_before)
    for name, value in encoder.state_dict().items():
        assert torch.equal(value, parameters_before[name])


def test_context_encoder_saturates_at_configured_delta_scale() -> None:
    config = _config()
    context_config = SupportQueryContextConfig(
        support_length=8,
        context_hidden_dim=12,
        context_layers=1,
        delta_scale=0.2,
    )
    encoder = FADASupportContextEncoder(config, context_config).eval()
    batch = _semantic_batch(config).validate(config, support_length=8)
    signed_bias = torch.where(
        torch.arange(config.hidden_dim) % 2 == 0,
        torch.tensor(100.0),
        torch.tensor(-100.0),
    )
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
        encoder.delta_head.bias.copy_(signed_bias)
    expected = context_config.delta_scale * torch.tanh(signed_bias)

    observed = encoder(
        batch.support,
        batch.query.observation_history[:, 0],
        batch.query.action_history[:, 0],
    )

    torch.testing.assert_close(
        observed,
        expected[None].expand(batch.batch_size, -1),
        rtol=0.0,
        atol=0.0,
    )
    assert bool((observed.abs() <= observed.new_tensor(context_config.delta_scale)).all())
    with pytest.raises(ValueError, match="delta_scale must be in"):
        replace(context_config, delta_scale=0.0)
    with pytest.raises(ValueError, match="delta_scale must be in"):
        replace(context_config, delta_scale=1.01)


def test_reconstruct_query_gathers_only_valid_rows_with_complete_support_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(4)
    config = replace(_config(), prediction_horizon=6)
    original = _batch(config, batch_size=2)
    valid_window_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, False]],
        dtype=torch.bool,
    )
    observation_history = torch.zeros_like(original.query.observation_history)
    action_history = torch.zeros_like(original.query.action_history)
    observation_history[0, 2, -1, 0] = 1.0
    observation_history[~valid_window_mask] = 10_000.0
    action_history[~valid_window_mask] = -10_000.0
    realized_future = original.query.realized_future.clone()
    realized_future[~valid_window_mask] = 20_000.0
    support = SupportContextBatch(
        target_future=original.support.target_future.clone(),
        realized_state=original.support.realized_state.clone(),
        executed_action=original.support.executed_action.clone(),
    )
    support.target_future[0].fill_(11.0)
    support.target_future[1].fill_(22.0)
    batch = replace(
        original,
        support=support,
        query=replace(
            original.query,
            observation_history=observation_history,
            action_history=action_history,
            realized_future=realized_future,
            valid_window_mask=valid_window_mask,
        ),
    ).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    _install_deterministic_nonzero_query_path(
        setup.policy.context_encoder,
        query_feature_index=0,
    )
    context_inputs: list[tuple[object, ...]] = []
    tracker_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    original_encode_latent = setup.policy.idm.encode_latent

    def record_tracker_inputs(
        observation: torch.Tensor,
        action: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        tracker_inputs.append((observation.detach(), action.detach(), future.detach()))
        return original_encode_latent(observation, action, future)

    monkeypatch.setattr(setup.policy.idm, "encode_latent", record_tracker_inputs)
    handle = setup.policy.context_encoder.register_forward_pre_hook(
        lambda _module, inputs: context_inputs.append(inputs)
    )
    try:
        output = setup.policy.reconstruct_query(batch)
    finally:
        handle.remove()

    pair_indices, window_indices = torch.nonzero(valid_window_mask, as_tuple=True)
    assert len(context_inputs) == len(tracker_inputs) == 1
    recorded_support, recorded_state, recorded_action = context_inputs[0]
    assert isinstance(recorded_support, SupportContextBatch)
    assert recorded_support.support_length == batch.support.support_length == 8
    torch.testing.assert_close(
        recorded_support.target_future,
        batch.support.target_future.index_select(0, pair_indices),
    )
    torch.testing.assert_close(recorded_state, observation_history[pair_indices, window_indices])
    torch.testing.assert_close(recorded_action, action_history[pair_indices, window_indices])
    torch.testing.assert_close(
        tracker_inputs[0][0],
        observation_history[pair_indices, window_indices],
    )
    torch.testing.assert_close(
        tracker_inputs[0][1],
        action_history[pair_indices, window_indices],
    )
    assert output.delta_z.shape == (2, 4, config.hidden_dim)
    assert output.query_latent.shape == (2, 4, 6, config.hidden_dim)
    assert output.action_chunk.shape == (2, 4, 6, config.action_dim)
    assert torch.equal(
        output.delta_z[~valid_window_mask], torch.zeros_like(output.delta_z[~valid_window_mask])
    )
    assert torch.equal(
        output.action_chunk[~valid_window_mask],
        torch.zeros_like(output.action_chunk[~valid_window_mask]),
    )
    assert not torch.allclose(output.delta_z[0, 0], output.delta_z[0, 2])
    assert torch.equal(output.action, output.action_chunk[..., 0, :])


def test_reconstruct_query_singleton_shapes() -> None:
    config = _config()
    batch = _semantic_batch(config, pairs=1, windows=1).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )

    output = setup.policy.reconstruct_query(batch)

    assert output.delta_z.shape == (1, 1, config.hidden_dim)
    assert output.query_latent.shape == (
        1,
        1,
        config.prediction_horizon,
        config.hidden_dim,
    )
    assert output.repaired_latent.shape == output.query_latent.shape
    assert output.action_chunk.shape == (
        1,
        1,
        config.prediction_horizon,
        config.action_dim,
    )
    assert output.action.shape == (1, 1, config.action_dim)


def test_reconstruct_query_rejects_invalid_mask_and_rollout_ownership_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    monkeypatch.setattr(
        setup.policy.context_encoder,
        "forward",
        lambda *_args, **_kwargs: pytest.fail("invalid batch reached Context forward"),
    )
    zero_valid = replace(
        batch,
        query=replace(
            batch.query,
            valid_window_mask=torch.zeros_like(batch.query.valid_window_mask),
        ),
    )
    role_copy = replace(batch, query_rollout_id=batch.support_rollout_id.clone())

    with pytest.raises(ValueError, match="at least one valid window"):
        setup.policy.reconstruct_query(zero_valid)
    with pytest.raises(ValueError, match="different rollout ids"):
        setup.policy.reconstruct_query(role_copy)


def test_reconstruct_query_pair_window_permutation_and_repeat_are_covariant_and_pure() -> None:
    torch.manual_seed(103)
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    pair_order = torch.tensor([1, 0])
    window_order = torch.tensor([2, 0, 1])
    inverse_pair = torch.argsort(pair_order)
    inverse_window = torch.argsort(window_order)
    query_values = {
        name: value.index_select(0, pair_order).index_select(1, window_order)
        for name, value in zip(
            (
                "observation_history",
                "action_history",
                "command",
                "planner_intent",
                "realized_future",
                "executed_action",
                "window_anchor",
                "valid_window_mask",
            ),
            batch.query.tensors(),
            strict=True,
        )
    }
    permuted_batch = SupportQueryBatch(
        support=batch.support.index_select(pair_order),
        query=ContextQueryBatch(**query_values),
        support_command=batch.support_command.index_select(0, pair_order),
        pair_id=batch.pair_id.index_select(0, pair_order),
        support_rollout_id=batch.support_rollout_id.index_select(0, pair_order),
        query_rollout_id=batch.query_rollout_id.index_select(0, pair_order),
    ).validate(config, support_length=8)
    support_before = tuple(value.clone() for value in batch.support.tensors())
    parameters_before = {
        name: value.detach().clone() for name, value in setup.policy.state_dict().items()
    }

    original = setup.policy.reconstruct_query(batch)
    repeated = setup.policy.reconstruct_query(batch)
    permuted = setup.policy.reconstruct_query(permuted_batch)

    for name in ("delta_z", "query_latent", "repaired_latent", "action_chunk"):
        expected = getattr(original, name)
        assert torch.equal(getattr(repeated, name), expected)
        restored = (
            getattr(permuted, name).index_select(0, inverse_pair).index_select(1, inverse_window)
        )
        torch.testing.assert_close(restored, expected, rtol=1.0e-5, atol=1.0e-6)
    for observed, expected in zip(batch.support.tensors(), support_before, strict=True):
        assert torch.equal(observed, expected)
    for name, value in setup.policy.state_dict().items():
        assert torch.equal(value, parameters_before[name])


def test_reconstruct_query_padded_rows_have_zero_input_and_output_gradient() -> None:
    torch.manual_seed(104)
    config = _config()
    base = _semantic_batch(config).validate(config, support_length=8)
    observation = base.query.observation_history.clone().requires_grad_(True)
    action = base.query.action_history.clone().requires_grad_(True)
    future = base.query.realized_future.clone().requires_grad_(True)
    batch = replace(
        base,
        query=replace(
            base.query,
            observation_history=observation,
            action_history=action,
            realized_future=future,
        ),
    )
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )

    output = setup.policy.reconstruct_query(batch)
    output.action_chunk.retain_grad()
    (output.action_chunk.square().sum() + output.delta_z.square().sum()).backward()

    invalid = ~batch.query.valid_window_mask
    assert output.action_chunk.grad is not None
    assert torch.equal(
        output.action_chunk.grad[invalid],
        torch.zeros_like(output.action_chunk.grad[invalid]),
    )
    for value in (observation, action, future):
        assert value.grad is not None
        assert torch.equal(value.grad[invalid], torch.zeros_like(value.grad[invalid]))


def test_act_with_context_recomputes_planner_and_query_conditioned_delta() -> None:
    torch.manual_seed(41)
    config = replace(_config(), prediction_horizon=6)
    batch = _batch(config, batch_size=2)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    _install_deterministic_nonzero_query_path(
        setup.policy.context_encoder,
        query_feature_index=0,
    )
    state = torch.zeros(2, config.history_length, config.obs_dim)
    action = torch.zeros(2, config.history_length, config.action_dim)
    planner_inputs: list[tuple[object, ...]] = []
    context_inputs: list[tuple[object, ...]] = []
    planner_handle = setup.policy.planner.register_forward_pre_hook(
        lambda _module, inputs: planner_inputs.append(inputs)
    )
    context_handle = setup.policy.context_encoder.register_forward_pre_hook(
        lambda _module, inputs: context_inputs.append(inputs)
    )
    try:
        first = setup.policy.act_with_context(
            batch.support,
            state,
            action,
            batch.support_command,
        )
        changed_state = state.clone()
        changed_state[:, -1, 0] = 1.0
        second = setup.policy.act_with_context(
            batch.support,
            changed_state,
            action,
            batch.support_command,
        )
    finally:
        context_handle.remove()
        planner_handle.remove()

    assert len(planner_inputs) == len(context_inputs) == 2
    torch.testing.assert_close(planner_inputs[0][0], state)
    torch.testing.assert_close(planner_inputs[0][1], batch.support_command)
    assert context_inputs[0][0] is batch.support
    torch.testing.assert_close(context_inputs[0][1], state)
    torch.testing.assert_close(context_inputs[0][2], action)
    assert not torch.allclose(first.delta_z, second.delta_z)
    assert first.action_chunk.shape == (2, 6, config.action_dim)
    assert torch.equal(first.action, first.action_chunk[:, 0, :])


def test_act_with_context_zero_residual_matches_hand_linear_tracker_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    batch = _semantic_batch(config, pairs=1, windows=1).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    state = batch.query.observation_history[:, 0]
    action = batch.query.action_history[:, 0]
    command = batch.support_command

    def planner_answer(observation: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        return observation[:, -1:, :].expand(-1, config.prediction_horizon, -1) + cmd[
            :, :1
        ].unsqueeze(-1)

    def tracker_answer(
        observation: torch.Tensor,
        action_history: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        scalar = observation[:, -1, 0, None] + action_history[:, -1, 0, None] + future[:, :, 0]
        offsets = torch.arange(config.hidden_dim, dtype=scalar.dtype)[None, None, :]
        return scalar[:, :, None] + offsets

    def decoder_answer(latent: torch.Tensor) -> torch.Tensor:
        return 2.0 * latent[..., : config.action_dim] + 1.0

    monkeypatch.setattr(setup.policy.planner, "forward", planner_answer)
    monkeypatch.setattr(setup.policy.idm, "encode_latent", tracker_answer)
    monkeypatch.setattr(setup.policy.idm, "decode_latent", decoder_answer)
    expected_future = planner_answer(state, command)
    expected_latent = tracker_answer(state, action, expected_future)
    expected_chunk = decoder_answer(expected_latent)

    output = setup.policy.act_with_context(
        batch.support,
        state,
        action,
        command,
    )

    assert torch.equal(output.delta_z, torch.zeros_like(output.delta_z))
    torch.testing.assert_close(output.query_latent, expected_latent, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output.repaired_latent, expected_latent, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output.action_chunk, expected_chunk, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output.action, expected_chunk[:, 0], rtol=0.0, atol=0.0)


def test_act_with_context_pair_permutation_covariance() -> None:
    torch.manual_seed(105)
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    with torch.no_grad():
        setup.policy.context_encoder.delta_head.weight.fill_(0.01)
    state = batch.query.observation_history[:, 0]
    action = batch.query.action_history[:, 0]
    order = torch.tensor([1, 0])

    original = setup.policy.act_with_context(
        batch.support,
        state,
        action,
        batch.support_command,
    )
    permuted = setup.policy.act_with_context(
        batch.support.index_select(order),
        state.index_select(0, order),
        action.index_select(0, order),
        batch.support_command.index_select(0, order),
    )

    for name in ("delta_z", "query_latent", "repaired_latent", "action_chunk"):
        torch.testing.assert_close(
            getattr(permuted, name),
            getattr(original, name).index_select(0, order),
            rtol=1.0e-5,
            atol=1.0e-6,
        )


def test_act_with_context_malformed_support_fails_before_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    batch = _semantic_batch(config, pairs=1, windows=1).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    malformed_support = SupportContextBatch(*(value[:, :-1] for value in batch.support.tensors()))
    decoder_calls = 0

    def record_decoder(*_args, **_kwargs):
        nonlocal decoder_calls
        decoder_calls += 1
        pytest.fail("malformed Support reached action decoder")

    monkeypatch.setattr(setup.policy.idm, "decode_latent", record_decoder)

    with pytest.raises(ValueError, match="support length mismatch"):
        setup.policy.act_with_context(
            malformed_support,
            batch.query.observation_history[:, 0],
            batch.query.action_history[:, 0],
            batch.support_command,
        )

    assert decoder_calls == 0


def test_context_loss_supervises_only_executed_first_action_and_updates_context() -> None:
    torch.manual_seed(5)
    config = _config()
    batch = _batch(config)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    changed_action = batch.query.executed_action.clone()
    changed_action[:, -1] += 1000.0
    changed_mask = batch.query.valid_window_mask.clone()
    changed_mask[:, -1] = False
    baseline_mask = changed_mask.clone()
    baseline = replace(batch, query=replace(batch.query, valid_window_mask=baseline_mask))
    changed = replace(
        batch,
        query=replace(
            batch.query,
            executed_action=changed_action,
            valid_window_mask=changed_mask,
        ),
    )

    loss = context_first_action_loss(setup.policy, baseline)
    torch.testing.assert_close(context_first_action_loss(setup.policy, changed), loss)
    loss.backward()

    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in setup.policy.context_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in setup.policy.planner.parameters())
    assert all(parameter.grad is None for parameter in setup.policy.idm.parameters())


def test_context_loss_weights_pairs_equally_with_unequal_valid_window_counts() -> None:
    torch.manual_seed(6)
    config = _config()
    batch = _batch(config, batch_size=2)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    predicted_action = setup.policy.reconstruct_query(batch).action.detach()
    error_offset = torch.tensor([1.0, 3.0])[:, None, None]
    executed_action = predicted_action + error_offset
    valid_window_mask = torch.tensor([[True, True, True, True], [True, False, False, False]])
    unequal_windows = replace(
        batch,
        query=replace(
            batch.query,
            executed_action=executed_action,
            valid_window_mask=valid_window_mask,
        ),
    )

    loss = context_first_action_loss(setup.policy, unequal_windows)

    torch.testing.assert_close(loss, torch.tensor(5.0))


@pytest.mark.parametrize("invalid_kind", ["zero_valid", "nan_target"])
def test_context_first_action_loss_rejects_invalid_batch_before_reduction(
    invalid_kind: str,
) -> None:
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    if invalid_kind == "zero_valid":
        batch = replace(
            batch,
            query=replace(
                batch.query,
                valid_window_mask=torch.zeros_like(batch.query.valid_window_mask),
            ),
        )
        message = "at least one valid window"
    else:
        target = batch.query.executed_action.clone()
        target[0, 0, 0] = torch.nan
        batch = replace(batch, query=replace(batch.query, executed_action=target))
        message = "finite"

    with pytest.raises(ValueError, match=message):
        context_first_action_loss(setup.policy, batch)


def test_context_first_action_loss_pair_window_permutation_preserves_loss_and_gradients() -> None:
    config = _config()
    base = _semantic_batch(config).validate(config, support_length=8)
    observation = base.query.observation_history.clone().requires_grad_(True)
    batch = replace(
        base,
        query=replace(base.query, observation_history=observation),
    )
    scale = torch.nn.Parameter(torch.tensor(0.25))

    class _ControlledPolicy:
        def reconstruct_query(self, value: SupportQueryBatch) -> ContextActionOutput:
            predicted = value.query.observation_history[..., -1, 0] * scale
            action_chunk = predicted[:, :, None, None].expand(
                -1,
                -1,
                config.prediction_horizon,
                config.action_dim,
            )
            return ContextActionOutput(
                delta_z=torch.zeros(
                    value.batch_size,
                    value.query.window_count,
                    config.hidden_dim,
                ),
                query_latent=torch.zeros(
                    value.batch_size,
                    value.query.window_count,
                    config.prediction_horizon,
                    config.hidden_dim,
                ),
                repaired_latent=torch.zeros(
                    value.batch_size,
                    value.query.window_count,
                    config.prediction_horizon,
                    config.hidden_dim,
                ),
                action_chunk=action_chunk,
            )

    policy = _ControlledPolicy()
    original_loss = context_first_action_loss(policy, batch)  # type: ignore[arg-type]
    original_loss.backward()
    original_input_grad = observation.grad.detach().clone()
    assert scale.grad is not None
    original_scale_grad = scale.grad.detach().clone()

    pair_order = torch.tensor([1, 0])
    window_order = torch.tensor([2, 0, 1])
    permuted_observation = (
        base.query.observation_history.index_select(0, pair_order)
        .index_select(1, window_order)
        .clone()
        .requires_grad_(True)
    )
    query_values = {
        name: value.index_select(0, pair_order).index_select(1, window_order)
        for name, value in zip(
            (
                "action_history",
                "command",
                "planner_intent",
                "realized_future",
                "executed_action",
                "window_anchor",
                "valid_window_mask",
            ),
            base.query.tensors()[1:],
            strict=True,
        )
    }
    permuted_batch = SupportQueryBatch(
        support=base.support.index_select(pair_order),
        query=ContextQueryBatch(
            observation_history=permuted_observation,
            **query_values,
        ),
        support_command=base.support_command.index_select(0, pair_order),
        pair_id=base.pair_id.index_select(0, pair_order),
        support_rollout_id=base.support_rollout_id.index_select(0, pair_order),
        query_rollout_id=base.query_rollout_id.index_select(0, pair_order),
    ).validate(config, support_length=8)
    scale.grad = None

    permuted_loss = context_first_action_loss(  # type: ignore[arg-type]
        policy,
        permuted_batch,
    )
    permuted_loss.backward()

    torch.testing.assert_close(permuted_loss, original_loss, rtol=1.0e-6, atol=1.0e-6)
    assert permuted_observation.grad is not None
    restored_grad = permuted_observation.grad.index_select(
        0, torch.argsort(pair_order)
    ).index_select(1, torch.argsort(window_order))
    torch.testing.assert_close(restored_grad, original_input_grad, rtol=0.0, atol=0.0)
    assert scale.grad is not None
    torch.testing.assert_close(scale.grad, original_scale_grad, rtol=1.0e-6, atol=1.0e-6)


def test_context_first_action_loss_repeated_zero_grad_backward_has_no_stale_state() -> None:
    torch.manual_seed(106)
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    state_before = {
        name: value.detach().clone() for name, value in setup.policy.state_dict().items()
    }

    gradients = []
    losses = []
    for _ in range(2):
        setup.optimizer.zero_grad(set_to_none=True)
        loss = context_first_action_loss(setup.policy, batch)
        loss.backward()
        losses.append(loss.detach().clone())
        gradients.append(
            tuple(
                parameter.grad.detach().clone()
                for parameter in setup.policy.context_encoder.parameters()
                if parameter.grad is not None
            )
        )

    assert torch.equal(losses[0], losses[1])
    assert len(gradients[0]) == len(gradients[1]) > 0
    for first, second in zip(*gradients, strict=True):
        assert torch.equal(first, second)
    for name, value in setup.policy.state_dict().items():
        assert torch.equal(value, state_before[name])


def test_first_action_loss_gives_nonexecuted_tail_exactly_zero_gradient() -> None:
    config = replace(_config(), prediction_horizon=6)
    batch = _batch(config, batch_size=2)
    valid_window_mask = torch.tensor(
        [[True, True, True, True], [True, False, False, False]],
        dtype=torch.bool,
    )
    target = torch.zeros_like(batch.query.executed_action)
    action_chunk = torch.zeros(
        2,
        batch.query.window_count,
        config.prediction_horizon,
        config.action_dim,
        requires_grad=True,
    )
    with torch.no_grad():
        action_chunk[0, :, 0].fill_(1.0)
        action_chunk[1, :, 0].fill_(3.0)
        action_chunk[..., 1:, :].fill_(1000.0)
    output = ContextActionOutput(
        delta_z=torch.zeros(2, batch.query.window_count, config.hidden_dim),
        query_latent=torch.zeros(
            2,
            batch.query.window_count,
            config.prediction_horizon,
            config.hidden_dim,
        ),
        repaired_latent=torch.zeros(
            2,
            batch.query.window_count,
            config.prediction_horizon,
            config.hidden_dim,
        ),
        action_chunk=action_chunk,
    )

    class _FixedOutputPolicy:
        def reconstruct_query(self, _batch: SupportQueryBatch) -> ContextActionOutput:
            return output

    weighted_batch = replace(
        batch,
        query=replace(
            batch.query,
            executed_action=target,
            valid_window_mask=valid_window_mask,
        ),
    )
    loss = context_first_action_loss(_FixedOutputPolicy(), weighted_batch)  # type: ignore[arg-type]
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(5.0))
    assert action_chunk.grad is not None
    assert torch.equal(
        action_chunk.grad[..., 1:, :],
        torch.zeros_like(action_chunk.grad[..., 1:, :]),
    )


def test_support_query_validate_seals_full_asymmetric_single_valid_contract() -> None:
    config = _config()
    raw = _semantic_batch(config, pairs=2, windows=3, support_length=8)

    validated = raw.validate(config, support_length=8)

    assert validated is raw
    expected_shapes = {
        "support_target_future": (2, 8, 3, 7),
        "support_realized_state": (2, 8, 7),
        "support_executed_action": (2, 8, 3),
        "query_observation_history": (2, 3, 5, 7),
        "query_action_history": (2, 3, 5, 3),
        "query_command": (2, 3, 2),
        "query_planner_intent": (2, 3, 3, 7),
        "query_realized_future": (2, 3, 3, 7),
        "query_executed_action": (2, 3, 3),
        "query_window_anchor": (2, 3),
        "query_valid_window_mask": (2, 3),
        "support_command": (2, 2),
        "pair_id": (2,),
        "support_rollout_id": (2,),
        "query_rollout_id": (2,),
    }
    assert {
        name: tuple(value.shape) for name, value in _named_batch_tensors(validated).items()
    } == expected_shapes
    assert validated.query.valid_window_mask.tolist() == [
        [True, False, False],
        [False, True, False],
    ]
    assert validated.query.valid_window_mask.sum(dim=1).tolist() == [1, 1]
    assert validated.pair_id.tolist() == [11, 12]
    assert validated.support_rollout_id.tolist() == [101, 103]
    assert validated.query_rollout_id.tolist() == [102, 104]


def test_support_query_validate_pair_permutation_covariance_and_role_copy_rejection() -> None:
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    order = torch.tensor([1, 0])

    permuted = batch.index_select(order).validate(config, support_length=8)

    for name, value in _named_batch_tensors(batch).items():
        torch.testing.assert_close(
            _named_batch_tensors(permuted)[name],
            value.index_select(0, order),
            rtol=0.0,
            atol=0.0,
        )
    with pytest.raises(ValueError, match="different rollout ids"):
        replace(batch, query_rollout_id=batch.support_rollout_id.clone()).validate(
            config,
            support_length=8,
        )


def test_support_query_rejects_command_or_rollout_identity_mismatch() -> None:
    config = _config()
    batch = _batch(config)
    with pytest.raises(ValueError, match="commands must match"):
        replace(batch, support_command=batch.support_command + 1.0).validate(config)
    with pytest.raises(ValueError, match="different rollout ids"):
        replace(batch, query_rollout_id=batch.support_rollout_id).validate(config)
    with pytest.raises(ValueError, match="share one dtype"):
        replace(batch, support_command=batch.support_command.double()).validate(config)
    invalid_mask = torch.zeros_like(batch.query.valid_window_mask)
    with pytest.raises(ValueError, match="at least one valid window"):
        replace(
            batch,
            query=replace(batch.query, valid_window_mask=invalid_mask),
        ).validate(config)


def test_support_query_dataset_round_trip(tmp_path: Path) -> None:
    config = _config()
    batch = _batch(config)
    path = save_support_query_dataset(
        tmp_path / "support_query.pt",
        batch,
        config,
        support_length=8,
        query_length=10,
        metadata={
            "source_checkpoint_sha256": "abc",
            "task_config": "fault-070",
            "fault_joint": "left_knee",
            "fault_strength": 0.7,
            "command": [0.4, 0.0],
            "seed": 7,
        },
    )

    loaded, metadata = load_support_query_dataset(path, config, support_length=8, query_length=10)

    torch.testing.assert_close(loaded.query.realized_future, batch.query.realized_future)
    torch.testing.assert_close(loaded.support.target_future, batch.support.target_future)
    assert metadata["fault_strength"] == 0.7


def test_dataset_singleton_every_field_roundtrip_is_stable_and_role_tagged(
    tmp_path: Path,
) -> None:
    config = _config()
    batch = _semantic_batch(config, pairs=1, windows=1).validate(
        config,
        support_length=8,
    )
    metadata = {
        "source_checkpoint_sha256": "source-role-tag",
        "task_config": "fault-070",
        "fault_joint": "left_knee",
        "fault_strength": 0.7,
        "command": [7.0, 7.001],
        "seed": 17,
        "semantic_tag": "singleton",
    }
    path = save_support_query_dataset(
        tmp_path / "singleton.pt",
        batch,
        config,
        support_length=8,
        query_length=5,
        metadata=metadata,
    )
    digest_before = sha256_file(path)

    first, first_metadata = load_support_query_dataset(
        path,
        config,
        support_length=8,
        query_length=5,
    )
    second, second_metadata = load_support_query_dataset(
        path,
        config,
        support_length=8,
        query_length=5,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)

    for name, expected in _named_batch_tensors(batch).items():
        torch.testing.assert_close(_named_batch_tensors(first)[name], expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(_named_batch_tensors(second)[name], expected, rtol=0.0, atol=0.0)
    assert first.query.valid_window_mask.tolist() == [[True]]
    assert first_metadata == second_metadata == metadata
    assert payload["schema_version"] == 2
    assert payload["architecture"] == asdict(config)
    assert payload["support_length"] == 8
    assert payload["query_length"] == 5
    assert sha256_file(path) == digest_before
    assert digest_before == sha256_file(path)


@pytest.mark.parametrize("mutation", ["schema", "omission", "same_shape_role_swap"])
def test_dataset_direct_schema_digest_omission_and_role_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    config = _config()
    batch = _semantic_batch(config, pairs=1, windows=1).validate(config, support_length=8)
    path = save_support_query_dataset(
        tmp_path / f"mutation-{mutation}.pt",
        batch,
        config,
        support_length=8,
        query_length=5,
        metadata={
            "source_checkpoint_sha256": "source",
            "task_config": "fault-070",
            "fault_joint": "left_knee",
            "fault_strength": 0.7,
            "command": [7.0, 7.001],
            "seed": 17,
        },
    )
    original_digest = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if mutation == "schema":
        payload["schema_version"] = 99
    elif mutation == "omission":
        payload.pop("query_action_history")
    else:
        payload["query_planner_intent"], payload["query_realized_future"] = (
            payload["query_realized_future"],
            payload["query_planner_intent"],
        )
    torch.save(payload, path)

    assert sha256_file(path) != original_digest
    if mutation == "schema":
        with pytest.raises(ValueError, match="unsupported Support-Query dataset schema"):
            load_support_query_dataset(path, config, support_length=8, query_length=5)
    elif mutation == "omission":
        with pytest.raises(ValueError, match="missing tensor fields"):
            load_support_query_dataset(path, config, support_length=8, query_length=5)
    else:
        swapped, _ = load_support_query_dataset(path, config, support_length=8, query_length=5)
        torch.testing.assert_close(
            swapped.query.planner_intent,
            batch.query.realized_future,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            swapped.query.realized_future,
            batch.query.planner_intent,
            rtol=0.0,
            atol=0.0,
        )
        assert not torch.equal(
            swapped.query.planner_intent,
            batch.query.planner_intent,
        )


def test_active_dataset_loader_has_no_legacy_schema_surface(tmp_path: Path) -> None:
    config = _config()
    batch = _semantic_batch(config, pairs=1, windows=1).validate(config, support_length=8)
    path = save_support_query_dataset(
        tmp_path / "active-schema.pt",
        batch,
        config,
        support_length=8,
        query_length=5,
        metadata={
            "source_checkpoint_sha256": "source",
            "task_config": "fault-070",
            "fault_joint": "left_knee",
            "fault_strength": 0.7,
            "command": [7.0, 7.001],
            "seed": 17,
        },
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema_version"] = 1
    torch.save(payload, path)

    assert (
        "allow_legacy_single_anchor" not in inspect.signature(load_support_query_dataset).parameters
    )
    with pytest.raises(ValueError, match="unsupported Support-Query dataset schema"):
        load_support_query_dataset(
            path,
            config,
            support_length=8,
            query_length=5,
        )


def test_support_query_split_keeps_rollout_groups_disjoint() -> None:
    config = _config()
    batch = _batch(config, batch_size=6)
    grouped = replace(
        batch,
        support_rollout_id=torch.tensor([0, 0, 2, 2, 4, 4]),
        query_rollout_id=torch.tensor([1, 1, 3, 3, 5, 5]),
    ).validate(config)

    train, validation = split_support_query_by_rollout(grouped, validation_fraction=0.34, seed=7)

    train_groups = set(zip(train.support_rollout_id.tolist(), train.query_rollout_id.tolist()))
    validation_groups = set(
        zip(validation.support_rollout_id.tolist(), validation.query_rollout_id.tolist())
    )
    assert train_groups.isdisjoint(validation_groups)
    assert train.batch_size + validation.batch_size == grouped.batch_size
    assert support_query_split_identity_sha256(train) != support_query_split_identity_sha256(
        validation
    )


def _artifact_fixture(
    tmp_path: Path,
) -> tuple[
    FADAPlannerIDMPolicy,
    SupportQueryContextConfig,
    Path,
    Path,
    Path,
]:
    config = _config()
    healthy = FADAPlannerIDMPolicy(config)
    context_config = SupportQueryContextConfig(
        support_length=8, context_hidden_dim=12, context_layers=1
    )
    source_checkpoint = tmp_path / "healthy.pt"
    source_checkpoint.write_bytes(b"healthy-checkpoint")
    source_sha = sha256_file(source_checkpoint)
    batch = _batch(config, batch_size=6)
    grouped = replace(
        batch,
        support_rollout_id=torch.tensor([0, 0, 2, 2, 4, 4]),
        query_rollout_id=torch.tensor([1, 1, 3, 3, 5, 5]),
    ).validate(config)
    dataset_path = save_support_query_dataset(
        tmp_path / "support_query.pt",
        grouped,
        config,
        support_length=8,
        query_length=10,
        metadata={
            "source_checkpoint_sha256": source_sha,
            "task_config": "fault-070",
            "fault_joint": "left_knee",
            "fault_strength": 0.7,
            "command": [0.4, 0.0],
            "seed": 7,
        },
    )
    train, validation = split_support_query_by_rollout(grouped, validation_fraction=0.34, seed=7)
    checkpoint_path = save_context_support_query_checkpoint(
        tmp_path / "context.pt",
        prepare_support_query_training(healthy, context_config, learning_rate=3.0e-4),
        source_checkpoint_sha256=source_sha,
        dataset_sha256=sha256_file(dataset_path),
        train_split_sha256=support_query_split_identity_sha256(train),
        validation_split_sha256=support_query_split_identity_sha256(validation),
        step=3,
        split_seed=7,
        metrics={"validation_mse": 0.25},
        resolved_config={},
    )
    return healthy, context_config, source_checkpoint, dataset_path, checkpoint_path


def _set_borrowed_owner_lifecycle(healthy: FADAPlannerIDMPolicy) -> None:
    for index, module in enumerate((*healthy.planner.modules(), *healthy.idm.modules())):
        module.training = index % 2 == 0
    for index, parameter in enumerate((*healthy.planner.parameters(), *healthy.idm.parameters())):
        parameter.requires_grad_(index % 2 == 0)


def _borrowed_owner_lifecycle(
    healthy: FADAPlannerIDMPolicy,
) -> tuple[tuple[tuple[int, bool], ...], tuple[tuple[int, bool], ...]]:
    modules = tuple(
        (id(module), module.training)
        for module in (*healthy.planner.modules(), *healthy.idm.modules())
    )
    parameters = tuple(
        (id(parameter), parameter.requires_grad)
        for parameter in (*healthy.planner.parameters(), *healthy.idm.parameters())
    )
    return modules, parameters


def _prepare_artifact_fixture(tmp_path: Path):
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    return prepare_context_support_query_artifact(
        healthy,
        context_config,
        source_checkpoint_path=source_checkpoint,
        dataset_path=dataset_path,
        context_checkpoint_path=checkpoint_path,
        support_length=8,
        query_length=10,
        validation_fraction=0.34,
        split_seed=7,
    )


def test_prepared_context_artifact_keeps_rollout_groups_disjoint(tmp_path: Path) -> None:
    prepared = _prepare_artifact_fixture(tmp_path)

    train_groups = set(
        zip(prepared.train.support_rollout_id.tolist(), prepared.train.query_rollout_id.tolist())
    )
    validation_groups = set(
        zip(
            prepared.validation.support_rollout_id.tolist(),
            prepared.validation.query_rollout_id.tolist(),
        )
    )
    assert train_groups.isdisjoint(validation_groups)
    assert prepared.split_contract == "rollout_group_split"
    assert prepared.method_contract_id == FADA_CONTEXT_METHOD_CONTRACT_ID
    assert prepared.checkpoint_schema == 4
    assert prepared.checkpoint_step == 3
    assert not hasattr(prepared, "checkpoint_payload")


def test_artifact_admission_owner_reports_typed_v006_query_evidence(tmp_path: Path) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    owner = getattr(training_module, "preflight_context_support_query_artifact", None)

    assert callable(owner)
    result = owner(
        healthy,
        context_config,
        source_checkpoint_path=source_checkpoint,
        dataset_path=dataset_path,
        context_checkpoint_path=checkpoint_path,
        support_length=8,
        query_length=10,
        validation_fraction=0.34,
        split_seed=7,
    )

    assert result.method_contract_id == FADA_CONTEXT_METHOD_CONTRACT_ID
    assert result.checkpoint_schema == 4
    assert result.checkpoint_step == 3
    assert result.pair_ids == tuple(range(6))
    assert result.support_rollout_ids == (0, 0, 2, 2, 4, 4)
    assert result.query_rollout_ids == (1, 1, 3, 3, 5, 5)
    assert result.delta_z_shape == (
        6,
        result.window_count,
        healthy.config.hidden_dim,
    )


@pytest.mark.parametrize(
    "field,wrong_value",
    [
        ("source_checkpoint_sha256", "wrong-source"),
        ("dataset_sha256", "wrong-dataset"),
        ("train_split_sha256", "wrong-train"),
        ("validation_split_sha256", "wrong-validation"),
        ("split_seed", 99),
    ],
)
def test_prepared_context_artifact_rejects_identity_mismatch(
    tmp_path: Path,
    field: str,
    wrong_value: str | int,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload[field] = wrong_value
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="mismatch"):
        prepare_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_source_checkpoint_sha256",
        "remove_dataset_sha256",
        "remove_train_split_sha256",
        "remove_validation_split_sha256",
        "remove_split_seed",
        "swap_train_validation_splits",
    ],
)
def test_artifact_missing_or_swapped_identity_rejects_before_policy_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if mutation.startswith("remove_"):
        payload.pop(mutation.removeprefix("remove_"))
    else:
        payload["train_split_sha256"], payload["validation_split_sha256"] = (
            payload["validation_split_sha256"],
            payload["train_split_sha256"],
        )
    torch.save(payload, checkpoint_path)
    constructions = 0

    def record_construction(*_args, **_kwargs):
        nonlocal constructions
        constructions += 1
        pytest.fail("invalid artifact identity reached Context policy construction")

    monkeypatch.setattr(
        training_module,
        "prepare_context_support_query_policy",
        record_construction,
    )

    with pytest.raises(ValueError, match="mismatch"):
        prepare_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )

    assert constructions == 0


@pytest.mark.parametrize(("section", "field"), STRUCTURAL_IDENTITY_MUTATIONS)
def test_artifact_structural_identity_matrix_rejects_before_policy_or_state_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload[section][field] = int(payload[section][field]) + 1
    torch.save(payload, checkpoint_path)
    mutable_calls: list[str] = []

    def record_policy(*_args, **_kwargs):
        mutable_calls.append("policy")
        pytest.fail("structural mismatch reached Context policy construction")

    def record_load(*_args, **_kwargs):
        mutable_calls.append("load_state_dict")
        pytest.fail("structural mismatch reached Context state load")

    monkeypatch.setattr(
        training_module,
        "prepare_context_support_query_policy",
        record_policy,
    )
    monkeypatch.setattr(
        training_module.FADASupportContextEncoder,
        "load_state_dict",
        record_load,
    )

    with pytest.raises(ValueError, match="mismatch"):
        prepare_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )

    assert mutable_calls == []


@pytest.mark.parametrize("route", ["artifact", "resume", "preflight"])
def test_nonfinite_context_checkpoint_rejects_before_all_mutable_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload["context_state_dict"]
    state_key = next(
        name
        for name, value in state.items()
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel() > 0
    )
    nonfinite = state[state_key].clone()
    nonfinite.reshape(-1)[0] = float("nan")
    state[state_key] = nonfinite
    torch.save(payload, checkpoint_path)
    mutable_calls: list[str] = []

    def record_boundary(name: str):
        def recorder(*_args, **_kwargs):
            mutable_calls.append(name)
            pytest.fail(f"nonfinite Context state reached {name}")

        return recorder

    monkeypatch.setattr(
        training_module,
        "prepare_context_support_query_policy",
        record_boundary("policy construction"),
    )
    monkeypatch.setattr(
        training_module,
        "prepare_support_query_training",
        record_boundary("training construction"),
    )
    monkeypatch.setattr(
        training_module.FADASupportContextEncoder,
        "load_state_dict",
        record_boundary("state load"),
    )
    monkeypatch.setattr(
        training_module.FrozenIDMSupportQueryPolicy,
        "reconstruct_query",
        record_boundary("artifact reconstruction"),
    )
    monkeypatch.setattr(
        torch.optim,
        "Adam",
        record_boundary("optimizer construction"),
    )

    with pytest.raises(ValueError, match="non-finite tensor"):
        if route == "artifact":
            prepare_context_support_query_artifact(
                healthy,
                context_config,
                source_checkpoint_path=source_checkpoint,
                dataset_path=dataset_path,
                context_checkpoint_path=checkpoint_path,
                support_length=8,
                query_length=10,
                validation_fraction=0.34,
                split_seed=7,
            )
        elif route == "resume":
            resume_context_support_query_training(
                healthy,
                context_config,
                checkpoint_path,
                learning_rate=3.0e-4,
                expected_source_checkpoint_sha256=payload["source_checkpoint_sha256"],
                expected_dataset_sha256=payload["dataset_sha256"],
                expected_train_split_sha256=payload["train_split_sha256"],
                expected_validation_split_sha256=payload["validation_split_sha256"],
                expected_split_seed=payload["split_seed"],
            )
        else:
            training_module.preflight_context_support_query_artifact(
                healthy,
                context_config,
                source_checkpoint_path=source_checkpoint,
                dataset_path=dataset_path,
                context_checkpoint_path=checkpoint_path,
                support_length=8,
                query_length=10,
                validation_fraction=0.34,
                split_seed=7,
            )

    assert mutable_calls == []


def test_artifact_preflight_rejects_nonfinite_delta_computed_from_finite_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    with torch.no_grad():
        for parameter in healthy.idm.parameters():
            parameter.zero_()
    context_outputs: list[torch.Tensor] = []
    decoder_inputs: list[torch.Tensor] = []
    original_context_forward = FADASupportContextEncoder.forward
    original_decode_latent = FADAInverseDynamicsModel.decode_latent

    def record_context_forward(
        model: FADASupportContextEncoder,
        support: SupportContextBatch,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        delta_z = original_context_forward(
            model,
            support,
            observation_history,
            action_history,
        )
        context_outputs.append(delta_z.detach().clone())
        return delta_z

    def record_decode_latent(
        model: FADAInverseDynamicsModel,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        decoder_inputs.append(latent.detach().clone())
        return original_decode_latent(model, latent)

    monkeypatch.setattr(
        FADASupportContextEncoder,
        "forward",
        record_context_forward,
    )
    monkeypatch.setattr(
        FADAInverseDynamicsModel,
        "decode_latent",
        record_decode_latent,
    )
    dataset_payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    for name, value in dataset_payload.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            alternating = torch.full_like(value, 1.0e38)
            alternating.reshape(-1)[1::2] = -1.0e38
            dataset_payload[name] = alternating
    dataset_payload["query_command"] = (
        dataset_payload["support_command"][:, None, :]
        .expand_as(dataset_payload["query_command"])
        .clone()
    )
    torch.save(dataset_payload, dataset_path)

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for name, value in checkpoint_payload["context_state_dict"].items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            checkpoint_payload["context_state_dict"][name] = torch.full_like(value, 1.0e38)
    checkpoint_payload["dataset_sha256"] = sha256_file(dataset_path)
    torch.save(checkpoint_payload, checkpoint_path)

    assert all(
        bool(torch.isfinite(value).all())
        for value in dataset_payload.values()
        if isinstance(value, torch.Tensor)
    )
    assert all(
        bool(torch.isfinite(value).all())
        for value in checkpoint_payload["context_state_dict"].values()
        if isinstance(value, torch.Tensor)
    )

    result = None
    with pytest.raises(ValueError) as error:
        result = training_module.preflight_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )
    assert result is None
    assert context_outputs and not bool(torch.isfinite(context_outputs[0]).all())
    assert decoder_inputs == [], "non-finite Context delta reached the action decoder"
    assert "Context delta_z must contain only finite values" in str(error.value)


def test_context_checkpoint_state_requires_typed_tensor_leaves_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_key = next(iter(payload["context_state_dict"]))
    payload["context_state_dict"][state_key] = "not-a-tensor"
    torch.save(payload, checkpoint_path)
    constructions = 0

    def record_construction(*_args, **_kwargs):
        nonlocal constructions
        constructions += 1
        pytest.fail("untyped Context state reached policy construction")

    monkeypatch.setattr(
        training_module,
        "prepare_context_support_query_policy",
        record_construction,
    )

    with pytest.raises(ValueError, match="tensor leaf"):
        prepare_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )

    assert constructions == 0


def test_context_checkpoint_meta_tensor_rejects_without_scalar_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_key = next(iter(payload["context_state_dict"]))
    payload["context_state_dict"][state_key] = torch.empty_like(
        payload["context_state_dict"][state_key],
        device="meta",
    )
    torch.save(payload, checkpoint_path)
    constructions = 0

    def record_construction(*_args, **_kwargs):
        nonlocal constructions
        constructions += 1
        pytest.fail("meta Context state reached policy construction")

    monkeypatch.setattr(
        training_module,
        "prepare_context_support_query_policy",
        record_construction,
    )

    with pytest.raises(ValueError, match="must be materialized before admission"):
        prepare_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )

    assert constructions == 0


def test_resume_nonfinite_optimizer_state_rejects_before_training_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    context_config = SupportQueryContextConfig(
        support_length=8,
        context_hidden_dim=12,
        context_layers=1,
    )
    source = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        context_config,
        learning_rate=3.0e-4,
    )
    path = save_context_support_query_checkpoint(
        tmp_path / "nonfinite-optimizer.pt",
        source,
        source_checkpoint_sha256="healthy-sha",
        **CHECKPOINT_IDENTITIES,
        step=3,
        split_seed=7,
        metrics={"validation_mse": 0.25},
        resolved_config={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["optimizer_state_dict"]["state"] = {0: {"exp_avg": torch.tensor([float("nan")])}}
    torch.save(payload, path)
    constructions = 0

    def record_construction(*_args, **_kwargs):
        nonlocal constructions
        constructions += 1
        pytest.fail("nonfinite optimizer state reached training construction")

    monkeypatch.setattr(
        training_module,
        "prepare_support_query_training",
        record_construction,
    )

    with pytest.raises(ValueError, match="non-finite tensor"):
        resume_context_support_query_training(
            FADAPlannerIDMPolicy(config),
            context_config,
            path,
            learning_rate=3.0e-4,
            expected_source_checkpoint_sha256="healthy-sha",
            **EXPECTED_CHECKPOINT_IDENTITIES,
            expected_split_seed=7,
        )

    assert constructions == 0


@pytest.mark.parametrize("historical_schema", [1, 2, 3])
def test_historical_artifact_rejected_before_context_policy_or_optimizer_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    historical_schema: int,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["schema_version"] = historical_schema
    payload.pop("method_contract_id", None)
    torch.save(payload, checkpoint_path)
    monkeypatch.setattr(
        training_module,
        "prepare_context_support_query_policy",
        lambda *_args, **_kwargs: pytest.fail(
            "historical schema reached Context policy/optimizer construction"
        ),
    )

    with pytest.raises(ValueError, match="historical fixed-residual checkpoint schema"):
        prepare_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )


@pytest.mark.parametrize("method_id", [None, "FADA-CONTEXT-METHOD-v005", "wrong"])
def test_v4_artifact_requires_exact_v006_method_before_policy_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_id: str | None,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if method_id is None:
        payload.pop("method_contract_id")
    else:
        payload["method_contract_id"] = method_id
    torch.save(payload, checkpoint_path)
    monkeypatch.setattr(
        training_module,
        "prepare_context_support_query_policy",
        lambda *_args, **_kwargs: pytest.fail("wrong method reached Context policy construction"),
    )

    with pytest.raises(ValueError, match="method Contract mismatch"):
        prepare_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )


@pytest.mark.parametrize("checkpoint_step", [-1, 1.5, "3", None])
def test_v4_artifact_requires_typed_nonnegative_step_before_policy_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_step: object,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["step"] = checkpoint_step
    torch.save(payload, checkpoint_path)
    monkeypatch.setattr(
        training_module,
        "prepare_context_support_query_policy",
        lambda *_args, **_kwargs: pytest.fail("invalid step reached Context policy construction"),
    )

    with pytest.raises(ValueError, match="checkpoint step must be a non-negative integer"):
        prepare_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )


def test_policy_only_artifact_preparation_never_constructs_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        torch.optim,
        "Adam",
        lambda *_args, **_kwargs: pytest.fail("policy-only artifact constructed optimizer"),
    )

    prepared = prepare_context_support_query_artifact(
        healthy,
        context_config,
        source_checkpoint_path=source_checkpoint,
        dataset_path=dataset_path,
        context_checkpoint_path=checkpoint_path,
        support_length=8,
        query_length=10,
        validation_fraction=0.34,
        split_seed=7,
    )

    assert prepared.policy.training is False
    assert not hasattr(prepared, "optimizer")


def test_active_artifact_rejects_schema_v1_dataset_before_mutable_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    dataset_payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    dataset_payload["schema_version"] = 1
    dataset_payload["query_observation_history"] = dataset_payload["query_observation_history"][
        :, 0
    ]
    dataset_payload["query_action_history"] = dataset_payload["query_action_history"][:, 0]
    dataset_payload["query_command"] = dataset_payload["query_command"][:, 0]
    dataset_payload["query_planner_intent"] = dataset_payload["query_planner_intent"][:, 0]
    dataset_payload["query_realized_future"] = dataset_payload["query_realized_future"][:, 0]
    first_action = dataset_payload.pop("query_executed_action")[:, 0]
    dataset_payload["query_executed_action_chunk"] = first_action[:, None, :].expand(
        -1,
        healthy.config.prediction_horizon,
        -1,
    )
    dataset_payload.pop("query_window_anchor")
    dataset_payload.pop("query_valid_window_mask")
    torch.save(dataset_payload, dataset_path)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_payload["dataset_sha256"] = sha256_file(dataset_path)
    torch.save(checkpoint_payload, checkpoint_path)
    boundary_calls: list[str] = []
    original_prepare = training_module.prepare_context_support_query_policy
    original_load_state_dict = training_module.FADASupportContextEncoder.load_state_dict

    def record_prepare(*args, **kwargs):
        boundary_calls.append("policy")
        return original_prepare(*args, **kwargs)

    def record_load_state_dict(*args, **kwargs):
        boundary_calls.append("state_dict")
        return original_load_state_dict(*args, **kwargs)

    monkeypatch.setattr(training_module, "prepare_context_support_query_policy", record_prepare)
    monkeypatch.setattr(
        training_module.FADASupportContextEncoder,
        "load_state_dict",
        record_load_state_dict,
    )
    monkeypatch.setattr(
        torch.optim,
        "Adam",
        lambda *_args, **_kwargs: boundary_calls.append("optimizer"),
    )

    with pytest.raises(ValueError, match="unsupported Support-Query dataset schema"):
        prepare_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )

    assert boundary_calls == []


def test_playback_rejects_checkpoint_dataset_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from omegaconf import OmegaConf
    from scripts import play_fada_context_viser as playback_script

    healthy, _, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(tmp_path)
    dataset_payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    dataset_payload["query_length"] = 8
    torch.save(dataset_payload, dataset_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["dataset_sha256"] = "wrong-dataset"
    torch.save(payload, checkpoint_path)
    cfg = OmegaConf.create(
        {
            "context_playback": {
                "healthy_checkpoint": str(source_checkpoint),
                "context_checkpoint": str(checkpoint_path),
                "dataset": str(dataset_path),
                "support_length": 8,
                "query_length": 8,
                "validation_fraction": 0.34,
                "split_seed": 7,
                "support_index": 0,
                "hidden_dim": 12,
                "num_layers": 1,
                "delta_scale": 0.1,
            }
        }
    )
    monkeypatch.setattr(
        playback_script,
        "load_fada_policy_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(policy=healthy),
    )

    with pytest.raises(ValueError, match="dataset_sha256 mismatch"):
        playback_script._context_controller(cfg, device="cpu")


def test_training_paths_must_be_fresh(tmp_path: Path) -> None:
    artifact = tmp_path / "dataset.pt"
    output_dir = tmp_path / "run"
    require_fresh_support_query_run_paths(artifact, output_dir)
    artifact.touch()
    with pytest.raises(FileExistsError, match="dataset artifact already exists"):
        require_fresh_support_query_run_paths(artifact, output_dir)
    artifact.unlink()
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="output directory already exists"):
        require_fresh_support_query_run_paths(artifact, output_dir)


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
    resumed = resume_context_support_query_training(
        FADAPlannerIDMPolicy(config),
        context_config,
        path,
        learning_rate=3.0e-4,
        expected_source_checkpoint_sha256="healthy-sha",
        **EXPECTED_CHECKPOINT_IDENTITIES,
    )

    assert resumed.checkpoint_schema == 4
    assert resumed.method_contract_id == FADA_CONTEXT_METHOD_CONTRACT_ID
    assert resumed.checkpoint_step == 3
    torch.testing.assert_close(
        resumed.setup.policy.context_encoder.delta_head.bias,
        source.policy.context_encoder.delta_head.bias,
    )
    with pytest.raises(ValueError, match="source identity mismatch"):
        resume_context_support_query_training(
            FADAPlannerIDMPolicy(config),
            context_config,
            path,
            learning_rate=3.0e-4,
            expected_source_checkpoint_sha256="wrong-sha",
            **EXPECTED_CHECKPOINT_IDENTITIES,
        )
    with pytest.raises(ValueError, match="dataset_sha256 mismatch"):
        resume_context_support_query_training(
            FADAPlannerIDMPolicy(config),
            context_config,
            path,
            learning_rate=3.0e-4,
            expected_source_checkpoint_sha256="healthy-sha",
            expected_dataset_sha256="wrong-dataset",
            expected_train_split_sha256="train-split-sha",
            expected_validation_split_sha256="validation-split-sha",
        )


def test_resume_checkpoint_admits_raw_identity_before_fresh_training_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    context_config = SupportQueryContextConfig(
        support_length=8,
        context_hidden_dim=12,
        context_layers=1,
    )
    source = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        context_config,
        learning_rate=3.0e-4,
    )
    with torch.no_grad():
        source.policy.context_encoder.delta_head.bias.fill_(0.07)
    path = save_context_support_query_checkpoint(
        tmp_path / "context.pt",
        source,
        source_checkpoint_sha256="healthy-sha",
        **CHECKPOINT_IDENTITIES,
        step=9,
        split_seed=7,
        metrics={"validation_mse": 0.2},
        resolved_config={},
    )
    resume = getattr(training_module, "resume_context_support_query_training", None)
    assert callable(resume)
    constructions = 0
    original_prepare = training_module.prepare_support_query_training

    def record_prepare(*args, **kwargs):
        nonlocal constructions
        constructions += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(training_module, "prepare_support_query_training", record_prepare)

    with pytest.raises(ValueError, match="dataset_sha256 mismatch"):
        resume(
            FADAPlannerIDMPolicy(config),
            context_config,
            path,
            learning_rate=3.0e-4,
            expected_source_checkpoint_sha256="healthy-sha",
            expected_dataset_sha256="wrong-dataset",
            expected_train_split_sha256="train-split-sha",
            expected_validation_split_sha256="validation-split-sha",
            expected_split_seed=7,
        )
    assert constructions == 0

    resumed = resume(
        FADAPlannerIDMPolicy(config),
        context_config,
        path,
        learning_rate=3.0e-4,
        expected_source_checkpoint_sha256="healthy-sha",
        **EXPECTED_CHECKPOINT_IDENTITIES,
        expected_split_seed=7,
    )

    assert constructions == 1
    assert resumed.method_contract_id == FADA_CONTEXT_METHOD_CONTRACT_ID
    assert resumed.checkpoint_schema == 4
    assert resumed.checkpoint_step == 9
    torch.testing.assert_close(
        resumed.setup.policy.context_encoder.delta_head.bias,
        source.policy.context_encoder.delta_head.bias,
    )


def test_resume_step_zero_restores_typed_identity_and_context_state(tmp_path: Path) -> None:
    config = _config()
    context_config = SupportQueryContextConfig(
        support_length=8,
        context_hidden_dim=12,
        context_layers=1,
    )
    source = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        context_config,
        learning_rate=3.0e-4,
    )
    with torch.no_grad():
        source.policy.context_encoder.delta_head.bias.copy_(
            torch.linspace(-0.1, 0.1, config.hidden_dim)
        )
    path = save_context_support_query_checkpoint(
        tmp_path / "step-zero.pt",
        source,
        source_checkpoint_sha256="healthy-sha",
        **CHECKPOINT_IDENTITIES,
        step=0,
        split_seed=7,
        metrics={"validation_mse": 0.25},
        resolved_config={},
    )

    resumed = resume_context_support_query_training(
        FADAPlannerIDMPolicy(config),
        context_config,
        path,
        learning_rate=3.0e-4,
        expected_source_checkpoint_sha256="healthy-sha",
        **EXPECTED_CHECKPOINT_IDENTITIES,
        expected_split_seed=7,
    )

    assert resumed.method_contract_id == FADA_CONTEXT_METHOD_CONTRACT_ID
    assert resumed.checkpoint_schema == 4
    assert resumed.checkpoint_step == 0
    for name, value in resumed.setup.policy.context_encoder.state_dict().items():
        assert torch.equal(value, source.policy.context_encoder.state_dict()[name])
    assert not any(
        parameter.requires_grad
        for module in (resumed.setup.policy.planner, resumed.setup.policy.idm)
        for parameter in module.parameters()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_1",
        "schema_2",
        "schema_3",
        "wrong_method",
        "remove_method_contract_id",
        "remove_source_checkpoint_sha256",
        "remove_dataset_sha256",
        "remove_train_split_sha256",
        "remove_validation_split_sha256",
        "remove_split_seed",
        "swap_train_validation_splits",
    ],
)
def test_resume_invalid_identity_matrix_rejects_before_setup_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = _config()
    context_config = SupportQueryContextConfig(
        support_length=8,
        context_hidden_dim=12,
        context_layers=1,
    )
    source = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        context_config,
        learning_rate=3.0e-4,
    )
    path = save_context_support_query_checkpoint(
        tmp_path / f"resume-{mutation}.pt",
        source,
        source_checkpoint_sha256="healthy-sha",
        **CHECKPOINT_IDENTITIES,
        step=3,
        split_seed=7,
        metrics={"validation_mse": 0.25},
        resolved_config={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if mutation.startswith("schema_"):
        payload["schema_version"] = int(mutation.removeprefix("schema_"))
    elif mutation == "wrong_method":
        payload["method_contract_id"] = "FADA-CONTEXT-METHOD-v005"
    elif mutation.startswith("remove_"):
        payload.pop(mutation.removeprefix("remove_"))
    else:
        payload["train_split_sha256"], payload["validation_split_sha256"] = (
            payload["validation_split_sha256"],
            payload["train_split_sha256"],
        )
    torch.save(payload, path)
    constructions = 0

    def record_construction(*_args, **_kwargs):
        nonlocal constructions
        constructions += 1
        pytest.fail("invalid resume checkpoint reached fresh setup construction")

    monkeypatch.setattr(
        training_module,
        "prepare_support_query_training",
        record_construction,
    )

    with pytest.raises(ValueError):
        resume_context_support_query_training(
            FADAPlannerIDMPolicy(config),
            context_config,
            path,
            learning_rate=3.0e-4,
            expected_source_checkpoint_sha256="healthy-sha",
            **EXPECTED_CHECKPOINT_IDENTITIES,
            expected_split_seed=7,
        )

    assert constructions == 0


@pytest.mark.parametrize(("section", "field"), STRUCTURAL_IDENTITY_MUTATIONS)
def test_resume_structural_identity_matrix_rejects_before_setup_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
) -> None:
    config = _config()
    context_config = SupportQueryContextConfig(
        support_length=8,
        context_hidden_dim=12,
        context_layers=1,
    )
    source = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        context_config,
        learning_rate=3.0e-4,
    )
    path = save_context_support_query_checkpoint(
        tmp_path / f"resume-structure-{section}-{field}.pt",
        source,
        source_checkpoint_sha256="healthy-sha",
        **CHECKPOINT_IDENTITIES,
        step=3,
        split_seed=7,
        metrics={"validation_mse": 0.25},
        resolved_config={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload[section][field] = int(payload[section][field]) + 1
    torch.save(payload, path)
    constructions = 0

    def record_construction(*_args, **_kwargs):
        nonlocal constructions
        constructions += 1
        pytest.fail("structural mismatch reached fresh setup construction")

    monkeypatch.setattr(
        training_module,
        "prepare_support_query_training",
        record_construction,
    )

    with pytest.raises(ValueError, match="mismatch"):
        resume_context_support_query_training(
            FADAPlannerIDMPolicy(config),
            context_config,
            path,
            learning_rate=3.0e-4,
            expected_source_checkpoint_sha256="healthy-sha",
            **EXPECTED_CHECKPOINT_IDENTITIES,
            expected_split_seed=7,
        )

    assert constructions == 0


def test_resume_failure_does_not_return_partially_prepared_training(tmp_path: Path) -> None:
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
    resumed = None
    with pytest.raises((KeyError, ValueError)):
        resumed = resume_context_support_query_training(
            FADAPlannerIDMPolicy(config),
            context_config,
            path,
            learning_rate=3.0e-4,
            expected_source_checkpoint_sha256="healthy-sha",
            **EXPECTED_CHECKPOINT_IDENTITIES,
        )

    assert resumed is None


@pytest.mark.parametrize("failed_load", ["context", "optimizer"])
def test_resume_failed_load_restores_borrowed_owner_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_load: str,
) -> None:
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
    healthy = FADAPlannerIDMPolicy(config)
    _set_borrowed_owner_lifecycle(healthy)
    before = _borrowed_owner_lifecycle(healthy)

    def fail_load(*_args, **_kwargs):
        raise RuntimeError(f"injected {failed_load} load failure")

    if failed_load == "context":
        monkeypatch.setattr(
            training_module.FADASupportContextEncoder,
            "load_state_dict",
            fail_load,
        )
    else:
        monkeypatch.setattr(torch.optim.Adam, "load_state_dict", fail_load)

    resumed = None
    with pytest.raises(RuntimeError, match=f"injected {failed_load} load failure"):
        resumed = resume_context_support_query_training(
            healthy,
            context_config,
            path,
            learning_rate=3.0e-4,
            expected_source_checkpoint_sha256="healthy-sha",
            **EXPECTED_CHECKPOINT_IDENTITIES,
        )

    assert resumed is None
    assert _borrowed_owner_lifecycle(healthy) == before


def test_artifact_context_load_failure_restores_borrowed_owner_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy, context_config, source_checkpoint, dataset_path, checkpoint_path = _artifact_fixture(
        tmp_path
    )
    _set_borrowed_owner_lifecycle(healthy)
    before = _borrowed_owner_lifecycle(healthy)

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("injected artifact Context load failure")

    monkeypatch.setattr(
        training_module.FADASupportContextEncoder,
        "load_state_dict",
        fail_load,
    )

    prepared = None
    with pytest.raises(RuntimeError, match="injected artifact Context load failure"):
        prepared = prepare_context_support_query_artifact(
            healthy,
            context_config,
            source_checkpoint_path=source_checkpoint,
            dataset_path=dataset_path,
            context_checkpoint_path=checkpoint_path,
            support_length=8,
            query_length=10,
            validation_fraction=0.34,
            split_seed=7,
        )

    assert prepared is None
    assert _borrowed_owner_lifecycle(healthy) == before


def test_training_owner_controls_validation_and_checkpoint_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    batch = _batch(config, batch_size=4)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    evaluations = iter((10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0))
    monkeypatch.setattr(
        training_module,
        "evaluate_context_action_mse",
        lambda *_args, **_kwargs: next(evaluations),
    )
    events: list[tuple[str, dict[str, object]]] = []
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    result = run_support_query_training(
        setup,
        batch,
        batch,
        output_dir=output_dir,
        source_checkpoint_sha256="source",
        dataset_sha256="dataset",
        train_split_sha256="train",
        validation_split_sha256="validation",
        split_seed=7,
        resolved_config={},
        config=SupportQueryTrainingLoopConfig(
            steps=2,
            batch_size=2,
            log_interval=1,
            checkpoint_interval=2,
            gradient_clip_norm=1.0,
            minimum_zero_context_mse=0.0,
        ),
        emit=lambda event, **payload: events.append((event, payload)),
    )

    assert {path.name for path in output_dir.iterdir()} == {
        "best.pt",
        "context_2.pt",
        "final.pt",
    }
    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    interval = torch.load(output_dir / "context_2.pt", map_location="cpu", weights_only=True)
    final = torch.load(output_dir / "final.pt", map_location="cpu", weights_only=True)
    assert best["step"] == 2
    assert interval["step"] == 2
    assert final["step"] == 2
    assert final["metrics"]["best_step"] == 2.0
    assert result.best_step == 2
    assert result.final_checkpoint == output_dir / "final.pt"
    assert [event for event, _ in events] == [
        "training_started",
        "training_step",
        "training_step",
        "training_completed",
    ]
    assert [payload["step"] for event, payload in events if event == "training_step"] == [
        1,
        2,
    ]


def test_training_owner_one_step_has_exact_context_ids_update_and_transaction_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(107)
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    learning_rate = 1.0e-3
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=learning_rate,
    )
    context_ids = {id(parameter) for parameter in setup.policy.context_encoder.parameters()}
    optimizer_ids = {
        id(parameter) for group in setup.optimizer.param_groups for parameter in group["params"]
    }
    owner_ids = {
        id(parameter)
        for module in (setup.policy.planner, setup.policy.idm)
        for parameter in module.parameters()
    }
    assert optimizer_ids == context_ids
    assert optimizer_ids.isdisjoint(owner_ids)
    planner_before = {
        name: value.detach().clone() for name, value in setup.policy.planner.state_dict().items()
    }
    idm_before = {
        name: value.detach().clone() for name, value in setup.policy.idm.state_dict().items()
    }
    bias = setup.policy.context_encoder.delta_head.bias
    bias_before = bias.detach().clone()
    trace: list[str] = []
    captured_gradient: torch.Tensor | None = None
    original_zero_grad = setup.optimizer.zero_grad
    original_step = setup.optimizer.step
    original_clip = torch.nn.utils.clip_grad_norm_

    def record_zero_grad(*args, **kwargs):
        trace.append("zero_grad")
        return original_zero_grad(*args, **kwargs)

    def record_clip(parameters, max_norm, *args, **kwargs):
        materialized = tuple(parameters)
        assert {id(parameter) for parameter in materialized} == context_ids
        trace.append("clip")
        return original_clip(materialized, max_norm, *args, **kwargs)

    def record_step(*args, **kwargs):
        nonlocal captured_gradient
        assert bias.grad is not None
        captured_gradient = bias.grad.detach().clone()
        trace.append("step")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(setup.optimizer, "zero_grad", record_zero_grad)
    monkeypatch.setattr(setup.optimizer, "step", record_step)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    monkeypatch.setattr(
        training_module,
        "evaluate_context_action_mse",
        lambda *_args, **_kwargs: 1.0,
    )
    backward_handle = bias.register_hook(lambda gradient: trace.append("backward") or gradient)
    output_dir = tmp_path / "one-step"
    output_dir.mkdir()
    try:
        result = run_support_query_training(
            setup,
            batch,
            batch,
            output_dir=output_dir,
            source_checkpoint_sha256="source",
            dataset_sha256="dataset",
            train_split_sha256="train",
            validation_split_sha256="validation",
            split_seed=7,
            resolved_config={},
            config=SupportQueryTrainingLoopConfig(
                steps=1,
                batch_size=2,
                log_interval=1,
                checkpoint_interval=10,
                gradient_clip_norm=1.0e9,
                minimum_zero_context_mse=0.0,
            ),
        )
    finally:
        backward_handle.remove()

    assert trace == ["zero_grad", "backward", "clip", "step"]
    assert captured_gradient is not None
    assert bool((captured_gradient != 0.0).any())
    expected_bias = bias_before - learning_rate * captured_gradient / (
        captured_gradient.abs() + setup.optimizer.defaults["eps"]
    )
    torch.testing.assert_close(bias, expected_bias, rtol=1.0e-5, atol=1.0e-7)
    assert result.final_checkpoint == output_dir / "final.pt"
    assert torch.load(result.final_checkpoint, map_location="cpu", weights_only=True)["step"] == 1
    for name, value in setup.policy.planner.state_dict().items():
        assert torch.equal(value, planner_before[name])
    for name, value in setup.policy.idm.state_dict().items():
        assert torch.equal(value, idm_before[name])


def test_training_owner_zero_step_preserves_context_and_emits_baseline_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    context_before = {
        name: value.detach().clone()
        for name, value in setup.policy.context_encoder.state_dict().items()
    }
    monkeypatch.setattr(
        setup.optimizer,
        "step",
        lambda *_args, **_kwargs: pytest.fail("zero-step run reached optimizer.step"),
    )
    monkeypatch.setattr(
        training_module,
        "evaluate_context_action_mse",
        lambda *_args, **_kwargs: 1.0,
    )
    events: list[str] = []
    output_dir = tmp_path / "zero-step"
    output_dir.mkdir()

    result = run_support_query_training(
        setup,
        batch,
        batch,
        output_dir=output_dir,
        source_checkpoint_sha256="source",
        dataset_sha256="dataset",
        train_split_sha256="train",
        validation_split_sha256="validation",
        split_seed=7,
        resolved_config={},
        config=SupportQueryTrainingLoopConfig(
            steps=0,
            batch_size=2,
            log_interval=1,
            checkpoint_interval=10,
            gradient_clip_norm=1.0,
            minimum_zero_context_mse=0.0,
        ),
        emit=lambda event, **_payload: events.append(event),
    )

    assert {path.name for path in output_dir.iterdir()} == {"best.pt", "final.pt"}
    assert torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)["step"] == 0
    assert torch.load(output_dir / "final.pt", map_location="cpu", weights_only=True)["step"] == 0
    assert result.best_step == 0
    assert events == ["training_started", "training_completed"]
    for name, value in setup.policy.context_encoder.state_dict().items():
        assert torch.equal(value, context_before[name])


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_training_owner_nonfinite_transaction_never_steps_or_saves_post_step_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    config = _config()
    batch = _semantic_batch(config).validate(config, support_length=8)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    context_before = {
        name: value.detach().clone()
        for name, value in setup.policy.context_encoder.state_dict().items()
    }
    monkeypatch.setattr(
        training_module,
        "evaluate_context_action_mse",
        lambda *_args, **_kwargs: 1.0,
    )
    step_calls = 0

    def record_step(*_args, **_kwargs):
        nonlocal step_calls
        step_calls += 1
        pytest.fail("nonfinite transaction reached optimizer.step")

    monkeypatch.setattr(setup.optimizer, "step", record_step)
    gradient_handle = None
    if failure == "loss":
        monkeypatch.setattr(
            training_module,
            "context_first_action_loss",
            lambda *_args, **_kwargs: torch.tensor(float("nan"), requires_grad=True),
        )
        message = "non-finite Context loss"
    else:
        gradient_handle = setup.policy.context_encoder.delta_head.bias.register_hook(
            lambda gradient: torch.full_like(gradient, float("inf"))
        )
        message = "non-finite Context gradient"
    output_dir = tmp_path / f"nonfinite-{failure}"
    output_dir.mkdir()
    try:
        with pytest.raises(ValueError, match=message):
            run_support_query_training(
                setup,
                batch,
                batch,
                output_dir=output_dir,
                source_checkpoint_sha256="source",
                dataset_sha256="dataset",
                train_split_sha256="train",
                validation_split_sha256="validation",
                split_seed=7,
                resolved_config={},
                config=SupportQueryTrainingLoopConfig(
                    steps=1,
                    batch_size=2,
                    log_interval=1,
                    checkpoint_interval=1,
                    gradient_clip_norm=1.0,
                    minimum_zero_context_mse=0.0,
                ),
            )
    finally:
        if gradient_handle is not None:
            gradient_handle.remove()

    assert step_calls == 0
    assert {path.name for path in output_dir.iterdir()} == {"best.pt"}
    assert torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)["step"] == 0
    for name, value in setup.policy.context_encoder.state_dict().items():
        assert torch.equal(value, context_before[name])


@pytest.mark.parametrize("frozen_owner", ["planner", "idm"])
def test_training_owner_rolls_back_frozen_mutation_before_validation_or_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_owner: str,
) -> None:
    config = _config()
    batch = _batch(config, batch_size=2)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    planner_before = {
        name: value.detach().clone() for name, value in setup.policy.planner.state_dict().items()
    }
    idm_before = {
        name: value.detach().clone() for name, value in setup.policy.idm.state_dict().items()
    }
    original_step = setup.optimizer.step

    def mutate_frozen_owner(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            next(getattr(setup.policy, frozen_owner).parameters()).add_(1.0)
        return result

    monkeypatch.setattr(setup.optimizer, "step", mutate_frozen_owner)
    validation_calls = 0

    def record_validation(*_args, **_kwargs):
        nonlocal validation_calls
        validation_calls += 1
        return float(10 - validation_calls)

    monkeypatch.setattr(training_module, "evaluate_context_action_mse", record_validation)
    events: list[str] = []
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="changed during Context training"):
        run_support_query_training(
            setup,
            batch,
            batch,
            output_dir=output_dir,
            source_checkpoint_sha256="source",
            dataset_sha256="dataset",
            train_split_sha256="train",
            validation_split_sha256="validation",
            split_seed=7,
            resolved_config={},
            config=SupportQueryTrainingLoopConfig(
                steps=1,
                batch_size=2,
                log_interval=1,
                checkpoint_interval=10,
                gradient_clip_norm=1.0,
                minimum_zero_context_mse=0.0,
            ),
            emit=lambda event, **_payload: events.append(event),
        )

    assert validation_calls == 2
    assert events == ["training_started"]
    assert {path.name for path in output_dir.iterdir()} == {"best.pt"}
    pre_step = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    assert pre_step["step"] == 0
    for name, value in setup.policy.planner.state_dict().items():
        torch.testing.assert_close(value, planner_before[name], rtol=0.0, atol=0.0)
    for name, value in setup.policy.idm.state_dict().items():
        torch.testing.assert_close(value, idm_before[name], rtol=0.0, atol=0.0)


def test_training_owner_rejects_zero_context_gradient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    batch = _batch(config, batch_size=2)
    setup = prepare_support_query_training(
        FADAPlannerIDMPolicy(config),
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
    )
    monkeypatch.setattr(
        training_module,
        "evaluate_context_action_mse",
        lambda *_args, **_kwargs: 1.0,
    )
    monkeypatch.setattr(
        training_module,
        "context_first_action_loss",
        lambda policy, _batch: sum(
            parameter.sum() * 0.0 for parameter in policy.context_encoder.parameters()
        ),
    )
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="zero Context gradient"):
        run_support_query_training(
            setup,
            batch,
            batch,
            output_dir=output_dir,
            source_checkpoint_sha256="source",
            dataset_sha256="dataset",
            train_split_sha256="train",
            validation_split_sha256="validation",
            split_seed=7,
            resolved_config={},
            config=SupportQueryTrainingLoopConfig(
                steps=1,
                batch_size=2,
                log_interval=1,
                checkpoint_interval=10,
                gradient_clip_norm=1.0,
                minimum_zero_context_mse=0.0,
            ),
        )


def test_training_owner_owns_complete_preflight_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    healthy = FADAPlannerIDMPolicy(config)
    batch = _batch(config, batch_size=2)
    planner_before = {
        name: value.detach().clone() for name, value in healthy.planner.state_dict().items()
    }
    idm_before = {name: value.detach().clone() for name, value in healthy.idm.state_dict().items()}
    monkeypatch.setattr(
        torch.optim.Adam,
        "step",
        lambda *_args, **_kwargs: pytest.fail("preflight must not take an optimizer step"),
    )
    owner = getattr(training_module, "run_support_query_preflight", None)

    assert callable(owner)
    result = owner(
        healthy,
        batch,
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
        learning_rate=3.0e-4,
        minimum_zero_context_mse=0.0,
    )

    assert result.method_contract_id == FADA_CONTEXT_METHOD_CONTRACT_ID
    assert result.delta_z_shape == (2, batch.query.window_count, config.hidden_dim)
    assert result.optimizer_steps == 0
    assert result.context_grad_norm > 0.0
    for name, value in healthy.planner.state_dict().items():
        torch.testing.assert_close(value, planner_before[name], rtol=0.0, atol=0.0)
    for name, value in healthy.idm.state_dict().items():
        torch.testing.assert_close(value, idm_before[name], rtol=0.0, atol=0.0)


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


class _InvalidatingEnv(_IndependentResetEnv):
    def __init__(self, config: FADAArchitectureConfig, event: str) -> None:
        super().__init__(config)
        self.event = event

    def step(self, action: np.ndarray) -> _State:
        state = super().step(action)
        if self.event == "terminated":
            state.terminated[0] = True
        elif self.event == "truncated":
            state.truncated[0] = True
        elif self.event == "command":
            state.info["commands"][0, 0] += 0.1
        else:
            raise AssertionError(f"unknown invalidation event: {self.event}")
        return state


class _PermutedLabeledEnv(_IndependentResetEnv):
    def __init__(
        self,
        config: FADAArchitectureConfig,
        row_order: tuple[int, int],
    ) -> None:
        super().__init__(config)
        self.row_order = np.asarray(row_order, dtype=np.int64)

    def reset_all(self) -> _State:
        self.reset_count += 1
        labels = np.asarray([0.0, 1.0], dtype=np.float32)[self.row_order]
        feature_offsets = np.arange(self.config.obs_dim, dtype=np.float32) / 100.0
        obs = (labels[:, None] * 10.0 + self.reset_count / 10.0 + feature_offsets[None, :]).astype(
            np.float32
        )
        command = np.stack(
            (0.2 + labels / 10.0, -0.2 - labels / 10.0),
            axis=1,
        ).astype(np.float32)
        self.state = _State(
            obs={"obs": obs},
            info={"commands": command},
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
        SupportQueryCollectionConfig(
            num_pairs=2, support_length=8, query_length=10, max_reset_pairs=1
        ),
    )

    assert env.reset_count == 2
    assert env.autoreset is False
    assert result.accepted_pairs == 2
    assert torch.all(result.batch.support_rollout_id == 0)
    assert torch.all(result.batch.query_rollout_id == 1)
    torch.testing.assert_close(
        result.batch.query.executed_action,
        torch.full_like(result.batch.query.executed_action, 0.25),
    )
    assert not torch.equal(
        result.batch.support.realized_state[:, 0],
        result.batch.query.observation_history[:, 0, 0],
    )


def test_collector_row_permutation_covariance_preserves_all_fields_and_opaque_id_roles() -> None:
    torch.manual_seed(108)
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    torch.nn.init.zeros_(policy.idm.action_head.weight)
    torch.nn.init.constant_(policy.idm.action_head.bias, 0.25)
    spec = SupportQueryCollectionConfig(
        num_pairs=2,
        support_length=8,
        query_length=10,
        max_reset_pairs=1,
    )

    original = collect_support_query_pairs(
        _PermutedLabeledEnv(config, (0, 1)),
        policy,
        spec,
    )
    row_order = torch.tensor([1, 0])
    inverse_order = torch.argsort(row_order)
    permuted = collect_support_query_pairs(
        _PermutedLabeledEnv(config, (1, 0)),
        policy,
        spec,
    )
    restored = permuted.batch.index_select(inverse_order)

    for name, expected in _named_batch_tensors(original.batch).items():
        if name in {"pair_id", "support_rollout_id", "query_rollout_id"}:
            continue
        observed = _named_batch_tensors(restored)[name]
        if expected.is_floating_point():
            torch.testing.assert_close(observed, expected, rtol=1.0e-5, atol=1.0e-6)
        else:
            assert torch.equal(observed, expected)
    for name in ("pair_id", "support_rollout_id", "query_rollout_id"):
        expected_ids = getattr(original.batch, name)
        restored_ids = getattr(restored, name)
        assert torch.equal(
            expected_ids[:, None] == expected_ids[None, :],
            restored_ids[:, None] == restored_ids[None, :],
        )
    assert bool(torch.all(original.batch.support_rollout_id != original.batch.query_rollout_id))
    assert bool(torch.all(restored.support_rollout_id != restored.query_rollout_id))
    assert original.accepted_pairs == permuted.accepted_pairs == 2
    assert original.rejected_pairs == permuted.rejected_pairs == 0
    assert original.reset_pairs == permuted.reset_pairs == 1


def test_collector_minimum_query_length_emits_one_causal_window_and_stable_identities() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    torch.nn.init.zeros_(policy.idm.action_head.weight)
    torch.nn.init.constant_(policy.idm.action_head.bias, 0.25)
    minimum_query_length = config.history_length + config.prediction_horizon - 1

    result = collect_support_query_pairs(
        _IndependentResetEnv(config),
        policy,
        SupportQueryCollectionConfig(
            num_pairs=2,
            support_length=8,
            query_length=minimum_query_length,
            max_reset_pairs=1,
        ),
    )

    assert result.batch.query.window_count == 1
    assert result.batch.query.window_anchor.tolist() == [
        [config.history_length - 1],
        [config.history_length - 1],
    ]
    assert result.batch.query.valid_window_mask.tolist() == [[True], [True]]
    torch.testing.assert_close(
        result.batch.query.executed_action,
        torch.full_like(result.batch.query.executed_action, 0.25),
    )
    assert torch.unique(result.batch.pair_id).numel() == 2
    assert torch.unique(result.batch.support_rollout_id).numel() == 1
    assert torch.unique(result.batch.query_rollout_id).numel() == 1
    assert bool(torch.all(result.batch.support_rollout_id != result.batch.query_rollout_id))


def test_collector_nonfinite_policy_output_fails_before_sealed_batch() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    with torch.no_grad():
        policy.idm.action_head.bias.fill_(float("nan"))
    result = None

    with pytest.raises(ValueError, match="non-finite output"):
        result = collect_support_query_pairs(
            _IndependentResetEnv(config),
            policy,
            SupportQueryCollectionConfig(
                num_pairs=2,
                support_length=8,
                query_length=10,
                max_reset_pairs=1,
            ),
        )

    assert result is None


def test_collector_builds_every_causally_aligned_query_window() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    torch.nn.init.zeros_(policy.idm.action_head.weight)
    torch.nn.init.constant_(policy.idm.action_head.bias, 0.25)
    env = _IndependentResetEnv(config)

    result = collect_support_query_pairs(
        env,
        policy,
        SupportQueryCollectionConfig(
            num_pairs=2, support_length=8, query_length=10, max_reset_pairs=1
        ),
    )
    query = result.batch.query

    assert query.window_count == 4
    torch.testing.assert_close(
        query.window_anchor,
        torch.tensor([[4, 5, 6, 7], [4, 5, 6, 7]], dtype=torch.int64),
    )
    assert bool(query.valid_window_mask.all())
    # The fixture starts Query at 0.2 and each executed action adds 0.25 to the first 3 state dims.
    for window, anchor in enumerate((4, 5, 6, 7)):
        expected_state_t = 0.2 + 0.25 * anchor
        expected_next = expected_state_t + 0.25
        torch.testing.assert_close(
            query.observation_history[:, window, -1, :3],
            torch.full((2, 3), expected_state_t),
        )
        torch.testing.assert_close(
            query.realized_future[:, window, 0, :3],
            torch.full((2, 3), expected_next),
        )
        torch.testing.assert_close(query.executed_action[:, window], torch.full((2, 3), 0.25))


def test_collector_rejects_query_too_short_for_one_causal_window() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    env = _IndependentResetEnv(config)

    with pytest.raises(ValueError, match="cannot supply one causal window"):
        collect_support_query_pairs(
            env,
            policy,
            SupportQueryCollectionConfig(
                num_pairs=2,
                support_length=8,
                query_length=config.history_length + config.prediction_horizon - 2,
                max_reset_pairs=1,
            ),
        )


@pytest.mark.parametrize("event", ["terminated", "truncated", "command"])
def test_collector_rejects_rows_with_invalid_rollout_events(event: str) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    env = _InvalidatingEnv(config, event)

    result = collect_support_query_pairs(
        env,
        policy,
        SupportQueryCollectionConfig(
            num_pairs=1,
            support_length=8,
            query_length=10,
            max_reset_pairs=1,
        ),
    )

    assert result.accepted_pairs == 1
    assert result.rejected_pairs == 1
