from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from unilab.algos.torch.distill import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada_target_data import FADATargetBatch


def _owner() -> Any:
    try:
        return importlib.import_module("unilab.algos.torch.distill.fada_adaptation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Stage-D adaptation owner is missing: {exc}")


def _config() -> FADAArchitectureConfig:
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


def _inputs(config: FADAArchitectureConfig) -> tuple[torch.Tensor, ...]:
    observation_history = (
        torch.arange(2 * config.history_length * config.obs_dim, dtype=torch.float32).reshape(
            2, config.history_length, config.obs_dim
        )
        / 17.0
    )
    action_history = (
        torch.arange(2 * config.history_length * config.action_dim, dtype=torch.float32).reshape(
            2, config.history_length, config.action_dim
        )
        / 11.0
    )
    command = torch.tensor([[0.4, -0.2, 0.1], [-0.3, 0.25, 0.05]])
    return observation_history, action_history, command


def _target_batch(config: FADAArchitectureConfig) -> FADATargetBatch:
    rows = 6
    observation_history = (
        torch.arange(rows * config.history_length * config.obs_dim, dtype=torch.float32).reshape(
            rows, config.history_length, config.obs_dim
        )
        / 29.0
    )
    action_history = (
        torch.arange(rows * config.history_length * config.action_dim, dtype=torch.float32).reshape(
            rows, config.history_length, config.action_dim
        )
        / 23.0
    )
    command = torch.tensor(
        [
            [0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.3, 0.1, 0.0],
            [0.3, 0.1, 0.0],
            [0.2, -0.1, 0.0],
            [0.2, -0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    realized_future = observation_history[:, : config.prediction_horizon] + 0.125
    executed_action_chunk = torch.tensor(
        [
            [[0.10, -0.20], [9.0, 8.0]],
            [[0.20, -0.10], [7.0, 6.0]],
            [[0.30, 0.05], [5.0, 4.0]],
            [[0.40, 0.15], [3.0, 2.0]],
            [[0.50, 0.25], [1.0, 0.0]],
            [[0.60, 0.35], [-1.0, -2.0]],
        ],
        dtype=torch.float32,
    )
    return FADATargetBatch(
        observation_history=observation_history,
        action_history=action_history,
        command=command,
        realized_future=realized_future,
        executed_action_chunk=executed_action_chunk,
        episode_id=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
        start_timestep=torch.tensor([3, 4, 11, 12, 21, 22], dtype=torch.int64),
    ).validate(config)


def test_lora_injection_is_zero_delta_and_has_exact_trainable_owner() -> None:
    owner = _owner()
    torch.manual_seed(7)
    policy = FADAPlannerIDMPolicy(_config()).eval()
    inputs = _inputs(policy.config)
    baseline = policy(*inputs)

    adapted = owner.inject_fada_idm_lora(policy, owner.FADALoRAConfig())
    observed = adapted.policy(*inputs)

    assert adapted.lora_config.rank == 8
    assert adapted.lora_config.alpha == 16.0
    assert adapted.lora_config.dropout == 0.05
    assert adapted.lora_config.target_modules == (
        "observation_embedding",
        "action_embedding",
        "history_encoder.layers.0.linear1",
        "history_encoder.layers.0.linear2",
        "future_embedding",
        "future_decoder.layers.0.linear1",
        "future_decoder.layers.0.linear2",
        "action_head",
    )
    torch.testing.assert_close(
        observed.predicted_future, baseline.predicted_future, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(observed.action_chunk, baseline.action_chunk, rtol=1e-6, atol=2e-7)

    trainable = dict(owner.fada_adapter_named_parameters(adapted.policy))
    assert trainable
    assert all(name.endswith(("lora_A.weight", "lora_B.weight")) for name in trainable)
    assert {
        name for name, parameter in adapted.policy.named_parameters() if parameter.requires_grad
    } == (set(trainable))
    assert all(not parameter.requires_grad for parameter in adapted.policy.planner.parameters())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rank": 0}, "rank"),
        ({"alpha": 0.0}, "alpha"),
        ({"dropout": 1.0}, "dropout"),
        ({"target_modules": ("action_head",)}, "manifest"),
    ],
)
def test_lora_injection_rejects_invalid_or_incomplete_manifest(
    kwargs: dict[str, Any], match: str
) -> None:
    owner = _owner()
    policy = FADAPlannerIDMPolicy(_config())

    with pytest.raises(ValueError, match=match):
        owner.inject_fada_idm_lora(policy, owner.FADALoRAConfig(**kwargs))


def test_lora_backward_reaches_every_adapter_and_no_frozen_parameter() -> None:
    owner = _owner()
    torch.manual_seed(11)
    adapted = owner.inject_fada_idm_lora(
        FADAPlannerIDMPolicy(_config()), owner.FADALoRAConfig(dropout=0.0)
    )
    batch = _target_batch(adapted.policy.config)

    loss = owner.fada_adaptation_loss(adapted.policy, batch)
    loss.backward()

    adapters = dict(owner.fada_adapter_named_parameters(adapted.policy))
    assert all(parameter.grad is not None for parameter in adapters.values())
    assert all(torch.isfinite(parameter.grad).all() for parameter in adapters.values())
    assert any(
        bool(torch.any(parameter.grad != 0.0))
        for name, parameter in adapters.items()
        if name.endswith("lora_B.weight")
    )
    assert all(
        parameter.grad is None
        for name, parameter in adapted.policy.named_parameters()
        if name not in adapters
    )


def test_target_split_is_episode_owned_deterministic_and_permutation_safe() -> None:
    owner = _owner()
    batch = _target_batch(_config())

    first = owner.split_fada_target_batch(batch, validation_fraction=0.34, seed=19)
    second = owner.split_fada_target_batch(batch, validation_fraction=0.34, seed=19)

    torch.testing.assert_close(first.train_indices, second.train_indices)
    torch.testing.assert_close(first.validation_indices, second.validation_indices)
    train_episodes = set(batch.episode_id[first.train_indices].tolist())
    validation_episodes = set(batch.episode_id[first.validation_indices].tolist())
    assert train_episodes.isdisjoint(validation_episodes)
    assert sorted(first.train_indices.tolist() + first.validation_indices.tolist()) == list(
        range(6)
    )

    permutation = torch.tensor([4, 0, 2, 5, 1, 3], dtype=torch.int64)
    permuted = owner.select_fada_target_rows(batch, permutation)
    permuted_split = owner.split_fada_target_batch(permuted, validation_fraction=0.34, seed=19)
    assert set(permuted.episode_id[permuted_split.train_indices].tolist()) == train_episodes
    assert (
        set(permuted.episode_id[permuted_split.validation_indices].tolist()) == validation_episodes
    )


def test_target_split_supports_one_episode_with_temporal_purge() -> None:
    owner = _owner()
    batch = _target_batch(_config())
    repeats = 3
    long_episode = FADATargetBatch(
        **{
            name: (
                torch.zeros(len(batch.episode_id) * repeats, dtype=torch.int64)
                if name == "episode_id"
                else (
                    torch.arange(len(batch.episode_id) * repeats, dtype=torch.int64)
                    if name == "start_timestep"
                    else torch.cat([getattr(batch, name)] * repeats)
                )
            )
            for name in FADATargetBatch.__dataclass_fields__
        }
    )
    split = owner.split_fada_target_batch(long_episode, validation_fraction=0.34, seed=0)
    assert int(split.train_indices.max()) + 4 <= int(split.validation_indices.min())
    with pytest.raises(ValueError, match="unique"):
        owner.select_fada_target_rows(batch, torch.tensor([0, 0]))


def test_adaptation_loss_is_exact_first_action_mse() -> None:
    owner = _owner()
    torch.manual_seed(13)
    adapted = owner.inject_fada_idm_lora(
        FADAPlannerIDMPolicy(_config()), owner.FADALoRAConfig(dropout=0.0)
    )
    batch = _target_batch(adapted.policy.config)
    predicted = adapted.policy.idm(
        batch.observation_history, batch.action_history, batch.realized_future
    )
    expected = F.mse_loss(predicted[:, 0], batch.executed_action_chunk[:, 0])

    observed = owner.fada_adaptation_loss(adapted.policy, batch)
    changed_later = FADATargetBatch(
        **{
            **{name: getattr(batch, name) for name in FADATargetBatch.__dataclass_fields__},
            "executed_action_chunk": torch.cat(
                (batch.executed_action_chunk[:, :1], batch.executed_action_chunk[:, 1:] + 1000.0),
                dim=1,
            ),
        }
    )

    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(owner.fada_adaptation_loss(adapted.policy, changed_later), expected)


def test_one_update_changes_only_adapters_and_steps_exactly_once() -> None:
    owner = _owner()
    torch.manual_seed(17)
    adapted = owner.inject_fada_idm_lora(
        FADAPlannerIDMPolicy(_config()), owner.FADALoRAConfig(dropout=0.0)
    )
    adapters = dict(owner.fada_adapter_named_parameters(adapted.policy))

    class _CountingSGD(torch.optim.SGD):
        steps = 0

        def step(self, closure=None):
            self.steps += 1
            return super().step(closure)

    optimizer = _CountingSGD(adapters.values(), lr=0.05)
    trainer = owner.FADAAdaptationTrainer(
        adapted.policy,
        optimizer,
        lora_config=adapted.lora_config,
    )
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in adapted.policy.named_parameters()
        if name not in adapters
    }
    adapter_before = {name: parameter.detach().clone() for name, parameter in adapters.items()}

    stats = trainer.update(_target_batch(adapted.policy.config))

    assert optimizer.steps == 1
    assert stats.optimizer_steps == 1
    assert stats.loss >= 0.0
    assert stats.grad_norm > 0.0
    assert any(
        not torch.equal(adapter_before[name], parameter.detach())
        for name, parameter in adapters.items()
    )
    for name, before in frozen_before.items():
        torch.testing.assert_close(
            dict(adapted.policy.named_parameters())[name], before, rtol=0, atol=0
        )


def test_trainer_rejects_optimizer_ownership_that_includes_base_parameter() -> None:
    owner = _owner()
    adapted = owner.inject_fada_idm_lora(FADAPlannerIDMPolicy(_config()), owner.FADALoRAConfig())
    wrong = list(owner.fada_adapter_parameters(adapted.policy)) + [
        next(adapted.policy.planner.parameters())
    ]

    with pytest.raises(ValueError, match="only adapter"):
        owner.FADAAdaptationTrainer(
            adapted.policy,
            torch.optim.SGD(wrong, lr=0.01),
            lora_config=adapted.lora_config,
        )
