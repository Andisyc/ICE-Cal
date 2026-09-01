from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.algos._fada_training_test_support import _config, _source_batch
from unilab.algos.torch.distill import (
    FADA_CHECKPOINT_SCHEMA_VERSION,
    FADAPlannerIDMPolicy,
    FADATrainer,
    load_fada_policy_checkpoint,
    save_fada_checkpoint,
)
from unilab.algos.torch.distill.fada.replay import FADAReplayBuffer


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(before[name], value) for name, value in module.state_dict().items())


def test_trainer_orders_idm_then_fixed_idm_planner_updates() -> None:
    torch.manual_seed(11)
    policy = FADAPlannerIDMPolicy(_config())
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )
    batch = _source_batch(policy.config, size=4)
    idm_before = _snapshot(policy.idm)
    planner_before = _snapshot(policy.planner)

    stats = trainer.update(batch, idm_updates=1, planner_updates=1)

    assert _changed(idm_before, policy.idm)
    assert _changed(planner_before, policy.planner)
    assert stats.idm_grad_norm > 0.0
    assert stats.planner_grad_norm > 0.0
    assert all(parameter.grad is None for parameter in policy.idm.parameters())


def test_trainer_allows_idm_only_budget_without_mutating_planner() -> None:
    policy = FADAPlannerIDMPolicy(_config())
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )
    idm_before = _snapshot(policy.idm)
    planner_before = _snapshot(policy.planner)

    stats = trainer.update(
        _source_batch(policy.config, size=2),
        idm_updates=1,
        planner_updates=0,
    )

    assert _changed(idm_before, policy.idm)
    assert not _changed(planner_before, policy.planner)
    assert stats.idm_grad_norm > 0.0
    assert stats.planner_grad_norm == 0.0


def test_non_v005_replay_keeps_uniform_training_without_role_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config())
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )
    replay = FADAReplayBuffer(policy.config, capacity=8)
    replay.add(_source_batch(policy.config, size=8))
    uniform_calls = 0
    original_sample = replay.sample

    def traced_sample(*args, **kwargs):
        nonlocal uniform_calls
        uniform_calls += 1
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(replay, "sample", traced_sample)
    monkeypatch.setattr(
        replay,
        "sample_idm",
        lambda *args, **kwargs: pytest.fail("non-v005 must not use stratified IDM sampling"),
    )
    monkeypatch.setattr(
        replay,
        "sample_planner",
        lambda *args, **kwargs: pytest.fail("non-v005 must not use stratified Planner sampling"),
    )
    idm_before = _snapshot(policy.idm)
    planner_before = _snapshot(policy.planner)

    trainer.update_from_replay(
        replay,
        batch_size=4,
        idm_updates=1,
        planner_updates=1,
        device="cpu",
    )

    assert _changed(idm_before, policy.idm)
    assert _changed(planner_before, policy.planner)
    assert uniform_calls == 2


def test_trainer_rejects_missing_idm_budget() -> None:
    policy = FADAPlannerIDMPolicy(_config())
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )

    for idm_updates, planner_updates in ((0, 1), (0, 0), (1, -1)):
        try:
            trainer.update(
                _source_batch(policy.config, size=2),
                idm_updates=idm_updates,
                planner_updates=planner_updates,
            )
        except ValueError as exc:
            assert "IDM updates must be positive" in str(exc)
        else:
            raise AssertionError("invalid IDM-only budget must fail closed")


def test_schema5_checkpoint_binds_alternating_schedule_and_both_optimizers(
    tmp_path: Path,
) -> None:
    policy = FADAPlannerIDMPolicy(_config())
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=2.0e-3),
    )
    checkpoint = tmp_path / "alternating.pt"

    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=1,
        samples_seen=4,
        runtime_config={"v005_replay": {"enabled": False}},
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert FADA_CHECKPOINT_SCHEMA_VERSION == 5
    assert payload["schema_version"] == 5
    assert payload["training_schedule"] == "alternating_idm_then_planner"
    assert set(payload) >= {
        "idm_optimizer_state_dict",
        "planner_optimizer_state_dict",
        "idm_sha256",
    }
    loaded = load_fada_policy_checkpoint(checkpoint, device="cpu")
    assert loaded.checkpoint["completed_iterations"] == 1


def test_schema5_checkpoint_binds_idm_pretrain_schedule(tmp_path: Path) -> None:
    policy = FADAPlannerIDMPolicy(_config())
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=2.0e-3),
    )
    checkpoint = tmp_path / "idm.pt"

    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=1,
        samples_seen=4,
        runtime_config={
            "training_schedule": "idm_pretrain",
            "v005_replay": {"enabled": False},
        },
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["training_schedule"] == "idm_pretrain"
    assert payload["idm_sha256"]
    load_fada_policy_checkpoint(checkpoint, device="cpu")
