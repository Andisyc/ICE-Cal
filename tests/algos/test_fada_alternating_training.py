from __future__ import annotations

from pathlib import Path

import pytest
import torch

import unilab.algos.torch.distill.fada_checkpoint as fada_checkpoint
from tests.algos._fada_training_test_support import _config, _source_batch
from unilab.algos.torch.distill import (
    FADA_CHECKPOINT_SCHEMA_VERSION,
    FADAPlannerIDMPolicy,
    FADATrainer,
    load_fada_policy_checkpoint,
    save_fada_checkpoint,
)
from unilab.algos.torch.distill.fada_replay import FADAReplayBuffer


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


def test_planner_only_update_changes_planner_and_preserves_frozen_idm() -> None:
    torch.manual_seed(20260829)
    policy = FADAPlannerIDMPolicy(_config())
    for parameter in policy.idm.parameters():
        parameter.requires_grad_(False)
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )
    replay = FADAReplayBuffer(policy.config, capacity=8)
    replay.add(_source_batch(policy.config, size=4))
    idm_before = _snapshot(policy.idm)
    planner_before = _snapshot(policy.planner)

    update_planner = getattr(trainer, "update_planner_from_replay", None)
    assert callable(update_planner), "Planner-only update boundary is missing"
    stats = update_planner(
        replay,
        batch_size=2,
        planner_updates=2,
        device="cpu",
    )

    assert _changed(planner_before, policy.planner)
    assert not _changed(idm_before, policy.idm)
    assert all(parameter.grad is None for parameter in policy.idm.parameters())
    assert stats.idm_loss == 0.0
    assert stats.idm_grad_norm == 0.0
    assert stats.planner_grad_norm > 0.0


def test_planner_only_update_rejects_trainable_idm() -> None:
    policy = FADAPlannerIDMPolicy(_config())
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )
    replay = FADAReplayBuffer(policy.config, capacity=4)
    replay.add(_source_batch(policy.config, size=2))

    update_planner = getattr(trainer, "update_planner_from_replay", None)
    assert callable(update_planner), "Planner-only update boundary is missing"
    with pytest.raises(ValueError, match="frozen"):
        update_planner(replay, batch_size=2, planner_updates=1, device="cpu")


def test_planner_initialization_loads_only_idm_and_freezes_it(tmp_path: Path) -> None:
    torch.manual_seed(17)
    donor = FADAPlannerIDMPolicy(_config())
    with torch.no_grad():
        for parameter in donor.idm.parameters():
            parameter.fill_(0.25)
        for parameter in donor.planner.parameters():
            parameter.fill_(0.75)
    donor_trainer = FADATrainer(
        donor,
        idm_optimizer=torch.optim.Adam(donor.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(donor.planner.parameters(), lr=1.0e-3),
    )
    checkpoint = tmp_path / "idm_pretrain.pt"
    save_fada_checkpoint(
        checkpoint,
        donor,
        donor_trainer,
        completed_iterations=8,
        samples_seen=64,
        runtime_config={"training_schedule": "idm_pretrain"},
    )

    torch.manual_seed(23)
    target = FADAPlannerIDMPolicy(_config())
    planner_before = _snapshot(target.planner)
    initialize = getattr(fada_checkpoint, "initialize_fada_planner_from_idm", None)
    assert callable(initialize), "IDM-to-Planner checkpoint gateway is missing"

    payload = initialize(checkpoint, target, map_location="cpu")

    assert payload["training_schedule"] == "idm_pretrain"
    assert not _changed(planner_before, target.planner)
    assert all(
        torch.equal(target.idm.state_dict()[name], value)
        for name, value in donor.idm.state_dict().items()
    )
    assert all(not parameter.requires_grad for parameter in target.idm.parameters())


def test_planner_initialization_rejects_non_idm_checkpoint(tmp_path: Path) -> None:
    policy = FADAPlannerIDMPolicy(_config())
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )
    checkpoint = tmp_path / "alternating.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=1,
        samples_seen=4,
        runtime_config={"training_schedule": "alternating_idm_then_planner"},
    )
    initialize = getattr(fada_checkpoint, "initialize_fada_planner_from_idm", None)
    assert callable(initialize), "IDM-to-Planner checkpoint gateway is missing"

    with pytest.raises(ValueError, match="idm_pretrain"):
        initialize(checkpoint, FADAPlannerIDMPolicy(_config()), map_location="cpu")


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
