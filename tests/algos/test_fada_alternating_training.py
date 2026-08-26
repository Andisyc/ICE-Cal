from __future__ import annotations

from pathlib import Path

import torch

from tests.algos._fada_training_test_support import _config, _source_batch
from unilab.algos.torch.distill import (
    FADA_CHECKPOINT_SCHEMA_VERSION,
    FADAPlannerIDMPolicy,
    FADATrainer,
    load_fada_policy_checkpoint,
    save_fada_checkpoint,
)


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


def test_trainer_rejects_zero_pass_budget() -> None:
    policy = FADAPlannerIDMPolicy(_config())
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
    )

    for idm_updates, planner_updates in ((0, 1), (1, 0)):
        try:
            trainer.update(
                _source_batch(policy.config, size=2),
                idm_updates=idm_updates,
                planner_updates=planner_updates,
            )
        except ValueError as exc:
            assert "must both be positive" in str(exc)
        else:
            raise AssertionError("zero pass budget must fail closed")


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
