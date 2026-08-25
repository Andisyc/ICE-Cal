from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tests.algos._fada_training_test_support import _config, _FakeEnv, _Oracle, _source_batch
from unilab.algos.torch.distill.async_runtime import DaggerCollectRequest
from unilab.algos.torch.distill.fada import (
    FADA_IDM_SOURCE_ROLE_IDS,
    FADAPlannerIDMPolicy,
)
from unilab.algos.torch.distill.fada_adaptation_checkpoint import (
    assert_fada_adaptation_source_checkpoint,
    load_fada_deployable_policy_checkpoint,
)
from unilab.algos.torch.distill.fada_async_runtime import (
    FADA_ASYNC_SCENARIO,
    PersistentFADACollectorWorker,
)
from unilab.algos.torch.distill.fada_checkpoint import (
    load_fada_policy_checkpoint,
    load_pretrained_idm_checkpoint,
    save_fada_checkpoint,
)
from unilab.algos.torch.distill.fada_source_artifact import load_fada_source_batch
from unilab.algos.torch.distill.fada_trainer import FADATrainer
from unilab.algos.torch.distill.fada_training_phase import (
    FADATrainingPhase,
    canonical_module_sha256,
    canonical_state_dict_sha256,
)
from unilab.algos.torch.distill.fada_workflow_setup import resolve_fada_training_phase


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def _state_changed(module: torch.nn.Module, before: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(tensor, before[name]) for name, tensor in module.state_dict().items())


def test_phase_owner_selects_source_families_and_rollout_policy() -> None:
    idm = FADATrainingPhase.IDM_PRETRAIN
    planner = FADATrainingPhase.PLANNER

    assert idm.collect_intermediate_oracles is True
    assert idm.main_rollout_uses_student(iteration=0) is False
    assert idm.main_rollout_uses_student(iteration=3) is False
    assert planner.collect_intermediate_oracles is False
    assert planner.main_rollout_uses_student(iteration=0) is False
    assert planner.main_rollout_uses_student(iteration=3) is True


def _phase_cfg(
    phase: str,
    *,
    pretrained_idm_path: str | None = None,
    checkpoint_path: str = "output.pt",
    paper_source_enabled: bool = True,
):
    return OmegaConf.create(
        {
            "training": {
                "fada": {
                    "phase": phase,
                    "pretrained_idm_path": pretrained_idm_path,
                    "checkpoint_path": checkpoint_path,
                    "paper_source_enabled": paper_source_enabled,
                    "resume_path": None,
                    "initial_weights_path": None,
                }
            }
        }
    )


def test_phase_config_rejects_half_open_and_aliased_states() -> None:
    assert (
        resolve_fada_training_phase(_phase_cfg("idm_pretrain"))
        is FADATrainingPhase.IDM_PRETRAIN
    )
    with pytest.raises(ValueError, match="pretrained_idm_path is required"):
        resolve_fada_training_phase(
            _phase_cfg("planner", paper_source_enabled=False)
        )
    with pytest.raises(ValueError, match="paper_source_enabled=false"):
        resolve_fada_training_phase(
            _phase_cfg("planner", pretrained_idm_path="idm.pt")
        )
    with pytest.raises(ValueError, match="must differ"):
        resolve_fada_training_phase(
            _phase_cfg(
                "planner",
                pretrained_idm_path="same.pt",
                checkpoint_path="same.pt",
                paper_source_enabled=False,
            )
        )
    cfg = _phase_cfg("idm_pretrain")
    cfg.training.fada.resume_path = "resume.pt"
    with pytest.raises(ValueError, match="resume_path and initial_weights_path must both be null"):
        resolve_fada_training_phase(cfg)


@pytest.mark.parametrize(
    ("phase", "paper_source_enabled", "pretrained_name"),
    [("idm_pretrain", True, None), ("planner", False, "pretrained-idm.pt")],
)
def test_phase_config_rejects_existing_output_before_runtime(
    tmp_path: Path,
    phase: str,
    paper_source_enabled: bool,
    pretrained_name: str | None,
) -> None:
    output = tmp_path / "existing-output.pt"
    output.write_bytes(b"do-not-overwrite")
    pretrained = None if pretrained_name is None else str(tmp_path / pretrained_name)
    cfg = _phase_cfg(
        phase,
        pretrained_idm_path=pretrained,
        checkpoint_path=str(output),
        paper_source_enabled=paper_source_enabled,
    )

    with pytest.raises(FileExistsError, match="fresh output"):
        resolve_fada_training_phase(cfg)

    assert output.read_bytes() == b"do-not-overwrite"


def test_idm_phase_updates_only_idm() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    idm_before = _state(policy.idm)
    planner_before = _state(policy.planner)
    trainer = FADATrainer(
        policy,
        phase=FADATrainingPhase.IDM_PRETRAIN,
        optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
    )

    stats = trainer.update(_source_batch(config, size=4), updates=1)

    assert stats.idm_loss is not None
    assert stats.planner_loss is None
    assert _state_changed(policy.idm, idm_before)
    assert not _state_changed(policy.planner, planner_before)


def test_trainer_rejects_duplicate_optimizer_parameter() -> None:
    policy = FADAPlannerIDMPolicy(_config())
    parameters = list(policy.idm.parameters())
    with pytest.warns(UserWarning, match="duplicate parameters"):
        optimizer = torch.optim.SGD([*parameters, parameters[0]], lr=1.0e-3)

    with pytest.raises(ValueError, match="exactly idm parameters"):
        FADATrainer(
            policy,
            phase=FADATrainingPhase.IDM_PRETRAIN,
            optimizer=optimizer,
        )


def test_planner_phase_updates_only_planner_and_keeps_idm_sealed() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    idm_before = _state(policy.idm)
    idm_digest = canonical_module_sha256(policy.idm)
    planner_before = _state(policy.planner)
    trainer = FADATrainer(
        policy,
        phase=FADATrainingPhase.PLANNER,
        optimizer=torch.optim.Adam(policy.planner.parameters(), lr=1.0e-3),
        pretrained_idm_sha256=idm_digest,
    )

    stats = trainer.update(_source_batch(config, size=4), updates=1)

    assert stats.idm_loss is None
    assert stats.planner_loss is not None
    assert not _state_changed(policy.idm, idm_before)
    assert _state_changed(policy.planner, planner_before)
    assert canonical_module_sha256(policy.idm) == idm_digest
    assert policy.idm.training is False
    assert all(parameter.requires_grad is False for parameter in policy.idm.parameters())
    assert all(parameter.grad is None for parameter in policy.idm.parameters())


def test_canonical_idm_digest_is_order_independent_and_value_sensitive() -> None:
    state = _state(FADAPlannerIDMPolicy(_config()).idm)
    reversed_state = dict(reversed(tuple(state.items())))
    changed_state = {name: tensor.clone() for name, tensor in state.items()}
    first_name = next(iter(changed_state))
    changed_state[first_name].view(-1)[0] += 1.0

    assert canonical_state_dict_sha256(reversed_state) == canonical_state_dict_sha256(state)
    assert canonical_state_dict_sha256(changed_state) != canonical_state_dict_sha256(state)


def test_schema4_completed_idm_checkpoint_loads_only_idm(tmp_path: Path) -> None:
    config = _config()
    source = FADAPlannerIDMPolicy(config)
    trainer = FADATrainer(
        source,
        phase=FADATrainingPhase.IDM_PRETRAIN,
        optimizer=torch.optim.Adam(source.idm.parameters(), lr=1.0e-3),
    )
    checkpoint = tmp_path / "idm.pt"
    save_fada_checkpoint(
        checkpoint,
        source,
        trainer,
        completed_iterations=2,
        samples_seen=8,
        runtime_config={},
        phase_completed=True,
    )
    target = FADAPlannerIDMPolicy(config)
    with torch.no_grad():
        for parameter in target.planner.parameters():
            parameter.fill_(7.0)
    planner_before = _state(target.planner)

    loaded_digest = load_pretrained_idm_checkpoint(checkpoint, target, map_location="cpu")

    deployable = load_fada_deployable_policy_checkpoint(checkpoint, device="cpu")
    assert deployable.checkpoint["schema_version"] == 4

    assert loaded_digest == canonical_module_sha256(source.idm)
    assert canonical_module_sha256(target.idm) == loaded_digest
    assert not _state_changed(target.planner, planner_before)


@pytest.mark.parametrize("schema_version", [1, 2, 4])
def test_adaptation_source_contract_rejects_non_schema3(schema_version: int) -> None:
    with pytest.raises(ValueError, match="requires schema-3"):
        assert_fada_adaptation_source_checkpoint(
            type("Loaded", (), {"checkpoint": {"schema_version": schema_version}})()
        )


def test_pretrained_idm_loader_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    trainer = FADATrainer(
        policy,
        phase=FADATrainingPhase.IDM_PRETRAIN,
        optimizer=torch.optim.Adam(policy.idm.parameters(), lr=1.0e-3),
    )
    checkpoint = tmp_path / "incomplete-idm.pt"
    save_fada_checkpoint(
        checkpoint,
        policy,
        trainer,
        completed_iterations=1,
        samples_seen=4,
        runtime_config={},
        phase_completed=False,
    )

    with pytest.raises(ValueError, match="completed IDM-pretrain"):
        load_pretrained_idm_checkpoint(
            checkpoint,
            FADAPlannerIDMPolicy(config),
            map_location="cpu",
        )


def test_pretrained_idm_loader_rejects_tampered_identity_before_mutation(
    tmp_path: Path,
) -> None:
    config = _config()
    source = FADAPlannerIDMPolicy(config)
    trainer = FADATrainer(
        source,
        phase=FADATrainingPhase.IDM_PRETRAIN,
        optimizer=torch.optim.Adam(source.idm.parameters(), lr=1.0e-3),
    )
    checkpoint = tmp_path / "tampered-idm.pt"
    save_fada_checkpoint(
        checkpoint,
        source,
        trainer,
        completed_iterations=1,
        samples_seen=4,
        runtime_config={},
        phase_completed=True,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    first_name = next(iter(payload["idm_state_dict"]))
    payload["idm_state_dict"][first_name].view(-1)[0] += 1.0
    torch.save(payload, checkpoint)
    target = FADAPlannerIDMPolicy(config)
    before = _state(target.idm)

    with pytest.raises(ValueError, match="identity mismatch"):
        load_fada_policy_checkpoint(checkpoint, device="cpu")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_pretrained_idm_checkpoint(checkpoint, target, map_location="cpu")

    assert not _state_changed(target.idm, before)


@pytest.mark.parametrize("corruption", ["missing", "extra"])
def test_pretrained_idm_loader_rejects_malformed_optimizer_owner_before_mutation(
    tmp_path: Path,
    corruption: str,
) -> None:
    config = _config()
    source = FADAPlannerIDMPolicy(config)
    trainer = FADATrainer(
        source,
        phase=FADATrainingPhase.IDM_PRETRAIN,
        optimizer=torch.optim.Adam(source.idm.parameters(), lr=1.0e-3),
    )
    checkpoint = tmp_path / f"malformed-{corruption}.pt"
    save_fada_checkpoint(
        checkpoint,
        source,
        trainer,
        completed_iterations=1,
        samples_seen=4,
        runtime_config={},
        phase_completed=True,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if corruption == "missing":
        payload.pop("optimizer_state_dict")
    else:
        payload["idm_optimizer_state_dict"] = {}
    torch.save(payload, checkpoint)
    target = FADAPlannerIDMPolicy(config)
    before = _state(target.idm)

    with pytest.raises(ValueError, match="optimizer"):
        load_pretrained_idm_checkpoint(checkpoint, target, map_location="cpu")

    assert not _state_changed(target.idm, before)


@pytest.mark.parametrize(
    ("phase", "expected_main_mode", "expected_intermediate_loads"),
    [
        (FADATrainingPhase.IDM_PRETRAIN, "oracle", 2),
        (FADATrainingPhase.PLANNER, "planner_idm", 0),
    ],
)
def test_async_collection_obeys_phase_source_family(
    tmp_path: Path,
    phase: FADATrainingPhase,
    expected_main_mode: str,
    expected_intermediate_loads: int,
) -> None:
    config = _config()
    worker = PersistentFADACollectorWorker.__new__(PersistentFADACollectorWorker)
    worker.config = config
    worker.device = "cpu"
    worker.cfg = __import__("omegaconf").OmegaConf.create(
        {
            "training": {
                "fada": {
                    "phase": phase.value,
                    "windows_per_iteration": 1,
                    "oracle_shadow_enabled": True,
                    "observation_key": "obs",
                    "teacher_projection": "identity",
                    "student_projection": "identity",
                    "student_drop_index": None,
                    "command_info_keys": ["commands"],
                    "max_env_steps": 12,
                }
            }
        }
    )
    worker.env = _FakeEnv()
    worker.standing_env = None
    worker.student = FADAPlannerIDMPolicy(config)
    worker.final_teacher = _Oracle()
    worker.standing_teacher = None
    worker.teacher_spec = object()
    worker.source_allocations = tuple(
        (str(tmp_path / name), 1)
        for name in ("intermediate-a.pt", "intermediate-b.pt")
    )
    worker.intermediate_teacher = None
    worker.intermediate_teacher_checkpoint = None
    loads: list[str] = []
    reloads: list[str] = []

    def load_teacher(path, *_args, **_kwargs):
        loads.append(str(path))
        return _Oracle()

    worker._teacher_loader = load_teacher
    def reload_teacher(_teacher, path, *_args, **_kwargs):
        reloads.append(str(path))

    worker._teacher_reloader = reload_teacher

    class _WeightSync:
        def read_weights_into(self, _state_dict) -> int:
            return 1

    worker.weight_sync = _WeightSync()
    output = tmp_path / f"{phase.value}.pt"
    worker.collect(
        DaggerCollectRequest(
            request_id=f"{phase.value}-1",
            scenario=FADA_ASYNC_SCENARIO,
            iteration=1,
            checkpoint_path=str(tmp_path / "policy.pt"),
            output_path=str(output),
            expected_weight_version=1,
        )
    )

    loaded = load_fada_source_batch(output, config=config)
    modes = [item["rollout_mode"] for item in loaded.metadata["collections"]]
    assert modes[0] == expected_main_mode
    assert len(loads) + len(reloads) == expected_intermediate_loads
    assert ("intermediate_oracle" in modes) is phase.collect_intermediate_oracles
    if phase is FADATrainingPhase.IDM_PRETRAIN:
        assert modes == ["oracle", "intermediate_oracle", "intermediate_oracle"]
        assert loaded.batch.idm_source_role.tolist() == [
            FADA_IDM_SOURCE_ROLE_IDS["oracle_shadow"],
            FADA_IDM_SOURCE_ROLE_IDS["trajectory"],
            FADA_IDM_SOURCE_ROLE_IDS["trajectory"],
        ]
    else:
        assert modes == ["planner_idm"]
        assert loaded.batch.planner_eligible.tolist() == [True]
