from __future__ import annotations

import importlib
from pathlib import Path

import unilab.algos.torch.distill as distill
import unilab.algos.torch.distill.fada_training as training_facade
import unilab.algos.torch.distill.fada_workflow as workflow_facade

TRAINING_OWNER_SYMBOLS = {
    "fada_source_plan": (
        "FADA_INTERMEDIATE_ORACLE_COUNT",
        "FADAPaperSourcePlan",
        "build_fada_paper_source_plan",
    ),
    "fada_source_artifact": (
        "FADA_SOURCE_BATCH_SCHEMA_VERSION",
        "LoadedFADASourceBatch",
        "load_fada_source_batch",
        "save_fada_source_batch",
    ),
    "fada_replay": (
        "FADAReplayBuffer",
        "FADAReplayRoleCounts",
    ),
    "fada_source_evaluation": ("evaluate_fada_source_batch",),
    "fada_trainer": (
        "FADATrainer",
        "FADATrainingStats",
    ),
    "fada_checkpoint": (
        "FADA_CHECKPOINT_SCHEMA_VERSION",
        "FADA_TRAINING_SCHEDULE",
        "FADA_V005_REQUIRED_QUALITY_METRICS",
        "LoadedFADAPlannerIDMPolicy",
        "load_fada_checkpoint",
        "load_fada_policy_checkpoint",
        "save_fada_checkpoint",
    ),
    "fada_oracle": ("load_fada_oracle_policy",),
}


def test_training_facade_reexports_each_owner_symbol_by_identity() -> None:
    for module_name, symbol_names in TRAINING_OWNER_SYMBOLS.items():
        owner = importlib.import_module(f"unilab.algos.torch.distill.{module_name}")
        for symbol_name in symbol_names:
            assert getattr(training_facade, symbol_name) is getattr(owner, symbol_name)


def test_distill_package_keeps_existing_training_exports_by_identity() -> None:
    public_names = (
        "FADA_CHECKPOINT_SCHEMA_VERSION",
        "FADA_TRAINING_SCHEDULE",
        "FADA_SOURCE_BATCH_SCHEMA_VERSION",
        "FADAReplayBuffer",
        "FADATrainer",
        "FADATrainingStats",
        "LoadedFADAPlannerIDMPolicy",
        "LoadedFADASourceBatch",
        "evaluate_fada_source_batch",
        "load_fada_checkpoint",
        "load_fada_policy_checkpoint",
        "load_fada_oracle_policy",
        "load_fada_source_batch",
        "save_fada_checkpoint",
        "save_fada_source_batch",
    )
    for symbol_name in public_names:
        assert getattr(distill, symbol_name) is getattr(training_facade, symbol_name)


def test_training_facade_keeps_model_symbols_from_pre_refactor_module() -> None:
    model_owner = importlib.import_module("unilab.algos.torch.distill.fada")
    for symbol_name in (
        "FADA_COMMAND_SCENARIOS",
        "FADA_SCENARIO_IDS",
        "FADAArchitectureConfig",
        "FADAInverseDynamicsModel",
        "FADAPlanner",
        "FADAPlannerIDMPolicy",
        "FADASourceBatch",
        "idm_source_loss",
        "planner_source_loss",
    ):
        assert getattr(training_facade, symbol_name) is getattr(model_owner, symbol_name)


def test_workflow_facade_reexports_owner_boundaries_by_identity() -> None:
    setup = importlib.import_module("unilab.algos.torch.distill.fada_workflow_setup")
    admission = importlib.import_module("unilab.algos.torch.distill.fada_artifact_admission")
    persistent = importlib.import_module("unilab.algos.torch.distill.fada_persistent_workflow")
    legacy = importlib.import_module("unilab.algos.torch.distill.fada_legacy_workflow")

    expected = {
        "FADAWorkflowDependencies": setup.FADAWorkflowDependencies,
        "build_fada_architecture_config": setup.build_fada_architecture_config,
        "_paper_source_plan": setup.paper_source_plan,
        "_fada_execution_mode": setup.fada_execution_mode,
        "_fada_v005_replay_settings": setup.fada_v005_replay_settings,
        "_require_fada_curriculum_artifact": admission.require_fada_curriculum_artifact,
        "_slice_fada_batch": admission.slice_fada_batch,
        "_fada_quality_batch": admission.fada_quality_batch,
        "_run_fada_persistent_async": persistent.run_fada_persistent_async,
        "_run_fada_legacy": legacy.run_fada_legacy,
    }
    for symbol_name, owner_symbol in expected.items():
        assert getattr(workflow_facade, symbol_name) is owner_symbol


def test_workflow_facade_keeps_observation_contract_symbols() -> None:
    observation_owner = importlib.import_module("unilab.algos.torch.distill.fada_observation")
    for symbol_name in (
        "FADA_G1_STATE_OBSERVATION_CONTRACT",
        "assert_fada_active_route_contract",
        "assert_fada_projection_matches_contract",
    ):
        assert getattr(workflow_facade, symbol_name) is getattr(observation_owner, symbol_name)


def test_collector_facade_reexports_decomposed_owners_by_identity() -> None:
    facade = importlib.import_module("unilab.algos.torch.distill.fada_collector")
    owners = {
        "fada.collection_contract": (
            "FADACollectionResult",
            "FADACollectionSpec",
            "FADACollectionTransition",
        ),
        "fada.collection_io": (
            "_command_array",
            "_fada_actions",
            "_obs_array",
            "_oracle_shadow_pair",
            "_policy_actions",
        ),
        "fada.collection_transaction": ("collect_fada_source_windows",),
    }
    for module_name, symbol_names in owners.items():
        owner = importlib.import_module(f"unilab.algos.torch.distill.{module_name}")
        for symbol_name in symbol_names:
            assert getattr(facade, symbol_name) is getattr(owner, symbol_name)


def test_async_facade_reexports_config_owner_and_keeps_worker_owner() -> None:
    facade = importlib.import_module("unilab.algos.torch.distill.fada_async_runtime")
    config_owner = importlib.import_module("unilab.algos.torch.distill.fada.async_config")
    collection_owner = importlib.import_module(
        "unilab.algos.torch.distill.fada.async_collection"
    )

    assert facade.allocate_fada_command_scenarios is config_owner.allocate_fada_command_scenarios
    assert facade._curriculum_and_allocations is config_owner.curriculum_and_allocations
    assert facade._standing_owner_cfg is config_owner.standing_owner_cfg
    assert facade._stand_transition_curriculum_cfg is config_owner.stand_transition_curriculum_cfg
    assert facade._v005_replay_cfg is config_owner.v005_replay_cfg
    assert facade.collect_fada_iteration is collection_owner.collect_fada_iteration
    assert facade.PersistentFADACollectorWorker.__module__ == (
        "unilab.algos.torch.distill.fada.async_runtime"
    )


def test_decomposed_owners_do_not_import_compatibility_facades() -> None:
    forbidden = {
        "fada_collection_contract": ("fada_collector", "fada_async_runtime"),
        "fada_collection_io": ("fada_collector", "fada_async_runtime"),
        "fada_collection_windows": ("fada_collector", "fada_async_runtime"),
        "fada_collection_transaction": ("fada_collector", "fada_async_runtime"),
        "fada_async_config": ("fada_async_runtime",),
        "fada_async_collection": ("fada_async_runtime",),
    }
    for module_name, forbidden_names in forbidden.items():
        owner = importlib.import_module(f"unilab.algos.torch.distill.{module_name}")
        source = Path(owner.__file__).read_text(encoding="utf-8")
        for forbidden_name in forbidden_names:
            assert f"import {forbidden_name}" not in source
            assert f"from .{forbidden_name}" not in source


def test_collector_and_async_facades_are_bounded_composition_surfaces() -> None:
    collector = importlib.import_module("unilab.algos.torch.distill.fada_collector")
    async_runtime = importlib.import_module("unilab.algos.torch.distill.fada_async_runtime")

    collector_lines = len(Path(collector.__file__).read_text(encoding="utf-8").splitlines())
    async_lines = len(Path(async_runtime.__file__).read_text(encoding="utf-8").splitlines())
    assert collector_lines <= 120
    assert async_lines <= 500
