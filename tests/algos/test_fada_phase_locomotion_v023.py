from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.adaptation_checkpoint import (
    assert_fada_adaptation_source_checkpoint,
)
from unilab.algos.torch.distill.fada.checkpoint import LoadedFADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.oracle import (
    LoadedFADAPrivilegedOraclePolicy,
    validate_loaded_fada_oracle_lineage,
)
from unilab.algos.torch.distill.fada.privileged_oracle import (
    FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION,
    FADA_ORACLE_FINAL_ITERATION,
    FADA_ORACLE_INTERMEDIATE_ITERATIONS,
    FADA_ORACLE_PHASE_LOCOMOTION_PROFILE,
    FADA_ORACLE_PHASE_NEUTRAL_PROFILE,
    FADAOracleCheckpointContract,
    normalize_fada_oracle_checkpoint_identity,
    validate_fada_oracle_lineage,
)

ROOT = Path(__file__).resolve().parents[2]


def _contract(*, behavior_profile: str) -> FADAOracleCheckpointContract:
    return FADAOracleCheckpointContract(
        oracle_lineage_id="phase-lineage",
        privileged_schema="g1_fada_privileged_v1",
        task_name="G1WalkFlat",
        backend="mujoco",
        action_scale=(1.0,),
        seed=1,
        obs_dim=98,
        critic_obs_dim=276,
        action_dim=29,
        body_names=("world", "pelvis"),
        actuated_joint_names=tuple(f"joint_{index}" for index in range(29)),
        privileged_field_slices=(("privileged", 0, 178),),
        asset_sha256="a" * 64,
        config_hashes=(("env", "b" * 64),),
        behavior_profile=behavior_profile,
    )


def _compose(config_dir: Path, config_name: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides))


def test_phase_oracle_checkpoint_seals_explicit_behavior_profile() -> None:
    record = (
        _contract(behavior_profile=FADA_ORACLE_PHASE_LOCOMOTION_PROFILE)
        .identity_for_iteration(FADA_ORACLE_FINAL_ITERATION)
        .to_record()
    )

    assert record["schema_version"] == FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION == 3
    assert record["behavior_profile"] == FADA_ORACLE_PHASE_LOCOMOTION_PROFILE


def test_legacy_oracle_schema_is_only_phase_neutral() -> None:
    legacy = (
        _contract(behavior_profile=FADA_ORACLE_PHASE_NEUTRAL_PROFILE)
        .identity_for_iteration(FADA_ORACLE_FINAL_ITERATION)
        .to_record()
    )
    legacy["schema_version"] = 2
    legacy.pop("behavior_profile")

    normalized = normalize_fada_oracle_checkpoint_identity(legacy)

    assert normalized["behavior_profile"] == FADA_ORACLE_PHASE_NEUTRAL_PROFILE
    legacy["behavior_profile"] = FADA_ORACLE_PHASE_LOCOMOTION_PROFILE
    with pytest.raises(ValueError, match="legacy.*phase-neutral"):
        normalize_fada_oracle_checkpoint_identity(legacy)


def test_oracle_lineage_rejects_mixed_behavior_profiles() -> None:
    records = [
        _contract(behavior_profile=FADA_ORACLE_PHASE_LOCOMOTION_PROFILE)
        .identity_for_iteration(iteration)
        .to_record()
        for iteration in (*FADA_ORACLE_INTERMEDIATE_ITERATIONS, FADA_ORACLE_FINAL_ITERATION)
    ]
    records[0]["behavior_profile"] = FADA_ORACLE_PHASE_NEUTRAL_PROFILE

    with pytest.raises(ValueError, match="behavior_profile"):
        validate_fada_oracle_lineage(records)


def test_oracle_lineage_rejects_mixed_schema_versions() -> None:
    records = [
        _contract(behavior_profile=FADA_ORACLE_PHASE_LOCOMOTION_PROFILE)
        .identity_for_iteration(iteration)
        .to_record()
        for iteration in (*FADA_ORACLE_INTERMEDIATE_ITERATIONS, FADA_ORACLE_FINAL_ITERATION)
    ]
    records[0]["schema_version"] = 2
    records[0].pop("behavior_profile")

    with pytest.raises(ValueError, match="checkpoint schema mismatch"):
        validate_fada_oracle_lineage(records)


def test_phase_oracle_and_stage_b_configs_share_walk_only_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "phase-lineage")
    oracle = _compose(
        ROOT / "conf/offpolicy",
        "config",
        "task=sac/g1_walk_flat/mujoco_fada_phase",
    )
    stage_b = _compose(
        ROOT / "conf/distill",
        "config",
        "task=g1_walk_flat/mujoco_fada_phase_locomotion",
    )

    assert oracle.algo.actor.oracle_behavior_profile == FADA_ORACLE_PHASE_LOCOMOTION_PROFILE
    assert oracle.env.gait_phase_enabled is True
    assert oracle.env.commands.rel_standing_envs == pytest.approx(0.0)
    assert oracle.reward.scales.feet_phase == pytest.approx(5.0)
    assert stage_b.training.fada.source_behavior_profile == FADA_ORACLE_PHASE_LOCOMOTION_PROFILE
    assert stage_b.teacher.behavior_profile == FADA_ORACLE_PHASE_LOCOMOTION_PROFILE
    assert stage_b.env.gait_phase_enabled is True
    assert stage_b.env.commands.rel_standing_envs == pytest.approx(0.0)
    assert stage_b.training.fada.stand_transition_curriculum.enabled is False
    assert stage_b.training.fada.v005_replay.enabled is False

    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        resolve_privileged_locomotion_sac_runtime,
    )

    resolved_algo = OmegaConf.to_container(oracle.algo, resolve=True)
    assert isinstance(resolved_algo, dict)
    runtime = resolve_privileged_locomotion_sac_runtime(cast(dict[str, Any], resolved_algo))
    assert runtime is not None
    runtime.validate_training_config(oracle)


def test_slope_stage_c_uses_phase_locomotion_source_contract() -> None:
    cfg = _compose(ROOT / "conf/offpolicy", "fada_slope_target", "target_domain=slope_10")

    assert cfg.collection.source_behavior_profile == FADA_ORACLE_PHASE_LOCOMOTION_PROFILE
    assert cfg.env.gait_phase_enabled is True
    assert cfg.env.commands.rel_standing_envs == pytest.approx(0.0)
    assert cfg.reward.scales.feet_phase == pytest.approx(5.0)
    assert "phase_v023" in cfg.collection.output_dir


def test_slope_stage_c_rejects_same_shape_phase_neutral_override() -> None:
    cfg = _compose(
        ROOT / "conf/offpolicy",
        "fada_slope_target",
        "target_domain=slope_10",
        "env.gait_phase_enabled=false",
    )
    from unilab.algos.torch.distill.fada.target_domain import (
        assert_phase_locomotion_target_environment,
    )

    with pytest.raises(ValueError, match="gait_phase_enabled"):
        assert_phase_locomotion_target_environment(
            cfg,
            behavior_profile=FADA_ORACLE_PHASE_LOCOMOTION_PROFILE,
        )


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ("env.ctrl_dt=0.01", "ctrl_dt"),
        ("env.mode_observation=true", "mode_observation"),
        ("env.commands.resampling_time=1.0", "resampling_time"),
        ("env.commands.heading_command=true", "heading_command"),
        ("reward.gait_constraint.enabled=true", "gait_constraint.enabled"),
        ("reward.gait_constraint.penalty_scale=1.0", "gait_constraint.penalty_scale"),
    ],
)
def test_slope_stage_c_rejects_phase_profile_semantic_overrides(
    override: str,
    field: str,
) -> None:
    cfg = _compose(
        ROOT / "conf/offpolicy",
        "fada_slope_target",
        "target_domain=slope_10",
        override,
    )

    from unilab.algos.torch.distill.fada.target_domain import (
        assert_phase_locomotion_target_environment,
    )

    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        assert_phase_locomotion_target_environment(
            cfg,
            behavior_profile=FADA_ORACLE_PHASE_LOCOMOTION_PROFILE,
        )


def test_stage_b_rejects_phase_neutral_lineage_before_training_setup() -> None:
    contract = _contract(behavior_profile=FADA_ORACLE_PHASE_NEUTRAL_PROFILE)
    policies = []
    for iteration in (*FADA_ORACLE_INTERMEDIATE_ITERATIONS, FADA_ORACLE_FINAL_ITERATION):
        policies.append(
            LoadedFADAPrivilegedOraclePolicy(
                actor=torch.nn.Linear(1, 1),
                obs_dim=98,
                critic_obs_dim=276,
                action_dim=29,
                obs_normalizer=None,
                checkpoint_identity=contract.identity_for_iteration(iteration).to_record(),
            )
        )

    with pytest.raises(ValueError, match="lineage behavior profile mismatch"):
        validate_loaded_fada_oracle_lineage(
            policies[-1],
            policies[:-1],
            expected_behavior_profile=FADA_ORACLE_PHASE_LOCOMOTION_PROFILE,
        )


def test_stage_d_rejects_same_shape_phase_neutral_source() -> None:
    loaded = cast(
        LoadedFADAPlannerIDMPolicy,
        SimpleNamespace(
            checkpoint={
                "schema_version": 5,
                "runtime_config": {
                    "source_behavior_profile": FADA_ORACLE_PHASE_NEUTRAL_PROFILE,
                },
            },
        ),
    )

    with pytest.raises(ValueError, match="behavior profile"):
        assert_fada_adaptation_source_checkpoint(
            loaded,
            expected_behavior_profile=FADA_ORACLE_PHASE_LOCOMOTION_PROFILE,
        )
