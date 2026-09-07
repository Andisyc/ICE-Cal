from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.observation import (
    FADA_G1_ACTION_DIM,
    FADA_G1_ACTOR_OBS_DIM,
    FADA_G1_COMMAND_DIM,
    FADA_G1_STATE_DIM,
)
from unilab.algos.torch.distill.fada_privileged_oracle import (
    FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE,
    FADA_ORACLE_PHASE_LOCOMOTION_PROFILE,
    FADAOracleCheckpointContract,
    seal_fada_oracle_checkpoint,
    validate_fada_oracle_behavior_environment,
    validate_fada_oracle_checkpoint_payload,
    validate_fada_single_reward,
)
from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
    resolve_privileged_locomotion_sac_runtime,
)
from unilab.envs.locomotion.g1.walk_observation import (
    assemble_walk_observation,
    build_obs_groups_spec,
)

ROOT = Path(__file__).resolve().parents[2]


def _compose_profile(monkeypatch: pytest.MonkeyPatch, task: str):
    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "v024-test-lineage")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        return compose("config", overrides=["algo=sac", f"task={task}"])


def test_v024_profile_composes_as_one_final_command_gated_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose_profile(monkeypatch, "sac/g1_walk_flat/mujoco_fada_phase_contact")

    assert cfg.algo.actor.oracle_behavior_profile == FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE
    assert cfg.env.gait_phase_enabled is True
    assert cfg.env.gait_phase_init_mode == "command_gated"
    assert cfg.env.commands.rel_standing_envs == pytest.approx(0.3)
    assert cfg.env.commands.rel_transition_envs == pytest.approx(0.0)
    assert cfg.env.commands.resampling_time == pytest.approx(4.0)
    assert cfg.env.commands.heading_command is False
    assert cfg.env.command_gated_phase_contact.enabled is True
    assert cfg.env.command_gated_phase_contact.command_xy_dead_zone == pytest.approx(0.1)
    assert cfg.env.command_gated_phase_contact.command_yaw_dead_zone == pytest.approx(0.1)
    assert cfg.env.command_gated_phase_contact.min_frequency == pytest.approx(0.7)
    assert cfg.env.command_gated_phase_contact.max_frequency == pytest.approx(1.5)
    assert cfg.env.command_gated_phase_contact.duty_factor == pytest.approx(0.55)
    assert cfg.env.command_gated_phase_contact.contact_force_threshold == pytest.approx(1.0)
    assert cfg.env.command_gated_phase_contact.force_balance_epsilon == pytest.approx(1.0e-6)
    assert cfg.env.domain_rand.actuator_strength.enabled is False
    assert cfg.env.domain_rand.actuator_strength.curriculum_enabled is False
    assert cfg.env.domain_rand.actuator_strength.group_curriculum_enabled is False
    assert cfg.env.domain_rand.randomize_kp is True
    assert cfg.env.domain_rand.randomize_kd is True
    assert cfg.reward.scales.feet_phase == pytest.approx(0.0)
    assert cfg.reward.scales.feet_phase_contact == pytest.approx(0.0)
    assert cfg.reward.scales.phase_contact == pytest.approx(1.0)
    assert cfg.reward.scales.null_foot_force_balance == pytest.approx(-1.0)
    assert cfg.reward.scales.null_torque_relaxation == pytest.approx(-0.1)
    assert cfg.play_profile.env.commands.resampling_time == pytest.approx(0.0)

    validate_fada_oracle_behavior_environment(
        cfg.env, FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE
    )
    validate_fada_single_reward(
        reward_scales=OmegaConf.to_container(cfg.reward.scales, resolve=True),
        reward_config=OmegaConf.to_container(cfg.reward, resolve=True),
        behavior_profile=FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE,
    )
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )
    runtime.validate_training_config(cfg)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("env.domain_rand.actuator_strength.curriculum_enabled", "its curriculum"),
        ("env.domain_rand.actuator_strength.group_curriculum_enabled", "grouped curriculum"),
        ("env.domain_rand.actuator_strength.include_in_critic_obs", "duplicate Critic tail"),
    ],
)
def test_v024_preflight_rejects_side_effects_from_disabled_left_knee_strength(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    message: str,
) -> None:
    cfg = _compose_profile(monkeypatch, "sac/g1_walk_flat/mujoco_fada_phase_contact")
    OmegaConf.update(cfg, path, True, merge=False)
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )

    with pytest.raises(ValueError, match=message):
        runtime.validate_training_config(cfg)


def test_v023_profile_remains_a_separate_unchanged_compatibility_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose_profile(monkeypatch, "sac/g1_walk_flat/mujoco_fada_phase")

    assert cfg.algo.actor.oracle_behavior_profile == FADA_ORACLE_PHASE_LOCOMOTION_PROFILE
    assert cfg.env.gait_phase_init_mode == "offset_phase"
    assert cfg.env.commands.rel_standing_envs == pytest.approx(0.0)
    assert cfg.env.commands.resampling_time == pytest.approx(0.0)
    assert OmegaConf.select(cfg, "env.command_gated_phase_contact.enabled", default=False) is False
    assert cfg.reward.scales.feet_phase == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("command_gated_phase_contact.command_xy_dead_zone", 0.11, "command_xy_dead_zone"),
        ("command_gated_phase_contact.command_yaw_dead_zone", 0.11, "command_yaw_dead_zone"),
        ("command_gated_phase_contact.min_frequency", 0.8, "min_frequency"),
        ("command_gated_phase_contact.max_frequency", 1.4, "max_frequency"),
        ("command_gated_phase_contact.contact_force_threshold", 1.1, "contact_force_threshold"),
    ],
)
def test_v024_environment_identity_rejects_behavior_parameter_drift(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    value: float,
    message: str,
) -> None:
    cfg = _compose_profile(monkeypatch, "sac/g1_walk_flat/mujoco_fada_phase_contact")
    OmegaConf.update(cfg.env, path, value)

    with pytest.raises(ValueError, match=message):
        validate_fada_oracle_behavior_environment(
            cfg.env, FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE
        )


def test_v024_keeps_command_phase_actor_layout_and_fada_66_29_3_contract() -> None:
    rows = 2
    zeros3 = np.zeros((rows, 3), dtype=np.float32)
    zeros29 = np.zeros((rows, 29), dtype=np.float32)
    commands = np.array([[0.2, 0.0, 0.1], [-0.4, 0.2, -0.3]], dtype=np.float32)
    phase = np.array([[0.0, np.pi], [np.pi, 0.0]], dtype=np.float32)
    obs = assemble_walk_observation(
        noisy_gyro=zeros3,
        noisy_gravity=zeros3,
        noisy_diff=zeros29,
        noisy_dof_vel=zeros29,
        gyro=zeros3,
        gravity=zeros3,
        diff=zeros29,
        dof_vel=zeros29,
        last_actions=zeros29,
        command_obs=commands,
        gait_phase=phase,
        mode_obs=np.zeros((rows, 1), dtype=np.float32),
        linvel=zeros3,
        mode_observation=False,
        walk_profile=True,
        fada_privileged=False,
        privileged_strength=None,
        fada_privileged_obs=None,
        dtype=np.dtype(np.float32),
    )

    assert (
        build_obs_groups_spec(
            mode_observation=False,
            height_observation=False,
            privileged_strength=False,
            fada_privileged=False,
            fada_body_count=0,
        )["obs"]
        == FADA_G1_ACTOR_OBS_DIM
        == 98
    )
    np.testing.assert_allclose(obs["obs"][:, 93:96], commands)
    np.testing.assert_allclose(obs["obs"][:, 96:98], phase)
    assert (FADA_G1_STATE_DIM, FADA_G1_ACTION_DIM, FADA_G1_COMMAND_DIM) == (66, 29, 3)


def _checkpoint_contract(profile: str) -> FADAOracleCheckpointContract:
    return FADAOracleCheckpointContract(
        oracle_lineage_id="v024-test-lineage",
        privileged_schema="g1_fada_privileged_v1",
        task_name="G1WalkFlat",
        backend="mujoco",
        action_scale=(1.0,),
        seed=1,
        obs_dim=3,
        critic_obs_dim=4,
        action_dim=1,
        body_names=("world",),
        actuated_joint_names=("joint",),
        privileged_field_slices=(("privileged", 0, 1),),
        asset_sha256="a" * 64,
        config_hashes=(("effective", "b" * 64),),
        behavior_profile=profile,
    )


def test_same_shape_v023_checkpoint_is_rejected_by_v024_identity() -> None:
    legacy = _checkpoint_contract(FADA_ORACLE_PHASE_LOCOMOTION_PROFILE)
    current = _checkpoint_contract(FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE)
    payload = seal_fada_oracle_checkpoint({}, legacy, iteration=5000)

    with pytest.raises(ValueError, match="behavior_profile"):
        validate_fada_oracle_checkpoint_payload(payload, current, expected_iteration=5000)


def test_v024_checkpoint_rejects_missing_identity_and_config_hash_drift() -> None:
    current = _checkpoint_contract(FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE)
    with pytest.raises(ValueError, match="missing fada_privileged_oracle identity"):
        validate_fada_oracle_checkpoint_payload({}, current, expected_iteration=5000)

    stale = replace(current, config_hashes=(("effective", "c" * 64),))
    payload = seal_fada_oracle_checkpoint({}, stale, iteration=5000)
    with pytest.raises(ValueError, match="config_hashes mismatch"):
        validate_fada_oracle_checkpoint_payload(payload, current, expected_iteration=5000)
