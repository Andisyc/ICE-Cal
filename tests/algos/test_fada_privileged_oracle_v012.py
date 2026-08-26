from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill.fada_privileged_oracle import (
    FADA_ORACLE_FINAL_ITERATION,
    FADA_ORACLE_INTERMEDIATE_ITERATIONS,
    G1FADAPrivilegedObservation,
    build_g1_fada_privileged_layout,
    pack_g1_fada_privileged_observation,
    validate_fada_oracle_lineage,
    validate_no_gait_reward,
)


def _bundle(batch: int, body_count: int) -> G1FADAPrivilegedObservation:
    def values(width: int, start: int) -> np.ndarray:
        return np.arange(start, start + batch * width, dtype=np.float32).reshape(batch, width)

    return G1FADAPrivilegedObservation(
        base_linear_velocity=values(3, 1),
        foot_contact_resultants=values(6, 10),
        foot_contact_flags=values(2, 30),
        terrain_heights=values(9, 40),
        root_clearance=values(1, 60),
        kp_scale=values(29, 70),
        kd_scale=values(29, 140),
        normalized_torque=values(29, 210),
        ground_friction=values(1, 280),
        base_com_shift=values(3, 290),
        added_base_mass=values(1, 300),
        body_mass_scale=values(body_count, 310),
        dof_position_bias=values(29, 400),
        torque_rfi=values(29, 470),
        control_delay=values(1, 540),
        push_interval=values(1, 550),
        push_velocity=values(1, 560),
    )


def test_g1_fada_privileged_layout_is_typed_ordered_and_round_trips() -> None:
    body_names = ("world", "pelvis", "left_hip", "right_hip")
    layout = build_g1_fada_privileged_layout(body_names)
    bundle = _bundle(batch=2, body_count=len(body_names))

    packed = pack_g1_fada_privileged_observation(bundle, layout)

    assert layout.schema == "g1_fada_privileged_v1"
    assert layout.body_names == body_names
    assert layout.width == 174 + len(body_names)
    assert packed.shape == (2, layout.width)
    for field_name in layout.field_names:
        np.testing.assert_array_equal(
            packed[:, layout.slice_for(field_name)], getattr(bundle, field_name)
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda b: setattr(b, "foot_contact_resultants", np.zeros((2, 5))),
            "foot_contact_resultants",
        ),
        (lambda b: setattr(b, "kp_scale", np.full((2, 29), np.nan)), "finite"),
    ],
)
def test_g1_fada_privileged_layout_fails_closed_on_bad_payload(mutation, match: str) -> None:
    layout = build_g1_fada_privileged_layout(("world", "pelvis"))
    bundle = _bundle(batch=2, body_count=2)
    mutation(bundle)

    with pytest.raises(ValueError, match=match):
        pack_g1_fada_privileged_observation(bundle, layout)


def test_no_gait_reward_guard_rejects_nonzero_phase_conditioned_rewards() -> None:
    validate_no_gait_reward({"tracking_lin_vel": 2.0, "feet_phase": 0.0})
    validate_no_gait_reward({"tracking_lin_vel": 2.0})

    for name in ("feet_phase", "feet_phase_contact", "gait_tracking", "footfall_target"):
        with pytest.raises(ValueError, match=name):
            validate_no_gait_reward({name: 0.1})


def _lineage_records(lineage_id: str = "lineage-a") -> list[dict[str, object]]:
    contract = _sealed_checkpoint_contract(lineage_id=lineage_id)
    return [
        contract.identity_for_iteration(iteration).to_record()
        for iteration in (*FADA_ORACLE_INTERMEDIATE_ITERATIONS, FADA_ORACLE_FINAL_ITERATION)
    ]


def test_oracle_lineage_accepts_exactly_twenty_intermediates_and_one_final() -> None:
    admitted = validate_fada_oracle_lineage(_lineage_records())
    assert admitted.oracle_lineage_id == "lineage-a"
    assert admitted.intermediate_iterations == FADA_ORACLE_INTERMEDIATE_ITERATIONS
    assert admitted.final_iteration == 5000


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda xs: xs.pop(3), "iterations"),
        (lambda xs: xs[0].update(oracle_lineage_id="other"), "lineage"),
        (lambda xs: xs[-1].update(iteration=4800), "iteration"),
        (lambda xs: xs[0].update(role="planner_label"), "role"),
        (lambda xs: xs[-1].update(task_name="G1StandStill"), "task"),
    ],
)
def test_oracle_lineage_rejects_semantically_mixed_campaign(mutate, match: str) -> None:
    records = _lineage_records()
    mutate(records)
    with pytest.raises(ValueError, match=match):
        validate_fada_oracle_lineage(records)


def _apply_valid_privileged_oracle_profile(cfg: SimpleNamespace) -> None:
    cfg.algo.gamma = 0.99
    cfg.algo.value_support_min = -30.0
    cfg.algo.value_support_max = 30.0
    cfg.algo.obs_normalization = True
    if isinstance(cfg.algo.actor, dict):
        cfg.algo.actor["priv_info_normalization"] = True
    else:
        cfg.algo.actor.priv_info_normalization = True
    cfg.env.mode_observation = False
    cfg.env.ctrl_dt = 0.02
    cfg.env.commands = getattr(
        cfg.env,
        "commands",
        SimpleNamespace(rel_transition_envs=0.0),
    )
    cfg.env.commands.vel_limit = [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]]
    cfg.env.commands.resampling_time = 0.0
    cfg.env.commands.heading_command = False
    cfg.env.curriculum = SimpleNamespace(enabled=False)


def test_runtime_preflight_rejects_gait_reward_before_env_creation() -> None:
    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        resolve_privileged_locomotion_sac_runtime,
    )

    runtime = resolve_privileged_locomotion_sac_runtime(
        {"runtime_impl": "privileged_locomotion_sac"}
    )
    cfg = SimpleNamespace(
        training=SimpleNamespace(task_name="G1WalkFlat", sim_backend="mujoco"),
        algo=SimpleNamespace(
            max_iterations=5000,
            save_interval=240,
            use_symmetry=False,
            actor={"oracle_lineage_id": "test-lineage"},
        ),
        env=SimpleNamespace(
            fada_privileged_observation=SimpleNamespace(
                enabled=True, schema="g1_fada_privileged_v1"
            )
        ),
        reward=SimpleNamespace(scales=SimpleNamespace(feet_phase=1.0)),
    )
    _apply_valid_privileged_oracle_profile(cfg)

    with pytest.raises(ValueError, match="feet_phase"):
        runtime.validate_training_config(cfg)

    cfg.reward.scales.feet_phase = 0.0
    cfg.reward.gait_constraint = SimpleNamespace(enabled=True, penalty_scale=0.5)
    with pytest.raises(ValueError, match="gait constraint penalty"):
        runtime.validate_training_config(cfg)

    cfg.reward.gait_constraint.penalty_scale = 0.0
    with pytest.raises(ValueError, match="gait constraint mode"):
        runtime.validate_training_config(cfg)


def test_runtime_preflight_rejects_standing_mode_mechanisms() -> None:
    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        resolve_privileged_locomotion_sac_runtime,
    )

    runtime = resolve_privileged_locomotion_sac_runtime(
        {"runtime_impl": "privileged_locomotion_sac"}
    )
    cfg = SimpleNamespace(
        training=SimpleNamespace(task_name="G1WalkFlat", sim_backend="mujoco"),
        algo=SimpleNamespace(
            max_iterations=5000,
            save_interval=240,
            use_symmetry=False,
            actor={"oracle_lineage_id": "test-lineage"},
        ),
        env=SimpleNamespace(
            mode_observation=False,
            commands=SimpleNamespace(rel_transition_envs=0.0),
            fada_privileged_observation=SimpleNamespace(
                enabled=True, schema="g1_fada_privileged_v1"
            ),
        ),
        reward=SimpleNamespace(
            scales=SimpleNamespace(feet_phase=0.0),
            mode=SimpleNamespace(enabled=True),
        ),
    )

    _apply_valid_privileged_oracle_profile(cfg)

    with pytest.raises(ValueError, match="reward.mode"):
        runtime.validate_training_config(cfg)

    cfg.reward.mode.enabled = False
    cfg.env.commands.rel_transition_envs = 0.2
    with pytest.raises(ValueError, match="transition"):
        runtime.validate_training_config(cfg)


def test_runtime_preflight_rejects_hydra_gait_reward_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        resolve_privileged_locomotion_sac_runtime,
    )

    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "unit-test-lineage")
    conf_dir = Path(__file__).resolve().parents[2] / "conf/offpolicy"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "algo=sac",
                "task=sac/g1_walk_flat/mujoco_fada_privileged_oracle",
                "reward.scales.feet_phase=1.0",
            ],
        )
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )

    with pytest.raises(ValueError, match="feet_phase"):
        runtime.validate_training_config(cfg)


def test_privileged_oracle_hydra_profile_is_single_task_moderate_and_gait_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "unit-test-lineage")
    conf_dir = Path(__file__).resolve().parents[2] / "conf/offpolicy"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "algo=sac",
                "task=sac/g1_walk_flat/mujoco_fada_privileged_oracle",
            ],
        )

    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.algo.max_iterations == 5000
    assert cfg.algo.save_interval == 240
    assert cfg.algo.gamma == pytest.approx(0.99)
    assert cfg.algo.value_support_min == pytest.approx(-30.0)
    assert cfg.algo.value_support_max == pytest.approx(30.0)
    assert cfg.algo.obs_normalization is True
    assert cfg.algo.actor.priv_info_normalization is True
    assert cfg.env.mode_observation is False
    assert cfg.env.ctrl_dt == pytest.approx(0.02)
    assert cfg.env.commands.rel_standing_envs == 0.3
    assert cfg.env.commands.rel_transition_envs == 0.0
    assert cfg.env.commands.vel_limit == [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]]
    assert cfg.env.commands.resampling_time == pytest.approx(0.0)
    assert cfg.env.commands.heading_command is False
    assert cfg.env.curriculum.enabled is False
    assert cfg.algo.gamma ** round(2.0 / cfg.env.ctrl_dt) > 0.35
    ideal_return_upper = (
        cfg.env.ctrl_dt
        * (
            cfg.reward.scales.alive
            + cfg.reward.scales.tracking_lin_vel
            + cfg.reward.scales.tracking_ang_vel
        )
        / (1.0 - cfg.algo.gamma)
    )
    assert ideal_return_upper == pytest.approx(27.0)
    assert cfg.algo.value_support_max >= ideal_return_upper
    assert OmegaConf.select(cfg, "reward.mode") is None
    assert OmegaConf.select(cfg, "reward.gait_constraint") is None
    assert cfg.reward.scales.feet_phase == 0.0
    assert OmegaConf.select(cfg, "reward.scales.feet_phase_contrast") is None
    assert OmegaConf.select(cfg, "reward.scales.feet_phase_contact") is None
    assert cfg.env.domain_rand.dof_position_bias_range == [-0.025, 0.025]
    assert cfg.env.domain_rand.torque_rfi_fraction == 0.05

    from unilab.training import BackendAdapter

    override = BackendAdapter(
        cfg, root_dir=Path.cwd(), algo_name="sac"
    ).build_task_env_cfg_override()
    assert override["fada_privileged_observation"]["schema"] == "g1_fada_privileged_v1"
    assert override["domain_rand"]["randomize_control_delay"] is True
    assert override["mode_observation"] is False
    assert override["commands"]["rel_standing_envs"] == 0.3
    assert override["commands"]["rel_transition_envs"] == 0.0
    assert override["commands"]["vel_limit"] == [
        [-0.6, -0.4, -0.8],
        [1.0, 0.4, 0.8],
    ]
    assert override["commands"]["resampling_time"] == pytest.approx(0.0)
    assert override["commands"]["heading_command"] is False
    assert override["ctrl_dt"] == pytest.approx(0.02)
    assert override["curriculum"]["enabled"] is False
    assert "mode" not in override["reward_config"]
    assert "gait_constraint" not in override["reward_config"]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("algo.gamma", 0.97, "gamma"),
        ("algo.value_support_max", 20.0, "value support"),
        ("algo.obs_normalization", False, "observation normalization"),
        ("algo.actor.priv_info_normalization", False, "privileged normalization"),
        ("env.ctrl_dt", 0.01, "ctrl_dt"),
        ("env.commands.vel_limit", [[0.0, 0.0, 0.0], [1.0, 0.4, 0.8]], "vel_limit"),
        ("env.commands.resampling_time", 1.0, "resampling"),
        ("env.commands.heading_command", True, "heading command"),
        ("env.curriculum.enabled", True, "curriculum"),
    ],
)
def test_privileged_oracle_preflight_rejects_unsealed_training_profile(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    value: object,
    message: str,
) -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        resolve_privileged_locomotion_sac_runtime,
    )

    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "unit-test-lineage")
    conf_dir = Path(__file__).resolve().parents[2] / "conf/offpolicy"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "algo=sac",
                "task=sac/g1_walk_flat/mujoco_fada_privileged_oracle",
            ],
        )
    OmegaConf.update(cfg, path, value, merge=False, force_add=True)
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )

    with pytest.raises(ValueError, match=message):
        runtime.validate_training_config(cfg)


def test_privileged_oracle_normalizers_update_and_round_trip() -> None:
    from unilab.algos.torch.distill.fada_privileged_oracle import (
        seal_fada_oracle_checkpoint,
    )
    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        FADAPrivilegedSACLearner,
    )

    contract = _sealed_checkpoint_contract(
        lineage_id="normalized-lineage", obs_dim=5, critic_obs_dim=8, action_dim=2
    )

    def build_learner() -> FADAPrivilegedSACLearner:
        return FADAPrivilegedSACLearner(
            obs_dim=5,
            critic_obs_dim=8,
            priv_info_dim=3,
            action_dim=2,
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            priv_info_embed_dim=2,
            priv_mlp_hidden_dims=(4, 2),
            num_atoms=5,
            use_layer_norm=False,
            use_compile=False,
            obs_normalization=True,
            priv_info_normalization=True,
            oracle_lineage_id="normalized-lineage",
            checkpoint_contract=contract,
        )

    learner = build_learner()
    assert float(learner.obs_normalizer.count) == 0.0
    assert float(learner.actor.priv_info_normalizer.count) == 0.0
    obs = torch.tensor([[1.0, 10.0, -1.0, 0.5, 3.0], [3.0, 14.0, 1.0, 1.5, 7.0]])
    next_obs = obs + torch.tensor([1.0, 2.0, 0.5, -0.25, 1.0])
    priv = torch.tensor([[100.0, -5.0, 0.2], [400.0, 7.0, 0.8]])
    next_priv = priv + torch.tensor([50.0, 1.0, 0.1])
    batch = {
        "obs": obs,
        "critic": torch.cat([obs, priv], dim=-1),
        "actions": torch.tensor([[0.1, -0.2], [0.2, -0.1]]),
        "rewards": torch.tensor([1.0, 0.5]),
        "next_obs": next_obs,
        "next_critic": torch.cat([next_obs, next_priv], dim=-1),
        "dones": torch.zeros(2),
        "truncated": torch.zeros(2),
    }

    learner.update_critic(batch)
    assert float(learner.obs_normalizer.count) > 0.0
    assert float(learner.actor.priv_info_normalizer.count) > 0.0

    from unilab.algos.torch.common.actor_factory import build_actor

    collector_actor = build_actor(
        "privileged_locomotion_sac",
        obs_dim=5,
        action_dim=2,
        actor_hidden_dim=16,
        use_layer_norm=False,
        device="cpu",
        priv_info_dim=3,
        priv_info_embed_dim=2,
        priv_mlp_hidden_dims=(4, 2),
        priv_info_normalization=True,
    )
    collector_actor.load_state_dict(learner.actor.state_dict())
    learner.actor.eval()
    collector_actor.eval()
    torch.testing.assert_close(
        collector_actor.explore(obs, priv, deterministic=True),
        learner.actor.explore(obs, priv, deterministic=True),
    )

    state = learner.get_state_dict()
    assert "obs_normalizer" in state
    assert any(key.startswith("priv_info_normalizer.") for key in state["actor"])

    payload = seal_fada_oracle_checkpoint(state, contract, iteration=240)
    restored = build_learner()
    restored.load_state_dict(payload)
    torch.testing.assert_close(restored.obs_normalizer.mean, learner.obs_normalizer.mean)
    torch.testing.assert_close(
        restored.actor.priv_info_normalizer.mean,
        learner.actor.priv_info_normalizer.mean,
    )

    rejected = build_learner()
    rejected_actor = {
        key: value.detach().clone() for key, value in rejected.actor.state_dict().items()
    }
    missing_normalizer = dict(payload)
    missing_normalizer.pop("obs_normalizer")
    with pytest.raises(ValueError, match="missing obs_normalizer"):
        rejected.load_state_dict(missing_normalizer)
    assert all(
        torch.equal(rejected.actor.state_dict()[key], value)
        for key, value in rejected_actor.items()
    )


def test_privileged_oracle_checkpoint_persists_lineage_and_actor_privilege_identity() -> None:
    from unilab.algos.torch.distill.fada_privileged_oracle import (
        seal_fada_oracle_checkpoint,
    )
    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        FADAPrivilegedSACLearner,
    )

    contract = _sealed_checkpoint_contract(
        lineage_id="lineage-persisted", obs_dim=5, critic_obs_dim=8, action_dim=2
    )
    learner = FADAPrivilegedSACLearner(
        obs_dim=5,
        critic_obs_dim=8,
        priv_info_dim=3,
        action_dim=2,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        priv_info_embed_dim=2,
        priv_mlp_hidden_dims=(4, 2),
        num_atoms=5,
        use_layer_norm=False,
        use_compile=False,
        oracle_lineage_id="lineage-persisted",
        checkpoint_contract=contract,
    )

    payload = seal_fada_oracle_checkpoint(learner.get_state_dict(), contract, iteration=240)
    metadata = payload["fada_privileged_oracle"]
    assert metadata["oracle_lineage_id"] == "lineage-persisted"
    assert metadata["actor_directly_privileged"] is True
    assert metadata["privileged_schema"] == "g1_fada_privileged_v1"


def _sealed_checkpoint_contract(
    *,
    lineage_id: str = "lineage-sealed",
    obs_dim: int = 98,
    critic_obs_dim: int = 276,
    action_dim: int = 29,
):
    from unilab.algos.torch.distill.fada_privileged_oracle import (
        FADAOracleCheckpointContract,
    )

    return FADAOracleCheckpointContract(
        oracle_lineage_id=lineage_id,
        privileged_schema="g1_fada_privileged_v1",
        task_name="G1WalkFlat",
        backend="mujoco",
        action_scale=(1.0,),
        seed=7,
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
        action_dim=action_dim,
        body_names=("world", "pelvis"),
        actuated_joint_names=tuple(f"joint_{index}" for index in range(action_dim)),
        privileged_field_slices=(
            ("base_linear_velocity", 0, 3),
            ("body_mass", 3, 5),
        ),
        asset_sha256="a" * 64,
        config_hashes=(
            ("algo", "b" * 64),
            ("env", "c" * 64),
            ("reward", "d" * 64),
        ),
    )


def test_checkpoint_contract_seals_complete_asymmetric_identity() -> None:
    contract = _sealed_checkpoint_contract()

    intermediate = contract.identity_for_iteration(240).to_record()
    final = contract.identity_for_iteration(5000).to_record()

    assert intermediate["role"] == "idm_coverage"
    assert final["role"] == "final_oracle"
    assert intermediate["iteration"] == 240
    assert intermediate["config_hashes"]["reward"] == "d" * 64
    assert intermediate["body_names"] == ["world", "pelvis"]
    assert intermediate["actuated_joint_names"][:2] == ["joint_0", "joint_1"]
    assert intermediate["dimensions"] == {
        "obs": 98,
        "critic": 276,
        "privileged": 178,
        "action": 29,
    }
    with pytest.raises(ValueError, match="iteration"):
        contract.identity_for_iteration(4801)


def test_checkpoint_load_rejects_identity_before_learner_mutation() -> None:
    from unilab.algos.torch.distill.fada_privileged_oracle import (
        seal_fada_oracle_checkpoint,
    )
    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        FADAPrivilegedSACLearner,
    )

    contract = _sealed_checkpoint_contract(obs_dim=5, critic_obs_dim=8, action_dim=2)
    learner = FADAPrivilegedSACLearner(
        obs_dim=5,
        critic_obs_dim=8,
        priv_info_dim=3,
        action_dim=2,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        priv_info_embed_dim=2,
        priv_mlp_hidden_dims=(4, 2),
        num_atoms=5,
        use_layer_norm=False,
        use_compile=False,
        oracle_lineage_id="lineage-sealed",
        checkpoint_contract=contract,
    )
    valid = seal_fada_oracle_checkpoint(learner.get_state_dict(), contract, iteration=240)
    tampered = dict(valid)
    tampered["fada_privileged_oracle"] = dict(valid["fada_privileged_oracle"])
    tampered["fada_privileged_oracle"]["action_scale"] = [0.5]
    before = {name: value.detach().clone() for name, value in learner.actor.state_dict().items()}
    tampered["actor"] = {
        name: torch.full_like(value, 123.0) for name, value in valid["actor"].items()
    }

    with pytest.raises(ValueError, match="action_scale"):
        learner.load_state_dict(tampered)

    after = learner.actor.state_dict()
    assert all(torch.equal(after[name], value) for name, value in before.items())


def test_checkpoint_schema_matrix_is_explicit_and_mixed_campaigns_are_not_admitted() -> None:
    from unilab.algos.torch.distill.fada_privileged_oracle import (
        seal_fada_oracle_checkpoint,
    )
    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        FADAPrivilegedSACLearner,
    )
    from unilab.algos.torch.hora.sac_learner import HoraSACLearner

    contract = _sealed_checkpoint_contract(obs_dim=5, critic_obs_dim=8, action_dim=2)

    def build_learner(cls, *, sealed: bool):
        kwargs = {
            "obs_dim": 5,
            "critic_obs_dim": 8,
            "priv_info_dim": 3,
            "action_dim": 2,
            "actor_hidden_dim": 16,
            "critic_hidden_dim": 16,
            "priv_info_embed_dim": 2,
            "priv_mlp_hidden_dims": (4, 2),
            "num_atoms": 5,
            "use_layer_norm": False,
            "use_compile": False,
        }
        if sealed:
            kwargs.update(
                oracle_lineage_id="lineage-sealed",
                checkpoint_contract=contract,
            )
        return cls(**kwargs)

    old_writer = build_learner(HoraSACLearner, sealed=False)
    old_payload = old_writer.get_state_dict()
    old_payload["fada_privileged_oracle"] = {
        "schema_version": 1,
        "oracle_lineage_id": "lineage-sealed",
    }
    new_writer = build_learner(FADAPrivilegedSACLearner, sealed=True)
    new_payload = seal_fada_oracle_checkpoint(new_writer.get_state_dict(), contract, iteration=240)

    build_learner(HoraSACLearner, sealed=False).load_state_dict(old_payload)
    with pytest.raises(ValueError, match="iteration"):
        build_learner(FADAPrivilegedSACLearner, sealed=True).load_state_dict(old_payload)
    build_learner(HoraSACLearner, sealed=False).load_state_dict(new_payload)
    build_learner(FADAPrivilegedSACLearner, sealed=True).load_state_dict(new_payload)


def test_checkpoint_gateway_finalizes_exact_production_20_plus_1(tmp_path: Path) -> None:
    from unilab.algos.torch.distill.fada_privileged_oracle import (
        FADAOracleCheckpointGateway,
    )

    class TinyLearner:
        def get_state_dict(self):
            return {"actor": {"weight": torch.tensor([3.0])}, "update_count": 9}

    gateway = FADAOracleCheckpointGateway(_sealed_checkpoint_contract())
    iterations = (*FADA_ORACLE_INTERMEDIATE_ITERATIONS, FADA_ORACLE_FINAL_ITERATION)
    for iteration in iterations:
        gateway.save(TinyLearner(), tmp_path / f"model_{iteration}.pt", iteration)

    manifest = json.loads((tmp_path / "fada_oracle_lineage.json").read_text())
    assert manifest["oracle_lineage_id"] == "lineage-sealed"
    assert manifest["intermediate_iterations"] == list(FADA_ORACLE_INTERMEDIATE_ITERATIONS)
    assert manifest["final_iteration"] == FADA_ORACLE_FINAL_ITERATION
    assert len(manifest["checkpoint_sha256"]) == 21


def test_checkpoint_gateway_rejects_missing_intermediate_at_finalization(tmp_path: Path) -> None:
    from unilab.algos.torch.distill.fada_privileged_oracle import (
        FADAOracleCheckpointGateway,
    )

    class TinyLearner:
        def get_state_dict(self):
            return {"actor": {"weight": torch.tensor([3.0])}}

    gateway = FADAOracleCheckpointGateway(_sealed_checkpoint_contract())
    for iteration in (*FADA_ORACLE_INTERMEDIATE_ITERATIONS[1:], FADA_ORACLE_FINAL_ITERATION):
        if iteration == FADA_ORACLE_FINAL_ITERATION:
            with pytest.raises(ValueError, match="missing"):
                gateway.save(TinyLearner(), tmp_path / f"model_{iteration}.pt", iteration)
        else:
            gateway.save(TinyLearner(), tmp_path / f"model_{iteration}.pt", iteration)


def test_g1_checkpoint_layout_identity_hashes_cold_path_asset_tree(tmp_path: Path) -> None:
    from unilab.envs.locomotion.g1.fada_privileged import (
        build_g1_fada_checkpoint_layout_identity,
    )

    model = tmp_path / "scene.xml"
    included = tmp_path / "robot.xml"
    model.write_text('<mujoco><include file="robot.xml"/></mujoco>', encoding="utf-8")
    included.write_text("<mujoco/>", encoding="utf-8")
    first = build_g1_fada_checkpoint_layout_identity(
        body_names=("world", "pelvis"),
        actuated_joint_names=("joint_0", "joint_1"),
        model_file=model,
    )
    included.write_text('<mujoco model="changed"/>', encoding="utf-8")
    second = build_g1_fada_checkpoint_layout_identity(
        body_names=("world", "pelvis"),
        actuated_joint_names=("joint_0", "joint_1"),
        model_file=model,
    )

    assert first.body_names == ("world", "pelvis")
    assert first.actuated_joint_names == ("joint_0", "joint_1")
    assert first.field_slices[0] == ("base_linear_velocity", 0, 3)
    assert first.asset_sha256 != second.asset_sha256


def test_runtime_builds_training_contract_and_owner_checkpoint_saver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from unilab.algos.torch.distill.fada_privileged_oracle import (
        FADAOracleCheckpointGateway,
    )
    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        resolve_privileged_locomotion_sac_runtime,
    )
    from unilab.envs.locomotion.g1.fada_privileged import (
        G1FADAPrivilegedCheckpointLayoutIdentity,
    )

    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "runtime-lineage")
    conf_dir = Path(__file__).resolve().parents[2] / "conf/offpolicy"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "algo=sac",
                "task=sac/g1_walk_flat/mujoco_fada_privileged_oracle",
            ],
        )
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )
    assert runtime is not None
    env = SimpleNamespace(
        get_fada_privileged_checkpoint_identity=lambda: G1FADAPrivilegedCheckpointLayoutIdentity(
            body_names=("world", "pelvis"),
            actuated_joint_names=tuple(f"joint_{index}" for index in range(29)),
            field_slices=(("base_linear_velocity", 0, 3), ("body_mass", 3, 5)),
            asset_sha256="e" * 64,
        )
    )

    kwargs = runtime.build_training_model_kwargs(
        cfg=cfg,
        env=env,
        obs_dim=98,
        critic_obs_dim=276,
        action_dim=29,
    )
    assert kwargs["obs_normalization"] is True
    assert kwargs["priv_info_normalization"] is True
    assert kwargs["v_min"] == pytest.approx(-30.0)
    assert kwargs["v_max"] == pytest.approx(30.0)
    contract = kwargs["checkpoint_contract"]
    assert contract.oracle_lineage_id == "runtime-lineage"
    assert dict(contract.config_hashes).keys() == {"algo", "env", "reward", "training"}
    sealed_env_hash = dict(contract.config_hashes)["env"]
    cfg.env.ctrl_dt = 0.01
    changed = runtime.build_training_model_kwargs(
        cfg=cfg,
        env=env,
        obs_dim=98,
        critic_obs_dim=276,
        action_dim=29,
    )["checkpoint_contract"]
    assert dict(changed.config_hashes)["env"] != sealed_env_hash
    cfg.env.ctrl_dt = 0.02
    learner = SimpleNamespace(checkpoint_contract=contract)
    saver = runtime.build_checkpoint_saver(learner)
    assert isinstance(saver.__self__, FADAOracleCheckpointGateway)


def test_fada_privileged_obs_contract_does_not_duplicate_base_velocity() -> None:
    from unilab.envs.locomotion.g1.fada_privileged import (
        build_g1_fada_privileged_layout,
    )
    from unilab.envs.locomotion.g1.joystick import G1WalkEnv

    env = object.__new__(G1WalkEnv)
    env._cfg = SimpleNamespace(mode_observation=False)
    env._fada_body_names = tuple(f"body_{index}" for index in range(31))
    env._uses_height_command_observation = lambda: False
    env._includes_privileged_actuator_strength_obs = lambda: False
    env._fada_privileged_enabled = lambda: True

    dims = env.obs_groups_spec
    layout = build_g1_fada_privileged_layout(env._fada_body_names)
    assert dims == {"obs": 98, "critic": 98 + layout.width}
