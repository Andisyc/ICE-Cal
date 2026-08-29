from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tests.algos._fada_training_test_support import _CommandControlledEnv, _curriculum_config
from unilab.algos.torch.distill import (
    DistillationTeacherSpec,
    FADACollectionSpec,
    MLPStudentPolicy,
    collect_fada_source_windows,
    save_distillation_checkpoint,
)


def _save_privileged_oracle(path: Path) -> None:
    from unilab.algos.torch.distill.fada_privileged_oracle import (
        FADAOracleCheckpointContract,
        seal_fada_oracle_checkpoint,
    )
    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        FADAPrivilegedSACLearner,
    )

    contract = FADAOracleCheckpointContract(
        oracle_lineage_id="lineage-idm",
        privileged_schema="g1_fada_privileged_v1",
        task_name="G1WalkFlat",
        backend="mujoco",
        action_scale=(1.0,),
        seed=7,
        obs_dim=3,
        critic_obs_dim=5,
        action_dim=2,
        body_names=("world", "pelvis"),
        actuated_joint_names=("joint_0", "joint_1"),
        privileged_field_slices=(("privileged", 0, 2),),
        asset_sha256="a" * 64,
        config_hashes=(("algo", "b" * 64),),
    )
    learner = FADAPrivilegedSACLearner(
        obs_dim=3,
        critic_obs_dim=5,
        priv_info_dim=2,
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
        oracle_lineage_id="lineage-idm",
        checkpoint_contract=contract,
    )
    payload = seal_fada_oracle_checkpoint(learner.get_state_dict(), contract, iteration=240)
    torch.save(payload, path)


def _save_distilled_oracle(path: Path, *, obs_dim: int = 98, action_dim: int = 29) -> None:
    policy = MLPStudentPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(8,),
        activation="elu",
        squash_action=True,
    )
    with torch.no_grad():
        for index, parameter in enumerate(policy.parameters(), start=1):
            parameter.fill_(index / 100.0)
    save_distillation_checkpoint(
        path,
        student=policy,
        agent_steps=17,
        teacher_metadata={"source": "walk-stand-dagger"},
        distill_runtime_cfg={
            "student_model_type": "mlp",
            "student_obs_dim": obs_dim,
            "student_action_dim": action_dim,
            "student_hidden_dims": [8],
            "student_activation": "elu",
            "student_squash_action": True,
        },
    )


def test_fada_oracle_loader_accepts_frozen_distillation_student(tmp_path: Path) -> None:
    from unilab.algos.torch.distill.fada_oracle import load_fada_oracle_policy

    checkpoint = tmp_path / "dagger_iteration_8.pt"
    _save_distilled_oracle(checkpoint)

    oracle = load_fada_oracle_policy(
        checkpoint,
        DistillationTeacherSpec(obs_dim=98, action_dim=29),
        device="cpu",
    )

    action = oracle(torch.arange(98, dtype=torch.float32).reshape(1, 98) / 100.0)
    assert action.shape == (1, 29)
    assert oracle.obs_dim == 98
    assert oracle.action_dim == 29
    assert not oracle.training
    assert all(not parameter.requires_grad for parameter in oracle.parameters())


def test_fada_oracle_loader_rejects_distillation_dimension_mismatch(tmp_path: Path) -> None:
    from unilab.algos.torch.distill.fada_oracle import load_fada_oracle_policy

    checkpoint = tmp_path / "wrong-dim.pt"
    _save_distilled_oracle(checkpoint, obs_dim=97)

    with pytest.raises(ValueError, match="obs dim mismatch"):
        load_fada_oracle_policy(
            checkpoint,
            DistillationTeacherSpec(obs_dim=98, action_dim=29),
            device="cpu",
        )


def test_fada_oracle_loader_runs_privileged_actor_from_env_observation(tmp_path: Path) -> None:
    from unilab.algos.torch.distill.fada_oracle import load_fada_oracle_policy

    checkpoint = tmp_path / "model_240.pt"
    _save_privileged_oracle(checkpoint)
    oracle = load_fada_oracle_policy(
        checkpoint,
        DistillationTeacherSpec(
            obs_dim=3,
            action_dim=2,
            algo_type="privileged_locomotion_sac",
            actor_hidden_dim=16,
            use_layer_norm=False,
            obs_normalization=True,
            priv_info_embed_dim=2,
            priv_mlp_hidden_dims=(4, 2),
            priv_info_normalization=True,
        ),
        device="cpu",
    )

    actions = oracle.actions_from_env_observation(
        {
            "obs": torch.tensor([[0.1, 0.2, 0.3]]).numpy(),
            "critic": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]).numpy(),
        },
        {},
    )
    assert actions.shape == (1, 2)
    assert torch.isfinite(torch.from_numpy(actions)).all()


def test_fada_collector_supplies_critic_tail_to_privileged_oracle() -> None:
    class PrivilegedOracle(torch.nn.Module):
        obs_dim = 3
        action_dim = 2
        calls = 0

        def actions_from_env_observation(self, obs, info):
            del info
            self.calls += 1
            assert obs["critic"].shape == (1, 5)
            return obs["critic"][:, -2:].astype("float32")

    class PrivilegedEnv(_CommandControlledEnv):
        def _state(self, commands):
            state = super()._state(commands)
            state.obs["critic"] = torch.cat(
                [torch.from_numpy(state.obs["obs"]), torch.ones((1, 2))], dim=1
            ).numpy()
            return state

    oracle = PrivilegedOracle()
    result = collect_fada_source_windows(
        PrivilegedEnv(),
        teacher_policy=oracle,
        config=_curriculum_config(),
        num_windows=1,
        spec=FADACollectionSpec(
            command_info_keys=("commands",),
            max_env_steps=16,
            collect_oracle_shadow=True,
        ),
    )
    assert result.batch.command.shape == (1, 3)
    assert oracle.calls > 0


def test_privileged_idm_task_composes_full_static_source_randomization() -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    conf_dir = Path(__file__).resolve().parents[2] / "conf" / "distill"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["task=g1_walk_flat/mujoco_fada_privileged_idm"],
        )

    assert cfg.training.fada.training_schedule == "idm_pretrain"
    assert cfg.training.fada.planner_updates == 0
    assert cfg.training.fada.async_artifact_dir == (
        "logs/fada/idm_pretrain_privileged_v022/source_batches"
    )
    assert cfg.training.fada.checkpoint_path == "logs/fada/idm_pretrain_privileged_v022.pt"
    assert cfg.teacher.task.endswith("mujoco_fada_privileged_oracle_grouped_dr_lineage")
    assert cfg.env.fada_privileged_observation.enabled is True
    assert cfg.env.ctrl_dt == pytest.approx(0.02)
    assert cfg.env.mode_observation is False
    assert cfg.env.gait_phase_enabled is False
    assert cfg.env.commands.rel_standing_envs == pytest.approx(0.3)
    assert cfg.env.commands.rel_transition_envs == pytest.approx(0.0)
    assert cfg.env.commands.vel_limit == [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]]
    assert cfg.env.commands.resampling_time == pytest.approx(0.0)
    assert cfg.env.commands.heading_command is False
    assert cfg.env.domain_rand.actuator_strength.curriculum_enabled is False
    assert list(cfg.env.domain_rand.actuator_strength.multiplier_range) == [0.8, 1.0]
    assert cfg.env.domain_rand.randomize_kp is True
    assert cfg.env.domain_rand.randomize_control_delay is False
    assert cfg.env.domain_rand.push_robots is False


def test_privileged_planner_task_composes_frozen_idm_stage() -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    conf_dir = Path(__file__).resolve().parents[2] / "conf" / "distill"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["task=g1_walk_flat/mujoco_fada_privileged_planner"],
        )

    assert cfg.training.fada.training_schedule == "planner_from_idm"
    assert cfg.training.fada.idm_updates == 0
    assert cfg.training.fada.planner_updates == 128
    assert cfg.training.fada.idm_initialization_path is None
    assert cfg.training.fada.async_artifact_dir == (
        "logs/fada/planner_from_privileged_idm_v022/source_batches"
    )
    assert cfg.training.fada.checkpoint_path == ("logs/fada/planner_from_privileged_idm_v022.pt")
    assert cfg.env.mujoco_num_threads == 1


def _privileged_collector_contract_fixture():
    cfg = OmegaConf.create(
        {
            "training": {"task_name": "G1WalkFlat", "sim_backend": "mujoco"},
            "env": {
                "ctrl_dt": 0.02,
                "mode_observation": False,
                "gait_phase_enabled": False,
                "control_config": {"action_scale": 1.0},
                "fada_privileged_observation": {
                    "enabled": True,
                    "schema": "g1_fada_privileged_v1",
                },
                "commands": {
                    "rel_standing_envs": 0.3,
                    "rel_transition_envs": 0.0,
                    "vel_limit": [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]],
                    "resampling_time": 0.0,
                    "heading_command": False,
                },
            },
        }
    )
    checkpoint_identity = {
        "privileged_schema": "g1_fada_privileged_v1",
        "task_name": "G1WalkFlat",
        "backend": "mujoco",
        "action_scale": [1.0],
        "body_names": ["world", "pelvis"],
        "actuated_joint_names": ["joint_0", "joint_1"],
        "privileged_field_slices": [["privileged", 0, 2]],
        "asset_sha256": "a" * 64,
    }
    env = SimpleNamespace(
        get_fada_privileged_checkpoint_identity=lambda: SimpleNamespace(
            body_names=("world", "pelvis"),
            actuated_joint_names=("joint_0", "joint_1"),
            field_slices=(("privileged", 0, 2),),
            asset_sha256="a" * 64,
        )
    )
    return cfg, checkpoint_identity, env


def test_privileged_oracle_collector_contract_accepts_matching_environment() -> None:
    import unilab.algos.torch.distill.fada_oracle as oracle_module

    validator = getattr(oracle_module, "validate_fada_oracle_environment_contract", None)
    assert callable(validator), "FADA Oracle environment contract validator is missing"
    cfg, checkpoint_identity, env = _privileged_collector_contract_fixture()

    validator(checkpoint_identity, env, cfg)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda cfg, _identity, _env: setattr(cfg.env, "gait_phase_enabled", True), "gait"),
        (
            lambda _cfg, identity, _env: identity.update({"action_scale": [0.5]}),
            "action_scale",
        ),
        (
            lambda _cfg, _identity, env: setattr(
                env.get_fada_privileged_checkpoint_identity(), "asset_sha256", "b" * 64
            ),
            "asset_sha256",
        ),
    ],
)
def test_privileged_oracle_collector_contract_rejects_semantic_mismatch(
    mutation,
    match: str,
) -> None:
    import unilab.algos.torch.distill.fada_oracle as oracle_module

    validator = getattr(oracle_module, "validate_fada_oracle_environment_contract", None)
    assert callable(validator), "FADA Oracle environment contract validator is missing"
    cfg, checkpoint_identity, env = _privileged_collector_contract_fixture()
    if match == "asset_sha256":
        mismatched_identity = SimpleNamespace(
            body_names=("world", "pelvis"),
            actuated_joint_names=("joint_0", "joint_1"),
            field_slices=(("privileged", 0, 2),),
            asset_sha256="b" * 64,
        )
        env.get_fada_privileged_checkpoint_identity = lambda: mismatched_identity
    else:
        mutation(cfg, checkpoint_identity, env)

    with pytest.raises(ValueError, match=match):
        validator(checkpoint_identity, env, cfg)


def test_walk_to_stand_uses_one_oracle_without_second_policy() -> None:
    class CountingOracle(torch.nn.Module):
        obs_dim = 3
        action_dim = 2

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.full((obs.shape[0], 2), 0.125, device=obs.device)

    oracle = CountingOracle()
    result = collect_fada_source_windows(
        _CommandControlledEnv(),
        teacher_policy=oracle,
        config=_curriculum_config(),
        num_windows=1,
        spec=FADACollectionSpec(
            command_info_keys=("commands",),
            command_scenario="walk_to_stand",
            transition_walk_command=(0.4, 0.0, 0.0),
            transition_pre_switch_steps=2,
            transition_post_switch_steps=4,
            max_env_steps=16,
            collect_oracle_shadow=True,
        ),
    )

    assert result.batch.command.shape == (1, 3)
    assert oracle.calls > 0
