from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "conf/offpolicy/task/sac/g1_walk_flat"


def _compose_task(task: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "architecture-test")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        return compose(config_name="config", overrides=["algo=sac", f"task={task}"])


def _assert_randomization_disabled(domain_rand) -> None:
    disabled_flags = (
        "randomize_ground_friction",
        "random_com",
        "randomize_base_mass",
        "randomize_body_mass",
        "randomize_gravity",
        "randomize_dof_armature",
        "randomize_kp",
        "randomize_kd",
        "randomize_dof_position_bias",
        "randomize_control_delay",
        "push_robots",
    )
    for field in disabled_flags:
        assert OmegaConf.select(domain_rand, field, default=False) is False
    assert float(OmegaConf.select(domain_rand, "torque_rfi_fraction", default=0.0)) == 0.0
    assert OmegaConf.select(domain_rand, "actuator_strength.enabled", default=False) is False


def test_clean_baseline_is_gait_free_privilege_free_and_randomization_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose_task("sac/g1_walk_flat/mujoco_clean_baseline", monkeypatch)

    assert cfg.env.gait_phase_enabled is False
    assert cfg.env.fada_privileged_observation.enabled is False
    assert cfg.reward.scales.feet_phase == pytest.approx(0.0)
    assert cfg.reward.gait_constraint.enabled is False
    _assert_randomization_disabled(cfg.env.domain_rand)
    assert OmegaConf.select(cfg, "algo.runtime_impl") is None


def test_canonical_fada_source_disables_left_knee_strength_without_gait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose_task("sac/g1_walk_flat/mujoco_fada_source", monkeypatch)

    assert cfg.algo.runtime_impl == "privileged_locomotion_sac"
    assert cfg.algo.actor.oracle_behavior_profile == "phase_neutral_mixed_v1"
    assert cfg.env.gait_phase_enabled is False
    assert cfg.reward.scales.feet_phase == pytest.approx(0.0)
    assert list(cfg.env.domain_rand.actuator_strength.candidate_actuator_indices) == [3]
    assert list(cfg.env.domain_rand.actuator_strength.multiplier_range) == [0.8, 1.0]
    assert cfg.env.domain_rand.actuator_strength.enabled is False
    assert cfg.env.domain_rand.actuator_strength.curriculum_enabled is False
    assert cfg.env.domain_rand.actuator_strength.group_curriculum_enabled is False
    assert cfg.env.domain_rand.randomize_kp is True
    assert cfg.env.domain_rand.randomize_kd is True
    assert cfg.algo.privileged_grouped_dr_lineage is True


def test_canonical_fada_phase_owns_phase_behavior_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose_task("sac/g1_walk_flat/mujoco_fada_phase", monkeypatch)

    assert cfg.algo.actor.oracle_behavior_profile == "phase_locomotion_v1"
    assert cfg.env.gait_phase_enabled is True
    assert cfg.env.commands.rel_standing_envs == pytest.approx(0.0)
    assert cfg.reward.scales.feet_phase == pytest.approx(5.0)
    assert cfg.algo.privileged_grouped_dr_lineage is True


def test_context_teacher_has_one_canonical_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _compose_task("sac/g1_walk_flat/mujoco_context_teacher", monkeypatch)

    assert cfg.algo.runtime_impl == "privileged_full_action_sac"
    assert cfg.algo.actor_lr == pytest.approx(0.00003)
    assert cfg.algo.actor.nominal_action_anchor_coef == pytest.approx(10.0)
    assert cfg.env.forward_progress_termination.enabled is True
    assert cfg.reward.scales.penalty_lateral_corridor_violation == pytest.approx(-20.0)


def test_slope_identity_is_owned_by_target_domain_group() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        cfg = compose(
            config_name="fada_slope_target",
            overrides=["target_domain=slope_10"],
            return_hydra_config=True,
        )

    assert cfg.hydra.runtime.choices.task == "sac/g1_walk_flat/mujoco_fada_target_phase_locomotion"
    assert cfg.env.scene.model_file.endswith("scene_slope_10.xml")
    assert cfg.env.domain_rand.actuator_strength.enabled is False
    assert not (TASK_DIR / "mujoco_fada_slope_10.yaml").exists()
    assert not (TASK_DIR / "mujoco_fada_slope_15.yaml").exists()


def test_privileged_runtime_validates_structure_without_owning_fault_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unilab.algos.torch.distill.fada.source import (
        resolve_privileged_locomotion_sac_runtime,
    )

    cfg = _compose_task("sac/g1_walk_flat/mujoco_fada_source", monkeypatch)
    cfg.env.domain_rand.actuator_strength.candidate_actuator_indices = [4]
    cfg.env.domain_rand.actuator_strength.multiplier_range = [0.7, 0.9]
    cfg.env.domain_rand.actuator_strength.nominal_probability = 0.25
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )

    assert runtime is not None
    runtime.validate_training_config(cfg)


@pytest.mark.parametrize(
    ("profile", "owner"),
    [
        ("mujoco_clean_baseline", "/task/sac/g1_walk_flat/mujoco"),
        ("mujoco_fada_source", "/task/sac/g1_walk_flat/mujoco"),
        ("mujoco_fada_phase", "/task/sac/g1_walk_flat/mujoco_fada_source"),
    ],
)
def test_canonical_profiles_have_one_declared_owner(profile: str, owner: str) -> None:
    payload = OmegaConf.load(TASK_DIR / f"{profile}.yaml")
    defaults = [str(item) for item in payload.defaults if str(item) != "_self_"]

    assert defaults == [owner]


def test_historical_fada_profiles_and_launchers_are_not_active() -> None:
    retired = {
        "mujoco_no_gait_single_reward.yaml",
        "mujoco_fada_privileged_oracle.yaml",
        "mujoco_fada_privileged_oracle_fixed_input_curriculum.yaml",
        "mujoco_fada_privileged_oracle_live_input_curriculum.yaml",
        "mujoco_fada_privileged_oracle_live_input_nominal_5k.yaml",
        "mujoco_fada_privileged_oracle_live_input_dr_curriculum.yaml",
        "mujoco_fada_privileged_oracle_grouped_dr_lineage.yaml",
        "mujoco_fada_privileged_oracle_phase_locomotion_grouped_dr_lineage.yaml",
        "mujoco_fada_privileged_oracle_command_phase_grouped_dr_lineage.yaml",
        "mujoco_fada_privileged_oracle_simple_height_grouped_dr_lineage.yaml",
        "mujoco_fada_privileged_oracle_phase_height_v3_grouped_dr_lineage.yaml",
        "mujoco_fada_privileged_oracle_original_height_grouped_dr_lineage.yaml",
        "mujoco_context_teacher_full_action_left_knee_090.yaml",
        "mujoco_context_teacher_full_action_v005.yaml",
        "mujoco_context_teacher_full_action_v006.yaml",
        "mujoco_context_teacher_full_action_v007.yaml",
    }

    assert retired.isdisjoint(path.name for path in TASK_DIR.glob("*.yaml"))
    assert not (ROOT / "scripts/train_original_height_oracle.sh").exists()
    assert not (ROOT / "scripts/train_simple_height_oracle.sh").exists()


def test_offpolicy_cli_is_a_thin_composition_root() -> None:
    script = (ROOT / "scripts/train_offpolicy.py").read_text(encoding="utf-8")

    assert "from unilab.training.offpolicy import" in script
    assert "def build_runner(" not in script
    assert "def play_offpolicy(" not in script
    assert len(script.splitlines()) < 100


def test_fada_exposes_method_stage_namespaces() -> None:
    from unilab.algos.torch.distill.fada.planner_idm import FADAPlannerIDMPolicy
    from unilab.algos.torch.distill.fada.source.runtime import (
        resolve_privileged_locomotion_sac_runtime,
    )
    from unilab.algos.torch.distill.fada.target.adaptation import (
        run_fada_target_adaptation,
    )
    from unilab.algos.torch.distill.fada.target.collection import run_fada_target_collection
    from unilab.algos.torch.distill.fada.target.evaluation import run_fada_target_evaluation

    assert callable(resolve_privileged_locomotion_sac_runtime)
    assert FADAPlannerIDMPolicy is not None
    assert callable(run_fada_target_collection)
    assert callable(run_fada_target_adaptation)
    assert callable(run_fada_target_evaluation)
