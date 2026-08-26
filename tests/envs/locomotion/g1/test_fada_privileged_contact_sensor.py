from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from unilab.envs.locomotion.g1.fada_privileged import (
    DOF_POSITION_BIAS_LIMIT_RAD,
    TORQUE_RFI_FRACTION,
    apply_fada_pd_target_perturbation,
    split_net_contact_sensor,
)

ROOT = Path(__file__).resolve().parents[4]


def test_g1_locomotion_task_exposes_batched_net_foot_contact_wrenches() -> None:
    model_path = ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))

    for name in ("left_foot_net_contact", "right_foot_net_contact"):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        assert sensor_id >= 0
        assert model.sensor_dim[sensor_id] == 4  # found + global-frame net force xyz


def test_net_contact_sensor_preserves_force_and_derives_binary_flag() -> None:
    force, flag = split_net_contact_sensor(
        np.array([[0.0, 1.0, 2.0, 3.0], [2.0, -4.0, 5.0, 6.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(force, [[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(flag, [[0.0], [1.0]])


def test_moderate_pd_perturbation_has_confirmed_bounds_and_exact_torque_equivalence() -> None:
    kp = np.array([[20.0, 40.0]], dtype=np.float64)
    tau_max = np.array([100.0, 50.0], dtype=np.float64)
    target = np.array([[0.2, -0.3]], dtype=np.float64)
    dof_bias = np.array([[DOF_POSITION_BIAS_LIMIT_RAD, -DOF_POSITION_BIAS_LIMIT_RAD]])
    torque_rfi = np.array([[TORQUE_RFI_FRACTION * tau_max[0], -TORQUE_RFI_FRACTION * tau_max[1]]])

    perturbed = apply_fada_pd_target_perturbation(
        target,
        dof_position_bias=dof_bias,
        torque_rfi=torque_rfi,
        kp=kp,
        tau_max=tau_max,
    )

    np.testing.assert_allclose(kp * (perturbed - target - dof_bias), torque_rfi)


@pytest.mark.parametrize("bad_kp", [0.0, -1.0])
def test_pd_perturbation_rejects_nonpositive_kp(bad_kp: float) -> None:
    with pytest.raises(ValueError, match="Kp"):
        apply_fada_pd_target_perturbation(
            np.zeros((1, 1)),
            dof_position_bias=np.zeros((1, 1)),
            torque_rfi=np.zeros((1, 1)),
            kp=np.full((1, 1), bad_kp),
            tau_max=np.ones(1),
        )


def test_privileged_oracle_env_reset_materializes_exact_critic_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.base.registry import ensure_registries
    from unilab.training import BackendAdapter, create_env

    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "env-contract-test")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "algo=sac",
                "task=sac/g1_walk_flat/mujoco_fada_privileged_oracle",
            ],
        )
    ensure_registries()
    override = BackendAdapter(cfg, root_dir=ROOT, algo_name="sac").build_task_env_cfg_override()
    env = create_env(
        cfg,
        num_envs=4,
        env_cfg_override=override,
        sim_backend="mujoco",
    )
    try:
        root_clearance = np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
        monkeypatch.setattr(env, "_terrain_relative_base_height", lambda: root_clearance)
        obs, _ = env.reset(np.asarray([1, 3], dtype=np.int32))
        identity = env.get_fada_privileged_checkpoint_identity()
        body_count = len(identity.body_names)
        expected_critic_dim = 98 + 174 + body_count
        assert obs["obs"].shape == (2, 98)
        assert obs["critic"].shape == (2, expected_critic_dim)
        assert env.obs_groups_spec["critic"] == expected_critic_dim
        root_start = next(
            start for name, start, _ in identity.field_slices if name == "root_clearance"
        )
        np.testing.assert_allclose(
            obs["critic"][:, 98 + root_start],
            np.asarray([0.2, 0.4], dtype=np.float32),
        )
    finally:
        env.close()
