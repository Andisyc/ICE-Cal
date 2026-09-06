from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings


@pytest.mark.parametrize("phase", [[np.pi / 2, 3 * np.pi / 2], [3 * np.pi / 2, np.pi / 2]])
def test_matching_left_or_right_peak_has_zero_cost(phase):
    heights = [0.06, 0.0] if phase[0] < np.pi else [0.0, 0.06]

    class Backend:
        def get_sensor_data(self, name):
            if name.endswith("upvector"):
                return np.array([[0.0, 0.0, 1.0]])
            height = heights[0] if name.startswith("left") else heights[1]
            return np.array([[0.0, 0.0, height - 0.002]])

    owner = SimpleNamespace(
        _backend=Backend(),
        _num_envs=1,
        _reward_cfg=SimpleNamespace(
            feet_phase_mode="command_height_v2",
            feet_phase_swing_height=0.06,
            feet_phase_command_speed_scale=0.3,
            feet_phase_turn_length=0.3,
            feet_phase_height_scale=0.09,
        ),
    )
    ctx = SimpleNamespace(
        info={"commands": np.array([[0.3, 0, 0]]), "gait_phase": np.array([phase])}
    )
    np.testing.assert_allclose(G1WalkRewardBindings._reward_feet_phase(owner, ctx), 0, atol=1e-12)
    heights.reverse()
    np.testing.assert_allclose(
        G1WalkRewardBindings._reward_feet_phase(owner, ctx), -4 / 9, atol=1e-12
    )


def test_simple_height_binding_tracks_current_command_and_phase():
    class Backend:
        def get_sensor_data(self, name):
            if name.endswith("upvector"):
                return np.tile([0.0, 0.0, 1.0], (4, 1))
            return np.tile([0.0, 0.0, -0.002], (4, 1))

    owner = SimpleNamespace(
        _backend=Backend(),
        _num_envs=4,
        _reward_cfg=SimpleNamespace(
            feet_phase_mode="command_height_v2",
            feet_phase_swing_height=0.06,
            feet_phase_command_speed_scale=0.3,
            feet_phase_turn_length=0.3,
            feet_phase_height_scale=0.09,
        ),
    )
    info = {
        "commands": np.array([[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0], [0, 0, 1.0]]),
        "gait_phase": np.tile([np.pi / 2, 3 * np.pi / 2], (4, 1)),
    }
    result = G1WalkRewardBindings._reward_feet_phase(owner, SimpleNamespace(info=info))
    np.testing.assert_allclose(result, [0, -2 / 9, -8 / 9, -2 / 9], atol=1e-12)
    info["commands"][:] = 0
    np.testing.assert_allclose(
        G1WalkRewardBindings._reward_feet_phase(owner, SimpleNamespace(info=info)),
        0,
        atol=1e-12,
    )
    info["commands"][:, 0] = 0.3
    info["gait_phase"][:] = [0.0, np.pi]
    np.testing.assert_allclose(
        G1WalkRewardBindings._reward_feet_phase(owner, SimpleNamespace(info=info)),
        -1 / 36,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "task, mode",
    [
        ("simple_height", "command_height_v2"),
        ("phase_height_v3", "command_height_v3"),
    ],
)
def test_simple_height_task_runtime_admission(monkeypatch, task, mode):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
        resolve_privileged_locomotion_sac_runtime,
    )

    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "simple-height-test")
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(config_dir=str(root / "conf/offpolicy"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"task=sac/g1_walk_flat/mujoco_fada_privileged_oracle_{task}_grouped_dr_lineage"
            ],
        )
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )
    runtime.validate_training_config(cfg)
    assert cfg.reward.feet_phase_mode == mode
    assert cfg.reward.scales.feet_phase == 1.0
    assert cfg.reward.scales.feet_phase_contact == 0
    assert cfg.reward.scales.feet_phase_contrast == 0
    from unilab.training.backend_adapter import BackendAdapter

    override = BackendAdapter(cfg, root_dir=root).build_task_env_cfg_override()
    assert override["reward_config"]["feet_phase_mode"] == mode
    assert override["gait_phase_enabled"] is True
    assert cfg.algo.max_iterations == 5000
    assert cfg.algo.save_interval == 240
