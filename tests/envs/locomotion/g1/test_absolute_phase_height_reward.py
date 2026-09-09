"""Absolute phase-height reward integration, without simulation."""

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.registry import apply_cfg_overrides
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.walk_config import G1WalkFlatCfg
from unilab.envs.locomotion.g1.walk_math import (
    compute_planar_command_speed_gate,
)
from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings


class _Backend:
    def __init__(self, count: int):
        self.left_foot = np.zeros((count, 3), dtype=get_global_dtype())
        self.right_foot = np.zeros((count, 3), dtype=get_global_dtype())

    def get_base_pos(self) -> np.ndarray:
        base = np.zeros_like(self.left_foot)
        base[:, 2] = 0.754
        return base

    def get_sensor_data(self, name: str) -> np.ndarray:
        if name == "left_foot_pos":
            return self.left_foot
        if name == "right_foot_pos":
            return self.right_foot
        raise KeyError(name)


class _RewardOwner(G1WalkRewardBindings):
    def __init__(self, config: G1WalkFlatCfg, count: int):
        self._cfg = self.cfg = config
        self._reward_cfg = config.reward_config
        self._num_envs = count
        self._enable_reward_log = False
        self.default_angles = np.zeros(29, dtype=get_global_dtype())
        self._pose_weights = np.asarray(
            self._reward_cfg.pose_weights, dtype=get_global_dtype()
        )
        self._backend = _Backend(count)
        self._init_reward_functions()

    def _log_current_action_authority(self, info: dict) -> None:
        del info


def _fixed_contact_config(monkeypatch) -> G1WalkFlatCfg:
    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "absolute-phase-height-test")
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(config_dir=str(root / "conf/offpolicy"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["algo=sac", "task=sac/g1_walk_flat/mujoco_fada_fixed_contact"],
        )
    typed = G1WalkFlatCfg()
    apply_cfg_overrides(
        typed,
        dict(
            OmegaConf.to_container(cfg.env, resolve=True),
            reward_config=OmegaConf.to_container(cfg.reward, resolve=True),
        ),
    )
    typed.validate()
    return typed


def test_absolute_phase_height_reward_is_gated_and_aggregated(monkeypatch):
    cfg = _fixed_contact_config(monkeypatch)
    reward_cfg = cfg.reward_config
    assert cfg.gait_phase_enabled is True
    assert cfg.gait_phase_init_mode == "offset_phase"
    assert cfg.gait_clock_mode == "continuous"
    assert reward_cfg.feet_phase_mode == "absolute_phase_height"
    assert reward_cfg.feet_phase_swing_height == 0.04
    assert reward_cfg.feet_phase_tracking_sigma == 0.008
    assert reward_cfg.min_planar_command_speed_for_gait_reward == 0.4
    assert reward_cfg.scales["feet_phase"] == 5.0
    assert reward_cfg.scales["phase_contact"] == 0.0
    assert reward_cfg.scales["feet_phase_contact"] == 0.0
    assert reward_cfg.scales["feet_phase_contrast"] == 0.0
    assert "feet_phase" not in reward_cfg.penalty_curriculum_terms

    commands = np.asarray(
        [[0.0, 0.0, 0.0], [0.39, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.5, 0.0]],
        dtype=get_global_dtype(),
    )
    np.testing.assert_array_equal(
        compute_planar_command_speed_gate(commands, 0.4),
        [0.0, 0.0, 1.0, 1.0],
    )

    # Keep one existing term to prove the new reward is added by the normal
    # aggregation path instead of bypassing it.
    reward_cfg.scales = {"alive": 10.0, "feet_phase": 5.0}
    owner = _RewardOwner(cfg, len(commands))
    owner._backend.left_foot[:, 2] = 0.04
    info = {
        "commands": commands,
        "gait_phase": np.tile([0.0, np.pi], (len(commands), 1)),
    }
    zeros3 = np.zeros((len(commands), 3), dtype=get_global_dtype())
    zeros29 = np.zeros((len(commands), 29), dtype=get_global_dtype())
    total = owner._compute_reward(info, zeros3, zeros3, zeros3, zeros29, zeros29)
    np.testing.assert_allclose(
        total,
        np.asarray([10.0, 10.0, 15.0, 15.0]) * cfg.ctrl_dt,
        rtol=1e-6,
    )

    # Absolute height, not height relative to the lower foot: translating both
    # feet upward by 1 cm must reduce the phase score.
    owner._backend.left_foot[:, 2] = 0.05
    owner._backend.right_foot[:, 2] = 0.01
    translated = owner._compute_reward(info, zeros3, zeros3, zeros3, zeros29, zeros29)
    translated_score = np.exp(-2.0 * 0.01**2 / 0.008)
    expected = np.asarray(
        [10.0, 10.0, 10.0 + 5.0 * translated_score, 10.0 + 5.0 * translated_score]
    ) * cfg.ctrl_dt
    np.testing.assert_allclose(translated, expected, rtol=1e-6)
