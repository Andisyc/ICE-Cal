from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from unilab.algos.torch.distill import FADAArchitectureConfig
from unilab.algos.torch.distill.fada.target_collector import FADASlopeEpisodePolicy
from unilab.algos.torch.distill.fada.target_domain import FADASlopeGeometry
from unilab.algos.torch.distill.fada.target_evaluation import _run_pair

ROOT = Path(__file__).resolve().parents[2]


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = FADAArchitectureConfig(
            obs_dim=3,
            action_dim=2,
            command_dim=3,
            history_length=2,
            prediction_horizon=2,
            hidden_dim=8,
            num_heads=2,
            planner_layers=1,
            idm_encoder_layers=1,
            idm_decoder_layers=1,
            feedforward_dim=16,
        )

    def forward(self, observation_history, _action_history, _command):
        return SimpleNamespace(action=observation_history[:, -1, :2])


class _Env:
    num_envs = 1

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(ctrl_dt=0.02)
        self.restore_calls = 0
        self.step_count = 0
        self._state()

    def _state(self) -> None:
        self.state = SimpleNamespace(
            obs={"obs": np.asarray([[float(self.step_count), 0.0, 0.0]], dtype=np.float32)},
            info={"commands": np.zeros((1, 3), dtype=np.float32)},
            terminated=np.zeros(1, dtype=np.bool_),
            truncated=np.zeros(1, dtype=np.bool_),
        )

    def set_autoreset(self, _enabled: bool) -> None:
        pass

    def reset_all(self):
        self.step_count = 0
        self._state()
        return self.state

    def capture_rollout_snapshot(self):
        return self.step_count

    def restore_rollout_snapshot(self, snapshot):
        self.restore_calls += 1
        self.step_count = snapshot
        self._state()

    def refresh_state(self) -> None:
        pass

    def step(self, _action):
        self.step_count += 1
        self._state()
        return self.state

    def get_base_pos(self):
        return np.asarray([[0.1 * self.step_count, 0.0, 0.8]])

    def get_base_quat(self):
        return np.asarray([[1.0, 0.0, 0.0, 0.0]])

    def get_foot_pos(self):
        x = 0.1 * self.step_count
        return np.asarray([[[x, 0.1, 0.0], [x, -0.1, 0.0]]])

    def get_base_lin_vel(self):
        return np.asarray([[0.1, 0.0, 0.0]])

    def get_physics_state_snapshot(self):
        return np.asarray([[self.step_count]], dtype=np.float32)


def test_evaluation_config_owns_same_condition_pair_and_flat_regression() -> None:
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        cfg = compose(config_name="fada_slope_evaluate", return_hydra_config=True)

    assert cfg.hydra.runtime.choices.task == "sac/g1_walk_flat/mujoco_fada_slope_15"
    assert list(cfg.evaluation.command) == [0.8, 0.0, 0.0]
    assert cfg.evaluation.run_flat_regression is True


def test_rollout_pair_restores_complete_snapshot_before_each_policy() -> None:
    env = _Env()
    policy = _Policy()
    geometry = FADASlopeGeometry(15.0, 0.8, 1.5, 8.0, 0.25, 0.5)

    zero, adapted = _run_pair(
        env,
        policy,
        policy,
        command=np.asarray([0.8, 0.0, 0.0], dtype=np.float32),
        control_steps=4,
        ramp_steps=2,
        episode_policy=FADASlopeEpisodePolicy(geometry, ((0.8, 0.0, 0.0),)),
    )

    assert env.restore_calls == 2
    np.testing.assert_array_equal(zero.base_pos_w, adapted.base_pos_w)
    np.testing.assert_array_equal(zero.command_forward_mps, adapted.command_forward_mps)
    assert len(zero.physics_states) == len(adapted.physics_states) == 4
