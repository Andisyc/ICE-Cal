"""Accepted reward/clock revision through owner boundaries, without simulation."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.curriculum import EpisodeLengthTracker, PenaltyCurriculum
from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.g1.walk_config import G1RewardConfig, G1WalkFlatCfg
from unilab.envs.locomotion.g1.walk_control_bindings import G1WalkControlBindings
from unilab.envs.locomotion.g1.walk_domain_randomization import G1WalkDomainRandomizationProvider
from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings
from unilab.envs.locomotion.g1.walk_runtime_bindings import G1WalkRuntimeBindings
from unilab.algos.torch.offpolicy.collector_session import OffPolicyCollectorSession


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(config_dir=str(root / "conf/offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=[
            "algo=sac", "task=sac/g1_walk_flat/mujoco_fada_fixed_contact",
        ])
    typed = G1WalkFlatCfg()
    apply_cfg_overrides(typed, dict(
        OmegaConf.to_container(cfg.env, resolve=True),
        reward_config=OmegaConf.to_container(cfg.reward, resolve=True),
    ))
    typed.validate()
    return typed


class ClockOwner(G1WalkControlBindings, G1WalkRewardBindings):
    def __init__(self, config, count=2):
        self._cfg = self.cfg = config
        self._reward_cfg = config.reward_config
        self._num_envs = count
        self._enable_reward_log = False
        self.default_angles = np.zeros(29)
        self._gait_phase_delta = 2 * np.pi * self._reward_cfg.gait_frequency * config.ctrl_dt
        self._init_reward_functions()

    def _fada_privileged_enabled(self):
        return False

    def _debug_action_trace_enabled(self):
        return False


def test_blended_tracking_and_restored_penalties(config):
    owner = ClockOwner(config, 6)
    commands = np.array([[.1, 0, 0], [-.1, 0, 0], [0, .1, 0], [0, 0, .2], [0, 0, 0], [.001, 0, 0]])
    ctx = rewards.RewardContext(info={"commands": commands}, linvel=np.zeros((6, 3)),
                                gyro=np.zeros((6, 3)), dof_pos=np.zeros((6, 29)),
                                num_envs=6, tracking_sigma=.25)
    rc = owner._reward_cfg
    assert rc.scales["tracking_lin_vel"] == 2
    assert rc.scales["tracking_ang_vel"] == 1.5
    assert rc.scales["under_speed"] == -1
    assert rc.under_speed_mode == "forward"
    assert rc.scales["penalty_action_rate"] == -4
    assert rc.scales["pose"] == -.5
    np.testing.assert_equal(rc.pose_weights[:12], [.01, 1, 5, .01, 5, 5] * 2)
    np.testing.assert_equal(rc.pose_weights[12:], [50] * 17)

    # Same relative progress in forward, backward and lateral low-speed commands.
    for fraction, expected in [(0, .0366312778), (.5, .7357588823), (1, 2.0)]:
        ctx.linvel = fraction * commands
        score = owner._reward_tracking_lin_vel(ctx)
        np.testing.assert_allclose(2 * score[:3], expected, rtol=1e-6)
        np.testing.assert_allclose(owner._reward_under_speed(ctx), rewards.under_speed(ctx))
        dispatched = rewards.run_reward_dispatch(
            scales={"tracking_lin_vel": 2}, fns={"tracking_lin_vel": owner._reward_tracking_lin_vel},
            ctx=ctx, info=ctx.info, enable_log=False, ctrl_dt=.02,
        )
        np.testing.assert_allclose(dispatched[:3], expected * .02, rtol=1e-6)

    # High-speed commands retain the full original curve, including reverse motion.
    commands[:] = [[.5, 0, 0], [.8, 0, 0], [-.8, 0, 0], [0, .8, 0], [.6, .8, 0], [0, 0, .2]]
    for fraction in [-1, 0, .5, 1, 1.5]:
        ctx.linvel = fraction * commands
        np.testing.assert_allclose(owner._reward_tracking_lin_vel(ctx), rewards.tracking_lin_vel(ctx))
    commands[:] = 0
    ctx.linvel[:] = [.05, .02, 0]
    np.testing.assert_allclose(owner._reward_tracking_lin_vel(ctx), rewards.tracking_lin_vel(ctx))

    # Continuous transitions and finite near-zero scores; no command mutation.
    for boundary in [0, .05, .25, .5]:
        commands[:] = 0
        commands[:, 0] = [max(0, boundary - 1e-8), boundary, boundary + 1e-8] * 2
        ctx.linvel[:] = [.02, .01, 0]
        before = commands.copy()
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            score = owner._reward_tracking_lin_vel(ctx)
        assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
        np.testing.assert_allclose(score, score[1], atol=2e-6)
        np.testing.assert_array_equal(commands, before)
    commands[:] = [.375, 0, 0]
    ctx.linvel[:] = [.1875, 0, 0]
    np.testing.assert_allclose(2 * owner._reward_tracking_lin_vel(ctx), 1.236694498, rtol=1e-6)

    # Old saved configurations still select their original formula.
    assert G1RewardConfig.__dataclass_fields__["tracking_lin_relative_blend"].default is False
    rc.tracking_lin_relative_blend = False
    owner._reward_cfg.tracking_lin_error_scale = None
    owner._reward_cfg.under_speed_mode = "forward"
    np.testing.assert_allclose(owner._reward_tracking_lin_vel(ctx), rewards.tracking_lin_vel(ctx))
    np.testing.assert_allclose(owner._reward_under_speed(ctx), rewards.under_speed(ctx))
    rc.tracking_lin_error_scale = .2
    np.testing.assert_allclose(owner._reward_tracking_lin_vel(ctx), np.exp(-.1875 / .2))
    assert G1WalkFlatCfg().gait_clock_mode == "continuous"


def test_fixed_clock_reset_keyboard_resampling_and_advance(config):
    owner = ClockOwner(config)
    provider = object.__new__(G1WalkDomainRandomizationProvider)
    commands = np.array([[0., 0., 0.], [0., 0., .2]])
    info = provider._build_extra_info_updates_for_commands(owner, 2, commands)
    info["commands"] = commands.copy()
    state = NpEnvState({}, np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), info)
    np.testing.assert_allclose(info["gait_phase"], [[np.pi, np.pi], [0, np.pi]])
    owner.apply_action(np.zeros((2, 29)), state)
    np.testing.assert_allclose(info["gait_phase"][0], [np.pi, np.pi])
    np.testing.assert_allclose(info["gait_phase"][1], [owner._gait_phase_delta, np.pi+owner._gait_phase_delta])
    # Keyboard changes do not advance time; tiny nonzero commands are not a dead zone.
    info["commands"][0] = [.001, 0, 0]
    owner._synchronize_external_command_for_observation(info)
    np.testing.assert_allclose(info["gait_phase"][0], [0, np.pi])
    phase = info["gait_phase"].copy()
    owner._synchronize_external_command_for_observation(info)
    np.testing.assert_array_equal(info["gait_phase"], phase)
    assert owner._current_command_gait_mask(info).all()
    owner.apply_action(np.zeros((2, 29)), state)
    np.testing.assert_allclose(info["gait_phase"][0], [owner._gait_phase_delta, np.pi+owner._gait_phase_delta])
    # Training resampling uses exactly the same transition as keyboard input.
    owner._sample_next_commands = lambda _info, values: values.fill(0)
    owner._commit_command_state_for_next_observation(info)
    np.testing.assert_allclose(info["gait_phase"], np.full((2, 2), np.pi))
    owner._sample_next_commands = lambda _info, values: values.__setitem__((slice(None), 0), -.1)
    owner._commit_command_state_for_next_observation(info)
    np.testing.assert_allclose(info["gait_phase"], [[0, np.pi], [0, np.pi]])
    config.gait_clock_mode = "continuous"
    info["commands"].fill(0)
    owner.apply_action(np.zeros((2, 29)), state)
    np.testing.assert_allclose(info["gait_phase"], np.tile([owner._gait_phase_delta, np.pi+owner._gait_phase_delta], (2, 1)))


def test_playback_command_probe_synchronizes_null_without_reward(config):
    from unilab.visualization.playback_controls import _policy_obs_contains_command

    class ProbeOwner(G1WalkRuntimeBindings, ClockOwner):
        @property
        def state(self):
            return self._state

        def get_local_linvel(self):
            return np.zeros((1, 3))

        get_gyro = get_dof_pos = get_dof_vel = get_local_linvel

        def _compute_obs(self, info, *args):
            # Exercise the same consistency check that rejected the live probe.
            self._exact_null_command_mask(NS(info=info))
            return {"obs": info["commands"].copy()}

        def update_state(self, state):
            self._exact_null_command_mask(NS(info=state.info))
            pytest.fail("An observation probe must not run the reward transaction")

    owner = ProbeOwner(config, 1)
    owner._backend = NS(get_sensor_data=lambda _: np.array([[0., 0., 1.]]))
    provider = object.__new__(G1WalkDomainRandomizationProvider)
    commands = np.zeros((1, 3))
    info = provider._build_extra_info_updates_for_commands(owner, 1, commands)
    info["commands"] = commands.copy()
    info["steps"] = np.array([7], dtype=np.uint32)
    owner._state = NpEnvState(
        {"obs": commands.copy()}, np.array([.25]), np.zeros(1, bool),
        np.zeros(1, bool), info,
    )
    # External reset callbacks can leave the live command unchanged: the probe
    # must synchronize both its temporary moving command and the restored zero.
    assert _policy_obs_contains_command(owner, reset_fn=lambda: None)
    np.testing.assert_array_equal(owner.state.info["commands"], commands)
    np.testing.assert_array_equal(owner.state.info["command_is_null"], [True])
    np.testing.assert_allclose(owner.state.info["gait_phase"], [[np.pi, np.pi]])
    np.testing.assert_array_equal(owner.state.obs["obs"], commands)
    np.testing.assert_array_equal(owner.state.reward, [.25])
    np.testing.assert_array_equal(owner.state.info["steps"], [7])


class LifecycleOwner(G1WalkRuntimeBindings):
    """Actual step/update/curriculum sequence with a non-physical backend."""
    def __init__(self, config, fallen):
        self._cfg = self.cfg = config
        self._reward_cfg = config.reward_config
        self._num_envs = 1
        self.step_counter = config.max_episode_steps - 1
        self.fallen = fallen
        self._state = NpEnvState({"obs": np.zeros((1, 1)), "critic": np.zeros((1, 2))},
            np.zeros(1), np.zeros(1, bool), np.zeros(1, bool),
            {"steps": np.array([self.step_counter], dtype=np.uint32)})
        self._backend = NS(step=lambda *a: None, get_sensor_data=lambda _: np.array([[0., 0., 1.]]))
        self._dr_manager = self._nan_guard = None
        self._autoreset = False
        self._truncated_scratch = np.zeros(1, bool)
        self._fada_dr_provider = NS(update_iteration_curriculum=lambda *a: None)
        self._episode_tracker = EpisodeLengthTracker(1, window_size=1)
        self._penalty_curriculum = PenaltyCurriculum(self, initial_scale=.5)

    def _command_gated_phase_contact_enabled(self): return False
    def _advance_command_state_for_reward(self, info): pass
    def get_local_linvel(self): return np.zeros((1, 3))
    get_gyro = get_dof_pos = get_dof_vel = get_local_linvel
    def _refresh_command_gated_torque(self, *a): pass
    def _terrain_relative_base_height(self): return np.array([.2 if self.fallen else .754])
    def _forward_progress_failure(self, info): return np.zeros(1, bool)
    def _compute_reward(self, *a): return np.array([.25])
    def _debug_action_trace(self, *a, **kw): pass
    def _commit_command_state_for_next_observation(self, info): pass
    def _compute_obs(self, *a): return self._state.obs
    def _write_curriculum_log(self, info): pass
    def apply_action(self, actions, state): return actions
    def _clear_step_final_observation(self): pass
    def _check_physics_envelope(self): return None
    _compute_truncated = NpEnv._compute_truncated


@pytest.mark.parametrize("fallen", [False, True])
def test_completed_episode_and_collector_bootstrap(config, fallen):
    owner = LifecycleOwner(config, fallen)
    state = NpEnv.step(owner, np.zeros((1, 29)))
    assert state.truncated[0] and bool(state.terminated[0]) == fallen
    assert owner._episode_tracker.average_length == 1000
    assert owner._penalty_curriculum.current_scale == pytest.approx(.5005)
    captured = {}
    collector = object.__new__(OffPolicyCollectorSession)
    collector.spec = NS(num_envs=1, replay_buffer=NS(add=lambda *a, **kw: captured.update(args=a)))
    collector.dependencies = NS(terminal_contract_fn=resolve_terminal_observation_contract)
    collector._split_observations = lambda obs: (obs["obs"], obs["critic"])
    collector.obs_np, collector.critic_np = state.obs.values()
    collector.trace_recorder = None
    collector.done_count_window = collector.timeout_count_window = collector.terminated_count_window = 0
    collector.total_steps = collector.env_steps_since_sync = 0
    collector._service_pack_requests = lambda: None
    collector._record_episode_rows = lambda *a: None
    collector._store_transition(state, np.zeros((1, 29)))
    args = captured["args"]
    assert args[2].item() == .25
    bootstrap = torch.clamp(1-args[4]+args[5], 0, 1)
    assert bootstrap.item() == (0 if fallen else 1)


def test_old_checkpoint_does_not_inherit_new_clock(config, monkeypatch):
    from unilab.visualization import playback_checkpoint_contract as playback
    monkeypatch.setattr(playback, "_load_checkpoint_run_config", lambda _: {
        "config": {"env": {"mode_observation": False}, "reward": {"tracking_sigma": .25}}
    })
    result = playback.apply_checkpoint_env_contract(
        {"gait_clock_mode": config.gait_clock_mode}, NS(task="g1_walk_flat", algo="sac")
    )
    assert result["gait_clock_mode"] == "continuous"
