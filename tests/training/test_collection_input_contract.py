"""Offline regressions for collection normalization and seeded G1 commands."""

from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.offpolicy.collector_session import OffPolicyCollectorSession
from unilab.envs.locomotion.g1.walk_commands import sample_g1_walk_commands
from unilab.training.seed import apply_training_seed


@pytest.mark.parametrize("algo_type", ["sac", "privileged_locomotion_sac"])
def test_collector_matches_learner_and_reloaded_normalizer(algo_type):
    normalizer = EmpiricalNormalization(shape=4, device="cpu")
    normalizer.update(torch.tensor([[1.0, 0.0, -0.1, -1.0], [1.0, 0.0002, 0.1, 1.0]]))
    raw = np.array([[1.01, 0.0003, 0.2, 2.0]], dtype=np.float32)
    expected = normalizer(torch.from_numpy(raw), update=False)
    restored = EmpiricalNormalization(shape=4, device="cpu").eval()
    restored.load_state_dict(normalizer.state_dict())

    class CaptureActor:
        def explore(self, obs, *args, **kwargs):
            self.observed = obs.clone()
            self.privileged = args
            return torch.zeros((len(obs), 1))

    session = object.__new__(OffPolicyCollectorSession)
    session.spec = SimpleNamespace(
        algo_type=algo_type,
        obs_normalization=True,
        shared_obs_normalizer_stats=SimpleNamespace(
            get=lambda: (normalizer.mean.numpy(), normalizer.std.numpy())
        ),
    )
    session.actor = CaptureActor()
    session.obs_np = raw
    privileged = np.array([[3.0, 4.0]], dtype=np.float32)
    session.critic_np = np.concatenate((raw, privileged), axis=1)
    session.info_dict = {}
    session.prev_dones_np = np.zeros(1, dtype=np.float32)
    session.trace_recorder = None
    session._select_actions()

    torch.testing.assert_close(session.actor.observed, expected)
    torch.testing.assert_close(session.actor.observed, restored(torch.from_numpy(raw)))
    if algo_type == "privileged_locomotion_sac":
        torch.testing.assert_close(session.actor.privileged[0], torch.from_numpy(privileged))


@pytest.mark.parametrize("transition_prob", [0.0, 0.3, 0.7])
def test_g1_normal_and_transition_commands_follow_training_seed(transition_prob):
    env = SimpleNamespace(cfg=SimpleNamespace(commands=SimpleNamespace(
        vel_limit=[[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]],
        transition_vel_limit=[[0.05, 0.0, 0.0], [0.25, 0.0, 0.0]],
        rel_standing_envs=0.3,
        rel_transition_envs=transition_prob,
    )))
    numpy_state, random_state = np.random.get_state(), random.getstate()
    try:
        apply_training_seed(1, torch_runtime=False)
        first = sample_g1_walk_commands(env, 512)
        second = sample_g1_walk_commands(env, 512)
        apply_training_seed(1, torch_runtime=False)
        np.testing.assert_array_equal(first, sample_g1_walk_commands(env, 512))
        np.testing.assert_array_equal(second, sample_g1_walk_commands(env, 512))
        assert not np.array_equal(first, second)
        apply_training_seed(2, torch_runtime=False)
        assert not np.array_equal(first, sample_g1_walk_commands(env, 512))
        standing = np.all(first == 0.0, axis=1)
        assert 0 < standing.sum() < len(first)
        assert np.all(first >= np.asarray(env.cfg.commands.vel_limit[0]))
        assert np.all(first <= np.asarray(env.cfg.commands.vel_limit[1]))
        if transition_prob > 0:
            transition = (~standing) & np.all(first[:, 1:] == 0.0, axis=1)
            assert transition.any()
            assert np.all((first[transition, 0] >= 0.05) & (first[transition, 0] <= 0.25))
    finally:
        np.random.set_state(numpy_state)
        random.setstate(random_state)
