from __future__ import annotations

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.base.np_env import NpEnvState
from unilab.envs.common.rotation import np_yaw_to_quat


class _FakePairedEnv:
    def __init__(self, *, corrupt_restore: bool = False, emit_done: bool = False) -> None:
        self.num_envs = 3
        self.action_space = SimpleNamespace(shape=(3,))
        self._autoreset = True
        self._corrupt_restore = corrupt_restore
        self._emit_done = emit_done
        self._step_index = 0
        self._base_pos = np.zeros((3, 3), dtype=np.float32)
        self._yaw = np.asarray([0.0, np.pi / 2.0, -np.pi / 2.0], dtype=np.float32)
        self._local_linvel = np.zeros((3, 3), dtype=np.float32)
        strength = np.ones((3, 29), dtype=np.float32)
        strength[1, 3] = 0.9
        strength[2, 3] = 0.9
        self._state = NpEnvState(
            obs={"obs": np.zeros((3, 5), dtype=np.float32)},
            reward=np.zeros((3,), dtype=np.float32),
            terminated=np.zeros((3,), dtype=np.bool_),
            truncated=np.zeros((3,), dtype=np.bool_),
            info={
                "commands": np.asarray([[0.4, 0.0, 0.0]] * 3, dtype=np.float32),
                "privileged_actuator_strength": strength,
            },
        )

    @property
    def state(self) -> NpEnvState:
        return self._state

    def get_base_pos(self) -> np.ndarray:
        return self._base_pos

    def get_base_quat(self) -> np.ndarray:
        return np_yaw_to_quat(self._yaw)

    def get_local_linvel(self) -> np.ndarray:
        return self._local_linvel

    def set_autoreset(self, enabled: bool) -> None:
        self._autoreset = bool(enabled)

    def capture_rollout_snapshot(self):
        return copy.deepcopy(
            (
                self._state,
                self._base_pos,
                self._yaw,
                self._local_linvel,
                self._step_index,
                np.random.get_state(),
                self._autoreset,
            )
        )

    def restore_rollout_snapshot(self, snapshot) -> None:
        (
            self._state,
            self._base_pos,
            self._yaw,
            self._local_linvel,
            self._step_index,
            rng_state,
            self._autoreset,
        ) = copy.deepcopy(snapshot)
        np.random.set_state(rng_state)
        if self._corrupt_restore:
            self._base_pos[0, 0] += 0.01

    @contextmanager
    def preserve_rollout_state(self):
        snapshot = self.capture_rollout_snapshot()
        self.set_autoreset(False)
        try:
            yield
        finally:
            self.restore_rollout_snapshot(snapshot)

    def step(self, actions: np.ndarray) -> NpEnvState:
        assert self._autoreset is False
        actions = np.asarray(actions, dtype=np.float32)
        self._local_linvel = actions.copy()
        yaw = self._yaw
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        self._base_pos[:, 0] += cos_yaw * actions[:, 0] - sin_yaw * actions[:, 1]
        self._base_pos[:, 1] += sin_yaw * actions[:, 0] + cos_yaw * actions[:, 1]
        self._yaw += actions[:, 2]
        self._step_index += 1
        terminated = np.zeros((3,), dtype=np.bool_)
        truncated = np.zeros((3,), dtype=np.bool_)
        if self._emit_done and self._step_index == 1:
            terminated[0] = True
            truncated[1] = True
        self._state = self._state.replace(
            obs={"obs": np.full((3, 5), self._step_index, dtype=np.float32)},
            terminated=terminated,
            truncated=truncated,
        )
        return self._state


class _FakeResidualActor:
    obs_dim = 5
    priv_info_dim = 29
    action_dim = 3
    residual_scale = 0.5

    def __init__(self, *, nominal_forward: float = 0.0, residual_forward: float = 0.4) -> None:
        self.nominal_forward = float(nominal_forward)
        self.residual_forward = float(residual_forward)

    def nominal_action(self, obs: torch.Tensor) -> torch.Tensor:
        action = torch.zeros((obs.shape[0], 3), dtype=obs.dtype, device=obs.device)
        action[:, 0] = self.nominal_forward
        return action

    def residual_action(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
        *,
        deterministic: bool,
    ) -> torch.Tensor:
        assert deterministic is True
        assert priv_info.shape == (obs.shape[0], 29)
        action = torch.zeros((obs.shape[0], 3), dtype=obs.dtype, device=obs.device)
        action[:, 0] = self.residual_forward
        return action

    @staticmethod
    def fuse_action(nominal_action: torch.Tensor, delta_action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(nominal_action + delta_action, -1.0, 1.0)


class _FakeBaselineActor:
    obs_dim = 5
    action_dim = 3

    def explore(self, obs: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
        assert deterministic is True
        action = torch.zeros((obs.shape[0], 3), dtype=obs.dtype, device=obs.device)
        action[:, 0] = 0.2
        return action


class _FakeFullActionTeacher:
    obs_dim = 5
    priv_info_dim = 29
    action_dim = 3

    def explore(
        self, obs: torch.Tensor, strength: torch.Tensor, *, deterministic: bool
    ) -> torch.Tensor:
        assert deterministic is True
        assert strength.shape == (obs.shape[0], 29)
        action = torch.zeros((obs.shape[0], 3), dtype=obs.dtype, device=obs.device)
        action[:, 0] = 0.4
        return action


def _fixed_left_knee_env() -> _FakePairedEnv:
    env = _FakePairedEnv()
    strength = np.ones((env.num_envs, 29), dtype=np.float32)
    strength[:, 3] = 0.9
    env._state = env._state.replace(
        info={**env._state.info, "privileged_actuator_strength": strength}
    )
    return env


def test_full_action_evaluator_compares_independent_policies_at_fixed_090() -> None:
    from unilab.algos.torch.fada_context.full_action_paired_evaluation import (
        evaluate_full_action_paired_rollouts,
    )

    env = _fixed_left_knee_env()
    report = evaluate_full_action_paired_rollouts(
        env,
        _FakeBaselineActor(),
        _FakeFullActionTeacher(),
        steps=2,
        device="cpu",
    )
    assert report["pairing"]["exact_start_match"] is True
    assert report["baseline"]["overall"]["forward_velocity_mae_mps"] == pytest.approx(0.2)
    assert report["teacher"]["overall"]["forward_velocity_mae_mps"] == pytest.approx(0.0)
    assert report["teacher"]["overall"]["action_delta_l2_mean"] == pytest.approx(0.2)


def test_full_action_evaluator_rejects_mixed_strength_rows() -> None:
    from unilab.algos.torch.fada_context.full_action_paired_evaluation import (
        evaluate_full_action_paired_rollouts,
    )

    with pytest.raises(ValueError, match="every row to be fixed left-knee 0.9"):
        evaluate_full_action_paired_rollouts(
            _FakePairedEnv(),
            _FakeBaselineActor(),
            _FakeFullActionTeacher(),
            steps=1,
            device="cpu",
        )


def test_paired_evaluator_uses_exact_snapshot_and_initial_yaw_frame() -> None:
    from unilab.algos.torch.fada_context.paired_evaluation import evaluate_paired_rollouts

    env = _FakePairedEnv()
    initial_position = env.get_base_pos().copy()

    report = evaluate_paired_rollouts(
        env,
        _FakeResidualActor(),
        steps=2,
        device="cpu",
    )

    assert report["pairing"]["exact_start_match"] is True
    assert report["scenario_counts"] == {"left_knee": 2, "nominal": 1}
    assert report["nominal"]["overall"]["forward_velocity_mae_mps"] == pytest.approx(0.4)
    assert report["teacher"]["overall"]["forward_velocity_mae_mps"] == pytest.approx(0.0)
    assert report["teacher"]["overall"]["max_lateral_abs_m"] == pytest.approx(0.0, abs=1e-6)
    assert report["improvement_lower_is_better"]["overall"][
        "forward_velocity_mae_mps"
    ] == pytest.approx(0.4)
    assert report["teacher"]["overall"]["residual_l2_mean"] == pytest.approx(0.4)
    np.testing.assert_allclose(env.get_base_pos(), initial_position)
    assert env._autoreset is True


def test_paired_evaluator_separates_physical_fall_from_truncation() -> None:
    from unilab.algos.torch.fada_context.paired_evaluation import evaluate_paired_rollouts

    report = evaluate_paired_rollouts(
        _FakePairedEnv(emit_done=True),
        _FakeResidualActor(),
        steps=3,
        device="cpu",
    )

    assert report["nominal"]["overall"]["fall_rate"] == pytest.approx(1.0 / 3.0)
    assert report["nominal"]["overall"]["truncation_rate"] == pytest.approx(1.0 / 3.0)
    assert report["nominal"]["overall"]["survival_steps_mean"] == pytest.approx(5.0 / 3.0)


def test_paired_evaluator_measures_raw_fusion_clipping() -> None:
    from unilab.algos.torch.fada_context.paired_evaluation import evaluate_paired_rollouts

    report = evaluate_paired_rollouts(
        _FakePairedEnv(),
        _FakeResidualActor(nominal_forward=0.9, residual_forward=0.2),
        steps=2,
        device="cpu",
    )

    teacher = report["teacher"]["overall"]
    assert teacher["clipping_element_rate"] == pytest.approx(1.0 / 3.0)
    assert teacher["clipping_step_rate"] == pytest.approx(1.0)


def test_paired_evaluator_rejects_missing_required_strength_scenario() -> None:
    from unilab.algos.torch.fada_context.paired_evaluation import evaluate_paired_rollouts

    env = _FakePairedEnv()
    env.state.info["privileged_actuator_strength"][1:] = 1.0

    with pytest.raises(ValueError, match="missing actuator-strength scenarios.*left_knee"):
        evaluate_paired_rollouts(env, _FakeResidualActor(), steps=2, device="cpu")


def test_paired_evaluator_rejects_strength_outside_exact_phase1_profile() -> None:
    from unilab.algos.torch.fada_context.paired_evaluation import evaluate_paired_rollouts

    env = _FakePairedEnv()
    env.state.info["privileged_actuator_strength"][1] = 1.0
    env.state.info["privileged_actuator_strength"][1, 0] = 0.9

    with pytest.raises(ValueError, match="outside the exact Phase-1 profile"):
        evaluate_paired_rollouts(env, _FakeResidualActor(), steps=2, device="cpu")


def test_paired_evaluator_rejects_right_knee_or_nonfixed_left_strength() -> None:
    from unilab.algos.torch.fada_context.paired_evaluation import evaluate_paired_rollouts

    for actuator_index, multiplier in ((9, 0.9), (3, 0.85)):
        env = _FakePairedEnv()
        env.state.info["privileged_actuator_strength"][1] = 1.0
        env.state.info["privileged_actuator_strength"][1, actuator_index] = multiplier
        with pytest.raises(ValueError, match="outside the exact Phase-1 profile"):
            evaluate_paired_rollouts(env, _FakeResidualActor(), steps=2, device="cpu")


def test_paired_evaluator_rejects_inexact_snapshot_restore() -> None:
    from unilab.algos.torch.fada_context.paired_evaluation import evaluate_paired_rollouts

    with pytest.raises(ValueError, match="paired branch start mismatch"):
        evaluate_paired_rollouts(
            _FakePairedEnv(corrupt_restore=True),
            _FakeResidualActor(),
            steps=2,
            device="cpu",
        )


def test_paired_report_aggregation_preserves_seed_and_scenario_structure() -> None:
    from unilab.algos.torch.fada_context.paired_evaluation import (
        aggregate_paired_reports,
        evaluate_paired_rollouts,
    )

    first = evaluate_paired_rollouts(
        _FakePairedEnv(),
        _FakeResidualActor(),
        steps=2,
        device="cpu",
    )
    second = evaluate_paired_rollouts(
        _FakePairedEnv(),
        _FakeResidualActor(),
        steps=2,
        device="cpu",
    )
    first["seed"] = 1
    second["seed"] = 2

    aggregate = aggregate_paired_reports([first, second])

    assert aggregate["seed_count"] == 2
    assert aggregate["seeds"] == [1, 2]
    assert aggregate["scenario_counts"] == {"left_knee": 4, "nominal": 2}
    assert aggregate["teacher"]["overall"]["forward_velocity_mae_mps"] == pytest.approx(0.0)
