from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "deploy"
    / "check_unilab_g1_distill_student_transition_live.py"
)


def _load_script() -> Any:
    name = "check_unilab_g1_distill_student_transition_live_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class _FakeBackend:
    def __init__(self, env: "_FakeEnv") -> None:
        self._env = env

    def get_base_pos(self) -> np.ndarray:
        return np.asarray([[0.0, 0.0, self._env.height]], dtype=np.float32)

    def get_base_lin_vel(self) -> np.ndarray:
        return np.asarray([[self._env.speed, 0.0, 0.0]], dtype=np.float32)

    def get_sensor_data(self, name: str) -> np.ndarray:
        if name == "upvector":
            tilt_rad = np.deg2rad(self._env.tilt_deg)
            return np.asarray([[np.sin(tilt_rad), 0.0, np.cos(tilt_rad)]], dtype=np.float32)
        if name.startswith("left_foot_contact_"):
            return np.ones((1,), dtype=np.float32)
        if name.startswith("right_foot_contact_"):
            return np.full((1,), float(self._env.double_support), dtype=np.float32)
        raise AssertionError(f"unexpected sensor {name!r}")


class _FakeEnv:
    def __init__(
        self,
        *,
        recovery_height_error: float = 0.0,
        double_support: bool = True,
        target_obs_offset: float = 0.0,
    ) -> None:
        self.cfg = types.SimpleNamespace(sensor=types.SimpleNamespace(upvector="upvector"))
        self._backend = _FakeBackend(self)
        self.num_envs = 1
        self.state: Any | None = None
        self.height = 0.75
        self.tilt_deg = 0.0
        self.speed = 0.0
        self.recovery_height_error = float(recovery_height_error)
        self.double_support = bool(double_support)
        self.target_obs_offset = float(target_obs_offset)
        self.init_state_calls = 0
        self.refresh_state_calls = 0
        self.autoreset: bool | None = None
        self.closed = False

    def init_state(self) -> Any:
        self.init_state_calls += 1
        self.height = 0.75
        self.tilt_deg = 0.0
        self.speed = 0.0
        self.state = types.SimpleNamespace(
            obs={"obs": np.zeros((1, 99), dtype=np.float32)},
            reward=np.zeros(1, dtype=np.float32),
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
            info={
                "steps": np.zeros(1, dtype=np.uint32),
                "commands": np.zeros((1, 3), dtype=np.float32),
                "height_commands": np.full((1, 1), 0.754, dtype=np.float32),
            },
            final_observation=None,
        )
        self.refresh_state()
        return self.state

    def refresh_state(self) -> Any:
        assert self.state is not None
        self.refresh_state_calls += 1
        actor_obs = self.state.obs["obs"]
        actor_obs[:, 93:96] = self.state.info["commands"][:, :3]
        actor_obs[:, 96] = self.state.info["height_commands"][:, 0] + self.target_obs_offset
        return self.state

    def _terrain_relative_base_height(self) -> np.ndarray:
        return np.asarray([self.height], dtype=np.float32)

    def reset(self, _env_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        raise AssertionError("the sentinel must use a full state reset")

    def set_autoreset(self, enabled: bool) -> None:
        self.autoreset = bool(enabled)

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, env: _FakeEnv, *, terminal_kind: str | None = None) -> None:
        self.env = env
        self.terminal_kind = terminal_kind
        self.command = np.zeros(3, dtype=np.float32)
        self.actions = np.ones((1, 29), dtype=np.float32)
        self.obs: np.ndarray | None = None
        self.action_obs: Any | None = None
        self.step_count = 0
        self._terminal_emitted = False
        self.policy_inputs: list[dict[str, Any]] = []

    def refresh_observation(self) -> np.ndarray:
        assert self.env.state is not None
        self.obs = self.env.state.obs["obs"]
        return self.obs

    def set_external_command(self, command: np.ndarray) -> np.ndarray:
        assert self.env.state is not None
        self.command = np.asarray(command, dtype=np.float32).copy()
        command_rows = np.broadcast_to(
            self.command,
            self.env.state.info["commands"][:, :3].shape,
        )
        if np.array_equal(self.env.state.info["commands"][:, :3], command_rows):
            return self.refresh_observation()
        self.env.state.info["commands"][:, :3] = command_rows
        self.env.refresh_state()
        return self.refresh_observation()

    def step_once(self) -> np.ndarray:
        assert self.env.state is not None
        self.command = self.env.state.info["commands"][0, :3].copy()
        target_height = float(self.env.state.info["height_commands"][0, 0])
        self.policy_inputs.append(
            {
                "reset_id": self.env.init_state_calls,
                "command": self.command.copy(),
                "target_height": target_height,
                "target_obs": float(self.env.state.obs["obs"][0, 96]),
            }
        )
        self.actions = np.ones((1, 29), dtype=np.float32)
        self.env.state.info["steps"] += 1
        self.step_count += 1
        active = bool(np.max(np.abs(self.command)) > 0.0)
        self.env.speed = 0.2 if active else 0.1
        if not active:
            self.env.height = target_height + self.env.recovery_height_error
        self.env.state.terminated.fill(False)
        self.env.state.truncated.fill(False)
        if active and self.terminal_kind is not None and not self._terminal_emitted:
            self._terminal_emitted = True
            if self.terminal_kind == "terminated":
                self.env.height = 0.2
                self.env.state.terminated[:] = True
            else:
                self.env.state.truncated[:] = True
        self.env.refresh_state()
        return self.refresh_observation()


def _install_runtime(monkeypatch: pytest.MonkeyPatch, mod: Any, session: _FakeSession) -> list[Any]:
    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(
            task_name="G1WalkHeight",
            log_root=None,
        ),
        algo=types.SimpleNamespace(algo_log_name="distill"),
        reward=types.SimpleNamespace(min_base_height=0.3, max_tilt_deg=65.0),
    )
    seed_calls: list[Any] = []

    def apply_seed(seed: int, **kwargs: Any) -> int:
        seed_calls.append((seed, kwargs))
        return seed

    monkeypatch.setattr(mod, "_compose_cfg", lambda _task: cfg)
    monkeypatch.setattr(mod, "apply_training_seed", apply_seed)
    monkeypatch.setattr(
        mod,
        "create_distill_playback_session",
        lambda **_kwargs: (session, "actor", "/tmp/student.pt"),
    )
    return seed_calls


def test_transition_sentinel_reinitializes_every_episode_and_applies_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_script()
    env = _FakeEnv()
    session = _FakeSession(env)
    seed_calls = _install_runtime(monkeypatch, mod, session)

    report = mod.run_check(
        student_checkpoint=Path("student.pt"),
        task="g1_walk_height_nominal/mujoco",
        repeats=17,
        active_steps=1,
        stop_steps=1,
        device="cpu",
        seed=7,
        height_recovery_nominal_settle_steps=1,
        height_recovery_warmup_steps=0,
        height_recovery_evaluation_steps=1,
    )

    assert seed_calls == [(7, {"torch_runtime": True, "cuda": False})]
    assert env.init_state_calls == 26
    assert env.autoreset is False
    assert env.closed is True
    assert report["seed"] == 7
    assert report["summary"]["total_done_count"] == 0
    assert report["summary"]["total_terminated_count"] == 0
    assert report["summary"]["total_truncated_count"] == 0
    assert report["summary"]["completed_phase_count"] == 51
    assert report["summary"]["gate_pass"] is True
    assert {episode["reset_step_count"] for episode in report["episodes"]} == {0}


def test_controlled_phase_refreshes_state_only_at_the_input_boundary() -> None:
    mod = _load_script()
    env = _FakeEnv()
    session = _FakeSession(env)
    env.init_state()
    session.refresh_observation()
    initial_refresh_calls = env.refresh_state_calls

    phase = mod._run_phase(
        session,
        4,
        command=np.asarray([0.4, 0.0, 0.0], dtype=np.float32),
        target_height=0.65,
    )

    assert env.refresh_state_calls - initial_refresh_calls == 5
    assert phase["input_sync"]["sync_count"] == 5
    assert phase["input_sync"]["passed"] is True
    assert all(row["command"][0] == pytest.approx(0.4) for row in session.policy_inputs)
    assert [row["target_height"] for row in session.policy_inputs] == pytest.approx([0.65] * 4)


@pytest.mark.parametrize("terminal_kind", ["terminated", "truncated"])
def test_transition_sentinel_preserves_terminal_frame_and_skips_coupled_stop_score(
    monkeypatch: pytest.MonkeyPatch,
    terminal_kind: str,
) -> None:
    mod = _load_script()
    env = _FakeEnv()
    session = _FakeSession(env, terminal_kind=terminal_kind)
    _install_runtime(monkeypatch, mod, session)

    report = mod.run_check(
        student_checkpoint=Path("student.pt"),
        task="g1_walk_height_nominal/mujoco",
        repeats=1,
        active_steps=2,
        stop_steps=1,
        device="cpu",
        post_walk_target_heights=(0.65,),
        height_recovery_nominal_settle_steps=1,
        height_recovery_warmup_steps=0,
        height_recovery_evaluation_steps=1,
    )

    summary = report["summary"]
    episode = report["episodes"][0]
    terminal = report["terminal_events"][0]
    assert summary["total_done_count"] == 1
    assert report["seed"] == 1
    assert summary["total_terminated_count"] == int(terminal_kind == "terminated")
    assert summary["total_truncated_count"] == int(terminal_kind == "truncated")
    assert episode["walking"]["executed_steps"] == 1
    assert episode["stop"]["executed_steps"] == 0
    assert episode["stop"]["skip_reason"] == "walking_done"
    assert episode["stop_speed_le_active"] is False
    assert terminal["phase"] == "walking"
    assert terminal["terminated"] is (terminal_kind == "terminated")
    assert terminal["truncated"] is (terminal_kind == "truncated")
    assert terminal["state_step"] == 2
    assert terminal["height"] == pytest.approx(0.2 if terminal_kind == "terminated" else 0.754)
    assert report["command_summary"]["lateral"]["episodes"] == 0
    assert report["command_summary"]["lateral"]["min_base_height"] is None
    assert report["command_summary"]["lateral"]["max_tilt_deg"] is None
    assert report["failure_indices"] == [0]
    assert report["termination_indices"] == ([0] if terminal_kind == "terminated" else [])
    assert report["truncation_indices"] == ([0] if terminal_kind == "truncated" else [])


def test_transition_sentinel_covers_non_nominal_walk_to_stand_height_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_script()
    env = _FakeEnv()
    session = _FakeSession(env)
    _install_runtime(monkeypatch, mod, session)

    targets = (0.65, 0.702, 0.754)
    report = mod.run_check(
        student_checkpoint=Path("student.pt"),
        task="g1_walk_height/mujoco",
        repeats=1,
        active_steps=2,
        stop_steps=1,
        device="cpu",
        post_walk_target_heights=targets,
        height_recovery_nominal_settle_steps=2,
        height_recovery_warmup_steps=1,
        height_recovery_evaluation_steps=2,
    )

    recovery = report["height_recovery"]
    expected_grid = [
        (command_name, target_height)
        for command_name in ("forward", "lateral", "yaw")
        for target_height in targets
    ]
    assert [
        (scenario["command"], scenario["requested_target_height"])
        for scenario in recovery["scenarios"]
    ] == expected_grid
    assert recovery["verdict"] == "PASS"
    assert report["summary"]["height_recovery_gate_pass"] is True
    assert report["summary"]["gate_pass"] is True

    first_recovery_reset_id = 2
    for offset, ((command_name, target_height), scenario) in enumerate(
        zip(expected_grid, recovery["scenarios"], strict=True)
    ):
        rows = [
            row
            for row in session.policy_inputs
            if row["reset_id"] == first_recovery_reset_id + offset
        ]
        assert len(rows) == 8
        assert np.allclose(rows[0]["command"], 0.0)
        assert rows[0]["target_height"] == pytest.approx(0.754)
        assert all(np.max(np.abs(row["command"])) > 0.0 for row in rows[1:3])
        assert all(row["target_height"] == pytest.approx(0.754) for row in rows[1:3])
        assert all(np.allclose(row["command"], 0.0) for row in rows[3:])
        assert all(row["target_height"] == pytest.approx(0.754) for row in rows[3:5])
        assert all(row["target_height"] == pytest.approx(target_height) for row in rows[5:])
        assert scenario["command"] == command_name
        assert scenario["walking_target_height"] == pytest.approx(0.754)
        assert scenario["nominal_settle"]["executed_steps"] == 2
        assert scenario["recovery"]["verdict"] == "PASS"
        assert scenario["recovery"]["metrics"]["target_obs_max_error"] <= 1.0e-6


@pytest.mark.parametrize(
    ("env_kwargs", "failed_check"),
    [
        ({"recovery_height_error": 0.06}, "quality/height_mae"),
        ({"double_support": False}, "quality/double_support_fraction"),
        ({"target_obs_offset": 0.01}, "rollout/target_obs_roundtrip"),
    ],
)
def test_transition_sentinel_fails_closed_on_height_recovery_quality_contract(
    monkeypatch: pytest.MonkeyPatch,
    env_kwargs: dict[str, Any],
    failed_check: str,
) -> None:
    mod = _load_script()
    env = _FakeEnv(**env_kwargs)
    session = _FakeSession(env)
    _install_runtime(monkeypatch, mod, session)

    report = mod.run_check(
        student_checkpoint=Path("student.pt"),
        task="g1_walk_height/mujoco",
        repeats=1,
        active_steps=1,
        stop_steps=1,
        device="cpu",
        post_walk_target_heights=(0.65,),
        height_recovery_nominal_settle_steps=1,
        height_recovery_warmup_steps=0,
        height_recovery_evaluation_steps=2,
    )

    scenario = report["height_recovery"]["scenarios"][0]
    failed_names = {
        check["name"] for check in scenario["recovery"]["checks"] if check["level"] == "FAIL"
    }
    transition_levels = {check["name"]: check["level"] for check in scenario["transition_checks"]}
    assert failed_check in failed_names
    assert transition_levels["transition/recovery_completed"] == "PASS"
    expected_sync_level = "FAIL" if failed_check == "rollout/target_obs_roundtrip" else "PASS"
    assert transition_levels["transition/recovery_input_synchronized"] == expected_sync_level
    assert report["height_recovery"]["verdict"] == "FAIL"
    assert report["summary"]["height_recovery_gate_pass"] is False
    assert report["summary"]["gate_pass"] is False
