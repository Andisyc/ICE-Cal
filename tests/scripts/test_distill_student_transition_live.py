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
        assert name == "upvector"
        tilt_rad = np.deg2rad(self._env.tilt_deg)
        return np.asarray([[np.sin(tilt_rad), 0.0, np.cos(tilt_rad)]], dtype=np.float32)


class _FakeEnv:
    def __init__(self) -> None:
        self.cfg = types.SimpleNamespace(sensor=types.SimpleNamespace(upvector="upvector"))
        self._backend = _FakeBackend(self)
        self.state: Any | None = None
        self.height = 0.75
        self.tilt_deg = 0.0
        self.speed = 0.0
        self.init_state_calls = 0
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
            },
            final_observation=None,
        )
        return self.state

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
        self.action_obs: Any | None = None
        self.step_count = 0
        self._terminal_emitted = False

    def refresh_observation(self) -> np.ndarray:
        assert self.env.state is not None
        return self.env.state.obs["obs"]

    def set_external_command(self, command: np.ndarray) -> np.ndarray:
        assert self.env.state is not None
        self.command = np.asarray(command, dtype=np.float32).copy()
        self.env.state.info["commands"][:, :3] = self.command
        return self.refresh_observation()

    def step_once(self) -> np.ndarray:
        assert self.env.state is not None
        self.actions = np.ones((1, 29), dtype=np.float32)
        self.env.state.info["steps"] += 1
        self.step_count += 1
        active = bool(np.max(np.abs(self.command)) > 0.0)
        self.env.speed = 0.2 if active else 0.1
        self.env.state.terminated.fill(False)
        self.env.state.truncated.fill(False)
        if active and self.terminal_kind is not None and not self._terminal_emitted:
            self._terminal_emitted = True
            if self.terminal_kind == "terminated":
                self.env.height = 0.2
                self.env.state.terminated[:] = True
            else:
                self.env.state.truncated[:] = True
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
    )

    assert seed_calls == [(7, {"torch_runtime": True, "cuda": False})]
    assert env.init_state_calls == 17
    assert env.autoreset is False
    assert env.closed is True
    assert report["seed"] == 7
    assert report["summary"]["total_done_count"] == 0
    assert report["summary"]["total_terminated_count"] == 0
    assert report["summary"]["total_truncated_count"] == 0
    assert report["summary"]["completed_phase_count"] == 51
    assert report["summary"]["gate_pass"] is True
    assert {episode["reset_step_count"] for episode in report["episodes"]} == {0}


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
    assert terminal["height"] == pytest.approx(0.2 if terminal_kind == "terminated" else 0.75)
    assert report["command_summary"]["lateral"]["episodes"] == 0
    assert report["command_summary"]["lateral"]["min_base_height"] is None
    assert report["command_summary"]["lateral"]["max_tilt_deg"] is None
    assert report["failure_indices"] == [0]
    assert report["termination_indices"] == ([0] if terminal_kind == "terminated" else [])
    assert report["truncation_indices"] == ([0] if terminal_kind == "truncated" else [])
