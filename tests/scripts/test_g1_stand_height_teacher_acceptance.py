from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pytest

from unilab.training import g1_stand_height_acceptance as acceptance


def _write_run_identity(
    tmp_path: Path, *, task_name: str = "G1StandHeight"
) -> tuple[Path, Path, str]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model_5000.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = {
        "run": {"effective_seed": 7},
        "config": {
            "training": {
                "task_name": task_name,
                "sim_backend": "mujoco",
                "log_dir": str(run_dir),
            },
            "algo": {
                "algo": "sac",
                "actor_hidden_dim": 512,
                "use_layer_norm": True,
                "obs_normalization": False,
            },
            "env": {
                "commands": {
                    "height_range": [0.754, 0.754],
                    "default_height": 0.754,
                }
            },
            "reward": {"max_tilt_deg": 65.0},
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir, checkpoint, digest


def test_evaluate_samples_applies_teacher_quality_thresholds() -> None:
    samples = acceptance.RolloutSamples()
    samples.append(
        target_height=np.asarray([0.754, 0.754], dtype=np.float32),
        measured_height=np.asarray([0.744, 0.764], dtype=np.float32),
        double_support=np.asarray([True, True]),
        tilt_deg=np.asarray([1.0, 2.0], dtype=np.float32),
        terminated=np.asarray([False, False]),
        truncated=np.asarray([False, False]),
        commands=np.zeros((2, 3), dtype=np.float32),
        target_obs=np.asarray([0.754, 0.754], dtype=np.float32),
        actions=np.zeros((2, 29), dtype=np.float32),
        scored=True,
    )

    report = acceptance.evaluate_samples(
        samples,
        expected_target_height=0.754,
        max_height_mae=0.05,
        min_double_support_fraction=0.90,
        max_tilt_deg=65.0,
        requested_steps=1,
        executed_steps=1,
    )

    assert report["metrics"]["height_mae"] == pytest.approx(0.01)
    assert report["metrics"]["double_support_fraction"] == pytest.approx(1.0)
    assert report["verdict"] == "PASS"
    assert all(check["level"] == "PASS" for check in report["checks"])


def test_evaluate_samples_fails_closed_on_physical_quality() -> None:
    samples = acceptance.RolloutSamples()
    samples.append(
        target_height=np.asarray([0.754, 0.754], dtype=np.float32),
        measured_height=np.asarray([0.674, 0.674], dtype=np.float32),
        double_support=np.asarray([True, False]),
        tilt_deg=np.asarray([2.0, 66.0], dtype=np.float32),
        terminated=np.asarray([False, True]),
        truncated=np.asarray([False, False]),
        commands=np.zeros((2, 3), dtype=np.float32),
        target_obs=np.asarray([0.754, 0.70], dtype=np.float32),
        actions=np.zeros((2, 29), dtype=np.float32),
        scored=True,
    )

    report = acceptance.evaluate_samples(
        samples,
        expected_target_height=0.754,
        max_height_mae=0.05,
        min_double_support_fraction=0.90,
        max_tilt_deg=65.0,
        requested_steps=2,
        executed_steps=1,
    )

    failed = {check["name"] for check in report["checks"] if check["level"] == "FAIL"}
    assert report["verdict"] == "FAIL"
    assert {
        "rollout/completed_window",
        "rollout/no_termination",
        "rollout/target_obs_roundtrip",
        "quality/height_mae",
        "quality/double_support_fraction",
        "quality/tilt_below_limit",
    } <= failed


def test_load_run_identity_rejects_hash_and_contract_mismatch(tmp_path: Path) -> None:
    run_dir, checkpoint, digest = _write_run_identity(tmp_path)

    identity = acceptance.load_run_identity(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        expected_target_height=0.754,
    )
    assert identity.checkpoint_sha256 == digest
    assert identity.task_name == "G1StandHeight"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        acceptance.load_run_identity(
            run_dir=run_dir,
            checkpoint_path=checkpoint,
            expected_sha256="0" * 64,
            expected_target_height=0.754,
        )

    bad_run_dir, bad_checkpoint, bad_digest = _write_run_identity(
        tmp_path / "bad", task_name="G1WalkFlat"
    )
    with pytest.raises(ValueError, match="G1StandHeight"):
        acceptance.load_run_identity(
            run_dir=bad_run_dir,
            checkpoint_path=bad_checkpoint,
            expected_sha256=bad_digest,
            expected_target_height=0.754,
        )


def test_run_acceptance_connects_identity_policy_env_and_metrics(tmp_path: Path) -> None:
    run_dir, checkpoint, digest = _write_run_identity(tmp_path)
    state = types.SimpleNamespace(
        obs={
            "obs": np.pad(
                np.full((2, 1), 0.754, dtype=np.float32),
                ((0, 0), (96, 2)),
            ),
            "critic": np.zeros((2, 102), dtype=np.float32),
        },
        info={
            "commands": np.zeros((2, 3), dtype=np.float32),
            "height_commands": np.full((2, 1), 0.754, dtype=np.float32),
            "steps": np.zeros(2, dtype=np.uint32),
        },
        reward=np.zeros(2, dtype=np.float32),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.zeros(2, dtype=bool),
    )

    class FakeBackend:
        def get_sensor_data(self, name: str) -> np.ndarray:
            if name == "upvector":
                return np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            if name.startswith(("left_foot_contact_", "right_foot_contact_")):
                return np.ones(2, dtype=np.float32)
            raise KeyError(name)

    class FakeEnv:
        num_envs = 2
        obs_groups_spec = {"obs": 99, "critic": 102}
        action_space = types.SimpleNamespace(shape=(29,))
        cfg = types.SimpleNamespace(sensor=types.SimpleNamespace(upvector="upvector"))
        _backend = FakeBackend()
        closed = False

        def init_state(self):
            return state

        def set_autoreset(self, enabled: bool) -> None:
            assert enabled is False

        def step(self, actions: np.ndarray):
            assert actions.shape == (2, 29)
            state.info["steps"] += 1
            return state

        def _terrain_relative_base_height(self) -> np.ndarray:
            return np.asarray([0.75, 0.758], dtype=np.float32)

        def close(self) -> None:
            self.closed = True

    fake_env = FakeEnv()

    def create_env_fn(*args, **kwargs):
        assert kwargs["num_envs"] == 2
        assert kwargs["sim_backend"] == "mujoco"
        return fake_env

    def load_policy_fn(*args, **kwargs):
        assert Path(args[0]).resolve() == checkpoint.resolve()
        assert kwargs["obs_dim"] == 99
        assert kwargs["action_dim"] == 29
        return lambda obs: np.zeros((obs.shape[0], 29), dtype=np.float32)

    report = acceptance.run_acceptance(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        expected_target_height=0.754,
        num_envs=2,
        warmup_steps=1,
        evaluation_steps=2,
        seed=7,
        device="cpu",
        create_env_fn=create_env_fn,
        load_policy_fn=load_policy_fn,
        ensure_registries_fn=lambda: None,
    )

    assert report["verdict"] == "PASS"
    assert report["identity"]["checkpoint_sha256"] == digest
    assert report["rollout"]["executed_steps"] == 3
    assert report["metrics"]["scored_sample_count"] == 4
    assert fake_env.closed is True
