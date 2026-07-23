from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pytest

from unilab.training import g1_walk_height_acceptance as acceptance


def _write_identity(tmp_path: Path, *, task_name: str = "G1WalkHeight"):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model_5000.pt"
    checkpoint.write_bytes(b"synthetic-walk-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = {
        "run": {"effective_seed": 7},
        "config": {
            "training": {"task_name": task_name, "sim_backend": "mujoco"},
            "algo": {
                "algo": "sac",
                "seed": 1,
                "actor_hidden_dim": 512,
                "use_layer_norm": True,
                "obs_normalization": True,
            },
            "env": {
                "commands": {
                    "height_range": [0.754, 0.754],
                    "default_height": 0.754,
                    "random_height_during_walking": False,
                    "observe_height_command": True,
                }
            },
            "reward": {"max_tilt_deg": 65.0},
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir, checkpoint, digest


def _samples(*, measured_velocity=(0.45, 0.0, 0.0)):
    samples = acceptance.WalkRolloutSamples()
    command = np.asarray([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    samples.append(
        command=command,
        measured_linvel=np.asarray([measured_velocity, measured_velocity], dtype=np.float32),
        measured_gyro=np.zeros((2, 3), dtype=np.float32),
        target_height=np.full(2, 0.754, dtype=np.float32),
        measured_height=np.asarray([0.74, 0.75], dtype=np.float32),
        tilt_deg=np.asarray([2.0, 3.0], dtype=np.float32),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.zeros(2, dtype=bool),
        command_obs=command,
        target_height_obs=np.full(2, 0.754, dtype=np.float32),
        actions=np.zeros((2, 29), dtype=np.float32),
        scored=True,
    )
    return samples


def test_walk_quality_evaluator_passes_and_fails_closed() -> None:
    kwargs = {
        "expected_command": (0.5, 0.0, 0.0),
        "expected_target_height": 0.754,
        "max_linear_velocity_error": 0.25,
        "max_yaw_velocity_error": 0.35,
        "max_height_mae": 0.08,
        "max_tilt_deg": 65.0,
        "requested_steps": 1,
        "executed_steps": 1,
    }
    passed = acceptance.evaluate_samples(_samples(), **kwargs)
    failed = acceptance.evaluate_samples(_samples(measured_velocity=(0.0, 0.0, 0.0)), **kwargs)

    assert passed["verdict"] == "PASS"
    assert passed["metrics"]["linear_velocity_error"] == pytest.approx(0.05)
    assert failed["verdict"] == "FAIL"
    assert "quality/linear_velocity_error" in {
        item["name"] for item in failed["checks"] if item["level"] == "FAIL"
    }


def test_walk_identity_is_hash_and_task_strict(tmp_path: Path) -> None:
    run_dir, checkpoint, digest = _write_identity(tmp_path)
    identity = acceptance.load_run_identity(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        expected_target_height=0.754,
    )
    assert identity.checkpoint_sha256 == digest
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        acceptance.load_run_identity(
            run_dir=run_dir,
            checkpoint_path=checkpoint,
            expected_sha256="0" * 64,
            expected_target_height=0.754,
        )
    bad_run, bad_checkpoint, bad_digest = _write_identity(
        tmp_path / "bad", task_name="G1StandHeight"
    )
    with pytest.raises(ValueError, match="G1WalkHeight"):
        acceptance.load_run_identity(
            run_dir=bad_run,
            checkpoint_path=bad_checkpoint,
            expected_sha256=bad_digest,
            expected_target_height=0.754,
        )


def test_walk_probe_fixes_command_and_connects_runtime_metrics(tmp_path: Path) -> None:
    run_dir, checkpoint, digest = _write_identity(tmp_path)
    identity = acceptance.load_run_identity(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        expected_target_height=0.754,
    )
    captured = {}

    class Backend:
        def get_sensor_data(self, name):
            assert name == "upvector"
            return np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)

    class Env:
        num_envs = 1
        obs_groups_spec = {"obs": 99, "critic": 102}
        action_space = types.SimpleNamespace(shape=(29,))
        cfg = types.SimpleNamespace(sensor=types.SimpleNamespace(upvector="upvector"))
        _backend = Backend()

        def __init__(self, command):
            obs = np.zeros((1, 99), dtype=np.float32)
            obs[:, 93:96] = command
            obs[:, 96] = 0.754
            self.state = types.SimpleNamespace(
                obs={"obs": obs, "critic": np.zeros((1, 102), dtype=np.float32)},
                info={
                    "commands": np.asarray([command], dtype=np.float32),
                    "height_commands": np.asarray([[0.754]], dtype=np.float32),
                    "steps": np.zeros(1, dtype=np.uint32),
                },
                terminated=np.zeros(1, dtype=bool),
                truncated=np.zeros(1, dtype=bool),
            )

        def set_autoreset(self, enabled):
            assert enabled is False

        def init_state(self):
            return self.state

        def step(self, actions):
            self.state.info["steps"] += 1
            return self.state

        def get_local_linvel(self):
            return np.asarray([[0.45, 0.0, 0.0]], dtype=np.float32)

        def get_gyro(self):
            return np.zeros((1, 3), dtype=np.float32)

        def _terrain_relative_base_height(self):
            return np.asarray([0.75], dtype=np.float32)

        def close(self):
            captured["closed"] = True

    def create_env_fn(cfg, **kwargs):
        captured["vel_limit"] = [list(row) for row in cfg.env.commands.vel_limit]
        return Env(tuple(cfg.env.commands.vel_limit[0]))

    report = acceptance.run_probe(
        identity=identity,
        probe_name="forward_nominal",
        command=(0.5, 0.0, 0.0),
        expected_target_height=0.754,
        num_envs=1,
        warmup_steps=0,
        evaluation_steps=1,
        seed=7,
        device="cpu",
        max_linear_velocity_error=0.25,
        max_yaw_velocity_error=0.35,
        max_height_mae=0.08,
        create_env_fn=create_env_fn,
        load_policy_fn=lambda *args, **kwargs: (
            lambda obs: np.zeros((obs.shape[0], 29), dtype=np.float32)
        ),
        ensure_registries_fn=lambda: None,
    )
    assert captured == {
        "vel_limit": [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "closed": True,
    }
    assert report["verdict"] == "PASS"


def test_walk_acceptance_requires_every_probe(tmp_path: Path) -> None:
    run_dir, checkpoint, digest = _write_identity(tmp_path)
    seen = []

    def run_probe_fn(**kwargs):
        name = kwargs["probe_name"]
        seen.append(name)
        passed = name != "lateral"
        return {
            "verdict": "PASS" if passed else "FAIL",
            "checks": [
                {
                    "level": "PASS" if passed else "FAIL",
                    "name": "quality/linear_velocity_error",
                    "detail": name,
                }
            ],
        }

    report = acceptance.run_acceptance(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        expected_target_height=0.754,
        num_envs=1,
        warmup_steps=0,
        evaluation_steps=1,
        seed=7,
        device="cpu",
        run_probe_fn=run_probe_fn,
    )
    assert seen == ["forward_slow", "forward_nominal", "lateral", "yaw"]
    assert report["verdict"] == "FAIL"
    assert any(
        item["name"] == "probe/lateral/quality/linear_velocity_error" and item["level"] == "FAIL"
        for item in report["checks"]
    )
