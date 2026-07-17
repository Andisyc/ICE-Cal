from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from unilab.algos.torch.distill.persistent_resources import (
    PersistentResourceCache,
    PersistentResourceIdentity,
)


class _FakeTeacher:
    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _FakeEnv:
    def __init__(self, *, num_envs: int) -> None:
        self.num_envs = num_envs
        self.command = np.ones((num_envs, 3), dtype=np.float32)
        self.done = np.ones((num_envs,), dtype=bool)
        self.transition_age = np.full((num_envs,), 99, dtype=np.int64)
        self.close_count = 0

    def reset(self, env_indices):
        self.command[...] = 0.0
        self.done[...] = False
        self.transition_age[...] = -1
        return {"obs": np.zeros((self.num_envs, 2), dtype=np.float32)}, {}

    def step(self, _actions):
        return {"obs": np.zeros((self.num_envs, 2), dtype=np.float32)}, {}, None

    def close(self) -> None:
        self.close_count += 1


def _identity(*, role: str, num_envs: int = 2, teacher_hash: str = "teacher-v1"):
    return PersistentResourceIdentity(
        task_owner=f"g1_{role}/mujoco",
        task_name=f"G1{role.title().replace('_', '')}",
        sim_backend="mujoco",
        env_cfg_fingerprint=f"env-{role}",
        num_envs=num_envs,
        teacher_checkpoint_path=f"/checkpoints/{role}.pt",
        teacher_checkpoint_sha256=teacher_hash,
        teacher_spec_fingerprint=f"teacher-spec-{role}",
    )


def test_persistent_resource_cache_uses_exact_identity_and_resets_every_request() -> None:
    created_teachers: list[_FakeTeacher] = []
    created_envs: list[_FakeEnv] = []

    def teacher_factory(identity: PersistentResourceIdentity):
        teacher = _FakeTeacher(identity.teacher_checkpoint_sha256)
        created_teachers.append(teacher)
        return teacher

    def env_factory(identity: PersistentResourceIdentity):
        env = _FakeEnv(num_envs=identity.num_envs)
        created_envs.append(env)
        return env

    cache = PersistentResourceCache(
        teacher_factory=teacher_factory,
        env_factory=env_factory,
    )
    walk = _identity(role="walk_flat")
    stand = _identity(role="stand")
    observed_first_rows: list[tuple[float, bool, int]] = []

    def collect(env, teacher, reset_output):
        assert isinstance(reset_output, tuple)
        observed_first_rows.append(
            (float(env.command[0, 0]), bool(env.done[0]), int(env.transition_age[0]))
        )
        assert teacher is not None
        env.command[...] = 0.4
        env.done[...] = True
        env.transition_age[...] = 7
        env.step(np.zeros((env.num_envs, 1), dtype=np.float32))
        return teacher.identity

    try:
        assert cache.run_request(walk, collect) == "teacher-v1"
        assert cache.run_request(walk, collect) == "teacher-v1"
        assert cache.run_request(stand, collect) == "teacher-v1"
        assert observed_first_rows == [(0.0, False, -1)] * 3
        assert cache.counters() == {
            "request_count": 3,
            "request_error_count": 0,
            "cache_hit_count": 1,
            "teacher_init_count": 2,
            "env_init_count": 2,
            "reset_count": 3,
            "teacher_close_count": 0,
            "env_close_count": 0,
        }

        changed_num_envs = replace(walk, num_envs=4)
        changed_teacher = replace(walk, teacher_checkpoint_sha256="teacher-v2")
        cache.acquire(changed_num_envs)
        cache.acquire(changed_teacher)
        assert len(cache.cache_keys) == 4
        assert len(set(cache.cache_keys)) == 4
    finally:
        cache.close()

    assert all(teacher.close_count == 1 for teacher in created_teachers)
    assert all(env.close_count == 1 for env in created_envs)
    assert cache.counters()["teacher_close_count"] == 4
    assert cache.counters()["env_close_count"] == 4


def test_persistent_resource_cache_closes_once_after_exception() -> None:
    created_teachers: list[_FakeTeacher] = []
    created_envs: list[_FakeEnv] = []
    cache = PersistentResourceCache(
        teacher_factory=lambda identity: (
            created_teachers.append(_FakeTeacher(identity.teacher_checkpoint_sha256))
            or created_teachers[-1]
        ),
        env_factory=lambda identity: (
            created_envs.append(_FakeEnv(num_envs=identity.num_envs)) or created_envs[-1]
        ),
    )

    def fail_after_reset(_env, _teacher, _reset_output):
        raise RuntimeError("collector failed")

    with pytest.raises(RuntimeError, match="collector failed"):
        cache.run_request(_identity(role="walk_flat"), fail_after_reset)
    assert cache.counters()["request_error_count"] == 1

    cache.close()
    cache.close()

    assert created_teachers[0].close_count == 1
    assert created_envs[0].close_count == 1
