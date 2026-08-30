"""Exact-identity resource cache for persistent distillation workers.

状态: HP-3b2 active, fake lifecycle and bounded real G1 lifecycle confirmed.
上游: persistent scenario worker cold-path resource resolution.
下游: cached teacher/env bundle and per-request reset boundary.
证据: S1 lifecycle, identity, reset, exception, and cleanup tests.
边界: cache identity/reset/cleanup are owned here; collection semantics remain
in the collector/data owners.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class PersistentResourceIdentity:
    """Cold-path identity for one compatible teacher/env resource bundle."""

    task_owner: str
    task_name: str
    sim_backend: str
    env_cfg_fingerprint: str
    num_envs: int
    teacher_checkpoint_path: str
    teacher_checkpoint_sha256: str
    teacher_spec_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "task_owner",
            "task_name",
            "sim_backend",
            "env_cfg_fingerprint",
            "teacher_checkpoint_path",
            "teacher_checkpoint_sha256",
            "teacher_spec_fingerprint",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"persistent resource identity {name} must be non-empty")
        if int(self.num_envs) <= 0:
            raise ValueError(f"persistent resource num_envs must be positive, got {self.num_envs}")

    @property
    def cache_key(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def as_metadata(self) -> dict[str, Any]:
        return {**asdict(self), "cache_key": self.cache_key}


@dataclass
class PersistentResourceBundle:
    identity: PersistentResourceIdentity
    teacher: Any
    env: Any


class _ResetAuditedEnv:
    """Per-request proxy proving reset precedes any environment step."""

    def __init__(self, env: Any, *, on_reset: Callable[[], None]) -> None:
        self._env = env
        self._on_reset = on_reset
        self._reset_seen = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, env_indices: Any):
        result = self._env.reset(env_indices)
        self._reset_seen = True
        self._on_reset()
        return result

    def step(self, actions: Any):
        if not self._reset_seen:
            raise AssertionError("persistent resource request must reset before step")
        return self._env.step(actions)


class PersistentResourceCache:
    """Own exact-key initialization, request reset, counters, and cleanup."""

    def __init__(
        self,
        *,
        teacher_factory: Callable[[PersistentResourceIdentity], Any],
        env_factory: Callable[[PersistentResourceIdentity], Any],
    ) -> None:
        self._teacher_factory = teacher_factory
        self._env_factory = env_factory
        self._bundles: dict[str, PersistentResourceBundle] = {}
        self._counters: dict[str, int] = {
            "request_count": 0,
            "request_error_count": 0,
            "cache_hit_count": 0,
            "teacher_init_count": 0,
            "env_init_count": 0,
            "reset_count": 0,
            "teacher_close_count": 0,
            "env_close_count": 0,
        }
        self._closed = False

    @property
    def cache_keys(self) -> tuple[str, ...]:
        return tuple(self._bundles)

    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    @staticmethod
    def _close_resource(resource: Any) -> bool:
        close = getattr(resource, "close", None)
        if callable(close):
            close()
            return True
        return False

    def acquire(self, identity: PersistentResourceIdentity) -> PersistentResourceBundle:
        if self._closed:
            raise RuntimeError("cannot acquire from a closed persistent resource cache")
        key = identity.cache_key
        existing = self._bundles.get(key)
        if existing is not None:
            if existing.identity != identity:
                raise RuntimeError("persistent resource cache key collision")
            self._counters["cache_hit_count"] += 1
            return existing

        teacher = self._teacher_factory(identity)
        self._counters["teacher_init_count"] += 1
        try:
            env = self._env_factory(identity)
        except Exception:
            if self._close_resource(teacher):
                self._counters["teacher_close_count"] += 1
            raise
        self._counters["env_init_count"] += 1
        bundle = PersistentResourceBundle(identity=identity, teacher=teacher, env=env)
        self._bundles[key] = bundle
        return bundle

    def run_request(
        self,
        identity: PersistentResourceIdentity,
        collect: Callable[[Any, Any, Any], Any],
    ) -> Any:
        """Reset one cached env, then execute one semantic collection request."""

        self._counters["request_count"] += 1
        try:
            bundle = self.acquire(identity)
            audited_env = _ResetAuditedEnv(
                bundle.env,
                on_reset=lambda: self._counters.__setitem__(
                    "reset_count", self._counters["reset_count"] + 1
                ),
            )
            env_indices = np.arange(int(identity.num_envs), dtype=np.int32)
            reset_output = audited_env.reset(env_indices)
            return collect(audited_env, bundle.teacher, reset_output)
        except Exception:
            self._counters["request_error_count"] += 1
            raise

    def metadata(self) -> dict[str, Any]:
        return {
            "cache_keys": list(self.cache_keys),
            "resources": [bundle.identity.as_metadata() for bundle in self._bundles.values()],
            "counters": self.counters(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for bundle in self._bundles.values():
            if self._close_resource(bundle.env):
                self._counters["env_close_count"] += 1
            if self._close_resource(bundle.teacher):
                self._counters["teacher_close_count"] += 1
