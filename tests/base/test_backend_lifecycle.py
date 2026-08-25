from __future__ import annotations

import pytest

from unilab.base.np_env import NpEnv


class _CloseRecordingBackend:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _ClosableNpEnv(NpEnv):
    @property
    def action_space(self):
        return None

    def apply_action(self, actions):
        return None

    def update_state(self, state):
        return state


def test_np_env_close_delegates_to_backend_owner() -> None:
    env = object.__new__(_ClosableNpEnv)
    backend = _CloseRecordingBackend()
    env._backend = backend

    env.close()

    assert backend.close_count == 1


def test_mujoco_backend_close_releases_pool_once_and_is_idempotent() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    class _Pool:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    class _CleanupHandle:
        def __init__(self) -> None:
            self.cleanup_count = 0

        def cleanup(self) -> None:
            self.cleanup_count += 1

    backend = object.__new__(MuJoCoBackend)
    pool = _Pool()
    cleanup_handle = _CleanupHandle()
    backend._pool = pool
    backend._scene_cleanup_handle = cleanup_handle

    backend.close()
    backend.close()

    assert pool.close_count == 1
    assert cleanup_handle.cleanup_count == 1
    assert backend._pool is None
    assert backend._scene_cleanup_handle is None


def test_mujoco_backend_close_cleans_scene_assets_when_pool_close_fails() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    class _FailingPool:
        def close(self) -> None:
            raise RuntimeError("pool close failed")

    class _CleanupHandle:
        def __init__(self) -> None:
            self.cleanup_count = 0

        def cleanup(self) -> None:
            self.cleanup_count += 1

    backend = object.__new__(MuJoCoBackend)
    cleanup_handle = _CleanupHandle()
    backend._pool = _FailingPool()
    backend._scene_cleanup_handle = cleanup_handle

    with pytest.raises(RuntimeError, match="pool close failed"):
        backend.close()

    assert cleanup_handle.cleanup_count == 1
    assert backend._pool is None
    assert backend._scene_cleanup_handle is None
