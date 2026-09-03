from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.locomotion.g1.walk_runtime_bindings import (
    publish_walk_termination_provenance,
)


def test_walk_runtime_publishes_owned_termination_provenance() -> None:
    info: dict[str, object] = {}
    fall = np.asarray([True, False])
    forward = np.asarray([False, True])

    publish_walk_termination_provenance(
        info,
        fall_terminated=fall,
        forward_progress_terminated=forward,
    )
    fall[:] = False
    forward[:] = False

    np.testing.assert_array_equal(info["fall_terminated"], [True, False])
    np.testing.assert_array_equal(info["forward_progress_terminated"], [False, True])


def test_walk_runtime_rejects_misaligned_termination_provenance() -> None:
    with pytest.raises(ValueError, match="matching rank-1"):
        publish_walk_termination_provenance(
            {},
            fall_terminated=np.zeros((2,), dtype=np.bool_),
            forward_progress_terminated=np.zeros((1,), dtype=np.bool_),
        )
