"""Compatibility facade for :mod:`unilab.algos.torch.distill.fada.privileged_oracle`."""

from .fada.privileged_oracle import *  # noqa: F401,F403
from .fada.privileged_oracle import _SHA256_RE as _SHA256_RE  # noqa: F401
from .fada.privileged_oracle import _canonical_json_sha256 as _canonical_json_sha256  # noqa: F401
from .fada.privileged_oracle import (
    _reject_stand_reward_authority as _reject_stand_reward_authority,  # noqa: F401
)
