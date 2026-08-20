"""Compatibility imports for the base-layer NaN/Inf guard.

New code should import :mod:`unilab.base.nan_guard` directly.
"""

from unilab.base.nan_guard import NanGuard, NanGuardCfg

__all__ = ["NanGuard", "NanGuardCfg"]
