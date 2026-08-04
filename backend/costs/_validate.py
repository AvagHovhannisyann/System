"""Shared numeric guards for the cost model (P9.3, package-private).

Every public entry point in :mod:`backend.costs` validates before it computes.
A NaN that reaches the cost model propagates into a total, then into a net
return, then into a Sharpe ratio, and the first place it becomes visible is a
chart. These helpers make the refusal happen at the boundary and name the
offending field.
"""

from __future__ import annotations

import math

from backend.costs.errors import CostParameterError

__all__ = [
    "require_finite",
    "require_non_negative",
    "require_positive",
]


def require_finite(name: str, value: float) -> float:
    """Return ``value`` if it is a finite real number, else raise.

    Args:
        name: field name, used in the error message.
        value: the value to check.

    Returns:
        ``value`` unchanged.

    Raises:
        CostParameterError: if ``value`` is NaN or infinite.
    """
    if not math.isfinite(value):
        msg = f"{name} must be a finite number; got {value!r}"
        raise CostParameterError(msg)
    return value


def require_non_negative(name: str, value: float) -> float:
    """Return ``value`` if it is finite and ``>= 0``, else raise.

    Args:
        name: field name, used in the error message.
        value: the value to check.

    Returns:
        ``value`` unchanged.

    Raises:
        CostParameterError: if ``value`` is non-finite or negative. A negative
            cost parameter is a subsidy, and there is no such thing.
    """
    require_finite(name, value)
    if value < 0.0:
        msg = f"{name} must be >= 0; got {value!r}"
        raise CostParameterError(msg)
    return value


def require_positive(name: str, value: float) -> float:
    """Return ``value`` if it is finite and ``> 0``, else raise.

    Args:
        name: field name, used in the error message.
        value: the value to check.

    Returns:
        ``value`` unchanged.

    Raises:
        CostParameterError: if ``value`` is non-finite or not strictly
            positive.
    """
    require_finite(name, value)
    if value <= 0.0:
        msg = f"{name} must be > 0; got {value!r}"
        raise CostParameterError(msg)
    return value
