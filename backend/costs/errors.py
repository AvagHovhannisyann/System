"""Failure taxonomy for the transaction cost model (P9.3).

Bad cost parameters raise here rather than producing a number. A cost model
that accepts a negative half-spread, or a spread the caller meant as a
fraction and this package reads as basis points, produces a backtest that is
wrong by orders of magnitude and looks entirely normal (directive §8, I4).
"""

from __future__ import annotations

__all__ = [
    "CostCalibrationError",
    "CostModelError",
    "CostParameterError",
]


class CostModelError(Exception):
    """Base class for transaction-cost-model failures."""


class CostParameterError(CostModelError, ValueError):
    """Raised when a cost parameter or order field is not usable.

    Covers non-finite values, negative costs, negative sizes, and a
    non-positive average daily volume (which would make the participation rate
    undefined). Subclasses :class:`ValueError` so ordinary
    ``pytest.raises(ValueError)`` and caller-side validation keep working.
    """


class CostCalibrationError(CostModelError, ValueError):
    """Raised when a calibrated parameter set would loosen the conservative floor.

    DECISIONS.md **D-013**: IBKR paper fills are optimistic — they fill at the
    touch far more readily than reality and model no queue position — so
    realized paper slippage is a **lower bound** on cost, never an estimate of
    it. A parameter set may therefore only be marked calibrated if every
    component is at least as expensive as the conservative uncalibrated
    default. Cheaper-than-default parameters are refused at construction, so
    optimistic fills cannot silently tighten the model and turn a losing
    strategy into a winning backtest.

    Attributes:
        parameter: name of the offending field.
        value: the value that was supplied.
        floor: the conservative default it fell below.
    """

    def __init__(self, *, parameter: str, value: float, floor: float) -> None:
        """Build the error from the offending parameter, its value and the floor."""
        self.parameter = parameter
        self.value = value
        self.floor = floor
        super().__init__(
            f"calibrated cost parameter {parameter}={value!r} is below the "
            f"conservative uncalibrated default {floor!r}. Per DECISIONS.md D-013 "
            f"paper fills are a lower bound on slippage, never an estimate, so a "
            f"calibration may only make costs more expensive than the default, "
            f"never cheaper. If a cheaper value is genuinely justified, record the "
            f"reasoning as a new decision first — this refusal is the control that "
            f"keeps optimistic fills from silently tightening the model."
        )
