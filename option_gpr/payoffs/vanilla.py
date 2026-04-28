"""Vanilla terminal payoffs in log-price coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def call_payoff_log(x: ArrayLike, strike: float) -> NDArray[np.float64]:
    """Return the undiscounted terminal call payoff ``max(exp(x) - K, 0)``."""

    _validate_positive_strike(strike)
    return np.maximum(np.exp(np.asarray(x, dtype=float)) - strike, 0.0)


def put_payoff_log(x: ArrayLike, strike: float) -> NDArray[np.float64]:
    """Return the undiscounted terminal put payoff ``max(K - exp(x), 0)``."""

    _validate_positive_strike(strike)
    return np.maximum(strike - np.exp(np.asarray(x, dtype=float)), 0.0)


def _validate_positive_strike(strike: float) -> None:
    if not np.isfinite(strike) or strike <= 0:
        raise ValueError(f"strike must be positive and finite, got {strike!r}.")
