"""Vanilla terminal payoffs in log-price coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def call_payoff_log(x: ArrayLike, strike: float) -> NDArray[np.float64]:
    """Return the undiscounted terminal call payoff ``max(exp(x) - K, 0)``."""

    _validate_positive_strike(strike)
    x_arr = _validate_log_price(x)
    return np.maximum(np.exp(x_arr) - strike, 0.0)


def put_payoff_log(x: ArrayLike, strike: float) -> NDArray[np.float64]:
    """Return the undiscounted terminal put payoff ``max(K - exp(x), 0)``."""

    _validate_positive_strike(strike)
    x_arr = _validate_log_price(x)
    return np.maximum(strike - np.exp(x_arr), 0.0)


def _validate_positive_strike(strike: float) -> None:
    if not np.isfinite(strike) or strike <= 0:
        raise ValueError(f"strike must be positive and finite, got {strike!r}.")


def _validate_log_price(x: ArrayLike) -> NDArray[np.float64]:
    x_arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must contain only finite log-prices.")
    return x_arr
