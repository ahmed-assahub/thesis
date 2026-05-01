"""Black-Scholes reference prices in log-price coordinates."""

from __future__ import annotations

from math import erf

import numpy as np
from numpy.typing import ArrayLike, NDArray

from option_gpr.payoffs import call_payoff_log


def black_scholes_call_price_log(
    X: ArrayLike,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Return Black-Scholes call prices for rows ``(t, log(S))``."""

    X_arr = _validate_points(X)
    _validate_positive_finite("strike", strike)
    _validate_positive_finite("maturity", maturity)
    _validate_finite("r", r)
    _validate_positive_finite("sigma", sigma)

    t = X_arr[:, 0]
    x = X_arr[:, 1]
    if np.any(t > maturity):
        raise ValueError("all evaluation times must be less than or equal to maturity.")

    prices = np.empty(X_arr.shape[0], dtype=float)
    at_maturity = np.isclose(t, maturity, atol=1e-14, rtol=0.0)
    prices[at_maturity] = call_payoff_log(x[at_maturity], strike)

    before_maturity = ~at_maturity
    if np.any(before_maturity):
        tau = maturity - t[before_maturity]
        S = np.exp(x[before_maturity])
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / strike) + (r + 0.5 * sigma**2) * tau) / (
            sigma * sqrt_tau
        )
        d2 = d1 - sigma * sqrt_tau
        prices[before_maturity] = S * _normal_cdf(d1) - strike * np.exp(
            -r * tau
        ) * _normal_cdf(d2)

    if not np.all(np.isfinite(prices)):
        raise ValueError("computed prices must be finite.")
    return prices


def _normal_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def _validate_points(X: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"X must have shape (n, 2), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X must contain only finite values.")
    return arr


def _validate_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")


def _validate_positive_finite(name: str, value: float) -> None:
    _validate_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
