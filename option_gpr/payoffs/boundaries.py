"""Boundary value helpers for option pricing domains."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from option_gpr.payoffs.vanilla import call_payoff_log


def call_boundary_values_log(
    X_bd: ArrayLike,
    strike: float,
    maturity: float,
    r: float,
    S_min: float,
    S_max: float,
    atol: float = 1e-10,
) -> NDArray[np.float64]:
    """Return Black-Scholes call boundary values for rows ``(t, x)``.

    Terminal rows take priority over spatial boundary rows at corner points.
    """

    X_arr = _validate_boundary_points(X_bd)
    _validate_positive_finite("strike", strike)
    _validate_positive_finite("maturity", maturity)
    _validate_finite("r", r)
    _validate_positive_finite("S_min", S_min)
    _validate_positive_finite("S_max", S_max)
    if S_min >= S_max:
        raise ValueError(f"S_min must be less than S_max, got {S_min!r}, {S_max!r}.")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError(f"atol must be nonnegative and finite, got {atol!r}.")

    t = X_arr[:, 0]
    x = X_arr[:, 1]
    terminal = np.isclose(t, maturity, atol=atol, rtol=0.0)
    lower = np.isclose(x, np.log(S_min), atol=atol, rtol=0.0) & ~terminal
    upper = np.isclose(x, np.log(S_max), atol=atol, rtol=0.0) & ~terminal
    unknown = ~(terminal | lower | upper)
    if np.any(unknown):
        raise ValueError("All boundary rows must be terminal, lower, or upper boundary points.")

    values = np.empty(X_arr.shape[0], dtype=float)
    values[terminal] = call_payoff_log(x[terminal], strike)
    values[lower] = 0.0
    values[upper] = S_max - strike * np.exp(-r * (maturity - t[upper]))
    if not np.all(np.isfinite(values)):
        raise ValueError("boundary values must be finite.")
    return values


def _validate_boundary_points(X_bd: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(X_bd, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"X_bd must have shape (n, 2), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X_bd must contain only finite values.")
    return arr


def _validate_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")


def _validate_positive_finite(name: str, value: float) -> None:
    _validate_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
