"""Merton jump-diffusion reference prices in log-price coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from option_gpr.benchmarks.black_scholes import black_scholes_call_price_log
from option_gpr.payoffs import call_payoff_log


def merton_jump_call_price_log(
    X: ArrayLike,
    *,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    tail_tol: float = 1e-12,
    max_terms: int = 200,
) -> NDArray[np.float64]:
    """Return Merton jump-diffusion call prices for rows ``(t, log(S))``.

    The price is evaluated as a Poisson-weighted Black-Scholes series. The
    Black-Scholes term already includes discounting, so no additional external
    discount factor is applied to the series.
    """

    X_arr = _validate_points(X)
    _validate_positive_finite("strike", strike)
    _validate_positive_finite("maturity", maturity)
    _validate_finite("r", r)
    _validate_positive_finite("sigma", sigma)
    _validate_nonnegative_finite("jump_intensity", jump_intensity)
    _validate_finite("jump_mean", jump_mean)
    _validate_nonnegative_finite("jump_std", jump_std)
    _validate_positive_finite("tail_tol", tail_tol)
    if max_terms < 1:
        raise ValueError(f"max_terms must be at least 1, got {max_terms!r}.")

    t = X_arr[:, 0]
    x = X_arr[:, 1]
    if np.any(t > maturity):
        raise ValueError("all evaluation times must be less than or equal to maturity.")

    prices = np.empty(X_arr.shape[0], dtype=float)
    at_maturity = np.isclose(t, maturity, atol=1e-14, rtol=0.0)
    prices[at_maturity] = call_payoff_log(x[at_maturity], strike)

    before_maturity = ~at_maturity
    for row_index in np.flatnonzero(before_maturity):
        prices[row_index] = _merton_jump_call_price_one_row(
            t=float(t[row_index]),
            x=float(x[row_index]),
            strike=strike,
            maturity=maturity,
            r=r,
            sigma=sigma,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std,
            tail_tol=tail_tol,
            max_terms=max_terms,
        )

    if not np.all(np.isfinite(prices)):
        raise ValueError("computed prices must be finite.")
    return prices


def _merton_jump_call_price_one_row(
    *,
    t: float,
    x: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    tail_tol: float,
    max_terms: int,
) -> float:
    tau = maturity - t
    poisson_mean = jump_intensity * tau
    jump_compensator = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    poisson_weight = float(np.exp(-poisson_mean))
    cumulative_probability = 0.0
    price = 0.0

    for n in range(max_terms):
        x_n = (
            x
            + n * jump_mean
            + 0.5 * n * jump_std**2
            - jump_intensity * tau * jump_compensator
        )
        sigma_n = float(np.sqrt(sigma**2 + n * jump_std**2 / tau))
        bs_price = black_scholes_call_price_log(
            np.array([[t, x_n]], dtype=float),
            strike=strike,
            maturity=maturity,
            r=r,
            sigma=sigma_n,
        )[0]
        price += poisson_weight * bs_price
        cumulative_probability += poisson_weight

        remaining_tail = max(0.0, 1.0 - cumulative_probability)
        if remaining_tail <= tail_tol:
            return float(price)

        poisson_weight *= poisson_mean / (n + 1.0)

    raise RuntimeError(
        "Merton series did not reach the requested Poisson tail tolerance "
        f"tail_tol={tail_tol!r} within max_terms={max_terms!r}."
    )


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


def _validate_nonnegative_finite(name: str, value: float) -> None:
    _validate_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}.")
