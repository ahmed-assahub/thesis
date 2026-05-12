"""Black-Scholes reference prices in log-price coordinates."""

from __future__ import annotations

from math import erf, pi

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


def black_scholes_call_delta(
    t: ArrayLike,
    S: ArrayLike,
    *,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Return Black-Scholes European call Delta in price coordinates."""

    d1, _ = _d1_d2(t, S, strike=strike, maturity=maturity, r=r, sigma=sigma)
    return _normal_cdf(d1)


def black_scholes_call_gamma(
    t: ArrayLike,
    S: ArrayLike,
    *,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Return Black-Scholes European call Gamma in price coordinates."""

    _t_arr, S_arr, tau = _validate_price_inputs(
        t, S, strike=strike, maturity=maturity, r=r, sigma=sigma
    )
    d1, _ = _d1_d2_from_validated(S_arr, tau, strike=strike, r=r, sigma=sigma)
    return _normal_pdf(d1) / (S_arr * sigma * np.sqrt(tau))


def black_scholes_call_theta(
    t: ArrayLike,
    S: ArrayLike,
    *,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Return Black-Scholes European call Theta ``dV/dt`` at fixed maturity."""

    _t_arr, S_arr, tau = _validate_price_inputs(
        t, S, strike=strike, maturity=maturity, r=r, sigma=sigma
    )
    d1, d2 = _d1_d2_from_validated(S_arr, tau, strike=strike, r=r, sigma=sigma)
    sqrt_tau = np.sqrt(tau)
    diffusion_term = -S_arr * sigma * _normal_pdf(d1) / (2.0 * sqrt_tau)
    discount_term = -r * strike * np.exp(-r * tau) * _normal_cdf(d2)
    return diffusion_term + discount_term


def black_scholes_call_greeks_log(
    X: ArrayLike,
    *,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return European call ``Delta``, ``Gamma``, and ``Theta`` for ``(t, log(S))``.

    The returned Delta and Gamma are Greeks with respect to the original asset
    price ``S``. Theta is the calendar-time derivative ``dV/dt`` at fixed
    maturity, not the derivative with respect to remaining time.
    """

    X_arr = _validate_points(X)
    t = X_arr[:, 0]
    S = np.exp(X_arr[:, 1])
    return (
        black_scholes_call_delta(
            t, S, strike=strike, maturity=maturity, r=r, sigma=sigma
        ),
        black_scholes_call_gamma(
            t, S, strike=strike, maturity=maturity, r=r, sigma=sigma
        ),
        black_scholes_call_theta(
            t, S, strike=strike, maturity=maturity, r=r, sigma=sigma
        ),
    )


def _normal_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def _normal_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * pi)


def _d1_d2(
    t: ArrayLike,
    S: ArrayLike,
    *,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    _t_arr, S_arr, tau = _validate_price_inputs(
        t, S, strike=strike, maturity=maturity, r=r, sigma=sigma
    )
    return _d1_d2_from_validated(S_arr, tau, strike=strike, r=r, sigma=sigma)


def _d1_d2_from_validated(
    S: NDArray[np.float64],
    tau: NDArray[np.float64],
    *,
    strike: float,
    r: float,
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / strike) + (r + 0.5 * sigma**2) * tau) / (
        sigma * sqrt_tau
    )
    d2 = d1 - sigma * sqrt_tau
    return d1, d2


def _validate_price_inputs(
    t: ArrayLike,
    S: ArrayLike,
    *,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    _validate_positive_finite("strike", strike)
    _validate_positive_finite("maturity", maturity)
    _validate_finite("r", r)
    _validate_positive_finite("sigma", sigma)

    t_arr = np.asarray(t, dtype=float)
    S_arr = np.asarray(S, dtype=float)
    try:
        t_broadcast, S_broadcast = np.broadcast_arrays(t_arr, S_arr)
    except ValueError as exc:
        raise ValueError("t and S must be broadcastable to a common shape.") from exc

    if not np.all(np.isfinite(t_broadcast)):
        raise ValueError("t must contain only finite values.")
    if not np.all(np.isfinite(S_broadcast)):
        raise ValueError("S must contain only finite values.")
    if np.any(S_broadcast <= 0):
        raise ValueError("S must be positive.")

    tau = maturity - t_broadcast
    if np.any(tau <= 0):
        raise ValueError("all evaluation times must be strictly before maturity.")
    return t_broadcast.astype(float), S_broadcast.astype(float), tau.astype(float)


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
