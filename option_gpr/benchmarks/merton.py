"""Merton jump-diffusion reference prices in log-price coordinates."""

from __future__ import annotations

from math import pi

import numpy as np
from numpy.typing import ArrayLike, NDArray

from option_gpr.benchmarks.black_scholes import (
    black_scholes_call_delta,
    black_scholes_call_gamma,
    black_scholes_call_price_log,
    black_scholes_call_theta,
)
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


def merton_jump_call_price_mc_log(
    X: ArrayLike,
    *,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    n_paths: int,
    seed: int | None = None,
    return_std_error: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Merton call prices by exact terminal Monte Carlo simulation.

    The simulation samples the terminal value under the risk-neutral Merton
    jump-diffusion distribution directly, so it has Monte Carlo sampling error
    but no Euler time-discretization error. ``n_paths`` is the number of
    terminal samples per input row. If ``return_std_error`` is true, also return
    standard errors computed from the discounted payoff samples.
    """

    X_arr = _validate_points(X)
    _validate_positive_finite("strike", strike)
    _validate_positive_finite("maturity", maturity)
    _validate_finite("r", r)
    _validate_positive_finite("sigma", sigma)
    _validate_nonnegative_finite("jump_intensity", jump_intensity)
    _validate_finite("jump_mean", jump_mean)
    _validate_nonnegative_finite("jump_std", jump_std)
    _validate_n_paths(n_paths, return_std_error=return_std_error)

    t = X_arr[:, 0]
    x = X_arr[:, 1]
    if np.any(t > maturity):
        raise ValueError("all evaluation times must be less than or equal to maturity.")

    rng = np.random.default_rng(seed)
    prices = np.empty(X_arr.shape[0], dtype=float)
    standard_errors = np.empty(X_arr.shape[0], dtype=float)

    for row_index, (t_i, x_i) in enumerate(X_arr):
        price, standard_error = _merton_jump_call_price_mc_one_row(
            t=float(t_i),
            x=float(x_i),
            strike=strike,
            maturity=maturity,
            r=r,
            sigma=sigma,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std,
            n_paths=n_paths,
            rng=rng,
            return_std_error=return_std_error,
        )
        prices[row_index] = price
        standard_errors[row_index] = standard_error

    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(standard_errors)):
        raise ValueError("computed Monte Carlo prices must be finite.")
    if return_std_error:
        return prices, standard_errors
    return prices


def merton_jump_call_delta(
    t: ArrayLike,
    S: ArrayLike,
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
    """Return Merton European call Delta with respect to asset price ``S``."""

    delta, _, _ = _merton_jump_call_greeks_price(
        t,
        S,
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
    return delta


def merton_jump_call_gamma(
    t: ArrayLike,
    S: ArrayLike,
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
    """Return Merton European call Gamma with respect to asset price ``S``."""

    _, gamma, _ = _merton_jump_call_greeks_price(
        t,
        S,
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
    return gamma


def merton_jump_call_theta(
    t: ArrayLike,
    S: ArrayLike,
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
    """Return Merton European call Theta ``dC/dt`` at fixed maturity."""

    _, _, theta = _merton_jump_call_greeks_price(
        t,
        S,
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
    return theta


def merton_jump_call_greeks_log(
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return Merton call ``Delta``, ``Gamma``, and ``Theta`` for ``(t, log(S))``.

    The semi-analytical Merton series is differentiated term by term. Delta
    and Gamma are with respect to the original asset price ``S``. Theta is the
    calendar-time derivative ``dC/dt`` at fixed maturity.
    """

    X_arr = _validate_points(X)
    t = X_arr[:, 0]
    S = np.exp(X_arr[:, 1])
    return _merton_jump_call_greeks_price(
        t,
        S,
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


def _merton_jump_call_greeks_price(
    t: ArrayLike,
    S: ArrayLike,
    *,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    tail_tol: float,
    max_terms: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    t_arr, S_arr = _validate_greek_price_inputs(
        t,
        S,
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

    delta = np.empty(t_arr.shape, dtype=float)
    gamma = np.empty(t_arr.shape, dtype=float)
    theta = np.empty(t_arr.shape, dtype=float)
    flat_t = t_arr.ravel()
    flat_S = S_arr.ravel()
    flat_delta = delta.ravel()
    flat_gamma = gamma.ravel()
    flat_theta = theta.ravel()

    for index, (t_i, S_i) in enumerate(zip(flat_t, flat_S, strict=True)):
        row_delta, row_gamma, row_theta = _merton_jump_call_greeks_one_row(
            t=float(t_i),
            S=float(S_i),
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
        flat_delta[index] = row_delta
        flat_gamma[index] = row_gamma
        flat_theta[index] = row_theta

    if (
        not np.all(np.isfinite(delta))
        or not np.all(np.isfinite(gamma))
        or not np.all(np.isfinite(theta))
    ):
        raise ValueError("computed Merton Greeks must be finite.")
    return delta, gamma, theta


def _merton_jump_call_greeks_one_row(
    *,
    t: float,
    S: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    tail_tol: float,
    max_terms: int,
) -> tuple[float, float, float]:
    tau = maturity - t
    poisson_mean = jump_intensity * tau
    jump_compensator = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    poisson_weight = float(np.exp(-poisson_mean))
    cumulative_probability = 0.0
    delta = 0.0
    gamma = 0.0
    d_price_dtau = 0.0

    for n in range(max_terms):
        scale = np.exp(
            n * jump_mean
            + 0.5 * n * jump_std**2
            - jump_intensity * tau * jump_compensator
        )
        S_n = S * scale
        sigma_n = float(np.sqrt(sigma**2 + n * jump_std**2 / tau))
        X_n = np.array([[t, np.log(S_n)]], dtype=float)
        bs_price = black_scholes_call_price_log(
            X_n,
            strike=strike,
            maturity=maturity,
            r=r,
            sigma=sigma_n,
        )[0]
        bs_delta = black_scholes_call_delta(
            np.array([t]),
            np.array([S_n]),
            strike=strike,
            maturity=maturity,
            r=r,
            sigma=sigma_n,
        )[0]
        bs_gamma = black_scholes_call_gamma(
            np.array([t]),
            np.array([S_n]),
            strike=strike,
            maturity=maturity,
            r=r,
            sigma=sigma_n,
        )[0]
        bs_theta = black_scholes_call_theta(
            np.array([t]),
            np.array([S_n]),
            strike=strike,
            maturity=maturity,
            r=r,
            sigma=sigma_n,
        )[0]
        vega = _black_scholes_call_vega(
            t=t,
            S=S_n,
            strike=strike,
            maturity=maturity,
            r=r,
            sigma=sigma_n,
        )

        delta += poisson_weight * bs_delta * scale
        gamma += poisson_weight * bs_gamma * scale**2

        if n == 0:
            poisson_weight_tau = -jump_intensity * poisson_weight
            sigma_n_tau = 0.0
        else:
            poisson_weight_tau = poisson_weight * (n / tau - jump_intensity)
            sigma_n_tau = -n * jump_std**2 / (2.0 * tau**2 * sigma_n)
        S_n_tau = -jump_intensity * jump_compensator * S_n
        d_term_dtau = poisson_weight_tau * bs_price + poisson_weight * (
            -bs_theta + bs_delta * S_n_tau + vega * sigma_n_tau
        )
        d_price_dtau += d_term_dtau

        cumulative_probability += poisson_weight
        remaining_tail = max(0.0, 1.0 - cumulative_probability)
        if remaining_tail <= tail_tol:
            return float(delta), float(gamma), float(-d_price_dtau)

        poisson_weight *= poisson_mean / (n + 1.0)

    raise RuntimeError(
        "Merton Greek series did not reach the requested Poisson tail tolerance "
        f"tail_tol={tail_tol!r} within max_terms={max_terms!r}."
    )


def _black_scholes_call_vega(
    *,
    t: float,
    S: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
) -> float:
    tau = maturity - t
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / strike) + (r + 0.5 * sigma**2) * tau) / (
        sigma * sqrt_tau
    )
    return float(S * _normal_pdf(np.array([d1]))[0] * sqrt_tau)


def _normal_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * pi)


def _merton_jump_call_price_mc_one_row(
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
    n_paths: int,
    rng: np.random.Generator,
    return_std_error: bool,
) -> tuple[float, float]:
    tau = maturity - t
    if np.isclose(tau, 0.0, atol=1e-14, rtol=0.0):
        payoff = float(call_payoff_log(np.array([x]), strike)[0])
        return payoff, 0.0

    jump_compensator = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    drift = r - 0.5 * sigma**2 - jump_intensity * jump_compensator
    normals = rng.normal(size=n_paths)
    jump_counts = rng.poisson(jump_intensity * tau, size=n_paths)
    if jump_std == 0.0:
        jump_total = jump_counts * jump_mean
    else:
        jump_total = rng.normal(
            loc=jump_counts * jump_mean,
            scale=np.sqrt(jump_counts) * jump_std,
        )

    log_terminal = (
        x
        + drift * tau
        + sigma * np.sqrt(tau) * normals
        + jump_total
    )
    terminal_spot = np.exp(log_terminal)
    discounted_payoffs = np.exp(-r * tau) * np.maximum(terminal_spot - strike, 0.0)
    price = float(np.mean(discounted_payoffs))
    if return_std_error:
        standard_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))
    else:
        standard_error = 0.0
    return price, standard_error


def _validate_points(X: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"X must have shape (n, 2), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X must contain only finite values.")
    return arr


def _validate_greek_price_inputs(
    t: ArrayLike,
    S: ArrayLike,
    *,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    tail_tol: float,
    max_terms: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    if np.any(maturity - t_broadcast <= 0):
        raise ValueError("all evaluation times must be strictly before maturity.")
    return t_broadcast.astype(float), S_broadcast.astype(float)


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


def _validate_n_paths(n_paths: int, *, return_std_error: bool) -> None:
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths!r}.")
    if return_std_error and n_paths < 2:
        raise ValueError(
            "n_paths must be at least 2 when return_std_error=True, "
            f"got {n_paths!r}."
        )
