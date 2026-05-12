import numpy as np
import pytest

from option_gpr.benchmarks import (
    black_scholes_call_delta,
    black_scholes_call_gamma,
    black_scholes_call_greeks_log,
    black_scholes_call_price_log,
    black_scholes_call_theta,
)
from option_gpr.payoffs import call_payoff_log


def test_black_scholes_call_price_log_shape() -> None:
    X = np.array([[0.0, np.log(100.0)], [0.5, np.log(110.0)]])

    prices = black_scholes_call_price_log(
        X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )

    assert prices.shape == (2,)


def test_black_scholes_call_price_log_matches_payoff_at_maturity() -> None:
    X = np.array([[1.0, np.log(80.0)], [1.0, np.log(100.0)], [1.0, np.log(120.0)]])

    prices = black_scholes_call_price_log(
        X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )

    np.testing.assert_allclose(prices, call_payoff_log(X[:, 1], 100.0))


def test_black_scholes_call_price_log_known_atm_value() -> None:
    X = np.array([[0.0, np.log(100.0)]])

    prices = black_scholes_call_price_log(
        X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )

    assert prices[0] == pytest.approx(10.4506, abs=1e-4)


def test_black_scholes_call_price_log_is_nonnegative() -> None:
    X = np.array([[0.0, np.log(80.0)], [0.5, np.log(100.0)], [1.0, np.log(120.0)]])

    prices = black_scholes_call_price_log(
        X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )

    assert np.all(prices >= 0.0)


def test_black_scholes_call_greeks_log_shapes() -> None:
    X = np.array(
        [[0.0, np.log(90.0)], [0.25, np.log(100.0)], [0.5, np.log(110.0)]]
    )

    delta, gamma, theta = black_scholes_call_greeks_log(
        X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )

    assert delta.shape == (3,)
    assert gamma.shape == (3,)
    assert theta.shape == (3,)
    assert np.all(np.isfinite(delta))
    assert np.all(np.isfinite(gamma))
    assert np.all(np.isfinite(theta))


def test_black_scholes_delta_matches_price_finite_difference() -> None:
    t = np.array([0.1, 0.3, 0.6])
    S = np.array([85.0, 100.0, 120.0])
    step = 1e-3

    delta = black_scholes_call_delta(
        t, S, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )
    price_plus = _price_from_t_and_S(t, S + step)
    price_minus = _price_from_t_and_S(t, S - step)
    finite_difference = (price_plus - price_minus) / (2.0 * step)

    np.testing.assert_allclose(delta, finite_difference, rtol=1e-5, atol=1e-7)


def test_black_scholes_gamma_matches_delta_finite_difference() -> None:
    t = np.array([0.1, 0.3, 0.6])
    S = np.array([85.0, 100.0, 120.0])
    step = 1e-3

    gamma = black_scholes_call_gamma(
        t, S, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )
    delta_plus = black_scholes_call_delta(
        t, S + step, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )
    delta_minus = black_scholes_call_delta(
        t, S - step, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )
    finite_difference = (delta_plus - delta_minus) / (2.0 * step)

    np.testing.assert_allclose(gamma, finite_difference, rtol=1e-4, atol=1e-6)


def test_black_scholes_theta_matches_calendar_time_finite_difference() -> None:
    t = np.array([0.1, 0.3, 0.6])
    S = np.array([85.0, 100.0, 120.0])
    step = 1e-5

    theta = black_scholes_call_theta(
        t, S, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )
    price_plus = _price_from_t_and_S(t + step, S)
    price_minus = _price_from_t_and_S(t - step, S)
    finite_difference = (price_plus - price_minus) / (2.0 * step)

    np.testing.assert_allclose(theta, finite_difference, rtol=1e-5, atol=1e-7)


def test_black_scholes_call_greeks_log_matches_price_coordinate_helpers() -> None:
    S = np.array([90.0, 100.0, 110.0])
    t = np.array([0.0, 0.25, 0.5])
    X = np.column_stack([t, np.log(S)])

    delta_log, gamma_log, theta_log = black_scholes_call_greeks_log(
        X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )

    np.testing.assert_allclose(
        delta_log,
        black_scholes_call_delta(
            t, S, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
        ),
    )
    np.testing.assert_allclose(
        gamma_log,
        black_scholes_call_gamma(
            t, S, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
        ),
    )
    np.testing.assert_allclose(
        theta_log,
        black_scholes_call_theta(
            t, S, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
        ),
    )


@pytest.mark.parametrize(
    "bad_X",
    [
        np.array([0.0, np.log(100.0)]),
        np.array([[0.0]]),
        np.array([[0.0, np.log(100.0), 1.0]]),
        np.array([[np.nan, np.log(100.0)]]),
    ],
)
def test_black_scholes_call_price_log_rejects_invalid_shape_or_values(
    bad_X: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        black_scholes_call_price_log(
            bad_X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
        )


def test_black_scholes_call_price_log_rejects_times_after_maturity() -> None:
    X = np.array([[1.01, np.log(100.0)]])

    with pytest.raises(ValueError):
        black_scholes_call_price_log(
            X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strike": 0.0},
        {"maturity": 0.0},
        {"r": np.inf},
        {"sigma": 0.0},
        {"sigma": np.nan},
    ],
)
def test_black_scholes_call_price_log_rejects_invalid_parameters(
    kwargs: dict[str, float],
) -> None:
    X = np.array([[0.0, np.log(100.0)]])
    params = {"strike": 100.0, "maturity": 1.0, "r": 0.05, "sigma": 0.2}
    params.update(kwargs)

    with pytest.raises(ValueError):
        black_scholes_call_price_log(X, **params)


@pytest.mark.parametrize(
    "X",
    [
        np.array([[1.0, np.log(100.0)]]),
        np.array([[1.01, np.log(100.0)]]),
    ],
)
def test_black_scholes_call_greeks_log_rejects_times_at_or_after_maturity(
    X: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        black_scholes_call_greeks_log(
            X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
        )


@pytest.mark.parametrize(
    "bad_X",
    [
        np.array([0.0, np.log(100.0)]),
        np.array([[0.0]]),
        np.array([[0.0, np.log(100.0), 1.0]]),
        np.array([[np.nan, np.log(100.0)]]),
    ],
)
def test_black_scholes_call_greeks_log_rejects_invalid_shape_or_values(
    bad_X: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        black_scholes_call_greeks_log(
            bad_X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
        )


@pytest.mark.parametrize(
    "helper",
    [
        black_scholes_call_delta,
        black_scholes_call_gamma,
        black_scholes_call_theta,
    ],
)
def test_black_scholes_call_greek_helpers_reject_nonpositive_spot(
    helper: object,
) -> None:
    with pytest.raises(ValueError):
        helper(0.0, 0.0, strike=100.0, maturity=1.0, r=0.05, sigma=0.2)


@pytest.mark.parametrize(
    "helper",
    [
        black_scholes_call_delta,
        black_scholes_call_gamma,
        black_scholes_call_theta,
    ],
)
def test_black_scholes_call_greek_helpers_reject_times_at_or_after_maturity(
    helper: object,
) -> None:
    with pytest.raises(ValueError):
        helper(1.0, 100.0, strike=100.0, maturity=1.0, r=0.05, sigma=0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strike": 0.0},
        {"maturity": 0.0},
        {"r": np.inf},
        {"sigma": 0.0},
        {"sigma": np.nan},
    ],
)
def test_black_scholes_call_greeks_log_rejects_invalid_parameters(
    kwargs: dict[str, float],
) -> None:
    X = np.array([[0.0, np.log(100.0)]])
    params = {"strike": 100.0, "maturity": 1.0, "r": 0.05, "sigma": 0.2}
    params.update(kwargs)

    with pytest.raises(ValueError):
        black_scholes_call_greeks_log(X, **params)


def _price_from_t_and_S(t: np.ndarray, S: np.ndarray) -> np.ndarray:
    X = np.column_stack([t, np.log(S)])
    return black_scholes_call_price_log(
        X, strike=100.0, maturity=1.0, r=0.05, sigma=0.2
    )
