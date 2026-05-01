import numpy as np
import pytest

from option_gpr.benchmarks import black_scholes_call_price_log
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
