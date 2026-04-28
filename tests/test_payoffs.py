import numpy as np
import pytest

from option_gpr.payoffs import call_payoff_log, put_payoff_log


def test_call_payoff_log_at_below_at_and_above_strike() -> None:
    strike = 100.0
    stock_prices = np.array([80.0, 100.0, 120.0])
    x = np.log(stock_prices)

    np.testing.assert_allclose(
        call_payoff_log(x, strike), np.array([0.0, 0.0, 20.0]), atol=1e-12
    )


def test_put_payoff_log_at_below_at_and_above_strike() -> None:
    strike = 100.0
    stock_prices = np.array([80.0, 100.0, 120.0])
    x = np.log(stock_prices)

    np.testing.assert_allclose(
        put_payoff_log(x, strike), np.array([20.0, 0.0, 0.0]), atol=1e-12
    )


def test_terminal_call_payoff_is_not_discounted() -> None:
    payoff = call_payoff_log(np.log(np.array([120.0])), strike=100.0)

    np.testing.assert_allclose(payoff, np.array([20.0]))


def test_payoffs_accept_scalar_log_price() -> None:
    assert call_payoff_log(np.log(120.0), strike=100.0) == pytest.approx(20.0)
    assert put_payoff_log(np.log(80.0), strike=100.0) == pytest.approx(20.0)


@pytest.mark.parametrize("bad_strike", [0.0, -1.0, np.inf])
def test_payoffs_require_positive_finite_strike(bad_strike: float) -> None:
    with pytest.raises(ValueError):
        call_payoff_log(np.array([0.0]), strike=bad_strike)

    with pytest.raises(ValueError):
        put_payoff_log(np.array([0.0]), strike=bad_strike)
