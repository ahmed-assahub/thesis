import numpy as np
import pytest

from option_gpr.benchmarks import (
    black_scholes_call_price_log,
    merton_jump_call_price_log,
    merton_jump_call_price_mc_log,
)
from option_gpr.payoffs import call_payoff_log


def _sample_points() -> np.ndarray:
    return np.array(
        [
            [0.0, np.log(80.0)],
            [0.25, np.log(100.0)],
            [0.5, np.log(120.0)],
        ]
    )


def _params() -> dict[str, float]:
    return {
        "strike": 100.0,
        "maturity": 1.0,
        "r": 0.05,
        "sigma": 0.25,
        "jump_intensity": 1.0,
        "jump_mean": -0.1,
        "jump_std": 0.2,
    }


def test_merton_jump_call_price_log_reduces_to_black_scholes_when_intensity_zero() -> None:
    X = _sample_points()
    params = _params()
    params["jump_intensity"] = 0.0

    merton_prices = merton_jump_call_price_log(X, **params)
    bs_prices = black_scholes_call_price_log(
        X,
        strike=params["strike"],
        maturity=params["maturity"],
        r=params["r"],
        sigma=params["sigma"],
    )

    np.testing.assert_allclose(merton_prices, bs_prices)


def test_merton_jump_call_price_log_matches_payoff_at_maturity() -> None:
    params = _params()
    X = np.array(
        [
            [params["maturity"], np.log(80.0)],
            [params["maturity"], np.log(100.0)],
            [params["maturity"], np.log(120.0)],
        ]
    )

    prices = merton_jump_call_price_log(X, **params)

    np.testing.assert_allclose(prices, call_payoff_log(X[:, 1], params["strike"]))


def test_merton_jump_call_price_log_positive_jump_prices_are_finite() -> None:
    X = _sample_points()

    prices = merton_jump_call_price_log(X, **_params())

    assert prices.shape == (3,)
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)


def test_merton_jump_call_price_log_allows_zero_jump_std() -> None:
    X = _sample_points()
    params = _params()
    params["jump_std"] = 0.0

    prices = merton_jump_call_price_log(X, **params)

    assert prices.shape == (3,)
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)


def test_merton_jump_call_price_log_is_nondecreasing_in_spot() -> None:
    params = _params()
    S = np.array([70.0, 90.0, 110.0, 130.0])
    X = np.column_stack([np.zeros_like(S), np.log(S)])

    prices = merton_jump_call_price_log(X, **params)

    assert np.all(np.diff(prices) >= -1e-12)


def test_merton_jump_call_price_log_tail_tolerance_stability() -> None:
    X = _sample_points()
    params = _params()

    prices_loose = merton_jump_call_price_log(X, **params, tail_tol=1e-10)
    prices_tight = merton_jump_call_price_log(X, **params, tail_tol=1e-12)

    np.testing.assert_allclose(prices_loose, prices_tight, rtol=1e-9, atol=1e-8)


def test_merton_jump_call_price_log_raises_when_max_terms_too_small() -> None:
    X = np.array([[0.0, np.log(100.0)]])
    params = _params()

    with pytest.raises(RuntimeError):
        merton_jump_call_price_log(X, **params, max_terms=1)


@pytest.mark.parametrize(
    "bad_X",
    [
        np.array([0.0, np.log(100.0)]),
        np.array([[0.0]]),
        np.array([[0.0, np.log(100.0), 1.0]]),
        np.array([[np.nan, np.log(100.0)]]),
    ],
)
def test_merton_jump_call_price_log_rejects_invalid_shape_or_values(
    bad_X: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        merton_jump_call_price_log(bad_X, **_params())


def test_merton_jump_call_price_log_rejects_times_after_maturity() -> None:
    X = np.array([[1.01, np.log(100.0)]])

    with pytest.raises(ValueError):
        merton_jump_call_price_log(X, **_params())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strike": 0.0},
        {"maturity": 0.0},
        {"r": np.inf},
        {"sigma": 0.0},
        {"jump_intensity": -0.1},
        {"jump_mean": np.inf},
        {"jump_std": -0.1},
        {"tail_tol": 0.0},
        {"max_terms": 0},
    ],
)
def test_merton_jump_call_price_log_rejects_invalid_parameters(
    kwargs: dict[str, float],
) -> None:
    X = np.array([[0.0, np.log(100.0)]])
    params = _params()
    params.update(kwargs)

    with pytest.raises(ValueError):
        merton_jump_call_price_log(X, **params)


def test_merton_jump_call_price_mc_log_shape_and_finite_values() -> None:
    X = _sample_points()

    prices = merton_jump_call_price_mc_log(X, **_params(), n_paths=1_000, seed=123)

    assert prices.shape == (3,)
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)


def test_merton_jump_call_price_mc_log_returns_standard_errors() -> None:
    X = _sample_points()

    prices, standard_errors = merton_jump_call_price_mc_log(
        X,
        **_params(),
        n_paths=1_000,
        seed=123,
        return_std_error=True,
    )

    assert prices.shape == (3,)
    assert standard_errors.shape == (3,)
    assert np.all(np.isfinite(prices))
    assert np.all(np.isfinite(standard_errors))
    assert np.all(standard_errors >= 0.0)


def test_merton_jump_call_price_mc_log_same_seed_is_reproducible() -> None:
    X = _sample_points()

    prices_1, standard_errors_1 = merton_jump_call_price_mc_log(
        X,
        **_params(),
        n_paths=500,
        seed=123,
        return_std_error=True,
    )
    prices_2, standard_errors_2 = merton_jump_call_price_mc_log(
        X,
        **_params(),
        n_paths=500,
        seed=123,
        return_std_error=True,
    )

    np.testing.assert_array_equal(prices_1, prices_2)
    np.testing.assert_array_equal(standard_errors_1, standard_errors_2)


def test_merton_jump_call_price_mc_log_different_seeds_change_prices() -> None:
    X = _sample_points()

    prices_1 = merton_jump_call_price_mc_log(X, **_params(), n_paths=500, seed=123)
    prices_2 = merton_jump_call_price_mc_log(X, **_params(), n_paths=500, seed=456)

    assert not np.array_equal(prices_1, prices_2)


def test_merton_jump_call_price_mc_log_matches_payoff_at_maturity() -> None:
    params = _params()
    X = np.array(
        [
            [params["maturity"], np.log(80.0)],
            [params["maturity"], np.log(100.0)],
            [params["maturity"], np.log(120.0)],
        ]
    )

    prices, standard_errors = merton_jump_call_price_mc_log(
        X,
        **params,
        n_paths=10,
        seed=123,
        return_std_error=True,
    )

    np.testing.assert_allclose(prices, call_payoff_log(X[:, 1], params["strike"]))
    np.testing.assert_allclose(standard_errors, np.zeros(3))


def test_merton_jump_call_price_mc_log_no_jumps_matches_black_scholes_loosely() -> None:
    X = np.array([[0.0, np.log(100.0)]])
    params = _params()
    params["jump_intensity"] = 0.0

    mc_price, standard_error = merton_jump_call_price_mc_log(
        X,
        **params,
        n_paths=20_000,
        seed=123,
        return_std_error=True,
    )
    bs_price = black_scholes_call_price_log(
        X,
        strike=params["strike"],
        maturity=params["maturity"],
        r=params["r"],
        sigma=params["sigma"],
    )

    assert abs(mc_price[0] - bs_price[0]) < 4.0 * standard_error[0] + 0.1


def test_merton_jump_call_price_mc_log_matches_series_loosely() -> None:
    X = np.array([[0.0, np.log(100.0)]])
    params = _params()

    mc_price, standard_error = merton_jump_call_price_mc_log(
        X,
        **params,
        n_paths=30_000,
        seed=123,
        return_std_error=True,
    )
    series_price = merton_jump_call_price_log(X, **params)

    assert abs(mc_price[0] - series_price[0]) < 4.0 * standard_error[0] + 0.1


def test_merton_jump_call_price_mc_log_allows_zero_jump_std() -> None:
    X = _sample_points()
    params = _params()
    params["jump_std"] = 0.0

    prices = merton_jump_call_price_mc_log(X, **params, n_paths=1_000, seed=123)

    assert prices.shape == (3,)
    assert np.all(np.isfinite(prices))
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
def test_merton_jump_call_price_mc_log_rejects_invalid_shape_or_values(
    bad_X: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        merton_jump_call_price_mc_log(bad_X, **_params(), n_paths=10, seed=123)


def test_merton_jump_call_price_mc_log_rejects_times_after_maturity() -> None:
    X = np.array([[1.01, np.log(100.0)]])

    with pytest.raises(ValueError):
        merton_jump_call_price_mc_log(X, **_params(), n_paths=10, seed=123)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strike": 0.0},
        {"maturity": 0.0},
        {"r": np.inf},
        {"sigma": 0.0},
        {"jump_intensity": -0.1},
        {"jump_mean": np.inf},
        {"jump_std": -0.1},
        {"n_paths": 0},
    ],
)
def test_merton_jump_call_price_mc_log_rejects_invalid_parameters(
    kwargs: dict[str, float],
) -> None:
    X = np.array([[0.0, np.log(100.0)]])
    params = _params()
    params["n_paths"] = 10
    params.update(kwargs)

    with pytest.raises(ValueError):
        merton_jump_call_price_mc_log(X, **params, seed=123)


def test_merton_jump_call_price_mc_log_rejects_one_path_with_standard_error() -> None:
    X = np.array([[0.0, np.log(100.0)]])

    with pytest.raises(ValueError):
        merton_jump_call_price_mc_log(
            X,
            **_params(),
            n_paths=1,
            seed=123,
            return_std_error=True,
        )
