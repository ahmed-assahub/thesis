import numpy as np
import pytest

from option_gpr.grids import make_random_price_grid
from option_gpr.payoffs import call_boundary_values_log, call_payoff_log


def test_call_boundary_values_terminal_rows_match_call_payoff() -> None:
    X_bd = np.array([[1.0, np.log(80.0)], [1.0, np.log(100.0)], [1.0, np.log(120.0)]])

    values = call_boundary_values_log(
        X_bd, strike=100.0, maturity=1.0, r=0.05, S_min=60.0, S_max=140.0
    )

    np.testing.assert_allclose(values, call_payoff_log(X_bd[:, 1], 100.0))


def test_call_boundary_values_lower_nonterminal_rows_are_zero() -> None:
    X_bd = np.array([[0.0, np.log(60.0)], [0.4, np.log(60.0)], [0.8, np.log(60.0)]])

    values = call_boundary_values_log(
        X_bd, strike=100.0, maturity=1.0, r=0.05, S_min=60.0, S_max=140.0
    )

    np.testing.assert_allclose(values, np.zeros(3))


def test_call_boundary_values_upper_nonterminal_rows_use_far_field_call_boundary() -> None:
    X_bd = np.array([[0.0, np.log(140.0)], [0.4, np.log(140.0)], [0.8, np.log(140.0)]])

    values = call_boundary_values_log(
        X_bd, strike=100.0, maturity=1.0, r=0.05, S_min=60.0, S_max=140.0
    )

    expected = 140.0 - 100.0 * np.exp(-0.05 * (1.0 - X_bd[:, 0]))
    np.testing.assert_allclose(values, expected)


def test_call_boundary_values_mixed_rows_preserve_row_order() -> None:
    X_bd = np.array(
        [
            [1.0, np.log(120.0)],
            [0.3, np.log(60.0)],
            [0.6, np.log(140.0)],
        ]
    )

    values = call_boundary_values_log(
        X_bd, strike=100.0, maturity=1.0, r=0.05, S_min=60.0, S_max=140.0
    )

    expected = np.array([20.0, 0.0, 140.0 - 100.0 * np.exp(-0.05 * (1.0 - 0.6))])
    np.testing.assert_allclose(values, expected)


def test_call_boundary_values_terminal_corners_use_terminal_payoff() -> None:
    X_bd = np.array(
        [
            [1.0, np.log(60.0)],
            [1.0, np.log(140.0)],
            [0.5, np.log(60.0)],
            [0.5, np.log(140.0)],
        ]
    )

    values = call_boundary_values_log(
        X_bd, strike=100.0, maturity=1.0, r=0.05, S_min=60.0, S_max=140.0
    )

    expected_upper = 140.0 - 100.0 * np.exp(-0.05 * (1.0 - 0.5))
    np.testing.assert_allclose(values, np.array([0.0, 40.0, 0.0, expected_upper]))


def test_call_boundary_values_rejects_unknown_boundary_rows() -> None:
    X_bd = np.array([[0.5, np.log(100.0)]])

    with pytest.raises(ValueError):
        call_boundary_values_log(
            X_bd, strike=100.0, maturity=1.0, r=0.05, S_min=60.0, S_max=140.0
        )


@pytest.mark.parametrize(
    "bad_X_bd",
    [
        np.array([1.0, np.log(100.0)]),
        np.array([[1.0]]),
        np.array([[1.0, np.log(100.0), 0.0]]),
        np.array([[np.nan, np.log(100.0)]]),
    ],
)
def test_call_boundary_values_rejects_malformed_boundary_points(
    bad_X_bd: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        call_boundary_values_log(
            bad_X_bd, strike=100.0, maturity=1.0, r=0.05, S_min=60.0, S_max=140.0
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strike": 0.0},
        {"maturity": 0.0},
        {"r": np.inf},
        {"S_min": 0.0},
        {"S_min": 140.0, "S_max": 60.0},
        {"S_max": np.inf},
        {"atol": -1e-10},
    ],
)
def test_call_boundary_values_rejects_invalid_parameters(
    kwargs: dict[str, float],
) -> None:
    X_bd = np.array([[1.0, np.log(120.0)]])
    params = {
        "strike": 100.0,
        "maturity": 1.0,
        "r": 0.05,
        "S_min": 60.0,
        "S_max": 140.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        call_boundary_values_log(X_bd, **params)


def test_call_boundary_values_work_as_random_price_grid_boundary_function() -> None:
    grid = make_random_price_grid(
        n_int=6,
        n_terminal=5,
        n_lower=3,
        n_upper=4,
        t_min=0.0,
        t_max=0.9,
        maturity=1.0,
        S_min=60.0,
        S_max=140.0,
        boundary_value_fn=lambda X: call_boundary_values_log(
            X, strike=100.0, maturity=1.0, r=0.05, S_min=60.0, S_max=140.0
        ),
        seed=5,
    )

    assert grid.y_bd.shape == (12,)
    assert np.all(np.isfinite(grid.y_bd))
