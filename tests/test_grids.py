import numpy as np
import pytest

from option_gpr.grids import GridSet, make_uniform_terminal_grid
from option_gpr.payoffs import call_payoff_log


def _make_grid() -> GridSet:
    strike = 100.0
    return make_uniform_terminal_grid(
        maturity=1.0,
        x_min=np.log(60.0),
        x_max=np.log(140.0),
        n_t_int=3,
        n_x_int=4,
        n_x_bd=5,
        payoff_fn=lambda x: call_payoff_log(x, strike),
    )


def test_make_uniform_terminal_grid_returns_grid_set() -> None:
    assert isinstance(_make_grid(), GridSet)


def test_make_uniform_terminal_grid_shapes() -> None:
    grid = _make_grid()

    assert grid.X_int.shape == (12, 2)
    assert grid.X_bd.shape == (5, 2)
    assert grid.y_bd.shape == (5,)


def test_uniform_terminal_grid_has_interior_times_and_terminal_boundary() -> None:
    grid = _make_grid()
    maturity = 1.0

    assert np.all(grid.X_int[:, 0] > 0.0)
    assert np.all(grid.X_int[:, 0] < maturity)
    np.testing.assert_allclose(grid.X_bd[:, 0], maturity)


def test_uniform_terminal_grid_boundary_spans_log_price_domain() -> None:
    grid = _make_grid()
    x_min = np.log(60.0)
    x_max = np.log(140.0)

    assert grid.X_bd[0, 1] == pytest.approx(x_min)
    assert grid.X_bd[-1, 1] == pytest.approx(x_max)


def test_uniform_terminal_grid_boundary_values_match_payoff() -> None:
    grid = _make_grid()

    np.testing.assert_allclose(grid.y_bd, call_payoff_log(grid.X_bd[:, 1], 100.0))


def test_uniform_terminal_grid_has_no_duplicate_rows_in_each_block() -> None:
    grid = _make_grid()

    assert np.unique(grid.X_int, axis=0).shape == grid.X_int.shape
    assert np.unique(grid.X_bd, axis=0).shape == grid.X_bd.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maturity": 0.0},
        {"maturity": np.inf},
        {"x_min": 1.0, "x_max": 1.0},
        {"x_min": np.nan},
        {"n_t_int": 0},
        {"n_x_int": 0},
        {"n_x_bd": 1},
    ],
)
def test_make_uniform_terminal_grid_rejects_invalid_grid_parameters(
    kwargs: dict[str, float | int]
) -> None:
    params = {
        "maturity": 1.0,
        "x_min": np.log(60.0),
        "x_max": np.log(140.0),
        "n_t_int": 3,
        "n_x_int": 4,
        "n_x_bd": 5,
        "payoff_fn": lambda x: call_payoff_log(x, 100.0),
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        make_uniform_terminal_grid(**params)


@pytest.mark.parametrize(
    "payoff_fn",
    [
        lambda x: np.array([[1.0, 2.0]]),
        lambda x: np.array([1.0, 2.0]),
        lambda x: np.full(x.shape, np.nan),
    ],
)
def test_make_uniform_terminal_grid_rejects_malformed_payoff_output(
    payoff_fn,
) -> None:
    with pytest.raises(ValueError):
        make_uniform_terminal_grid(
            maturity=1.0,
            x_min=np.log(60.0),
            x_max=np.log(140.0),
            n_t_int=3,
            n_x_int=4,
            n_x_bd=5,
            payoff_fn=payoff_fn,
        )
