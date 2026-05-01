import numpy as np
import pytest

from option_gpr.grids import (
    GridSet,
    combine_boundary_points,
    make_random_price_grid,
    make_random_price_interior_points,
    make_random_price_lower_boundary,
    make_random_price_terminal_boundary,
    make_random_price_upper_boundary,
)


def test_random_price_interior_points_shape_and_ranges() -> None:
    X = make_random_price_interior_points(
        n=10,
        t_min=0.1,
        t_max=0.8,
        S_min=60.0,
        S_max=140.0,
        maturity=1.0,
        seed=1,
    )

    assert X.shape == (10, 2)
    assert np.all(X[:, 0] >= 0.1)
    assert np.all(X[:, 0] <= 0.8)
    assert np.all(np.exp(X[:, 1]) >= 60.0)
    assert np.all(np.exp(X[:, 1]) <= 140.0)


def test_random_price_terminal_boundary_shape_and_ranges() -> None:
    X = make_random_price_terminal_boundary(
        n=8,
        maturity=1.0,
        S_min=60.0,
        S_max=140.0,
        seed=2,
    )

    assert X.shape == (8, 2)
    np.testing.assert_allclose(X[:, 0], 1.0)
    assert np.all(np.exp(X[:, 1]) >= 60.0)
    assert np.all(np.exp(X[:, 1]) <= 140.0)


def test_random_price_lower_boundary_shape_and_ranges() -> None:
    X = make_random_price_lower_boundary(
        n=7,
        t_min=0.2,
        maturity=1.0,
        S_min=60.0,
        seed=3,
    )

    assert X.shape == (7, 2)
    assert np.all(X[:, 0] >= 0.2)
    assert np.all(X[:, 0] <= 1.0)
    np.testing.assert_allclose(X[:, 1], np.log(60.0))


def test_random_price_upper_boundary_shape_and_ranges() -> None:
    X = make_random_price_upper_boundary(
        n=7,
        t_min=0.2,
        maturity=1.0,
        S_max=140.0,
        seed=4,
    )

    assert X.shape == (7, 2)
    assert np.all(X[:, 0] >= 0.2)
    assert np.all(X[:, 0] <= 1.0)
    np.testing.assert_allclose(X[:, 1], np.log(140.0))


def test_combine_boundary_points_stacks_terminal_lower_upper() -> None:
    X_terminal = make_random_price_terminal_boundary(5, 1.0, 60.0, 140.0, seed=1)
    X_lower = make_random_price_lower_boundary(3, 0.0, 1.0, 60.0, seed=2)
    X_upper = make_random_price_upper_boundary(4, 0.0, 1.0, 140.0, seed=3)

    X_bd = combine_boundary_points(X_terminal, X_lower, X_upper)

    assert X_bd.shape == (12, 2)
    np.testing.assert_allclose(X_bd[:5], X_terminal)
    np.testing.assert_allclose(X_bd[5:8], X_lower)
    np.testing.assert_allclose(X_bd[8:], X_upper)


def test_make_random_price_grid_returns_grid_set_with_expected_shapes() -> None:
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
        boundary_value_fn=lambda X: np.exp(X[:, 1]),
        seed=5,
    )

    assert isinstance(grid, GridSet)
    assert grid.X_int.shape == (6, 2)
    assert grid.X_bd.shape == (12, 2)
    assert grid.y_bd.shape == (12,)
    np.testing.assert_allclose(grid.y_bd, np.exp(grid.X_bd[:, 1]))


def test_random_price_grid_same_seed_reproduces_grid() -> None:
    kwargs = {
        "n_int": 6,
        "n_terminal": 5,
        "n_lower": 3,
        "n_upper": 4,
        "t_min": 0.0,
        "t_max": 0.9,
        "maturity": 1.0,
        "S_min": 60.0,
        "S_max": 140.0,
        "boundary_value_fn": lambda X: np.exp(X[:, 1]),
        "seed": 10,
    }

    first = make_random_price_grid(**kwargs)
    second = make_random_price_grid(**kwargs)

    np.testing.assert_allclose(first.X_int, second.X_int)
    np.testing.assert_allclose(first.X_bd, second.X_bd)
    np.testing.assert_allclose(first.y_bd, second.y_bd)


def test_random_price_grid_different_seed_changes_interior_points() -> None:
    kwargs = {
        "n": 6,
        "t_min": 0.0,
        "t_max": 0.9,
        "S_min": 60.0,
        "S_max": 140.0,
        "maturity": 1.0,
    }

    first = make_random_price_interior_points(**kwargs, seed=10)
    second = make_random_price_interior_points(**kwargs, seed=11)

    assert not np.allclose(first, second)


def test_random_price_grid_rejects_seed_and_rng_together() -> None:
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError):
        make_random_price_interior_points(
            n=3,
            t_min=0.0,
            t_max=0.9,
            S_min=60.0,
            S_max=140.0,
            maturity=1.0,
            seed=1,
            rng=rng,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"t_min": -0.1},
        {"t_min": 0.5, "t_max": 0.5},
        {"t_max": 1.1},
        {"maturity": 0.0},
        {"S_min": 0.0},
        {"S_min": 140.0, "S_max": 60.0},
        {"S_max": np.inf},
    ],
)
def test_random_price_interior_rejects_invalid_inputs(
    kwargs: dict[str, float | int]
) -> None:
    params = {
        "n": 3,
        "t_min": 0.0,
        "t_max": 0.9,
        "S_min": 60.0,
        "S_max": 140.0,
        "maturity": 1.0,
        "seed": 1,
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        make_random_price_interior_points(**params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"t_min": -0.1},
        {"t_min": 1.0},
        {"maturity": 0.0},
        {"S_min": 0.0},
    ],
)
def test_random_price_lower_boundary_rejects_invalid_inputs(
    kwargs: dict[str, float | int]
) -> None:
    params = {
        "n": 3,
        "t_min": 0.0,
        "maturity": 1.0,
        "S_min": 60.0,
        "seed": 1,
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        make_random_price_lower_boundary(**params)


@pytest.mark.parametrize(
    "boundary_value_fn",
    [
        lambda X: np.array([[1.0, 2.0]]),
        lambda X: np.array([1.0, 2.0]),
        lambda X: np.full(X.shape[0], np.nan),
    ],
)
def test_make_random_price_grid_rejects_malformed_boundary_values(
    boundary_value_fn,
) -> None:
    with pytest.raises(ValueError):
        make_random_price_grid(
            n_int=6,
            n_terminal=5,
            n_lower=3,
            n_upper=4,
            t_min=0.0,
            t_max=0.9,
            maturity=1.0,
            S_min=60.0,
            S_max=140.0,
            boundary_value_fn=boundary_value_fn,
            seed=5,
        )
