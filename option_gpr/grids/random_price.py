"""Random grids sampled uniformly in price and returned in log-price coordinates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from option_gpr.grids.base import GridSet


def make_random_price_interior_points(
    n: int,
    t_min: float,
    t_max: float,
    S_min: float,
    S_max: float,
    maturity: float,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample interior points with uniform time and uniform price."""

    _validate_count("n", n)
    _validate_maturity(maturity)
    _validate_time_interval(t_min, t_max, maturity)
    _validate_price_interval(S_min, S_max)
    generator = _resolve_rng(seed=seed, rng=rng)
    t = generator.uniform(t_min, t_max, size=n)
    S = generator.uniform(S_min, S_max, size=n)
    return _points_from_time_and_price(t, S)


def make_random_price_terminal_boundary(
    n: int,
    maturity: float,
    S_min: float,
    S_max: float,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample terminal boundary points with uniform price."""

    _validate_count("n", n)
    _validate_maturity(maturity)
    _validate_price_interval(S_min, S_max)
    generator = _resolve_rng(seed=seed, rng=rng)
    S = generator.uniform(S_min, S_max, size=n)
    return _points_from_time_and_price(np.full(n, maturity), S)


def make_random_price_lower_boundary(
    n: int,
    t_min: float,
    maturity: float,
    S_min: float,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample lower spatial boundary points at ``S_min``."""

    _validate_count("n", n)
    _validate_maturity(maturity)
    _validate_boundary_time_min(t_min, maturity)
    _validate_positive_price("S_min", S_min)
    generator = _resolve_rng(seed=seed, rng=rng)
    t = generator.uniform(t_min, maturity, size=n)
    return _points_from_time_and_price(t, np.full(n, S_min))


def make_random_price_upper_boundary(
    n: int,
    t_min: float,
    maturity: float,
    S_max: float,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample upper spatial boundary points at ``S_max``."""

    _validate_count("n", n)
    _validate_maturity(maturity)
    _validate_boundary_time_min(t_min, maturity)
    _validate_positive_price("S_max", S_max)
    generator = _resolve_rng(seed=seed, rng=rng)
    t = generator.uniform(t_min, maturity, size=n)
    return _points_from_time_and_price(t, np.full(n, S_max))


def combine_boundary_points(
    X_terminal: NDArray[np.float64],
    X_lower: NDArray[np.float64],
    X_upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Stack terminal, lower, and upper boundary points in that order."""

    arrays = [
        _validate_point_array("X_terminal", X_terminal),
        _validate_point_array("X_lower", X_lower),
        _validate_point_array("X_upper", X_upper),
    ]
    return np.vstack(arrays)


def make_random_price_grid(
    n_int: int,
    n_terminal: int,
    n_lower: int,
    n_upper: int,
    t_min: float,
    t_max: float,
    maturity: float,
    S_min: float,
    S_max: float,
    boundary_value_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GridSet:
    """Return a random price-sampled grid with terminal and spatial boundaries."""

    generator = _resolve_rng(seed=seed, rng=rng)
    X_int = make_random_price_interior_points(
        n_int,
        t_min,
        t_max,
        S_min,
        S_max,
        maturity,
        rng=generator,
    )
    X_terminal = make_random_price_terminal_boundary(
        n_terminal,
        maturity,
        S_min,
        S_max,
        rng=generator,
    )
    X_lower = make_random_price_lower_boundary(
        n_lower,
        t_min,
        maturity,
        S_min,
        rng=generator,
    )
    X_upper = make_random_price_upper_boundary(
        n_upper,
        t_min,
        maturity,
        S_max,
        rng=generator,
    )
    X_bd = combine_boundary_points(X_terminal, X_lower, X_upper)
    y_bd = np.asarray(boundary_value_fn(X_bd), dtype=float)
    if y_bd.ndim != 1 or y_bd.shape[0] != X_bd.shape[0]:
        raise ValueError(
            "boundary_value_fn must return a 1D array with length X_bd.shape[0], "
            f"got shape {y_bd.shape}."
        )
    _validate_finite_array("y_bd", y_bd)
    return GridSet(X_int=X_int, X_bd=X_bd, y_bd=y_bd)


def _resolve_rng(
    *,
    seed: int | None,
    rng: np.random.Generator | None,
) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both.")
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _points_from_time_and_price(
    t: NDArray[np.float64], S: NDArray[np.float64]
) -> NDArray[np.float64]:
    points = np.column_stack([t, np.log(S)])
    _validate_finite_array("points", points)
    return points


def _validate_count(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _validate_maturity(maturity: float) -> None:
    if not np.isfinite(maturity) or maturity <= 0:
        raise ValueError(f"maturity must be positive and finite, got {maturity!r}.")


def _validate_time_interval(t_min: float, t_max: float, maturity: float) -> None:
    if (
        not np.isfinite(t_min)
        or not np.isfinite(t_max)
        or t_min < 0
        or t_min >= t_max
        or t_max > maturity
    ):
        raise ValueError(
            "time interval must satisfy 0 <= t_min < t_max <= maturity, "
            f"got t_min={t_min!r}, t_max={t_max!r}, maturity={maturity!r}."
        )


def _validate_boundary_time_min(t_min: float, maturity: float) -> None:
    if not np.isfinite(t_min) or t_min < 0 or t_min >= maturity:
        raise ValueError(
            "boundary time range must satisfy 0 <= t_min < maturity, "
            f"got t_min={t_min!r}, maturity={maturity!r}."
        )


def _validate_price_interval(S_min: float, S_max: float) -> None:
    _validate_positive_price("S_min", S_min)
    _validate_positive_price("S_max", S_max)
    if S_min >= S_max:
        raise ValueError(f"S_min must be less than S_max, got {S_min!r}, {S_max!r}.")


def _validate_positive_price(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")


def _validate_point_array(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2), got {arr.shape}.")
    _validate_finite_array(name, arr)
    return arr


def _validate_finite_array(name: str, values: NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
