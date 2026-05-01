"""Uniform grids for terminal-only boundary experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GridSet:
    """Interior collocation points and terminal boundary observations."""

    X_int: NDArray[np.float64]
    X_bd: NDArray[np.float64]
    y_bd: NDArray[np.float64]


def make_uniform_terminal_grid(
    maturity: float,
    x_min: float,
    x_max: float,
    n_t_int: int,
    n_x_int: int,
    n_x_bd: int,
    payoff_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> GridSet:
    """Return a uniform interior grid with terminal payoff boundary values."""

    _validate_grid_parameters(maturity, x_min, x_max, n_t_int, n_x_int, n_x_bd)

    t_int = np.linspace(0.0, maturity, n_t_int + 2)[1:-1]
    x_int = np.linspace(x_min, x_max, n_x_int + 2)[1:-1]
    T_int, X_int_values = np.meshgrid(t_int, x_int, indexing="ij")
    X_int = np.column_stack([T_int.ravel(), X_int_values.ravel()])

    x_bd = np.linspace(x_min, x_max, n_x_bd)
    X_bd = np.column_stack([np.full(n_x_bd, maturity), x_bd])
    y_bd = np.asarray(payoff_fn(x_bd), dtype=float)
    if y_bd.ndim != 1 or y_bd.shape[0] != n_x_bd:
        raise ValueError(
            "payoff_fn must return a 1D array with length n_x_bd, "
            f"got shape {y_bd.shape}."
        )

    _validate_finite_array("X_int", X_int)
    _validate_finite_array("X_bd", X_bd)
    _validate_finite_array("y_bd", y_bd)
    return GridSet(X_int=X_int, X_bd=X_bd, y_bd=y_bd)


def _validate_grid_parameters(
    maturity: float,
    x_min: float,
    x_max: float,
    n_t_int: int,
    n_x_int: int,
    n_x_bd: int,
) -> None:
    if not np.isfinite(maturity) or maturity <= 0:
        raise ValueError(f"maturity must be positive and finite, got {maturity!r}.")
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        raise ValueError(
            f"x_min and x_max must be finite with x_min < x_max, got {x_min!r}, {x_max!r}."
        )
    if n_t_int < 1:
        raise ValueError(f"n_t_int must be at least 1, got {n_t_int!r}.")
    if n_x_int < 1:
        raise ValueError(f"n_x_int must be at least 1, got {n_x_int!r}.")
    if n_x_bd < 2:
        raise ValueError(f"n_x_bd must be at least 2, got {n_x_bd!r}.")


def _validate_finite_array(name: str, values: NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
