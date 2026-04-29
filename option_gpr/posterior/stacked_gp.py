"""Stacked operator Gaussian process posterior blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class StackedOperatorGP:
    """Build covariance blocks for interior operator and boundary observations."""

    model: Any
    kernel: Any
    operator: Any
    noise_int: float
    noise_bd: float
    jitter: float
    X_int: NDArray[np.float64] | None = field(default=None, init=False)
    X_bd: NDArray[np.float64] | None = field(default=None, init=False)
    y_A: NDArray[np.float64] | None = field(default=None, init=False)
    K_AA: NDArray[np.float64] | None = field(default=None, init=False)
    K_reg: NDArray[np.float64] | None = field(default=None, init=False)
    cholesky_factor: NDArray[np.float64] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _validate_nonnegative("noise_int", self.noise_int)
        _validate_nonnegative("noise_bd", self.noise_bd)
        _validate_nonnegative("jitter", self.jitter)

    def build_y_A(self, X_int: ArrayLike, y_bd: ArrayLike) -> NDArray[np.float64]:
        """Return stacked observations ``[0_int, y_bd]``."""

        X_int_arr = self.kernel._validate_points(X_int, "X_int")
        y_bd_arr = _validate_vector(y_bd, "y_bd")
        return np.concatenate([np.zeros(X_int_arr.shape[0]), y_bd_arr])

    def build_K_AA(self, X_int: ArrayLike, X_bd: ArrayLike) -> NDArray[np.float64]:
        """Return the stacked covariance matrix for interior and boundary rows."""

        X_int_arr = self.kernel._validate_points(X_int, "X_int")
        X_bd_arr = self.kernel._validate_points(X_bd, "X_bd")
        top = np.hstack(
            [
                self.operator.LkLp(X_int_arr, X_int_arr),
                self.operator.Lk(X_int_arr, X_bd_arr),
            ]
        )
        bottom = np.hstack(
            [
                self.operator.kLp(X_bd_arr, X_int_arr),
                self.kernel.K(X_bd_arr, X_bd_arr),
            ]
        )
        return np.vstack([top, bottom])

    def build_K_star_A(
        self, X_star: ArrayLike, X_int: ArrayLike, X_bd: ArrayLike
    ) -> NDArray[np.float64]:
        """Return cross-covariance between test points and stacked observations."""

        X_star_arr = self.kernel._validate_points(X_star, "X_star")
        X_int_arr = self.kernel._validate_points(X_int, "X_int")
        X_bd_arr = self.kernel._validate_points(X_bd, "X_bd")
        return np.hstack(
            [
                self.operator.kLp(X_star_arr, X_int_arr),
                self.kernel.K(X_star_arr, X_bd_arr),
            ]
        )

    def build_noise_diag(self, n_int: int, n_bd: int) -> NDArray[np.float64]:
        """Return the diagonal of the stacked observation noise matrix."""

        if n_int < 0 or n_bd < 0:
            raise ValueError("n_int and n_bd must be nonnegative.")
        return np.concatenate(
            [
                np.full(n_int, self.noise_int**2),
                np.full(n_bd, self.noise_bd**2),
            ]
        )

    def fit(
        self, X_int: ArrayLike, X_bd: ArrayLike, y_bd: ArrayLike
    ) -> "StackedOperatorGP":
        """Build and factorize the regularized stacked covariance matrix."""

        X_int_arr = self.kernel._validate_points(X_int, "X_int")
        X_bd_arr = self.kernel._validate_points(X_bd, "X_bd")
        y_bd_arr = _validate_vector(y_bd, "y_bd")
        if y_bd_arr.shape[0] != X_bd_arr.shape[0]:
            raise ValueError(
                "y_bd length must match the number of boundary points, "
                f"got {y_bd_arr.shape[0]} and {X_bd_arr.shape[0]}."
            )

        self.X_int = X_int_arr
        self.X_bd = X_bd_arr
        self.y_A = self.build_y_A(X_int_arr, y_bd_arr)
        self.K_AA = self.build_K_AA(X_int_arr, X_bd_arr)
        noise_diag = self.build_noise_diag(X_int_arr.shape[0], X_bd_arr.shape[0])
        self.K_reg = self.K_AA + np.diag(noise_diag + self.jitter)
        self.cholesky_factor = np.linalg.cholesky(self.K_reg)
        return self

    def predict(
        self,
        X_star: ArrayLike,
        return_cov: bool = False,
        return_var: bool = False,
    ) -> (
        NDArray[np.float64]
        | tuple[NDArray[np.float64], NDArray[np.float64]]
    ):
        """Return posterior mean, optionally with covariance or marginal variance."""

        if return_cov and return_var:
            raise ValueError("Only one of return_cov or return_var may be True.")
        if (
            self.X_int is None
            or self.X_bd is None
            or self.y_A is None
            or self.cholesky_factor is None
        ):
            raise RuntimeError("fit must be called before predict.")

        X_star_arr = self.kernel._validate_points(X_star, "X_star")
        K_star_A = self.build_K_star_A(X_star_arr, self.X_int, self.X_bd)
        lower_solve_y = np.linalg.solve(self.cholesky_factor, self.y_A)
        alpha = np.linalg.solve(self.cholesky_factor.T, lower_solve_y)
        mean = K_star_A @ alpha

        if not return_cov and not return_var:
            return mean

        V = np.linalg.solve(self.cholesky_factor, K_star_A.T)
        cov = self.kernel.K(X_star_arr, X_star_arr) - V.T @ V
        if return_cov:
            return mean, cov

        return mean, np.diag(cov)


def _validate_nonnegative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite, got {value!r}.")


def _validate_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr
