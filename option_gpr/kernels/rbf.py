"""Anisotropic RBF kernel in time and log-price coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class RBFKernel:
    """Anisotropic RBF kernel for points shaped ``(n, 2)``.

    Points use columns ``t`` and ``x``, where ``x`` is log-price.
    """

    ell_t: float
    ell_x: float
    sigma_f: float

    def __post_init__(self) -> None:
        for name, value in (
            ("ell_t", self.ell_t),
            ("ell_x", self.ell_x),
            ("sigma_f", self.sigma_f),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}.")

    def tau(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return pairwise time differences ``t_i - s_j``."""

        X_arr, Y_arr = self._validate_pair(X, Y)
        return X_arr[:, [0]] - Y_arr[:, 0]

    def chi(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return pairwise log-price differences ``x_i - xi_j``."""

        X_arr, Y_arr = self._validate_pair(X, Y)
        return X_arr[:, [1]] - Y_arr[:, 1]

    def K(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return the RBF kernel matrix for two point sets."""

        tau = self.tau(X, Y)
        chi = self.chi(X, Y)
        exponent = -0.5 * (
            (tau / self.ell_t) ** 2 + (chi / self.ell_x) ** 2
        )
        return self.sigma_f**2 * np.exp(exponent)

    @staticmethod
    def _validate_pair(
        X: ArrayLike, Y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return RBFKernel._validate_points(X, "X"), RBFKernel._validate_points(Y, "Y")

    @staticmethod
    def _validate_points(points: ArrayLike, name: str) -> NDArray[np.float64]:
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"{name} must be a 2D array with shape (n, 2), got {arr.shape}."
            )
        return arr
