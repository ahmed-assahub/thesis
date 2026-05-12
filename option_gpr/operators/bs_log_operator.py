"""Black-Scholes log-price differential operator applied to RBF kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from option_gpr.kernels import RBFKernel
from option_gpr.models import BlackScholesModel


@dataclass(frozen=True)
class BSLogOperator:
    """Black-Scholes log-price kernel identities.

    Sign convention:
    - ``Lk(X, Y)`` applies the pricing operator to the first kernel argument.
    - ``kLp(X, Y)`` applies the pricing operator to the second kernel argument.
    - ``LkLp(X, Y)`` applies the pricing operator to both kernel arguments.
    """

    model: BlackScholesModel
    kernel: RBFKernel

    def Lk(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return ``L_z k(z, y)`` for rows ``z`` in ``X`` and ``y`` in ``Y``."""

        return self._P(X, Y) * self.kernel.K(X, Y)

    def kLp(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return ``L_y k(z, y)`` for rows ``z`` in ``X`` and ``y`` in ``Y``."""

        return self._Q(X, Y) * self.kernel.K(X, Y)

    def d_t_kLp(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return ``d/dt [L_y k((t, x), y)]`` for the first kernel argument."""

        tau = self.kernel.tau(X, Y)
        ell_t = self.kernel.ell_t
        C = self._Q(X, Y)
        k = self.kernel.K(X, Y)
        return (1.0 / ell_t**2 - (tau / ell_t**2) * C) * k

    def d_x_kLp(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return ``d/dx [L_y k((t, x), y)]`` for log-price ``x``."""

        coeffs = self.model.coefficients()
        chi = self.kernel.chi(X, Y)
        ell_x = self.kernel.ell_x
        C = self._Q(X, Y)
        k = self.kernel.K(X, Y)
        multiplier = (
            coeffs.b / ell_x**2
            + 2.0 * coeffs.c * chi / ell_x**4
            - (chi / ell_x**2) * C
        )
        return multiplier * k

    def d_xx_kLp(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return ``d2/dx2 [L_y k((t, x), y)]`` for log-price ``x``."""

        coeffs = self.model.coefficients()
        chi = self.kernel.chi(X, Y)
        ell_x = self.kernel.ell_x
        C = self._Q(X, Y)
        k = self.kernel.K(X, Y)
        multiplier = (
            2.0 * coeffs.c / ell_x**4
            - 2.0 * coeffs.b * chi / ell_x**4
            - 4.0 * coeffs.c * chi**2 / ell_x**6
            + C * (chi**2 / ell_x**4 - 1.0 / ell_x**2)
        )
        return multiplier * k

    def LkLp(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return ``L_z L_y k(z, y)`` for rows ``z`` in ``X`` and ``y`` in ``Y``."""

        coeffs = self.model.coefficients()
        chi = self.kernel.chi(X, Y)
        ell_t = self.kernel.ell_t
        ell_x = self.kernel.ell_x
        correction = (
            1.0 / ell_t**2
            + coeffs.b**2 / ell_x**2
            + 2.0 * coeffs.c**2 / ell_x**4
            - 4.0 * coeffs.c**2 * chi**2 / ell_x**6
        )
        return (self._P(X, Y) * self._Q(X, Y) + correction) * self.kernel.K(X, Y)

    def _P(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        coeffs = self.model.coefficients()
        tau = self.kernel.tau(X, Y)
        chi = self.kernel.chi(X, Y)
        return (
            coeffs.a
            - tau / self.kernel.ell_t**2
            - coeffs.b * chi / self.kernel.ell_x**2
            + coeffs.c * self._D(chi)
        )

    def _Q(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        coeffs = self.model.coefficients()
        tau = self.kernel.tau(X, Y)
        chi = self.kernel.chi(X, Y)
        return (
            coeffs.a
            + tau / self.kernel.ell_t**2
            + coeffs.b * chi / self.kernel.ell_x**2
            + coeffs.c * self._D(chi)
        )

    def _D(self, chi: NDArray[np.float64]) -> NDArray[np.float64]:
        ell_x = self.kernel.ell_x
        return chi**2 / ell_x**4 - 1.0 / ell_x**2
