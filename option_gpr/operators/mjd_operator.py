"""Merton jump-diffusion log-price operator applied to RBF kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from option_gpr.kernels import RBFKernel
from option_gpr.models import MertonJumpDiffusionModel


@dataclass(frozen=True)
class MJDOperator:
    """Merton jump-diffusion log-price kernel identities.

    Sign convention:
    - ``Lk(X, Y)`` applies the pricing operator to the first kernel argument.
    - ``kLp(X, Y)`` applies the pricing operator to the second kernel argument.
    - ``LkLp(X, Y)`` applies the pricing operator to both kernel arguments.
    """

    model: MertonJumpDiffusionModel
    kernel: RBFKernel

    def Lk(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return ``L_z k(z, y)`` for rows ``z`` in ``X`` and ``y`` in ``Y``."""

        jump_intensity = self.model.jump_intensity
        return self._P(X, Y) * self.kernel.K(X, Y) + jump_intensity * self._Jxk(X, Y)

    def kLp(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return ``L_y k(z, y)`` for rows ``z`` in ``X`` and ``y`` in ``Y``."""

        jump_intensity = self.model.jump_intensity
        return self._Q(X, Y) * self.kernel.K(X, Y) + jump_intensity * self._Jyk(X, Y)

    def LkLp(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Return ``L_z L_y k(z, y)`` for rows ``z`` in ``X`` and ``y`` in ``Y``."""

        jump_intensity = self.model.jump_intensity
        return (
            self._Dx_Dyk(X, Y)
            + jump_intensity * self._Dx_Jyk(X, Y)
            + jump_intensity * self._Jx_Dyk(X, Y)
            + jump_intensity**2 * self._JxJyk(X, Y)
        )

    def _P(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        coeffs = self.model.coefficients()
        tau = self.kernel.tau(X, Y)
        chi = self.kernel.chi(X, Y)
        v = self.kernel.ell_x**2
        return (
            coeffs.a
            - tau / self.kernel.ell_t**2
            - coeffs.b * chi / v
            + coeffs.c * self._D_v(chi)
        )

    def _Q(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        coeffs = self.model.coefficients()
        tau = self.kernel.tau(X, Y)
        chi = self.kernel.chi(X, Y)
        v = self.kernel.ell_x**2
        return (
            coeffs.a
            + tau / self.kernel.ell_t**2
            + coeffs.b * chi / v
            + coeffs.c * self._D_v(chi)
        )

    def _Dx_Dyk(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        coeffs = self.model.coefficients()
        chi = self.kernel.chi(X, Y)
        ell_t = self.kernel.ell_t
        v = self.kernel.ell_x**2
        correction = (
            1.0 / ell_t**2
            + coeffs.b**2 / v
            + 2.0 * coeffs.c**2 / v**2
            - 4.0 * coeffs.c**2 * chi**2 / v**3
        )
        return (self._P(X, Y) * self._Q(X, Y) + correction) * self.kernel.K(X, Y)

    def _Dx_Jyk(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        coeffs = self.model.coefficients()
        tau = self.kernel.tau(X, Y)
        chi = self.kernel.chi(X, Y)
        q = chi - self.model.jump_mean
        w = self._jump_variance_sum()
        multiplier = (
            coeffs.a
            - tau / self.kernel.ell_t**2
            - coeffs.b * q / w
            + coeffs.c * self._D_w(q)
        )
        return multiplier * self._Jyk(X, Y)

    def _Jx_Dyk(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        coeffs = self.model.coefficients()
        tau = self.kernel.tau(X, Y)
        chi = self.kernel.chi(X, Y)
        q = chi + self.model.jump_mean
        w = self._jump_variance_sum()
        multiplier = (
            coeffs.a
            + tau / self.kernel.ell_t**2
            + coeffs.b * q / w
            + coeffs.c * self._D_w(q)
        )
        return multiplier * self._Jxk(X, Y)

    def _Jxk(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        tau = self.kernel.tau(X, Y)
        chi = self.kernel.chi(X, Y)
        v = self.kernel.ell_x**2
        w = self._jump_variance_sum()
        q = chi + self.model.jump_mean
        return (
            self.kernel.sigma_f**2
            * np.exp(-0.5 * (tau / self.kernel.ell_t) ** 2)
            * np.sqrt(v / w)
            * np.exp(-0.5 * q**2 / w)
        )

    def _Jyk(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        tau = self.kernel.tau(X, Y)
        chi = self.kernel.chi(X, Y)
        v = self.kernel.ell_x**2
        w = self._jump_variance_sum()
        q = chi - self.model.jump_mean
        return (
            self.kernel.sigma_f**2
            * np.exp(-0.5 * (tau / self.kernel.ell_t) ** 2)
            * np.sqrt(v / w)
            * np.exp(-0.5 * q**2 / w)
        )

    def _JxJyk(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        tau = self.kernel.tau(X, Y)
        chi = self.kernel.chi(X, Y)
        v = self.kernel.ell_x**2
        w2 = v + 2.0 * self.model.jump_std**2
        return (
            self.kernel.sigma_f**2
            * np.exp(-0.5 * (tau / self.kernel.ell_t) ** 2)
            * np.sqrt(v / w2)
            * np.exp(-0.5 * chi**2 / w2)
        )

    def _D_v(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        v = self.kernel.ell_x**2
        return q**2 / v**2 - 1.0 / v

    def _D_w(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        w = self._jump_variance_sum()
        return q**2 / w**2 - 1.0 / w

    def _jump_variance_sum(self) -> float:
        return self.kernel.ell_x**2 + self.model.jump_std**2
