"""Financial model parameter containers."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite


@dataclass(frozen=True)
class OperatorCoefficients:
    """Constant coefficients for the log-price pricing operator."""

    a: float
    b: float
    c: float


@dataclass(frozen=True)
class BlackScholesModel:
    """Black-Scholes model parameters in log-price coordinates."""

    r: float
    sigma: float
    strike: float
    maturity: float

    def __post_init__(self) -> None:
        _validate_finite("r", self.r)
        _validate_positive("sigma", self.sigma)
        _validate_positive("strike", self.strike)
        _validate_positive("maturity", self.maturity)

    def coefficients(self) -> OperatorCoefficients:
        """Return coefficients ``a, b, c`` for the Black-Scholes log operator."""

        sigma_sq = self.sigma**2
        return OperatorCoefficients(
            a=-self.r,
            b=self.r - 0.5 * sigma_sq,
            c=0.5 * sigma_sq,
        )


@dataclass(frozen=True)
class MertonJumpDiffusionModel:
    """Merton jump diffusion model parameters in log-price coordinates."""

    r: float
    sigma: float
    jump_intensity: float
    jump_mean: float
    jump_std: float
    strike: float
    maturity: float

    def __post_init__(self) -> None:
        _validate_finite("r", self.r)
        _validate_positive("sigma", self.sigma)
        _validate_nonnegative("jump_intensity", self.jump_intensity)
        _validate_finite("jump_mean", self.jump_mean)
        _validate_positive("jump_std", self.jump_std)
        _validate_positive("strike", self.strike)
        _validate_positive("maturity", self.maturity)

    def coefficients(self) -> OperatorCoefficients:
        """Return differential coefficients ``a, b, c`` for the Merton operator."""

        sigma_sq = self.sigma**2
        jump_compensator = exp(self.jump_mean + 0.5 * self.jump_std**2) - 1.0
        return OperatorCoefficients(
            a=-(self.r + self.jump_intensity),
            b=self.r - 0.5 * sigma_sq - self.jump_intensity * jump_compensator,
            c=0.5 * sigma_sq,
        )


def _validate_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")


def _validate_positive(name: str, value: float) -> None:
    _validate_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _validate_nonnegative(name: str, value: float) -> None:
    _validate_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}.")
