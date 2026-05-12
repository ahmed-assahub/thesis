from collections.abc import Callable

import numpy as np

from option_gpr.kernels import RBFKernel
from option_gpr.models import BlackScholesModel, MertonJumpDiffusionModel
from option_gpr.operators import BSLogOperator, MJDOperator


def _sample_points() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.15, 3.9], [0.65, 4.2]])
    Y = np.array([[0.25, 3.8], [0.75, 4.1], [0.9, 4.4]])
    return X, Y


def _central_difference(
    fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, *, axis: int, step: float
) -> np.ndarray:
    X_plus = X.copy()
    X_minus = X.copy()
    X_plus[:, axis] += step
    X_minus[:, axis] -= step
    return (fn(X_plus) - fn(X_minus)) / (2.0 * step)


def _second_central_difference(
    fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, *, axis: int, step: float
) -> np.ndarray:
    X_plus = X.copy()
    X_minus = X.copy()
    X_plus[:, axis] += step
    X_minus[:, axis] -= step
    return (fn(X_plus) - 2.0 * fn(X) + fn(X_minus)) / step**2


def test_bs_kLp_first_argument_derivatives_match_finite_differences() -> None:
    model = BlackScholesModel(r=0.05, sigma=0.2, strike=100.0, maturity=1.0)
    kernel = RBFKernel(ell_t=0.8, ell_x=1.1, sigma_f=1.4)
    operator = BSLogOperator(model=model, kernel=kernel)
    X, Y = _sample_points()

    np.testing.assert_allclose(
        operator.d_t_kLp(X, Y),
        _central_difference(lambda Z: operator.kLp(Z, Y), X, axis=0, step=1e-6),
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        operator.d_x_kLp(X, Y),
        _central_difference(lambda Z: operator.kLp(Z, Y), X, axis=1, step=1e-6),
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        operator.d_xx_kLp(X, Y),
        _second_central_difference(
            lambda Z: operator.kLp(Z, Y), X, axis=1, step=1e-4
        ),
        rtol=5e-4,
        atol=1e-6,
    )


def test_mjd_kLp_first_argument_derivatives_match_finite_differences() -> None:
    model = MertonJumpDiffusionModel(
        r=0.05,
        sigma=0.25,
        jump_intensity=0.7,
        jump_mean=-0.1,
        jump_std=0.2,
        strike=100.0,
        maturity=1.0,
    )
    kernel = RBFKernel(ell_t=0.8, ell_x=1.1, sigma_f=1.4)
    operator = MJDOperator(model=model, kernel=kernel)
    X, Y = _sample_points()

    np.testing.assert_allclose(
        operator.d_t_kLp(X, Y),
        _central_difference(lambda Z: operator.kLp(Z, Y), X, axis=0, step=1e-6),
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        operator.d_x_kLp(X, Y),
        _central_difference(lambda Z: operator.kLp(Z, Y), X, axis=1, step=1e-6),
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        operator.d_xx_kLp(X, Y),
        _second_central_difference(
            lambda Z: operator.kLp(Z, Y), X, axis=1, step=1e-4
        ),
        rtol=5e-4,
        atol=1e-6,
    )


def test_mjd_zero_jump_derivatives_reduce_to_black_scholes() -> None:
    mjd_model = MertonJumpDiffusionModel(
        r=0.05,
        sigma=0.25,
        jump_intensity=0.0,
        jump_mean=-0.1,
        jump_std=0.2,
        strike=100.0,
        maturity=1.0,
    )
    bs_model = BlackScholesModel(
        r=mjd_model.r,
        sigma=mjd_model.sigma,
        strike=mjd_model.strike,
        maturity=mjd_model.maturity,
    )
    kernel = RBFKernel(ell_t=0.8, ell_x=1.1, sigma_f=1.4)
    mjd_operator = MJDOperator(model=mjd_model, kernel=kernel)
    bs_operator = BSLogOperator(model=bs_model, kernel=kernel)
    X, Y = _sample_points()

    np.testing.assert_allclose(mjd_operator.d_t_kLp(X, Y), bs_operator.d_t_kLp(X, Y))
    np.testing.assert_allclose(mjd_operator.d_x_kLp(X, Y), bs_operator.d_x_kLp(X, Y))
    np.testing.assert_allclose(mjd_operator.d_xx_kLp(X, Y), bs_operator.d_xx_kLp(X, Y))
