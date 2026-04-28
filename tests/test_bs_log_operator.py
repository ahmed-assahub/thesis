import numpy as np
import pytest

from option_gpr.kernels import RBFKernel
from option_gpr.models import BlackScholesModel
from option_gpr.operators import BSLogOperator


def _make_operator() -> BSLogOperator:
    model = BlackScholesModel(r=0.05, sigma=0.2, strike=100.0, maturity=1.0)
    kernel = RBFKernel(ell_t=0.7, ell_x=1.3, sigma_f=1.5)
    return BSLogOperator(model=model, kernel=kernel)


def _sample_points() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 3.9], [0.4, 4.0], [0.8, 4.2]])
    Y = np.array([[0.2, 3.8], [0.6, 4.1]])
    return X, Y


def test_bs_log_operator_block_shapes() -> None:
    operator = _make_operator()
    X, Y = _sample_points()

    assert operator.Lk(X, Y).shape == (3, 2)
    assert operator.kLp(X, Y).shape == (3, 2)
    assert operator.LkLp(X, Y).shape == (3, 2)


def test_bs_log_operator_first_and_second_argument_symmetry() -> None:
    operator = _make_operator()
    X, Y = _sample_points()

    np.testing.assert_allclose(operator.Lk(X, Y), operator.kLp(Y, X).T)


def test_bs_log_operator_LkLp_symmetry() -> None:
    operator = _make_operator()
    X, Y = _sample_points()

    np.testing.assert_allclose(operator.LkLp(X, Y), operator.LkLp(Y, X).T)


def test_bs_log_operator_matches_closed_form_identities() -> None:
    operator = _make_operator()
    X, Y = _sample_points()
    kernel = operator.kernel
    coeffs = operator.model.coefficients()

    tau = X[:, [0]] - Y[:, 0]
    chi = X[:, [1]] - Y[:, 1]
    k = kernel.K(X, Y)
    D = chi**2 / kernel.ell_x**4 - 1.0 / kernel.ell_x**2
    P = (
        coeffs.a
        - tau / kernel.ell_t**2
        - coeffs.b * chi / kernel.ell_x**2
        + coeffs.c * D
    )
    Q = (
        coeffs.a
        + tau / kernel.ell_t**2
        + coeffs.b * chi / kernel.ell_x**2
        + coeffs.c * D
    )
    expected_LkLp = (
        P * Q
        + 1.0 / kernel.ell_t**2
        + coeffs.b**2 / kernel.ell_x**2
        + 2.0 * coeffs.c**2 / kernel.ell_x**4
        - 4.0 * coeffs.c**2 * chi**2 / kernel.ell_x**6
    ) * k

    np.testing.assert_allclose(operator.Lk(X, Y), P * k)
    np.testing.assert_allclose(operator.kLp(X, Y), Q * k)
    np.testing.assert_allclose(operator.LkLp(X, Y), expected_LkLp)


def test_bs_log_operator_reuses_kernel_input_validation() -> None:
    operator = _make_operator()
    _, Y = _sample_points()

    with pytest.raises(ValueError):
        operator.Lk(np.array([0.0, 1.0]), Y)
