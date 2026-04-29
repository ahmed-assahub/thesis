import numpy as np
import pytest

from option_gpr.kernels import RBFKernel
from option_gpr.models import BlackScholesModel
from option_gpr.operators import BSLogOperator
from option_gpr.posterior import StackedOperatorGP


def _make_gp(noise_int: float = 1e-4, noise_bd: float = 1e-4) -> StackedOperatorGP:
    model = BlackScholesModel(r=0.05, sigma=0.2, strike=100.0, maturity=1.0)
    kernel = RBFKernel(ell_t=0.7, ell_x=1.3, sigma_f=1.5)
    operator = BSLogOperator(model=model, kernel=kernel)
    return StackedOperatorGP(
        model=model,
        kernel=kernel,
        operator=operator,
        noise_int=noise_int,
        noise_bd=noise_bd,
        jitter=1e-8,
    )


def _sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_int = np.array([[0.2, 3.9], [0.5, 4.0]])
    X_bd = np.array([[1.0, 3.8], [1.0, 4.1], [1.0, 4.3]])
    y_bd = np.array([0.0, 5.0, 25.0])
    return X_int, X_bd, y_bd


def test_build_y_A_stacks_interior_zeros_and_boundary_values() -> None:
    gp = _make_gp()
    X_int, _, y_bd = _sample_data()

    y_A = gp.build_y_A(X_int, y_bd)

    np.testing.assert_allclose(y_A, np.array([0.0, 0.0, 0.0, 5.0, 25.0]))


def test_build_K_AA_shape_and_symmetry() -> None:
    gp = _make_gp()
    X_int, X_bd, _ = _sample_data()

    K_AA = gp.build_K_AA(X_int, X_bd)

    assert K_AA.shape == (5, 5)
    np.testing.assert_allclose(K_AA, K_AA.T)


def test_build_K_star_A_shape() -> None:
    gp = _make_gp()
    X_int, X_bd, _ = _sample_data()
    X_star = np.array([[0.1, 3.95], [0.6, 4.2], [0.9, 4.4], [1.0, 4.0]])

    K_star_A = gp.build_K_star_A(X_star, X_int, X_bd)

    assert K_star_A.shape == (4, 5)


def test_build_noise_diag_uses_separate_squared_noise_levels() -> None:
    gp = _make_gp(noise_int=0.1, noise_bd=0.2)

    noise_diag = gp.build_noise_diag(n_int=2, n_bd=3)

    np.testing.assert_allclose(noise_diag, np.array([0.01, 0.01, 0.04, 0.04, 0.04]))


def test_fit_builds_regularized_matrix_and_cholesky_factor() -> None:
    gp = _make_gp(noise_int=1e-3, noise_bd=1e-3)
    X_int, X_bd, y_bd = _sample_data()

    fitted = gp.fit(X_int, X_bd, y_bd)

    assert fitted is gp
    assert gp.y_A is not None
    assert gp.K_AA is not None
    assert gp.K_reg is not None
    assert gp.cholesky_factor is not None
    assert gp.K_reg.shape == (5, 5)
    assert gp.cholesky_factor.shape == (5, 5)


def test_fit_rejects_boundary_value_length_mismatch() -> None:
    gp = _make_gp()
    X_int, X_bd, _ = _sample_data()

    with pytest.raises(ValueError):
        gp.fit(X_int, X_bd, np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_int": -1e-4},
        {"noise_bd": -1e-4},
        {"jitter": -1e-8},
        {"noise_int": np.inf},
    ],
)
def test_stacked_gp_rejects_invalid_regularization_parameters(
    kwargs: dict[str, float]
) -> None:
    model = BlackScholesModel(r=0.05, sigma=0.2, strike=100.0, maturity=1.0)
    kernel = RBFKernel(ell_t=0.7, ell_x=1.3, sigma_f=1.5)
    operator = BSLogOperator(model=model, kernel=kernel)
    params = {
        "model": model,
        "kernel": kernel,
        "operator": operator,
        "noise_int": 1e-4,
        "noise_bd": 1e-4,
        "jitter": 1e-8,
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        StackedOperatorGP(**params)
